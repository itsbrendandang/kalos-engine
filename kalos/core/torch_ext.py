"""Guard against a stale torch JIT-extension lock hanging a request.

BoTorch's fused qLogEHVI kernel is JIT-compiled on first use
(`botorch.acquisition.multi_objective.logei._try_load_fused_kernel` ->
`torch.utils.cpp_extension.load`). That load serialises on a `lock` file in the
extension build dir and waits on it with no timeout. If a previous process was
killed mid-compile the lock is never removed, and every later multi-objective
call blocks until the caller's own timeout (the portal answers 504 after 120 s).

`clear_stale_extension_lock` removes a lock that is clearly abandoned (older
than any real compile) before the first load. A compile genuinely in progress
(young lock) is left alone, so concurrent first loads still wait correctly.
Numerics are untouched: the fused kernel still loads and is still used.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

EXTENSION_NAME = "logei_fused_ext"
# A real compile of the fused kernel takes ~7 s; five minutes is generous.
STALE_AFTER_SECONDS = 300.0


def _build_dir(name: str) -> Path | None:
    try:
        from torch.utils.cpp_extension import (
            _get_build_directory,  # private API: guarded
        )

        return Path(_get_build_directory(name, False))
    except Exception:  # noqa: BLE001 - private torch API moved: skip the guard, never break serving
        return None


def clear_stale_extension_lock(
    name: str = EXTENSION_NAME,
    *,
    stale_after: float = STALE_AFTER_SECONDS,
    build_dir: Path | None = None,
) -> bool:
    """Delete `<build_dir>/lock` if it is older than `stale_after` seconds.

    Returns True when a stale lock was removed. Never raises."""
    directory = build_dir if build_dir is not None else _build_dir(name)
    if directory is None:
        return False
    lock = directory / "lock"
    try:
        age = time.time() - lock.stat().st_mtime
        if age < stale_after:
            return False
        os.remove(lock)
    except FileNotFoundError:
        return False
    except OSError:
        log.warning("could not clear stale torch extension lock %s", lock, exc_info=True)
        return False
    log.warning("removed stale torch extension lock %s (%.0f s old)", lock, age)
    return True
