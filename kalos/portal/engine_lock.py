"""A single-process guard for the engine's state directory.

`CampaignStore` (portal.db) assumes ONE engine process per state directory:
its single-writer story is SQLite's own locking plus the in-process limiter in
kalos/portal/busy.py, and nothing stops a second `python -m kalos.portal` from
starting against the same `KALOS_STATE_DIR` and interleaving with the first. The
result would be two servers that each believe they own the queue and the
latest-analysis cache.

`acquire_engine_lock` takes an exclusive, non-blocking `flock` on
`<state_dir>/engine.lock` and returns the open file. The lock lives exactly as
long as that file object: the kernel releases it when the process exits, however
it exits (crash, SIGKILL, OOM), so there is no stale-lock cleanup to get wrong.
The PID written into the file is informational only, for the error message.

The job worker processes (kalos/portal/job_worker.py) do NOT take this lock:
they are children of the one engine and never start a second server.

Only `EWOULDBLOCK`/`EAGAIN` mean "held by another process". Any other failure
to lock (a filesystem that does not support flock, an unexpected errno) is
reported as `EngineLockUnsupported`, because "another engine holds it" would be
a false and misleading message. `fcntl` is POSIX-only and is imported when the
lock is taken, so importing this module never fails on another platform.

Limits: the lock is per host. `uvicorn kalos.portal.app:app --workers N` starts
N processes that do not take it; start the engine with `python -m kalos.portal`
(one process), or give each engine its own state directory.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import IO

LOCK_FILENAME = "engine.lock"


class EngineLockError(RuntimeError):
    """The state-directory lock could not be taken. See the subclasses."""


class EngineLockHeld(EngineLockError):
    """Another engine process already owns this state directory."""


class EngineLockUnsupported(EngineLockError):
    """The lock could not be taken for a reason other than another holder: no
    POSIX `fcntl` on this platform, or a filesystem that refuses flock."""


# errno values that mean "another process holds the lock" (non-blocking flock).
_HELD_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN})


def default_state_dir() -> Path:
    # The same rule CampaignStore uses for its default, kept in one expression
    # here so the lock and the store can never point at different directories.
    return Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos"))


def acquire_engine_lock(state_dir: Path | None = None) -> IO[str]:
    """Take the exclusive engine lock for `state_dir`.

    Raises `EngineLockHeld` when another process holds it, and
    `EngineLockUnsupported` for any other failure to lock. Keep the returned file
    open for the life of the process. Closing it, or letting it be
    garbage-collected, releases the lock."""
    try:
        import fcntl  # POSIX only; deferred so importing this module never fails
    except ImportError:
        raise EngineLockUnsupported(
            "the engine state-directory lock needs POSIX fcntl, which this platform "
            "does not have. Run the engine on Linux or macOS."
        ) from None

    directory = Path(state_dir) if state_dir is not None else default_state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOCK_FILENAME
    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in _HELD_ERRNOS:
            handle.close()
            raise EngineLockUnsupported(
                f"could not lock {path}: {exc.strerror or 'error'} (errno {exc.errno}). "
                "The filesystem of KALOS_STATE_DIR may not support flock (some network "
                "filesystems do not). Use a local directory for the state."
            ) from None
        holder = ""
        try:
            handle.seek(0)
            holder = handle.read().strip()
        except OSError:
            pass
        handle.close()
        pid = f" (pid {holder})" if holder.isdigit() else ""
        raise EngineLockHeld(
            f"another kalos engine{pid} already holds {path}; one engine process per "
            "state directory. Stop it, or point this one at a different KALOS_STATE_DIR."
        ) from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle
