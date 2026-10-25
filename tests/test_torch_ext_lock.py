"""A stale torch JIT-extension lock must not hang the fused-kernel load."""
from __future__ import annotations

import os
import time

from kalos.core.torch_ext import clear_stale_extension_lock


def test_stale_lock_is_removed(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    assert clear_stale_extension_lock(build_dir=tmp_path) is True
    assert not lock.exists()


def test_fresh_lock_is_left_alone(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("")  # a compile may genuinely be running
    assert clear_stale_extension_lock(build_dir=tmp_path) is False
    assert lock.exists()


def test_missing_lock_and_missing_dir_are_noops(tmp_path):
    assert clear_stale_extension_lock(build_dir=tmp_path) is False
    assert clear_stale_extension_lock(build_dir=tmp_path / "nope") is False


def test_default_build_dir_resolves_under_torch_extensions_dir(tmp_path, monkeypatch):
    # Hermetic: point torch at a temp cache so the real one is never touched.
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    from kalos.core.torch_ext import _build_dir

    d = _build_dir("logei_fused_ext")
    assert d is not None and str(d).startswith(str(tmp_path))


def test_stale_lock_does_not_block_the_fused_load(tmp_path):
    """End-to-end in a subprocess (the load is once-per-process): with a stale
    lock in a temp extension dir the guard clears it and the kernel loads."""
    import subprocess
    import sys
    import textwrap

    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path))
    code = textwrap.dedent(
        """
        import os, time
        from kalos.core.torch_ext import _build_dir, clear_stale_extension_lock
        d = _build_dir("logei_fused_ext"); d.mkdir(parents=True, exist_ok=True)
        lock = d / "lock"; lock.write_text("")
        old = time.time() - 3600; os.utime(lock, (old, old))
        assert clear_stale_extension_lock()
        from botorch.acquisition.multi_objective import logei
        logei._try_load_fused_kernel()
        print("FUSED" if logei._C is not None else "FALLBACK")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().endswith(("FUSED", "FALLBACK"))
    # Without the guard this would have blocked on the lock until the timeout.
