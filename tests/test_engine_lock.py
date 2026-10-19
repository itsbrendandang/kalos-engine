"""The engine's state-directory guard (kalos/portal/engine_lock.py): a second
engine on the same state directory fails loudly instead of interleaving."""
from __future__ import annotations

import subprocess
import sys

import pytest

from kalos.portal import __main__ as portal_main
from kalos.portal import engine_lock


def test_second_acquire_on_the_same_state_dir_is_refused(tmp_path):
    first = engine_lock.acquire_engine_lock(tmp_path)
    try:
        with pytest.raises(engine_lock.EngineLockHeld, match="already holds"):
            engine_lock.acquire_engine_lock(tmp_path)
    finally:
        first.close()


def test_lock_is_released_when_the_holder_closes(tmp_path):
    engine_lock.acquire_engine_lock(tmp_path).close()
    second = engine_lock.acquire_engine_lock(tmp_path)  # does not raise
    assert (tmp_path / engine_lock.LOCK_FILENAME).read_text().strip().isdigit()
    second.close()


def test_a_different_state_dir_is_independent(tmp_path):
    a = engine_lock.acquire_engine_lock(tmp_path / "a")
    try:
        b = engine_lock.acquire_engine_lock(tmp_path / "b")
        b.close()
    finally:
        a.close()


def test_lock_is_held_across_processes(tmp_path):
    """A real second process is refused while this one holds the lock."""
    holder = engine_lock.acquire_engine_lock(tmp_path)
    try:
        code = ("import sys; from pathlib import Path; from kalos.portal import engine_lock\n"
                "try:\n    engine_lock.acquire_engine_lock(Path(sys.argv[1]))\n"
                "except engine_lock.EngineLockHeld:\n    print('HELD')\n")
        out = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "HELD"
    finally:
        holder.close()


def test_portal_main_exits_2_with_a_reason_when_the_lock_is_held(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("KALOS_HOST", "127.0.0.1")
    monkeypatch.delenv("KALOS_PORT", raising=False)
    started = []
    monkeypatch.setattr(portal_main.uvicorn, "run", lambda *a, **k: started.append(a))
    holder = engine_lock.acquire_engine_lock(tmp_path)
    try:
        with pytest.raises(SystemExit) as exc:
            portal_main.main()
    finally:
        holder.close()
    assert exc.value.code == 2
    assert started == []
    assert "already holds" in capsys.readouterr().err


def test_portal_main_takes_the_lock_before_serving(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("KALOS_HOST", "127.0.0.1")
    monkeypatch.delenv("KALOS_PORT", raising=False)
    seen = []

    def fake_run(*args, **kwargs):
        # while "serving", a second engine on the same state dir is refused
        with pytest.raises(engine_lock.EngineLockHeld):
            engine_lock.acquire_engine_lock(tmp_path)
        seen.append(True)

    monkeypatch.setattr(portal_main.uvicorn, "run", fake_run)
    portal_main.main()
    assert seen == [True]
    # main() keeps the lock for the life of the process; releasing it here (as
    # process exit would) frees the directory again.
    portal_main._ENGINE_LOCK.close()
    engine_lock.acquire_engine_lock(tmp_path).close()


# --- failures that are not "held by another engine" ------------------------- #


def test_a_non_contention_flock_failure_is_not_reported_as_held(tmp_path, monkeypatch):
    """Regression: every OSError from flock used to read as 'already holds'. A
    filesystem that refuses flock (ENOLCK) must say so, not blame another engine."""
    import errno
    import fcntl

    def refuse(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    with pytest.raises(engine_lock.EngineLockUnsupported) as exc:
        engine_lock.acquire_engine_lock(tmp_path)
    assert not isinstance(exc.value, engine_lock.EngineLockHeld)
    assert "flock" in str(exc.value) and "already holds" not in str(exc.value)


def test_would_block_still_means_held(tmp_path, monkeypatch):
    import errno
    import fcntl

    def busy(fd, op):
        raise OSError(errno.EWOULDBLOCK, "Resource temporarily unavailable")

    monkeypatch.setattr(fcntl, "flock", busy)
    with pytest.raises(engine_lock.EngineLockHeld, match="already holds"):
        engine_lock.acquire_engine_lock(tmp_path)


def test_portal_main_exits_2_and_says_why_for_an_unlockable_filesystem(tmp_path, monkeypatch, capsys):
    import errno
    import fcntl

    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("KALOS_HOST", "127.0.0.1")
    monkeypatch.delenv("KALOS_PORT", raising=False)
    monkeypatch.setattr(fcntl, "flock", lambda fd, op: (_ for _ in ()).throw(
        OSError(errno.ENOLCK, "No locks available")))
    started = []
    monkeypatch.setattr(portal_main.uvicorn, "run", lambda *a, **k: started.append(a))
    with pytest.raises(SystemExit) as exc:
        portal_main.main()
    assert exc.value.code == 2 and started == []
    err = capsys.readouterr().err
    assert "flock" in err and "already holds" not in err


def test_module_imports_and_refuses_cleanly_without_fcntl(tmp_path):
    """fcntl is imported when the lock is taken, not at import: a platform
    without it can import the module and gets a clear refusal."""
    code = ("import sys; sys.modules['fcntl'] = None\n"
            "from pathlib import Path\n"
            "from kalos.portal import engine_lock\n"
            "try:\n    engine_lock.acquire_engine_lock(Path(sys.argv[1]))\n"
            "except engine_lock.EngineLockUnsupported as e:\n    print('UNSUPPORTED', 'POSIX' in str(e))\n")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "UNSUPPORTED True"


# --- a crash leaves nothing stale --------------------------------------------- #


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock")
def test_sigkilled_holder_releases_the_lock_for_the_next_engine(tmp_path):
    """A SIGKILLed engine (no cleanup, no atexit) leaves engine.lock on disk with
    its dead pid in it. The kernel released the flock, so the next engine starts."""
    import signal

    code = ("import sys; from pathlib import Path; from kalos.portal import engine_lock\n"
            "held = engine_lock.acquire_engine_lock(Path(sys.argv[1]))  # keep the handle\n"
            "print('LOCKED', flush=True)\n"
            "import time; time.sleep(600)\n")
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        # while the child lives, the lock is held
        with pytest.raises(engine_lock.EngineLockHeld):
            engine_lock.acquire_engine_lock(tmp_path)
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
    stale = (tmp_path / engine_lock.LOCK_FILENAME).read_text().strip()
    assert stale == str(child.pid)  # the file survives the kill, with the dead pid
    second = engine_lock.acquire_engine_lock(tmp_path)
    assert (tmp_path / engine_lock.LOCK_FILENAME).read_text().strip() != stale
    second.close()
