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
