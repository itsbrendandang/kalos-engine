"""The suite-wide guard in tests/conftest.py really refuses the home state dir.

Every probe path below is a name that does not exist under `~/.kalos`; the
audit hook raises BEFORE the operation, so nothing is created or read there.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

_PROBE = Path(os.path.expanduser("~")) / ".kalos" / "__pytest_guard_probe__"


def test_session_state_dir_is_never_the_home_one():
    state = os.path.realpath(os.environ["KALOS_STATE_DIR"])
    home = os.path.realpath(os.path.join(os.path.expanduser("~"), ".kalos"))
    assert state != home and not state.startswith(home + os.sep)


def test_guard_refuses_open_mkdir_and_sqlite_under_home_state(real_state_guard):
    with real_state_guard.expect() as caught:
        with pytest.raises(RuntimeError, match="real Kalos state directory"):
            open(_PROBE / "x.txt", "w")  # noqa: SIM115 - the open itself is refused
        with pytest.raises(RuntimeError, match="real Kalos state directory"):
            _PROBE.mkdir(parents=True)
        with pytest.raises(RuntimeError, match="real Kalos state directory"):
            sqlite3.connect(_PROBE / "portal.db")
    assert [c.split(":", 1)[0] for c in caught] == ["open", "os.mkdir", "sqlite3.connect"]
    assert not _PROBE.exists()


def test_guard_refuses_a_campaign_store_at_the_home_default(real_state_guard, monkeypatch):
    """The exact E-B1 failure: a store built with KALOS_STATE_DIR unset."""
    from kalos.portal import campaign

    monkeypatch.delenv("KALOS_STATE_DIR")
    with (
        real_state_guard.expect() as caught,
        pytest.raises(RuntimeError, match="real Kalos state directory"),
    ):
        campaign.CampaignStore()
    assert caught, "the guard did not see the store reach ~/.kalos"
