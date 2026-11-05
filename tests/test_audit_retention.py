"""Audit-log retention (review L8, revised for schema v4): `KALOS_AUDIT_RETENTION_DAYS`.

The engine never deletes an audit row. `audit_log` rejects UPDATE and DELETE
(tests/test_audit_append_only.py). The setting is now an opt-in ARCHIVE window:
unset (the default) keeps everything and archives nothing; a number N copies rows
older than N days to a per-tenant JSON-lines file under the state directory,
and the rows stay in `audit_log`. Archive runs at startup and daily, never fail
the boot they run in, and never copy a row twice.
"""
from __future__ import annotations

import json
import time

import pytest

from kalos.envconfig import ConfigError
from kalos.portal import app as portal
from kalos.portal import campaign as campaign_module
from kalos.portal.campaign import CampaignStore
from kalos.portal.config import audit_retention_days

DAY = 86_400.0


@pytest.fixture
def store(tmp_path):
    return CampaignStore(tmp_path)


def _age(store: CampaignStore, action: str, days_old: float) -> None:
    """Backdate the most recent row with `action`. The engine's own UPDATE path is
    closed by the v4 trigger, so this test seam runs as the schema owner with the
    trigger dropped for one statement, then restores it."""
    with store._txn(write=True) as conn:  # noqa: SLF001 - test seam for the row's age
        conn.execute("DROP TRIGGER audit_log_no_update")
        try:
            conn.execute(
                "UPDATE audit_log SET ts = ? WHERE id = (SELECT MAX(id) FROM audit_log WHERE action = ?)",
                (time.time() - days_old * DAY, action),
            )
        finally:
            conn.execute(
                "CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log BEGIN "
                "SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE is rejected'); END"
            )


def _actions(store: CampaignStore, tenant: str) -> list[str]:
    return [e["action"] for e in store.audit_entries(tenant=tenant, limit=1000)]


def _archive_lines(store: CampaignStore, tenant: str) -> list[dict]:
    path = store.state_dir / "audit-archive" / f"t-{tenant.encode().hex()}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_unset_retention_means_keep_everything(monkeypatch):
    monkeypatch.delenv("KALOS_AUDIT_RETENTION_DAYS", raising=False)
    assert audit_retention_days() is None


@pytest.mark.parametrize("raw, expected", [("30", 30), (" 7 ", 7), ("36500", 36_500)])
def test_retention_accepts_whole_days_in_range(monkeypatch, raw, expected):
    monkeypatch.setenv("KALOS_AUDIT_RETENTION_DAYS", raw)
    assert audit_retention_days() == expected


@pytest.mark.parametrize("raw", ["0", "-1", "36501", "forever", "1.5"])
def test_retention_refuses_out_of_range_or_malformed(monkeypatch, raw):
    monkeypatch.setenv("KALOS_AUDIT_RETENTION_DAYS", raw)
    with pytest.raises(ConfigError, match="KALOS_AUDIT_RETENTION_DAYS"):
        audit_retention_days()


def test_archive_copies_old_rows_and_deletes_nothing(store):
    store.record_audit(tenant="acme", action="old.action", outcome="200")
    _age(store, "old.action", 400)
    store.record_audit(tenant="acme", action="recent.action", outcome="200")
    _age(store, "recent.action", 364)
    store.record_audit(tenant="beta", action="old.other", outcome="200")
    _age(store, "old.other", 900)

    archived = store.archive_audit_log(365)

    assert archived == 2  # every tenant's old row, nothing else
    assert [r["action"] for r in _archive_lines(store, "acme")] == ["old.action"]
    assert [r["action"] for r in _archive_lines(store, "beta")] == ["old.other"]
    # the rows are still in the live table: archiving is a copy
    assert set(_actions(store, "acme")) == {"old.action", "recent.action"}
    assert _actions(store, "beta") == ["old.other"]


def test_archive_never_copies_a_row_twice(store):
    store.record_audit(tenant="acme", action="old", outcome="200")
    _age(store, "old", 500)
    assert store.archive_audit_log(365) == 1
    assert store.archive_audit_log(365) == 0
    store.record_audit(tenant="acme", action="older", outcome="200")
    _age(store, "older", 500)
    assert store.archive_audit_log(365) == 1
    assert [r["action"] for r in _archive_lines(store, "acme")] == ["old", "older"]


def test_archive_keeps_everything_when_nothing_is_old(store):
    for i in range(3):
        store.record_audit(tenant="acme", action=f"a{i}", outcome="200")
    assert store.archive_audit_log(365) == 0
    assert not (store.state_dir / "audit-archive").exists()
    assert len(_actions(store, "acme")) == 3


def test_archive_file_is_private(store):
    store.record_audit(tenant="acme", action="old", outcome="200")
    _age(store, "old", 500)
    store.archive_audit_log(365)
    path = store.state_dir / "audit-archive" / f"t-{b'acme'.hex()}.jsonl"
    assert path.stat().st_mode & 0o777 == 0o600
    assert (store.state_dir / "audit-archive").stat().st_mode & 0o777 == 0o700


def test_unset_setting_does_not_archive_at_startup(store, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    monkeypatch.delenv("KALOS_AUDIT_RETENTION_DAYS", raising=False)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal, "_STATE_DIR", store.state_dir)
    store.record_audit(tenant="acme", action="stale", outcome="200")
    _age(store, "stale", 4000)

    with TestClient(portal.app):
        pass

    assert _actions(store, "acme") == ["stale"]  # kept
    assert _archive_lines(store, "acme") == []  # and not archived either


def test_startup_archives_when_the_setting_is_set(store, monkeypatch):
    """With the setting, the lifespan archives once at startup. Rows stay live."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    monkeypatch.setenv("KALOS_AUDIT_RETENTION_DAYS", "30")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal, "_STATE_DIR", store.state_dir)
    store.record_audit(tenant="acme", action="stale", outcome="200")
    _age(store, "stale", 45)
    store.record_audit(tenant="acme", action="fresh", outcome="200")
    _age(store, "fresh", 5)

    with TestClient(portal.app):
        pass

    assert [r["action"] for r in _archive_lines(store, "acme")] == ["stale"]
    assert set(_actions(store, "acme")) == {"stale", "fresh"}


def test_a_failing_archive_does_not_stop_the_boot(store, monkeypatch, caplog):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    monkeypatch.setenv("KALOS_AUDIT_RETENTION_DAYS", "30")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal, "_STATE_DIR", store.state_dir)

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(CampaignStore, "archive_audit_log", boom)
    with TestClient(portal.app) as client:
        assert client.get("/healthz").status_code == 200
    assert "audit" in caplog.text.lower()
