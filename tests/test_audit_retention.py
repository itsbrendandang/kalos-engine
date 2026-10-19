"""Audit-log retention (review L8): `KALOS_AUDIT_RETENTION_DAYS`, default 365.

Rows older than the retention are pruned at startup and once a day. The prune
removes only rows older than the cut-off, for every tenant, and never fails the
request or the boot it runs in.
"""
from __future__ import annotations

import time

import pytest

from kalos.envconfig import ConfigError
from kalos.portal import app as portal
from kalos.portal import campaign as campaign_module
from kalos.portal.campaign import CampaignStore
from kalos.portal.config import AUDIT_RETENTION_DEFAULT_DAYS, audit_retention_days

DAY = 86_400.0


@pytest.fixture
def store(tmp_path):
    return CampaignStore(tmp_path)


def _age(store: CampaignStore, action: str, days_old: float) -> None:
    """Backdate the most recent row with `action` (the store API has no clock input)."""
    with store._txn(write=True) as conn:  # noqa: SLF001 - test seam for the row's age
        conn.execute(
            "UPDATE audit_log SET ts = ? WHERE id = (SELECT MAX(id) FROM audit_log WHERE action = ?)",
            (time.time() - days_old * DAY, action),
        )


def _actions(store: CampaignStore, tenant: str) -> list[str]:
    return [e["action"] for e in store.audit_entries(tenant=tenant, limit=1000)]


def test_default_retention_is_365_days(monkeypatch):
    monkeypatch.delenv("KALOS_AUDIT_RETENTION_DAYS", raising=False)
    assert audit_retention_days() == AUDIT_RETENTION_DEFAULT_DAYS == 365


@pytest.mark.parametrize("raw, expected", [("30", 30), (" 7 ", 7), ("36500", 36_500)])
def test_retention_accepts_whole_days_in_range(monkeypatch, raw, expected):
    monkeypatch.setenv("KALOS_AUDIT_RETENTION_DAYS", raw)
    assert audit_retention_days() == expected


@pytest.mark.parametrize("raw", ["0", "-1", "36501", "forever", "1.5"])
def test_retention_refuses_out_of_range_or_malformed(monkeypatch, raw):
    monkeypatch.setenv("KALOS_AUDIT_RETENTION_DAYS", raw)
    with pytest.raises(ConfigError, match="KALOS_AUDIT_RETENTION_DAYS"):
        audit_retention_days()


def test_prune_removes_only_rows_older_than_the_cut_off(store):
    store.record_audit(tenant="acme", action="old.action", outcome="200")
    _age(store, "old.action", 400)
    store.record_audit(tenant="acme", action="recent.action", outcome="200")
    _age(store, "recent.action", 364)
    store.record_audit(tenant="beta", action="old.other", outcome="200")
    _age(store, "old.other", 900)

    removed = store.prune_audit_log(365)

    assert removed == 2  # every tenant's old row, nothing else
    assert _actions(store, "acme") == ["recent.action"]
    assert _actions(store, "beta") == []


def test_prune_keeps_everything_when_nothing_is_old(store):
    for i in range(3):
        store.record_audit(tenant="acme", action=f"a{i}", outcome="200")
    assert store.prune_audit_log(365) == 0
    assert len(_actions(store, "acme")) == 3


def test_prune_is_idempotent(store):
    store.record_audit(tenant="acme", action="old", outcome="200")
    _age(store, "old", 500)
    assert store.prune_audit_log(365) == 1
    assert store.prune_audit_log(365) == 0


def test_startup_prunes_the_audit_log(store, monkeypatch):
    """The app lifespan prunes once at startup, with the configured retention."""
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

    assert _actions(store, "acme") == ["fresh"]


def test_a_failing_prune_does_not_stop_the_boot(store, monkeypatch, caplog):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal, "_STATE_DIR", store.state_dir)

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(CampaignStore, "prune_audit_log", boom)
    with TestClient(portal.app) as client:
        assert client.get("/healthz").status_code == 200
    assert "audit" in caplog.text.lower()
