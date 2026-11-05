"""A virtual-run batch and its `campaign.virtual_runs.create` audit row commit in
one transaction (kalos/portal/virtual_runs.py). Every database is in tmp_path."""
from __future__ import annotations

import json
import sqlite3

import pytest

from kalos.portal import virtual_runs as vr
from kalos.portal.campaign import CampaignStore


@pytest.fixture
def store(tmp_path):
    s = CampaignStore(tmp_path)
    yield s
    s.close()


def _cid(store: CampaignStore) -> str:
    return store.create_campaign(tenant="acme", name="c")["id"]


def _counts(store: CampaignStore) -> tuple[int, int]:
    c = store._conn
    return (
        c.execute("SELECT COUNT(*) FROM virtual_runs").fetchone()[0],
        c.execute("SELECT COUNT(*) FROM audit_log WHERE action = ?", (vr.AUDIT_CREATE,)).fetchone()[0],
    )


def test_create_writes_one_audit_row_with_the_batch(store):
    cid = _cid(store)
    out = vr.create_batch(store, tenant="acme", campaign_id=cid, generator_id="bench.gaussian_bump",
                          params={"dim": 2}, n=4, requested_by="bob@acme", auth_method="jwt",
                          audit=True, request_id="req-1")
    assert _counts(store) == (4, 1)
    (entry,) = [e for e in store.audit_entries(tenant="acme") if e["action"] == vr.AUDIT_CREATE]
    assert entry["subject"] == "bob@acme" and entry["auth_method"] == "jwt"
    assert entry["campaign_id"] == cid and entry["job_id"] == out["batch_id"]
    assert entry["outcome"] == "ok" and entry["request_id"] == "req-1"
    detail = json.loads(entry["detail"])
    assert detail == {
        "batch_id": out["batch_id"], "origin": "synthetic", "generator_id": "bench.gaussian_bump",
        "generator_version": out["generator_version"], "n": 4,
        "params_sha256": out["params_sha256"], "twin_version_id": None,
    }
    # the audit row and the rows share the server timestamp of the write
    ts = store._conn.execute("SELECT DISTINCT created_at FROM virtual_runs").fetchall()
    assert ts == [(entry["ts"],)]


def test_a_failing_audit_insert_stores_no_rows(store, monkeypatch):
    cid = _cid(store)

    def boom(*a, **k):
        raise sqlite3.OperationalError("audit unavailable")

    monkeypatch.setattr(CampaignStore, "_insert_audit", staticmethod(boom))
    with pytest.raises(sqlite3.OperationalError):
        vr.create_batch(store, tenant="acme", campaign_id=cid, generator_id="bench.ackley",
                        params=None, n=3, requested_by="bob", audit=True)
    assert _counts(store) == (0, 0)


def test_a_failing_row_insert_writes_no_audit_row(store, monkeypatch):
    cid = _cid(store)
    real = vr.gen.generate

    def bad_generate(*a, **k):
        b = real(*a, **k)
        object.__setattr__(b, "params_sha256", "short")  # violates the CHECK
        return b

    monkeypatch.setattr(vr.gen, "generate", bad_generate)
    with pytest.raises(sqlite3.IntegrityError):
        vr.create_batch(store, tenant="acme", campaign_id=cid, generator_id="bench.ackley",
                        params=None, n=3, requested_by="bob", audit=True)
    assert _counts(store) == (0, 0)


def test_without_audit_flag_no_audit_row(store):
    cid = _cid(store)
    vr.create_batch(store, tenant="acme", campaign_id=cid, generator_id="bench.ackley",
                    params=None, n=2, requested_by="bob")
    assert _counts(store) == (2, 0)


def test_routes_audit_create_refusals_and_exports(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.campaign import get_campaign_store

    s = CampaignStore(tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", s)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: s
    try:
        tc = TestClient(portal_module.app)
        cid = tc.post("/api/campaigns", json={"name": "c"}).json()["id"]
        ok = tc.post(f"/api/campaigns/{cid}/virtual-runs", json={"generator": "bench.ackley", "n": 2})
        assert ok.status_code == 201
        bad = tc.post(f"/api/campaigns/{cid}/virtual-runs", json={"generator": "nope", "n": 2})
        assert bad.status_code == 400
        assert tc.get(f"/api/campaigns/{cid}/virtual-runs/export").status_code == 200
        entries = [(e["action"], e["outcome"]) for e in s.audit_entries(tenant="default")]
        assert (vr.AUDIT_CREATE, "ok") in entries
        assert (vr.AUDIT_CREATE, "http_400") in entries
        assert (vr.AUDIT_EXPORT, "ok") in entries
        assert sum(1 for a, o in entries if a == vr.AUDIT_CREATE and o == "ok") == 1
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)
        s.close()
