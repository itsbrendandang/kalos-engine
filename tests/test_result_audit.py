"""The `campaign.result` audit row carries the change (E2): run id, old value, new
value, reason and actor, and it is written in the same transaction as the change.
A change that cannot be audited is not made. Refused submissions keep the
best-effort row they always had. Every database is in a pytest tmp dir.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.campaign import CampaignStore, get_campaign_store

RECIPE = {"Methanol": 2.5, "pH": 6.5}


def _sheet() -> pd.DataFrame:
    return pd.DataFrame([{"Methanol": 1.0 + i * 0.5, "pH": 6.0 + (i % 3) * 0.2,
                          "lipase_titer": 2.0 + i} for i in range(6)])


@pytest.fixture
def store(tmp_path):
    s = CampaignStore(tmp_path)
    s.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    yield s
    s.close()


@pytest.fixture
def client(store, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app)
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


def _start(store: CampaignStore) -> str:
    return store.start([{"recipe": RECIPE, "pred": 4.0, "std": 0.2, "mode": "explore",
                         "reason": "gap"}], tenant="default")[0]["id"]


def _result_rows(store: CampaignStore) -> list[dict]:
    return [e for e in store.audit_entries(tenant="default", limit=1000)
            if e["action"] == "campaign.result"]


def test_correction_audit_row_carries_old_new_reason_and_actor(client, store):
    run_id = _start(store)
    client.post("/api/campaign/result", json={"id": run_id, "value": 7.0})
    resp = client.post("/api/campaign/result",
                       json={"id": run_id, "value": 8.5, "reason": "re-weighed"})
    assert resp.status_code == 200

    newest = _result_rows(store)[0]  # audit_entries is newest first
    assert newest["outcome"] == "ok"
    assert newest["subject"] == "anonymous" and newest["auth_method"] == "anonymous"
    assert newest["campaign_id"] == store.default_campaign_id(tenant="default")
    detail = json.loads(newest["detail"])
    assert detail == {"run_id": run_id, "version": 2, "old_value": 7.0, "new_value": 8.5,
                      "reason": "re-weighed", "actor": "anonymous"}


def test_first_value_audit_row_has_no_old_value(client, store):
    run_id = _start(store)
    client.post("/api/campaign/result", json={"id": run_id, "value": 7.0})
    detail = json.loads(_result_rows(store)[0]["detail"])
    assert detail["old_value"] is None and detail["new_value"] == 7.0
    assert detail["reason"] is None and detail["version"] == 1


def test_explicit_campaign_route_writes_the_same_row(client, store):
    run_id = _start(store)
    cid = store.default_campaign_id(tenant="default")
    url = f"/api/campaigns/{cid}/runs/{run_id}/result"
    client.post(url, json={"value": 7.0})
    assert client.post(url, json={"value": 6.0, "reason": "transcription"}).status_code == 200
    newest = _result_rows(store)[0]
    assert newest["outcome"] == "ok" and newest["campaign_id"] == cid
    assert json.loads(newest["detail"])["old_value"] == 7.0


def test_refused_correction_keeps_its_best_effort_row(client, store):
    run_id = _start(store)
    client.post("/api/campaign/result", json={"id": run_id, "value": 7.0})
    assert client.post("/api/campaign/result", json={"id": run_id, "value": 8.0}).status_code == 400
    newest = _result_rows(store)[0]
    assert newest["outcome"] == "http_400"
    assert newest["detail"] is None  # no change, so no change content
    assert [r["value"] for r in store.result_history(run_id, tenant="default")] == [7.0]


def test_a_failed_audit_write_fails_the_change(store, monkeypatch):
    run_id = _start(store)
    before = store.get(tenant="default")

    def _broken(*_a, **kw):
        if kw.get("action") == "campaign.result":
            raise RuntimeError("audit disk full")
        return _real(*_a, **kw)

    _real = CampaignStore._insert_audit
    monkeypatch.setattr(CampaignStore, "_insert_audit", staticmethod(_broken))
    with pytest.raises(RuntimeError, match="audit disk full"):
        store.set_result(run_id, 7.0, tenant="default", audit=True, actor="alice")

    assert store.get(tenant="default") == before  # the run is still unmeasured
    assert store.result_history(run_id, tenant="default") == []  # no history row either


def test_a_failed_audit_write_is_a_server_error_on_the_route(store, monkeypatch, client):
    run_id = _start(store)

    def _broken(*_a, **kw):
        if kw.get("action") == "campaign.result":
            raise RuntimeError("audit disk full")
        return _real(*_a, **kw)

    _real = CampaignStore._insert_audit
    monkeypatch.setattr(CampaignStore, "_insert_audit", staticmethod(_broken))
    from fastapi.testclient import TestClient

    tc = TestClient(portal_module.app, raise_server_exceptions=False)
    resp = tc.post("/api/campaign/result", json={"id": run_id, "value": 7.0})
    assert resp.status_code == 500
    assert store.result_history(run_id, tenant="default") == []
