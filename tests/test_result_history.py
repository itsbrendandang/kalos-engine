"""Append-only recorded results (schema v5, docs/CAMPAIGNS_API.md "Recording results").

A submitted value is never overwritten. Each submission is a row in `run_results`
(value, the value it replaced, reason, actor, server time). The first value needs
no reason; a correction needs one. The campaign state keeps the latest value, so
the fold and the summary read what they always read. The history table rejects
UPDATE and DELETE. Every database is in a pytest tmp dir.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal import migrate
from kalos.portal.campaign import (
    CampaignReasonRequired,
    CampaignStore,
    get_campaign_store,
)

RECIPE = {"Methanol": 2.5, "pH": 6.5}


def _sheet() -> pd.DataFrame:
    rows = [{"Methanol": 1.0 + i * 0.5, "pH": 6.0 + (i % 3) * 0.2, "lipase_titer": 2.0 + i}
            for i in range(6)]
    return pd.DataFrame(rows)


@pytest.fixture
def store(tmp_path):
    s = CampaignStore(tmp_path)
    # "acme" for the store tests; "default" (open mode's tenant) for the HTTP tests
    s.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme")
    s.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    yield s
    s.close()


def _start(store: CampaignStore, tenant: str = "acme") -> str:
    return store.start([{"recipe": RECIPE, "pred": 4.0, "std": 0.2, "mode": "explore",
                         "reason": "gap"}], tenant=tenant)[0]["id"]


def _history_rows(store: CampaignStore, run_id: str, tenant: str = "acme") -> list[dict]:
    return store.result_history(run_id, tenant=tenant)


# --- the store --------------------------------------------------------------- #


def test_first_value_needs_no_reason_and_is_version_one(store):
    run_id = _start(store)
    out = store.set_result(run_id, 7.0, tenant="acme", actor="alice", auth_method="bearer")
    assert out["result"] == 7.0
    assert out["result_version"] == 1
    assert out["previous_result"] is None
    assert out["result_reason"] is None
    assert out["result_actor"] == "alice"
    assert out["result_recorded_at"] > 0
    (row,) = _history_rows(store, run_id)
    assert (row["version"], row["value"], row["previous_value"], row["reason"]) == (1, 7.0, None, None)
    assert row["actor"] == "alice" and row["auth_method"] == "bearer"


def test_correction_without_reason_is_refused_and_writes_nothing(store):
    run_id = _start(store)
    store.set_result(run_id, 7.0, tenant="acme")
    before = store.get(tenant="acme")
    with pytest.raises(CampaignReasonRequired):
        store.set_result(run_id, 8.0, tenant="acme")
    with pytest.raises(CampaignReasonRequired):
        store.set_result(run_id, 8.0, tenant="acme", reason="   ")  # whitespace is no reason
    assert store.get(tenant="acme") == before
    assert [r["value"] for r in _history_rows(store, run_id)] == [7.0]


def test_correction_with_reason_appends_and_keeps_the_old_value(store):
    run_id = _start(store)
    store.set_result(run_id, 7.0, tenant="acme", actor="alice")
    out = store.set_result(run_id, 7.4, tenant="acme", reason="  assay re-run  ", actor="bob")

    assert out["result"] == 7.4 and out["result_version"] == 2
    assert out["previous_result"] == 7.0
    assert out["result_reason"] == "assay re-run"
    assert out["result_actor"] == "bob"
    history = _history_rows(store, run_id)
    assert [(r["version"], r["value"], r["previous_value"], r["reason"], r["actor"])
            for r in history] == [
        (1, 7.0, None, None, "alice"),
        (2, 7.4, 7.0, "assay re-run", "bob"),
    ]
    # the current value is the latest: the state and the fold read 7.4
    run = next(r for r in store.get(tenant="acme")["pending"] if r["id"] == run_id)
    assert run["result"] == 7.4
    df, _target, _gen = store.plan_fold(tenant="acme")
    folded = [v for v in df["lipase_titer"] if v not in set(_sheet()["lipase_titer"])]
    assert folded == [7.4]  # the folded run carries the latest value, once


def test_history_rejects_update_and_delete_at_the_database_level(store, tmp_path):
    run_id = _start(store)
    store.set_result(run_id, 7.0, tenant="acme")
    store.set_result(run_id, 7.1, tenant="acme", reason="re-read")
    store.close()
    conn = sqlite3.connect(str(tmp_path / "portal.db"), isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="run_results is append-only: UPDATE"):
            conn.execute("UPDATE run_results SET value = 0")
        with pytest.raises(sqlite3.IntegrityError, match="run_results is append-only: DELETE"):
            conn.execute("DELETE FROM run_results")
        # a correction row without a reason is refused by the CHECK, not only by the store
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                "INSERT INTO run_results (tenant, campaign_id, run_id, version, value, "
                "previous_value, reason, actor, auth_method, recorded_at) "
                "SELECT tenant, campaign_id, run_id, 3, 9.0, 7.1, NULL, NULL, NULL, 1.0 "
                "FROM run_results WHERE version = 1"
            )
    finally:
        conn.close()
    assert [r["value"] for r in _history_rows(CampaignStore(tmp_path), run_id)] == [7.0, 7.1]


def test_history_is_tenant_scoped(store):
    run_id = _start(store)
    store.set_result(run_id, 7.0, tenant="acme")
    assert store.result_history(run_id, tenant="globex") == []


def test_v4_state_with_a_measured_run_needs_a_reason_after_migrating_to_v5(tmp_path):
    """A run measured before v5 has a value and no history. Its first v5 correction
    needs a reason, and the history records the value it replaces."""
    store = CampaignStore(tmp_path)
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme")
    run_id = _start(store)
    store.set_result(run_id, 7.0, tenant="acme")
    store.close()
    db = tmp_path / "portal.db"
    migrate.migrate_down(db, target=4)  # the blob keeps 7.0; run_results is renamed aside
    migrate.migrate_up(db)
    store = CampaignStore(tmp_path)
    with pytest.raises(CampaignReasonRequired):
        store.set_result(run_id, 7.5, tenant="acme")
    out = store.set_result(run_id, 7.5, tenant="acme", reason="transcription")
    assert out["previous_result"] == 7.0 and out["result_version"] == 1
    store.close()


# --- the HTTP route ---------------------------------------------------------- #


@pytest.fixture
def client(store, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app)
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


def test_route_keeps_the_old_response_fields_and_adds_the_new_ones(client, store):
    run_id = _start(store, "default")
    body = client.post("/api/campaign/result", json={"id": run_id, "value": 7.0}).json()
    # the fields a v1 client reads are still there
    for key in ("id", "recipe", "pred", "std", "mode", "reason", "result", "round",
                "created_at", "measured_at"):
        assert key in body
    assert body["reason"] == "gap"  # the PROPOSAL reason, not the correction reason
    assert body["result"] == 7.0
    # the additive fields
    assert body["result_version"] == 1
    assert body["previous_result"] is None
    assert body["result_reason"] is None
    assert body["result_actor"] == "anonymous"  # open mode, as the audit row says
    assert "result_recorded_at" in body


def test_route_refuses_a_correction_without_a_reason_with_a_code(client, store):
    run_id = _start(store, "default")
    assert client.post("/api/campaign/result", json={"id": run_id, "value": 7.0}).status_code == 200
    resp = client.post("/api/campaign/result", json={"id": run_id, "value": 8.0})
    assert resp.status_code == 400
    assert resp.json()["code"] == "reason_required"
    assert "reason" in resp.json()["error"]
    assert [r["value"] for r in _history_rows(store, run_id, "default")] == [7.0]


def test_route_accepts_a_correction_with_a_reason(client, store):
    run_id = _start(store, "default")
    client.post("/api/campaign/result", json={"id": run_id, "value": 7.0})
    resp = client.post("/api/campaign/result",
                       json={"id": run_id, "value": 8.0, "reason": "re-weighed the sample"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_version"] == 2 and body["previous_result"] == 7.0
    assert body["result_reason"] == "re-weighed the sample"


def test_route_bounds_the_reason(client, store):
    run_id = _start(store)
    resp = client.post("/api/campaign/result", json={"id": run_id, "value": 7.0, "reason": "x" * 257})
    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_body"


def test_route_summary_still_reports_the_latest_value(client, store):
    run_id = _start(store, "default")
    client.post("/api/campaign/result", json={"id": run_id, "value": 7.0})
    client.post("/api/campaign/result", json={"id": run_id, "value": 7.9, "reason": "re-read"})
    summary = client.get("/api/campaign").json()
    (run,) = [r for r in summary["pending"] if r["id"] == run_id]
    assert run["result"] == 7.9
    assert summary["n_measured"] == 1 and summary["n_awaiting"] == 0


def test_campaigns_api_result_route_takes_the_same_reason_rule(client, store):
    """The explicit-campaign route (`/api/campaigns/{id}/runs/{run_id}/result`) has
    the same rule: a correction needs a reason, and says so with a code."""
    run_id = _start(store, "default")
    cid = store.default_campaign_id(tenant="default")
    url = f"/api/campaigns/{cid}/runs/{run_id}/result"
    assert client.post(url, json={"value": 7.0}).status_code == 200
    resp = client.post(url, json={"value": 8.0})
    assert resp.status_code == 400 and resp.json()["code"] == "reason_required"
    ok = client.post(url, json={"value": 8.0, "reason": "re-read"})
    assert ok.status_code == 200 and ok.json()["result_version"] == 2
    assert ok.json()["previous_result"] == 7.0
