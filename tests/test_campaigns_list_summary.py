"""`GET /api/campaigns?summary=true`: the campaigns list with per-campaign card
fields (kalos/portal/campaign_list.py). Default response unchanged; summary is
tenant-scoped, paginated, and costs a constant number of SQL statements."""
from __future__ import annotations

import hashlib
import json
import time

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.campaign_list import (
    InvalidCursor,
    decode_cursor,
    encode_cursor,
    list_campaign_summaries,
)
from kalos.portal.campaign_settings import put_setting


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]
A = {"Authorization": "Bearer a-read"}
B = {"Authorization": "Bearer b-admin"}
B_SECRET = "GLOBEX-SECRET-Project-Hera"


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


DF = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "titer": [5.0, None, 2.0, 7.5]})
ANALYSIS = {
    "target": "titer", "engine_version": "9.9.9", "cv_spearman": 0.41, "conformal_q": 0.7,
    "conformal_coverage": 0.9, "proposals": [{"recipe": {"x": 1}}, {"recipe": {"x": 2}}],
    "reliability": {"clears_floor": True, "ci_excludes_zero": False, "spearman": 0.41,
                    "ci95": [-0.1, 0.7], "spearman_floor": 0.2, "producer_clears_floor": False},
}


def _seeded(store: CampaignStore, tenant: str = "acme", name: str = "c") -> str:
    cid = store.create_campaign(tenant=tenant, name=name)["id"]
    store.seed(DF, "titer", ["x"], tenant=tenant, campaign_id=cid, analysis=ANALYSIS,
               dataset="d.csv")
    return cid


def _sql(store: CampaignStore, sql: str, params: tuple) -> None:
    with store._txn(write=True) as conn:
        conn.execute(sql, params)


def _summary(tc: TestClient, headers=A, **params) -> dict:
    r = tc.get("/api/campaigns", params={"summary": "true", **params}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _by_id(body: dict) -> dict[str, dict]:
    return {c["id"]: c for c in body["campaigns"]}


# --- default response unchanged ------------------------------------------------- #


def test_default_list_is_unchanged(env):
    tc, store = env
    _seeded(store)
    store.create_campaign(tenant="acme", name="empty")
    r = tc.get("/api/campaigns", headers=A)
    assert r.status_code == 200
    assert r.json() == {"campaigns": store.list_campaigns(tenant="acme")}
    assert r.content == json.dumps(
        {"campaigns": store.list_campaigns(tenant="acme")}, separators=(",", ":")
    ).encode()
    r = tc.get("/api/campaigns", params={"summary": "false"}, headers=A)
    assert r.json() == {"campaigns": store.list_campaigns(tenant="acme")}


@pytest.mark.parametrize("params", [{"limit": "5"}, {"cursor": "abc"}])
def test_pagination_params_without_summary_are_refused(env, params):
    tc, _ = env
    r = tc.get("/api/campaigns", params=params, headers=A)
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_query"


# --- summary fields --------------------------------------------------------------- #


def test_summary_fields_for_seeded_and_empty_campaigns(env):
    tc, store = env
    cid = _seeded(store, name="seeded")
    empty = store.create_campaign(tenant="acme", name="empty")["id"]
    body = _summary(tc)
    assert body["next_cursor"] is None and body["limit"] == 50
    items = _by_id(body)
    c = items[cid]
    info = store.campaign_info(tenant="acme", campaign_id=cid)
    assert {k: c[k] for k in info} == info  # a superset of the plain list item
    assert c["has_data"] is True and c["target"] == "titer"
    assert c["n_base"] == 4 and c["round"] == 0
    assert c["n_awaiting"] == 0 and c["n_measured"] == 0
    assert c["goal"] == {"direction": "maximize", "source": "default"}
    assert c["best_measured"] == {"value": 7.5, "basis": "measured", "direction": "max"}
    assert c["analysis"] == {
        "present": True, "updated": c["analysis"]["updated"], "in_sync": True,
        "engine_version": "9.9.9", "cv_spearman": 0.41, "conformal_q": 0.7,
        "conformal_coverage": 0.9, "n_proposals": 2,
        "reliability": ANALYSIS["reliability"],
    }
    assert isinstance(c["analysis"]["updated"], float)
    assert c["current_stage"] is None and c["latest_gate_decision"] is None
    assert c["active_job"] is None

    e = items[empty]
    assert e["has_data"] is False and e["seeded"] is False and e["target"] is None
    assert e["n_base"] == 0 and e["round"] is None
    assert e["best_measured"]["value"] is None and e["best_measured"]["reason"]
    assert e["analysis"] == {"present": False}


def test_runs_and_minimize_goal(env):
    tc, store = env
    cid = _seeded(store)
    runs = store.start([{"recipe": {"x": 1.0}}, {"recipe": {"x": 2.0}}, {"recipe": {"x": 3.0}}],
                       tenant="acme", campaign_id=cid)
    store.set_result(runs[0]["id"], 0.5, tenant="acme", campaign_id=cid)
    put_setting(store, tenant="acme", campaign_id=cid, kind="setup",
                value={"goal": "minimize"}, updated_by="x")
    c = _by_id(_summary(tc))[cid]
    assert c["n_awaiting"] == 2 and c["n_measured"] == 1
    # measured-but-unfolded results are not base data yet; best is over base rows
    assert c["goal"] == {"direction": "minimize", "source": "setup"}
    assert c["best_measured"] == {"value": 2.0, "basis": "measured", "direction": "min"}
    # starting runs moved the generation past the published analysis
    assert c["analysis"]["in_sync"] is False


def test_unknown_goal_falls_back_to_max_and_says_so(env):
    tc, store = env
    cid = _seeded(store)
    put_setting(store, tenant="acme", campaign_id=cid, kind="setup",
                value={"goal": "sideways"}, updated_by=None)
    c = _by_id(_summary(tc))[cid]
    assert c["goal"] == {"direction": "maximize", "source": "default"}
    assert c["best_measured"]["value"] == 7.5


def test_best_ignores_booleans_strings_and_missing(env):
    tc, store = env
    cid = store.create_campaign(tenant="acme", name="mixed")["id"]
    df = pd.DataFrame({"x": [1, 2, 3, 4], "t": [True, "9", None, 3.0]})
    store.seed(df, "t", ["x"], tenant="acme", campaign_id=cid)
    assert _by_id(_summary(tc))[cid]["best_measured"]["value"] == 3.0


def test_stage_gate_and_active_job(env):
    tc, store = env
    cid = _seeded(store)
    put_setting(store, tenant="acme", campaign_id=cid, kind="stages",
                value={"current_stage": "stage_3", "stages": []}, updated_by=None)
    now = time.time()
    for gate, decision, at in (("stage_1", "go", now - 10), ("stage_2", "no_go", now - 5)):
        _sql(store, "INSERT INTO gate_decisions (tenant, campaign_id, gate, decision, note, "
             "decided_at) VALUES (?, ?, ?, ?, 'n', ?)", ("acme", cid, gate, decision, at))
    for jid, state, created in (("j_done", "succeeded", now - 9), ("j_old", "queued", now - 8),
                                ("j_new", "running", now - 1)):
        _sql(store, "INSERT INTO jobs (id, tenant, campaign_id, kind, state, created_at, "
             "updated_at, progress, stage) VALUES (?, 'acme', ?, 'analysis', ?, ?, ?, 0.25, 'fit')",
             (jid, cid, state, created, created))
    c = _by_id(_summary(tc))[cid]
    assert c["current_stage"] == "stage_3"
    assert c["latest_gate_decision"] == {"gate": "stage_2", "decision": "no_go",
                                         "decided_at": now - 5}
    assert c["active_job"] == {"id": "j_new", "kind": "analysis", "state": "running",
                               "progress": 0.25, "stage": "fit"}


def test_archived_only_with_include_archived(env):
    tc, store = env
    cid = _seeded(store)
    store.archive_campaign(tenant="acme", campaign_id=cid)
    assert cid not in _by_id(_summary(tc))
    c = _by_id(_summary(tc, include_archived="true"))[cid]
    assert c["archived"] is True and c["n_base"] == 4


# --- robustness ------------------------------------------------------------------ #


def test_invalid_json_blobs_degrade_gracefully(env):
    tc, store = env
    nan_state = store.create_campaign(tenant="acme", name="nan")["id"]
    garbage = store.create_campaign(tenant="acme", name="garbage")["id"]
    state = ('{"target": "t", "round": 1, "base_rows": [{"t": NaN}, {"t": 4.0}], '
             '"pending": [{"result": null}]}')
    _sql(store, "UPDATE campaigns SET state = ? WHERE id = ? AND tenant = ?",
         (state, nan_state, "acme"))
    _sql(store, "UPDATE campaigns SET state = ? WHERE id = ? AND tenant = ?",
         ("{not json", garbage, "acme"))
    for cid, blob in ((nan_state, '{"cv_spearman": NaN, "engine_version": "1"}'),
                      (garbage, "nope")):
        _sql(store, "INSERT INTO analyses (campaign_id, tenant, state, updated_at) "
             "VALUES (?, 'acme', ?, 1.0)", (cid, blob))
    items = _by_id(_summary(tc))
    n = items[nan_state]
    assert n["n_base"] == 2 and n["round"] == 1 and n["n_awaiting"] == 1
    assert n["best_measured"]["value"] == 4.0
    assert n["analysis"]["engine_version"] == "1" and n["analysis"]["cv_spearman"] is None
    g = items[garbage]
    assert g["seeded"] is True and g["n_base"] is None and g["has_data"] is False
    assert g["best_measured"]["value"] is None
    assert g["analysis"]["present"] is True and g["analysis"]["engine_version"] is None


def test_constant_statement_count(env):
    _, store = env

    def count() -> int:
        seen: list[str] = []
        store._conn.set_trace_callback(seen.append)
        try:
            list_campaign_summaries(store, tenant="acme")
        finally:
            store._conn.set_trace_callback(None)
        return len(seen)

    for i in range(2):
        _seeded(store, name=f"c{i}")
    small = count()
    for i in range(8):
        _seeded(store, name=f"d{i}")
    assert count() == small <= 8


# --- pagination ------------------------------------------------------------------- #


def test_pagination_walks_every_campaign_once(env):
    tc, store = env
    ids = [_seeded(store, name=f"c{i}") for i in range(5)]
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        params = {"limit": "2", **({"cursor": cursor} if cursor else {})}
        body = _summary(tc, **params)
        pages += 1
        seen += [c["id"] for c in body["campaigns"]]
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert pages == 3 and sorted(seen) == sorted(ids) and len(set(seen)) == 5
    assert seen == [c["id"] for c in _summary(tc)["campaigns"]]  # same order as one page


@pytest.mark.parametrize("limit", ["0", "201", "-1", "x"])
def test_limit_out_of_range_is_400(env, limit):
    tc, _ = env
    r = tc.get("/api/campaigns", params={"summary": "true", "limit": limit}, headers=A)
    assert r.status_code == 400


@pytest.mark.parametrize("cursor", ["!!!", "W10", encode_cursor(1.0, "cmp_x"), "a" * 300])
def test_bad_cursor_is_400_and_not_echoed(env, cursor):
    tc, _ = env
    r = tc.get("/api/campaigns", params={"summary": "true", "cursor": cursor}, headers=A)
    assert r.status_code == 400
    assert cursor not in r.text


def test_cursor_roundtrip():
    cid = "cmp_" + "a" * 32
    assert decode_cursor(encode_cursor(12.5, cid)) == (12.5, cid)
    with pytest.raises(InvalidCursor):
        decode_cursor(encode_cursor(float("inf"), cid))


# --- tenancy + auth ---------------------------------------------------------------- #


def test_summary_is_tenant_scoped(env):
    tc, store = env
    mine = _seeded(store, name="acme-one")
    theirs = _seeded(store, tenant="globex", name=B_SECRET)
    put_setting(store, tenant="globex", campaign_id=theirs, kind="stages",
                value={"current_stage": B_SECRET}, updated_by=None)
    _sql(store, "INSERT INTO gate_decisions (tenant, campaign_id, gate, decision, note, "
         "decided_at) VALUES ('globex', ?, ?, 'go', 'n', 1.0)", (theirs, B_SECRET))
    _sql(store, "INSERT INTO jobs (id, tenant, campaign_id, kind, state, created_at, "
         "updated_at) VALUES ('j_b', 'globex', ?, 'analysis', 'running', 1.0, 1.0)", (theirs,))
    r = tc.get("/api/campaigns", params={"summary": "true"}, headers=A)
    assert [c["id"] for c in r.json()["campaigns"]] == [mine]
    assert B_SECRET not in r.text and theirs not in r.text and "j_b" not in r.text
    b = _by_id(_summary(tc, headers=B))[theirs]
    assert b["current_stage"] == B_SECRET and b["active_job"]["id"] == "j_b"
    # a cursor never crosses tenants: acme paging past globex's id sees only acme
    body = _summary(tc, cursor=encode_cursor(time.time() + 100, theirs))
    assert [c["id"] for c in body["campaigns"]] == [mine]


def test_summary_requires_auth(env):
    tc, _ = env
    assert tc.get("/api/campaigns", params={"summary": "true"}).status_code == 401
