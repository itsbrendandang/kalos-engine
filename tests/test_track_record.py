"""Prediction history and the prospective track record (schema v8, BRE-32).

- Wilson score intervals against hand-computed values;
- a 3-round round trip through the store (start, record, correct, fold, x3)
  with hand-computed inside/outside, coverage so far and errors; the band is the
  one recorded at start, never the newer analysis; folded runs keep their id,
  prediction and band; every published analysis is kept;
- honesty: a run whose result predates its prediction is never scored;
- a re-upload starts a new dataset epoch;
- the routes: scope, tenant isolation (byte-identical 404), validation,
  pagination, the per-run history (corrections and reasons), the snapshots;
- every statement on a tenant table names the tenant;
- (slow) the same round trip with the real engine over HTTP.
"""
from __future__ import annotations

import hashlib
import io
import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal import migrate
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.history_api import MAX_ROW_ID
from kalos.portal.track_record import CAVEAT_NOT_RECORDED, MIN_JUDGED_N, wilson_interval


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
    {"token_sha256": _sha("a-write"), "subject": "wally@acme", "tenant": "acme", "scopes": ["write"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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
        store.close()


# --- Wilson ------------------------------------------------------------------------ #


@pytest.mark.parametrize(("k", "n", "lo", "hi"), [
    (2, 3, 0.2077, 0.9385),     # hand-computed, z = 1.96
    (9, 10, 0.5958, 0.9821),
    (0, 5, 0.0, 0.4345),
    (10, 10, 0.7225, 1.0),
])
def test_wilson_interval_matches_hand_computed_values(k, n, lo, hi):
    got = wilson_interval(k, n)
    assert got is not None
    assert got[0] == pytest.approx(lo, abs=1e-4) and got[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_interval_with_no_runs_is_none():
    assert wilson_interval(0, 0) is None


# --- a 3-round round trip through the store -------------------------------------------- #


def _sheet() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, 8).round(3)
    ph = rng.uniform(5, 7, 8).round(2)
    return pd.DataFrame({"Methanol": m, "pH": ph, "lipase_titer": (1.5 * m + 1).round(3)})


def _analysis(tag: str, proposals: list[tuple[float, float, float, float, float]], q: float) -> dict:
    """A published analysis with per-proposal 90/95% intervals (the shape
    `analysis_parts/intervals.py` writes) and a global conformal band."""
    props = []
    for m, ph, pred, w90, w95 in proposals:
        props.append({
            "recipe": {"Methanol": m, "pH": ph}, "pred": pred, "std": 0.1, "mode": "exploit",
            "intervals": [
                {"coverage": 0.9, "method": "conformal_normalized", "coverage_basis": "held_out_past_runs",
                 "caveat": {"code": "coverage_lower_at_proposals", "message": f"{tag} caveat at 90",
                            "coverage_at_proposals_estimate": None},
                 "available": True, "lo": round(pred - w90, 4), "hi": round(pred + w90, 4)},
                {"coverage": 0.95, "method": "conformal_normalized", "coverage_basis": "held_out_past_runs",
                 "caveat": {"code": "coverage_lower_at_proposals", "message": f"{tag} caveat at 95",
                            "coverage_at_proposals_estimate": None},
                 "available": True, "lo": round(pred - w95, 4), "hi": round(pred + w95, 4)},
            ],
        })
    return {"target": "lipase_titer", "tag": tag, "conformal_q": q, "conformal_coverage": 0.9,
            "conformal_caveat": {"code": "coverage_not_guaranteed", "coverage_basis": "held_out_past_runs",
                                 "message": f"{tag} global band caveat", "caveat_code": "x"},
            "proposals": props,
            "proposal_provenance": {"provenance_sha256": _sha(tag), "input_sha256": _sha("in" + tag),
                                    "config_sha256": _sha("c"), "engine_version": "0.1.0", "seed": 1,
                                    "timestamp": 1}}


A0 = _analysis("a0", [(1.0, 6.0, 3.0, 0.5, 0.7), (2.0, 6.5, 4.0, 0.5, 0.7), (3.0, 5.5, 5.0, 0.5, 0.7)], 0.6)
A1 = _analysis("a1", [(2.5, 6.0, 4.5, 0.4, 0.6), (3.5, 6.0, 5.5, 0.4, 0.6)], 0.3)
A2 = _analysis("a2", [(3.8, 6.1, 6.0, 0.5, 0.8), (3.9, 5.9, 6.1, 0.5, 0.8)], 0.3)


def _recipes(a: dict) -> list[dict]:
    return [{"recipe": dict(p["recipe"]), "pred": 0.0, "std": 0.0, "mode": "x", "reason": "x"}
            for p in a["proposals"]]


def _three_rounds(store: CampaignStore, tenant: str = "acme") -> tuple[str, dict[str, list[str]]]:
    cid = store.create_campaign(tenant=tenant, name="loop")["id"]
    kw = {"tenant": tenant, "campaign_id": cid}
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], analysis=A0, dataset="runs.csv", **kw)
    ids: dict[str, list[str]] = {}
    # round 0: three proposals and one edited recipe (no engine prediction)
    runs = store.start(_recipes(A0) + [{"recipe": {"Methanol": 1.5, "pH": 6.2}, "pred": 9.9}], **kw)
    ids["r0"] = [r["id"] for r in runs]
    for rid, v in zip(ids["r0"], (3.2, 5.0, 4.4, 2.0)):
        store.set_result(rid, v, actor="alice", **kw)
    store.set_result(ids["r0"][2], 4.8, reason="re-read the plate", actor="alice", **kw)
    store.commit_fold(store.plan_fold(**kw)[2], analysis=A1, **kw)
    # round 1
    ids["r1"] = [r["id"] for r in store.start(_recipes(A1), **kw)]
    for rid, v in zip(ids["r1"], (4.6, 6.5)):
        store.set_result(rid, v, **kw)
    store.commit_fold(store.plan_fold(**kw)[2], analysis=A2, **kw)
    # round 2: one measured, one awaiting
    ids["r2"] = [r["id"] for r in store.start(_recipes(A2), **kw)]
    store.set_result(ids["r2"][0], 6.2, **kw)
    return cid, ids


def _q(store: CampaignStore, sql: str, params: tuple = ()) -> list[tuple]:
    with store._txn(write=False) as conn:
        return conn.execute(sql, params).fetchall()


def test_three_round_trip_keeps_every_prediction_band_run_id_and_analysis(env):
    tc, store = env
    cid, ids = _three_rounds(store)
    # every started run kept its prediction row, folded or not
    preds = {r[0]: r for r in _q(store, "SELECT run_id, round, available, pred, lo_90, hi_90, lo_95, "
                                        "band_lo, band_hi, analysis_snapshot_id, model_version, "
                                        "analysis_sha256 FROM run_predictions WHERE tenant = 'acme'")}
    assert set(preds) == set(ids["r0"] + ids["r1"] + ids["r2"])
    assert [preds[i][1] for i in ids["r0"]] == [0, 0, 0, 0] and preds[ids["r2"][1]][1] == 2
    assert preds[ids["r0"][3]][2] == 0  # the edited recipe: no engine prediction
    assert preds[ids["r0"][1]][3:9] == (4.0, 3.5, 4.5, 3.3, 3.4, 4.6)  # A0's bands, q = 0.6
    assert preds[ids["r1"][0]][7:9] == (4.2, 4.8)  # A1's global band, q = 0.3
    # three snapshots, one per published analysis, and the runs point at theirs
    snaps = _q(store, "SELECT id, round, kind, analysis_sha256, analysis FROM analysis_snapshots "
                      "WHERE tenant = 'acme' ORDER BY id")
    assert [(s[1], s[2]) for s in snaps] == [(0, "upload"), (1, "reanalyze"), (2, "reanalyze")]
    assert [json.loads(s[4])["tag"] for s in snaps] == ["a0", "a1", "a2"]
    assert len({s[3] for s in snaps}) == 3
    assert {preds[i][9] for i in ids["r0"][:3]} == {snaps[0][0]}
    assert {preds[i][9] for i in ids["r1"]} == {snaps[1][0]}
    assert preds[ids["r0"][0]][11] == snaps[0][3] and preds[ids["r0"][0]][10] == "0.1.0"
    assert store.load_analysis(tenant="acme", campaign_id=cid)["tag"] == "a2"  # only the newest is live
    # the folds: which value and version went into base_rows
    folds = {r[0]: r[1:] for r in _q(store, "SELECT run_id, folded_in_round, result_version, value, "
                                            "round_started FROM run_folds WHERE tenant = 'acme'")}
    assert set(folds) == set(ids["r0"] + ids["r1"])
    assert folds[ids["r0"][2]] == (1, 2, 4.8, 0)  # the corrected value, version 2
    assert folds[ids["r1"][1]] == (2, 1, 6.5, 1)

    body = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()
    assert body["basis"] == "prospective" and body["current_round"] == 2
    assert [r["round"] for r in body["rounds"]] == [2, 1, 0]  # newest first
    r2, r1, r0 = body["rounds"]
    assert (r0["coverage"]["n"], r0["coverage"]["inside"]) == (3, 2)  # edited recipe not scored
    assert r0["excluded"] == {"no_engine_prediction": 1} and r0["n_corrected"] == 1
    assert r0["n_folded"] == 4 and r0["error"]["n"] == 3
    assert r0["error"]["mean"] == pytest.approx((0.2 + 1.0 - 0.2) / 3, abs=1e-6)
    assert r0["error"]["mean_abs"] == pytest.approx(1.4 / 3, abs=1e-6)
    assert (r1["coverage"]["n"], r1["coverage"]["inside"]) == (2, 1)
    assert (r2["coverage"]["n"], r2["coverage"]["inside"]) == (1, 1) and r2["excluded"] == {"awaiting": 1}
    so_far = [(r["coverage_so_far"]["inside"], r["coverage_so_far"]["n"]) for r in (r0, r1, r2)]
    assert so_far == [(2, 3), (3, 5), (4, 6)]
    lo, hi = wilson_interval(4, 6)
    assert r2["coverage_so_far"]["wilson_95"] == [round(lo, 4), round(hi, 4)]
    assert r2["coverage_so_far"]["judged"] is False
    assert r2["coverage_so_far"]["reason"] == f"needs {MIN_JUDGED_N} scored runs to judge; has 6"
    assert body["pooled"]["coverage"]["observed"] == round(4 / 6, 4)
    assert body["rounds_without_record"] == {"rounds": [], "count": 0, "reason": None}
    assert all(r["current_dataset"] and r["dataset_epoch"] == 1 for r in body["rounds"])


def test_the_band_is_the_one_recorded_at_start_never_the_newer_analysis(env):
    tc, store = env
    cid, ids = _three_rounds(store)
    # band=global: round 0 is scored with A0's q = 0.6 (in force when its runs
    # started), never with the q = 0.3 that A1 and A2 publish later.
    runs = tc.get(f"/api/campaigns/{cid}/track-record/runs", params={"band": "global", "round": 0},
                  headers=H("a-read")).json()["items"]
    bands = {r["run_id"]: (r["band"] or {}).get("lo") for r in runs}
    assert bands[ids["r0"][1]] == 3.4 and bands[ids["r0"][0]] == 2.4  # pred - 0.6, recorded at start
    body = tc.get(f"/api/campaigns/{cid}/track-record", params={"band": "global"},
                  headers=H("a-read")).json()
    r0 = body["rounds"][-1]
    assert (r0["coverage"]["inside"], r0["coverage"]["n"]) == (2, 3)
    # the global band was recorded at 0.9: at 0.95 there is no band, errors are still reported
    body = tc.get(f"/api/campaigns/{cid}/track-record", params={"band": "global", "coverage": 0.95},
                  headers=H("a-read")).json()
    r0 = body["rounds"][-1]
    assert r0["coverage"]["n"] == 0 and r0["excluded"]["no_band_at_coverage"] == 3
    assert r0["error"]["n"] == 3 and r0["coverage"]["wilson_95"] is None


def test_the_per_run_view_and_history_carry_corrections_and_reasons(env):
    tc, store = env
    cid, ids = _three_rounds(store)
    page = tc.get(f"/api/campaigns/{cid}/track-record/runs", params={"limit": 3},
                  headers=H("a-read")).json()
    assert len(page["items"]) == 3 and page["has_more"] is True
    rest = tc.get(f"/api/campaigns/{cid}/track-record/runs",
                  params={"limit": 100, "after_id": page["next_after_id"]}, headers=H("a-read")).json()
    items = page["items"] + rest["items"]
    assert [i["run_id"] for i in items] == ids["r0"] + ids["r1"] + ids["r2"]
    by = {i["run_id"]: i for i in items}
    corrected = by[ids["r0"][2]]
    assert corrected["status"] == "folded" and corrected["folded_in_round"] == 1
    assert corrected["measured"] == {**corrected["measured"], "value": 4.8, "version": 2,
                                     "n_versions": 2, "corrected": True}
    assert corrected["inside"] is True and corrected["error"] == pytest.approx(-0.2)
    assert corrected["band"]["method"] == "conformal_normalized" and corrected["band"]["lo"] == 4.5
    assert corrected["recipe"] == {"Methanol": 3.0, "pH": 5.5}
    assert corrected["prediction"]["intervals"][0] == {
        "coverage": 0.9, "lo": 4.5, "hi": 5.5, "method": "conformal_normalized",
        "coverage_basis": "held_out_past_runs",
        "caveat": {"code": "coverage_lower_at_proposals", "message": "a0 caveat at 90", "recorded": True}}
    edited = by[ids["r0"][3]]
    assert edited["scored"] is False and edited["excluded"] == "no_engine_prediction"
    assert edited["prediction"]["available"] is False and edited["prediction"]["pred"] is None
    assert by[ids["r2"][1]]["status"] == "awaiting" and by[ids["r2"][1]]["excluded"] == "awaiting"

    hist = tc.get(f"/api/campaigns/{cid}/runs/{ids['r0'][2]}/history", headers=H("a-read")).json()
    assert [(v["version"], v["value"], v["reason"], v["is_correction"]) for v in hist["result_versions"]] == [
        (1, 4.4, None, False), (2, 4.8, "re-read the plate", True)]
    assert hist["fold"]["folded_in_round"] == 1 and hist["fold"]["result_version"] == 2
    assert hist["prediction"]["pred"] == 5.0 and hist["status"] == "folded" and hist["round"] == 0
    missing = tc.get(f"/api/campaigns/{cid}/runs/{'0' * 32}/history", headers=H("a-read"))
    assert missing.status_code == 404 and missing.json()["code"] == "run_not_found"
    bad = tc.get(f"/api/campaigns/{cid}/runs/not-a-run/history", headers=H("a-read"))
    assert bad.status_code == 404 and bad.json() == missing.json()


def test_analysis_snapshots_are_listed_newest_first_and_served_as_published(env):
    tc, store = env
    cid, _ids = _three_rounds(store)
    first = tc.get(f"/api/campaigns/{cid}/analyses", params={"limit": 2}, headers=H("a-read")).json()
    assert [i["round"] for i in first["items"]] == [2, 1] and first["has_more"] is True
    assert "analysis" not in first["items"][0]  # summaries only
    rest = tc.get(f"/api/campaigns/{cid}/analyses", params={"before_id": first["next_before_id"]},
                  headers=H("a-read")).json()
    assert [i["kind"] for i in rest["items"]] == ["upload"] and rest["has_more"] is False
    sid = rest["items"][0]["id"]
    one = tc.get(f"/api/campaigns/{cid}/analyses/{sid}", headers=H("a-read")).json()
    assert one["analysis"]["tag"] == "a0" and one["analysis"]["proposals"] == A0["proposals"]
    raw = _q(store, "SELECT analysis FROM analysis_snapshots WHERE tenant = 'acme' AND id = ?", (sid,))[0][0]
    assert one["snapshot"]["analysis_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    gone = tc.get(f"/api/campaigns/{cid}/analyses/999999", headers=H("a-read"))
    assert gone.status_code == 404 and gone.json()["code"] == "snapshot_not_found"


def test_a_result_recorded_before_the_prediction_is_never_scored(env):
    tc, store = env
    cid = store.create_campaign(tenant="acme", name="planted")["id"]
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    with store._txn(write=True) as conn:  # planted rows: the result predates the prediction
        conn.execute(
            "INSERT INTO run_predictions (tenant, campaign_id, run_id, round, source, pred_source, "
            "recipe, available, pred, lo_90, hi_90, recorded_at) VALUES "
            "('acme', ?, ?, 0, 'migrated_v7_state', 'engine', '{}', 1, 3.0, 2.0, 4.0, 200.0)",
            (cid, "a" * 32))
        conn.execute(
            "INSERT INTO run_results (tenant, campaign_id, run_id, version, value, recorded_at) "
            "VALUES ('acme', ?, ?, 1, 3.1, 100.0)", (cid, "a" * 32))
    body = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()
    (r0,) = body["rounds"]
    assert r0["coverage"]["n"] == 0 and r0["excluded"] == {"result_before_prediction": 1}
    assert r0["error"]["n"] == 0  # not even its error counts
    item = tc.get(f"/api/campaigns/{cid}/track-record/runs", headers=H("a-read")).json()["items"][0]
    assert item["scored"] is False and item["inside"] is None and item["status"] == "discarded"


def test_a_re_upload_starts_a_new_dataset_epoch(env):
    tc, store = env
    cid, _ids = _three_rounds(store)
    seed_before = store.get(tenant="acme", campaign_id=cid)["seed_id"]
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid,
               analysis=A0, dataset="again.csv")
    assert store.get(tenant="acme", campaign_id=cid)["seed_id"] != seed_before
    new = store.start(_recipes(A0)[:1], tenant="acme", campaign_id=cid)[0]["id"]
    store.set_result(new, 3.1, tenant="acme", campaign_id=cid)
    body = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()
    assert [(r["dataset_epoch"], r["round"], r["current_dataset"]) for r in body["rounds"]] == [
        (2, 0, True), (1, 2, False), (1, 1, False), (1, 0, False)]
    items = tc.get(f"/api/campaigns/{cid}/track-record/runs", params={"round": 2},
                   headers=H("a-read")).json()["items"]
    assert {i["status"] for i in items} == {"discarded"}  # the re-upload dropped them from pending
    assert body["rounds"][1]["coverage"]["n"] == 1  # still scored: predicted before measured


def test_rounds_folded_before_v8_are_listed_not_scored(env):
    tc, store = env
    cid = store.create_campaign(tenant="acme", name="old")["id"]
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid, analysis=A0)
    with store._txn(write=True) as conn:  # a campaign two rounds in when v8 arrived
        state = json.loads(conn.execute("SELECT state FROM campaigns WHERE id = ? AND tenant = 'acme'",
                                        (cid,)).fetchone()[0])
        state["round"] = 2
        state.pop("seed_id")
        conn.execute("UPDATE campaigns SET state = ? WHERE id = ? AND tenant = 'acme'",
                     (json.dumps(state), cid))
    body = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()
    assert body["rounds"] == [] and body["rounds_without_record"]["rounds"] == [0, 1]
    assert "schema v8" in body["rounds_without_record"]["reason"]
    assert body["pooled"]["coverage"]["n"] == 0 and body["pooled"]["coverage"]["observed"] is None


# --- routes: scope, tenancy, validation, bounds ------------------------------------------ #


ROUTES = ("track-record", "track-record/runs", "analyses", f"runs/{'0' * 32}/history", "analyses/1")


@pytest.mark.parametrize("route", ROUTES)
def test_routes_need_read_and_hide_other_tenants_campaigns(env, route):
    tc, store = env
    cid, _ = _three_rounds(store)
    assert tc.get(f"/api/campaigns/{cid}/{route}").status_code == 401
    assert tc.get(f"/api/campaigns/{cid}/{route}", headers=H("a-write")).status_code == 403
    foreign = tc.get(f"/api/campaigns/{cid}/{route}", headers=H("b-admin"))
    missing = tc.get(f"/api/campaigns/cmp_{'f' * 32}/{route}", headers=H("a-read"))
    malformed = tc.get(f"/api/campaigns/nope/{route}", headers=H("a-read"))
    assert foreign.status_code == missing.status_code == malformed.status_code == 404
    assert foreign.content == missing.content == malformed.content
    assert cid not in foreign.text


def test_validation_and_bounds(env):
    tc, store = env
    cid, _ = _three_rounds(store)
    bad = tc.get(f"/api/campaigns/{cid}/track-record", params={"coverage": 0.8}, headers=H("a-read"))
    assert bad.status_code == 400 and bad.json()["code"] == "unsupported_coverage"
    assert bad.json()["supported"] == [0.9, 0.95]
    # /api/campaign* validation errors are 400 invalid_body (the app's envelope, #98)
    for params in ({"limit": 101}, {"limit": 0}, {"band": "current"}, {"offset": -1}):
        r = tc.get(f"/api/campaigns/{cid}/track-record", params=params, headers=H("a-read"))
        assert (r.status_code, r.json()["code"]) == (400, "invalid_body"), params
    assert tc.get(f"/api/campaigns/{cid}/track-record/runs", params={"limit": 501},
                  headers=H("a-read")).status_code == 400
    assert tc.get(f"/api/campaigns/{cid}/analyses", params={"limit": 201},
                  headers=H("a-read")).status_code == 400
    page = tc.get(f"/api/campaigns/{cid}/track-record", params={"limit": 1, "offset": 1},
                  headers=H("a-read")).json()
    assert [r["round"] for r in page["rounds"]] == [1]
    assert page["page"] == {"limit": 1, "offset": 1, "total_rounds": 3, "has_more": True, "next_offset": 2}


def test_an_unseeded_campaign_has_an_empty_record(env):
    tc, store = env
    cid = store.create_campaign(tenant="acme", name="empty")["id"]
    body = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()
    assert body["has_campaign"] is False and body["rounds"] == [] and body["current_round"] is None
    assert tc.get(f"/api/campaigns/{cid}/analyses", headers=H("a-read")).json()["items"] == []


# --- every statement on a tenant table names the tenant -------------------------------- #

_TENANT_TABLE_RE = re.compile(r"\b(" + "|".join(migrate.TENANT_TABLES) + r")\b")


def test_every_history_statement_on_a_tenant_table_mentions_tenant(env):
    tc, store = env
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    cid, ids = _three_rounds(store)
    store.save_analysis(A2, "x.csv", tenant="acme", campaign_id=cid, campaign_generation=None)
    for route in ("track-record", "track-record/runs", "analyses", f"runs/{ids['r0'][2]}/history",
                  "analyses/1"):
        assert tc.get(f"/api/campaigns/{cid}/{route}", headers=H("a-read")).status_code == 200
    store._conn.set_trace_callback(None)
    touching = [s for s in seen if _TENANT_TABLE_RE.search(re.sub(r"'(?:[^']|'')*'", "''", s))]
    assert any("run_predictions" in s for s in touching) and any("run_folds" in s for s in touching)
    assert any("analysis_snapshots" in s for s in touching)
    assert [s for s in touching if "tenant" not in s] == []


# --- the real engine, three rounds over HTTP --------------------------------------------- #


def _truth(m: float, ph: float) -> float:
    return 1.5 * m - 0.8 * (ph - 6.0) ** 2 + 2.0


@pytest.mark.slow
def test_three_rounds_with_the_real_engine_over_http(env):
    tc, _store = env
    rng = np.random.default_rng(3)
    m = rng.uniform(0, 4, 14).round(3)
    ph = rng.uniform(5, 7, 14).round(2)
    y = np.array([_truth(a, b) for a, b in zip(m, ph)]) + rng.normal(0, 0.15, 14)
    buf = io.StringIO()
    pd.DataFrame({"Methanol": m, "pH": ph, "lipase_titer": y.round(3)}).to_csv(buf, index=False)
    cid = tc.post("/api/campaigns", headers=H("a-admin"), json={"name": "real"}).json()["id"]
    up = tc.post(f"/api/campaigns/{cid}/upload", headers=H("a-admin"),
                 files={"file": ("runs.csv", buf.getvalue().encode(), "text/csv")},
                 data={"target": "lipase_titer"})
    assert up.status_code == 200, up.text[:300]
    started: dict[int, list[dict]] = {}
    for rnd in range(3):
        analysis = tc.get(f"/api/campaigns/{cid}/analysis", headers=H("a-read")).json()
        props = analysis["proposals"][:2]
        runs = tc.post(f"/api/campaigns/{cid}/runs", headers=H("a-admin"), json={"recipes": [
            {"recipe": p["recipe"], "mode": p.get("mode"), "reason": "t"} for p in props]}).json()["started"]
        started[rnd] = [{"run": r, "proposal": p} for r, p in zip(runs, props)]
        for r in runs:
            v = round(_truth(float(r["recipe"]["Methanol"]), float(r["recipe"]["pH"])) + rng.normal(0, 0.15), 3)
            assert tc.post(f"/api/campaigns/{cid}/runs/{r['id']}/result", headers=H("a-admin"),
                           json={"value": v}).status_code == 200
        if rnd < 2:
            re_ = tc.post(f"/api/campaigns/{cid}/reanalyze", headers=H("a-admin"))
            assert re_.status_code == 200, re_.text[:300]
    body = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()
    assert [r["round"] for r in body["rounds"]] == [2, 1, 0] and body["current_round"] == 2
    items = tc.get(f"/api/campaigns/{cid}/track-record/runs", headers=H("a-read")).json()["items"]
    assert len(items) == 6
    by = {i["run_id"]: i for i in items}
    compared = 0
    for rnd, pairs in started.items():
        for pair in pairs:
            item = by[pair["run"]["id"]]
            assert item["round"] == rnd and item["status"] == ("folded" if rnd < 2 else "measured")
            iv = next((x for x in pair["proposal"]["intervals"] if x["coverage"] == 0.9 and x["available"]), None)
            if iv is not None:  # the band recorded at start is the one the proposal carried
                assert (item["band"]["lo"], item["band"]["hi"]) == (iv["lo"], iv["hi"])
                served = item["band"]["caveat"]  # verbatim, as the engine served it
                assert (served["code"], served["message"], served["recorded"]) == (
                    iv["caveat"]["code"], iv["caveat"]["message"], True)
                assert item["inside"] == (iv["lo"] <= item["measured"]["value"] <= iv["hi"])
                compared += 1
            assert item["prediction"]["pred"] == pair["proposal"]["pred"]
            snap = tc.get(f"/api/campaigns/{cid}/analyses/{item['prediction']['analysis_snapshot_id']}",
                          headers=H("a-read")).json()
            assert snap["snapshot"]["round"] == rnd
            assert snap["analysis"]["proposals"][0]["recipe"] == started[rnd][0]["proposal"]["recipe"]
    assert compared >= 1  # the real analysis published per-proposal 90% intervals
    scored = [i for i in items if i["scored"]]
    pooled = body["pooled"]["coverage"]
    assert pooled["n"] == len(scored) and pooled["inside"] == sum(i["inside"] for i in scored)


# --- review fixes: caveats, pooled methods, judged boundary, bounds ------------------ #


def _all_bands(item: dict) -> list[dict]:
    pred = item["prediction"]
    return [*pred["intervals"], *([pred["global_band"]] if pred["global_band"] else []),
            *([item["band"]] if item.get("band") else [])]


@pytest.mark.parametrize("band", ["proposal", "global"])
def test_every_returned_interval_and_band_carries_the_caveat_served_at_start(env, band):
    tc, store = env
    cid, ids = _three_rounds(store)
    items = tc.get(f"/api/campaigns/{cid}/track-record/runs", params={"band": band},
                   headers=H("a-read")).json()["items"]
    tags = {**dict.fromkeys(ids["r0"], "a0"), **dict.fromkeys(ids["r1"], "a1"), **dict.fromkeys(ids["r2"], "a2")}
    n_bands = 0
    for item in items:
        for b in _all_bands(item):
            n_bands += 1
            assert b["caveat"]["recorded"] is True and b["caveat"]["code"]
            assert b["caveat"]["message"].startswith(tags[item["run_id"]])  # verbatim, from ITS analysis
            assert "calibrated" not in b["caveat"]["message"].lower()
        for iv in item["prediction"]["intervals"]:
            assert iv["caveat"]["message"] == f"{tags[item['run_id']]} caveat at {int(iv['coverage'] * 100)}"
        if item["prediction"]["global_band"]:
            gb = item["prediction"]["global_band"]
            assert gb["coverage_basis"] == "held_out_past_runs"
            assert gb["caveat"]["message"] == f"{tags[item['run_id']]} global band caveat"
    assert n_bands >= 7 * 3
    for rid in ids["r0"][:3] + ids["r1"]:
        hist = tc.get(f"/api/campaigns/{cid}/runs/{rid}/history", headers=H("a-read")).json()
        bands = [*hist["prediction"]["intervals"], hist["prediction"]["global_band"]]
        assert bands and all(b["caveat"]["recorded"] and b["caveat"]["message"] for b in bands)


def _plant(store: CampaignStore, cid: str, n: int, *, inside: int, caveat: bool = True,
           first: int = 0, method: str = "conformal_normalized", basis: str = "held_out_past_runs",
           with_result: bool = True, fold: bool = False) -> list[str]:
    """`n` runs with a prediction recorded at t=100+i and a result at t=200+i."""
    run_ids = []
    with store._txn(write=True) as conn:
        for i in range(first, first + n):
            rid = f"{i:032x}"
            run_ids.append(rid)
            conn.execute(
                "INSERT INTO run_predictions (tenant, campaign_id, run_id, round, source, pred_source, "
                "recipe, available, pred, lo_90, hi_90, method_90, basis_90, caveat_code_90, caveat_90, "
                "recorded_at) VALUES ('acme', ?, ?, 0, 'start', 'engine', '{}', 1, 3.0, 2.0, 4.0, ?, ?, ?, ?, ?)",
                (cid, rid, method, basis, "c" if caveat else None, "served" if caveat else None, 100.0 + i))
            if with_result:
                conn.execute("INSERT INTO run_results (tenant, campaign_id, run_id, version, value, recorded_at) "
                             "VALUES ('acme', ?, ?, 1, ?, ?)", (cid, rid, 3.5 if i - first < inside else 9.0, 200.0 + i))
            if fold:
                conn.execute("INSERT INTO run_folds (tenant, campaign_id, run_id, folded_in_round, value, "
                             "folded_at) VALUES ('acme', ?, ?, 1, 3.5, 300.0)", (cid, rid))
    return run_ids


def _empty_campaign(store: CampaignStore) -> str:
    cid = store.create_campaign(tenant="acme", name="planted")["id"]
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    return cid


def test_a_band_stored_without_a_caveat_is_served_with_caveat_not_recorded(env):
    tc, store = env
    cid = _empty_campaign(store)
    _plant(store, cid, 1, inside=1, caveat=False)
    item = tc.get(f"/api/campaigns/{cid}/track-record/runs", headers=H("a-read")).json()["items"][0]
    assert item["band"]["caveat"] == CAVEAT_NOT_RECORDED
    assert item["prediction"]["intervals"][0]["caveat"]["code"] == "caveat_not_recorded"
    assert not any(ch.isdigit() for ch in CAVEAT_NOT_RECORDED["message"])  # no coverage figures


@pytest.mark.parametrize(("n", "inside", "judged", "nominal_inside"), [
    (MIN_JUDGED_N - 1, MIN_JUDGED_N - 1, False, None),
    (MIN_JUDGED_N, MIN_JUDGED_N, True, True),     # 10/10: Wilson [0.7225, 1] holds 0.9
    (MIN_JUDGED_N, 5, True, False),                # 5/10: [0.2366, 0.7634] does not
])
def test_coverage_is_judged_from_ten_scored_runs(env, n, inside, judged, nominal_inside):
    tc, store = env
    cid = _empty_campaign(store)
    _plant(store, cid, n, inside=inside)
    pooled = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()["pooled"]["coverage"]
    assert (pooled["n"], pooled["inside"], pooled["judged"]) == (n, inside, judged)
    assert pooled["nominal_inside_interval"] is nominal_inside
    assert (pooled["reason"] is None) == judged


def test_a_folded_run_without_result_history_is_result_time_unknown(env):
    tc, store = env
    cid = _empty_campaign(store)
    (rid,) = _plant(store, cid, 1, inside=1, with_result=False, fold=True)
    item = tc.get(f"/api/campaigns/{cid}/track-record/runs", headers=H("a-read")).json()["items"][0]
    assert item["run_id"] == rid and item["status"] == "folded"
    assert item["excluded"] == "result_time_unknown" and item["scored"] is False and item["error"] is None
    (r0,) = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()["rounds"]
    assert r0["excluded"] == {"result_time_unknown": 1} and r0["coverage"]["n"] == 0


def test_coverage_shows_how_many_bands_of_each_method_it_pools(env):
    tc, store = env
    cid = _empty_campaign(store)
    _plant(store, cid, 3, inside=2, method="conformal_selection_aware", basis="held_out_top_ranked_runs")
    _plant(store, cid, 2, inside=1, first=3)
    cov = tc.get(f"/api/campaigns/{cid}/track-record", headers=H("a-read")).json()["rounds"][0]["coverage"]
    assert (cov["n"], cov["inside"]) == (5, 3)
    assert cov["by_method"] == {"conformal_normalized": {"n": 2, "inside": 1},
                                "conformal_selection_aware": {"n": 3, "inside": 2}}
    assert cov["by_coverage_basis"] == {"held_out_past_runs": {"n": 2, "inside": 1},
                                        "held_out_top_ranked_runs": {"n": 3, "inside": 2}}


def test_ids_beyond_sqlite_integers_are_refused_not_500(env):
    tc, store = env
    cid, _ = _three_rounds(store)
    for url, params in ((f"/api/campaigns/{cid}/track-record/runs", {"after_id": MAX_ROW_ID + 1}),
                        (f"/api/campaigns/{cid}/analyses", {"before_id": MAX_ROW_ID + 1}),
                        (f"/api/campaigns/{cid}/analyses/{MAX_ROW_ID + 1}", {})):
        r = tc.get(url, params=params, headers=H("a-read"))
        assert (r.status_code, r.json()["code"]) == (400, "invalid_body"), url
    assert tc.get(f"/api/campaigns/{cid}/track-record/runs", params={"after_id": MAX_ROW_ID},
                  headers=H("a-read")).json()["items"] == []
    assert tc.get(f"/api/campaigns/{cid}/analyses", params={"before_id": MAX_ROW_ID},
                  headers=H("a-read")).status_code == 200
    assert tc.get(f"/api/campaigns/{cid}/analyses/{MAX_ROW_ID}", headers=H("a-read")).status_code == 404
