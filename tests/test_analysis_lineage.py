"""One stored analysis, one id, on every read route; a refit says what it
replaced (kalos/portal/lineage.py, docs/CAMPAIGNS_API.md "Analysis identity")."""
from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal import jobs, pipeline
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store

CSV = (
    b"Carbon_conc,Gly,Met,SP,Titer\n"
    b"10,1,2,5,40\n20,2,3,6,55\n30,3,1,7,61\n15,4,2,8,70\n"
    b"25,1,4,9,82\n35,2,2,6,58\n12,3,3,7,66\n28,4,1,8,74\n"
)
INPUTS = ["Carbon_conc", "Gly", "Met", "SP"]


def stub_analyze(df, target, *, roles=None, direction=None, **_):
    """A fast stand-in for `_analyze` with the fields lineage reads. Without
    declared roles it leaves Carbon_conc out, as the name-based profile did."""
    target = target or (roles.target if roles is not None else "Titer")
    if roles is not None and roles.features:
        feats = [f for f in roles.features if f in df.columns]
    else:
        feats = [c for c in df.columns if c not in (target, "Carbon_conc")]
    n = int(df[target].notna().sum())
    share = 1.0 / len(feats)
    return {
        "target": target, "features": feats, "proposal_features": feats, "d": len(feats),
        "n": n, "best": float(df[target].max()), "seed": 1234, "timestamp": 1,
        "direction": direction or "maximize", "categorical_features": [],
        "reliability": {"clears_floor": n >= 8},
        "gp_shapes": {"available": True,
                      "features": [{"name": f, "relevance": share} for f in feats]},
        "proposals": [{"recipe": {f: 1.0 for f in feats}, "pred": 50.0, "std": 2.0}],
        "conformal_q": 10.0,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal_module, "_analyze", stub_analyze)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def upload(c, roles=None, *, path="/api/run", replace=False):
    data = {"target": "Titer"}
    if roles is not None:
        data["roles"] = json.dumps(roles)
    if replace:
        data["replace"] = "true"
    r = c.post(path, files={"file": ("pichia.csv", io.BytesIO(CSV), "text/csv")}, data=data)
    assert r.status_code == 200, r.text
    return r.json()


def test_every_read_route_names_the_same_stored_analysis(client):
    c, store = client
    run = upload(c)
    aid = run["analysis_id"]
    assert aid.startswith("an_") and run["supersedes"] is None
    cid = store.default_campaign_id(tenant="default")
    latest = c.get("/api/latest").json()
    assert latest["analysis_id"] == aid
    assert c.get("/api/campaign").json()["analysis_id"] == aid
    assert c.get(f"/api/campaigns/{cid}/analysis").json()["analysis_id"] == aid
    assert c.get(f"/api/campaigns/{cid}").json()["summary"]["analysis_id"] == aid
    # /api/run returns the stored object, not a second copy with other fields.
    assert {k: v for k, v in latest.items() if k != "has_data"} == run


def test_identical_reupload_is_the_identical_analysis_on_both_upload_routes(client, monkeypatch):
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "4")
    pipeline.ANALYSIS_CACHE.clear()
    c, store = client
    roles = {"target": "Titer", "features": INPUTS}
    try:
        first = upload(c, roles)
        cid = store.default_campaign_id(tenant="default")
        again = upload(c, roles)
        via_campaign = upload(c, roles, path=f"/api/campaigns/{cid}/upload")
    finally:
        pipeline.ANALYSIS_CACHE.clear()
    assert again["analysis_cache"] == {"hit": True}
    assert first["analysis_id"] == again["analysis_id"] == via_campaign["analysis_id"]
    for key in ("best", "n", "features", "proposals", "gp_shapes"):
        assert first[key] == again[key] == via_campaign[key]
    assert via_campaign["supersedes"] is None


def test_new_roles_make_a_new_analysis_that_says_what_it_replaced(client):
    c, _ = client
    first = upload(c)
    second = upload(c, {"target": "Titer", "features": INPUTS})
    assert second["analysis_id"] != first["analysis_id"]
    assert second["supersedes"]["analysis_id"] == first["analysis_id"]
    assert second["supersedes"]["code"] == "roles_changed"
    assert "column choices changed" in second["supersedes"]["why"]


def test_reanalyze_supersedes_the_upload_and_reads_back_identically(client):
    c, _ = client
    first = upload(c)
    started = c.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Gly": 2.0, "Met": 2.0, "SP": 7.0}, "pred": 60.0, "std": 3.0},
    ]}).json()["started"]
    c.post("/api/campaign/result", json={"id": started[0]["id"], "value": 63.0})
    body = c.post("/api/campaign/reanalyze").json()
    aid = body["analysis"]["analysis_id"]
    assert aid != first["analysis_id"]
    assert body["analysis"]["supersedes"] == {
        "analysis_id": first["analysis_id"], "code": "results_added",
        "why": "Kalos fit the model again with 1 recorded result added."}
    assert body["campaign"]["analysis_id"] == aid
    assert c.get("/api/latest").json()["analysis_id"] == aid


def test_job_view_names_the_published_analysis(client):
    _, store = client
    cid = store.create_campaign(tenant="default", name="jobs")["id"]
    gen = store.seed(None, "Titer", ["SP"], tenant="default", campaign_id=cid,
                     base_rows=[{"SP": 1.0, "Titer": 2.0}], analysis={"target": "Titer"})
    published = store.load_analysis(tenant="default", campaign_id=cid)
    row = {"state": "succeeded", "result_ref": gen, "tenant": "default", "campaign_id": cid}
    with store._txn(write=False) as conn:
        assert jobs._with_analysis_id(conn, row)["analysis_id"] == published["analysis_id"]
        stale = jobs._with_analysis_id(conn, {**row, "result_ref": "old"})
    assert stale["analysis_id"] is None


# --- declared roles persist with the campaign (finding 7) --------------------------- #


def _start_and_record(c, values):
    recipes = [{"recipe": {"Carbon_conc": 20.0 + i, "Gly": 2.0, "Met": 2.0, "SP": 7.0 + i},
                "pred": 60.0 + i, "std": 3.0} for i in range(3)]
    started = c.post("/api/campaign/start", json={"recipes": recipes}).json()["started"]
    for run, value in zip(started, values):
        assert c.post("/api/campaign/result", json={"id": run["id"], "value": value}).status_code == 200
    return started


def test_declared_roles_and_the_name_survive_a_reanalyze(client):
    c, store = client
    roles = {"target": "Titer", "features": INPUTS, "direction": "maximize"}
    first = upload(c, roles)
    assert first["features"] == INPUTS
    assert first["roles_used"] == {"source": "declared", "roles": roles}
    assert first["dataset"] == "pichia.csv"
    assert store.get(tenant="default")["settings"]["roles"] == roles

    _start_and_record(c, [63.0, 71.0])
    body = c.post("/api/campaign/reanalyze").json()
    after = body["analysis"]
    assert after["features"] == INPUTS
    assert after["dataset"] == "pichia.csv"
    assert after["roles_used"] == {"source": "declared", "roles": roles}
    assert after["run"]["roles_declared"] == roles and after["run"]["profile"] == "generic"
    assert after["campaign_findings"] == []
    # The roles stay with the campaign for the next round too.
    assert store.get(tenant="default")["settings"]["roles"] == roles


def test_without_declared_roles_the_reanalyze_infers_and_says_so(client):
    c, _ = client
    upload(c)
    _start_and_record(c, [63.0])
    after = c.post("/api/campaign/reanalyze").json()["analysis"]
    assert after["roles_used"]["source"] == "inferred"
    assert after["roles_used"]["roles"]["features"] == ["Gly", "Met", "SP"]
    assert after["run"]["profile"] == "bioprocess"


def test_changed_inputs_carry_an_inputs_changed_finding(client):
    c, _ = client
    upload(c)
    second = upload(c, {"target": "Titer", "features": INPUTS})
    (finding,) = second["campaign_findings"]
    assert finding["code"] == "inputs_changed"
    assert finding["added"] == ["Carbon_conc"] and finding["removed"] == []
    assert "Carbon_conc is now used as an input." in finding["message"]
    third = upload(c, {"target": "Titer", "features": ["Gly", "SP"]})
    assert third["campaign_findings"][0]["removed"] == ["Carbon_conc", "Met"]
    assert "Carbon_conc and Met are no longer used." in third["campaign_findings"][0]["message"]


def test_upload_settings_keep_other_keys_and_replace_roles(client):
    c, store = client
    upload(c, {"target": "Titer", "features": INPUTS})
    state = store.get(tenant="default")
    state["settings"]["allow_beyond"] = True
    with store._txn(write=True) as conn:
        row = store._row(conn, "default", store._default_id(conn, "default"))
        store._cas_write(conn, "default", row, state, row.generation)
    upload(c)
    settings = store.get(tenant="default")["settings"]
    assert settings == {"allow_beyond": True, "roles": None}


# --- which recorded runs entered the fit, and what changed (finding 5) --------------- #


def test_reanalyze_lists_the_recorded_runs_and_the_delta(client):
    c, store = client
    first = upload(c, {"target": "Titer", "features": INPUTS})
    assert first["runs_used"] == {"n_upload_rows": 8, "n_recorded": 0, "recorded": []}
    assert first["delta"] is None
    started = _start_and_record(c, [63.0, 90.0])  # 3 started, 2 recorded
    assert started[0]["likely_range_at_start"] == {"low": 50.0, "high": 70.0}

    after = c.post("/api/campaign/reanalyze").json()["analysis"]
    used = after["runs_used"]
    assert used["n_upload_rows"] == 8 and used["n_recorded"] == 2
    assert used["recorded"] == [
        {"run_id": started[0]["id"], "round": 0, "measured_value": 63.0,
         "predicted_at_start": 60.0, "inside_likely_range": True},
        {"run_id": started[1]["id"], "round": 0, "measured_value": 90.0,
         "predicted_at_start": 61.0, "inside_likely_range": False},
    ]
    assert after["n"] == used["n_upload_rows"] + used["n_recorded"]

    d = after["delta"]
    assert d["previous_analysis_id"] == first["analysis_id"] == after["supersedes"]["analysis_id"]
    assert d["best"] == {"before": 82.0, "after": 90.0}
    assert d["verdict"] == {"before": "reliable", "after": "reliable"}
    assert d["n_runs"] == {"before": 8, "after": 10}
    assert d["inputs_added"] == [] and d["inputs_removed"] == []
    assert d["driver_shares_source"] == "gp_shapes"
    assert {s["input"] for s in d["driver_shares"]} == set(INPUTS)
    assert d["summary"] == "Your 2 results were added. Best went from 82 to 90."
    # Stored, so every read route shows the same lists.
    assert c.get("/api/latest").json()["runs_used"] == used

    # The next round lists both rounds' recorded runs.
    more = _start_and_record(c, [70.0])
    again = c.post("/api/campaign/reanalyze").json()["analysis"]
    assert [r["run_id"] for r in again["runs_used"]["recorded"]] == [
        started[0]["id"], started[1]["id"], more[0]["id"]]
    assert again["runs_used"]["recorded"][2]["round"] == 1
    assert again["delta"]["summary"] == "Your 1 result was added. Best is still 90."
    assert store.get(tenant="default")["round"] == 2


def test_delta_summary_names_the_driver_shift_and_the_new_input(client):
    from kalos.portal import lineage

    before = {"analysis_id": "an_a", "best": 113.65, "n": 235, "features": ["SP", "Met", "Gly"],
              "reliability": {"clears_floor": True},
              "gp_shapes": {"available": True, "features": [
                  {"name": "Gly", "relevance": 0.44}, {"name": "SP", "relevance": 0.44},
                  {"name": "Met", "relevance": 0.12}]}}
    after = {"best": 113.65, "n": 237, "features": ["SP", "Met", "Gly", "Carbon_conc"],
             "reliability": {"clears_floor": False},
             "gp_shapes": {"available": True, "features": [
                 {"name": "SP", "relevance": 0.55}, {"name": "Carbon_conc", "relevance": 0.44},
                 {"name": "Gly", "relevance": 0.004}, {"name": "Met", "relevance": 0.006}]}}
    d = lineage.delta(after, before, n_folded=2, why="")
    assert d["summary"] == (
        "Your 2 results were added. Best is still 113.65. Gly dropped from 44% to under 1% "
        "after Carbon_conc was added. The verdict changed from reliable to not yet reliable.")
    assert d["inputs_added"] == ["Carbon_conc"]
    assert {"input": "Carbon_conc", "before": None, "after": 0.44} in d["driver_shares"]
    # The verdict object of a newer analysis wins over the reliability flag.
    assert lineage.verdict_code({"verdict": {"code": "trust_ranking"},
                                 "reliability": {"clears_floor": False}}) == "trust_ranking"
