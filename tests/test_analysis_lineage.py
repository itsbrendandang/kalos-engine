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


# --- stored roles that cannot be read are never replaced by inferred ones (S5) ------- #


def _write_stored_roles(store, roles):
    state = store.get(tenant="default")
    state["settings"]["roles"] = roles
    with store._txn(write=True) as conn:
        row = store._row(conn, "default", store._default_id(conn, "default"))
        store._cas_write(conn, "default", row, state, row.generation)


@pytest.mark.parametrize("bad", [
    {"target": "Titer", "features": INPUTS, "allow_beyond": 5},
    {"target": "Titer", "features": INPUTS, "linear_constraints": "x"},
    "not an object",
])
def test_unreadable_stored_roles_refuse_the_reanalyze_and_commit_nothing(client, bad):
    c, store = client
    upload(c, {"target": "Titer", "features": INPUTS})
    _start_and_record(c, [63.0])
    _write_stored_roles(store, bad)
    cid = store.default_campaign_id(tenant="default")
    before_state = store.get(tenant="default")
    before_analysis = store.load_analysis(tenant="default")
    for path in ("/api/campaign/reanalyze", f"/api/campaigns/{cid}/reanalyze"):
        r = c.post(path)
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["code"] == "roles_not_applied"
        assert "Upload the sheet again" in body["error"]
    assert store.get(tenant="default") == before_state
    assert store.load_analysis(tenant="default") == before_analysis


# --- the audit row of an upload names the settings it changed (S8) ------------------- #


def _last_audit(store, action):
    return next(e for e in store.audit_entries(tenant="default") if e["action"] == action)


def test_upload_audit_row_names_the_changed_settings_never_the_values_of_the_rest(client):
    c, store = client
    secret = "Carbon_conc"
    upload(c, {"target": "Titer", "features": INPUTS, "allow_beyond": 0.1})
    first = _last_audit(store, "run.upload")["detail"]
    assert "allow_beyond set to 0.1" in first and "features set" in first
    upload(c, {"target": "Titer", "features": ["Gly", "Met", "SP"], "allow_beyond": 0.2,
               "linear_constraints": [{"terms": {"Gly": 1, "Met": 1}, "op": "<=", "rhs": 9,
                                       "label": "budget"}]})
    second = _last_audit(store, "run.upload")["detail"]
    assert "allow_beyond 0.1 -> 0.2" in second
    assert "features changed" in second and "linear_constraints set" in second
    # Key names only for the rest: no column names, terms or labels.
    for leaked in (secret, "Gly", "budget", "pichia"):
        assert leaked not in second
    cid = store.default_campaign_id(tenant="default")
    upload(c, {"target": "Titer", "features": ["Gly", "Met", "SP"], "allow_beyond": 0.2,
               "direction": "minimize"}, path=f"/api/campaigns/{cid}/upload")
    third = _last_audit(store, "campaign.upload")["detail"]
    assert "direction set to minimize" in third and "linear_constraints removed" in third
    upload(c, path=f"/api/campaigns/{cid}/upload")
    assert "cleared" in _last_audit(store, "campaign.upload")["detail"]


def test_roles_change_detail_keeps_unsafe_values_out():
    from kalos.portal.audit_detail import roles_change_detail

    detail = roles_change_detail(
        {"target": "Titer", "allow_beyond": {"Glucose": 0.1}, "direction": "maximize"},
        {"target": "Yield", "allow_beyond": {"Glucose": 0.3}, "direction": "minimize",
         "secret_key": "x"},
    )
    assert detail == ("settings: target changed; direction maximize -> minimize; "
                      "allow_beyond changed; 1 other key(s) changed")


# --- the analysis cache never gives two campaigns one analysis_id ---------------------- #


def test_two_campaigns_never_share_an_analysis_id(client, monkeypatch):
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "8")
    pipeline.ANALYSIS_CACHE.clear()
    c, store = client
    roles = {"target": "Titer", "features": INPUTS}
    try:
        on_default = upload(c, roles)  # creates the tenant's default campaign
        other = c.post("/api/campaigns", json={"name": "second"}).json()["id"]
        in_other = upload(c, roles, path=f"/api/campaigns/{other}/upload")
        third = c.post("/api/campaigns", json={"name": "third"}).json()["id"]
        in_third = upload(c, roles, path=f"/api/campaigns/{third}/upload")
        # The same campaign again is still the cached analysis.
        again_other = upload(c, roles, path=f"/api/campaigns/{other}/upload")
        again_default = upload(c, roles)
    finally:
        pipeline.ANALYSIS_CACHE.clear()
    ids = [on_default["analysis_id"], in_other["analysis_id"], in_third["analysis_id"]]
    assert len(set(ids)) == 3
    assert in_other["analysis_cache"] == {"hit": False}
    assert again_other["analysis_cache"] == {"hit": True}
    assert again_other["analysis_id"] == in_other["analysis_id"]
    assert again_default["analysis_cache"] == {"hit": True}
    assert again_default["analysis_id"] == on_default["analysis_id"]
    cid = store.default_campaign_id(tenant="default")
    stored = {store.load_analysis(tenant="default", campaign_id=x)["analysis_id"]
              for x in (cid, other, third)}
    assert stored == set(ids)


def test_a_new_default_campaign_does_not_reuse_the_archived_ones_analysis(client, monkeypatch):
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "8")
    pipeline.ANALYSIS_CACHE.clear()
    c, store = client
    try:
        first = upload(c)
        old = store.default_campaign_id(tenant="default")
        store.archive_campaign(tenant="default", campaign_id=old)
        second = upload(c)  # a fresh default campaign
    finally:
        pipeline.ANALYSIS_CACHE.clear()
    assert store.default_campaign_id(tenant="default") != old
    assert second["analysis_id"] != first["analysis_id"]
