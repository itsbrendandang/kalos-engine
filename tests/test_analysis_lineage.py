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
