"""GET /api/campaigns/{id}/rows: every row the model learns from, labelled by
origin (kalos/portal/campaign_rows.py)."""
from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.campaign_rows import campaign_rows

CSV = (
    b"Carbon_conc,Gly,Met,SP,Titer\n"
    b"10,1,2,5,40\n20,2,3,6,55\n30,3,1,7,61\n15,4,2,8,70\n"
    b"25,1,4,9,82\n35,2,2,6,58\n12,3,3,7,66\n28,4,1,8,74\n"
)


def _stub_analyze(df, target, *, roles=None, direction=None, **_):
    feats = [c for c in df.columns if c != "Titer"]
    return {"target": "Titer", "features": feats, "proposal_features": feats, "d": len(feats),
            "n": len(df), "best": float(df["Titer"].max()), "seed": 1234, "timestamp": 1,
            "direction": "maximize", "categorical_features": [], "proposals": [],
            "reliability": {"clears_floor": True}, "conformal_q": 10.0}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal_module, "_analyze", _stub_analyze)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _setup(c):
    cid = c.post("/api/campaigns", json={"name": "pichia"}).json()["id"]
    r = c.post(f"/api/campaigns/{cid}/upload",
               files={"file": ("pichia.csv", io.BytesIO(CSV), "text/csv")}, data={"target": "Titer"})
    assert r.status_code == 200, r.text
    return cid


def test_rows_before_and_after_a_round(client):
    c, _store = client
    cid = _setup(c)
    body = c.get(f"/api/campaigns/{cid}/rows").json()
    assert body["has_data"] and body["n_rows"] == body["n_upload"] == 8
    assert {r["_origin"] for r in body["rows"]} == {"upload"}
    assert body["rows"][0]["_upload_row"] == 1 and body["rows"][0]["_kind"] == "measured"
    assert body["columns"][:5] == ["Carbon_conc", "Gly", "Met", "SP", "Titer"]

    recipes = [{"recipe": {"Carbon_conc": 20.0 + i, "Gly": 2.0, "Met": 2.0, "SP": 7.0},
                "pred": 60.0, "std": 3.0} for i in range(3)]
    started = c.post(f"/api/campaigns/{cid}/runs", json={"recipes": recipes}).json()["started"]
    for run, v in zip(started[:2], [63.0, 71.0]):
        assert c.post(f"/api/campaigns/{cid}/runs/{run['id']}/result", json={"value": v}).status_code == 200
    assert c.post(f"/api/campaigns/{cid}/reanalyze").status_code == 200

    body = c.get(f"/api/campaigns/{cid}/rows").json()
    assert body["n_rows"] == 10 and body["n_from_rounds"] == 2
    folded = [r for r in body["rows"] if r["_origin"] == "round"]
    assert [r["_run_id"] for r in folded] == [started[0]["id"], started[1]["id"]]
    assert [r["Titer"] for r in folded] == [63.0, 71.0]
    assert all(r["_round"] == 1 and r["_kind"] == "measured" for r in folded)
    page = c.get(f"/api/campaigns/{cid}/rows", params={"offset": 8, "limit": 1}).json()
    assert page["n_rows"] == 10 and len(page["rows"]) == 1 and page["rows"][0]["_index"] == 8


def test_rows_unknown_campaign_is_404(client):
    c, _ = client
    assert c.get("/api/campaigns/cmp_00000000000000000000000000000000/rows").status_code == 404
    assert c.get("/api/campaigns/not-an-id/rows").status_code == 404


def test_old_campaigns_without_recorded_runs_still_list_folded_rows():
    state = {"target": "y", "base_rows": [{"x": 1, "y": 2}, {"x": 2, "y": 3}, {"x": 3, "y": 4}],
             "history": [{"round": 0, "n_base": 2}]}
    out = campaign_rows(state)
    assert [r["_origin"] for r in out["rows"]] == ["upload", "upload", "round"]
    assert out["rows"][2]["_run_id"] is None and out["rows"][2]["_round"] is None
    assert campaign_rows({"base_rows": [{"x": 1}]})["n_upload"] == 1  # no history: all upload


def test_upload_rows_without_a_result_are_not_labelled_measured():
    state = {"target": "y", "base_rows": [{"x": 1, "y": 2.0}, {"x": 2, "y": None},
                                          {"x": 3, "y": float("nan")}, {"x": 4, "y": " "}],
             "history": [{"round": 0, "n_base": 4}]}
    out = campaign_rows(state)
    assert [r["_kind"] for r in out["rows"]] == ["measured", "no_result", "no_result", "no_result"]
    assert out["kinds"] == ["measured", "no_result"]
    assert "no_result" in out["kinds_note"]


def test_sheet_columns_named_like_the_metadata_are_renamed_not_overwritten():
    row = {"_origin": "lab A", "_kind": "fed-batch", "_index": 99, "_round": "R7",
           "_run_id": "B-12", "_upload_row": 5, "_origin (sheet)": "taken", "y": 1.0}
    out = campaign_rows({"target": "y", "base_rows": [row], "history": [{"n_base": 1}]})
    (r,) = out["rows"]
    # The metadata wins its own names ...
    assert r["_origin"] == "upload" and r["_kind"] == "measured" and r["_index"] == 0
    assert r["_round"] == 0 and r["_run_id"] is None and r["_upload_row"] == 1
    # ... and the sheet's values survive under a new name.
    assert r["_origin (sheet 2)"] == "lab A" and r["_origin (sheet)"] == "taken"
    assert r["_kind (sheet)"] == "fed-batch" and r["_index (sheet)"] == 99
    assert r["_round (sheet)"] == "R7" and r["_run_id (sheet)"] == "B-12"
    assert r["_upload_row (sheet)"] == 5
    assert {"column": "_kind", "shown_as": "_kind (sheet)"} in out["renamed_columns"]
    assert len(out["renamed_columns"]) == 6 and out["renamed_note"]
    assert "_kind (sheet)" in out["columns"] and "_kind" not in out["columns"]


def test_a_target_named_like_metadata_still_reads_its_result():
    out = campaign_rows({"target": "_kind", "base_rows": [{"x": 1, "_kind": 3.0}, {"x": 2, "_kind": None}]})
    assert [r["_kind"] for r in out["rows"]] == ["measured", "no_result"]
    assert [r["_kind (sheet)"] for r in out["rows"]] == [3.0, None]
    no_target = campaign_rows({"base_rows": [{"x": 1}]})
    assert no_target["renamed_columns"] == [] and no_target["rows"][0]["_kind"] == "measured"
