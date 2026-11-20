"""The campaign setup is ENFORCED in proposal generation (kalos.portal.analysis
`_analyze(setup=...)`, kalos.portal.setup_settings.plan_setup,
kalos.core.optimize `maximize=False`).

- no setup (and the default setup) leaves the proposals byte-identical;
- goal=minimize proposes toward LOW target values, keeps preds in the user's
  units, and makes `best`/the incumbent the minimum;
- every proposal satisfies the user bounds and linear constraints;
- a setup that cannot apply (categoricals + constraints, infeasible on this
  sheet) withholds proposals with a structured `setup.refusal`;
- end to end over HTTP: upload and re-analyze apply the stored setup and
  `settings_in_sync` tracks it.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis import _analyze
from kalos.portal.setup_settings import analysis_setup, normalize_setup


def _monotone(n: int = 18, seed: int = 3) -> pd.DataFrame:
    """titer rises with both inputs: maximize goes to the top corner, minimize to 0."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    y = 2.0 * a + b + rng.normal(0, 0.2, n)
    return pd.DataFrame({"a": a.round(3), "b": b.round(3), "titer": y.round(3)})


def _setup(**doc) -> dict:
    return analysis_setup(normalize_setup(doc))


@pytest.fixture(scope="module")
def baseline() -> dict:
    return _analyze(_monotone(), "titer")


@pytest.fixture(scope="module")
def minimized() -> dict:
    return _analyze(_monotone(), "titer", setup=_setup(goal="minimize"))


def test_no_setup_has_no_setup_fields(baseline):
    assert "setup" not in baseline and "goal" not in baseline
    assert baseline["proposals"]


def test_default_setup_gives_byte_identical_proposals(baseline):
    for doc in ({}, {"goal": "maximize", "target_unit": "g/L"}):
        out = _analyze(_monotone(), "titer", setup=_setup(**doc))
        assert json.dumps(out["proposals"], sort_keys=True) == json.dumps(
            baseline["proposals"], sort_keys=True)
        assert out["best"] == baseline["best"]
        assert out["setup"]["status"] == "applied" and out["goal"] == "maximize"
        assert out["setup"]["hash"] == hashlib.sha256(json.dumps(
            normalize_setup(doc), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_minimize_proposes_low_targets_in_user_units(baseline, minimized):
    y = _monotone()["titer"].to_numpy()
    assert minimized["goal"] == "minimize"
    assert minimized["setup"]["goal"] == "minimize"
    assert minimized["best"] == round(float(y.min()), 4)
    assert minimized["noise"]["best_single"] == pytest.approx(float(y.min()), abs=1e-3)
    preds = [p["pred"] for p in minimized["proposals"]]
    base_preds = [p["pred"] for p in baseline["proposals"]]
    assert minimized["proposals"]
    assert np.mean(preds) < np.mean(base_preds)
    # the best proposal goes below (or to) the lowest measured value's neighborhood
    assert min(preds) < np.quantile(y, 0.1)
    # preds are in the target's units and sign (no negation leaks): the GP's own
    # prediction at the proposed recipe matches 2a + b closely on this clean sheet
    for p in minimized["proposals"]:
        r = p["recipe"]
        assert abs(p["pred"] - (2 * r["a"] + r["b"])) < 2.0
    exploit = [p for p in minimized["proposals"] if p["mode"] == "exploit"]
    assert all("predicted low" in p["reason"] for p in exploit)
    # the response surface is fixed at the LOWEST run
    surf = minimized.get("response_surface")
    if surf:
        low = _monotone().iloc[int(np.argmin(y))]
        assert surf["fixed_recipe"]["a"] == pytest.approx(low["a"], abs=1e-3)
    # drivers keep their sign meaning: titer still rises with a
    drv = {d["name"]: d["rho"] for d in minimized["drivers"]}
    assert drv["a"] > 0
    assert any(w["code"] == "gp_shapes_swept_at_highest_run" for w in minimized["setup"]["warnings"])


def test_bounds_and_constraints_hold_for_every_proposal():
    setup = _setup(
        goal="maximize",
        bounds={"a": {"lo": 2, "hi": 6}, "b": {"lo": -5, "hi": 8}},
        linear_constraints=[{"coefficients": {"a": 1, "b": 1}, "op": "<=", "rhs": 9},
                            {"coefficients": {"a": 2, "b": -1}, "op": ">=", "rhs": 0}],
    )
    out = _analyze(_monotone(), "titer", setup=setup)
    block = out["setup"]
    assert block["status"] == "applied", block
    assert out["proposals"]
    for p in out["proposals"]:
        a, b = p["recipe"]["a"], p["recipe"]["b"]
        assert 2 - 1e-6 <= a <= 6 + 1e-6
        assert b <= 8 + 1e-6
        assert a + b <= 9 + 1e-5
        assert 2 * a - b >= -1e-5
    assert block["n_proposals_dropped"] == 0
    assert len(block["constraints_applied"]) == 2
    # b's lower bound reaches past the data: flagged, and the applied range is the data's
    bb = block["bounds_applied"]["b"]
    assert bb["extrapolates"] is True and bb["applied"][0] == bb["observed"][0]
    assert block["bounds_applied"]["a"]["extrapolates"] is False
    assert any(w["code"] == "bound_extrapolates" and w["feature"] == "b" for w in block["warnings"])


def test_minimize_with_constraints_stays_feasible():
    setup = _setup(goal="minimize", linear_constraints=[
        {"coefficients": {"a": 1, "b": 1}, "op": ">=", "rhs": 8}])
    out = _analyze(_monotone(), "titer", setup=setup)
    assert out["setup"]["status"] == "applied"
    for p in out["proposals"]:
        assert p["recipe"]["a"] + p["recipe"]["b"] >= 8 - 1e-5
    # pushed toward the low-titer edge of the feasible region: a small, b large
    assert min(p["pred"] for p in out["proposals"]) < 14


def test_bound_on_a_column_this_sheet_does_not_model_is_a_warning():
    out = _analyze(_monotone(), "titer", setup=_setup(bounds={"not_here": {"lo": 0, "hi": 1}}))
    assert out["setup"]["status"] == "applied"
    assert out["setup"]["warnings"][0]["code"] == "bound_feature_not_modeled"
    assert out["proposals"]


def test_constraints_with_categoricals_withhold_proposals():
    df = _monotone(20)
    df["medium"] = ["A", "B"] * 10
    out = _analyze(df, "titer", setup=_setup(linear_constraints=[
        {"coefficients": {"a": 1}, "op": "<=", "rhs": 5}]))
    block = out["setup"]
    assert block["status"] == "refused"
    assert block["refusal"]["code"] == "constraints_with_categoricals"
    assert block["constraints_applied"] == []
    assert out["proposals"] == []
    assert out["proposal_notice"]["code"] == "setup_not_applied"
    # the rest of the analysis is still there
    assert out["cv_spearman"] is not None and out["drivers"]


def test_constraint_infeasible_on_this_sheet_withholds_proposals():
    out = _analyze(_monotone(), "titer", setup=_setup(linear_constraints=[
        {"coefficients": {"a": 1, "b": 1}, "op": ">=", "rhs": 100}]))
    assert out["setup"]["status"] == "refused"
    assert out["setup"]["refusal"]["code"] == "infeasible_constraints"
    assert out["proposals"] == []


# --- end to end over HTTP ---------------------------------------------------------- #


def test_upload_and_reanalyze_apply_the_stored_setup(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        tc = TestClient(app)
        cid = tc.post("/api/campaigns", json={"name": "min"}).json()["id"]
        r = tc.put(f"/api/campaigns/{cid}/setup",
                   json={"goal": "minimize", "bounds": {"a": {"lo": 0, "hi": 5}}})
        assert r.status_code == 200, r.text
        h = r.json()["hash"]
        csv = _monotone().to_csv(index=False).encode()
        up = tc.post(f"/api/campaigns/{cid}/upload", files={"file": ("m.csv", csv, "text/csv")},
                     data={"target": "titer"})
        assert up.status_code == 200, up.text
        res = up.json()
        assert res["setup"]["hash"] == h and res["goal"] == "minimize"
        assert all(p["recipe"]["a"] <= 5 + 1e-6 for p in res["proposals"])
        view = tc.get(f"/api/campaigns/{cid}/setup").json()
        assert view["settings_in_sync"] is True
        assert view["analysis_setup_hash"] == h
        summary = tc.get(f"/api/campaigns/{cid}").json()["summary"]
        assert summary["goal"] == "minimize"
        assert summary["best"] == pytest.approx(float(_monotone()["titer"].min()))

        r = tc.put(f"/api/campaigns/{cid}/setup", json={"goal": "maximize"})
        assert r.json()["settings_in_sync"] is False
        run = tc.post(f"/api/campaigns/{cid}/runs",
                      json={"recipes": [res["proposals"][0]]}).json()["started"][0]
        assert tc.post(f"/api/campaigns/{cid}/runs/{run['id']}/result",
                       json={"value": 1.0}).status_code == 200
        re = tc.post(f"/api/campaigns/{cid}/reanalyze")
        assert re.status_code == 200, re.text
        assert re.json()["analysis"]["goal"] == "maximize"
        assert tc.get(f"/api/campaigns/{cid}/setup").json()["settings_in_sync"] is True
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def test_cv_topk_ranks_the_best_runs_for_the_goal(baseline, minimized):
    """E-S2: cv_topk asks "did the model pick the k recipes I would advance?".
    Under goal=minimize those are the k LOWEST, not the k highest."""
    from kalos.core.evaluation import top_k_overlap

    assert baseline["cv_topk"]["direction"] == "highest"
    assert minimized["cv_topk"]["direction"] == "lowest"
    for out, sign in ((baseline, 1.0), (minimized, -1.0)):
        oof = np.asarray(out["oof"], float)
        k = out["cv_topk"]["k"]
        expected = top_k_overlap(sign * oof[:, 0], sign * oof[:, 1], k)
        assert out["cv_topk"]["overlap"] == round(expected["overlap"], 3)


def test_cv_topk_under_minimize_is_not_the_maximize_number():
    """A sheet whose high end is learnable but whose low end is a noisy floor: the
    model picks the 5 highest perfectly and the 5 lowest poorly. Before the fix a
    minimize campaign was shown the highest-first overlap (1.0)."""
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 10, 24)
    b = rng.uniform(0, 10, 24)
    y = np.maximum(0, 3 * a - 15) + 5 + rng.normal(0, 0.3, 24)
    df = pd.DataFrame({"a": a.round(3), "b": b.round(3), "titer": y.round(3)})
    hi = _analyze(df, "titer", setup=_setup(goal="maximize"))["cv_topk"]
    lo = _analyze(df, "titer", setup=_setup(goal="minimize"))["cv_topk"]
    assert hi["direction"] == "highest" and hi["overlap"] == 1.0
    assert lo["direction"] == "lowest" and lo["overlap"] <= 0.4
