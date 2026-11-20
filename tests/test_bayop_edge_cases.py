"""BayOp's edge-case fixtures, ported as Kalos regression tests.

Source: the user's earlier BayOp service (Bioqore `voyager-ml-bayop`,
`tests/data/_generate_fixtures.py`, `tests/test_stress.py`,
`tests/test_errors.py`; see NOTICE). The fixtures are fabricated data (24 runs,
6 settings, a quadratic "titer" with a known optimum, seed 7); they are
regenerated here by the same recipe rather than copied as files.

What each fixture did in BayOp and what Kalos does (docs/COLD_START_DESIGN.md,
"BayOp edge cases" records the same table):

| fixture | BayOp | Kalos before this change | Kalos now |
|---|---|---|---|
| flat target (all 0.5) | proposed 3 "optimal" points from a flat GP | refused: `target_constant` (400) | same refusal, plus a space-filling plan from the design route |
| single row | proposed 3 points from a 1-point GP | refused: `too_few_rows` (400, needs 6) | same refusal, plus a gap-filling plan around the one run |
| NaN in an input and in the target | the GP fit raised; routed to FAIL | the blank-target run is excluded and counted; the blank `phosphate_gL` cell is read as absent (0) by the documented concentration-column convention and reported in `blank_cells_as_absent` (200) | same; the design skips an existing run with a blank setting and says so |
| column mismatch (bounds name a column the data lacks) | ValueError, routed to FAIL | setup bounds refused: `unknown_feature` (400) | same, and the design route refuses `unknown_factor` |
| missing bounds | routed to FAIL | (no equivalent: bounds were optional) | design refuses `factor_without_bounds` |
| large batch (20) stays in bounds | checked | (proposals are 5 per batch) | a 96-row design stays in bounds and on its steps |
"""
from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.core.design import ContinuousFactor, generate_design  # noqa: E402
from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402

# --- BayOp's fixture recipe (fabricated; names, ranges and values are synthetic) --- #
BOUNDS = {
    "glucose_gL": (1.0, 40.0), "nitrogen_gL": (0.1, 5.0), "phosphate_gL": (0.05, 3.0),
    "trace_metals_mLL": (0.1, 10.0), "temperature_C": (25.0, 37.0), "ph": (4.5, 7.5),
}
OPT = {"glucose_gL": 28.0, "nitrogen_gL": 3.2, "phosphate_gL": 1.4, "trace_metals_mLL": 6.0,
       "temperature_C": 30.0, "ph": 6.4}
TARGET = "titer_gL"


def bayop_input(n: int = 24, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({k: rng.uniform(lo, hi, size=n) for k, (lo, hi) in BOUNDS.items()})
    y = np.full(n, 6.0)
    for k, (lo, hi) in BOUNDS.items():
        y -= 4.0 * ((df[k].to_numpy() - OPT[k]) / (hi - lo)) ** 2
    df[TARGET] = np.clip(y + rng.normal(0.0, 0.05, size=n), 0.0, None)
    return df.round(4)


def flat() -> pd.DataFrame:
    df = bayop_input()
    df[TARGET] = 0.5
    return df


def single() -> pd.DataFrame:
    return bayop_input().head(1)


def with_nan() -> pd.DataFrame:
    df = bayop_input()
    df.loc[2, "phosphate_gL"] = np.nan
    df.loc[5, TARGET] = np.nan
    return df


def mismatch() -> pd.DataFrame:
    return bayop_input().rename(columns={"glucose_gL": "not_a_bound_key"})


# --- harness ------------------------------------------------------------------- #


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [{"token_sha256": _sha("w"), "subject": "bob@acme", "tenant": "acme", "scopes": ["read", "write"]}]
HDR = {"Authorization": "Bearer w"}


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


def _campaign(tc) -> str:
    r = tc.post("/api/campaigns", json={"name": "bayop"}, headers=HDR)
    assert r.status_code == 201
    return r.json()["id"]


def _upload(tc, cid: str, df: pd.DataFrame, target: str = TARGET):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return tc.post(f"/api/campaigns/{cid}/upload", files={"file": ("sheet.csv", buf.getvalue(), "text/csv")},
                   data={"target": target}, headers=HDR)


DESIGN_BODY = {"n": 8, "factors": {k: {"lo": lo, "hi": hi} for k, (lo, hi) in BOUNDS.items()}, "target": TARGET}


# --- the fixtures ----------------------------------------------------------------- #


def test_flat_target_is_refused_and_the_design_route_gives_a_plan(env):
    """Already handled before this change: `target_constant`. New: a plan."""
    tc, _ = env
    cid = _campaign(tc)
    r = _upload(tc, cid, flat())
    assert r.status_code == 400 and r.json()["code"] == "target_constant"
    d = tc.post(f"/api/campaigns/{cid}/design", json=DESIGN_BODY, headers=HDR)
    assert d.status_code == 200
    design = d.json()["design"]
    assert design["model_not_used"]["code"] == "no_measured_runs"
    assert len(design["rows"]) == 8


def test_single_row_is_refused_and_a_plan_fills_around_it(env):
    """Already handled: `too_few_rows` (a sheet needs 6 runs). New: gap filling
    around the one measured run (BayOp's `test_single_initial_experiment`)."""
    tc, _ = env
    cid = _campaign(tc)
    r = _upload(tc, cid, single())
    assert r.status_code == 400 and r.json()["code"] == "too_few_rows"
    row = single().iloc[0]
    facs = [ContinuousFactor(k, lo, hi) for k, (lo, hi) in BOUNDS.items()]
    res = generate_design(facs, n=3, seed=0, existing=np.array([[row[k] for k in BOUNDS]]))
    assert res.method == "maximin_fill" and res.n_existing == 1 and len(res.rows) == 3
    for rec in res.rows:
        assert all(lo <= rec[k] <= hi and np.isfinite(rec[k]) for k, (lo, hi) in BOUNDS.items())
    assert res.quality["min_distance_to_existing"] > 0.3


@pytest.mark.slow
def test_nan_in_an_input_and_in_the_target_is_counted_never_silently_imputed(env):
    """Already handled (BayOp's GP raised instead). The run with a blank target is
    excluded and counted. The blank `phosphate_gL` cell is read as absent (0)
    because a concentration-type column follows the documented absent-zero
    convention (docs/UPLOADS.md), and it is reported under
    `blank_cells_as_absent`, not hidden."""
    tc, _ = env
    cid = _campaign(tc)
    r = _upload(tc, cid, with_nan())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 23 and body["n_dropped_incomplete"] == 1
    drops = body["data_drops"]
    assert drops["cells_by_column"] == {TARGET: {"missing": 1}}
    assert drops["blank_cells_as_absent"] == {"phosphate_gL": 1}


def test_nan_existing_run_is_skipped_by_the_design_and_said(env):
    facs = [ContinuousFactor(k, lo, hi) for k, (lo, hi) in BOUNDS.items()]
    df = with_nan()
    res = generate_design(facs, n=4, seed=0, existing=df[list(BOUNDS)].to_numpy(float))
    assert res.n_existing == 23 and any("blank setting" in n for n in res.notes)


def test_column_mismatch_is_refused(env):
    """Already handled: setup bounds naming a column the analysis lacks are
    refused (`unknown_feature`), and an absent target is `target_not_found`.
    New: the design route refuses `unknown_factor`."""
    tc, store = env
    cid = _campaign(tc)
    df = bayop_input()
    store.seed(df, TARGET, list(BOUNDS), tenant="acme", campaign_id=cid, analysis={
        "target": TARGET, "features": list(BOUNDS), "categorical_features": [],
        "feature_ranges": {k: [float(df[k].min()), float(df[k].max())] for k in BOUNDS}, "proposals": [],
    })
    bad = {"bounds": {"not_a_bound_key": {"lo": 1.0, "hi": 40.0}}}
    r = tc.put(f"/api/campaigns/{cid}/setup", json=bad, headers=HDR)
    assert r.status_code == 400 and r.json()["code"] == "unknown_feature"
    r = tc.post(f"/api/campaigns/{cid}/design", json={"n": 4, "factors": {"not_a_bound_key": {"step": 1.0}}},
                headers=HDR)
    assert r.status_code == 400 and r.json()["code"] == "unknown_factor"
    cid2 = _campaign(tc)
    r = _upload(tc, cid2, mismatch(), target="not_a_column")
    assert r.status_code == 400 and r.json()["code"] == "target_not_found"


def test_missing_bounds_are_refused_by_the_design(env):
    tc, _ = env
    cid = _campaign(tc)
    r = tc.post(f"/api/campaigns/{cid}/design", json={"n": 4, "factors": {"glucose_gL": {"step": 0.5}}},
                headers=HDR)
    assert r.status_code == 400 and r.json()["code"] == "factor_without_bounds"


def test_largest_design_stays_in_bounds_and_on_its_steps(env):
    tc, _ = env
    cid = _campaign(tc)
    body = {"n": 96, "factors": {**DESIGN_BODY["factors"], "ph": {"lo": 4.5, "hi": 7.5, "step": 0.05}}}
    r = tc.post(f"/api/campaigns/{cid}/design", json=body, headers=HDR)
    assert r.status_code == 200
    rows = [x["recipe"] for x in r.json()["design"]["rows"]]
    assert len(rows) == 96
    for rec in rows:
        assert all(lo <= rec[k] <= hi for k, (lo, hi) in BOUNDS.items())
        k = (rec["ph"] - 4.5) / 0.05
        assert abs(k - round(k)) < 1e-9
