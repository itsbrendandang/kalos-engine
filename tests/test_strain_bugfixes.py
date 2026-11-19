"""Regression tests for the strain probe bugs (/tmp/probe_strain.py, STRAIN_VIEWS.md
section 1, ML_AUDIT d4 / N2). One synthetic sheet: 24 runs, 3 CHO clones with
true offsets 0 / +0.6 / +1.1 g/L, a passage covariate, pH / temperature / feed.

1. A clone column is categorical even when its IDs are numeric (no `clone: 329.44`).
2. Covariates (passage) are fitted but never proposed.
3. Categorical inputs appear in the driver / correlation panels as skipped with a
   reason, and in `categorical_effects` with a measured association.
4. gp_shapes works on a mixed (categorical) fit. Cause: the sweep passed the
   d_cont continuous columns to a d-column mixed GP (tensor-size RuntimeError).
5. A clone column plus a linear constraint no longer withholds every proposal.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis import _analyze
from kalos.portal.setup_settings import analysis_setup, normalize_setup

pytestmark = pytest.mark.slow

CLONES = ("CHO-A12", "CHO-B07", "CHO-C33")
OFFSET = {"CHO-A12": 0.0, "CHO-B07": 0.6, "CHO-C33": 1.1}


def probe_sheet(numeric: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 24
    clone = np.repeat(CLONES, 8)
    ph = rng.uniform(6.8, 7.2, n)
    t = rng.uniform(33, 37, n)
    feed = rng.uniform(2, 6, n)
    passage = rng.integers(8, 30, n)
    titer = (np.array([OFFSET[c] for c in clone]) + 2 + 0.3 * (feed - 4)
             - 0.4 * (t - 35) ** 2 / 4 + rng.normal(0, 0.2, n))
    ids = np.repeat([101, 204, 377], 8) if numeric else clone
    return pd.DataFrame({"run_id": range(n), "clone": ids, "passage": passage, "pH": ph,
                         "temp_C": t, "feed_pct": feed, "titer_g_L": titer})


@pytest.fixture(scope="module")
def text_result() -> dict:
    return _analyze(probe_sheet())


@pytest.fixture(scope="module")
def numeric_result() -> dict:
    return _analyze(probe_sheet(numeric=True))


# 1 ---------------------------------------------------------------------------- #
def test_numeric_clone_ids_are_categorical_and_proposals_name_real_clones(numeric_result):
    r = numeric_result
    assert "clone" in r["categorical_features"]
    assert "clone" not in r["feature_ranges"]  # not a continuous box dimension
    assert r["proposals"]
    for p in r["proposals"]:
        assert isinstance(p["recipe"]["clone"], str)
        assert p["recipe"]["clone"] in {"101", "204", "377"}
    s = r["strain_summary"]
    assert s["available"] and s["column"] == "clone" and s["numeric_looking_labels"]
    assert {lv["label"] for lv in s["levels"]} == {"101", "204", "377"}
    found = [f for f in r["validation"]["findings"] if f["check"] == "strain_column_detected"]
    assert found and found[0]["detail"]["numeric_looking"] is True


def test_declared_numeric_clone_with_roles_is_categorical():
    from kalos.domains import ColumnRoles

    df = probe_sheet(numeric=True)
    roles = ColumnRoles.from_dict({"target": "titer_g_L", "strain": "clone",
                                   "features": ["pH", "temp_C", "feed_pct"]})
    r = _analyze(df, roles=roles)
    assert r["categorical_features"] == ["clone"]
    assert r["strain_summary"]["policy"] == "fixed"
    assert {p["recipe"]["clone"] for p in r["proposals"]} == {r["strain_summary"]["fixed_level"]}


# 2 ---------------------------------------------------------------------------- #
def test_passage_is_fitted_but_never_proposed(text_result):
    r = text_result
    assert "passage" in r["features"]  # a model input
    assert r["covariates"]["fitted"] == ["passage"]
    held = r["covariates"]["held"]["passage"]["value"]
    for p in r["proposals"]:
        assert "passage" not in p["recipe"]
        assert p["held"]["passage"] == {"value": held, "basis": "median of the fitted runs",
                                       "proposed": False}
    assert "passage" not in r["proposal_features"]
    prov = {row["name"]: row for row in r["provenance"]}
    assert prov["passage"]["role"] == "covariate" and prov["clone"]["role"] == "strain"


def test_passage_missing_on_some_runs_is_recorded_not_fitted_and_runs_kept():
    df = probe_sheet()
    df.loc[[3, 11], "passage"] = np.nan
    r = _analyze(df)
    assert "passage" not in r["features"]
    assert "passage" in r["covariates"]["not_fitted"]
    assert r["n"] == 24  # nothing dropped, nothing imputed
    assert all("passage" not in p["recipe"] for p in r["proposals"])


# 3 ---------------------------------------------------------------------------- #
def test_clone_is_reported_in_driver_and_correlation_panels(text_result):
    r = text_result
    assert [d["feature"] for d in r["drivers_skipped"]] == ["clone"]
    assert "categorical" in r["drivers_skipped"][0]["reason"]
    assert [d["feature"] for d in r["correlation_skipped"]] == ["clone"]
    eff = {f["name"]: f for f in r["categorical_effects"]["features"]}["clone"]
    assert eff["p"] is not None and eff["p"] < 0.01 and eff["survives_fdr"]
    assert eff["epsilon_sq"] > 0.3
    assert [lv["level"] for lv in eff["levels"]] == list(CLONES)
    # and slices: skipped with a reason, plus per-level slices
    assert {"feature": "clone", "reason": "categorical"} in r["slices_skipped"]
    assert set(r["slices_by_level"]["clone"]) == set(CLONES)


# 4 ---------------------------------------------------------------------------- #
def test_gp_shapes_available_on_a_mixed_fit(text_result):
    g = text_result["gp_shapes"]
    assert g["available"] is True, g["reason"]
    assert {f["name"] for f in g["features"]} == {"passage", "pH", "temp_C", "feed_pct"}
    assert any("categorical" in u for u in g["unmodeled"])
    temp = next(f for f in g["features"] if f["name"] == "temp_C")
    assert temp["shape"] == "interior_optimum"  # the simulated peak at 35 C


def test_gp_shape_report_sweeps_full_width_rows():
    """Unit-level: the sweep keeps the categorical code; the old call (continuous
    slice only) still declines with a reason instead of raising."""
    from kalos.core.gp_shape import gp_shape_report
    from kalos.core.surrogate import Surrogate

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0, 1, 24), rng.uniform(0, 1, 24), np.repeat([0, 1, 2], 8)])
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 2] + rng.normal(0, 0.05, 24)
    s = Surrogate().fit(X, y, bounds=np.array([[0, 0, 0], [1, 1, 2]]), cat_dims=[2])
    old = gp_shape_report(s, X[:, :2], y, feature_names=["a", "b"]).to_dict()
    assert old["available"] is False and "RuntimeError" in old["reason"]
    new = gp_shape_report(s, X, y, feature_names=["a", "b"], sweep_columns=[0, 1]).to_dict()
    assert new["available"] is True and len(new["features"]) == 2


# 5 ---------------------------------------------------------------------------- #
def _lin_setup(**extra) -> dict:
    return analysis_setup(normalize_setup({
        "strain_column": "clone", **extra,
        "linear_constraints": [{"coefficients": {"feed_pct": 1.0}, "op": "<=", "rhs": 5.0}],
    }))


def test_clone_plus_linear_constraint_proposes_with_a_fixed_clone():
    r = _analyze(probe_sheet(), setup=_lin_setup(strain_level="CHO-C33"))
    assert r["setup"]["status"] == "applied", r["setup"]["refusal"]
    assert len(r["proposals"]) == 5
    for p in r["proposals"]:
        assert p["recipe"]["clone"] == "CHO-C33"
        assert p["recipe"]["feed_pct"] <= 5.0 + 1e-6


def test_clone_plus_linear_constraint_proposes_in_factor_mode():
    r = _analyze(probe_sheet(), setup=_lin_setup(strain_policy="factor"))
    assert r["setup"]["status"] == "applied", r["setup"]["refusal"]
    assert r["proposals"]
    assert all(p["recipe"]["feed_pct"] <= 5.0 + 1e-6 for p in r["proposals"])


def test_results_are_strict_json(text_result, numeric_result):
    json.dumps(text_result, allow_nan=False)
    json.dumps(numeric_result, allow_nan=False)
