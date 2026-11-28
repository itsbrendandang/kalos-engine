"""Clone keys in the campaign setup (docs/CLONES.md) and the optimizer's held
dimensions. No full analysis is run: the setup routes are checked against a
hand-written analysis blob, the optimizer against a small fitted GP."""
from __future__ import annotations

import numpy as np
import pytest

from kalos.portal.setup_settings import (
    DEFAULT_SETUP,
    DEFAULT_SETUP_HASH,
    SetupInvalid,
    analysis_setup,
    normalize_setup,
    plan_setup,
    setup_hash,
)

pytest.importorskip("fastapi")
from test_setup_api import H, _campaign, env  # noqa: E402,F401


def _strain_analysis(levels=("CHO-A12", "CHO-C33"), anonymize=False) -> dict:
    return {
        "target": "titer",
        "features": ["a", "b", "passage", "clone"],
        "categorical_features": ["clone"],
        "feature_ranges": {"a": [0.0, 10.0], "b": [1.0, 6.0], "passage": [8.0, 29.0]},
        "provenance": [{"name": n} for n in ("a", "b", "passage", "clone", "titer")],
        "strain_summary": {"available": True, "column": "clone",
                           "levels": [{"label": lv} for lv in levels]},
        "covariates": {"columns": ["passage"]},
        "run": {"anonymize": anonymize},
        "proposals": [],
    }


# --- normalization and hash stability ------------------------------------------ #
def test_a_setup_without_clone_keys_keeps_its_hash():
    assert setup_hash(normalize_setup({})) == DEFAULT_SETUP_HASH
    assert set(normalize_setup({})) == set(DEFAULT_SETUP)
    assert "strain_column" not in analysis_setup(normalize_setup({}))


def test_clone_keys_normalize_and_reach_the_analysis():
    doc = normalize_setup({"strain_column": "clone", "strain_level": " CHO-C33 ",
                           "covariates": ["thaw_id", "passage", "passage"]})
    assert doc["strain_column"] == "clone" and doc["strain_level"] == "CHO-C33"
    assert doc["covariates"] == ["passage", "thaw_id"]
    assert "strain_policy" not in doc  # the analysis defaults it to "fixed"
    a = analysis_setup(doc)
    assert a["strain_column"] == "clone" and a["covariates"] == ["passage", "thaw_id"]


@pytest.mark.parametrize("raw", [
    {"strain_policy": "fixed"},                                      # no column
    {"strain_column": "clone", "strain_policy": "pick"},
    {"strain_column": "clone", "strain_policy": "factor", "strain_level": "A"},
    {"strain_column": "clone", "strain_level": "x" * 65},
    {"strain_column": "clone", "covariates": ["clone"]},
    {"strain_column": "clone", "covariates": "passage"},
])
def test_bad_clone_keys_are_refused(raw):
    with pytest.raises(SetupInvalid):
        normalize_setup(raw)


# --- plan_setup: held dims and constraints with a clone ------------------------- #
def _box():
    return np.array([[0.0, 1.0, 8.0, 0.0], [10.0, 6.0, 29.0, 2.0]])


def test_constraints_with_a_clone_are_applied_when_enumerable():
    s = {"goal": "maximize", "bounds": {}, "linear_constraints": [
        {"coefficients": {"a": 1.0, "b": 1.0}, "op": "<=", "rhs": 12.0}]}
    names = ["a", "b", "passage", "clone"]
    refused = plan_setup(s, names=names, categorical=["clone"], bounds=_box(), target="t")
    assert refused.refusal["code"] == "constraints_with_categoricals"
    ok = plan_setup(s, names=names, categorical=["clone"], bounds=_box(), target="t",
                    held=["passage", "clone"])
    assert ok.refusal is None and ok.linear_constraints
    factor = plan_setup(s, names=names, categorical=["clone"], bounds=_box(), target="t",
                        held=["passage"], mixed_constraints_ok=True)
    assert factor.refusal is None


def test_a_bound_on_a_held_covariate_is_refused():
    s = {"goal": "maximize", "bounds": {"passage": {"lo": 8, "hi": 12}}, "linear_constraints": []}
    p = plan_setup(s, names=["a", "passage"], categorical=[], bounds=_box()[:, [0, 2]],
                   target="t", held=["passage"])
    assert p.refusal["code"] == "covariate_not_proposable"


# --- HTTP ------------------------------------------------------------------------ #
def test_put_setup_accepts_clone_keys_and_constraints(env):  # noqa: F811
    tc, store = env
    cid = _campaign(store, analysis=_strain_analysis())
    r = tc.put(f"/api/campaigns/{cid}/setup", json={
        "strain_column": "clone", "strain_level": "CHO-C33",
        "linear_constraints": [{"coefficients": {"a": 1, "b": 1}, "op": "<=", "rhs": 12}],
    }, headers=H("a-write"))
    assert r.status_code == 200, r.text
    got = tc.get(f"/api/campaigns/{cid}/setup", headers=H("a-read")).json()["setup"]
    assert got["strain_column"] == "clone" and got["strain_level"] == "CHO-C33"


def test_put_setup_refuses_an_unseen_clone_and_a_covariate_bound(env):  # noqa: F811
    tc, store = env
    cid = _campaign(store, analysis=_strain_analysis())
    r = tc.put(f"/api/campaigns/{cid}/setup", json={
        "strain_column": "clone", "strain_level": "CHO-NEW"}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "clone_not_in_training"
    assert "CHO-NEW" not in r.text  # never echoed
    r = tc.put(f"/api/campaigns/{cid}/setup", json={
        "bounds": {"passage": {"lo": 8, "hi": 12}}}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "covariate_not_proposable"
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"strain_column": "nope"}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "unknown_column"
    r = tc.put(f"/api/campaigns/{cid}/setup", json={"strain_column": "clone", "sequence": "MKT"},
               headers=H("a-write"))
    assert r.status_code in (400, 422)


def test_put_setup_cannot_check_levels_of_an_anonymized_analysis(env):  # noqa: F811
    tc, store = env
    cid = _campaign(store, analysis=_strain_analysis(levels=("level_ab12cd34",), anonymize=True))
    r = tc.put(f"/api/campaigns/{cid}/setup", json={
        "strain_column": "clone", "strain_level": "CHO-C33"}, headers=H("a-write"))
    assert r.status_code == 200, r.text


# --- optimizer: held dims -------------------------------------------------------- #
def _mixed_fit():
    from kalos.core.surrogate import Surrogate

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0, 1, 24), rng.uniform(0, 1, 24), rng.uniform(5, 30, 24),
                         np.repeat([0, 1, 2], 8)])
    y = X[:, 0] + X[:, 1] + 0.3 * X[:, 3] - 0.01 * X[:, 2] + rng.normal(0, 0.05, 24)
    b = np.array([[0, 0, 5, 0], [1, 1, 30, 2]], float)
    return Surrogate().fit(X, y, bounds=b, cat_dims=[3]), b


def test_fixed_features_hold_a_clone_and_a_covariate_with_a_linear_constraint():
    from kalos.core.optimize import propose_batch

    s, b = _mixed_fit()
    out = propose_batch(
        s, b, q=3, seed=0, cat_dims=[3], cat_cardinalities=[3],
        fixed_features={3: 0.0, 2: 12.0},
        linear_constraints=[([0, 1], [-1.0, -1.0], -1.2)],  # a + b <= 1.2
    ).X
    assert len(out)
    assert (out[:, 3] == 0).all() and np.allclose(out[:, 2], 12.0)
    assert (out[:, 0] + out[:, 1] <= 1.2 + 1e-6).all()


def test_linear_constraints_with_free_categoricals_use_exact_enumeration():
    from kalos.core.optimize import propose_batch

    s, b = _mixed_fit()
    out = propose_batch(
        s, b, q=2, seed=0, cat_dims=[3], cat_cardinalities=[3],
        linear_constraints=[([0, 1], [-1.0, -1.0], -1.2)],
    ).X
    assert len(out) and (out[:, 0] + out[:, 1] <= 1.2 + 1e-6).all()
    assert set(np.unique(out[:, 3])) <= {0.0, 1.0, 2.0}


def test_fixed_features_none_is_the_historical_call():
    from kalos.core.optimize import propose_batch

    s, b = _mixed_fit()
    a = propose_batch(s, b, q=2, seed=3, cat_dims=[3], cat_cardinalities=[3]).X
    c = propose_batch(s, b, q=2, seed=3, cat_dims=[3], cat_cardinalities=[3], fixed_features=None).X
    assert np.array_equal(a, c)


def test_the_registry_lead_reaches_the_analysis_setup(env):  # noqa: F811
    from kalos.campaigns.setup import setup_for_analysis

    tc, store = env
    cid = _campaign(store, analysis=_strain_analysis())
    assert tc.put(f"/api/campaigns/{cid}/clones/CHO-C33", json={"status": "lead"},
                  headers=H("a-write")).status_code == 200
    assert tc.put(f"/api/campaigns/{cid}/setup", json={"strain_column": "clone"},
                  headers=H("a-write")).status_code == 200
    s = setup_for_analysis(store, tenant="acme", campaign_id=cid)
    assert s["strain_column"] == "clone" and s["strain_lead"] == "CHO-C33"
    # a named level wins; the lead is not looked up
    assert tc.put(f"/api/campaigns/{cid}/setup", json={
        "strain_column": "clone", "strain_level": "CHO-A12"}, headers=H("a-write")).status_code == 200
    s = setup_for_analysis(store, tenant="acme", campaign_id=cid)
    assert s["strain_level"] == "CHO-A12" and "strain_lead" not in s
    # two leads: no default
    assert tc.put(f"/api/campaigns/{cid}/clones/CHO-A12", json={"status": "lead"},
                  headers=H("a-write")).status_code == 200
    assert tc.put(f"/api/campaigns/{cid}/setup", json={"strain_column": "clone"},
                  headers=H("a-write")).status_code == 200
    assert "strain_lead" not in setup_for_analysis(store, tenant="acme", campaign_id=cid)
