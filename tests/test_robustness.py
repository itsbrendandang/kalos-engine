"""Robustness reporting: proposal stability across seeds (opt-in) and the driver
second opinion (GP relevance vs rank drivers, collinearity).

Fast tests cover the comparison and assembly logic on constructed inputs; the
`slow` ones run real analyses: the feature off adds nothing, the feature on
changes no other number, two identical runs agree, an unstable case and a
collinear case are flagged, and every output matches
docs/schemas/analysis-robustness.schema.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from kalos.core.collinearity import COLLINEAR_LABEL, collinearity_report
from kalos.core.stability import (
    STABILITY_TOL,
    UNSTABLE_LABEL,
    batch_agreement,
    pairwise_separation,
    seed_schedule,
)
from kalos.envconfig import ConfigError
from kalos.portal.analysis_parts.driver_opinion import (
    GP_ONLY_LABEL,
    RANK_ONLY_LABEL,
    driver_second_opinion,
)
from kalos.portal.analysis_parts.stability import (
    SEEDS_VAR,
    annotate_proposals,
    configured_seeds,
    resolve_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "docs" / "schemas" / "analysis-robustness.schema.json").read_text())


# --------------------------------------------------------------------------- #
# A small JSON-schema checker (the subset the schema file uses), so the schema
# is enforced without adding a dependency.
# --------------------------------------------------------------------------- #

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def _is_type(value: Any, t: str) -> bool:
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[t])


def _check(value: Any, schema: dict, path: str = "$") -> list[str]:
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return _check(value, SCHEMA["$defs"][name], path)
    errs: list[str] = []
    types = schema.get("type")
    if types is not None:
        tl = types if isinstance(types, list) else [types]
        if not any(_is_type(value, t) for t in tl):
            return [f"{path}: {type(value).__name__} is not {tl}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > {schema['maximum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errs.append(f"{path}: missing {key}")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errs += _check(value[key], sub, f"{path}.{key}")
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict):
            for key, v in value.items():
                if key not in schema.get("properties", {}):
                    errs += _check(v, extra, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += _check(v, schema["items"], f"{path}[{i}]")
    return errs


def _schema_errors(result: dict) -> list[str]:
    return _check(json.loads(json.dumps(result, default=str)), SCHEMA)


def test_schema_checker_rejects_bad_values():
    bad = {"driver_second_opinion": {"available": "yes"}}
    assert _schema_errors(bad)
    bad_flag = {"proposals": [{"stability": {
        "agreement": 1.5, "seeds_matched": 0, "seeds_compared": 4,
        "median_nearest_distance": None, "unstable": False, "label": "fine"}}]}
    errs = _schema_errors(bad_flag)
    assert any("1.5" in e for e in errs) and any("fine" in e for e in errs)


# --------------------------------------------------------------------------- #
# Proposal stability: comparison logic
# --------------------------------------------------------------------------- #

LO = np.array([0.0, 0.0, 0.0])
HI = np.array([10.0, 1.0, 2.0])  # last column: a categorical code


def test_tolerance_is_the_optimizers_near_duplicate_rule():
    from kalos.core.optimize import NEAR_DUPLICATE_TOL, proposal_separation

    assert STABILITY_TOL == NEAR_DUPLICATE_TOL
    rng = np.random.default_rng(0)
    A = np.column_stack([rng.uniform(0, 10, 6), rng.uniform(0, 1, 6), rng.integers(0, 3, 6)])
    B = np.column_stack([rng.uniform(0, 10, 4), rng.uniform(0, 1, 4), rng.integers(0, 3, 4)])
    M = pairwise_separation(A, B, LO, HI, cat_dims=[2])
    for i in range(6):
        for j in range(4):
            assert M[i, j] == pytest.approx(proposal_separation(A[i], B[j], LO, HI, [2]))


def test_identical_reruns_agree_fully():
    S = np.array([[1.0, 0.2, 0], [5.0, 0.8, 1]])
    out = batch_agreement(S, [S.copy(), S[::-1].copy()], LO, HI, [2])
    assert out["batch_stability"] == 1.0
    assert out["n_unstable"] == 0
    assert all(p["label"] is None and p["agreement"] == 1.0 for p in out["per_proposal"])


def test_a_proposal_no_rerun_finds_is_flagged_unstable():
    S = np.array([[1.0, 0.2, 0], [5.0, 0.8, 1]])
    # Rerun batches all contain the first proposal (within 1% of range) but put
    # the second somewhere else: 3 of 4 off by > tol, or at another level.
    near = np.array([[1.05, 0.21, 0]])
    reruns = [
        np.vstack([near, [[9.0, 0.1, 1]]]),
        np.vstack([near, [[5.0, 0.8, 2]]]),          # same values, different level
        np.vstack([near, [[5.6, 0.8, 1]]]),          # 6% of range away: outside tol
        np.vstack([near, [[5.2, 0.82, 1]]]),         # 2% away: a match
    ]
    out = batch_agreement(S, reruns, LO, HI, [2])
    p0, p1 = out["per_proposal"]
    assert p0["agreement"] == 1.0 and not p0["unstable"]
    assert p1["seeds_matched"] == 1 and p1["seeds_compared"] == 4
    assert p1["agreement"] == 0.25 and p1["unstable"] and p1["label"] == UNSTABLE_LABEL
    assert p1["nearest_distance_per_seed"][1] == 1.0  # a differing level is the full range
    assert out["n_unstable"] == 1
    assert out["batch_stability"] == pytest.approx(0.625)


def test_failed_and_empty_reruns_are_counted_not_hidden():
    S = np.array([[1.0, 0.2, 0]])
    out = batch_agreement(S, [None, np.empty((0, 3)), S.copy()], LO, HI, [2])
    assert out["n_reruns_failed"] == 1 and out["n_reruns_compared"] == 2
    assert out["per_proposal"][0]["agreement"] == 0.5
    none = batch_agreement(S, [None, None], LO, HI, [2])
    assert none["batch_stability"] is None and none["per_proposal"][0]["agreement"] is None
    assert not none["per_proposal"][0]["unstable"]


def test_seed_schedule_starts_at_the_served_seed_and_never_repeats():
    seeds = seed_schedule(0, 10)
    assert seeds[0] == 0 and len(set(seeds)) == 10
    # The optimizer retries at seed + attempt (attempt < 4): no overlap either.
    assert len({s + a for s in seeds for a in range(4)}) == 40


def test_config_is_off_by_default_and_fails_closed(monkeypatch):
    monkeypatch.delenv(SEEDS_VAR, raising=False)
    assert configured_seeds() == 0
    monkeypatch.setenv(SEEDS_VAR, "1")
    assert configured_seeds() == 0  # one seed has nothing to compare against
    monkeypatch.setenv(SEEDS_VAR, "5")
    assert configured_seeds() == 5
    assert resolve_seeds(0) == 0 and resolve_seeds(3) == 3  # explicit wins
    for bad in ("11", "-1", "five"):
        monkeypatch.setenv(SEEDS_VAR, bad)
        with pytest.raises(ConfigError):
            configured_seeds()
    with pytest.raises(ValueError):
        resolve_seeds(99)


def test_annotate_copies_agreement_onto_each_proposal():
    block = batch_agreement(np.array([[1.0, 0.2, 0]]), [np.array([[9.0, 0.9, 2]])], LO, HI, [2])
    props = annotate_proposals([{"recipe": {}}], block)
    assert props[0]["stability"]["label"] == UNSTABLE_LABEL
    assert set(props[0]["stability"]) == {
        "agreement", "seeds_matched", "seeds_compared", "median_nearest_distance", "unstable", "label"}


# --------------------------------------------------------------------------- #
# Collinearity and the second-opinion assembly
# --------------------------------------------------------------------------- #

def _acid_base(n: int = 30, seed: int = 3) -> pd.DataFrame:
    """Acid and base dosed together (the Anagram case), one of them driving titer."""
    rng = np.random.default_rng(seed)
    acid = rng.uniform(0, 10, n)
    base = 0.9 * acid + rng.normal(0, 0.3, n)
    glucose = rng.uniform(5, 20, n)
    temp = rng.uniform(30, 37, n)
    y = 0.4 * acid + 0.1 * glucose - 0.05 * (temp - 34) ** 2 + rng.normal(0, 0.2, n)
    return pd.DataFrame({
        "acid_mM": acid.round(3), "base_mM": base.round(3), "glucose_gL": glucose.round(2),
        "temp_C": temp.round(2), "titer": y.round(3),
    })


def test_collinear_pair_is_flagged_with_vif():
    df = _acid_base()
    names = ["acid_mM", "base_mM", "glucose_gL", "temp_C"]
    rep = collinearity_report(df[names].to_numpy(float), names)
    assert [(p["a"], p["b"]) for p in rep["pairs"]] == [("acid_mM", "base_mM")]
    assert rep["pairs"][0]["r"] > 0.95
    assert set(rep["flagged"]) == {"acid_mM", "base_mM"}
    assert rep["vif"]["acid_mM"] > 10 and rep["vif"]["glucose_gL"] < 2
    assert rep["groups"] == [["acid_mM", "base_mM"]]
    assert rep["label"] == COLLINEAR_LABEL


def test_vif_catches_a_combination_with_no_strong_pair_and_exact_dependence():
    rng = np.random.default_rng(1)
    a, b, c = rng.uniform(0, 1, (3, 40))
    total = a + b + c  # each pair |r| ~ 0.55, but `total` is exactly determined
    rep = collinearity_report(np.column_stack([a, b, c, total]), ["a", "b", "c", "total"])
    assert rep["pairs"] == []
    assert set(rep["vif_perfect"]) == {"a", "b", "c", "total"}
    assert all(v is None for v in rep["vif"].values())
    assert set(rep["flagged"]) == {"a", "b", "c", "total"}


def test_independent_inputs_are_not_flagged_and_tiny_sheets_say_why():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (30, 4))
    rep = collinearity_report(X, list("abcd"))
    assert rep["flagged"] == [] and rep["pairs"] == [] and rep["vif_reason"] is None
    tiny = collinearity_report(X[:4], list("abcd"))
    assert tiny["vif"] == {} and "VIF needs more recipes" in tiny["vif_reason"]


def _gp(status: list[str], d: int) -> dict:
    return {
        "available": True, "reason": None, "method": "recipe_subsample_refit", "level": 0.95,
        "n_subsamples": 16, "n_fits_used": 16, "fraction": 0.8,
        "lengthscale": [0.5] * d, "lengthscale_ci": [[0.3, 4.0]] * d,
        "relevance": [1 / d] * d, "relevance_ci": [[0.1, 0.4]] * d, "status": status,
    }


def test_second_opinion_labels_each_kind_of_disagreement():
    drivers = [
        {"name": "acid", "rho": 0.8, "ci95": [0.6, 0.9], "significant": True},
        {"name": "base", "rho": 0.78, "ci95": [0.55, 0.9], "significant": True},
        {"name": "pH", "rho": 0.05, "ci95": [-0.3, 0.4], "significant": False},
        {"name": "glucose", "rho": 0.6, "ci95": [0.3, 0.8], "significant": True},
    ]
    col = {"pairs": [{"a": "acid", "b": "base", "r": 0.97}], "flagged": ["acid", "base"],
           "vif": {"acid": 20.0, "base": 20.0}, "vif_perfect": []}
    out = driver_second_opinion(
        drivers, ["acid", "base", "pH", "glucose"], col,
        _gp(["relevant", "negligible", "relevant", "relevant"], 4))
    kinds = {(d["name"], d["kind"]) for d in out["disagreements"]}
    assert kinds == {("acid", "collinear"), ("base", "collinear"), ("base", "rank_only"),
                     ("pH", "gp_only")}
    feats = {f["name"]: f for f in out["features"]}
    assert feats["base"]["flags"] == [RANK_ONLY_LABEL, COLLINEAR_LABEL]
    assert feats["pH"]["flags"] == [GP_ONLY_LABEL]
    assert feats["glucose"]["flags"] == []
    assert "acid" in next(d["detail"] for d in out["disagreements"] if d["kind"] == "rank_only")
    assert "Neither is a causal effect" in out["not_causal"]
    full = {**out, "collinearity": {**collinearity_report(np.zeros((0, 0)), []), **col}}
    assert not _schema_errors({"driver_second_opinion": full})


def test_second_opinion_without_a_gp_reading_judges_nothing_against_it():
    drivers = [{"name": "a", "rho": 0.9, "ci95": [0.8, 0.95], "significant": True}]
    out = driver_second_opinion(drivers, ["a"], collinearity_report(np.zeros((0, 0)), []),
                                {"available": False, "reason": "mixed"})
    assert out["features"][0]["gp_status"] == "not_judged"
    assert out["disagreements"] == []
    assert out["gp"]["reason"] == "mixed"
    assert not _schema_errors({"driver_second_opinion": out})


# --------------------------------------------------------------------------- #
# GP relevance (real fits)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_subsample_relevance_separates_relevant_inputs_and_is_deterministic():
    import torch

    from kalos.core.gp_relevance import subsample_relevance

    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (40, 3))
    y = np.sin(3 * X[:, 0]) + rng.normal(0, 0.05, 40)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    np_before = np.random.get_state()[1].copy()
    a = subsample_relevance(X, y, bounds=np.array([[0, 0, 0], [1, 1, 1]]), seed=0)
    assert torch.equal(torch.random.get_rng_state(), before)  # global RNG untouched
    assert np.array_equal(np.random.get_state()[1], np_before)
    b = subsample_relevance(X, y, bounds=np.array([[0, 0, 0], [1, 1, 1]]), seed=0)
    assert a == b
    assert a["available"] and a["status"][0] == "relevant"
    assert a["status"][1] != "relevant" and a["status"][2] != "relevant"
    assert a["relevance"][0] > 0.5


def test_subsample_relevance_refuses_too_few_recipes():
    from kalos.core.gp_relevance import subsample_relevance

    X = np.random.default_rng(0).uniform(0, 1, (12, 2))
    out = subsample_relevance(X, X[:, 0], bounds=np.array([[0, 0], [1, 1]]),
                              groups=[i // 2 for i in range(12)])
    assert not out["available"] and "6 distinct recipes" in out["reason"]


# --------------------------------------------------------------------------- #
# Full analyses
# --------------------------------------------------------------------------- #

def _two_feature() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 4, 24)
    ph = rng.uniform(5, 7, 24)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, 24)).round(3),
    })


def _egg_crate() -> pd.DataFrame:
    """Many equally good peaks: the acquisition has several equal optima, so
    which ones a seed lands on is arbitrary. A deliberately unstable case."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, (30, 2))
    y = np.cos(4 * np.pi * x[:, 0]) * np.cos(4 * np.pi * x[:, 1]) + rng.normal(0, 0.02, 30)
    return pd.DataFrame({"feed_a": x[:, 0].round(4), "feed_b": x[:, 1].round(4), "titer": y.round(4)})


def _strip_stability(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k not in ("proposal_stability", "timestamp")}
    out["proposals"] = [{k: v for k, v in p.items() if k != "stability"} for p in res["proposals"]]
    return out


def _volatile_free(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _volatile_free(v) for k, v in obj.items()
                if k not in ("timestamp", "fit_seconds", "created_at", "seconds", "workers")}
    if isinstance(obj, list):
        return [_volatile_free(v) for v in obj]
    return obj


def _dump(obj: Any) -> str:
    return json.dumps(_volatile_free(obj), sort_keys=True, default=str)


@pytest.mark.slow
def test_off_adds_nothing_and_on_changes_no_other_number(monkeypatch):
    from kalos.portal.analysis import _analyze

    monkeypatch.delenv(SEEDS_VAR, raising=False)
    off = _analyze(_two_feature(), "lipase_titer")
    assert "proposal_stability" not in off
    assert all("stability" not in p for p in off["proposals"])
    on = _analyze(_two_feature(), "lipase_titer", stability_seeds=3)
    block = on["proposal_stability"]
    assert block["available"] and block["n_seeds"] == 3 and block["n_reruns_compared"] == 2
    assert block["served_seed"] == on["seed"] == block["seeds"][0]
    assert _dump(_strip_stability(on)) == _dump(_strip_stability(off))
    assert all("stability" in p for p in on["proposals"])
    assert not _schema_errors(on) and not _schema_errors(off)


@pytest.mark.slow
def test_stability_is_deterministic_and_read_from_the_environment(monkeypatch):
    from kalos.portal.analysis import _analyze

    monkeypatch.setenv(SEEDS_VAR, "3")
    a = _analyze(_two_feature(), "lipase_titer")["proposal_stability"]
    b = _analyze(_two_feature(), "lipase_titer")["proposal_stability"]
    assert _dump(a) == _dump(b)
    assert a["varied"] == ["acquisition_optimizer_seed", "gp_hyperparameter_restart_seed"]


@pytest.mark.slow
def test_an_unstable_synthetic_case_is_flagged():
    from kalos.portal.analysis import _analyze

    res = _analyze(_egg_crate(), "titer", stability_seeds=3)
    block = res["proposal_stability"]
    assert block["available"], block["reason"]
    assert block["n_unstable"] >= 1
    assert block["batch_stability"] < 1.0
    flagged = [p for p in res["proposals"] if p["stability"]["unstable"]]
    assert flagged and all(p["stability"]["label"] == UNSTABLE_LABEL for p in flagged)
    assert not _schema_errors(res)


@pytest.mark.slow
def test_the_collinear_acid_base_case_is_flagged_in_the_analysis():
    from kalos.portal.analysis import _analyze

    res = _analyze(_acid_base(), "titer")
    so = res["driver_second_opinion"]
    assert so["available"] and so["gp"]["available"], so["gp"]["reason"]
    feats = {f["name"]: f for f in so["features"]}
    for name in ("acid_mM", "base_mM"):
        assert COLLINEAR_LABEL in feats[name]["flags"]
        assert feats[name]["collinear_with"] == [("base_mM" if name == "acid_mM" else "acid_mM")]
    assert COLLINEAR_LABEL not in feats["glucose_gL"]["flags"]
    assert {d["name"] for d in so["disagreements"] if d["kind"] == "collinear"} == {"acid_mM", "base_mM"}
    # The rank panel itself is unchanged by the second opinion.
    assert [d["name"] for d in res["drivers"]][:2] == ["base_mM", "acid_mM"] or \
        [d["name"] for d in res["drivers"]][:2] == ["acid_mM", "base_mM"]
    assert not _schema_errors(res)


@pytest.mark.slow
def test_parallel_reruns_match_serial(monkeypatch):
    """Runs only once perf/parallel-analysis (`kalos.core.parallel`) is on main."""
    parallel = pytest.importorskip("kalos.core.parallel")
    from kalos.portal.analysis import _analyze

    monkeypatch.setenv("KALOS_ANALYSIS_WORKERS", "1")
    serial = _analyze(_two_feature(), "lipase_titer", stability_seeds=3)
    monkeypatch.setenv("KALOS_ANALYSIS_WORKERS", "2")
    parallel.reset_stats()
    pooled = _analyze(_two_feature(), "lipase_titer", stability_seeds=3)
    assert parallel.stats["pool_tasks"] > 0
    assert _dump(pooled) == _dump(serial)
