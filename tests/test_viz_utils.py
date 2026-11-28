"""Tests for viz_utils fixes, leak_free_cv deduplication, parallel _oof, and operating_ranges."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from kalos.core.replicates import aggregate_replicates
from kalos.portal.analysis_parts.cv_grouping import leak_free_cv
from kalos.portal.viz_utils import (
    _correlation_spearman,
    _embedding_pca,
    _response_surface_gp,
)


def test_response_surface_gp_batched_matches_single_row_exactly() -> None:
    rng = np.random.default_rng(7)
    X_fit = rng.uniform(0.0, 10.0, size=(15, 3))
    y_fit = 2.0 * X_fit[:, 0] - 0.5 * X_fit[:, 1] + 0.1 * X_fit[:, 2]
    bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    drv = [
        {"name": "f0", "rho": 0.9},
        {"name": "f1", "rho": -0.6},
        {"name": "f2", "rho": 0.2},
    ]

    calls: list[tuple[int, int]] = []

    class _FakeSurrogate:
        def posterior(self, X: np.ndarray, *, observation_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
            assert not observation_noise
            arr = np.asarray(X, dtype=float)
            calls.append(arr.shape)
            mean = 1.5 * arr[:, 0] - 0.3 * (arr[:, 1] - 5.0) ** 2 + 0.25 * arr[:, 2]
            sd = 0.1 + 0.01 * np.abs(arr[:, 0] - arr[:, 1])
            return mean, sd

    s = _FakeSurrogate()
    res = _response_surface_gp(
        s,
        X_fit,
        y_fit,
        ["f0", "f1", "f2"],
        [],
        {},
        {},
        bounds,
        drv,
        None,
        float(y_fit.max()),
        False,
    )
    assert res is not None and "reason" not in res
    assert calls == [(441, 3)]

    # Verify exact row-major [i over y_vals][j over x_vals] correspondence against single-row calls.
    best_row = X_fit[int(np.argmax(y_fit))]
    x_vals = np.linspace(0.0, 10.0, 21)
    y_vals = np.linspace(0.0, 10.0, 21)
    expected_z: list[list[float]] = []
    expected_std: list[list[float]] = []
    for yv in y_vals:
        row_z: list[float] = []
        row_s: list[float] = []
        for xv in x_vals:
            pt = best_row.copy()
            pt[0] = xv
            pt[1] = yv
            m, sd = s.posterior(pt.reshape(1, -1))
            row_z.append(round(float(m[0]), 4))
            row_s.append(round(float(sd[0]), 4))
        expected_z.append(row_z)
        expected_std.append(row_s)

    assert res["x_feature"] == "f0"
    assert res["y_feature"] == "f1"
    assert res["z"] == expected_z
    assert res["std"] == expected_std


def test_embedding_pca_encodes_integer_categorical_codes() -> None:
    # Identical continuous column, varying integer categorical codes 0, 1, 2.
    X = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ])
    y = np.array([1.0, 1.2, 2.0, 2.1, 3.0, 3.2])
    res = _embedding_pca(
        X,
        y,
        ["temp"],
        ["medium"],
        cat_dims_map={"medium": ("M1", "M2", "M3")},
    )
    assert res is not None and "reason" not in res
    assert res["n_rows"] == 6
    assert res["n_features"] == 4  # 1 continuous + 3 one-hot levels
    pts = np.array(res["points"])
    # Rows with different categorical codes must separate in PCA space, while identical rows match.
    assert np.allclose(pts[0, :2], pts[1, :2])
    assert np.allclose(pts[2, :2], pts[3, :2])
    assert np.allclose(pts[4, :2], pts[5, :2])
    assert not np.allclose(pts[0, :2], pts[2, :2])
    assert not np.allclose(pts[2, :2], pts[4, :2])


def test_correlation_spearman_recipe_averaged_with_groups_and_backward_compat() -> None:
    rng = np.random.default_rng(11)
    # 6 distinct recipes, replicated unevenly (recipe 0 repeated 6 times, others twice).
    X_base = rng.uniform(0.0, 10.0, size=(6, 2))
    y_base = 2.0 * X_base[:, 0] - X_base[:, 1]
    repeats = [6, 2, 2, 2, 2, 2]
    X_rows = np.vstack([np.tile(X_base[i], (r, 1)) for i, r in enumerate(repeats)])
    y_rows = np.concatenate([np.full(r, y_base[i]) + rng.normal(0, 0.05, r) for i, r in enumerate(repeats)])
    groups = np.concatenate([np.full(r, f"recipe_{i}") for i, r in enumerate(repeats)])

    # Default groups=None stays backward-compatible (row-level).
    raw_corr = _correlation_spearman(X_rows, y_rows, ["a", "b"])
    assert raw_corr is not None and "reason" not in raw_corr
    assert raw_corr["n_units"] == len(y_rows) == raw_corr["n_rows"] == 16

    # Grouped by recipe: n_units == 6, n_rows == 16, correlations match aggregated recipe means.
    grp_corr = _correlation_spearman(X_rows, y_rows, ["a", "b"], groups=groups)
    assert grp_corr is not None and "reason" not in grp_corr
    assert grp_corr["unit"] == "recipe"
    assert grp_corr["n_units"] == 6
    assert grp_corr["n_rows"] == 16
    assert grp_corr["labels"] == ["a", "b", "target_mean"]

    Xr, yr, _, _ = aggregate_replicates(X_rows, y_rows, groups=groups)
    expected_rho = round(float(spearmanr(Xr[:, 0], yr).statistic), 3)
    assert grp_corr["matrix"][0][2] == expected_rho
    assert grp_corr["matrix"][2][0] == expected_rho


def test_explain_passes_recipe_key_to_correlation_spearman() -> None:
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(5)
    recipes = rng.uniform(0.0, 5.0, size=(8, 2))
    rows = []
    for x0, x1 in recipes:
        y_true = 1.5 * x0 - 0.8 * x1
        for _ in range(3):
            rows.append({"a": round(float(x0), 4), "b": round(float(x1), 4), "titer": y_true + rng.normal(0, 0.1)})
    res = _analyze(pd.DataFrame(rows), "titer")
    corr = res["correlation"]
    assert corr is not None
    assert corr["n_units"] == 8
    assert corr["n_rows"] == 24


def test_leak_free_cv_deduplicates_identical_candidate_partitions() -> None:
    def _rep(rho: float, n_oof: int = 24) -> dict[str, Any]:
        return {"spearman": rho, "ci95": (rho - 0.1, rho + 0.1), "n_groups": 6, "n_oof": n_oof}

    prim = _rep(0.8)
    prim_groups = np.arange(24)
    cand_groups = np.repeat(np.arange(6), 4)
    # Two candidates with distinct names/bases whose partitions differ from primary but equal each other.
    cands = [
        {"basis": "column", "column": "batch", "groups": cand_groups, "n_levels": 6, "merged_with_replicates": True},
        {"basis": "campaign_round", "column": "round", "groups": cand_groups + 100, "n_levels": 6, "merged_with_replicates": True},
    ]
    calls: list[int] = []

    def run_cv(g: Any) -> dict[str, Any]:
        calls.append(len(set(np.asarray(g).tolist())))
        return _rep(0.25)

    block, rep, _groups, basis = leak_free_cv(
        prim, prim_groups, "replicate_clusters", None, cands, run_cv,
        n_exact_recipes=24, n_clusters=24, strict_candidates=False,
    )
    assert calls == [6]
    assert basis == "column"
    assert rep["spearman"] == 0.25
    assert block["candidates"][1].get("same_partition_as") is None
    assert block["candidates"][2]["same_partition_as"] == "column"
    assert block["candidates"][2]["spearman"] == 0.25


def test_leak_free_cv_duplicate_of_coverage_gap_candidate_does_not_bind() -> None:
    prim = {"spearman": 0.7, "ci95": (0.5, 0.9), "n_groups": 6, "n_oof": 12}
    prim_groups = np.arange(12)
    cand_groups = np.repeat(np.arange(3), 4)
    cands = [
        {"basis": "column", "column": "batch", "groups": cand_groups, "n_levels": 3, "merged_with_replicates": True},
        {"basis": "column", "column": "plate", "groups": cand_groups + 10, "n_levels": 3, "merged_with_replicates": True},
    ]
    calls = 0

    def run_cv(_g: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"spearman": -0.4, "ci95": (-0.6, -0.2), "n_groups": 2, "n_oof": 8}

    block, rep, _groups, basis = leak_free_cv(
        prim, prim_groups, "replicate_clusters", None, cands, run_cv,
        n_exact_recipes=12, n_clusters=12, strict_candidates=True,
    )
    assert calls == 1
    assert basis == "replicate_clusters" and rep is prim
    assert block["candidates"][1]["evaluable"] is False
    assert block["candidates"][2]["evaluable"] is False
    assert block["candidates"][2]["same_partition_as"] == "column"


def test_parallel_oof_and_grouped_cv_report_match_sequential() -> None:
    pytest.importorskip("botorch")
    from kalos.core.evaluation import _oof, grouped_cv_report

    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(25, 2))
    y = np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0.0, 0.05, size=25)
    # Include a small group that causes one fold with fewer rows to test ordering + normal path.
    groups = np.repeat(np.arange(5), 5)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    idx1: list[int] = []
    fold1: list[int] = []
    p1, s1, a1, g1, nf1 = _oof(X, y, groups, 5, bounds, index_out=idx1, fold_out=fold1, n_workers=1)

    idx4: list[int] = []
    fold4: list[int] = []
    p4, s4, a4, g4, nf4 = _oof(X, y, groups, 5, bounds, index_out=idx4, fold_out=fold4, n_workers=4)

    assert nf1 == nf4 == 5
    assert idx1 == idx4
    assert fold1 == fold4
    assert np.array_equal(a1, a4)
    assert np.array_equal(g1, g4)
    assert np.allclose(p1, p4, atol=1e-12, rtol=1e-12)
    assert np.allclose(s1, s4, atol=1e-12, rtol=1e-12)

    rep1 = grouped_cv_report(X, y, groups=groups, n_splits=5, bounds=bounds, n_boot=100, n_repeats=2, n_workers=1)
    rep4 = grouped_cv_report(X, y, groups=groups, n_splits=5, bounds=bounds, n_boot=100, n_repeats=2, n_workers=4)
    assert rep1["spearman"] == pytest.approx(rep4["spearman"], abs=1e-12)
    assert rep1["oof_index"] == rep4["oof_index"]
    assert np.allclose(rep1["oof_pred"], rep4["oof_pred"], atol=1e-12, rtol=1e-12)
    assert np.allclose(rep1["oof_std"], rep4["oof_std"], atol=1e-12, rtol=1e-12)


def test_operating_ranges_replicate_aware_conditioning_and_cache(tmp_path: Path) -> None:
    pytest.importorskip("botorch")
    from kalos.portal.operating_ranges import NotComputed, build_plan, fit_and_score

    rng = np.random.default_rng(12)
    recipes = rng.uniform(0.0, 10.0, size=(12, 2))
    rows = []
    for a, b in recipes:
        y_true = 5.0 + 2.0 * a - 0.05 * (b - 5.0) ** 2
        for _ in range(3):
            rows.append({
                "glucose_gL": round(float(a), 3),
                "temperature_C": round(float(b), 3),
                "titer": round(float(y_true + rng.normal(0.0, 0.3)), 3),
            })
    df = pd.DataFrame(rows)
    state = {"base_rows": df.to_dict(orient="records"), "generation": 1, "target": "titer"}
    analysis = {
        "target": "titer",
        "features": ["glucose_gL", "temperature_C"],
        "categorical_features": [],
        "feature_ranges": {
            "glucose_gL": [float(df["glucose_gL"].min()), float(df["glucose_gL"].max())],
            "temperature_C": [float(df["temperature_C"].min()), float(df["temperature_C"].max())],
        },
        "n": len(df),
        "best": round(float(df["titer"].max()), 4),
        "noise": {"replicate_aware": True, "noise_sd": 0.3},
        "reliability": {
            "spearman": 0.9, "ci95": [0.8, 0.96], "spearman_floor": 0.2,
            "clears_floor": True, "ci_excludes_zero": True,
        },
        "run": {"anonymize": False, "kind": "upload"},
    }
    plan = build_plan(
        state, analysis, threshold=15.0, probability=0.8,
        goal="maximize", goal_source="default",
        column_for={c: c for c in df.columns}, display_level=lambda _c, lbl: lbl,
    )
    assert plan.replicate_aware is True
    assert plan.assay_noise_var is not None and plan.assay_noise_var > 0

    cache_dir = tmp_path / "models"
    r1 = fit_and_score(plan, cache_dir=cache_dir)
    assert r1["status"] == "computed"
    assert r1["model"]["noise"] == "fixed_assay_noise_floor"
    assert r1["model"]["refit_rows"] == 12
    assert list(cache_dir.glob("*.json"))

    # Second call hits cache and gives identical result.
    r2 = fit_and_score(plan, cache_dir=cache_dir)
    assert r2 == r1

    # Conditioning refusal is wired in fit_and_score (check_conditioning=True).
    from kalos.core import surrogate as surrogate_mod

    old_refuse = surrogate_mod.COND_REFUSE
    try:
        surrogate_mod.COND_REFUSE = 1.0  # type: ignore[misc]
        with pytest.raises(NotComputed) as exc:
            fit_and_score(plan, cache_dir=None)
        assert exc.value.reason_code == "model_fit_failed"
    finally:
        surrogate_mod.COND_REFUSE = old_refuse  # type: ignore[misc]
