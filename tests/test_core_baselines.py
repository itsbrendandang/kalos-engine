"""kalos.core.baselines: carry-forward rank agreement and random-split inflation.

All data here is synthetic (seeded generators below); nothing comes from a lab.
"""
from __future__ import annotations

import numpy as np
import pytest

from kalos.core.baselines import (
    MIN_SHARED_UNITS,
    carry_forward_rank_agreement,
    random_split_inflation,
)

N_BOOT = 300  # enough for stable tests, small enough to stay fast


def _units(n_units: int, seed: int = 0, noise_earlier: float = 0.5, noise_later: float = 0.2):
    """Synthetic scale-up: latent unit quality q; earlier-scale = q + big noise,
    target-scale = q + small noise. A model that sees q (as a process feature
    would) beats carry-forward; the earlier-scale value itself does not."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=n_units)
    earlier = q + rng.normal(scale=noise_earlier, size=n_units)
    later = q + rng.normal(scale=noise_later, size=n_units)
    ids = np.array([f"u{i:03d}" for i in range(n_units)])
    return q, earlier, later, ids


# ----------------------------------------------------------------- carry-forward: refusals


def test_refuses_under_five_shared_units():
    _, e, lt, ids = _units(4)
    r = carry_forward_rank_agreement(e, lt, ids, n_boot=N_BOOT)
    assert MIN_SHARED_UNITS == 5
    assert r["evaluable"] is False
    assert r["code"] == "too_few_shared_units"
    assert r["spearman"] is None and r["spearman_ci95"] is None
    assert r["reason"] and r["action"]


def test_five_shared_units_is_evaluable():
    _, e, lt, ids = _units(5)
    r = carry_forward_rank_agreement(e, lt, ids, n_boot=N_BOOT)
    assert r["evaluable"] is True
    assert r["n_units"] == 5 and r["k"] == 2
    assert -1.0 <= r["spearman"] <= 1.0


def test_units_missing_a_scale_are_not_shared_and_are_counted():
    _, e, lt, ids = _units(8)
    e = e.copy()
    lt = lt.copy()
    e[:2] = np.nan  # measured only at the target scale
    lt[2:4] = np.nan  # measured only at the earlier scale
    r = carry_forward_rank_agreement(e, lt, ids, n_boot=N_BOOT)
    assert r["n_units"] == 4
    assert r["n_units_later_only"] == 2 and r["n_units_earlier_only"] == 2
    assert r["code"] == "too_few_shared_units"


def test_constant_earlier_scale_is_refused_not_scored():
    _, _, lt, ids = _units(10)
    r = carry_forward_rank_agreement(np.ones(10), lt, ids, n_boot=N_BOOT)
    assert r["evaluable"] is False and r["code"] == "constant_values"


def test_unit_in_two_clusters_is_refused():
    _, e, lt, ids = _units(6)
    ids = np.concatenate([ids, ids[:1]])
    e = np.concatenate([e, e[:1]])
    lt = np.concatenate([lt, lt[:1]])
    clusters = ["c1"] * 6 + ["c2"]
    r = carry_forward_rank_agreement(e, lt, ids, clusters=clusters, n_boot=N_BOOT)
    assert r["evaluable"] is False and r["code"] == "unit_in_several_clusters"


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        carry_forward_rank_agreement([1.0, 2.0], [1.0], ["a", "b"])


# ----------------------------------------------------------------- carry-forward: values


def test_perfect_agreement_and_ci_bounds():
    x = np.arange(12, dtype=float)
    ids = [f"u{i}" for i in range(12)]
    r = carry_forward_rank_agreement(x, 2 * x + 1, ids, n_boot=N_BOOT)
    assert r["spearman"] == 1.0
    assert r["top_k_overlap"] == 1.0
    assert r["spearman_ci95"] == [1.0, 1.0]
    assert r["top_k_chance"] == pytest.approx(r["k"] / 12, abs=1e-4)


def test_independent_scales_ci_includes_zero():
    rng = np.random.default_rng(3)
    e, lt = rng.normal(size=30), rng.normal(size=30)
    r = carry_forward_rank_agreement(e, lt, [f"u{i}" for i in range(30)], n_boot=N_BOOT)
    lo, hi = r["spearman_ci95"]
    assert lo < 0.0 < hi


def test_replicate_rows_are_averaged_per_unit():
    _, e, lt, ids = _units(10, seed=1)
    base = carry_forward_rank_agreement(e, lt, ids, n_boot=N_BOOT)
    # Triplicate every unit at the target scale with symmetric noise: the
    # per-unit means and the ranking are unchanged.
    eps = np.array([-0.01, 0.0, 0.01])
    rep = carry_forward_rank_agreement(
        np.repeat(e, 3), np.repeat(lt, 3) + np.tile(eps, 10), np.repeat(ids, 3), n_boot=N_BOOT
    )
    assert rep["n_units"] == 10 and rep["n_rows"] == 30
    assert rep["spearman"] == base["spearman"]
    assert rep["top_k_overlap"] == base["top_k_overlap"]


def test_minimize_flips_top_k_not_spearman():
    x = np.arange(10, dtype=float)
    later = x.copy()
    later[-1], later[0] = later[0], later[-1]  # swap the extremes
    ids = [f"u{i}" for i in range(10)]
    up = carry_forward_rank_agreement(x, later, ids, k=1, maximize=True, n_boot=N_BOOT)
    down = carry_forward_rank_agreement(x, later, ids, k=1, maximize=False, n_boot=N_BOOT)
    assert up["spearman"] == down["spearman"]
    assert up["top_k_overlap"] == 0.0 and down["top_k_overlap"] == 0.0
    later2 = x.copy()
    later2[-1] = 100.0  # same top unit, lowest unit unchanged
    assert carry_forward_rank_agreement(x, later2, ids, k=1, maximize=False, n_boot=N_BOOT)["top_k_overlap"] == 1.0


def test_deterministic_for_a_seed():
    _, e, lt, ids = _units(15, seed=2)
    q = _units(15, seed=2)[0]
    a = carry_forward_rank_agreement(e, lt, ids, model_pred=q, n_boot=N_BOOT, seed=7)
    b = carry_forward_rank_agreement(e, lt, ids, model_pred=q, n_boot=N_BOOT, seed=7)
    assert a == b


def test_few_clusters_gives_point_but_no_ci_and_no_verdict():
    q, e, lt, ids = _units(12, seed=4)
    clusters = ["A"] * 6 + ["B"] * 6
    r = carry_forward_rank_agreement(e, lt, ids, clusters=clusters, model_pred=q, n_boot=N_BOOT)
    assert r["evaluable"] is True and r["spearman"] is not None
    assert r["spearman_ci95"] is None and "2 cluster" in r["ci_reason"]
    assert r["model"]["beats_carry_forward"] is None
    assert r["model"]["verdict"].startswith("not judged")


# ----------------------------------------------------------------- carry-forward vs a model


def test_model_that_copies_the_earlier_scale_does_not_beat_it():
    """Planted fault `earlier_scale_titer_as_feature`: a model that only re-ranks
    by the earlier-scale value has no skill of its own and must not read as one."""
    _, e, lt, ids = _units(40, seed=5)
    model = 0.9 * e + 0.1  # monotone in the earlier-scale value
    r = carry_forward_rank_agreement(e, lt, ids, model_pred=model, n_boot=N_BOOT)
    m = r["model"]
    assert m["evaluable"] is True
    assert m["spearman_difference"] == 0.0
    assert m["beats_carry_forward"] is False
    assert "does not beat" in m["verdict"]


def test_model_with_real_signal_beats_carry_forward():
    q, e, lt, ids = _units(40, seed=6, noise_earlier=1.0, noise_later=0.1)
    r = carry_forward_rank_agreement(e, lt, ids, model_pred=q, n_boot=N_BOOT)
    m = r["model"]
    assert m["spearman"] > m["carry_forward_spearman"]
    assert m["spearman_difference_ci95"][0] > 0.0
    assert m["beats_carry_forward"] is True
    assert "better than" in m["verdict"]


def test_model_missing_predictions_is_not_judged():
    q, e, lt, ids = _units(10, seed=8)
    q = q.copy()
    q[:6] = np.nan
    r = carry_forward_rank_agreement(e, lt, ids, model_pred=q, n_boot=N_BOOT)
    assert r["evaluable"] is True
    assert r["model"]["evaluable"] is False and r["model"]["code"] == "too_few_shared_units"
    assert r["model"]["beats_carry_forward"] is None


# ----------------------------------------------------------------- random-split inflation


def _nn_fit_predict(X_tr, y_tr, X_te):
    """1-nearest-neighbour: memorises replicates, so leakage shows up clearly."""
    d = ((X_te[:, None, :] - X_tr[None, :, :]) ** 2).sum(axis=2)
    return y_tr[np.argmin(d, axis=1)]


def _grouped_data(n_groups=20, reps=3, seed=0):
    """Each group (recipe) has its own random level and its replicates share a
    near-identical feature vector; the feature carries no signal across groups."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_groups, 2))
    level = rng.normal(size=n_groups)
    X = np.repeat(centers, reps, axis=0) + rng.normal(scale=1e-3, size=(n_groups * reps, 2))
    y = np.repeat(level, reps) + rng.normal(scale=0.05, size=n_groups * reps)
    g = np.repeat([f"r{i}" for i in range(n_groups)], reps)
    return X, y, g


def test_inflation_shows_the_gap_on_replicated_groups():
    X, y, g = _grouped_data()
    r = random_split_inflation(X, y, g, _nn_fit_predict, n_boot=N_BOOT)
    assert r["role"] == "diagnostic" and r["headline"] is False
    assert r["evaluable"] is True
    assert r["random_spearman"] > 0.9
    assert abs(r["grouped_spearman"]) < 0.5
    assert r["inflation"] > 0.4
    assert r["inflation_ci95"][0] > 0.0
    assert r["leaked_test_fraction"] > 0.9


def test_no_replicated_groups_is_refused():
    X, y, g = _grouped_data(n_groups=12, reps=1)
    r = random_split_inflation(X, y, g, _nn_fit_predict, n_boot=N_BOOT)
    assert r["evaluable"] is False and r["code"] == "no_replicated_groups"
    assert r["inflation"] is None


def test_too_few_groups_is_refused():
    X, y, g = _grouped_data(n_groups=2, reps=4)
    r = random_split_inflation(X, y, g, _nn_fit_predict, n_boot=N_BOOT)
    assert r["evaluable"] is False and r["code"] == "too_few_groups"


def test_nonfinite_targets_dropped_and_counted():
    X, y, g = _grouped_data(n_groups=10, reps=3, seed=1)
    y = y.copy()
    y[[0, 5]] = np.nan
    r = random_split_inflation(X, y, g, _nn_fit_predict, n_boot=N_BOOT)
    assert r["n_rows_dropped"] == 2 and r["n_rows"] == 28
    assert r["evaluable"] is True


def test_inflation_deterministic_for_a_seed():
    X, y, g = _grouped_data(n_groups=10, reps=2, seed=2)
    a = random_split_inflation(X, y, g, _nn_fit_predict, n_boot=N_BOOT, seed=3)
    b = random_split_inflation(X, y, g, _nn_fit_predict, n_boot=N_BOOT, seed=3)
    assert a == b


def test_fit_predict_wrong_length_raises():
    X, y, g = _grouped_data(n_groups=6, reps=2)
    with pytest.raises(ValueError):
        random_split_inflation(X, y, g, lambda a, b, c: np.zeros(1), n_boot=N_BOOT)


def test_module_is_pure_layering():
    """core/baselines.py imports only stdlib, numpy/pandas/scipy/sklearn and kalos.core."""
    import ast
    from pathlib import Path

    import kalos.core.baselines as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert not node.module.startswith("kalos.") or node.module.startswith(
                ("kalos.core", "kalos.util", "kalos.data", "kalos.config")
            ), node.module
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            assert node.module in {"splits", "gates"}, node.module
