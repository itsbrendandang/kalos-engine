"""Tests for upgraded MultiObjectiveSurrogate and propose_multiobjective."""
from __future__ import annotations

import numpy as np
import pytest

from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective


def test_mixed_maximize_and_minimize_objectives_pareto_posterior_refpoint():
    rng = np.random.default_rng(10)
    X = rng.uniform(0.0, 1.0, (16, 2))
    # Objective 0 (maximize): peaks near x0=0.8
    # Objective 1 (minimize): lowest near x0=0.2 (tradeoff along x0)
    y0 = 10.0 - 5.0 * (X[:, 0] - 0.8) ** 2 - (X[:, 1] - 0.5) ** 2
    y1 = 2.0 + 4.0 * (X[:, 0] - 0.2) ** 2 + (X[:, 1] - 0.5) ** 2
    Y = np.column_stack([y0, y1])
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)

    s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds, maximize=[True, False])

    px, py = s.pareto()
    assert len(px) >= 2 and py.shape[1] == 2
    # Every Pareto point must be non-dominated in (max y0, min y1) original units
    for i in range(len(Y)):
        dominated_by_other = any(
            (py[j, 0] >= Y[i, 0] and py[j, 1] <= Y[i, 1])
            and (py[j, 0] > Y[i, 0] or py[j, 1] < Y[i, 1])
            for j in range(len(py))
        )
        in_front = np.any(np.all(np.isclose(py, Y[i]), axis=1))
        assert in_front or dominated_by_other

    rp = s.default_ref_point()
    assert rp.shape == (2,)
    assert rp[0] < Y[:, 0].min()  # maximized objective: ref below min
    assert rp[1] > Y[:, 1].max()  # minimized objective: ref above max

    mean, std = s.posterior(X)
    assert mean.shape == (16, 2) and std.shape == (16, 2)
    assert np.corrcoef(mean[:, 0], y0)[0, 1] > 0.9
    assert np.corrcoef(mean[:, 1], y1)[0, 1] > 0.9

    _, std_obs = s.posterior(X, observation_noise=True)
    assert np.all(std_obs >= std - 1e-12)

    nxt = propose_multiobjective(
        s, bounds, ref_point=np.array([0.0, 10.0]), q=2, num_restarts=3, raw_samples=32, mc_samples=32, seed=42
    )
    assert nxt.shape == (2, 2)
    assert np.all(nxt >= 0.0) and np.all(nxt <= 1.0)


def test_fixed_noise_variants_and_validation():
    rng = np.random.default_rng(11)
    n, d, m = 14, 2, 2
    X = rng.uniform(0.0, 1.0, (n, d))
    Y = np.column_stack([np.sin(3 * X[:, 0]), np.cos(3 * X[:, 1])])
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)

    for noise in (0.01, np.full(n, 0.02), np.column_stack([np.full(n, 0.01), np.full(n, 0.05)])):
        s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds, noise=noise)
        mean, std = s.posterior(X, observation_noise=True)
        assert mean.shape == (n, m) and np.all(std > 0)
        assert all(hp["noise_fitted"] is None for hp in s.fit_diagnostics["per_objective_diagnostics"])

    with pytest.raises(ValueError, match="non-negative"):
        MultiObjectiveSurrogate().fit(X, Y, bounds=bounds, noise=-0.1)
    with pytest.raises(ValueError, match="finite"):
        MultiObjectiveSurrogate().fit(X, Y, bounds=bounds, noise=np.nan)
    with pytest.raises(ValueError):
        MultiObjectiveSurrogate().fit(X, Y, bounds=bounds, maximize=[True])


def test_mixed_categorical_fit_and_propose_with_constraints_fixed_pending():
    rng = np.random.default_rng(12)
    n = 18
    X = np.column_stack([
        rng.uniform(0.0, 1.0, n),
        rng.uniform(0.0, 1.0, n),
        rng.integers(0, 3, n),
    ])
    Y = np.column_stack([
        X[:, 0] + 0.4 * (X[:, 2] == 2),
        1.0 - X[:, 1] + 0.3 * (X[:, 2] == 0),
    ])
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], float)

    s = MultiObjectiveSurrogate().fit(
        X, Y, bounds=bounds, cat_dims=[2], restarts=2, restart_seed=5, check_conditioning=True
    )
    assert s.fit_diagnostics["restarts"] == 2
    assert "condition_number" in s.fit_diagnostics
    assert len(s.fit_diagnostics["per_objective_condition_numbers"]) == 2

    # x0 + x1 <= 1.2 => -x0 - x1 >= -1.2; hold dim 1 fixed at 0.4; pass pending points
    pending = np.array([[0.2, 0.4, 1.0]])
    nxt = propose_multiobjective(
        s,
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=32,
        mc_samples=32,
        seed=7,
        pending=pending,
        inequality_constraints=[([0, 1], [-1.0, -1.0], -1.2)],
        fixed_features={1: 0.4},
        cat_dims=[2],
        cat_cardinalities=[3],
    )
    assert nxt.shape == (2, 3)
    assert np.allclose(nxt[:, 1], 0.4)
    assert np.all(nxt[:, 0] + nxt[:, 1] <= 1.2 + 1e-6)
    assert np.all(np.isin(nxt[:, 2], [0.0, 1.0, 2.0]))


def test_mixed_categorical_alternating_branch():
    """Force > MAX_MIXED_COMBOS (64) to exercise optimize_acqf_mixed_alternating."""
    rng = np.random.default_rng(13)
    n = 16
    # 9 * 9 = 81 > 64 combinations
    X = np.column_stack([
        rng.uniform(0.0, 1.0, n),
        rng.integers(0, 9, n),
        rng.integers(0, 9, n),
    ])
    Y = np.column_stack([X[:, 0] + 0.1 * X[:, 1], 1.0 - X[:, 0] + 0.1 * X[:, 2]])
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 8.0, 8.0]], float)

    s = MultiObjectiveSurrogate().fit(X, Y, bounds=bounds, cat_dims=[1, 2])
    nxt = propose_multiobjective(
        s,
        bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        mc_samples=16,
        seed=9,
        cat_dims=[1, 2],
        cat_cardinalities=[9, 9],
    )
    assert nxt.shape == (2, 3)
    assert np.all(np.isin(nxt[:, 1], np.arange(9, dtype=float)))
    assert np.all(np.isin(nxt[:, 2], np.arange(9, dtype=float)))
