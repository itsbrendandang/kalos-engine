"""TuRBO-1 trust region (opt-in, d >= 8): state update rules from the BoTorch
tutorial, the region is centred on the incumbent and inside the box, proposals
land inside it, and it is ignored below d=8."""
from __future__ import annotations

import numpy as np

from kalos.bench.turbo_bench import run_turbo_one
from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate
from kalos.core.turbo import (
    TURBO_MIN_DIM,
    TurboState,
    trust_region_bounds,
    update_state,
)


def test_state_expands_shrinks_and_restarts():
    s = TurboState(dim=10, best_value=0.0)
    assert s.failure_tolerance == 10
    for i in range(s.success_tolerance):
        update_state(s, [float(i + 1)])
    assert s.length == 1.6
    for _ in range(s.failure_tolerance):
        update_state(s, [-1.0])
    assert s.length == 0.8
    while not s.restart_triggered:
        update_state(s, [-1.0])
    assert s.length < s.length_min


def _fit(d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(20, d))
    y = -((X - 0.3) ** 2).sum(axis=1)
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    return Surrogate().fit(X, y, bounds=bounds), bounds, X, y


def test_region_is_centred_on_the_incumbent_and_proposals_stay_inside():
    s, bounds, X, y = _fit(TURBO_MIN_DIM)
    st = TurboState(dim=TURBO_MIN_DIM, length=0.4)
    tr = trust_region_bounds(s, bounds, st)
    best = X[int(np.argmax(y))]
    assert np.all(tr[0] <= best + 1e-12) and np.all(best <= tr[1] + 1e-12)
    assert np.all(tr[0] >= 0) and np.all(tr[1] <= 1)
    assert np.all(tr[1] - tr[0] < 1.0)
    batch = propose(s, bounds, q=2, num_restarts=2, raw_samples=64, seed=0, trust_region=st)
    assert np.all(batch >= tr[0] - 1e-9) and np.all(batch <= tr[1] + 1e-9)


def test_region_is_ignored_below_min_dim():
    s, bounds, _, _ = _fit(3)
    st = TurboState(dim=3, length=0.01)
    a = propose(s, bounds, q=1, num_restarts=2, raw_samples=64, seed=1, trust_region=st)
    b = propose(s, bounds, q=1, num_restarts=2, raw_samples=64, seed=1)
    assert np.allclose(a, b)


def test_closed_loop_runs():
    r = run_turbo_one("sparse_bump_12d", "bo_turbo", seed=0, n_init=8, budget=4)
    assert 0.0 <= r <= 1.0
