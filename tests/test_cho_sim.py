"""kalos.bench.cho_sim: the promoted CHO fed-batch simulator is deterministic,
has an interior optimum, is registered as a bench objective, and the BO loop
beats a space-filling design on it."""
from __future__ import annotations

import numpy as np
import pytest

from kalos.bench import OBJECTIVES, get_objective, run_one
from kalos.bench.cho_sim import (
    CHO_OPTIMUM_MG_L,
    DESIGN_SPACE,
    ProcessParams,
    endpoint_titer,
    replicate_titers,
    true_optimum,
)


def test_registered_and_deterministic():
    assert "cho_fed_batch_4d" in OBJECTIVES
    obj = get_objective("cho_fed_batch_4d")
    x = np.array([[1.0, 40.0, 0.5, 7.0]])
    assert obj.evaluate(x)[0][0] == obj.evaluate(x)[0][0]
    a = obj.evaluate(x, noise=0.1, rng=np.random.default_rng(3))[1]
    b = obj.evaluate(x, noise=0.1, rng=np.random.default_rng(3))[1]
    assert a == b
    with pytest.raises(KeyError):
        get_objective("nope")


def test_replicates_are_seeded_and_spread_around_the_truth():
    p = ProcessParams(1.0, 40, 0.5, 7.0)
    r1 = replicate_titers(p, 6, noise_cv=0.1, rng=np.random.default_rng(0))
    r2 = replicate_titers(p, 6, noise_cv=0.1, rng=np.random.default_rng(0))
    assert np.array_equal(r1, r2) and r1.std() > 0
    assert abs(np.log(r1).mean() - np.log(endpoint_titer(p))) < 0.15


def test_optimum_is_interior_and_matches_the_recorded_value():
    p, v = true_optimum()
    assert v == pytest.approx(CHO_OPTIMUM_MG_L, rel=5e-3)
    for k in ("glc_feed_rate", "glc_0", "ph_setpoint"):
        lo, hi = DESIGN_SPACE[k]
        span = hi - lo
        assert lo + 0.02 * span < getattr(p, k) < hi - 0.02 * span, k
    # nothing in a dense random sample beats the recorded optimum
    obj = get_objective("cho_fed_batch_4d")
    X = np.random.default_rng(1).uniform(obj.bounds[0], obj.bounds[1], size=(300, 4))
    assert obj.evaluate(X)[0].max() <= obj.optimum * 1.002


def test_bo_beats_lhs_on_the_cho_surface():
    obj = get_objective("cho_fed_batch_4d")
    bo = np.mean([obj.optimum - run_one(obj, "bo", budget=10, n_init=6, noise=0.05, seed=s)[-1]
                  for s in range(3)])
    lhs = np.mean([obj.optimum - run_one(obj, "lhs", budget=10, n_init=6, noise=0.05, seed=s)[-1]
                   for s in range(3)])
    assert bo < lhs
