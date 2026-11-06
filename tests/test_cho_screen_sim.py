"""CHO screen generator (structure only, known optimum): deterministic per
seed, 109 x 49, and the engine recovers the optimum from the screen."""
from __future__ import annotations

import numpy as np

from kalos.bench import get_objective
from kalos.bench.cho_screen_sim import (
    FEATURE_COLUMNS,
    N_FEATURES,
    N_ROWS,
    cho_screen_objective,
    make_screen,
    optimum_point,
)
from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate


def test_deterministic_shape_and_exact_optimum():
    a, b = make_screen(2), make_screen(2)
    assert a.equals(b) and not a.equals(make_screen(3))
    assert a.shape == (N_ROWS, 1 + N_FEATURES + 2) and (N_ROWS, N_FEATURES) == (109, 49)
    obj = cho_screen_objective(2)
    assert obj.evaluate(optimum_point(2)[None])[0][0] == obj.optimum == 7.5
    X = np.random.default_rng(0).uniform(size=(5000, N_FEATURES))
    assert obj.evaluate(X)[0].max() < obj.optimum
    assert get_objective("cho_screen_49d").dim == 49


def test_engine_recovers_the_optimum_from_the_screen():
    df = make_screen(0)
    obj = cho_screen_objective(0)
    X = df[list(FEATURE_COLUMNS)].to_numpy()
    y = df["titer_g_L"].to_numpy()
    rng = np.random.default_rng(0)
    best = float(df["titer_true"].max())
    screen_best = best
    for i in range(4):
        s = Surrogate().fit(X, y, bounds=obj.bounds)
        x = propose(s, obj.bounds, q=1, num_restarts=4, raw_samples=256, seed=i)[0]
        yt = float(obj.evaluate(x[None])[0][0])
        X = np.vstack([X, x])
        y = np.append(y, yt * np.exp(rng.normal(0.0, 0.05)))
        best = max(best, yt)
    assert best > screen_best and best >= 0.95 * obj.optimum
