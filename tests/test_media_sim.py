"""Media campaign generator (structure only, known optimum): deterministic per
seed, 100 runs over 37 recipes with replicates and a scale column, and the
engine recovers the optimum at the target scale."""
from __future__ import annotations

import numpy as np

from kalos.bench import media_sim as ms
from kalos.core.optimize import propose
from kalos.core.replicates import aggregate_replicates
from kalos.core.surrogate import Surrogate


def test_deterministic_shape_replicates_and_scale():
    a, b = ms.make_campaign(1), ms.make_campaign(1)
    assert a.equals(b) and not a.equals(ms.make_campaign(2))
    assert len(a) == ms.N_RUNS == 100 and a["recipe_id"].nunique() == 37
    assert set(a["scale_L"]) <= set(ms.SCALES) and a["scale_L"].nunique() > 1
    # replicates share the truth and differ only by noise
    rep = a.groupby("recipe_id")
    assert (rep["titer_true"].nunique() == 1).all()
    multi = rep.filter(lambda g: len(g) > 1)
    assert len(multi) > 0 and multi.groupby("recipe_id")["titer_g_L"].std().gt(0).all()
    X = a[list(ms.COMPONENT_COLUMNS)].to_numpy()
    agg_X, _, _, _ = aggregate_replicates(X, a["titer_g_L"].to_numpy())
    assert len(agg_X) == 37


def test_known_optimum_is_exact():
    x, v = ms.optimum(0)
    assert np.isclose(ms.true_titer(x[None], ms.TARGET_SCALE, 0)[0], v)
    rng = np.random.default_rng(0)
    X = ms.LOWER + rng.uniform(size=(5000, 8)) * (ms.UPPER - ms.LOWER)
    assert ms.true_titer(X, ms.TARGET_SCALE, 0).max() < v
    assert ms.peak(0.001) > ms.peak(ms.TARGET_SCALE)


def test_engine_recovers_the_optimum_at_target_scale():
    df = ms.make_campaign(0)
    X = np.column_stack([df[list(ms.COMPONENT_COLUMNS)].to_numpy(), np.log10(df["scale_L"])])
    y = df["titer_g_L"].to_numpy()
    ls = np.log10(ms.TARGET_SCALE)
    box = np.vstack([np.append(ms.LOWER, np.log10(min(ms.SCALES))), np.append(ms.UPPER, ls)])
    at_target = np.vstack([np.append(ms.LOWER, ls), np.append(ms.UPPER, ls)])
    _, opt = ms.optimum(0)
    data_best = float(df.loc[df["scale_L"] == ms.TARGET_SCALE, "titer_true"].max())
    rng = np.random.default_rng(0)
    best = 0.0
    for i in range(8):
        s = Surrogate().fit(X, y, bounds=box)
        x = propose(s, at_target, q=1, num_restarts=4, raw_samples=256, seed=i)[0]
        x[-1] = ls
        yt = float(ms.true_titer(x[None, :-1], ms.TARGET_SCALE, 0)[0])
        X = np.vstack([X, x])
        y = np.append(y, yt * np.exp(rng.normal(0.0, ms.NOISE_CV)))
        best = max(best, yt)
    assert best > data_best and best >= 0.9 * opt
