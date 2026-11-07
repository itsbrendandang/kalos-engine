"""Clone-funnel generator (structure only, known latent): deterministic per
seed, funnel-shaped, the strain-grouped split catches the strain-fingerprint
leakage, and pool BO recovers a top clone."""
from __future__ import annotations

import numpy as np

from kalos.bench.clone_funnel_sim import (
    FEATURE_COLUMNS,
    N_CLONES,
    N_FEATURES,
    make_funnel,
)
from kalos.bench.pool import run_pool_one


def test_deterministic_shape_and_gates():
    a, b = make_funnel(3), make_funnel(3)
    assert a.frame.equals(b.frame) and np.array_equal(a.productivity, b.productivity)
    assert not make_funnel(4).frame.equals(a.frame)
    df = a.frame
    assert len(df) == N_CLONES and len(FEATURE_COLUMNS) == N_FEATURES == 49
    assert df["strain"].nunique() > 1
    # multi-stage gates: each stage is a subset of the previous one
    assert df["stage1_advanced"].sum() == N_CLONES // 2
    assert df["stage2_advanced"].sum() == N_CLONES // 4
    assert not (df["stage2_advanced"] & ~df["stage1_advanced"]).any()
    assert df.loc[~df["stage1_advanced"], "stage2_readout"].isna().all()
    assert df.loc[df["stage2_advanced"], "stage3_readout"].notna().all()


def test_grouped_split_catches_strain_leakage():
    from kalos.core.evaluation import grouped_cv_report

    f = make_funnel(0, n_clones=200)
    X, y = f.features(), f.frame["stage1_readout"].to_numpy()
    grouped = grouped_cv_report(X, y, groups=f.strain, n_boot=50)["spearman"]
    shuffled = grouped_cv_report(X, y, groups=np.arange(len(y)), n_boot=50)["spearman"]
    assert shuffled > grouped + 0.2  # a random split rewards memorising strains


def test_pool_bo_recovers_a_top_clone():
    f = make_funnel(0)
    X, y = f.features(), f.productivity
    q99 = float(np.quantile(y, 0.99))
    bo = [run_pool_one(X, y, "bo", n_init=10, budget=30, seed=s)[-1] for s in range(2)]
    rnd = [run_pool_one(X, y, "random", n_init=10, budget=30, seed=s)[-1] for s in range(2)]
    assert min(bo) >= q99 and np.mean(bo) > np.mean(rnd)
