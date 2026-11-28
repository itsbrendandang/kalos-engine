"""Tests for torch-free small-N space-filling DoE and config flags."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from kalos.core.space_filling import (
    ProposalBatch,
    proposal_separation,
    propose_space_filling,
)
from kalos.envconfig import ConfigError
from kalos.portal.config import (
    check_env_settings,
    cv_workers_setting,
    space_filling_small_n_setting,
)


def test_space_filling_module_is_torch_free_at_import_and_call():
    code = (
        "import sys, numpy as np; "
        "from kalos.core.space_filling import propose_space_filling; "
        "loaded_at_import = 'torch' in sys.modules; "
        "bounds = np.array([[0.0, 0.0], [1.0, 1.0]]); "
        "X_tr = np.array([[0.1, 0.1], [0.9, 0.9]]); "
        "b = propose_space_filling(bounds, 3, X_train=X_tr, seed=0); "
        "loaded_at_call = 'torch' in sys.modules; "
        "print(loaded_at_import, loaded_at_call, b.returned)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False False 3"


def test_space_filling_respects_bounds_and_determinism():
    bounds = np.array([[1.0, -2.0, 10.0], [5.0, 4.0, 20.0]])
    X_train = np.array([[2.0, 0.0, 12.0], [4.0, 2.0, 18.0]])

    b1 = propose_space_filling(bounds, 5, X_train=X_train, seed=7)
    b2 = propose_space_filling(bounds, 5, X_train=X_train, seed=7)
    b3 = propose_space_filling(bounds, 5, X_train=X_train, seed=8)

    assert isinstance(b1, ProposalBatch)
    assert b1.returned == 5
    assert b1.notice is None
    assert np.array_equal(b1.X, b2.X)
    assert not np.array_equal(b1.X, b3.X)
    assert np.all(bounds[0] - 1e-12 <= b1.X)
    assert np.all(bounds[1] + 1e-12 >= b1.X)


def test_space_filling_linear_constraints_and_infeasible_raises():
    bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    X_train = np.array([[1.0, 1.0, 5.0]])
    # x0 + x1 <= 4.0  <=>  -x0 - x1 >= -4.0, and x0 >= 1.0
    constraints = [
        ([0, 1], [-1.0, -1.0], -4.0),
        ([0], [1.0], 1.0),
    ]
    batch = propose_space_filling(
        bounds,
        5,
        X_train=X_train,
        inequality_constraints=constraints,
        seed=0,
    )
    assert batch.returned == 5
    assert np.all(batch.X[:, 0] + batch.X[:, 1] <= 4.0 + 1e-6)
    assert np.all(batch.X[:, 0] >= 1.0 - 1e-6)

    with pytest.raises(ValueError, match="infeasible"):
        propose_space_filling(
            bounds,
            3,
            X_train=X_train,
            inequality_constraints=[([0], [1.0], 12.0)],
            seed=0,
        )


def test_space_filling_fixed_features_and_pending_avoidance():
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    X_train = np.array([[0.1, 0.1, 1.0]])
    pending = np.array([[0.9, 0.9, 1.0], [0.1, 0.9, 1.0]])
    batch = propose_space_filling(
        bounds,
        4,
        X_train=X_train,
        cat_dims=[2],
        cat_cardinalities=[3],
        fixed_features={2: 1.0, 0: 0.5},
        pending=pending,
        seed=0,
        min_separation=0.05,
    )
    assert batch.returned == 4
    assert np.allclose(batch.X[:, 0], 0.5)
    assert np.allclose(batch.X[:, 2], 1.0)
    for row in batch.X:
        for ref in np.vstack([X_train, pending]):
            assert proposal_separation(row, ref, bounds[0], bounds[1], [2]) >= 0.05


def test_space_filling_categorical_balance_and_short_batch_notice():
    # Mixed space: 1 continuous, 1 categorical with 3 levels.
    bounds = np.array([[0.0, 0.0], [1.0, 2.0]])
    # Empty X_train, q=6 -> balanced across all 3 levels (2 per level).
    balanced = propose_space_filling(
        bounds,
        6,
        X_train=np.empty((0, 2)),
        cat_dims=[1],
        cat_cardinalities=[3],
        seed=0,
    )
    assert balanced.returned == 6
    counts = {lvl: int((np.rint(balanced.X[:, 1]) == lvl).sum()) for lvl in range(3)}
    assert counts == {0: 2, 1: 2, 2: 2}

    # X_train already has 2 points in level 0 -> maximin prioritizes unvisited levels 1 & 2.
    X_train = np.array([[0.2, 0.0], [0.8, 0.0]])
    batch = propose_space_filling(
        bounds,
        4,
        X_train=X_train,
        cat_dims=[1],
        cat_cardinalities=[3],
        seed=0,
    )
    assert batch.returned == 4
    assert {int(round(v)) for v in batch.X[:, 1]} == {1, 2}

    # Purely categorical 3-level space with 1 existing point -> only 2 distinct remaining.
    cat_bounds = np.array([[0.0], [2.0]])
    short = propose_space_filling(
        cat_bounds,
        5,
        X_train=np.array([[0.0]]),
        cat_dims=[0],
        cat_cardinalities=[3],
        seed=0,
    )
    assert short.returned == 2
    assert short.notice is not None
    assert short.notice["code"] == "fewer_distinct_candidates"
    assert short.notice["requested"] == 5
    assert short.notice["returned"] == 2
    assert set(short.X[:, 0].astype(int)) == {1, 2}


def test_space_filling_and_cv_workers_config_flags(monkeypatch):
    monkeypatch.delenv("KALOS_SPACE_FILLING_SMALL_N", raising=False)
    monkeypatch.delenv("KALOS_CV_WORKERS", raising=False)
    assert space_filling_small_n_setting() is False
    assert cv_workers_setting() == 1
    check_env_settings()

    monkeypatch.setenv("KALOS_SPACE_FILLING_SMALL_N", "1")
    monkeypatch.setenv("KALOS_CV_WORKERS", "4")
    assert space_filling_small_n_setting() is True
    assert cv_workers_setting() == 4
    check_env_settings()

    monkeypatch.setenv("KALOS_SPACE_FILLING_SMALL_N", "maybe")
    with pytest.raises(ConfigError, match="KALOS_SPACE_FILLING_SMALL_N"):
        check_env_settings()

    monkeypatch.setenv("KALOS_SPACE_FILLING_SMALL_N", "0")
    monkeypatch.setenv("KALOS_CV_WORKERS", "9")
    with pytest.raises(ConfigError, match="KALOS_CV_WORKERS"):
        check_env_settings()


def test_analyze_small_n_space_filling_flag_on_and_off(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    import kalos.core.optimize as optimize_mod
    import kalos.core.space_filling as sf_mod
    from kalos.portal.analysis import _analyze

    # 6 distinct recipes, d=4 continuous features -> 6 < 2*4=8, verdict is not reliable.
    rng = np.random.default_rng(11)
    n = 6
    df = pd.DataFrame({
        "f1": rng.uniform(0, 1, n).round(3),
        "f2": rng.uniform(0, 1, n).round(3),
        "f3": rng.uniform(0, 1, n).round(3),
        "f4": rng.uniform(0, 1, n).round(3),
        "titer": np.linspace(1.0, 2.0, n) + rng.normal(0, 0.05, n),
    })

    calls_bo = 0
    calls_sf = 0
    orig_bo = optimize_mod.propose_batch
    orig_sf = sf_mod.propose_space_filling

    def spy_bo(*args, **kwargs):
        nonlocal calls_bo
        calls_bo += 1
        return orig_bo(*args, **kwargs)

    def spy_sf(*args, **kwargs):
        nonlocal calls_sf
        calls_sf += 1
        return orig_sf(*args, **kwargs)

    monkeypatch.setattr(optimize_mod, "propose_batch", spy_bo)
    monkeypatch.setattr(sf_mod, "propose_space_filling", spy_sf)

    # Flag OFF (default): uses BoTorch qLogNEI
    monkeypatch.delenv("KALOS_SPACE_FILLING_SMALL_N", raising=False)
    out_off = _analyze(df, "titer")
    assert calls_bo == 1
    assert calls_sf == 0
    assert len(out_off["proposals"]) == 5

    # Flag ON: uses space-filling DoE instead of qLogNEI, still scores posterior & intervals
    calls_bo = calls_sf = 0
    monkeypatch.setenv("KALOS_SPACE_FILLING_SMALL_N", "1")
    out_on = _analyze(df, "titer")
    assert out_on["verdict"]["state"] != "reliable"
    assert calls_sf == 1
    assert calls_bo == 0
    assert len(out_on["proposals"]) == 5
    for p in out_on["proposals"]:
        assert p["pred"] is not None
        assert p["std"] is not None
        assert p["predictive_std"] is not None
        assert p["intervals"] is not None
