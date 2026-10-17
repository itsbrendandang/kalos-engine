"""A proposed batch never contains duplicate or near-duplicate recipes.

Regression for a QA finding: on a monotonic sheet (optimum at the box edge) a
q=5 batch returned two IDENTICAL recipes - the joint qLogNEI optimization
collapsed two points onto the same box corner. `propose_batch` now repairs a
collapsed batch by sequential, pending-aware selection from a candidate pool
(no random jitter), and returns a SHORT batch with a notice when the space has
fewer than q distinct candidates. Synthetic data only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")

import torch  # noqa: E402

import kalos.core.optimize as optimize  # noqa: E402
from kalos.core.optimize import (  # noqa: E402
    NEAR_DUPLICATE_TOL,
    _is_distinct,
    propose,
    propose_batch,
    proposal_separation,
)
from kalos.core.surrogate import Surrogate, sanitize_bounds  # noqa: E402


def _pairwise_min(X, bounds, cat_dims=None) -> float:
    lo, hi = sanitize_bounds(bounds)
    seps = [
        proposal_separation(X[i], X[j], lo, hi, cat_dims)
        for i in range(len(X)) for j in range(i + 1, len(X))
    ]
    return min(seps) if seps else float("inf")


def _monotonic(n=30, seed=0):
    """Titer rises in both factors: the optimum is the upper box corner."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0, 3, n).round(2), rng.uniform(5, 7, n).round(2)])
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(0, 0.05, n)
    return X, y, np.vstack([X.min(0), X.max(0)])


def _fit(X, y, bounds, **kw):
    torch.manual_seed(1234)
    np.random.seed(1234)
    return Surrogate().fit(X, y, bounds=bounds, **kw)


@pytest.fixture
def collapse_joint_batch(monkeypatch):
    """Make the JOINT optimizer return a batch whose rows all collapsed onto its
    first row - the failure mode under test - while leaving every q=1 call
    (the sequential repair) untouched."""
    real = optimize.optimize_acqf

    def collapsed(*args, **kwargs):
        cands, vals = real(*args, **kwargs)
        if kwargs.get("q", 1) > 1 and kwargs.get("return_best_only", True):
            cands = cands[:1].expand(cands.shape[0], -1).clone()
        return cands, vals

    monkeypatch.setattr(optimize, "optimize_acqf", collapsed)


# --- the distance rule ---------------------------------------------------------- #


def test_separation_is_range_normalised_chebyshev():
    lo, hi = np.array([0.0, 5.0]), np.array([10.0, 7.0])
    a = np.array([5.0, 6.0])
    # 1.0 on a 10-wide factor = 0.1; 0.1 on a 2-wide factor = 0.05 -> max = 0.1
    assert proposal_separation(a, a + [1.0, 0.1], lo, hi) == pytest.approx(0.1)
    assert proposal_separation(a, a, lo, hi) == 0.0


def test_categorical_levels_compare_exactly():
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 3.0])
    a = np.array([0.5, 1.0])
    assert proposal_separation(a, np.array([0.5, 2.0]), lo, hi, cat_dims=[1]) == 1.0
    assert proposal_separation(a, np.array([0.5, 1.0]), lo, hi, cat_dims=[1]) == 0.0


def test_tolerance_boundary():
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    a = np.array([0.5, 0.5])
    at_tol = a + [NEAR_DUPLICATE_TOL, 0.0]
    just_below = a + [NEAR_DUPLICATE_TOL * (1 - 1e-6), NEAR_DUPLICATE_TOL * (1 - 1e-6)]
    assert _is_distinct(at_tol, [a], lo, hi, None, NEAR_DUPLICATE_TOL)
    assert not _is_distinct(just_below, [a], lo, hi, None, NEAR_DUPLICATE_TOL)


# --- propose_batch ---------------------------------------------------------------- #


def test_box_edge_optimum_batch_is_distinct_and_in_bounds():
    X, y, bounds = _monotonic()
    s = _fit(X, y, bounds)
    pb = propose_batch(s, bounds, q=5, seed=0)
    assert pb.returned == 5 and pb.notice is None
    assert _pairwise_min(pb.X, bounds) >= NEAR_DUPLICATE_TOL
    assert (pb.X >= bounds[0] - 1e-9).all() and (pb.X <= bounds[1] + 1e-9).all()
    # the plain `propose` wrapper returns the same rows
    np.testing.assert_allclose(propose(s, bounds, q=5, seed=0), pb.X)


def test_a_collapsed_joint_batch_is_refilled_not_jittered(collapse_joint_batch):
    X, y, bounds = _monotonic(seed=3)
    s = _fit(X, y, bounds)
    pb = propose_batch(s, bounds, q=5, seed=0)
    assert pb.returned == 5 and pb.notice is None
    assert pb.near_duplicates_replaced == 4
    assert _pairwise_min(pb.X, bounds) >= NEAR_DUPLICATE_TOL
    assert (pb.X >= bounds[0] - 1e-9).all() and (pb.X <= bounds[1] + 1e-9).all()
    # deterministic: the repair is a selection, not random noise
    again = propose_batch(s, bounds, q=5, seed=0)
    np.testing.assert_allclose(again.X, pb.X)
    rep = pb.diversity_report()
    assert rep["requested"] == 5 and rep["returned"] == 5
    assert rep["min_separation"] == NEAR_DUPLICATE_TOL
    json.dumps(rep)


def test_constrained_batch_repair_still_honours_the_constraint(collapse_joint_batch):
    # target rewards high x1; purity falls with x1, so the floor caps x1.
    rng = np.random.default_rng(11)
    n = 60
    x1, x2 = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
    titer = 2.0 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n)
    purity = 99.0 - 1.2 * x1 + rng.normal(0, 0.3, n)
    X = np.column_stack([x1, x2])
    bounds = np.vstack([X.min(0), X.max(0)])
    s = _fit(X, titer, bounds)
    c = _fit(X, purity, bounds)
    floor = 92.0  # purity >= 92  <=>  x1 <~ 5.8
    pb = propose_batch(s, bounds, q=5, seed=0, constraint_surrogate=c, constraint_floor=floor)
    free = propose_batch(s, bounds, q=5, seed=0)
    assert pb.returned == 5 and pb.near_duplicates_replaced == 4
    assert _pairwise_min(pb.X, bounds) >= NEAR_DUPLICATE_TOL
    # Rows 1..4 were refilled by the sequential repair. They are scored by the
    # SAME constrained acquisition, so they sit at the purity floor, unlike the
    # refill of an unconstrained batch which chases titer into low purity. (The
    # constraint is BoTorch's probabilistic one, so "at the floor" is in
    # posterior mean, with a small margin.)
    refilled = np.asarray(c.posterior(pb.X[1:])[0])
    refilled_free = np.asarray(c.posterior(free.X[1:])[0])
    assert (refilled >= floor - 1.5).all(), refilled
    assert refilled.mean() > refilled_free.mean() + 2.0, (refilled, refilled_free)


def test_tiny_discrete_space_returns_fewer_with_a_notice():
    # Two categorical factors with 2 levels each: only 4 distinct recipes exist.
    codes = np.array([[i, j] for i in range(2) for j in range(2)] * 3, float)
    rng = np.random.default_rng(0)
    y = codes[:, 0] + 0.5 * codes[:, 1] + rng.normal(0, 0.05, len(codes))
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    s = _fit(codes, y, bounds, cat_dims=[0, 1])
    pb = propose_batch(s, bounds, q=5, cat_dims=[0, 1], cat_cardinalities=[2, 2], seed=0)
    assert pb.returned == 4
    assert len({tuple(r) for r in pb.X}) == 4  # never an exact duplicate
    assert pb.notice is not None
    assert pb.notice["code"] == "fewer_distinct_candidates"
    assert pb.notice["requested"] == 5 and pb.notice["returned"] == 4
    assert "4 of the 5" in pb.notice["message"]
    json.dumps(pb.notice)


def test_mixed_space_batch_is_distinct(collapse_joint_batch):
    rng = np.random.default_rng(2)
    n = 24
    x = rng.uniform(0, 1, n)
    lvl = rng.integers(0, 3, n).astype(float)
    y = 3 * x + lvl + rng.normal(0, 0.05, n)
    X = np.column_stack([x, lvl])
    bounds = np.array([[0.0, 0.0], [1.0, 2.0]])
    s = _fit(X, y, bounds, cat_dims=[1])
    # collapse only affects optimize_acqf; the mixed path is joint through
    # optimize_acqf_mixed, so this checks the ordinary mixed path end to end
    pb = propose_batch(s, bounds, q=5, cat_dims=[1], cat_cardinalities=[3], seed=0)
    assert pb.returned == 5
    assert _pairwise_min(pb.X, bounds, cat_dims=[1]) >= NEAR_DUPLICATE_TOL
    assert set(np.unique(pb.X[:, 1])) <= {0.0, 1.0, 2.0}


# --- the portal response ---------------------------------------------------------- #

pytest.importorskip("fastapi")
from kalos.portal.analysis import _analyze  # noqa: E402


def test_analyze_box_edge_sheet_has_no_duplicate_recipes():
    X, y, _ = _monotonic(seed=0)
    df = pd.DataFrame({"Methanol": X[:, 0], "pH": X[:, 1], "lipase_titer": y.round(3)})
    out = _analyze(df)
    recipes = [tuple(p["recipe"].values()) for p in out["proposals"]]
    assert len(recipes) == 5 and len(set(recipes)) == 5
    assert out["proposal_notice"] is None
    div = out["batch_diversity"]
    assert div["requested"] == 5 and div["returned"] == 5
    assert div["min_separation"] == NEAR_DUPLICATE_TOL
    json.dumps({k: out[k] for k in ("proposal_notice", "batch_diversity")})
