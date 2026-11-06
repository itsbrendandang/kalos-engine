"""Hard linear + box constraints on proposals (e.g. a media-component budget)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import kalos.core.optimize as optimize
from kalos.core.optimize import proposal_separation, propose, propose_batch
from kalos.core.surrogate import Surrogate, sanitize_bounds


def _fit():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (15, 3))
    y = X.sum(1) + 0.01 * rng.normal(size=15)  # unconstrained optimum is the (1,1,1) corner
    b = np.array([[0, 0, 0], [1, 1, 1]], float)
    return Surrogate().fit(X, y, bounds=b), b


def test_total_budget_constraint_is_respected_by_every_proposal():
    s, b = _fit()
    free = propose(s, b, q=3, seed=0)
    assert (free.sum(1) > 1.5).any()  # the objective really does push past the budget
    out = propose(s, b, q=3, seed=0, linear_constraints=[([0, 1, 2], [-1.0, -1.0, -1.0], -1.5)])  # sum <= 1.5
    assert out.sum(1).max() <= 1.5 + 1e-6
    assert (out >= b[0] - 1e-9).all() and (out <= b[1] + 1e-9).all()  # box still holds


def test_lower_bound_on_a_component_ratio_style_constraint():
    s, b = _fit()
    out = propose(s, b, q=2, seed=1, linear_constraints=[([0, 1], [1.0, -1.0], 0.2)])  # x0 - x1 >= 0.2
    assert (out[:, 0] - out[:, 1] >= 0.2 - 1e-6).all()


def test_linear_constraints_rejected_with_categoricals():
    s, b = _fit()
    with pytest.raises(ValueError, match="categorical"):
        propose(s, b, q=1, cat_dims=[2], cat_cardinalities=[2], linear_constraints=[([0], [1.0], 0.0)])


# ---- validation, infeasibility, edge cases (review of PR #78) -------------

@pytest.mark.parametrize(
    "bad, match",
    [
        ([([-1], [1.0], 0.0)], "negative indices"),
        ([([3], [1.0], 0.0)], "indices must be in"),
        ([([0, 1], [1.0], 0.0)], "indices but"),
        ([([], [], 0.0)], "no indices"),
        ([([0], [float("nan")], 0.0)], "finite"),
        ([([0], [1.0], float("inf"))], "finite"),
        ([([0], [0.0], 0.0)], "all coefficients are zero"),
        ([([0, 0], [1.0, 1.0], 0.0)], "repeated index"),
        ([(0, 1.0, 0.0)], "must be"),
    ],
)
def test_invalid_constraint_inputs_are_rejected_clearly(bad, match):
    s, b = _fit()
    with pytest.raises(ValueError, match=match):
        propose(s, b, q=1, linear_constraints=bad)


def test_infeasible_constraint_set_raises_up_front():
    s, b = _fit()
    # x0 + x1 >= 5 cannot hold with both in [0, 1]
    with pytest.raises(ValueError, match="infeasible"):
        propose(s, b, q=1, linear_constraints=[([0, 1], [1.0, 1.0], 5.0)])
    # two constraints that contradict each other
    with pytest.raises(ValueError, match="infeasible"):
        propose(s, b, q=1, linear_constraints=[([0], [1.0], 0.8), ([0], [-1.0], -0.2)])


def test_constraint_active_exactly_at_a_box_edge():
    s, b = _fit()
    out = propose(s, b, q=2, seed=0, linear_constraints=[([0], [1.0], 1.0)])  # x0 >= 1 == upper bound
    assert np.allclose(out[:, 0], 1.0, atol=1e-6)


def test_very_tight_feasible_region():
    s, b = _fit()
    out = propose(s, b, q=2, seed=0, linear_constraints=[([0, 1, 2], [-1.0, -1.0, -1.0], -0.05)])  # sum <= 0.05
    assert out.sum(1).max() <= 0.05 + 1e-6 and (out >= -1e-9).all()


def test_every_row_of_a_q_batch_satisfies_every_constraint():
    s, b = _fit()
    cons = [([0, 1, 2], [-1.0, -1.0, -1.0], -2.0), ([0, 1], [1.0, -1.0], -0.3)]  # sum <= 2 ; x0 - x1 >= -0.3
    out = propose(s, b, q=4, seed=3, linear_constraints=cons)
    assert out.shape == (4, 3)
    assert (out.sum(1) <= 2.0 + 1e-6).all() and (out[:, 0] - out[:, 1] >= -0.3 - 1e-6).all()


def test_pending_points_with_constraints():
    s, b = _fit()
    pending = np.array([[0.2, 0.2, 0.2]])
    out = propose(s, b, q=2, seed=0, pending=pending, linear_constraints=[([0, 1, 2], [-1.0, -1.0, -1.0], -1.5)])
    assert out.sum(1).max() <= 1.5 + 1e-6


def test_fixed_seed_is_repeatable_with_constraints():
    s, b = _fit()
    cons = [([0, 1, 2], [-1.0, -1.0, -1.0], -1.5)]
    a = propose(s, b, q=2, seed=11, linear_constraints=cons)
    c = propose(s, b, q=2, seed=11, linear_constraints=cons)
    assert np.allclose(a, c)


# ---- near-duplicate repair under hard constraints --------------------------------


def _monotonic(seed=3):
    """Titer rises in both factors: the unconstrained optimum is the upper box corner."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0, 3, 30).round(2), rng.uniform(5, 7, 30).round(2)])
    y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(0, 0.05, 30)
    return X, y, np.vstack([X.min(0), X.max(0)])


def _fit_seeded(X, y, bounds):
    torch.manual_seed(1234)
    np.random.seed(1234)
    return Surrogate().fit(X, y, bounds=bounds)


def _patch_joint(monkeypatch, rewrite):
    """Rewrite only the JOINT (q>1) optimizer output, so the q=1 candidate-pool calls
    stay real. Returns the list of kwargs of every `optimize_acqf` call."""
    real = optimize.optimize_acqf
    calls: list[dict] = []

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        cands, vals = real(*args, **kwargs)
        if kwargs.get("q", 1) > 1 and kwargs.get("return_best_only", True):
            cands = rewrite(cands)
        return cands, vals

    monkeypatch.setattr(optimize, "optimize_acqf", wrapped)
    return calls


def _min_sep(X, bounds) -> float:
    lo, hi = sanitize_bounds(bounds)
    seps = [proposal_separation(X[i], X[j], lo, hi) for i in range(len(X)) for j in range(i + 1, len(X))]
    return min(seps) if seps else float("inf")


X_CAP = [([0], [-1.0], -2.0)]  # x0 <= 2; the unconstrained optimum sits at x0 = 3


def test_constrained_near_duplicate_repair_never_violates(monkeypatch):
    X, y, bounds = _monotonic()
    s = _fit_seeded(X, y, bounds)
    # Joint batch collapses onto its first row (the QA failure mode); that row is feasible.
    _patch_joint(monkeypatch, lambda c: c[:1].expand(c.shape[0], -1).clone())
    pb = propose_batch(s, bounds, q=5, seed=0, linear_constraints=X_CAP)
    assert pb.returned == 5 and pb.notice is None
    assert pb.near_duplicates_replaced == 4
    assert (pb.X[:, 0] <= 2.0 + 1e-6).all(), pb.X
    assert _min_sep(pb.X, bounds) >= optimize.NEAR_DUPLICATE_TOL
    assert (bounds[0] - 1e-9 <= pb.X).all() and (bounds[1] + 1e-9 >= pb.X).all()


def test_joint_rows_that_break_a_constraint_are_refilled_feasibly(monkeypatch):
    X, y, bounds = _monotonic()
    s = _fit_seeded(X, y, bounds)
    # Every joint row is the infeasible box corner (x0 = 3 > 2): all are dropped, so the
    # repair starts from an empty accepted set and must still return feasible rows.
    _patch_joint(monkeypatch, lambda c: torch.as_tensor(bounds[1], dtype=c.dtype).expand(c.shape[0], -1).clone())
    pb = propose_batch(s, bounds, q=5, seed=0, linear_constraints=X_CAP)
    assert pb.returned == 5 and pb.notice is None
    assert pb.near_duplicates_replaced == 5
    assert (pb.X[:, 0] <= 2.0 + 1e-6).all(), pb.X
    assert _min_sep(pb.X, bounds) >= optimize.NEAR_DUPLICATE_TOL


def test_every_optimizer_call_receives_the_constraints(monkeypatch):
    X, y, bounds = _monotonic()
    s = _fit_seeded(X, y, bounds)
    calls = _patch_joint(monkeypatch, lambda c: c[:1].expand(c.shape[0], -1).clone())
    propose_batch(s, bounds, q=5, seed=0, linear_constraints=X_CAP)
    assert any(k.get("return_best_only") is False for k in calls), "candidate-pool call never ran"
    assert all(k.get("inequality_constraints") is not None for k in calls), [
        k.get("q") for k in calls
    ]


def test_unconstrained_optimizer_calls_are_unchanged(monkeypatch):
    X, y, bounds = _monotonic()
    s = _fit_seeded(X, y, bounds)
    calls = _patch_joint(monkeypatch, lambda c: c[:1].expand(c.shape[0], -1).clone())
    propose_batch(s, bounds, q=5, seed=0)
    assert calls and all(k.get("inequality_constraints") is None for k in calls)


# ---- equalities (mixtures) and the best bet ------------------------------------------

MIX_EQ = [([0, 1, 2], [1.0, 1.0, 1.0], 1.5)]  # x0 + x1 + x2 == 1.5


def test_equality_holds_for_every_row_and_the_best_bet():
    s, b = _fit()
    pb = propose_batch(s, b, q=4, seed=0, equality_constraints=MIX_EQ)
    assert pb.returned == 4
    assert np.abs(pb.X.sum(1) - 1.5).max() <= 1e-6
    opt = optimize.posterior_mean_optimum(s, b, seed=0, equality_constraints=MIX_EQ)
    assert abs(opt.sum() - 1.5) <= 1e-6


def test_best_bet_respects_an_inequality():
    s, b = _fit()
    opt = optimize.posterior_mean_optimum(s, b, seed=0, linear_constraints=[([0, 1, 2], [-1.0] * 3, -1.5)])
    assert opt.sum() <= 1.5 + 1e-6


def test_equality_repair_pool_stays_on_the_plane(monkeypatch):
    X, y, bounds = _monotonic()
    s = _fit_seeded(X, y, bounds)
    eq = [([0, 1], [1.0, 1.0], 7.0)]  # x0 + x1 == 7
    calls = _patch_joint(monkeypatch, lambda c: c[:1].expand(c.shape[0], -1).clone())
    pb = propose_batch(s, bounds, q=4, seed=0, equality_constraints=eq)
    assert pb.returned == 4 and pb.near_duplicates_replaced == 3
    assert np.abs(pb.X.sum(1) - 7.0).max() <= 1e-6
    assert _min_sep(pb.X, bounds) >= optimize.NEAR_DUPLICATE_TOL
    assert all(k.get("equality_constraints") is not None for k in calls)


def test_infeasible_equality_raises():
    s, b = _fit()
    with pytest.raises(ValueError, match="infeasible"):
        propose(s, b, q=1, equality_constraints=[([0, 1], [1.0, 1.0], 3.0)])
