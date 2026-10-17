"""Hard linear + box constraints on proposals (e.g. a media-component budget)."""
from __future__ import annotations

import numpy as np
import pytest

from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate


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
