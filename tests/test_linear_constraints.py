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
