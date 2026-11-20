"""kalos.core.resolution: proposals rounded to a makeable resolution, toward the
feasible side (BIOQORE_PORT item 1, WS-A PR 1). Pure-function tests; nothing in
the analysis pipeline calls this module yet, so no model output changes."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pytest

from kalos.core.resolution import (
    DEFAULT_SIG_FIGS,
    MAX_EXHAUSTIVE_DIMS,
    REFUSAL_CODE,
    ResolutionFactor,
    round_to_resolution,
)


@dataclass(frozen=True)
class _Factor:
    """Structural stand-in with the attribute names of the cold-start design's
    factor (`kalos.core.design.ContinuousFactor`: name, lo, hi, step, levels).
    WS-A PR 2 replaces it with the real class once feat/cold-start-design merges
    and asserts equality with `ContinuousFactor.grid()`."""

    name: str = "x"
    lo: float = 0.0
    hi: float = 1.0
    step: float | None = None
    levels: tuple[float, ...] | None = None


def _sig_digits(v: float) -> int:
    return len(Decimal(repr(v)).normalize().as_tuple().digits)


def _ok(x: np.ndarray, bounds, cons) -> bool:
    lo, hi = np.asarray(bounds, float)
    if not np.all((x >= lo) & (x <= hi)):
        return False
    return all(float(x[list(i)] @ np.asarray(c)) >= rhs for i, c, rhs in cons)


# --------------------------------------------------------------------------- #
# acceptance: sub-milligram input (observed range 0.000208 to 0.000898)
# --------------------------------------------------------------------------- #


def test_sub_milligram_input_never_leaves_the_observed_range_or_hits_zero():
    lo, hi = 0.000208, 0.000898
    xs = np.concatenate([np.linspace(lo, hi, 401), [lo, hi, lo - 1e-9, hi + 1e-9, 0.00020800001]])
    for x in xs:
        res = round_to_resolution([x], [None], [[lo], [hi]])
        assert res.feasible, x
        assert res.x is not None
        v = float(res.x[0])
        assert lo <= v <= hi and v != 0.0, (x, v)
        assert _sig_digits(v) <= DEFAULT_SIG_FIGS, (x, v)
        assert abs(v - min(max(x, lo), hi)) <= 1e-7 + 1e-12, (x, v)  # within one 4-sig-fig step
    # The behaviour this replaces: 3 decimals put every one of these at 0.0 or 0.001.
    assert all(not (lo <= round(float(x), 3) <= hi) for x in np.linspace(lo, hi, 401))


def test_sig_figs_handles_negative_values_and_zero():
    res = round_to_resolution([-0.0123456, 0.0, 12345.678], [None] * 3, [[-1, -1, 0], [1, 1, 1e5]])
    assert res.x is not None
    assert res.x.tolist() == [-0.01235, 0.0, 12350.0]
    assert res.rules == ("sig_figs",) * 3


def test_bounds_pinned_tighter_than_the_sig_fig_step_serve_the_declared_value():
    v = 0.000208123
    res = round_to_resolution([v], [None], [[v], [v]])
    assert res.x is not None and res.x.tolist() == [v]


# --------------------------------------------------------------------------- #
# acceptance: an active linear constraint gives zero violations
# --------------------------------------------------------------------------- #


def test_constraint_on_the_boundary_gives_zero_violations_where_3_decimals_break_it():
    rng = np.random.default_rng(20261120)
    n, d = 1000, 4
    violations = refused = old_violations = 0
    for _ in range(n):
        a = rng.uniform(0.2, 3.0, d)
        x = rng.uniform(0.0, 1.0, d)
        cap = float(a @ x)  # active:  a @ x <= cap, i.e. -a @ x >= -cap
        cons = [(list(range(d)), list(-a), -cap)]
        res = round_to_resolution(x, [None] * d, [[0.0] * d, [1.0] * d], cons)
        if res.x is None:
            refused += 1
            continue
        violations += int(not _ok(res.x, [[0.0] * d, [1.0] * d], cons))
        old_violations += int(float(-a @ np.round(x, 3)) < -cap)
    assert violations == 0 and refused == 0
    assert old_violations > n // 4  # the bug the module fixes is real in this probe


def test_one_halfspace_and_a_feasible_point_always_round_feasibly():
    """With a single linear constraint and a feasible point, the cell corner that
    maximises the constraint's left side is feasible, so the search never refuses
    (steps aligned with the bounds, so no candidate is clipped)."""
    rng = np.random.default_rng(7)
    for _ in range(400):
        d = int(rng.integers(1, 7))
        factors: list[ResolutionFactor | None] = [
            None if rng.random() < 0.5 else _Factor(step=float(rng.choice([0.05, 0.1, 0.25])))
            for _ in range(d)
        ]
        x = rng.uniform(0.0, 1.0, d)
        coef = rng.normal(size=d)
        rhs = float(coef @ x) - float(rng.uniform(0, 0.05))
        cons = [(list(range(d)), list(coef), rhs)]
        bounds = [[0.0] * d, [1.0] * d]
        res = round_to_resolution(x, factors, bounds, cons)
        assert res.x is not None, res.refusal
        assert _ok(res.x, bounds, cons)


def test_the_closest_feasible_corner_is_chosen():
    rng = np.random.default_rng(3)
    for _ in range(200):
        d = 3
        x = rng.uniform(0.0, 1.0, d)
        coef = rng.uniform(0.5, 2.0, d)
        cons = [(list(range(d)), list(coef), float(coef @ x))]
        factors: list[ResolutionFactor | None] = [_Factor(step=0.1)] * d
        res = round_to_resolution(x, factors, [[0.0] * d, [1.0] * d], cons)
        assert res.x is not None
        corners = [np.array(c) for c in itertools.product(*[(np.floor(v * 10) / 10, np.ceil(v * 10) / 10) for v in x])]
        feas = [c for c in corners if float(c @ coef) >= cons[0][2] - 1e-12]
        best = min(float((((c - x)) ** 2).sum()) for c in feas)
        assert float(((res.x - x) ** 2).sum()) <= best + 1e-12


def test_a_feasible_nearest_recipe_is_not_moved():
    cons = [([0, 1], [1.0, 1.0], 0.5)]
    res = round_to_resolution([0.41234, 0.51234], [None, None], [[0, 0], [1, 1]], cons)
    assert res.x is not None and res.x.tolist() == [0.4123, 0.5123]
    assert res.search == "nearest" and res.moved == ()


def test_rounding_moves_the_needed_input_toward_the_feasible_side():
    # x + y >= 1 holds at the proposal; nearest 4-sig-fig values sum to 0.9999.
    cons = [([0, 1], [1.0, 1.0], 1.0)]
    res = round_to_resolution([0.33335, 0.66665], [None, None], [[0, 0], [1, 1]], cons)
    assert res.x is not None
    assert float(res.x.sum()) >= 1.0
    assert res.moved and res.search == "exhaustive"


def test_greedy_repair_above_the_exhaustive_limit():
    d = MAX_EXHAUSTIVE_DIMS + 4
    rng = np.random.default_rng(11)
    for _ in range(50):
        x = rng.uniform(0.1, 0.9, d)
        x = np.round(x, 6) + 0.0000004  # every input has two distinct candidates
        coef = rng.uniform(0.5, 2.0, d)
        cons = [(list(range(d)), list(coef), float(coef @ x))]
        res = round_to_resolution(x, [None] * d, [[0.0] * d, [1.0] * d], cons)
        if res.search == "nearest":
            continue
        assert res.search == "greedy"
        assert res.x is not None and _ok(res.x, [[0.0] * d, [1.0] * d], cons)


# --------------------------------------------------------------------------- #
# declared steps and levels
# --------------------------------------------------------------------------- #


def test_step_snaps_to_the_grid_without_binary_noise():
    factors = [_Factor(step=0.1), _Factor(hi=2.0, step=0.25), _Factor(hi=10.0, step=2.0)]
    res = round_to_resolution([0.29999, 1.26, 7.0], factors, [[0, 0, 0], [1, 2, 10]])
    assert res.x is not None
    assert res.x.tolist() == [0.3, 1.25, 6.0]  # 3 * 0.1 served as 0.3; tie 7.0 -> lower (6.0)
    assert res.rules == ("step",) * 3


def test_step_grid_is_anchored_at_the_factor_lo_like_the_design_grid():
    # ContinuousFactor.grid(): lo + k * step, k = 0..K -> 0.37, 0.87, 1.37, ...
    f = _Factor(lo=0.37, hi=3.0, step=0.5)
    res = round_to_resolution([1.0, 2.99], [f, f], [[0.37, 0.37], [3.0, 3.0]])
    assert res.x is not None and res.x.tolist() == [0.87, 2.87]


def test_every_step_output_is_a_point_of_the_factor_grid():
    """Membership in a locally built lo + k*step grid (k = 0..K), for random
    factors, proposal boxes inside the factor and random points."""
    rng = np.random.default_rng(5)
    for _ in range(2000):
        f_lo = float(np.round(rng.uniform(-5, 5), int(rng.integers(0, 4))))
        step = float(rng.choice([0.001, 0.01, 0.05, 0.1, 0.25, 0.3, 1.0, 2.5]))
        k_last = int(rng.integers(1, 40))
        f_hi = f_lo + k_last * step + float(rng.uniform(0, step * 0.99))
        f = _Factor(lo=f_lo, hi=f_hi, step=step)
        lo, hi = np.sort(rng.uniform(f_lo, f_hi, 2))
        x = float(rng.uniform(lo - step, hi + step))
        res = round_to_resolution([x], [f], [[lo], [hi]])
        grid = np.array([f_lo + k * step for k in range(int(np.floor((f_hi - f_lo) / step + 1e-9)) + 1)])
        inside = grid[(grid >= lo - 1e-9) & (grid <= hi + 1e-9)]
        if res.x is None:
            assert inside.size == 0 or not np.any((inside >= lo) & (inside <= hi)), (f, lo, hi)
            continue
        v = float(res.x[0])
        assert lo <= v <= hi
        assert np.min(np.abs(grid - v)) < 1e-9, (f, lo, hi, x, v)
        # and it is a nearest grid point inside the box
        assert abs(v - x) <= np.min(np.abs(inside - x)) + 1e-9


def test_step_rounds_into_the_bounds_not_across_them():
    res = round_to_resolution([0.3], [_Factor(step=0.25)], [[0.26], [1.0]])
    assert res.x is not None and res.x.tolist() == [0.5]
    res = round_to_resolution([0.99], [_Factor(step=0.25)], [[0.0], [0.9]])
    assert res.x is not None and res.x.tolist() == [0.75]
    # proposal box wider than the factor: the factor's own grid still bounds it
    res = round_to_resolution([1.4], [_Factor(step=0.25)], [[0.0], [2.0]])
    assert res.x is not None and res.x.tolist() == [1.0]


def test_levels_use_the_nearest_declared_level_inside_the_bounds():
    f = _Factor(hi=2.0, levels=(0.0, 0.5, 1.0, 5.0))  # 5.0 is outside the factor
    res = round_to_resolution([0.7], [f], [[0.0], [2.0]])
    assert res.x is not None and res.x.tolist() == [0.5]
    res = round_to_resolution([1.9], [f], [[0.0], [10.0]])
    assert res.x is not None and res.x.tolist() == [1.0]
    res = round_to_resolution([0.9], [f], [[0.0], [0.8]])  # 1.0 is outside the proposal box
    assert res.x is not None and res.x.tolist() == [0.5]
    assert res.rules == ("levels",)


def test_a_factor_with_neither_step_nor_levels_uses_sig_figs():
    res = round_to_resolution([0.123456], [_Factor()], [[0.0], [1.0]], sig_figs=2)
    assert res.x is not None and res.x.tolist() == [0.12]
    assert res.rules == ("sig_figs",)


def test_categorical_codes_are_left_unchanged():
    res = round_to_resolution([0.123456, 2.0], [None, None], [[0.0, 0.0], [1.0, 3.0]], cat_dims=[1])
    assert res.x is not None and res.x.tolist() == [0.1235, 2.0]
    assert res.rules == ("sig_figs", "categorical")


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_no_step_multiple_inside_the_bounds_is_refused():
    res = round_to_resolution([0.3], [_Factor(step=0.25)], [[0.26], [0.49]])
    assert res.x is None and not res.feasible
    assert res.refusal is not None
    assert res.refusal["code"] == REFUSAL_CODE
    assert res.refusal["inputs"] == [0]
    assert res.refusal["reason"] and res.refusal["what_would_fix_it"]


def test_constraints_no_grid_corner_can_meet_are_refused_not_served_unrounded():
    # x + y == 1 (two inequalities) with a 0.3 step: the corners sum to 0.6, 0.9 or 1.2.
    cons = [([0, 1], [1.0, 1.0], 1.0), ([0, 1], [-1.0, -1.0], -1.0)]
    res = round_to_resolution([0.5, 0.5], [_Factor(step=0.3)] * 2, [[0, 0], [1, 1]], cons)
    assert res.x is None
    assert res.refusal is not None and res.refusal["code"] == REFUSAL_CODE
    assert res.refusal["inputs"] == [0, 1]


# --------------------------------------------------------------------------- #
# purity, determinism, argument checks
# --------------------------------------------------------------------------- #


def test_deterministic_and_inputs_not_mutated():
    x = np.array([0.33335, 0.66665])
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    cons = [([0, 1], [1.0, 1.0], 1.0)]
    a = round_to_resolution(x, [None, None], bounds, cons)
    b = round_to_resolution(x, [None, None], bounds, cons)
    assert a.x is not None and b.x is not None
    assert (a.x.tolist(), a.rules, a.moved, a.search) == (b.x.tolist(), b.rules, b.moved, b.search)
    assert x.tolist() == [0.33335, 0.66665] and bounds.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert a.x is not x


def test_empty_point():
    res = round_to_resolution(np.empty(0), [], np.empty((2, 0)))
    assert res.x is not None and res.x.shape == (0,)


@pytest.mark.parametrize(
    ("x", "factors", "bounds", "cons", "kw"),
    [
        ([np.nan], [None], [[0], [1]], [], {}),
        ([0.5], [None, None], [[0], [1]], [], {}),
        ([0.5], [None], [[1], [0]], [], {}),
        ([0.5], [None], [[0], [np.inf]], [], {}),
        ([0.5], [_Factor(step=0.0)], [[0], [1]], [], {}),
        ([0.5], [_Factor(step=0.1, levels=(0.5,))], [[0], [1]], [], {}),
        ([0.5], [_Factor(levels=())], [[0], [1]], [], {}),
        ([0.5], [_Factor(levels=(2.0,))], [[0], [1]], [], {}),
        ([0.5], [_Factor(step=1.5)], [[0], [1]], [], {}),
        ([0.5], [_Factor(lo=1.0, hi=1.0)], [[0], [1]], [], {}),
        ([0.5], [None], [[0], [1]], [([1], [1.0], 0.0)], {}),
        ([0.5], [None], [[0], [1]], [([0], [1.0, 2.0], 0.0)], {}),
        ([0.5], [None], [[0], [1]], [], {"sig_figs": 0}),
        ([0.5], [None], [[0], [1]], [], {"cat_dims": [3]}),
    ],
)
def test_malformed_arguments_raise(x, factors, bounds, cons, kw):
    with pytest.raises(ValueError):
        round_to_resolution(x, factors, bounds, cons, **kw)
