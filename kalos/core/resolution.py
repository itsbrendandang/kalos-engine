"""Round a proposed recipe to the resolution a lab can actually make it at.

The optimizer works on continuous inputs and returns setpoints such as
0.000412873 g/L or 0.3333334 mL. A recipe is only useful if it can be weighed or
pipetted, and a served recipe is only honest if it is the point the model was
asked about. This module does the first half: it maps one proposed point to the
nearest recipe on each input's declared resolution that still satisfies the
proposal bounds and every linear constraint. (The caller re-predicts and
re-checks at the returned point; nothing here predicts.)

Per input, in order of precedence:

- a categorical input (`cat_dims`): left exactly as given (an integer level code);
- a factor with `levels` or `step`: a point of the factor's own grid, exactly
  as `kalos.core.design.ContinuousFactor.grid()` defines it (the declared levels
  inside the factor's [lo, hi]; or ``lo + k * step`` for k = 0..K, rounded to the
  step's decimals plus two), that also lies inside the proposal bounds. This
  module does not define a second grid; it adds only what the factor lacks:
  rounding toward the feasible side of the proposal bounds and constraints;
- otherwise (no factor, or a factor with neither): `sig_figs` significant
  figures (default 4). Kalos never invents a step for an undeclared input, and
  significant figures keep sub-milligram inputs (e.g. 0.000208 to 0.000898)
  meaningful where rounding to a fixed number of decimals sends them to 0.

Intentional difference from the design module for undeclared factors:
`ContinuousFactor.snap` rounds them to 1e-4 of the factor's range (a number of
decimals), which is right for laying out a space-filling plan, but sends a small
value in a wide range to 0 (0.0012 in [0, 100] keeps 2 decimals: 0.0). A
proposal must stay the recipe the model was asked about, so here an undeclared
input keeps 4 significant figures.

Each input has at most two candidates, the grid values just below and just
above the proposed value (moved inside the bounds when one falls outside).
The nearest candidate on every input is tried first. When that recipe breaks a
bound or a linear constraint, the other corners of the grid cell are searched,
nearest first (distance in units of each input's bound width), and the closest
recipe that satisfies everything is returned: the rounding goes toward the
feasible side. Up to `MAX_EXHAUSTIVE_DIMS` inputs with two candidates the search
is exhaustive; above that it repairs greedily (one input at a time, the change
that most reduces the total constraint violation). When no candidate recipe is
feasible the result carries a refusal (`rounding_infeasible`) with a reason and
what would fix it, and no point: such a row is withheld, never served unrounded.

Feasibility here is strict (``lo <= v <= hi`` and ``sum(coef * x) >= rhs`` with
no tolerance), so a returned recipe also passes the optimizer's and the setup
post-check's tolerant checks. Significant-figure values are computed in decimal
arithmetic, so a bound is never crossed by binary noise.

Factors are accepted structurally (`ResolutionFactor`): anything with read-only
`name`, `lo`, `hi`, `step` and `levels` attributes, as on
`kalos.core.design.ContinuousFactor` (the cold-start design's factor model, which
this module reuses rather than forks; `name` is not read here).

Pure and deterministic: numpy and the standard library only, no RNG, no I/O,
inputs are never mutated.

Idea credit: snapping generated values to a declared resolution, from
bioqore-synthetic-data-project (`generators/base.py`, by Srihan Vege and James
Hassett for Bioqore). Re-implemented, no code copied (see NOTICE).
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Protocol

import numpy as np

# Exhaustive corner search up to 2**12 = 4096 candidate recipes; greedy above.
MAX_EXHAUSTIVE_DIMS = 12
DEFAULT_SIG_FIGS = 4
REFUSAL_CODE = "rounding_infeasible"

# One `(indices, coefficients, rhs)` triple per constraint, meaning
# ``sum(coefficients[i] * x[indices[i]]) >= rhs`` in raw design units: the
# convention of `kalos.core.optimize` and `SetupPlan.linear_constraints`.
LinearConstraint = tuple[Sequence[int], Sequence[float], float]


class ResolutionFactor(Protocol):
    """The resolution of one input, read like `kalos.core.design.ContinuousFactor`:
    a grid over the factor's own [lo, hi]. At most one of `step` / `levels` is
    set; neither means "no declared resolution" (significant figures apply)."""

    @property
    def name(self) -> str: ...

    @property
    def lo(self) -> float: ...

    @property
    def hi(self) -> float: ...

    @property
    def step(self) -> float | None: ...

    @property
    def levels(self) -> Sequence[float] | None: ...


@dataclass(frozen=True, eq=False)
class Rounded:
    """The outcome of rounding one proposed point.

    `x` is the rounded recipe (a new array), or None when no recipe at the
    declared resolution is feasible; then `refusal` is
    ``{"code": "rounding_infeasible", "reason": ..., "what_would_fix_it": ...}``.
    `rules` names the rule applied per input (``categorical``, ``levels``,
    ``step``, ``sig_figs``). `moved` lists the inputs that were rounded to the
    farther of their two grid neighbours to satisfy a linear constraint. `search` is
    ``nearest``, ``exhaustive`` or ``greedy``."""

    x: np.ndarray | None
    rules: tuple[str, ...]
    moved: tuple[int, ...]
    search: str
    refusal: dict[str, Any] | None = None

    @property
    def feasible(self) -> bool:
        return self.x is not None


def _dec(v: float) -> Decimal:
    """The shortest decimal that round-trips to `v` (0.1 -> Decimal('0.1'))."""
    return Decimal(repr(float(v)))


def _sig_quantum(v: Decimal, sig_figs: int) -> Decimal:
    return Decimal(1).scaleb(v.adjusted() - sig_figs + 1)


def _sig_down(v: Decimal, sig_figs: int) -> Decimal:
    if v == 0:
        return v
    return v.quantize(_sig_quantum(v, sig_figs), rounding=ROUND_FLOOR)


def _sig_up(v: Decimal, sig_figs: int) -> Decimal:
    if v == 0:
        return v
    return v.quantize(_sig_quantum(v, sig_figs), rounding=ROUND_CEILING)


def _neighbours(lo_c: float | None, hi_c: float | None, xj: float) -> list[float]:
    """The (one or two) candidates nearest first; ties go to the lower value."""
    cands = sorted({c for c in (lo_c, hi_c) if c is not None})
    return sorted(cands, key=lambda c: (abs(c - xj), c))


def _candidates_levels(
    levels: Sequence[float], f_lo: float, f_hi: float, lo: float, hi: float, xj: float,
) -> list[float]:
    """Neighbours among the factor's grid (its levels inside its own [f_lo, f_hi],
    with `ContinuousFactor.grid()`'s 1e-12 slack) that lie inside [lo, hi]."""
    grid = {float(v) for v in levels if f_lo - 1e-12 <= float(v) <= f_hi + 1e-12}
    inside = sorted(v for v in grid if lo <= v <= hi)
    if not inside:
        return []
    below = [v for v in inside if v <= xj]
    above = [v for v in inside if v >= xj]
    return _neighbours(below[-1] if below else None, above[0] if above else None, xj)


def _decimals(v: float) -> int:
    exp = Decimal(repr(float(v))).normalize().as_tuple().exponent
    return max(0, -int(exp)) if isinstance(exp, int) else 12


def _step_value(step: float, f_lo: float, k: int) -> float:
    """Grid point k of a step factor: ``lo + k * step`` with the binary noise
    stripped (rounded to two more decimals than the step or lo, at most 12).
    `ContinuousFactor.grid()` rounds to the step's decimals only, which equals
    this whenever lo has no more decimals than the step (lo = 4.984 with
    step = 1.0 would otherwise give 5.98, off its own grid)."""
    return round(f_lo + k * step, min(max(_decimals(step), _decimals(f_lo)) + 2, 12))


def _candidates_step(step: float, f_lo: float, f_hi: float, lo: float, hi: float, xj: float) -> list[float]:
    """Neighbours among the factor's grid points k = 0..K that lie inside [lo, hi]."""
    k_last = int(math.floor((f_hi - f_lo) / step + 1e-9))

    def val(k: int) -> float:
        return _step_value(step, f_lo, k)

    # Integer estimates, then exact correction against the computed grid values.
    k_min = min(max(0, math.ceil((lo - f_lo) / step - 1e-9)), k_last + 1)
    while k_min <= k_last and val(k_min) < lo:
        k_min += 1
    while k_min > 0 and val(k_min - 1) >= lo:
        k_min -= 1
    k_max = max(min(k_last, math.floor((hi - f_lo) / step + 1e-9)), -1)
    while k_max >= 0 and val(k_max) > hi:
        k_max -= 1
    while k_max < k_last and val(k_max + 1) <= hi:
        k_max += 1
    if k_min > k_max:
        return []
    q = (xj - f_lo) / step
    k_dn = min(max(math.floor(q), k_min), k_max)
    k_up = min(max(math.ceil(q), k_min), k_max)
    if k_dn > k_min and val(k_dn) > xj:  # float noise in q
        k_dn -= 1
    if k_up < k_max and val(k_up) < xj:
        k_up += 1
    return _neighbours(val(k_dn), val(k_up), xj)


def _candidates_sig(sig_figs: int, lo: float, hi: float, xj: float) -> list[float]:
    d = _dec(xj)
    lo_in = float(_sig_up(_dec(lo), sig_figs))  # the smallest sig-fig value >= lo
    hi_in = float(_sig_down(_dec(hi), sig_figs))  # the largest sig-fig value <= hi
    if lo_in > hi_in:
        # The bounds are narrower than one sig-fig step (e.g. an input pinned
        # at lo == hi = 0.000208123): the declared bound values are the recipe.
        return _neighbours(lo, hi, min(max(xj, lo), hi))
    out = []
    for c in (float(_sig_down(d, sig_figs)), float(_sig_up(d, sig_figs))):
        out.append(min(max(c, lo_in), hi_in))
    return _neighbours(out[0], out[1], xj)


def _validate(
    x: np.ndarray, factors: Sequence[ResolutionFactor | None], bounds: np.ndarray,
    constraints: Sequence[LinearConstraint], sig_figs: int, cat_dims: Collection[int],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    d = x.shape[0]
    if not np.isfinite(x).all():
        raise ValueError("the point to round must be finite")
    if len(factors) != d:
        raise ValueError(f"{len(factors)} factors for a point with {d} inputs")
    if bounds.shape != (2, d) or not np.isfinite(bounds).all() or (bounds[0] > bounds[1]).any():
        raise ValueError("bounds must be finite (lower, upper) rows with lower <= upper, one column per input")
    if isinstance(sig_figs, bool) or not isinstance(sig_figs, int) or sig_figs < 1:
        raise ValueError("sig_figs must be a positive integer")
    if any(not 0 <= int(j) < d for j in cat_dims):
        raise ValueError("cat_dims must index the point's inputs")
    for j, f in enumerate(factors):
        if f is None:
            continue
        step, levels, f_lo, f_hi = f.step, f.levels, float(f.lo), float(f.hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo < f_hi):
            raise ValueError(f"input {j}: a factor needs finite lo < hi")
        if step is not None and levels is not None:
            raise ValueError(f"input {j}: a factor takes a step or levels, not both")
        if step is not None and not (math.isfinite(step) and 0 < step <= f_hi - f_lo):
            raise ValueError(f"input {j}: a step must be positive and no larger than the factor's range")
        if levels is not None and (len(levels) == 0 or not np.isfinite(np.asarray(levels, float)).all()
                                   or not any(f_lo - 1e-12 <= v <= f_hi + 1e-12 for v in levels)):
            raise ValueError(f"input {j}: levels must be finite with at least one inside the factor's bounds")
    lin = []
    for k, (idx, coef, rhs) in enumerate(constraints):
        ia = np.asarray(idx, dtype=int)
        ca = np.asarray(coef, dtype=float)
        if ia.ndim != 1 or ia.shape != ca.shape or ia.size == 0:
            raise ValueError(f"linear constraint {k}: indices and coefficients must be equal-length, non-empty")
        if ((ia < 0) | (ia >= d)).any() or not np.isfinite(ca).all() or not math.isfinite(float(rhs)):
            raise ValueError(f"linear constraint {k}: indices in range and finite numbers required")
        lin.append((ia, ca, float(rhs)))
    return lin


def _violation(X: np.ndarray, lin: list[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    """Total linear-constraint shortfall per row (0 = every constraint holds)."""
    v = np.zeros(X.shape[0])
    for ia, ca, rhs in lin:
        v += np.maximum(rhs - X[:, ia] @ ca, 0.0)
    return v


def _refusal(reason: str, fix: str, inputs: Sequence[int]) -> dict[str, Any]:
    return {"code": REFUSAL_CODE, "reason": reason, "what_would_fix_it": fix, "inputs": list(inputs)}


def round_to_resolution(
    x: Sequence[float] | np.ndarray,
    factors: Sequence[ResolutionFactor | None],
    bounds: Sequence[Sequence[float]] | np.ndarray,
    constraints: Sequence[LinearConstraint] = (),
    *,
    sig_figs: int = DEFAULT_SIG_FIGS,
    cat_dims: Collection[int] = (),
) -> Rounded:
    """Round the proposed point `x` (raw design units, one value per input) to
    each input's declared resolution, toward the feasible side of `bounds`
    (shape ``(2, d)``: lower row, upper row) and `constraints`
    (``sum(coef * x[idx]) >= rhs`` triples). `factors[j]` is input j's
    resolution (`ResolutionFactor`) or None for "undeclared" (`sig_figs`
    significant figures). Inputs listed in `cat_dims` are returned unchanged.
    See the module docstring for the rules. Raises ValueError on malformed
    arguments; an un-roundable point is a `Rounded` with a refusal, not an
    exception."""
    xv = np.array(x, dtype=float).reshape(-1)
    bv = np.array(bounds, dtype=float)
    lin = _validate(xv, factors, bv, constraints, sig_figs, cat_dims)
    d = xv.shape[0]
    cats = {int(j) for j in cat_dims}
    lo, hi = bv[0], bv[1]

    rules: list[str] = []
    cands: list[list[float]] = []
    empty: list[int] = []
    for j in range(d):
        f = factors[j]
        lj, hj, xj = float(lo[j]), float(hi[j]), float(xv[j])
        if j in cats:
            rules.append("categorical")
            cands.append([xj])
            continue
        if f is not None and f.levels is not None:
            rules.append("levels")
            c = _candidates_levels(f.levels, float(f.lo), float(f.hi), lj, hj, xj)
        elif f is not None and f.step is not None:
            rules.append("step")
            c = _candidates_step(float(f.step), float(f.lo), float(f.hi), lj, hj, xj)
        else:
            rules.append("sig_figs")
            c = _candidates_sig(sig_figs, lj, hj, xj)
        if not c:
            empty.append(j)
        cands.append(c)
    if empty:
        return Rounded(
            x=None, rules=tuple(rules), moved=(), search="nearest",
            refusal=_refusal(
                "no value at the declared resolution lies inside the bounds for "
                + ", ".join(f"input {j}" for j in empty),
                "widen the bounds, or declare a finer step (or a level inside the bounds) for that input",
                empty,
            ),
        )

    nearest = np.array([c[0] for c in cands])
    width = np.where(hi - lo > 0, hi - lo, 1.0)

    def result(point: np.ndarray, search: str) -> Rounded:
        moved = tuple(j for j in range(d) if point[j] != nearest[j])
        return Rounded(x=point.copy(), rules=tuple(rules), moved=moved, search=search)

    if _violation(nearest[None, :], lin)[0] == 0.0:
        return result(nearest, "nearest")

    two = [j for j in range(d) if len(cands[j]) == 2]
    if len(two) <= MAX_EXHAUSTIVE_DIMS:
        # Every corner of the grid cell, nearest first; ties broken by the
        # lexicographic corner order, so the choice is deterministic.
        picks = np.array(list(itertools.product((0, 1), repeat=len(two))), dtype=int).reshape(-1, len(two))
        X = np.tile(nearest, (picks.shape[0], 1))
        for col, j in enumerate(two):
            X[:, j] = np.asarray(cands[j])[picks[:, col]]
        ok = _violation(X, lin) == 0.0
        if ok.any():
            dist = (((X - xv) / width) ** 2).sum(axis=1)
            order = np.lexsort((np.arange(len(X)), dist))
            best = next(int(i) for i in order if ok[i])
            return result(X[best], "exhaustive")
        search = "exhaustive"
    else:
        point = nearest.copy()
        used: set[int] = set()
        viol = float(_violation(point[None, :], lin)[0])
        while viol > 0.0:
            trials = []
            for j in two:
                if j in used:
                    continue
                trial = point.copy()
                trial[j] = cands[j][1]
                v = float(_violation(trial[None, :], lin)[0])
                trials.append((v, float((((trial - xv) / width) ** 2).sum()), j, trial))
            if not trials:
                break
            v, _, j, trial = min(trials, key=lambda t: (t[0], t[1], t[2]))
            if v >= viol:
                break
            point, viol = trial, v
            used.add(j)
        if viol == 0.0:
            return result(point, "greedy")
        search = "greedy"

    bound_inputs = sorted({int(i) for ia, _, _ in lin for i in ia})
    return Rounded(
        x=None, rules=tuple(rules), moved=(), search=search,
        refusal=_refusal(
            "no recipe at the declared resolution next to this proposal satisfies every linear constraint",
            "declare a finer resolution for the inputs in the constraint, or loosen the constraint",
            bound_inputs,
        ),
    )
