"""Cold-start experiment designs: space-filling plans with no model in them.

Idea credit: the Latin-hypercube mode of the user's earlier BayOp service
(Bioqore `voyager-ml-bayop`, `bayop_sampling.latin_hypercube_sample` and
`LhsState`: 30 space-filling runs, no target needed). See NOTICE. This module is
a re-implementation to Kalos's rules, not a copy.

What a design here is, and is not:

- It is a plan of recipes chosen ONLY for coverage of the declared search
  region. No surrogate is fitted, nothing is predicted, and no row carries a
  predicted value or an interval. Callers label every row as exploration.
- It respects the declared bounds, linear constraints (by rejection, counted and
  reported), discrete setpoints (snapped to a declared step or level list; a
  step is never invented), categorical columns (levels balanced, counting runs
  that already exist), a held level (e.g. a clone under policy ``fixed``),
  optional centre points and optional replicate slots.
- With runs already measured (or in flight), it fills gaps: each new point is
  chosen to maximise its minimum distance to every existing point and to the
  points already chosen (greedy maximin over a feasible scrambled-Sobol pool),
  instead of starting a fresh design that ignores them.
- It is deterministic: everything random comes from `numpy.random.default_rng`
  seeded by the caller's `seed` (no global RNG is read or advanced), so the same
  inputs give byte-identical rows.

Methods (`METHODS`): ``sobol`` (scrambled Sobol, Owen scrambling), ``lhs_maximin``
(Latin hypercube, best of several starts then element swaps that raise the
smallest pairwise distance), ``random`` (uniform; a baseline), ``lhs`` (plain,
unoptimised Latin hypercube) and ``lhs_cd`` (Latin hypercube refined for
centered discrepancy: BayOp's default, ``optimization="random-cd"``). Which one
is the default is decided by the benchmark in `benchmarks/cold_start/`.

Distances and discrepancy are measured on the continuous factors, each scaled to
[0, 1] by its declared range; categorical columns are reported as level counts.

numpy + scipy only (torch-free), so the HTTP layer can import it at start-up.
"""
from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

METHODS: tuple[str, ...] = ("sobol", "lhs_maximin", "random", "lhs", "lhs_cd")
# Same relative tolerance as the optimizer's own constraint check
# (`kalos.core.optimize._LIN_TOL`) and the setup's (`setup_settings.LIN_TOL`).
LIN_TOL = 1e-6
# Candidate pool for gap filling / refilling rejected rows: at least this many
# scrambled-Sobol points (a power of two), and at least POOL_PER_ROW per row.
MIN_POOL = 1024
POOL_PER_ROW = 64
MAX_POOL = 1 << 15
# A fully discrete space (every continuous factor has a step or level list)
# with at most this many grid points is enumerated exactly instead of sampled.
MAX_ENUMERATED_GRID = 1 << 14
# Categorical combinations balanced jointly when there are at most this many.
MAX_JOINT_COMBOS = 512
# Gap filling picks from a Sobol prefix of FILL_PREFIX_FACTOR * (existing + new).
FILL_PREFIX_FACTOR = 4
# lhs_maximin: random starts, then element-swap iterations per row.
LHS_MAXIMIN_STARTS = 32
LHS_MAXIMIN_SWAPS_PER_ROW = 60
# Random-design reference for the quality report (draws from the feasible pool).
REFERENCE_DRAWS = 64


class DesignError(ValueError):
    """A design that cannot be built. `code` is machine readable; `reason` and
    `what_would_fix_it` are engine-authored and never quote caller input."""

    def __init__(self, code: str, reason: str, what_would_fix_it: str | None = None) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.what_would_fix_it = what_would_fix_it

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"error": self.reason, "code": self.code}
        if self.what_would_fix_it:
            body["what_would_fix_it"] = self.what_would_fix_it
        return body


@dataclass(frozen=True)
class ContinuousFactor:
    """A numeric setting over [lo, hi]. `step`: values are snapped to the absolute grid k*step inside [lo, hi].
    `levels`: values are snapped to the nearest listed setpoint (inside [lo, hi]).
    At most one of the two; neither means the factor is truly continuous."""

    name: str
    lo: float
    hi: float
    step: float | None = None
    levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo < self.hi):
            raise DesignError("invalid_factor", "a factor needs finite bounds with lo < hi")
        if self.step is not None and self.levels is not None:
            raise DesignError("invalid_factor", "a factor takes a step or a list of levels, not both")
        if self.step is not None and not (math.isfinite(self.step) and 0 < self.step <= self.hi - self.lo):
            raise DesignError(
                "invalid_step", "a factor's step must be positive and no larger than its range",
            )
        if self.levels is not None:
            inside = [v for v in self.levels if self.lo - 1e-12 <= v <= self.hi + 1e-12]
            if not inside:
                raise DesignError(
                    "invalid_levels", "a factor's levels must include at least one value inside its bounds",
                )

    @property
    def discrete(self) -> bool:
        return self.step is not None or self.levels is not None

    def grid(self) -> np.ndarray:
        """Every allowed value (discrete factors only)."""
        if self.levels is not None:
            return np.array(sorted({float(v) for v in self.levels if self.lo - 1e-12 <= v <= self.hi + 1e-12}))
        assert self.step is not None
        # Absolute grid k * step (pipetting / weighing increments; pH 6.05, not
        # lo + 0.05 when lo = 6.02), the same anchor as kalos.core.resolution.
        k_lo = int(math.ceil(self.lo / self.step - 1e-9))
        k_hi = int(math.floor(self.hi / self.step + 1e-9))
        return np.array([self._round(k * self.step) for k in range(k_lo, k_hi + 1)])

    def _round(self, v: float) -> float:
        """Strip binary noise (0.1 * 3 = 0.30000000000000004) to the step's decimals."""
        if self.step is None:
            return float(v)
        exp = Decimal(repr(self.step)).normalize().as_tuple().exponent
        decimals = max(0, -int(exp)) if isinstance(exp, int) else 12
        return round(float(v), min(decimals + 2, 12))

    def snap(self, values: np.ndarray) -> np.ndarray:
        if self.levels is not None:
            g = self.grid()
            idx = np.abs(values[:, None] - g[None, :]).argmin(axis=1)
            return g[idx]
        if self.step is not None:
            g = self.grid()
            idx = np.abs(values[:, None] - g[None, :]).argmin(axis=1)
            return g[idx]
        # Truly continuous: no step is invented, but 1e-4 of the range is the
        # finest resolution reported (a 15-digit setpoint is noise, not a plan).
        decimals = max(0, int(math.ceil(-math.log10((self.hi - self.lo) / 1e4))))
        return np.clip(np.round(values, decimals), self.lo, self.hi)


@dataclass(frozen=True)
class CategoricalFactor:
    """A categorical column. `fixed` holds every row at one level (e.g. a clone
    under policy `fixed`); otherwise rows are balanced across `levels`."""

    name: str
    levels: tuple[str, ...]
    fixed: str | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise DesignError("invalid_levels", "a categorical factor needs at least one level")
        if len(set(self.levels)) != len(self.levels):
            raise DesignError("invalid_levels", "a categorical factor's levels must be distinct")
        if self.fixed is not None and self.fixed not in self.levels:
            raise DesignError("invalid_levels", "the held level is not one of the factor's levels")


@dataclass(frozen=True)
class LinearConstraint:
    """sum(coefficients[name] * x[name]) `op` rhs, over continuous factors."""

    coefficients: Mapping[str, float]
    op: str
    rhs: float


@dataclass
class DesignResult:
    rows: list[dict[str, Any]]
    roles: list[str]  # "space_filling" | "centre" | "replicate", one per row
    replicate_of: list[int | None]  # 0-based index of the row a replicate repeats
    method: str  # what was actually run (e.g. "maximin_fill" when augmenting)
    requested_method: str
    seed: int
    quality: dict[str, Any]
    constraints: dict[str, Any]
    snapping: dict[str, Any]
    levels: dict[str, Any]
    n_existing: int
    shortfall: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _sub_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _constraint_matrix(
    factors: Sequence[ContinuousFactor], constraints: Sequence[LinearConstraint],
) -> tuple[np.ndarray, np.ndarray]:
    """Every constraint as  a @ x <= b  (natural units)."""
    pos = {f.name: i for i, f in enumerate(factors)}
    a = np.zeros((len(constraints), len(factors)))
    b = np.zeros(len(constraints))
    for r, c in enumerate(constraints):
        if c.op not in ("<=", ">="):
            raise DesignError("invalid_constraint", "a constraint's op must be '<=' or '>='")
        sign = 1.0 if c.op == "<=" else -1.0
        for name, coef in c.coefficients.items():
            if name not in pos:
                raise DesignError(
                    "constraint_unknown_factor",
                    "a linear constraint names a column that is not a continuous design factor",
                    "constraints apply to continuous factors with bounds",
                )
            a[r, pos[name]] = sign * float(coef)
        b[r] = sign * float(c.rhs)
    return a, b


def _feasible(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or x.shape[0] == 0:
        return np.ones(x.shape[0], dtype=bool)
    lhs = x @ a.T
    return np.all(lhs <= b + LIN_TOL * np.maximum(1.0, np.abs(b)), axis=1)


def _to_unit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (x - lo) / (hi - lo)


def _from_unit(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + u * (hi - lo)


def _sobol(d: int, n: int, seed: int) -> np.ndarray:
    with warnings.catch_warnings():
        # "balance properties ... n is not a power of 2": a prefix of a scrambled
        # Sobol sequence is still a valid low-discrepancy set; silence the notice.
        warnings.simplefilter("ignore", UserWarning)
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def _lhs_maximin(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Best-of-starts LHS by the smallest pairwise distance, then column swaps
    (which keep the Latin property) accepted when they raise that distance."""
    if n < 2:
        return qmc.LatinHypercube(d=d, seed=_sub_seed(rng)).random(n)
    best: np.ndarray | None = None
    best_d = -1.0
    for _ in range(LHS_MAXIMIN_STARTS):
        cand = qmc.LatinHypercube(d=d, seed=_sub_seed(rng)).random(n)
        md = float(pdist(cand).min())
        if md > best_d:
            best, best_d = cand, md
    assert best is not None
    x = best.copy()
    for _ in range(LHS_MAXIMIN_SWAPS_PER_ROW * n):
        j = int(rng.integers(d))
        r1, r2 = (int(v) for v in rng.choice(n, size=2, replace=False))
        x[[r1, r2], j] = x[[r2, r1], j]
        md = float(pdist(x).min())
        if md > best_d:
            best_d = md
        else:
            x[[r1, r2], j] = x[[r2, r1], j]
    return x


def _raw_unit(method: str, d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if d == 0:
        return np.empty((n, 0))
    if method == "sobol":
        return _sobol(d, n, _sub_seed(rng))
    if method == "random":
        return rng.random((n, d))
    if method == "lhs":
        return qmc.LatinHypercube(d=d, seed=_sub_seed(rng)).random(n)
    if method == "lhs_cd":
        return qmc.LatinHypercube(d=d, seed=_sub_seed(rng), optimization="random-cd").random(n)
    if method == "lhs_maximin":
        return _lhs_maximin(d, n, rng)
    raise DesignError("invalid_method", f"method must be one of {', '.join(METHODS)}")


def _unique_rows(x: np.ndarray) -> np.ndarray:
    """Indices of the first occurrence of each distinct row, in order."""
    if x.shape[0] == 0:
        return np.arange(0)
    seen: dict[bytes, int] = {}
    keep = []
    for i, row in enumerate(np.round(x, 12)):
        key = row.tobytes()
        if key not in seen:
            seen[key] = i
            keep.append(i)
    return np.asarray(keep, dtype=int)


def _greedy_maximin(pool_u: np.ndarray, fixed_u: np.ndarray, k: int) -> list[int]:
    """Pick `k` pool rows one at a time, each maximising its minimum distance to
    `fixed_u` and to the rows already picked. Ties go to the lowest pool index
    (the pool order is the seeded Sobol order, so this is deterministic). With
    no anchor the first pick is the pool point closest to the centre."""
    if k <= 0 or pool_u.shape[0] == 0:
        return []
    dmin: np.ndarray | None = cdist(pool_u, fixed_u).min(axis=1) if fixed_u.shape[0] else None
    centre = np.full(pool_u.shape[1], 0.5)
    picked: list[int] = []
    available = np.ones(pool_u.shape[0], dtype=bool)
    for _ in range(min(k, pool_u.shape[0])):
        score = -np.linalg.norm(pool_u - centre, axis=1) if dmin is None else dmin.copy()
        score[~available] = -np.inf
        i = int(np.argmax(score))
        picked.append(i)
        available[i] = False
        dist_i = np.linalg.norm(pool_u - pool_u[i], axis=1)
        dmin = dist_i if dmin is None else np.minimum(dmin, dist_i)
    return picked


def _fill_candidates(order: np.ndarray, n_anchor: int, k: int, enumerated: bool) -> np.ndarray:
    """Mask of the pool rows gap filling may pick from: a Sobol PREFIX of at
    least 4 * (anchors + new points) (itself a balanced Sobol set), doubled until
    it holds 2k feasible candidates. Greedy maximin over the whole 1024+ pool
    sends new points into the corners; over this prefix it keeps most of the
    distance at about random-like discrepancy (benchmarks/cold_start/
    augment_study.py). An enumerated setpoint grid is used whole."""
    if enumerated or order.size == 0:
        return np.ones(order.size, dtype=bool)
    m = 1 << int(math.ceil(math.log2(max(2, FILL_PREFIX_FACTOR * (n_anchor + k)))))
    top = int(order.max()) + 1
    while True:
        mask = order < m
        if int(mask.sum()) >= 2 * k or m >= top:
            return mask
        m *= 2


def _farthest_order(u: np.ndarray, anchor_u: np.ndarray) -> list[int]:
    """An order of the rows of `u` in which each next row is the one farthest
    from the anchors and the rows before it: round-robin level assignment over
    this order gives every level a spread-out subset."""
    n = u.shape[0]
    if n == 0:
        return []
    if u.shape[1] == 0:
        return list(range(n))
    return _greedy_maximin(u, anchor_u, n)


def _quality(design_u: np.ndarray, existing_u: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "space": "continuous factors scaled to [0, 1] by their declared ranges",
        "n_points": int(design_u.shape[0]),
        "min_pairwise_distance": None,
        "mean_nearest_neighbour_distance": None,
        "centered_l2_discrepancy": None,
        "min_distance_to_existing": None,
        "centered_l2_discrepancy_with_existing": None,
    }
    if design_u.shape[1] == 0:
        out["note"] = "no continuous factors: coverage is the level balance only"
        return out
    if design_u.shape[0] >= 2:
        dm = cdist(design_u, design_u)
        np.fill_diagonal(dm, np.inf)
        out["min_pairwise_distance"] = round(float(dm.min()), 6)
        out["mean_nearest_neighbour_distance"] = round(float(dm.min(axis=1).mean()), 6)
        out["centered_l2_discrepancy"] = round(float(qmc.discrepancy(np.clip(design_u, 0, 1), method="CD")), 6)
    if existing_u.shape[0] and design_u.shape[0]:
        out["min_distance_to_existing"] = round(float(cdist(design_u, existing_u).min()), 6)
        both = np.clip(np.vstack([existing_u, design_u]), 0, 1)
        if both.shape[0] >= 2:
            out["centered_l2_discrepancy_with_existing"] = round(float(qmc.discrepancy(both, method="CD")), 6)
    return out


def _random_reference(
    pool_u: np.ndarray, n: int, rng: np.random.Generator, existing_u: np.ndarray,
) -> dict[str, Any] | None:
    """Median min-pairwise distance and discrepancy of `REFERENCE_DRAWS` random
    designs of the same size drawn from the same feasible pool: what 'no design
    effort' would have given under the same bounds, steps and constraints."""
    if pool_u.shape[1] == 0 or n < 2 or pool_u.shape[0] < n:
        return None
    mins, cds, cds_ex, to_ex = [], [], [], []
    for _ in range(REFERENCE_DRAWS):
        idx = rng.choice(pool_u.shape[0], size=n, replace=False)
        pts = pool_u[idx]
        mins.append(float(pdist(pts).min()))
        cds.append(float(qmc.discrepancy(np.clip(pts, 0, 1), method="CD")))
        if existing_u.shape[0]:
            to_ex.append(float(cdist(pts, existing_u).min()))
            cds_ex.append(float(qmc.discrepancy(np.clip(np.vstack([existing_u, pts]), 0, 1), method="CD")))
    out: dict[str, Any] = {
        "basis": f"median of {REFERENCE_DRAWS} random designs of the same size from the same feasible candidates",
        "min_pairwise_distance": round(float(np.median(mins)), 6),
        "centered_l2_discrepancy": round(float(np.median(cds)), 6),
    }
    if existing_u.shape[0]:
        out["min_distance_to_existing"] = round(float(np.median(to_ex)), 6)
        out["centered_l2_discrepancy_with_existing"] = round(float(np.median(cds_ex)), 6)
    return out


def _assign_levels(
    order: Sequence[int],
    cats: Sequence[CategoricalFactor],
    existing_labels: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Level of every categorical for each row (indexed like `order`'s values).
    Fixed factors get their held level. Free factors are balanced: each next row
    (in `order`) takes the level combination whose levels are least used so far,
    counting existing runs; ties go to a seeded permutation."""
    n = len(order)
    out: list[dict[str, str]] = [{} for _ in range(n)]
    free = [c for c in cats if c.fixed is None]
    for c in cats:
        if c.fixed is not None:
            for r in range(n):
                out[r][c.name] = c.fixed
    counts: dict[str, dict[str, int]] = {c.name: dict.fromkeys(c.levels, 0) for c in free}
    for row in existing_labels:
        for c in free:
            v = row.get(c.name)
            if isinstance(v, str) and v in counts[c.name]:
                counts[c.name][v] += 1
    existing_counts = {k: dict(v) for k, v in counts.items()}
    if free:
        n_combos = math.prod(len(c.levels) for c in free)
        if n_combos <= MAX_JOINT_COMBOS:
            combos = list(itertools.product(*(c.levels for c in free)))
            tie = rng.permutation(len(combos))
            joint: dict[tuple[str, ...], int] = dict.fromkeys(combos, 0)
            for row in existing_labels:
                key = tuple(row.get(c.name) for c in free)
                if key in joint:
                    joint[key] += 1  # type: ignore[index]
            for r in order:
                best = min(
                    range(len(combos)),
                    key=lambda i: (
                        sum(counts[c.name][combos[i][j]] for j, c in enumerate(free)),
                        joint[combos[i]],
                        int(tie[i]),
                    ),
                )
                combo = combos[best]
                joint[combo] += 1
                for j, c in enumerate(free):
                    counts[c.name][combo[j]] += 1
                    out[r][c.name] = combo[j]
        else:
            ties = {c.name: rng.permutation(len(c.levels)) for c in free}
            for r in order:
                for c in free:
                    cnt, tie_c = counts[c.name], ties[c.name]
                    k = min(range(len(c.levels)), key=lambda i: (cnt[c.levels[i]], int(tie_c[i])))  # noqa: B023
                    cnt[c.levels[k]] += 1
                    out[r][c.name] = c.levels[k]
    report: dict[str, Any] = {}
    for c in cats:
        design_counts = dict.fromkeys(c.levels, 0)
        for row in out:
            design_counts[row[c.name]] += 1
        report[c.name] = {
            "policy": "held" if c.fixed is not None else "balanced",
            "held_level": c.fixed,
            "design_counts": design_counts,
            "existing_counts": existing_counts.get(c.name),
        }
    return out, report


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #


def generate_design(
    continuous: Sequence[ContinuousFactor],
    categorical: Sequence[CategoricalFactor] = (),
    constraints: Sequence[LinearConstraint] = (),
    *,
    n: int,
    method: str = "sobol",
    seed: int = 0,
    existing: np.ndarray | None = None,
    existing_labels: Sequence[Mapping[str, Any]] = (),
    centre_points: int = 0,
    replicate_slots: int = 0,
) -> DesignResult:
    """A space-filling plan of `n` new points (+ `centre_points` + `replicate_slots`).

    `existing` (natural units, one column per continuous factor in order; rows
    with a NaN are ignored) and `existing_labels` (categorical levels per
    existing run) are the runs already measured or in flight. With any existing
    point the new points fill gaps (greedy maximin) whatever `method` says, and
    `DesignResult.method` reads ``maximin_fill``.
    """
    if method not in METHODS:
        raise DesignError("invalid_method", f"method must be one of {', '.join(METHODS)}")
    if n < 0 or centre_points < 0 or replicate_slots < 0:
        raise DesignError("invalid_size", "sizes must be non-negative")
    if not continuous and not categorical:
        raise DesignError("no_factors", "a design needs at least one factor")
    names = [f.name for f in continuous] + [c.name for c in categorical]
    if len(set(names)) != len(names):
        raise DesignError("duplicate_factor", "each factor may be declared once")
    rng = np.random.default_rng(seed)
    d = len(continuous)
    lo = np.array([f.lo for f in continuous], dtype=float)
    hi = np.array([f.hi for f in continuous], dtype=float)
    a, b = _constraint_matrix(continuous, constraints)

    def snap(x: np.ndarray) -> np.ndarray:
        if d == 0 or x.shape[0] == 0:
            return x
        return np.column_stack([f.snap(x[:, j]) for j, f in enumerate(continuous)])

    ex = np.empty((0, d)) if existing is None else np.asarray(existing, dtype=float).reshape(-1, d)
    n_existing_total = ex.shape[0]
    ex = ex[np.all(np.isfinite(ex), axis=1)] if ex.size else ex
    ex_u = _to_unit(ex, lo, hi) if d else np.empty((ex.shape[0], 0))
    notes: list[str] = []
    if n_existing_total != ex.shape[0]:
        notes.append(f"{n_existing_total - ex.shape[0]} existing run(s) with a blank setting were not used for gap filling")

    # Feasible, snapped candidate pool (shared by gap filling, refills, centre
    # points and the random reference).
    all_discrete = d > 0 and all(f.discrete for f in continuous)
    grid_size = math.prod(len(f.grid()) for f in continuous) if all_discrete else None
    if d == 0:
        pool = np.empty((1, 0))
        pool_order = np.zeros(1, dtype=int)
        pool_rejected = 0
        pool_drawn = 1
        pool_kind = "none (no continuous factors)"
    elif grid_size is not None and grid_size <= MAX_ENUMERATED_GRID:
        pool = np.array(list(itertools.product(*(f.grid() for f in continuous))), dtype=float)
        pool_drawn = pool.shape[0]
        ok = _feasible(pool, a, b)
        pool_rejected = int((~ok).sum())
        pool = pool[ok]
        pool_order = np.arange(pool.shape[0])
        pool_kind = "every point of the declared setpoint grid"
    else:
        size = max(MIN_POOL, POOL_PER_ROW * max(1, n + centre_points))
        size = min(MAX_POOL, 1 << int(math.ceil(math.log2(size))))
        raw = snap(_from_unit(_sobol(d, size, _sub_seed(rng)), lo, hi))
        pool_drawn = raw.shape[0]
        ok = _feasible(raw, a, b)
        pool_rejected = int((~ok).sum())
        pool_order = np.flatnonzero(ok)
        pool = raw[ok]
        uniq_rows = _unique_rows(pool)
        pool, pool_order = pool[uniq_rows], pool_order[uniq_rows]
        pool_kind = f"{size} scrambled-Sobol candidates, snapped to the declared setpoints"
    if d and pool.shape[0] == 0:
        raise DesignError(
            "no_feasible_candidates",
            "no candidate recipe satisfies the bounds, setpoints and linear constraints together "
            f"({pool_rejected} of {pool_drawn} candidates were rejected)",
            "loosen a constraint, widen a bound, or use a finer step",
        )
    pool_u = _to_unit(pool, lo, hi) if d else pool
    enumerated = pool_kind.startswith("every point")

    # ---- the space-filling points ---------------------------------------- #
    raw_rejected = 0
    raw_drawn = 0
    snap_merged = 0
    if d == 0:
        chosen = np.empty((n, 0))
        method_used = "levels_only"
    elif ex_u.shape[0]:
        cand = np.flatnonzero(_fill_candidates(pool_order, ex_u.shape[0], n, enumerated))
        picks = _greedy_maximin(pool_u[cand], ex_u, n)
        chosen = pool[cand[picks]]
        method_used = "maximin_fill"
    else:
        raw = snap(_from_unit(_raw_unit(method, d, n, rng), lo, hi))
        raw_drawn = raw.shape[0]
        ok = _feasible(raw, a, b)
        raw_rejected = int((~ok).sum())
        kept = raw[ok]
        uniq = _unique_rows(kept)
        snap_merged = int(kept.shape[0] - len(uniq))
        kept = kept[uniq]
        missing = n - kept.shape[0]
        if missing > 0:
            # Refill rejected / merged rows from the pool, away from what is kept.
            keep_keys = {r.tobytes() for r in np.round(kept, 12)}
            avail = np.array([r.tobytes() not in keep_keys for r in np.round(pool, 12)], dtype=bool)
            avail &= _fill_candidates(pool_order, kept.shape[0], missing + kept.shape[0], enumerated)
            cand = np.flatnonzero(avail)
            extra = _greedy_maximin(pool_u[cand], _to_unit(kept, lo, hi), missing)
            kept = np.vstack([kept, pool[cand[extra]]]) if extra else kept
        chosen = kept
        method_used = method
    shortfall = None
    if chosen.shape[0] < n:
        shortfall = {
            "requested": int(n), "returned": int(chosen.shape[0]),
            "reason": "only this many distinct recipes satisfy the bounds, setpoints and constraints "
                      "(and differ from the existing runs)",
            "what_would_fix_it": "use replicate_slots for deliberate repeats, a finer step, or wider bounds",
        }

    # ---- centre points --------------------------------------------------- #
    centre = np.empty((0, d))
    if centre_points:
        if d == 0:
            centre = np.empty((centre_points, 0))
        else:
            mid = snap(((lo + hi) / 2.0)[None, :])
            if not _feasible(mid, a, b)[0]:
                # The box centre breaks a constraint: the feasible candidate
                # nearest the centroid of all feasible candidates.
                c = pool_u.mean(axis=0)
                mid = pool[[int(np.argmin(np.linalg.norm(pool_u - c, axis=1)))]]
                notes.append("the box centre violates a linear constraint; centre points use the "
                             "feasible candidate nearest the centre of the feasible region")
            centre = np.repeat(mid, centre_points, axis=0)

    # ---- order, levels, replicates --------------------------------------- #
    chosen_u = _to_unit(chosen, lo, hi) if d else chosen
    order = _farthest_order(chosen_u, ex_u)
    base_rows = chosen.shape[0]
    all_order = order + list(range(base_rows, base_rows + centre.shape[0]))
    stacked = np.vstack([chosen, centre]) if d else np.empty((base_rows + centre.shape[0], 0))
    labels_by_row, levels_report = _assign_levels(
        all_order, categorical, existing_labels, rng,
    )
    # `_assign_levels` returns labels in `all_order` positions; map back to rows.
    labels = [dict[str, str]() for _ in range(stacked.shape[0])]
    for pos, r in enumerate(all_order):
        labels[r] = labels_by_row[pos]

    rows: list[dict[str, Any]] = []
    roles: list[str] = []
    rep_of: list[int | None] = []

    def as_row(i: int) -> dict[str, Any]:
        rec: dict[str, Any] = {f.name: float(stacked[i, j]) for j, f in enumerate(continuous)}
        rec.update(labels[i])
        return rec

    for i in all_order:  # space-filling rows first in farthest order, then centre points
        rows.append(as_row(i))
        roles.append("space_filling" if i < base_rows else "centre")
        rep_of.append(None)
    if replicate_slots:
        sources = list(range(len(rows))) if rows else []
        if not sources:
            raise DesignError("invalid_size", "replicate slots need at least one design row to repeat")
        for k in range(replicate_slots):
            src = sources[k % len(sources)]
            rows.append(dict(rows[src]))
            roles.append("replicate")
            rep_of.append(src)
        # Replicates change level counts: recount.
        for c in categorical:
            levels_report[c.name]["design_counts"] = {
                lv: sum(1 for r in rows if r[c.name] == lv) for lv in c.levels
            }

    distinct_u = chosen_u[order] if d else chosen_u
    if d and centre.shape[0]:
        distinct_u = np.vstack([distinct_u, _to_unit(centre[:1], lo, hi)])
    quality = _quality(distinct_u, ex_u)
    quality["random_reference"] = _random_reference(
        pool_u, distinct_u.shape[0], np.random.default_rng([seed, 7919]), ex_u,
    )
    quality["counted"] = "distinct design points (space-filling rows plus one centre point; replicates excluded)"
    return DesignResult(
        rows=rows, roles=roles, replicate_of=rep_of, method=method_used, requested_method=method,
        seed=int(seed), quality=quality,
        constraints={
            "n_constraints": len(constraints),
            "method": "rejection: candidates that break any constraint are discarded and counted",
            "design_candidates_drawn": raw_drawn,
            "design_candidates_rejected": raw_rejected,
            "pool_candidates_drawn": int(pool_drawn) if d else 0,
            "pool_candidates_rejected": int(pool_rejected) if d else 0,
            "pool_acceptance_rate": (
                round(1.0 - pool_rejected / pool_drawn, 4) if d and pool_drawn else None
            ),
            "pool": pool_kind,
        },
        snapping={
            "factors": {
                f.name: ({"levels": [float(v) for v in f.grid()]} if f.levels is not None
                         else {"step": f.step} if f.step is not None else None)
                for f in continuous
            },
            "merged_after_snapping": snap_merged,
        },
        levels=levels_report,
        n_existing=int(ex.shape[0]),
        shortfall=shortfall,
        notes=notes,
    )


__all__ = [
    "METHODS", "CategoricalFactor", "ContinuousFactor", "DesignError", "DesignResult",
    "LinearConstraint", "generate_design",
]
