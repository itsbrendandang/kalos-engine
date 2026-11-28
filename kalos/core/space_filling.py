"""Torch-free space-filling Design of Experiments (DoE) for small-N regimes.

Provides deterministic scrambled Sobol + greedy maximin selection over mixed
continuous/categorical spaces with box bounds, fixed features, linear inequality
constraints, and pending-point avoidance. Kept 100% torch-free at import and
execution time so callers can use it without paying the torch import budget.
"""
from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

logger = logging.getLogger(__name__)

NEAR_DUPLICATE_TOL = 0.05
_LIN_TOL = 1e-6
_MAX_ENUMERATED_DISCRETE = 4096


@dataclass(frozen=True)
class ProposalBatch:
    """A proposed batch plus an honest account of how distinct it is.

    `X` has `returned <= requested` rows; every pair of rows is at least
    `min_separation` apart (`proposal_separation`). `near_duplicates_replaced`
    counts rows of the jointly-optimized batch that collapsed onto another row
    and were replaced by sequential, pending-aware selection (with linear
    constraints, rows that break one are replaced the same way). `notice` is None
    when the full batch was returned, else a machine- and human-readable dict
    explaining why fewer rows came back.
    """

    X: np.ndarray
    requested: int
    near_duplicates_replaced: int
    min_separation: float
    notice: dict | None

    @property
    def returned(self) -> int:
        return int(self.X.shape[0])

    def diversity_report(self) -> dict:
        return {
            "requested": int(self.requested),
            "returned": self.returned,
            "near_duplicates_replaced": int(self.near_duplicates_replaced),
            "min_separation": float(self.min_separation),
            "metric": (
                "max over continuous factors of |difference| / factor range; "
                "any differing categorical level counts as fully distinct"
            ),
        }


def _fewer_notice(requested: int, returned: int, tol: float) -> dict:
    return {
        "code": "fewer_distinct_candidates",
        "requested": int(requested),
        "returned": int(returned),
        "min_separation": float(tol),
        "message": (
            f"Only {returned} of the {requested} requested proposals are distinct. "
            "The design space has no further candidate that differs from every "
            "proposal already in the batch, and from every run already in flight, "
            f"by at least {tol:.0%} of a factor's range or by a categorical level, "
            "so the batch is returned short rather than repeating a recipe."
        ),
    }


def proposal_separation(
    a: np.ndarray,
    b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cat_dims: list[int] | None = None,
) -> float:
    """Distance between two proposals in the range-normalised design space.

    Each continuous coordinate is scaled by its design range (`upper - lower`),
    and the distance is the largest per-factor difference (Chebyshev), so 0.02
    means "no factor differs by more than 2% of its range". Categorical
    coordinates (integer level codes) are compared EXACTLY: any differing level
    makes the two recipes maximally different (distance 1.0, the full range),
    because there is no "nearby" level of an unordered factor.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    cats = set(cat_dims or [])
    for j in cats:
        if int(round(a[j])) != int(round(b[j])):
            return 1.0
    cont = [j for j in range(a.shape[0]) if j not in cats]
    if not cont:
        return 0.0
    width = np.asarray(upper, float)[cont] - np.asarray(lower, float)[cont]
    width = np.where(width > 0, width, 1.0)
    return float(np.max(np.abs(a[cont] - b[cont]) / width))


def _validate_linear_constraints(constraints, lower: np.ndarray, upper: np.ndarray):
    """Normalise and sanity-check `(indices, coefficients, rhs)` triples, and raise a
    clear ValueError if they cannot all hold inside the box (checked by LP)."""
    if not constraints:
        return []
    d = len(lower)
    out = []
    for k, c in enumerate(constraints):
        try:
            idx, coef, rhs = c
        except (TypeError, ValueError):
            raise ValueError(
                f"linear constraint {k} must be (indices, coefficients, rhs)"
            ) from None
        try:
            idx = [int(i) if float(i).is_integer() else -1 for i in idx]
            coef = [float(x) for x in coef]
            rhs = float(rhs)
        except (TypeError, ValueError):
            raise ValueError(
                f"linear constraint {k} must be (indices, coefficients, rhs) of numbers"
            ) from None
        if not idx:
            raise ValueError(f"linear constraint {k} has no indices")
        if len(idx) != len(coef):
            raise ValueError(
                f"linear constraint {k}: {len(idx)} indices but {len(coef)} coefficients"
            )
        if any(i < 0 or i >= d for i in idx):
            raise ValueError(
                f"linear constraint {k}: indices must be in [0, {d - 1}] (negative indices are not allowed)"
            )
        if len(set(idx)) != len(idx):
            raise ValueError(f"linear constraint {k}: repeated index")
        if not np.isfinite(coef).all() or not np.isfinite(float(rhs)):
            raise ValueError(f"linear constraint {k}: coefficients and rhs must be finite")
        if not any(coef):
            raise ValueError(f"linear constraint {k}: all coefficients are zero")
        out.append((idx, coef, float(rhs)))
    from scipy.optimize import linprog

    a = np.zeros((len(out), d))
    for r, (idx, coef, _) in enumerate(out):
        a[r, idx] = coef
    b = np.array([rhs for _, _, rhs in out])
    # sum(coef*x) >= rhs   <=>   -sum(coef*x) <= -rhs
    res = linprog(np.zeros(d), A_ub=-a, b_ub=-b, bounds=list(zip(lower, upper)), method="highs")
    if res.status == 2:
        raise ValueError(
            "linear constraints are infeasible within the bounds: no recipe can satisfy them"
        )
    return out


def _linear_ok(X: np.ndarray, lin) -> np.ndarray:
    """Boolean mask over the rows of `X`: True where every constraint in `lin` holds
    (to `_LIN_TOL`). A NaN row is False. With no constraints every row is True."""
    ok = np.ones(X.shape[0], dtype=bool)
    for idx, coef, rhs in lin:
        lhs = X[:, idx] @ np.asarray(coef)
        ok &= lhs >= rhs - _LIN_TOL * max(1.0, abs(rhs))
    return ok


def _resolve_cardinalities(
    cat_dims: list[int],
    cat_cardinalities: list[int] | None,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list[int]:
    """One level count per categorical dim, derived from the box when omitted."""
    given = [max(int(k), 1) for k in (cat_cardinalities or [])]
    out: list[int] = []
    for i, j in enumerate(cat_dims):
        if i < len(given):
            out.append(given[i])
        else:
            out.append(max(int(np.floor(upper[j])) + 1, 1))
    return out


def _sanitize_bounds_np(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Torch-free copy of `surrogate.sanitize_bounds`."""
    arr = np.asarray(bounds, float)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("bounds must have shape (2, d) = [lower_row, upper_row]")
    lower = arr[0].astype(float).copy()
    upper = arr[1].astype(float).copy()
    for row, partner in ((lower, upper), (upper, lower)):
        bad = ~np.isfinite(row)
        if bad.any():
            fill = np.where(np.isfinite(partner), partner, 0.0)
            row[bad] = fill[bad]
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    degenerate = upper - lower <= 0.0
    if degenerate.any():
        pad = 1e-6 * np.maximum(np.abs(lower), 1.0)
        upper = np.where(degenerate, lower + pad, upper)
    return lower, upper


def _sanitize_pending_np(
    pending,
    lower: np.ndarray,
    upper: np.ndarray,
    cats: list[int],
) -> np.ndarray | None:
    if pending is None:
        return None
    arr = np.asarray(pending, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    if arr.shape[1] != lower.shape[0]:
        logger.warning(
            "pending block rejected: expected width %d, got width %d (%d rows)",
            lower.shape[0],
            arr.shape[1],
            arr.shape[0],
        )
        return None
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        return None
    arr = np.clip(arr, lower, upper)
    for j in cats:
        arr[:, j] = np.rint(arr[:, j])
    return arr


def _pairwise_separation(
    pool: np.ndarray,
    refs: np.ndarray,
    cont: list[int],
    cats: list[int],
    width: np.ndarray,
) -> np.ndarray:
    """Normalized Chebyshev separation between `pool` (N x d) and `refs` (M x d)."""
    if cont:
        cont_d = np.max(np.abs(pool[:, None, cont] - refs[None, :, cont]) / width, axis=2)
    else:
        cont_d = np.zeros((pool.shape[0], refs.shape[0]), dtype=float)
    if cats:
        cat_mismatch = np.any(
            np.rint(pool[:, None, cats]) != np.rint(refs[None, :, cats]), axis=2
        )
        return np.where(cat_mismatch, 1.0, cont_d)
    return cont_d


def _build_candidate_pool(
    d: int,
    lower: np.ndarray,
    upper: np.ndarray,
    cont: list[int],
    cats: list[int],
    cards: list[int],
    fixed: dict[int, float],
    lin: list,
    q: int,
    seed: int,
) -> np.ndarray:
    free_cont = [j for j in cont if j not in fixed]
    free_cats: list[tuple[int, list[int]]] = []
    for j, card in zip(cats, cards):
        if j in fixed:
            continue
        levels = [c for c in range(card) if lower[j] - 1e-9 <= c <= upper[j] + 1e-9]
        if not levels:
            levels = list(range(card))
        free_cats.append((j, levels))

    if not free_cont:
        if not free_cats:
            cand = np.empty((1, d), dtype=float)
            for j in range(d):
                cand[0, j] = fixed.get(j, lower[j])
        else:
            n_combos = int(np.prod([len(lvls) for _, lvls in free_cats]))
            if n_combos <= _MAX_ENUMERATED_DISCRETE:
                combos = list(itertools.product(*(lvls for _, lvls in free_cats)))
            else:
                rng = np.random.default_rng(seed)
                combos = [
                    tuple(int(rng.choice(lvls)) for _, lvls in free_cats)
                    for _ in range(_MAX_ENUMERATED_DISCRETE)
                ]
            cand = np.zeros((len(combos), d), dtype=float)
            for j, v in fixed.items():
                cand[:, j] = v
            for col_idx, (dim_j, _) in enumerate(free_cats):
                cand[:, dim_j] = [row[col_idx] for row in combos]
        return cand[_linear_ok(cand, lin)]

    sobol_dim = len(free_cont) + len(free_cats)
    min_feasible = max(64, int(q) * 16)
    exponents = (10, 12, 14) if lin else (10,)
    feasible_pool = np.empty((0, d), dtype=float)
    for m in exponents:
        u = Sobol(d=sobol_dim, scramble=True, seed=int(seed)).random_base2(m=m)
        n = u.shape[0]
        cand = np.zeros((n, d), dtype=float)
        for j, v in fixed.items():
            cand[:, j] = v
        for k, dim_j in enumerate(free_cont):
            cand[:, dim_j] = lower[dim_j] + u[:, k] * (upper[dim_j] - lower[dim_j])
        for k, (dim_j, levels) in enumerate(free_cats):
            u_cat = u[:, len(free_cont) + k]
            idx = np.clip(np.floor(u_cat * len(levels)).astype(int), 0, len(levels) - 1)
            cand[:, dim_j] = np.asarray([levels[i] for i in idx], dtype=float)
        cand = np.clip(cand, lower, upper)
        for j in cats:
            cand[:, j] = np.rint(cand[:, j])
        feasible_pool = cand[_linear_ok(cand, lin)]
        if len(feasible_pool) >= min_feasible:
            break
    return feasible_pool


def propose_space_filling(
    bounds,
    q: int,
    *,
    X_train,
    cat_dims: Sequence[int] | None = None,
    cat_cardinalities: Sequence[int] | None = None,
    fixed_features: Mapping[int, float] | None = None,
    inequality_constraints: Sequence[tuple[Sequence[int], Sequence[float], float]] | None = None,
    pending=None,
    seed: int = 0,
    min_separation: float = NEAR_DUPLICATE_TOL,
) -> ProposalBatch:
    """Propose up to `q` space-filling points via deterministic scrambled Sobol + maximin.

    Respects box bounds, `fixed_features`, `cat_dims`/`cat_cardinalities`, linear
    `inequality_constraints` (`sum(x[indices] * coefficients) >= rhs`), `pending`
    recipes, and `min_separation` against `X_train` union `pending` union selected.
    """
    lower, upper = _sanitize_bounds_np(bounds)
    d = len(lower)
    q_int = max(int(q), 0)
    if q_int == 0:
        return ProposalBatch(np.empty((0, d), dtype=float), 0, 0, float(min_separation), None)

    cats = [int(j) for j in (cat_dims or ())]
    cat_set = set(cats)
    cont = [j for j in range(d) if j not in cat_set]

    fixed = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    for j, v in fixed.items():
        if not 0 <= j < d:
            raise ValueError("fixed_features names a dimension outside the design")
        if not np.isfinite(v):
            raise ValueError("fixed_features values must be finite")
        if j in cat_set:
            v = float(np.rint(v))
            fixed[j] = v
        lower[j] = upper[j] = v

    cards = (
        _resolve_cardinalities(
            cats,
            [int(c) for c in cat_cardinalities] if cat_cardinalities is not None else None,
            lower,
            upper,
        )
        if cats
        else []
    )
    lin = _validate_linear_constraints(inequality_constraints, lower, upper)
    pool = _build_candidate_pool(d, lower, upper, cont, cats, cards, fixed, lin, q_int, seed)

    refs_parts: list[np.ndarray] = []
    if X_train is not None:
        xt = np.asarray(X_train, dtype=float)
        if xt.ndim == 1 and xt.size:
            xt = xt.reshape(1, -1)
        if xt.ndim == 2 and xt.shape[0] > 0 and xt.shape[1] == d:
            xt = xt[np.isfinite(xt).all(axis=1)]
            if xt.shape[0] > 0:
                for j in cats:
                    xt[:, j] = np.rint(xt[:, j])
                refs_parts.append(xt)
    pend = _sanitize_pending_np(pending, lower, upper, cats)
    if pend is not None:
        refs_parts.append(pend)

    if len(pool) == 0:
        return ProposalBatch(
            np.empty((0, d), dtype=float),
            q_int,
            0,
            float(min_separation),
            _fewer_notice(q_int, 0, min_separation),
        )

    width = np.where((upper[cont] - lower[cont]) > 0, upper[cont] - lower[cont], 1.0) if cont else np.empty(0)
    if refs_parts:
        refs = np.vstack(refs_parts)
        min_dist = _pairwise_separation(pool, refs, cont, cats, width).min(axis=1)
    else:
        min_dist = np.full(len(pool), np.inf, dtype=float)

    selected: list[np.ndarray] = []
    for _ in range(q_int):
        best_idx = int(np.argmax(min_dist))
        if min_dist[best_idx] < float(min_separation):
            break
        chosen = pool[best_idx].copy()
        selected.append(chosen)
        d_new = _pairwise_separation(pool, chosen.reshape(1, -1), cont, cats, width).reshape(-1)
        min_dist = np.minimum(min_dist, d_new)

    X_out = np.vstack(selected) if selected else np.empty((0, d), dtype=float)
    notice = None if X_out.shape[0] >= q_int else _fewer_notice(q_int, X_out.shape[0], min_separation)
    return ProposalBatch(X_out, q_int, 0, float(min_separation), notice)


__all__ = [
    "NEAR_DUPLICATE_TOL",
    "ProposalBatch",
    "proposal_separation",
    "propose_space_filling",
]
