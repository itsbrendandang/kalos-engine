"""A space-filling batch: recipes spread over the untested parts of the box.

Used when the model cannot be trusted to rank recipes yet (QUEUE round 3 item
6; engine-lit section 6: the initial step "is typically conducted using
space-filling designs such as Sobol sequences or Latin hypercube sampling").
The rule that decides WHEN lives in `kalos.portal.analysis_parts.exploration`.

Greedy maximin over a scrambled Sobol candidate set: each pick is the
candidate farthest (in range-normalised units, a differing categorical level
counting as distance 1) from every training row, every running recipe and every
recipe already picked. Seeded and torch-free.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import qmc


def _normalise(X: np.ndarray, lower: np.ndarray, upper: np.ndarray, cats: Sequence[int]) -> np.ndarray:
    width = np.where(upper - lower > 0, upper - lower, 1.0)
    Z = (np.asarray(X, float) - lower) / width
    for j in cats:
        Z[:, j] = np.asarray(X, float)[:, j]  # codes compared exactly below
    return Z


def _distances(C: np.ndarray, P: np.ndarray, cats: Sequence[int]) -> np.ndarray:
    """Euclidean distance from every row of C to every row of P (continuous
    dims normalised; a categorical dim adds 1 when the level differs)."""
    cont = [j for j in range(C.shape[1]) if j not in set(cats)]
    d2 = np.zeros((C.shape[0], P.shape[0]))
    if cont:
        diff = C[:, None, cont] - P[None, :, cont]
        d2 += (diff ** 2).sum(-1)
    for j in cats:
        d2 += (np.rint(C[:, None, j]) != np.rint(P[None, :, j])).astype(float)
    return np.sqrt(d2)


def maximin_batch(
    X_train: np.ndarray,
    bounds: np.ndarray,
    q: int,
    *,
    cat_dims: Sequence[int] | None = None,
    cat_cardinalities: Sequence[int] | None = None,
    pending: np.ndarray | None = None,
    seed: int = 0,
    n_candidates: int = 2048,
    feasible: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """`q` recipes (raw units, categorical dims as integer codes) chosen
    greedily to be as far as possible from the training rows, `pending` and
    each other. `feasible(rows) -> bool mask` can drop candidates (for example
    rows that break a linear constraint)."""
    b = np.asarray(bounds, float)
    lower, upper = b[0], b[1]
    d = lower.shape[0]
    cats = [int(j) for j in (cat_dims or [])]
    cards = list(cat_cardinalities or [])
    sob = qmc.Sobol(d=d, scramble=True, seed=int(seed)).random(int(n_candidates))
    C = lower + sob * (upper - lower)
    for j, k in zip(cats, cards, strict=False):
        C[:, j] = np.clip(np.floor(sob[:, j] * int(k)), 0, int(k) - 1)
    if feasible is not None:
        C = C[np.asarray(feasible(C), bool)]
    if len(C) == 0 or q <= 0:
        return np.empty((0, d))
    anchors = [np.asarray(X_train, float)]
    if pending is not None and len(pending):
        anchors.append(np.asarray(pending, float).reshape(-1, d))
    P = np.vstack(anchors) if anchors[0].size else np.empty((0, d))
    Cn = _normalise(C, lower, upper, cats)
    nearest = (
        _distances(Cn, _normalise(P, lower, upper, cats), cats).min(axis=1)
        if len(P) else np.full(len(C), np.inf)
    )
    picked: list[int] = []
    for _ in range(min(int(q), len(C))):
        i = int(np.argmax(nearest))
        if picked and nearest[i] <= 0:
            break
        picked.append(i)
        nearest = np.minimum(nearest, _distances(Cn, Cn[i : i + 1], cats)[:, 0])
    return C[picked]


__all__ = ["maximin_batch"]
