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


# Cells of one (candidates x rows) distance block. Security review S4: the old
# code built a candidates x rows x inputs array (2048 x 2000 x 40 = 1.3 GB);
# blocks of at most this many cells keep peak memory near 32 MB.
_BLOCK_CELLS = 1 << 22


def _nearest(C: np.ndarray, P: np.ndarray, cats: Sequence[int]) -> np.ndarray:
    """Distance from every row of C to its nearest row of P (continuous dims
    normalised, Euclidean; a categorical dim adds 1 when the level differs).
    Computed in row blocks with `cdist`, never a C x P x inputs array."""
    from scipy.spatial.distance import cdist

    catset = set(cats)
    cont = [j for j in range(C.shape[1]) if j not in catset]
    out = np.full(C.shape[0], np.inf)
    if P.shape[0] == 0:
        return out
    step = max(1, _BLOCK_CELLS // max(1, P.shape[0]))
    Pc = P[:, cont]
    for start in range(0, C.shape[0], step):
        blk = C[start : start + step]
        d2 = cdist(blk[:, cont], Pc, "sqeuclidean") if cont else np.zeros((len(blk), len(P)))
        for j in cats:
            d2 += np.rint(blk[:, j])[:, None] != np.rint(P[:, j])[None, :]
        out[start : start + step] = np.sqrt(d2.min(axis=1))
    return out


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
    nearest = _nearest(Cn, _normalise(P, lower, upper, cats), cats)
    picked: list[int] = []
    for _ in range(min(int(q), len(C))):
        i = int(np.argmax(nearest))
        if picked and nearest[i] <= 0:
            break
        picked.append(i)
        nearest = np.minimum(nearest, _nearest(Cn, Cn[i : i + 1], cats))
    return C[picked]


__all__ = ["maximin_batch"]
