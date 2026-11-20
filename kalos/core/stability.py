"""Proposal stability across optimizer seeds: does the batch depend on the seed?

The served batch comes from ONE seeded run of the proposal step (hyperparameter
restarts plus the multi-start acquisition optimizer). Both are stochastic
searches, so a different seed can return a different batch from the same data.
When it does, a scientist reading the served batch as "the model's answer" is
reading one draw of a search, not a property of the data.

This module only COMPARES batches; it never runs an optimizer. The portal
(`kalos.portal.analysis_parts.stability`) reruns the proposal step with extra
seeds and hands the batches here.

Agreement, per served proposal: the fraction of the rerun seeds whose batch
contains a recipe within `tol` of it. Distance is the one the optimizer already
uses for "the same recipe" (`kalos.core.optimize.proposal_separation`): each
continuous factor divided by its design range, the largest per-factor difference
(Chebyshev), and any differing categorical level counts as the full range. So
with the default `tol` = 0.05, "matched" means "some rerun proposed a recipe with
the same levels and no factor more than 5% of its range away", the same rule that
makes two proposals in one batch near-duplicates.

The overall `batch_stability` is the mean agreement over the served proposals.
Both describe the optimizer's seed sensitivity on THIS data; neither is a
statement about whether the proposals are good, or about uncertainty in the
data (resampling the rows is a different question and is not done here).

Torch-free.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

STABILITY_TOL = 0.05
"""Matching tolerance in range-normalized Chebyshev units. Equal to
`kalos.core.optimize.NEAR_DUPLICATE_TOL` (asserted by a test, not imported, so
this module stays torch-free): the engine already treats two recipes closer than
this as the same recipe."""

UNSTABLE_BELOW = 0.5
"""A served proposal found by fewer than half of the rerun seeds is flagged.
With the default 5 seeds (4 reruns) that is 0 or 1 of 4."""

UNSTABLE_LABEL = "unstable across seeds"

SEED_STRIDE = 7919
"""Rerun k uses seed `base + k * SEED_STRIDE`. A prime stride far above the
optimizer's own retry offsets (`seed + attempt`, attempt < 4) so two reruns never
share a seed."""


def seed_schedule(base: int, n_seeds: int) -> list[int]:
    """The `n_seeds` seeds of a stability check; the first is the served seed."""
    return [int(base) + k * SEED_STRIDE for k in range(int(n_seeds))]


def pairwise_separation(
    A: np.ndarray, B: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    cat_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """(len(A), len(B)) matrix of `proposal_separation` distances (vectorized)."""
    A = np.asarray(A, float).reshape(-1, np.asarray(lower).size)
    B = np.asarray(B, float).reshape(-1, np.asarray(lower).size)
    d = A.shape[1]
    cats = sorted({int(c) for c in (cat_dims or [])})
    cont = [j for j in range(d) if j not in set(cats)]
    out = np.zeros((A.shape[0], B.shape[0]))
    if cont:
        width = np.asarray(upper, float)[cont] - np.asarray(lower, float)[cont]
        width = np.where(width > 0, width, 1.0)
        diff = np.abs(A[:, None, cont] - B[None, :, cont]) / width
        out = diff.max(axis=2)
    if cats:
        differ = (np.rint(A[:, None, cats]) != np.rint(B[None, :, cats])).any(axis=2)
        out = np.where(differ, 1.0, out)
    return out


def batch_agreement(
    served: np.ndarray,
    reruns: Sequence[np.ndarray | None],
    lower: np.ndarray,
    upper: np.ndarray,
    cat_dims: Sequence[int] | None = None,
    *,
    tol: float = STABILITY_TOL,
    unstable_below: float = UNSTABLE_BELOW,
) -> dict[str, Any]:
    """Per-proposal agreement of `served` with the rerun batches.

    A rerun that failed (`None`) is not compared and is counted in
    `n_reruns_failed`; agreement is over the reruns that returned a batch. An
    empty rerun batch counts as compared and matches nothing.
    """
    S = np.asarray(served, float)
    lower = np.asarray(lower, float)
    ok = [np.asarray(r, float).reshape(-1, lower.size) for r in reruns if r is not None]
    n_failed = sum(1 for r in reruns if r is None)
    n_cmp = len(ok)
    per: list[dict[str, Any]] = []
    for i in range(S.shape[0] if S.ndim == 2 else 0):
        nearest: list[float | None] = []
        matched = 0
        for R in ok:
            if R.shape[0] == 0:
                nearest.append(None)
                continue
            dist = float(pairwise_separation(S[i:i + 1], R, lower, upper, cat_dims).min())
            nearest.append(round(dist, 4))
            matched += int(dist <= tol)
        agreement = matched / n_cmp if n_cmp else None
        finite = [v for v in nearest if v is not None]
        unstable = agreement is not None and agreement < unstable_below
        per.append({
            "index": i,
            "agreement": None if agreement is None else round(agreement, 4),
            "seeds_matched": matched,
            "seeds_compared": n_cmp,
            "nearest_distance_per_seed": nearest,
            # How far off the misses are: 0.06 (just outside tol) reads very
            # differently from 0.6 (a different region of the design).
            "median_nearest_distance": round(float(np.median(finite)), 4) if finite else None,
            "unstable": bool(unstable),
            "label": UNSTABLE_LABEL if unstable else None,
        })
    values = [p["agreement"] for p in per if p["agreement"] is not None]
    return {
        "batch_stability": round(float(np.mean(values)), 4) if values else None,
        "n_unstable": sum(1 for p in per if p["unstable"]),
        "n_reruns_compared": n_cmp,
        "n_reruns_failed": n_failed,
        "per_proposal": per,
    }


__all__ = [
    "SEED_STRIDE", "STABILITY_TOL", "UNSTABLE_BELOW", "UNSTABLE_LABEL",
    "batch_agreement", "pairwise_separation", "seed_schedule",
]
