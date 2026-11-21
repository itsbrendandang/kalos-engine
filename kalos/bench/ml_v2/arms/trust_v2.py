"""ml-trust candidates (research/trust-v2): engine-patch arms for the trust layer.

Every piece here is an in-process patch of an engine internal (PROTOCOL.md 6.1); the
production `_analyze` path runs around it. Pieces:

- `served_oof` (audit S2): grouped-CV folds fit the way the SERVED model is fit
  (multi-start hyperparameters with the proposal's restart rule and seed; the
  replicate-aware recipe-mean fit with fixed per-recipe noise when the fold's training
  rows are replicated, scored against one new measurement with latent + assay-floor sd).
  Patch point: `kalos.core.evaluation._oof`.
"""
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..adapter import patched
from .kalos_ref import KalosEngineArm


# ------------------------------------------------------------------ S2: served-config CV
def served_oof(
    X, y, groups, n_splits, bounds, cat_dims=None, shuffle: bool = False, random_state: int = 42,
    index_out: list[int] | None = None, fold_out: list[int] | None = None,
):
    """Drop-in for `kalos.core.evaluation._oof` whose fold models match the served one."""
    from kalos.core.replicates import aggregate_replicates, noise_report
    from kalos.core.splits import make_splits, row_hash_groups
    from kalos.core.surrogate import Surrogate, proposal_restarts
    from kalos.portal.analysis_parts.policy import ANALYZE_SEED

    import kalos.portal.analysis as an

    requested = int(getattr(an, "PROPOSAL_FIT_RESTARTS", 3))
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    if groups is None:
        groups = row_hash_groups(X)
    groups = np.asarray(groups)
    if bounds is None and len(X):
        bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    pred, sd, actual, grp, n_folds = [], [], [], [], 0
    for tr, te in make_splits(X, y, groups, n_splits=n_splits, shuffle=shuffle, random_state=random_state):
        if len(tr) < 4 or len(te) < 1:
            continue
        n_folds += 1
        Xtr, ytr = X[tr], y[tr]
        nr = noise_report(Xtr, ytr)
        s2 = float(nr["noise_var"])
        if nr["n_replicated"] >= 1 and nr["n_recipes"] >= 6 and np.isfinite(s2) and s2 > 0:
            Xf, yf, _v, nrep = aggregate_replicates(Xtr, ytr)
            s = Surrogate().fit(Xf, yf, bounds=bounds, noise=s2 / np.maximum(nrep, 1).astype(float),
                                cat_dims=cat_dims, restarts=proposal_restarts(len(yf), Xf.shape[1], requested),
                                restart_seed=ANALYZE_SEED)
            mean, lat = s.posterior(X[te])
            std = np.sqrt(np.asarray(lat, float) ** 2 + s2)
        else:
            s = Surrogate().fit(Xtr, ytr, bounds=bounds, cat_dims=cat_dims,
                                restarts=proposal_restarts(len(ytr), Xtr.shape[1], requested),
                                restart_seed=ANALYZE_SEED)
            mean, std = s.posterior(X[te], observation_noise=True)
        pred.extend(np.asarray(mean).ravel().tolist())
        sd.extend(np.asarray(std).ravel().tolist())
        actual.extend(y[te].tolist())
        grp.extend(groups[te].tolist())
        if index_out is not None:
            index_out.extend(int(i) for i in te)
        if fold_out is not None:
            fold_out.extend([n_folds - 1] * len(te))
    return (np.asarray(pred), np.asarray(sd), np.asarray(actual), np.asarray(grp, dtype=object), n_folds)


# ------------------------------------------------- hidden batch / drift from run order
ORDER_ALPHA = 0.01
ORDER_PERMUTATIONS = 999
ORDER_SEED = 20261120
SEGMENT_MIN_SIZE = 3


def lag1_order_test(y, *, n_perm: int = ORDER_PERMUTATIONS, seed: int = ORDER_SEED) -> dict[str, Any]:
    """Are consecutive runs (sheet order) more alike than chance? One-sided permutation
    test of the lag-1 autocorrelation of the target in row order. Deterministic."""
    v = np.asarray(y, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 8 or np.ptp(v) == 0:
        return {"evaluable": False, "n": n, "r1": None, "p": None}
    z = v - v.mean()
    den = float(z @ z)
    r1 = float(z[:-1] @ z[1:]) / den
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_perm, n)), axis=1)
    zp = z[perms]
    null = np.einsum("ij,ij->i", zp[:, :-1], zp[:, 1:]) / den
    p = float((1 + np.sum(null >= r1)) / (n_perm + 1))
    return {"evaluable": True, "n": n, "r1": r1, "p": p}


def order_segments(y, *, min_size: int = SEGMENT_MIN_SIZE, penalty: float | None = None) -> np.ndarray:
    """Piecewise-constant segmentation of the target in row order (optimal partitioning,
    BIC-type penalty 2 sigma^2 log n, sigma from the MAD of first differences, so a
    jump between runs, not the spread of the whole sheet, sets the scale). Returns a
    segment label per row."""
    v = np.asarray(y, float)
    n = len(v)
    if n < 2 * min_size:
        return np.zeros(n, int)
    d = np.diff(v)
    sig = float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.std(v)) or 1.0
    pen = penalty if penalty is not None else 2.0 * sig ** 2 * np.log(n)
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, int)
    for t in range(min_size, n + 1):
        s = np.arange(0, t - min_size + 1)
        s = s[np.isfinite(F[s])]
        if not len(s):
            continue
        m = t - s
        cost = (c2[t] - c2[s]) - (c1[t] - c1[s]) ** 2 / m
        tot = F[s] + cost + pen
        k = int(np.argmin(tot))
        F[t], last[t] = tot[k], s[k]
    cuts = []
    t = n
    while t > 0:
        cuts.append(t)
        t = last[t]
    cuts = sorted(cuts)
    lab = np.zeros(n, int)
    start = 0
    for i, c in enumerate(cuts):
        lab[start:c] = i
        start = c
    return lab


LAST_ORDER_CHECK: dict[str, Any] = {}


def run_order_candidates(orig):
    """Wrap `structure_candidates`: when no campaign round index reaches the analysis
    and consecutive runs are alike beyond chance (lag-1 test at ORDER_ALPHA), the
    run-order segments become one more grouping candidate (held out together), so a
    model whose skill came from neighbours in time is caught by the lowest-score rule."""
    from kalos.portal.analysis_parts.cv_grouping import MIN_GROUPS_FOR_CV, _n_groups, _union

    def wrapped(frame, *, exclude, round_index, clusters):
        out = orig(frame, exclude=exclude, round_index=round_index, clusters=clusters)
        LAST_ORDER_CHECK.clear()
        if round_index is not None:
            LAST_ORDER_CHECK.update(skipped="campaign_round_index_present")
            return out
        tcol = [c for c in exclude if c in frame.columns][:1]
        if not tcol:
            return out
        yv = pd_numeric(frame[tcol[0]])
        test = lag1_order_test(yv)
        LAST_ORDER_CHECK.update(test)
        if not test["evaluable"] or test["p"] > ORDER_ALPHA:
            return out
        seg = order_segments(np.where(np.isfinite(yv), yv, np.nanmedian(yv)))
        k = len(set(seg.tolist()))
        LAST_ORDER_CHECK["n_segments"] = k
        if k < MIN_GROUPS_FOR_CV:
            return out
        lab = [f"seg{s}" for s in seg]
        merged = _union(lab, clusters.tolist())
        use_merged = _n_groups(merged) >= MIN_GROUPS_FOR_CV
        out.append({"basis": "run_order", "column": "run_order_segments", "labels": lab,
                    "groups": merged if use_merged else np.asarray(seg), "n_levels": k,
                    "merged_with_replicates": bool(use_merged)})
        return out

    return wrapped


def pd_numeric(s) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(s, errors="coerce").to_numpy(float)


class TrustV2Arm(KalosEngineArm):
    """Config keys select pieces: {"cv": "served", "order": true} ..."""

    name = "trust_v2"

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        import kalos.core.evaluation as ev
        import kalos.portal.analysis as an

        stack = contextlib.ExitStack()
        if self.config.get("cv") == "served":
            stack.enter_context(patched(ev, "_oof", served_oof))
        if self.config.get("order"):
            stack.enter_context(patched(an, "structure_candidates", run_order_candidates(an.structure_candidates)))
        return stack


def make_arm(config: Mapping[str, Any] | None = None) -> TrustV2Arm:
    return TrustV2Arm(config)
