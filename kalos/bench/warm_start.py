"""Three-regime benchmark for cross-campaign warm start (`kalos.core.multitask`).

Synthetic campaigns share a response SHAPE (one Gaussian peak, jittered a
little per campaign) and differ in LEVEL and AMPLITUDE, which is the
multi-campaign situation the warm start is for. The target campaign is held
out and predicted in three regimes:

- `within`: the target campaign's own runs are mostly in training (12 of 20);
  predict the rest. Its level is known, so pooling has the least to add.
- `new`: the target campaign has ZERO runs. No model can know its level; a
  multi-task GP cannot even address an unseen task, so the arm reported is the
  pooled single GP (task ignored) against the training mean. Expect little.
- `warm_k`: reveal k runs of the target campaign, predict the rest. This is the
  warm-start claim: the multi-task GP should beat a single-task GP fitted on
  the same k runs alone (the cold start) when k is small.

Arms: `st_target` (single-task GP on target rows only), `pooled` (single GP on
every row, campaign ignored), `mtgp` (MultiTaskGP, ICM kernel). Metrics on the
target campaign's held-out rows: RMSE and Spearman. Made-up surfaces; not
evidence about any real process.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import qmc, spearmanr

D = 3
BOUNDS = np.vstack([np.zeros(D), np.ones(D)])


def make_campaigns(seed: int, *, n_campaigns: int = 4, n_runs: int = 20, noise: float = 0.05):
    """`(X, y, campaign)` for `n_campaigns` campaigns; campaign 0 is the target."""
    rng = np.random.default_rng(seed)
    center = rng.uniform(0.3, 0.7, D)
    Xs, ys, cs = [], [], []
    for c in range(n_campaigns):
        level = rng.uniform(-2.0, 2.0)
        amp = rng.uniform(3.0, 6.0)
        cc = center + rng.normal(0.0, 0.03, D)
        X = qmc.LatinHypercube(d=D, seed=rng).random(n_runs)
        f = level + amp * np.exp(-6.0 * ((X - cc) ** 2).sum(axis=1))
        ys.append(f + rng.normal(0.0, noise * amp, n_runs))
        Xs.append(X)
        cs.extend([c] * n_runs)
    return np.vstack(Xs), np.concatenate(ys), np.asarray(cs)


def _metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    rho = spearmanr(pred, actual).statistic if np.ptp(pred) > 0 and np.ptp(actual) > 0 else np.nan
    return {"rmse": rmse, "spearman": float(rho)}


def _arms(X, y, c, train, test, *, target: int = 0) -> dict:
    from kalos.core.multitask import MultiTaskSurrogate
    from kalos.core.surrogate import FitError, Surrogate

    out: dict[str, dict] = {}
    Xte, yte = X[test], y[test]
    tr_t = train & (c == target)
    if tr_t.sum() >= 2:
        try:
            st = Surrogate().fit(X[tr_t], y[tr_t], bounds=BOUNDS)
            out["st_target"] = _metrics(st.posterior(Xte)[0], yte)
        except FitError:
            pass
    pooled = Surrogate().fit(X[train], y[train], bounds=BOUNDS)
    out["pooled"] = _metrics(pooled.posterior(Xte)[0], yte)
    out["train_mean"] = {"rmse": float(np.sqrt(np.mean((y[train].mean() - yte) ** 2))), "spearman": float("nan")}
    if tr_t.sum() >= 1:
        mt = MultiTaskSurrogate().fit(X[train], y[train], c[train], BOUNDS, target_task=target)
        out["mtgp"] = _metrics(mt.posterior(Xte)[0], yte)
    return out


def run_regimes(seed: int, *, ks: Iterable[int] = (2, 4, 8)) -> dict:
    X, y, c = make_campaigns(seed)
    rng = np.random.default_rng(seed + 7)
    tgt = np.flatnonzero(c == 0)
    order = rng.permutation(tgt)
    res: dict[str, dict] = {}
    # within: 12 target runs known
    train = np.ones(len(y), bool)
    train[order[12:]] = False
    res["within"] = _arms(X, y, c, train, ~train)
    # new: no target runs
    train = c != 0
    res["new"] = _arms(X, y, c, train, ~train)
    for k in ks:
        train = c != 0
        train[order[:k]] = True
        test = np.zeros(len(y), bool)
        test[order[k:]] = True
        res[f"warm_{k}"] = _arms(X, y, c, train, test)
    return res


def warm_start_benchmark(seeds: Iterable[int] = range(5), *, ks: Iterable[int] = (2, 4, 8)) -> dict:
    """Mean RMSE / Spearman per regime and arm over `seeds`."""
    ks = tuple(ks)
    runs = [run_regimes(s, ks=ks) for s in seeds]
    out: dict[str, dict] = {}
    for regime in runs[0]:
        arms = sorted({a for r in runs for a in r[regime]})
        out[regime] = {
            a: {
                m: float(np.nanmean([r[regime][a][m] for r in runs if a in r[regime]]))
                for m in ("rmse", "spearman")
            }
            for a in arms
        }
    return out


__all__ = ["make_campaigns", "run_regimes", "warm_start_benchmark"]
