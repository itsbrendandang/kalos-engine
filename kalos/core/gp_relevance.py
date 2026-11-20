"""ARD relevance of each input, with an uncertainty, from the GP that proposes.

`kalos.core.gp_shape` reads per-input relevance off the proposal GP's ARD
lengthscales. A point value alone cannot say whether "the GP ignores this input"
is a property of the data or of this particular sample of runs, so this module
adds an interval by SUBSAMPLING RECIPES AND REFITTING:

- `B` subsamples, each keeping `FRACTION` of the recipes (replicates of one
  recipe stay together), drawn without replacement from a seeded Generator;
- each subsample is fit with the served GP's configuration (bounds, fixed
  noise when the served fit used it) from the default start, and its ARD
  lengthscales are read;
- the interval is the central `LEVEL` range of the B lengthscales (and of the
  relevance shares, inverse lengthscales normalized to sum to 1).

Why subsampling and not a bootstrap or a Laplace approximation (measured on a
5-input simulator with 2 relevant inputs, 15-40 seeds per cell, n = 20/40/80;
docs/CAMPAIGNS_API.md "Driver second opinion"):
- A Laplace approximation at the fitted hyperparameters is nearly free, but it
  called a truly irrelevant input "relevant" in 1-43% of cases (overconfident
  toward short lengthscales); subsampling did so in 0-16%, with more power on
  the relevant inputs.
- A with-replacement bootstrap duplicates rows, which a GP reads as exact
  replicates and which biases the fitted noise and lengthscales.

Status per input (named thresholds, in units of the design range: inputs are
normalized to the design box, so lengthscales are comparable across inputs):
- `relevant` when the whole interval is at or below `RELEVANT_MAX_LS` (the
  fitted response changes by about a standard deviation or more across the
  range);
- `negligible` when the whole interval is at or above `NEGLIGIBLE_MIN_LS`;
- `unclear` otherwise.

What it is not: a causal effect. Relevance is how fast the FITTED response
changes along an input; collinear inputs can share it arbitrarily, and a model
that cannot predict held-out runs fits short lengthscales to noise (the caller
gates on the out-of-fold score for that reason).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

LEVEL = 0.95
N_SUBSAMPLES = 16
FRACTION = 0.8
MIN_RECIPES = 8
RELEVANT_MAX_LS = 1.0
NEGLIGIBLE_MIN_LS = 2.0


def _fit_lengthscales(
    X: np.ndarray, y: np.ndarray, bounds: np.ndarray, noise: np.ndarray | None, d: int,
) -> np.ndarray | None:
    """Module-level (picklable) task: one single-start fit, its ARD lengthscales."""
    from kalos.core.gp_shape import ard_lengthscales
    from kalos.core.surrogate import Surrogate

    try:
        s = Surrogate().fit(X, y, bounds=bounds, noise=noise)
    except Exception:  # a failed subsample is counted, never fatal
        return None
    return ard_lengthscales(s, d)


def _run(fn: Callable[..., Any], tasks: list[tuple]) -> list[Any]:
    try:
        from kalos.core import parallel  # perf/parallel-analysis
    except ImportError:
        return [fn(*t) for t in tasks]
    return list(parallel.run_ordered(fn, tasks))


def status_from_interval(lo: float, hi: float) -> str:
    if hi <= RELEVANT_MAX_LS:
        return "relevant"
    if lo >= NEGLIGIBLE_MIN_LS:
        return "negligible"
    return "unclear"


def subsample_relevance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    bounds: np.ndarray,
    groups: Sequence[Any] | None = None,
    noise: np.ndarray | float | None = None,
    point_lengthscales: np.ndarray | None = None,
    n_subsamples: int = N_SUBSAMPLES,
    fraction: float = FRACTION,
    level: float = LEVEL,
    seed: int = 0,
) -> dict[str, Any]:
    """Intervals for ARD lengthscales and relevance shares (continuous GP only).

    `point_lengthscales` are the served fit's (reported as the point value).
    Fits run inside a forked torch RNG with numpy's global state restored, so
    calling this never changes any other number of an analysis.
    """
    import torch

    Xa = np.asarray(X, float)
    ya = np.asarray(y, float).reshape(-1)
    n, d = Xa.shape
    g = np.asarray(list(groups) if groups is not None else np.arange(n))
    uniq = list(dict.fromkeys(g.tolist()))
    pos = {v: i for i, v in enumerate(uniq)}
    codes = np.array([pos[v] for v in g.tolist()], dtype=int)
    if len(uniq) < MIN_RECIPES:
        return {"available": False, "reason": (
            f"{len(uniq)} distinct recipes; subsampling needs at least {MIN_RECIPES}")}
    noise_arr = None if noise is None else np.broadcast_to(np.asarray(noise, float), (n,)).copy()
    rng = np.random.default_rng(seed)
    keep = max(MIN_RECIPES - 1, int(round(fraction * len(uniq))))
    tasks = []
    for _b in range(int(n_subsamples)):
        chosen = rng.choice(len(uniq), size=keep, replace=False)
        mask = np.isin(codes, chosen)
        tasks.append((Xa[mask], ya[mask], np.asarray(bounds, float),
                      None if noise_arr is None else noise_arr[mask], d))
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            fits = _run(_fit_lengthscales, tasks)
    finally:
        np.random.set_state(np_state)
    L = np.array([f for f in fits if f is not None], dtype=float).reshape(-1, d)
    n_ok = int(L.shape[0])
    if n_ok < max(4, n_subsamples // 2):
        return {"available": False, "reason": (
            f"only {n_ok} of {n_subsamples} subsample fits gave per-input lengthscales")}
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    inv = 1.0 / L
    shares = inv / inv.sum(axis=1, keepdims=True)
    ls_ci = np.quantile(L, [lo_q, hi_q], axis=0)
    sh_ci = np.quantile(shares, [lo_q, hi_q], axis=0)
    pt = np.asarray(point_lengthscales, float) if point_lengthscales is not None else np.median(L, axis=0)
    pt_share = (1 / pt) / (1 / pt).sum()
    return {
        "available": True,
        "reason": None,
        "method": "recipe_subsample_refit",
        "level": level,
        "n_subsamples": int(n_subsamples),
        "n_fits_used": n_ok,
        "fraction": fraction,
        "lengthscale": [round(float(v), 4) for v in pt],
        "lengthscale_ci": [[round(float(ls_ci[0, j]), 4), round(float(ls_ci[1, j]), 4)] for j in range(d)],
        "relevance": [round(float(v), 4) for v in pt_share],
        "relevance_ci": [[round(float(sh_ci[0, j]), 4), round(float(sh_ci[1, j]), 4)] for j in range(d)],
        "status": [status_from_interval(float(ls_ci[0, j]), float(ls_ci[1, j])) for j in range(d)],
    }


__all__ = [
    "FRACTION", "LEVEL", "MIN_RECIPES", "NEGLIGIBLE_MIN_LS", "N_SUBSAMPLES", "RELEVANT_MAX_LS",
    "status_from_interval", "subsample_relevance",
]
