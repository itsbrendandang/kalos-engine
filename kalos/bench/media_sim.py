"""Seeded media-optimization campaign: 100 runs over 37 recipes, a scale
column, replicate noise, and a KNOWN optimum.

STRUCTURE ONLY. The table has the shape of a media-optimization dataset from a
prior prototype, shape only (about 100 rows over 37 distinct recipes, so most
recipes are replicated; a culture-scale column).
Every value comes from the latent function below; nothing was read from,
fitted on, or calibrated to any real media data. Component names are generic
(`component_01` ... `component_08`, g/L on made-up ranges).

Latent titer (g/L) at scale `s` (litres, one of `SCALES`):

    titer(x, s) = peak(s) * exp(-0.5 * sum_i ((u_i - c_i) / w_i)^2)
    peak(s)     = 5 * (1 - 0.06 * log10(s / 0.001))

with `u` the components scaled to [0, 1]. Larger scale costs a little titer
but does not move the optimum, so the optimum at `TARGET_SCALE` is exact:
`peak(TARGET_SCALE)` at `u = c`. Replicates of one recipe share the truth and
differ by 8% lognormal assay noise plus a per-batch offset.

A benchmark fixture. Not evidence about any real medium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

N_COMPONENTS = 8
N_RECIPES = 37
N_RUNS = 100
SCALES = (0.001, 0.05, 2.0)  # litres: plate well, shake flask, bench bioreactor
TARGET_SCALE = 2.0
NOISE_CV = 0.08
COMPONENT_COLUMNS = tuple(f"component_{i + 1:02d}" for i in range(N_COMPONENTS))
LOWER = np.zeros(N_COMPONENTS)
UPPER = np.array([20.0, 10.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0])  # g/L, made up


def _layout(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.3, 0.7, N_COMPONENTS), rng.uniform(0.45, 0.65, N_COMPONENTS)


def peak(scale_L: float | np.ndarray) -> np.ndarray:
    return 5.0 * (1.0 - 0.06 * np.log10(np.asarray(scale_L, float) / 0.001))


def true_titer(X: np.ndarray, scale_L, seed: int = 0) -> np.ndarray:
    """Noiseless titer for raw component rows `X` (n x 8) at `scale_L`."""
    c, w = _layout(seed)
    u = (np.atleast_2d(np.asarray(X, float)) - LOWER) / (UPPER - LOWER)
    return peak(scale_L) * np.exp(-0.5 * (((u - c) / w) ** 2).sum(axis=1))


def optimum(seed: int = 0, scale_L: float = TARGET_SCALE) -> tuple[np.ndarray, float]:
    """(raw component vector, titer) of the exact optimum at `scale_L`."""
    c, _ = _layout(seed)
    return LOWER + c * (UPPER - LOWER), float(peak(scale_L))


def make_campaign(seed: int = 0, *, n_recipes: int = N_RECIPES, n_runs: int = N_RUNS) -> pd.DataFrame:
    """Rows: `run_id`, `recipe_id`, `batch`, 8 components, `scale_L`,
    `titer_g_L` (noisy) and `titer_true` (noiseless; scoring only)."""
    rng = np.random.default_rng(seed + 101)
    recipes = LOWER + qmc.LatinHypercube(d=N_COMPONENTS, seed=rng).random(n_recipes) * (UPPER - LOWER)
    recipe_scale = rng.choice(SCALES, size=n_recipes, p=[0.5, 0.3, 0.2])
    # every recipe once, the remaining runs as replicates spread over recipes
    idx = np.concatenate([np.arange(n_recipes), rng.integers(0, n_recipes, n_runs - n_recipes)])
    idx = np.sort(idx)
    batch = rng.integers(0, 6, size=n_runs)
    batch_offset = rng.normal(0.0, 0.03, size=6)
    X = recipes[idx]
    s = recipe_scale[idx]
    y_true = true_titer(X, s, seed)
    y_obs = y_true * np.exp(rng.normal(0.0, NOISE_CV, n_runs) + batch_offset[batch])
    df = pd.DataFrame(X, columns=list(COMPONENT_COLUMNS))
    df.insert(0, "batch", [f"batch_{b}" for b in batch])
    df.insert(0, "recipe_id", [f"recipe_{i:02d}" for i in idx])
    df.insert(0, "run_id", [f"run_{i:03d}" for i in range(n_runs)])
    df["scale_L"] = s
    df["titer_g_L"] = y_obs
    df["titer_true"] = y_true
    return df


__all__ = [
    "COMPONENT_COLUMNS", "N_RUNS", "SCALES", "TARGET_SCALE",
    "make_campaign", "optimum", "peak", "true_titer",
]
