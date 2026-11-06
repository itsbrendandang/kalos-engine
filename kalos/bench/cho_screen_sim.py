"""Seeded CHO screen: 109 runs x 49 features with a KNOWN optimum.

STRUCTURE ONLY. The table has the shape of a CHO screening dataset (109 rows,
49 numeric process/media features, one titer outcome) as described in
`/tmp/redesign/research/bioqore-mine.md`. Every value comes from the latent
function below; nothing was read from, fitted on, or calibrated to any real
screen. Column names are generic (`feature_01` ... `feature_49`), ranges are
the unit interval.

Latent titer (g/L), sparse like a real screen (most features do nothing):

    titer(x) = 1 + 0.5 * x_5 + 6 * exp(-0.5 * sum_{i in A} ((x_i - c_i) / w_i)^2)

over 4 active features `A` plus one weak linear driver (feature 5), so the
optimum is exact: 7.5 at `x_A = c`, `x_5 = 1`, any value elsewhere. Observed
titer adds 5% lognormal assay noise. The 109 screen rows are a Latin
hypercube over the 49-d box.

A benchmark fixture. Not evidence about any real process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .objectives import Objective

N_ROWS = 109
N_FEATURES = 49
FEATURE_COLUMNS = tuple(f"feature_{i + 1:02d}" for i in range(N_FEATURES))
NOISE_CV = 0.05
_LINEAR = 4  # feature_05


def _layout(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    active = np.sort(rng.choice([i for i in range(N_FEATURES) if i != _LINEAR], size=4, replace=False))
    centers = rng.uniform(0.3, 0.7, size=4)
    widths = rng.uniform(0.2, 0.3, size=4)
    return active, centers, widths


def cho_screen_objective(seed: int = 0) -> Objective:
    """The latent surface as an `Objective` (49-d unit box). `optimum` = 7.5."""
    active, centers, widths = _layout(seed)

    def f(X: np.ndarray) -> np.ndarray:
        z = ((X[:, active] - centers) / widths) ** 2
        return 1.0 + 0.5 * X[:, _LINEAR] + 6.0 * np.exp(-0.5 * z.sum(axis=1))

    bounds = np.vstack([np.zeros(N_FEATURES), np.ones(N_FEATURES)])
    obj = Objective(f"cho_screen_49d_s{seed}", N_FEATURES, bounds, 7.5, 6.0, f)
    return obj


def optimum_point(seed: int = 0) -> np.ndarray:
    active, centers, _ = _layout(seed)
    x = np.full(N_FEATURES, 0.5)
    x[active] = centers
    x[_LINEAR] = 1.0
    return x


def make_screen(seed: int = 0, *, n_rows: int = N_ROWS) -> pd.DataFrame:
    """The 109-row screen: `run_id`, 49 features, `titer_g_L` (noisy) and
    `titer_true` (noiseless, for scoring only; never a model input)."""
    obj = cho_screen_objective(seed)
    rng = np.random.default_rng(seed + 1)
    X = qmc.LatinHypercube(d=N_FEATURES, seed=rng).random(n_rows)
    y_true = obj.evaluate(X)[0]
    y_obs = y_true * np.exp(rng.normal(0.0, NOISE_CV, n_rows))
    df = pd.DataFrame(X, columns=list(FEATURE_COLUMNS))
    df.insert(0, "run_id", [f"run_{i:03d}" for i in range(n_rows)])
    df["titer_g_L"] = y_obs
    df["titer_true"] = y_true
    return df


__all__ = ["FEATURE_COLUMNS", "N_FEATURES", "N_ROWS", "cho_screen_objective", "make_screen", "optimum_point"]
