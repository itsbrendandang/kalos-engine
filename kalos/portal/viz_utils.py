"""Kalos portal — visualization payloads for the dashboard.

These are torch-free helpers that compute PCA, Spearman correlation, and a
bounded response surface from the fitted GP. They are called inside `_analyze`
while the GP is still alive, then discarded — no serialization, no persistence.

All visualizations use the `null-with-reason` pattern:
  - When available: the data structure + reason=null
  - When unavailable: None + a reason is logged (caller picks it up)
This matches the existing `cv_logo` / `alternative_scale` pattern.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr

from kalos.core.replicates import GroupKeys, aggregate_replicates
from kalos.portal.analysis_parts.slices import reference_point

logger = logging.getLogger(__name__)


def _embedding_pca(
    X: np.ndarray,
    y: np.ndarray,
    cont_feats: list,
    kept_cats: list,
    cat_dims_map: dict | None = None,
) -> dict | None:
    """Compute a 2D PCA embedding of the design matrix.

    Encodes continuous features as standardized values and categorical features
    as one-hot (excluding the target/outcome columns from encoding).

    Returns a dict with the structure expected by RunLandscape, or None if
    insufficient features/rank exist.

    Uses deterministic full SVD with a fixed sign convention for reproducibility.
    """
    n, d = X.shape

    if d < 2:
        return {"reason": "need at least 2 features for PCA"}

    try:
        # Build encoding: standardized continuous + one-hot categorical
        encoding_parts: list[np.ndarray] = []

        # Continuous features (standardized)
        if cont_feats:
            Xc = X[:, : len(cont_feats)]
            if Xc.shape[1] > 0:
                mean = Xc.mean(axis=0)
                std = Xc.std(axis=0)
                std[std < 1e-12] = 1.0
                Xc_std = (Xc - mean) / std
                encoding_parts.append(Xc_std)

        # One-hot categorical
        if cat_dims_map and kept_cats:
            for c in kept_cats:
                if c in cat_dims_map:
                    levels = cat_dims_map[c]
                    cat_col = X[:, len(cont_feats) + kept_cats.index(c)]
                    for code, _lvl in enumerate(levels):
                        encoding_parts.append((np.rint(cat_col) == code).astype(float))

        if not encoding_parts:
            return {"reason": "no features to encode for PCA"}

        X_enc = np.column_stack(encoding_parts)

        # Check rank
        n_features = X_enc.shape[1]
        if n_features < 2:
            return {"reason": "encoding produces fewer than 2 dimensions for PCA"}

        # Center
        X_centered = X_enc - X_enc.mean(axis=0)

        # Deterministic SVD
        U, S, _Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Fixed sign convention: first column of U has positive sum
        signs = np.sign(U[:, 0].sum())
        if signs == 0:
            signs = 1.0
        U *= signs

        # 2D projection: PC1, PC2
        pc1 = U[:, 0] * S[0]
        pc2 = U[:, 1] * S[1] if S.shape[0] > 1 else np.zeros(n)

        return {
            "method": "pca",
            "encoding": "standardized_continuous_one_hot_categorical",
            "unit": "run",
            "n_rows": int(n),
            "n_features": int(n_features),
            "points": [[round(float(pc1[i]), 4), round(float(pc2[i]), 4), round(float(y[i]), 4)] for i in range(n)],
        }
    except Exception:
        logger.exception("viz computation failed")
        return {"reason": "PCA computation failed"}


def _correlation_spearman(
    X: np.ndarray,
    y: np.ndarray,
    cont_feats: list,
    groups: GroupKeys | None = None,
) -> dict | None:
    """Compute descriptive Spearman correlations on recipe means.

    Continuous features + target only; nominal categories are excluded.
    Returns a square matrix with null for undefined cells.

    When `groups` is given, rows sharing a group key are averaged via
    `aggregate_replicates` before computing pairwise Spearman correlations.
    """
    if len(cont_feats) < 1:
        return {"reason": "no continuous features for correlation"}

    try:
        X_cont = X[:, : len(cont_feats)]
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        n_rows = int(X.shape[0])
        if groups is not None:
            Xr, yr, _rv, _rn = aggregate_replicates(X_cont, y_arr, groups=groups)
            n_units = int(Xr.shape[0])
            features_with_target = np.column_stack([Xr, yr.reshape(-1, 1)])
        else:
            n_units = n_rows
            features_with_target = np.column_stack([X_cont, y_arr.reshape(-1, 1)])

        labels = [str(c) for c in cont_feats] + ["target_mean"]
        n_labels = len(labels)
        matrix: list[list[float | None]] = [[None] * n_labels for _ in range(n_labels)]
        for i in range(n_labels):
            matrix[i][i] = 1.0
            for j in range(i + 1, n_labels):
                valid = np.isfinite(features_with_target[:, i]) & np.isfinite(features_with_target[:, j])
                if valid.sum() < 3:
                    val: float | None = None
                else:
                    rho, _ = spearmanr(features_with_target[valid, i], features_with_target[valid, j])
                    val = round(float(rho), 3) if np.isfinite(rho) else None
                matrix[i][j] = val
                matrix[j][i] = val

        return {
            "unit": "recipe",
            "n_units": n_units,
            "n_rows": n_rows,
            "labels": labels,
            "matrix": matrix,
        }
    except Exception:
        logger.exception("viz computation failed")
        return {"reason": "Spearman correlation computation failed"}


def _response_surface_gp(
    s,  # Surrogate (torch object — caller keeps it alive during this call)
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cont_feats: list,
    kept_cats: list,
    cat_dims_map: dict,
    code_maps: dict,
    bounds: np.ndarray,
    drv: list,  # driver list (already sorted by abs_rho desc)
    cat_dims: list | None,
    incumbent: float,
    replicate_aware: bool,
) -> dict | None:
    """Evaluate a 21x21 surface from the fitted GP.

    Other inputs are fixed at the incumbent (including categorical values).
    Two continuous axes are selected from the top drivers.

    Called BEFORE `del s` in _analyze, so the surrogate is still alive.
    """
    if not cont_feats:
        return {"reason": "no continuous features for surface sweep"}

    if not drv:
        return {"reason": "no drivers to select sweep axes"}

    # Select two varying continuous axes by driver ranking
    # drv is already sorted by abs_rho desc
    sweep_cols: list[tuple[str, int]] = []
    for d in drv[:4]:  # top 4
        name = d["name"]
        if name in cont_feats:
            idx = cont_feats.index(name)
            sweep_cols.append((name, idx))
            if len(sweep_cols) == 2:
                break

    if len(sweep_cols) < 2:
        return {"reason": "fewer than 2 continuous drivers for surface sweep"}

    x_name, x_idx = sweep_cols[0]
    y_name, y_idx = sweep_cols[1]

    try:
        # Build sweep grids from bounds
        x_vals = np.linspace(float(bounds[0, x_idx]), float(bounds[1, x_idx]), 21)
        y_vals = np.linspace(float(bounds[0, y_idx]), float(bounds[1, y_idx]), 21)

        n = len(y_fit)

        # Fixed recipe at the incumbent: the row whose target equals `incumbent`
        # (the best measured value for the campaign's goal: the max, or the min
        # when minimizing). Shared with the 1D slices (analysis_parts/slices.py)
        # so both use the same point.
        incumbent_recipe, fixed_recipe, incumbent_basis = reference_point(
            X_fit, y_fit, cont_feats, kept_cats, cat_dims_map, replicate_aware, incumbent
        )

        # Evaluate surface in a single batched posterior call (row-major: i over y_vals, j over x_vals)
        grid = np.tile(incumbent_recipe, (21 * 21, 1))
        xv_grid, yv_grid = np.meshgrid(x_vals, y_vals)
        grid[:, x_idx] = xv_grid.ravel()
        grid[:, y_idx] = yv_grid.ravel()
        mean_all, sd_all = s.posterior(grid)
        z = np.asarray(mean_all, dtype=float).reshape(21, 21)
        std = np.asarray(sd_all, dtype=float).reshape(21, 21)

        # Also collect observed runs at the swept points
        runs = [[round(float(X_fit[k, x_idx]), 4), round(float(X_fit[k, y_idx]), 4), round(float(y_fit[k]), 4)] for k in range(n)]

        return {
            "x_feature": x_name,
            "y_feature": y_name,
            "x_vals": [round(float(v), 4) for v in x_vals],
            "y_vals": [round(float(v), 4) for v in y_vals],
            "z": [[round(float(z[i, j]), 4) for j in range(21)] for i in range(21)],
            # R3: GP posterior SD of the LATENT mean (observation_noise=False),
            # same grid/shape as z. Model-based, not held-out calibrated.
            "std": [[round(float(std[i, j]), 4) for j in range(21)] for i in range(21)],
            "std_kind": "gp_posterior_sd",
            "std_basis": "latent_mean",
            "z_basis": "predicted",
            "runs": runs,
            "swept_at": "incumbent",
            "fixed_recipe": fixed_recipe,
            "incumbent_basis": incumbent_basis,
        }
    except Exception:
        logger.exception("viz computation failed")
        return {"reason": "Response surface computation failed"}
