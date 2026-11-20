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

from kalos.analysis.viz_fields import pairwise_complete_n
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
                    for lvl in levels:
                        encoding_parts.append((cat_col == lvl).astype(float))

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
) -> dict | None:
    """Descriptive pairwise Spearman correlations over the fitted ROWS (runs).

    Continuous features + target only; nominal categories are excluded.
    Returns a square matrix with null for undefined cells (fewer than 3
    complete pairs, or a constant column). `unit` is "row": every row of `X`
    counts once (replicates are NOT averaged into recipe means). `n_matrix`
    is the pairwise complete-case n behind each cell
    (`kalos.analysis.viz_fields.pairwise_complete_n`, VIZ_SPEC_V2 E5).
    """
    if len(cont_feats) < 1:
        return {"reason": "no continuous features for correlation"}

    try:
        X_cont = X[:, : len(cont_feats)]
        # Compute recipe means via groupby
        # Use the deterministic row-hash groups key from the analysis
        # For correlation, we average across all rows (simple mean)
        recipe_means = []
        for j in range(len(cont_feats)):
            recipe_means.append(X_cont[:, j].mean())
        recipe_means.append(y.mean())

        labels = [str(c) for c in cont_feats] + ["target_mean"]

        # Pairwise Spearman on the recipe means (single point per recipe)
        # For a single observation, we compute Spearman on the continuous features + target
        # This is a descriptive matrix of feature correlations
        n_labels = len(labels)
        matrix: list[list] = []
        for i in range(n_labels):
            row: list[float | None] = []
            for j in range(n_labels):
                if i == j:
                    row.append(1.0)
                elif j < i:
                    # Symmetric — should have been computed already
                    row.append(matrix[j][i])
                else:
                    # Compute correlation between feature i and feature j
                    # Use the raw values (not recipe means) for descriptive correlation
                    row.append(None)  # placeholder, fill below
            matrix.append(row)

        # Actually compute pairwise — use all rows for descriptive feature+target correlations
        features_with_target = np.column_stack([X_cont, y.reshape(-1, 1)])
        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                ci, cj = i, j
                valid = np.isfinite(features_with_target[:, ci]) & np.isfinite(features_with_target[:, cj])
                if valid.sum() < 3:
                    matrix[i][j] = None
                    matrix[j][i] = None
                else:
                    rho, _ = spearmanr(features_with_target[valid, ci], features_with_target[valid, cj])
                    if not np.isfinite(rho):
                        matrix[i][j] = None
                        matrix[j][i] = None
                    else:
                        val = round(float(rho), 3)
                        matrix[i][j] = val
                        matrix[j][i] = val

        return {
            # E5: the label matches the computation (raw rows, not recipe means).
            "unit": "row",
            "n_units": int(X.shape[0]),
            "n_rows": int(X.shape[0]),
            "labels": labels,
            "matrix": matrix,
            "n_matrix": pairwise_complete_n(features_with_target),
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
    return surface_grid(
        s, X_fit, y_fit, cont_feats, kept_cats, cat_dims_map, bounds,
        (x_name, x_idx), (y_name, y_idx), incumbent, replicate_aware,
    )


def surface_grid(
    s,  # Surrogate, alive for the call
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cont_feats: list,
    kept_cats: list,
    cat_dims_map: dict,
    bounds: np.ndarray,
    x_axis: tuple[str, int],
    y_axis: tuple[str, int],
    incumbent: float,
    replicate_aware: bool,
) -> dict:
    """The 21x21 GP mean/std grid over one pair of continuous inputs (`x_axis`,
    `y_axis` = (name, column index)), every other input at the incumbent recipe.
    Shared by `response_surface` (the top pair) and `response_surfaces[]`
    (`kalos.analysis.viz_fields`), so every pair has the same grid, rounding and
    labels."""
    x_name, x_idx = x_axis
    y_name, y_idx = y_axis
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

        # Evaluate surface
        z = np.zeros((21, 21))
        std = np.zeros((21, 21))

        for i, yv in enumerate(y_vals):
            for j, xv in enumerate(x_vals):
                # Build the point: sweep cols replaced, others at incumbent
                point = incumbent_recipe.copy()
                point[x_idx] = xv
                point[y_idx] = yv

                mean, sd = s.posterior(point.reshape(1, -1))
                z[i, j] = float(mean[0])
                std[i, j] = float(sd[0])

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
