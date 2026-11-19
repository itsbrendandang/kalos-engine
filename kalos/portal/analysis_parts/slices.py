"""1D model slices (engine request R2): one sweep per continuous input, read off
the fitted proposal GP, with every other input held at the SAME reference point
the 2D `response_surface` uses.

What a slice is, and is not:
  - Model-only. `mean` is the GP posterior mean of the LATENT response (the
    process, not a new measurement); `lo`/`hi` is a `coverage` central band of
    that latent posterior (`mean -/+ z * sd`, z the normal quantile). This is a
    model-based band (`band_method: "gp_posterior"`), NOT held-out calibrated and
    never conformal. Measured runs are deliberately not part of the payload: the
    UI must not mix predicted and measured values in one field.
  - Not partial dependence in the averaged sense: the other inputs are FIXED at
    the reference recipe (`reference`), not marginalized over the data.

Reference point: the row of `X_fit` (the rows the proposal surrogate was fit
on) with the best `y_fit`, i.e. the incumbent. On the replicate-aware path that
row is a recipe mean (`incumbent_basis: "recipe_mean"`), otherwise a single run
(`"single"`). `reference_point` is shared with the surface builder in
`kalos.portal.viz_utils` so the two can never disagree.

Feature order: driver importance (`drivers`, |Spearman rho| descending - the
same ranking the surface uses to pick its axes), then any remaining continuous
inputs in design order. At most `SLICE_MAX_FEATURES` are sliced; the rest, and
every categorical input, are listed in `slices_skipped` with a machine reason.

Torch-free at import: the surrogate is passed in and only `.posterior` is called.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SLICE_MAX_FEATURES = 12
SLICE_N_POINTS = 41
SLICE_COVERAGE = 0.9
SLICE_NOTE = "model-based band, not held-out calibrated"
_DIGITS = 4  # same rounding as the response surface


def reference_point(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cont_feats: Sequence[Any],
    kept_cats: Sequence[Any],
    cat_dims_map: dict,
    replicate_aware: bool,
    incumbent: float | None = None,
) -> tuple[np.ndarray, dict, str]:
    """The incumbent row of the surrogate's training data, decoded.

    Returns `(vector, recipe, incumbent_basis)`: the encoded design vector (copy),
    the human-readable recipe (continuous rounded, categoricals as level names),
    and `"recipe_mean"` / `"single"`.

    `incumbent` is the best measured value for the campaign's goal (the max, or
    the min when minimizing). The first row equal to it is the reference; with
    no incumbent (or no exact hit) the row with the highest target is used.
    """
    hits = (
        np.flatnonzero(np.asarray(y_fit, dtype=float) == incumbent)
        if incumbent is not None else np.empty(0, dtype=int)
    )
    idx = int(hits[0]) if hits.size else int(np.argmax(y_fit))
    vec = np.asarray(X_fit[idx], dtype=float).copy()
    recipe: dict = {}
    for j, c in enumerate(cont_feats):
        recipe[str(c)] = round(float(vec[j]), _DIGITS)
    for j, c in enumerate(kept_cats):
        if c in cat_dims_map:
            recipe[str(c)] = cat_dims_map[c][int(vec[len(cont_feats) + j])]
    basis = "recipe_mean" if replicate_aware else "single"
    return vec, recipe, basis


def _r(values: np.ndarray) -> list[float]:
    return [round(float(v), _DIGITS) for v in values]


def compute_slices(
    s: Any,  # Surrogate (torch object - caller keeps it alive)
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cont_feats: Sequence[Any],
    kept_cats: Sequence[Any],
    cat_dims_map: dict,
    bounds: np.ndarray,
    drv: Sequence[dict],
    replicate_aware: bool,
    *,
    n_points: int = SLICE_N_POINTS,
    coverage: float = SLICE_COVERAGE,
    max_features: int = SLICE_MAX_FEATURES,
    incumbent: float | None = None,
) -> dict:
    """Build `{"slices", "slices_skipped", "reason"}`.

    `slices` is a list (possibly empty only if nothing continuous exists, in
    which case it is None with `reason` set). `reason` is None on success.
    Never raises: a failure returns `slices: None` with a generic reason.
    """
    skipped: list[dict] = [
        {"feature": str(c), "reason": "categorical"} for c in kept_cats
    ]
    if s is None:
        return {"slices": None, "slices_skipped": skipped,
                "reason": "GP surrogate unavailable for slices"}
    if not cont_feats:
        return {"slices": None, "slices_skipped": skipped,
                "reason": "no continuous features to slice"}

    names = [str(c) for c in cont_feats]
    order: list[int] = []
    for d in drv:
        name = str(d.get("name", ""))
        if name in names and names.index(name) not in order:
            order.append(names.index(name))
    order.extend(j for j in range(len(names)) if j not in order)
    for j in order[max_features:]:
        skipped.append({"feature": names[j], "reason": "feature_cap"})
    order = order[:max_features]

    try:
        ref_vec, ref_recipe, inc_basis = reference_point(
            X_fit, y_fit, cont_feats, kept_cats, cat_dims_map, replicate_aware, incumbent
        )
        reference = {"basis": "incumbent", "incumbent_basis": inc_basis, "recipe": ref_recipe}
        z = NormalDist().inv_cdf(0.5 + coverage / 2.0)
        out: list[dict] = []
        for j in order:
            lo_b, hi_b = float(bounds[0, j]), float(bounds[1, j])
            xs = np.linspace(lo_b, hi_b, n_points)
            pts = np.tile(ref_vec, (n_points, 1))
            pts[:, j] = xs
            mean, sd = s.posterior(pts)  # latent posterior (observation_noise=False)
            mean = np.asarray(mean, dtype=float).reshape(-1)
            sd = np.clip(np.asarray(sd, dtype=float).reshape(-1), 0.0, None)
            if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(sd))):
                skipped.append({"feature": names[j], "reason": "non_finite_posterior"})
                continue
            out.append({
                "feature": names[j],
                "x": _r(xs),
                "mean": _r(mean),
                "sd": _r(sd),
                "lo": _r(mean - z * sd),
                "hi": _r(mean + z * sd),
                "coverage": coverage,
                "band_method": "gp_posterior",
                "band_of": "latent_mean",
                "note": SLICE_NOTE,
                "basis": "predicted",
                "range": [round(lo_b, 6), round(hi_b, 6)],
                "swept_at": "incumbent",
                "reference": reference,
                "n_points": int(n_points),
            })
        return {"slices": out, "slices_skipped": skipped, "reason": None}
    except Exception:
        logger.exception("slice computation failed")
        return {"slices": None, "slices_skipped": skipped,
                "reason": "slice computation failed"}
