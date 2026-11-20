"""Additive visualization fields for kalos-web heat maps (VIZ_SPEC_V2 E3, E5, E12).

Three fields, all descriptive or model-only, none of them feeding a proposal,
an interval, a verdict or any other number in the analysis:

- E5 `correlation.n_matrix`: the pairwise complete-case n behind each cell of the
  Spearman matrix (`pairwise_complete_n`, called from
  `kalos.portal.viz_utils._correlation_spearman`, which also labels the matrix
  `unit: "row"`: it is computed over the fitted rows, replicates not averaged).
- E12 `categorical_levels`: `{column: [level, ...]}` for every categorical input
  and the clone column, listing every non-blank label the uploaded sheet holds
  in that column, INCLUDING levels whose runs were all dropped (no usable
  target, a missing cell) and so have 0 fitted runs. Sorted, the order the
  engine encodes levels in. A level with 0 runs is listed, never invented: a
  label that is not in the sheet is not here (e.g. a clone only in the registry;
  `GET /api/campaigns/{id}/clones/summary` lists those).
- E3 `response_surfaces[]`: the same 21x21 GP mean / std grid as
  `response_surface` (one shared builder, `viz_utils.surface_grid`: same grid,
  rounding, labels `z_basis: "predicted"`, `std_kind: "gp_posterior_sd"`,
  `std_basis: "latent_mean"`, other inputs at the incumbent recipe) for the top
  `k` pairs of continuous drivers. Model-only: the std is the latent GP
  posterior SD, not a held-out calibrated band.

Determinism: nothing here draws from any RNG (no torch, numpy or `random`
state is consumed), so every number computed after this block in
`_analyze_body` is unchanged. The surrogate is only read (`posterior`).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos.envconfig import env_int

logger = logging.getLogger(__name__)

# E3: number of driver pairs returned in `response_surfaces`. Default 3 (the top
# three continuous drivers give exactly 3 pairs). `KALOS_VIZ_SURFACE_PAIRS=0`
# turns the field off (an empty list with a reason); max 6 (top four drivers).
SURFACE_PAIRS_ENV = "KALOS_VIZ_SURFACE_PAIRS"
SURFACE_PAIRS_DEFAULT = 3
SURFACE_PAIRS_MAX = 6
PAIR_RULE = (
    "pairs of continuous drivers in `drivers` order: (1,2), (1,3), (2,3), (1,4), (2,4), (3,4); "
    "the first is the `response_surface` pair"
)


def surface_pairs_k() -> int:
    """`KALOS_VIZ_SURFACE_PAIRS` (0..6), default 3. Out-of-range values raise."""
    return env_int(SURFACE_PAIRS_ENV, SURFACE_PAIRS_DEFAULT, minimum=0, maximum=SURFACE_PAIRS_MAX)


# --- E5 ---------------------------------------------------------------------- #


def pairwise_complete_n(columns: np.ndarray) -> list[list[int]]:
    """Square matrix: entry (i, j) is the number of rows where columns i and j
    are both finite; the diagonal is each column's own finite count."""
    finite = np.isfinite(np.asarray(columns, dtype=float))
    counts = finite.T.astype(np.int64) @ finite.astype(np.int64)
    return [[int(v) for v in row] for row in counts]


# --- E12 --------------------------------------------------------------------- #


def categorical_levels(df: pd.DataFrame, columns: Sequence[Any]) -> dict[str, list[str]]:
    """Every non-blank label per column over ALL rows of the sheet (rows with no
    usable target included), normalized and sorted the way the engine builds its
    categorical levels (`str(...).strip()`, null tokens dropped)."""
    from kalos.portal.cells import is_null_label

    out: dict[str, list[str]] = {}
    for c in columns:
        if c is None or c not in df.columns or str(c) in out:
            continue
        labels = df[c].fillna("").astype(str).str.strip()
        out[str(c)] = sorted({v for v in labels.tolist() if not is_null_label(v)})
    return out


# --- E3 ---------------------------------------------------------------------- #


def driver_pairs(drv: Sequence[Mapping[str, Any]], cont_feats: Sequence[Any], k: int) -> list[tuple[int, int]]:
    """The first `k` pairs (positions in `cont_feats`) of continuous drivers in
    driver order: (1,2), (1,3), (2,3), (1,4), ... (`PAIR_RULE`)."""
    names = [str(c) for c in cont_feats]
    ranked: list[int] = []
    for d in drv:
        name = str(d.get("name"))
        if name in names and names.index(name) not in ranked:
            ranked.append(names.index(name))
    pairs: list[tuple[int, int]] = []
    for b in range(1, len(ranked)):
        for a in range(b):
            if len(pairs) < k:
                pairs.append((ranked[a], ranked[b]))
    return pairs


def response_surfaces(
    s: Any,
    *,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cont_feats: Sequence[Any],
    kept_cats: Sequence[Any],
    cat_dims_map: Mapping[Any, Any],
    bounds: np.ndarray,
    drv: Sequence[Mapping[str, Any]],
    incumbent: float,
    replicate_aware: bool,
    first: Mapping[str, Any] | None,
    k: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """(`response_surfaces`, reason). The reason is None when `k` pairs were
    returned, else why fewer. `first` (the existing `response_surface`) is
    reused for its own pair rather than recomputed."""
    from kalos.portal.viz_utils import surface_grid

    if k <= 0:
        return [], f"turned off ({SURFACE_PAIRS_ENV}=0)"
    pairs = driver_pairs(drv, cont_feats, k)
    names = [str(c) for c in cont_feats]
    out: list[dict[str, Any]] = []
    failed = 0
    for rank, (xi, yi) in enumerate(pairs, start=1):
        if first is not None and first.get("x_feature") == names[xi] and first.get("y_feature") == names[yi]:
            grid: dict[str, Any] = dict(first)
        else:
            grid = surface_grid(
                s, X_fit, y_fit, list(cont_feats), list(kept_cats), dict(cat_dims_map), bounds,
                (names[xi], xi), (names[yi], yi), incumbent, replicate_aware,
            )
        if "reason" in grid:
            failed += 1
            continue
        out.append({"pair_rank": rank, **grid})
    if len(out) == k:
        return out, None
    if not pairs:
        return out, "fewer than 2 continuous drivers for a surface pair"
    why = f"{len(out)} of {k} pairs returned"
    if len(pairs) < k:
        why += f": only {len(pairs)} pair(s) from the continuous drivers"
    if failed:
        why += f"; {failed} surface computation(s) failed"
    return out, why


# --- the hook ---------------------------------------------------------------- #


def viz_fields(
    s: Any,
    *,
    df: pd.DataFrame,
    level_columns: Sequence[Any],
    response_surface: Mapping[str, Any] | None,
    **surface_args: Any,
) -> dict[str, Any]:
    """The new top-level analysis keys, plus `reasons` for
    `visualization_reasons` (popped by the caller)."""
    try:
        levels: dict[str, list[str]] | None = categorical_levels(df, level_columns)
    except Exception:
        logger.exception("categorical_levels failed")
        levels = None
    k = surface_pairs_k()  # outside the try: a bad setting is refused, not swallowed
    try:
        surfaces, why = response_surfaces(s, first=response_surface, k=k, **surface_args)
    except Exception:
        logger.exception("response_surfaces failed")
        surfaces, why = [], "response surfaces computation failed"
    return {
        "categorical_levels": levels,
        "response_surfaces": surfaces,
        "reasons": {"response_surfaces": why},
    }


def anonymize_viz_fields(
    out: dict[str, Any],
    *,
    alias: Callable[[str], str],
    alias_recipe: Callable[[object], object],
    alias_value: Callable[[str, object], object],
) -> dict[str, Any]:
    """`anonymize=True`: column names through `alias`, level labels of identifier
    and clone columns through `alias_value`, surface recipes through
    `alias_recipe` (the same functions `response_surface` goes through)."""
    levels = out.get("categorical_levels")
    if isinstance(levels, dict):
        out["categorical_levels"] = {
            alias(str(c)): [alias_value(str(c), v) for v in vals] for c, vals in levels.items()
        }
    surfaces = out.get("response_surfaces")
    if isinstance(surfaces, list):
        aliased = []
        for surf in surfaces:
            if not isinstance(surf, dict):
                aliased.append(surf)
                continue
            surf = dict(surf)
            for key in ("x_feature", "y_feature"):
                if isinstance(surf.get(key), str):
                    surf[key] = alias(surf[key])
            if isinstance(surf.get("fixed_recipe"), dict):
                surf["fixed_recipe"] = alias_recipe(surf["fixed_recipe"])
            aliased.append(surf)
        out["response_surfaces"] = aliased
    return out
