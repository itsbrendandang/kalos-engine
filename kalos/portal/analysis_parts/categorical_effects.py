"""Categorical inputs in the driver / correlation panels (bug fix, docs/CLONES.md).

The driver panel is a signed Spearman rank correlation and the correlation
matrix is Spearman too; neither is defined for an unordered label (a clone, a
medium), so a categorical input used to be absent from both with no trace, even
when it carried the largest effect on the sheet. This module:

- reports every categorical input as SKIPPED in `drivers_skipped` and
  `correlation_skipped`, with the reason, and
- adds `categorical_effects`: a MEASURED, recipe-unit association per
  categorical input (Kruskal-Wallis H across the levels' recipe means, its
  p-value, epsilon-squared effect size, BH across the categorical inputs) and
  the per-level n and mean. It is an association in observational data, not a
  process-adjusted effect (for the clone, see `strain_summary.levels[].adjusted_effect`).

numpy/scipy only.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from kalos.core.drivers import benjamini_hochberg

SKIP_REASON = (
    "categorical: a signed rank correlation is not defined for unordered labels; "
    "see categorical_effects"
)
MIN_RECIPES_PER_LEVEL = 2


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def categorical_effects(
    y: np.ndarray,
    labels_by_feature: dict[str, Sequence[str]],
    recipe_key: Sequence[Any],
    *,
    fdr_q: float,
) -> dict[str, Any]:
    """`{unit, method, features: [...]}` over the fitted rows. Replicates of one
    recipe are averaged first (the driver panel's unit of analysis)."""
    from scipy.stats import kruskal

    y = np.asarray(y, float)
    keys = [str(k) for k in recipe_key]
    feats: list[dict[str, Any]] = []
    pvals: list[float] = []
    for name, labels in labels_by_feature.items():
        lab = [str(v) for v in labels]
        # recipe means (a recipe key includes the categorical level)
        groups: dict[str, list[float]] = {}
        level_of: dict[str, str] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(float(y[i]))
            level_of[k] = lab[i]
        by_level: dict[str, list[float]] = {}
        for k, vals in groups.items():
            by_level.setdefault(level_of[k], []).append(float(np.mean(vals)))
        levels = sorted(by_level)
        per_level = [
            {"level": lv, "n_recipes": len(by_level[lv]), "mean": _r(float(np.mean(by_level[lv]))),
             "basis": "measured"}
            for lv in levels
        ]
        usable = [by_level[lv] for lv in levels if len(by_level[lv]) >= MIN_RECIPES_PER_LEVEL]
        entry: dict[str, Any] = {
            "name": name, "kind": "categorical", "n_levels": len(levels),
            "n_units": sum(len(v) for v in by_level.values()), "levels": per_level,
            "h": None, "p": None, "epsilon_sq": None, "survives_fdr": False, "reason": None,
        }
        if len(usable) < 2:
            entry["reason"] = (
                f"fewer than 2 levels with at least {MIN_RECIPES_PER_LEVEL} recipes; "
                "no test is reported"
            )
            feats.append(entry)
            continue
        flat = np.concatenate([np.asarray(u) for u in usable])
        if float(np.ptp(flat)) <= 0:
            entry["reason"] = "the target does not vary across these recipes"
            feats.append(entry)
            continue
        h, p = kruskal(*usable)
        n_units, n_groups = len(flat), len(usable)
        eps = (float(h) - n_groups + 1) / (n_units - n_groups) if n_units > n_groups else None
        entry.update({
            "h": _r(h), "p": round(float(p), 6),
            "epsilon_sq": None if eps is None else _r(max(eps, 0.0)),
            "n_levels_tested": n_groups,
        })
        pvals.append(float(p))
        feats.append(entry)
    tested = [f for f in feats if f["p"] is not None]
    if tested:
        keep = benjamini_hochberg(np.asarray(pvals, float), q=fdr_q)
        for f, ok in zip(tested, keep):
            f["survives_fdr"] = bool(ok)
    return {
        "unit": "recipe",
        "method": "kruskal_wallis",
        "effect_size": "epsilon_squared",
        "basis": "measured",
        "note": (
            "association across levels in the measured data, not adjusted for the other "
            "inputs; for the clone see strain_summary.levels[].adjusted_effect"
        ),
        "fdr_method": "benjamini_hochberg",
        "fdr_q": fdr_q,
        "features": feats,
    }


def skipped(features: Sequence[str]) -> list[dict[str, str]]:
    return [{"feature": str(f), "reason": SKIP_REASON} for f in features]


__all__ = ["SKIP_REASON", "categorical_effects", "skipped"]
