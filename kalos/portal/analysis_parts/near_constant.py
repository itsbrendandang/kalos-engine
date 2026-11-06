"""Near-constant inputs: a continuous input whose spread on the fitted runs is
within sensor precision is fixed at its median, not searched.

An inline pH of 6.98-7.03 or a temperature of 36.98-37.01 C is one setpoint
read with sensor noise. Searching that range gives proposals that differ only
by noise (CHO 20829178 proposed pH 6.978-6.993). The rule is conservative:

- A column whose name maps to a dimension with a precision default
  (`PRECISION_DEFAULTS`: pH 0.05 units, temperature 0.1 C, percent 0.5
  points) is near-constant when its range (max - min) is at most that default
  AND at most `MAX_RELATIVE_RANGE` (1%) of |median|.
- Any other column is near-constant only when its range is below
  `FALLBACK_RELATIVE_RANGE` (0.1%) of |median|.
- A median of 0 never qualifies (no relative scale).

A fixed input is left out of the model and the search box. Every proposal's
`recipe` carries its fixed value, `fixed_inputs` lists them, provenance marks
them `fixed_near_constant`, and an info finding (`near_constant_input`) says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kalos.validation.bounds import infer_dimension
from kalos.validation.report import Finding

PRECISION_DEFAULTS = {"ph": 0.05, "temperature_c": 0.1, "percent": 0.5}
MAX_RELATIVE_RANGE = 0.01
FALLBACK_RELATIVE_RANGE = 0.001
_TOL = 1e-9


def near_constant_inputs(block: pd.DataFrame, columns: list) -> dict[str, dict]:
    """`{column: {"value", "min", "max", "rule"}}` for each near-constant input,
    judged on the fitted rows in `block` (NaN cells ignored)."""
    out: dict[str, dict] = {}
    for c in columns:
        x = pd.to_numeric(block[c], errors="coerce").to_numpy(float)
        x = x[~np.isnan(x)]
        if x.size < 2:
            continue
        lo, hi, med = float(x.min()), float(x.max()), float(np.median(x))
        spread = hi - lo
        if spread <= 0.0 or med == 0.0:
            continue
        rel = spread / abs(med)
        precision = PRECISION_DEFAULTS.get(infer_dimension(str(c)) or "")
        if precision is not None:
            ok = spread <= precision * (1 + _TOL) + _TOL and rel <= MAX_RELATIVE_RANGE
            rule = f"range <= {precision:g} and <= {MAX_RELATIVE_RANGE:.0%} of median"
        else:
            ok = rel < FALLBACK_RELATIVE_RANGE
            rule = f"range < {FALLBACK_RELATIVE_RANGE:.1%} of median"
        if ok:
            out[str(c)] = {"value": round(med, 6), "min": lo, "max": hi, "rule": rule}
    return out


def near_constant_findings(fixed: dict[str, dict]) -> list[Finding]:
    return [
        Finding(
            check="near_constant_input", code="near_constant_input", severity="info", column=c,
            message=(
                f"Input {c!r} varies only from {v['min']:g} to {v['max']:g}, which is within "
                f"sensor precision. It was fixed at its median ({v['value']:g}) and not searched. "
                "Proposals carry this value."
            ),
            detail={"value": v["value"], "min": v["min"], "max": v["max"], "rule": v["rule"]},
        )
        for c, v in fixed.items()
    ]


def with_fixed_inputs(proposals: list, fixed: dict[str, dict]) -> list:
    """Add each fixed input's value to every proposal's `recipe`."""
    if not fixed:
        return proposals
    for row in proposals:
        recipe = row.get("recipe")
        if isinstance(recipe, dict):
            for c, v in fixed.items():
                recipe.setdefault(c, v["value"])
    return proposals


__all__ = [
    "FALLBACK_RELATIVE_RANGE", "MAX_RELATIVE_RANGE", "PRECISION_DEFAULTS",
    "near_constant_findings", "near_constant_inputs", "with_fixed_inputs",
]
