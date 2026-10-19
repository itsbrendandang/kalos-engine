"""Does the TARGET column actually vary? A precondition for fitting any model.

A run sheet whose target is the same number on every row (every titer 5.0)
carries no information about which recipe is better. A GP will still "fit" it -
`Standardize` divides by a standard deviation of zero, or of 1e-12 float noise,
and the acquisition then ranks recipes by rounding error - and the engine used
to hand back five confident-looking proposals built on nothing. This module is
the torch-free test the portal runs BEFORE any surrogate is fit, so such a
sheet is refused with an explanation instead.

Torch-free by contract (numpy/pandas only), like the rest of this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "TARGET_CONSTANT_RTOL",
    "MIN_DISTINCT_TARGET_VALUES",
    "TargetVariation",
    "assess_target_variation",
]

TARGET_CONSTANT_RTOL = 1e-9
# A target is EFFECTIVELY constant when its spread is indistinguishable from
# floating-point noise: range(y) <= TARGET_CONSTANT_RTOL * max(|y|).
#
# Why relative, and why 1e-9:
#   - Relative to the values' own magnitude (max |y|), so the rule is unit-free:
#     the same titer in g/L or mg/L gets the same verdict, and a target that
#     legitimately lives at 1e-12 (a pM concentration in molar units) is not
#     mistaken for noise around zero - an ABSOLUTE floor would do exactly that.
#     max |y| rather than |mean y| because a target centred on zero (e.g. -1..1)
#     has a mean near 0 and would otherwise get a near-zero scale.
#   - The floor it has to clear is representation noise. float64 has a relative
#     precision of ~2.2e-16; a value that went through text -> float parsing,
#     Excel's 15-significant-digit storage, and a unit conversion or two
#     carries relative error of order 1e-15 to 1e-13. 1e-9 sits ~4 orders of
#     magnitude above that, so jitter of that kind (the "5.000000000001" cells a
#     spreadsheet formula can produce) is always caught.
#   - The ceiling it must stay under is real measurement resolution. The finest
#     bioprocess assays report ~5-6 significant figures (relative resolution
#     ~1e-6 to 1e-5); 1e-9 sits ~3 orders of magnitude below that, so any target
#     a lab instrument could genuinely resolve still counts as varying and is fit.

MIN_DISTINCT_TARGET_VALUES = 2
# Fewer than two distinct numeric target values (exact comparison) means there is
# nothing to rank. Deliberately NOT higher: a two-level target (e.g. a screen
# scored 0/1) is coarse but genuinely informative, and refusing it would reject
# real data. The tolerance rule above handles the "distinct only by float noise"
# case; this one handles the exact-duplicate case without any arithmetic.

TargetStatus = Literal["ok", "empty", "not_numeric", "constant"]


@dataclass(frozen=True)
class TargetVariation:
    """Verdict on whether a target column can support a model fit.

    `status`:
      - "ok": at least two finite numeric values that differ by more than
        float noise - the column can be modeled.
      - "empty": no non-blank cell at all.
      - "not_numeric": non-blank cells exist, but none parses as a finite number.
      - "constant": finite numeric values exist but do not vary (exactly, or
        only within `TARGET_CONSTANT_RTOL`) - no model can learn from them.
    """

    status: TargetStatus
    n_nonblank: int
    n_numeric: int
    n_distinct: int
    value_min: float | None
    value_max: float | None

    @property
    def varies(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "n_nonblank": self.n_nonblank,
            "n_numeric": self.n_numeric,
            "n_distinct": self.n_distinct,
            "min": self.value_min,
            "max": self.value_max,
        }


def assess_target_variation(values) -> TargetVariation:
    """Classify a target column (any array-like / Series) per `TargetVariation`.

    Blank cells (NaN, None, empty or whitespace-only strings) are ignored, so a
    constant column padded with blanks is still constant. Cells that do not
    parse as numbers and non-finite numbers (+-inf) are not counted as values.
    """
    s = pd.Series(values, dtype="object") if not isinstance(values, pd.Series) else values
    nonblank_mask = s.notna() & (s.astype(str).str.strip() != "")
    n_nonblank = int(nonblank_mask.sum())
    if n_nonblank == 0:
        return TargetVariation("empty", 0, 0, 0, None, None)
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        # Dates and durations are not measured outcomes: `pd.to_numeric` would
        # turn them into nanosecond counts that "vary" and get fit. Refused as
        # not numeric, explicitly. (Object columns of date cells already
        # coerce to NaN below.)
        return TargetVariation("not_numeric", n_nonblank, 0, 0, None, None)
    numeric = pd.to_numeric(s[nonblank_mask], errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    n_numeric = int(finite.size)
    if n_numeric == 0:
        return TargetVariation("not_numeric", n_nonblank, 0, 0, None, None)
    lo, hi = float(finite.min()), float(finite.max())
    n_distinct = int(np.unique(finite).size)
    scale = float(np.max(np.abs(finite)))
    constant = n_distinct < MIN_DISTINCT_TARGET_VALUES or (hi - lo) <= TARGET_CONSTANT_RTOL * scale
    return TargetVariation(
        "constant" if constant else "ok", n_nonblank, n_numeric, n_distinct, lo, hi
    )
