"""Feature-range helpers: which continuous features are absent-as-zero, and the
physically valid observed range that bounds the design box.
"""
from __future__ import annotations

import numpy as np

from kalos.domains import ColumnRoles
from kalos.validation.bounds import DIMENSION_BOUNDS, infer_dimension


def _absent_is_zero_columns(cont_feats: list, roles: ColumnRoles | None) -> set[str]:
    """Continuous feature columns where a BLANK cell means "absent" (0).

    Only two kinds of column qualify: a column the caller declared `sparse` in
    roles, and a concentration-type column by name (`infer_dimension` ==
    "concentration": a titer-like analyte, glucose, methanol, a salt, a feed
    component). For a component like that a blank is a real zero. Every other
    feature cell is an unknown value: a blank pH is not a pH of 0.0, so its run is
    dropped through the row accounting instead of being imputed (review C1).
    """
    declared = set(roles.sparse) if roles is not None else set()
    return {
        str(c) for c in cont_feats
        if str(c) in declared or infer_dimension(str(c)) == "concentration"
    }


def _physical_range(
    name: str, col: np.ndarray, *, absent: int = 0, coded: bool = False
) -> tuple[float, float, int]:
    """Observed range of `col`, computed over PHYSICALLY POSSIBLE values only.

    The design box handed to the optimizer is the observed [min, max] of each
    feature, so a single impossible cell silently widens the search space to
    include impossible recipes. This is not hypothetical: one `-999` "sensor
    offline" sentinel in a temperature column stretched the box to [-999, 37]
    and the optimizer duly proposed a bioreactor run at -422 C, below absolute
    zero, with a confidence interval attached. A recommendation engine must not
    be able to emit that, in any mode, whatever the validation report says.

    So values outside the column's HARD physical bounds
    (`kalos.validation.bounds`) are excluded from the min/max, and the count of
    excluded cells is returned so the caller can report the narrowing rather
    than perform it silently. Note this only shapes the SEARCH SPACE - the rows
    themselves still reach the surrogate, and the validation gate is what tells
    the client their data has impossible values in it.

    A column whose header maps to no known dimension is returned unchanged
    (`infer_dimension` is deliberately silent rather than guessing), as is a
    column with no physically valid values at all - clamping the latter would
    invent a range from nothing, which is a worse lie than reporting the real
    one alongside an error-severity finding.

    A `coded` column (`bounds.is_coded_column`, reported by the validation gate
    as `coded_column`) is not in physical units, so its observed range is kept.
    """
    lo_obs, hi_obs = float(col.min()), float(col.max())
    lo_obs, hi_obs = _with_absent(lo_obs, hi_obs, absent)
    dim = infer_dimension(name)
    if dim is None or coded:
        return lo_obs, hi_obs, 0
    b = DIMENSION_BOUNDS[dim]
    valid = col[(col >= b.hard_lo) & (col <= b.hard_hi)]
    n_excluded = int(col.size - valid.size)
    if n_excluded == 0 or valid.size == 0:
        if valid.size:
            return lo_obs, hi_obs, 0
        return lo_obs, hi_obs, n_excluded
    lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        # The valid cells are all one value: a zero-width box would collapse the
        # dimension. Keep the real observed span rather than emit a degenerate
        # design; the validation report carries the impossible-value error.
        return lo_obs, hi_obs, n_excluded
    lo, hi = _with_absent(lo, hi, absent)
    return lo, hi, n_excluded


def widen_box(
    name: str, lower: float, upper: float, fraction: float, *, coded: bool = False,
) -> tuple[float, float]:
    """The search box for one input when the lab allows `fraction` of the tried
    range past each end (`roles.allow_beyond`). Never past the input's hard
    physical bounds (`kalos.validation.bounds`), and an input of unknown kind
    that was never negative is not taken below 0. A coded input is widened
    without a floor. A zero-width range is not widened."""
    width = float(upper) - float(lower)
    if fraction <= 0 or not width > 0:
        return float(lower), float(upper)
    lo, hi = float(lower) - fraction * width, float(upper) + fraction * width
    dim = None if coded else infer_dimension(name)
    if dim is not None:
        b = DIMENSION_BOUNDS[dim]
        lo, hi = max(lo, b.hard_lo), min(hi, b.hard_hi)
    elif not coded and lower >= 0:
        lo = max(lo, 0.0)
    return lo, hi


def _with_absent(lo: float, hi: float, absent: int) -> tuple[float, float]:
    """Widen the box to include 0 when blank cells were read as absent (0).

    Those rows carry a 0 in the fitted design, so the box must contain that point
    (a sparse component at zero is a real recipe). This is the only value that is
    not an observed cell, and it is added only for a sparse/concentration column
    that actually had blanks.
    """
    if absent and lo > 0.0:
        return 0.0, hi
    return lo, hi
