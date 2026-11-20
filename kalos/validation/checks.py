"""The eleven data-quality checks the validation gate runs over a run sheet.

Every function here is pure: `df: pd.DataFrame -> list[Finding]`, never
mutating the input frame, never logging a raw cell value or column name to
anything global (findings carry column names/values because that is the
report's whole job, but nothing here writes them anywhere else). A check
that needs domain vocabulary (which columns look like ids, which look like
outcomes) takes that as an explicit parameter instead of hardcoding a
regex, so the caller's `DomainProfile` stays the single source of truth for
that vocabulary.

Two families of numeric check live here, on purpose using different
machinery:

  - `check_units_consistency` and `check_physical_bounds` are UNIT-AWARE:
    they run every cell through `kalos.normalize.units.parse_value` so
    "34.6 C" and "94.3 F" are compared as what they actually are, not as the
    strings "34.6" and "94.3". This is the only unit parser used anywhere in
    this package - see `units.py`'s own docstring for why a second one must
    never be written.
  - `check_outliers` and `check_constant_columns` work on `pd.to_numeric`
    coercion instead. They care about the numeric *shape* of a column
    (spread, variance), not what unit it is in, and a cell with an embedded
    unit token ("34.6 C") is not a plain number to `pd.to_numeric` - it
    coerces to NaN and is correctly excluded from a statistic that would
    otherwise be nonsense on mixed unit-tagged text.

The single most important check in this file is `check_units_consistency`:
a column that silently mixes g/L and mg/mL (or C and F) is the bug class
that quietly ruins a client dataset, because every downstream statistic on
that column is computed as if the numbers were all in the same unit.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import pairwise
from re import Pattern

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kalos.core.replicates import noise_report
from kalos.normalize import units

from .bounds import DIMENSION_BOUNDS, column_bounds
from .report import Finding, UnitConversion, cap_rows

__all__ = [
    "check_constant_columns",
    "check_constant_within_group",
    "check_controls_present",
    "check_duplicate_rows",
    "check_informative_missingness",
    "check_missingness",
    "check_outliers",
    "check_physical_bounds",
    "check_provenance_metadata",
    "check_replicate_adequacy",
    "check_result_duplicates",
    "check_units_consistency",
    "detached_from_reference",
    "significant_digits",
]


def _nonblank_mask(series: pd.Series) -> pd.Series:
    """True where a cell is neither NaN nor an all-whitespace string.

    Matches `kalos.portal.validate`'s own blank test so "blank" means the
    same thing everywhere in the codebase.
    """
    return series.notna() & (series.astype(str).str.strip() != "")


def _row_index(idx: object) -> int:
    """Coerce a pandas row label to a plain `int`.

    `Series.items()` types its index as `Hashable` (pandas-stubs cannot know
    it is actually an integer RangeIndex label), so a bare `int(idx)` does
    not satisfy mypy even though every row label in this package's frames is
    integer-valued. Centralizing the cast here keeps the justification in
    one place instead of a `# type: ignore` scattered across every check.
    """
    return int(idx)  # type: ignore[call-overload]


# --- 1. unit consistency ---------------------------------------------------- #


def check_units_consistency(
    df: pd.DataFrame,
) -> tuple[list[Finding], list[UnitConversion]]:
    """Flag columns whose cells carry more than one distinct unit token.

    Per column, every non-blank cell is run through `units.parse_value`
    to get `(value, unit_token)`. What happens next depends on the set of
    distinct `unit_token`s seen (ignoring cells that fail to parse at all -
    that is a different failure mode, not this check's concern):

      - Zero or only-`None` tokens (every parsed cell was a bare number):
        nothing to report. Most numeric columns look like this.
      - Exactly one distinct non-`None` token: the column consistently uses
        one unit. INFO finding plus a `UnitConversion` record so the caller
        can auto-convert the column to its base unit via
        `apply_unit_conversions`.
      - Two or more distinct non-`None` tokens: ERROR. If every token maps
        to the same dimension via `units.canonical_suffix` (e.g. "C" and
        "F", both temperature), the message says the column mixes units
        within one dimension - the classic g/L-vs-mg/mL bug. If the tokens
        map to different dimensions, the message says the column is
        ambiguous (it is not clear what quantity the column even measures).

    A bare number mixed into a column that otherwise uses one explicit unit
    token is deliberately NOT flagged here: `canonical_suffix(None)` is `""`
    same as a dimensionless unit, but bare numbers are excluded from the
    token set entirely (not conflated with a `""`-dimension unit) because a
    spreadsheet routinely omits the unit on repeat rows once it has been
    stated once. Only genuinely differing explicit unit tokens are an error.
    """
    findings: list[Finding] = []
    conversions: list[UnitConversion] = []

    for col in df.columns:
        name = str(col)
        series = df[col]
        mask = _nonblank_mask(series)
        if not mask.any():
            continue

        token_rows: dict[str | None, list[int]] = {}
        for idx, raw in series[mask].items():
            value, token = units.parse_value(raw)
            if value is None:
                continue
            token_rows.setdefault(token, []).append(_row_index(idx))

        non_none_tokens = [t for t in token_rows if t is not None]
        if len(non_none_tokens) == 0:
            continue

        # Split on whether the registry can actually convert the token. This
        # matters because `canonical_suffix` returns "" both for an unknown token
        # and for a known dimensionless one, so suffix alone cannot tell them
        # apart. Recording a conversion for an unknown token is actively harmful:
        # a vessel column holding "5L"/"500L" ("L" is not in the registry) would
        # be "converted" to bare 5/500, silently turning a label into a
        # measurement. Unknown tokens therefore never produce a conversion.
        known_tokens = [t for t in non_none_tokens if units.is_known_unit(t)]
        unknown_tokens = [t for t in non_none_tokens if not units.is_known_unit(t)]

        if unknown_tokens:
            unknown_rows = sorted(r for t in unknown_tokens for r in token_rows[t])
            capped_unknown, total_unknown = cap_rows(unknown_rows)
            findings.append(
                Finding(
                    check="units_consistency",
                    severity="warning",
                    message=(
                        f"column '{name}' carries unit token(s) "
                        f"{sorted(unknown_tokens)} that this engine cannot "
                        "convert; the column is left exactly as uploaded"
                    ),
                    column=name,
                    rows=capped_unknown,
                    detail={
                        "unrecognized_tokens": sorted(unknown_tokens),
                        "n_rows_affected": total_unknown,
                    },
                )
            )

        if len(known_tokens) == 1 and not unknown_tokens:
            token = known_tokens[0]
            suffix = units.canonical_suffix(token)
            n_cells = len(token_rows[token])
            findings.append(
                Finding(
                    check="units_consistency",
                    severity="info",
                    message=(
                        f"column '{name}' consistently uses unit '{token}'; "
                        "eligible for automatic conversion to its base unit"
                    ),
                    column=name,
                    detail={"unit": token, "cells_converted": n_cells},
                )
            )
            conversions.append(
                UnitConversion(
                    column=name, from_units=(token,), to_unit=suffix, cells_converted=n_cells
                )
            )
            continue

        # Two or more CONVERTIBLE units in one column is the real bug class this
        # check exists for (the g/L-vs-mg/mL scale break). Judge the dimension
        # question on known tokens only - unknown ones were reported above and
        # would otherwise all collapse to the "" suffix and fake agreement.
        if len(known_tokens) < 2:
            continue

        dims = {t: units.canonical_suffix(t) for t in known_tokens}
        distinct_dims = set(dims.values())
        offending_rows = sorted(r for t in known_tokens for r in token_rows[t])
        capped, total = cap_rows(offending_rows)
        # Known tokens only: an unrecognized token already has its own warning,
        # and naming it here too would imply this error was about a unit the
        # engine had actually resolved.
        tokens_sorted = sorted(known_tokens)
        if len(distinct_dims) == 1:
            message = (
                f"column '{name}' mixes units {tokens_sorted} that all resolve to the "
                "same dimension - values are not comparable without conversion"
            )
        else:
            message = (
                f"column '{name}' mixes units {tokens_sorted} from different "
                "dimensions - the column is ambiguous about what it measures"
            )
        findings.append(
            Finding(
                check="units_consistency",
                severity="error",
                message=message,
                column=name,
                rows=capped,
                detail={"tokens": tokens_sorted, "n_rows_affected": total},
            )
        )

    return findings, conversions


# --- 2. physical bounds ------------------------------------------------------ #


def check_physical_bounds(df: pd.DataFrame) -> list[Finding]:
    """Flag values outside a column's HARD (ERROR) or TYPICAL (WARNING) range.

    The bounds come from `bounds.column_bounds`: the unit the header states
    (`DO_pct` is a percentage, `antifoam_mgL` is mg/L) through a unit-aware
    table, else the column name (`temperature_C`, `pH`). A column whose unit and
    name say nothing is skipped: no bounds check without a confident reading of
    what the column measures.

    A bare number is judged in the header's unit. A cell carrying its own unit
    token ("94.3 F") is converted to its dimension's base unit
    (`units.convert`) and judged against that dimension's base-unit bounds, so a
    Fahrenheit cell in a Celsius column is judged correctly instead of compared
    as a raw number.

    Outside HARD range -> ERROR (physically impossible). Inside HARD but
    outside TYPICAL -> WARNING (possible but operationally suspect). Each
    finding names an example row and value, the allowed range, and where the
    bounds came from; `detail` carries min/max seen and the capped row list.
    """
    findings: list[Finding] = []

    for col in df.columns:
        name = str(col)
        cb = column_bounds(name)
        if cb is None:
            continue
        series = df[col]
        mask = _nonblank_mask(series)
        if not mask.any():
            continue

        error: list[tuple[int, float]] = []
        warning: list[tuple[int, float]] = []
        for idx, raw in series[mask].items():
            value, token = units.parse_value(raw)
            if value is None:
                continue
            lo, hi, tlo, thi = cb.hard_lo, cb.hard_hi, cb.typical_lo, cb.typical_hi
            if token is not None:
                base_value, suffix = units.convert(value, token)
                dim = _SUFFIX_DIMENSION.get(suffix)
                if dim is None or not units.is_known_unit(token):
                    continue  # a unit of another kind: check_units_consistency's job
                if dim != cb.cell_dimension:
                    b = DIMENSION_BOUNDS[dim]
                    lo, hi, tlo, thi = b.hard_lo, b.hard_hi, b.typical_lo, b.typical_hi
                value = base_value
            if value < lo or value > hi:
                error.append((_row_index(idx), value))
            elif value < tlo or value > thi:
                warning.append((_row_index(idx), value))

        unit = f" {cb.unit}" if cb.unit and cb.unit != "%" else cb.unit
        for severity, hits, rng, what in (
            ("error", error, (cb.hard_lo, cb.hard_hi), "not physically possible"),
            ("warning", warning, (cb.typical_lo, cb.typical_hi), "possible but outside the usual range"),
        ):
            if not hits:
                continue
            rows = [r for r, _ in hits]
            vals = [v for _, v in hits]
            capped, total = cap_rows(rows)
            lo_txt = f"{rng[0]:g}"
            allowed = (f"{lo_txt}{unit} or more" if np.isinf(rng[1])
                       else f"{lo_txt} to {rng[1]:g}{unit}")
            note = f" {cb.note}" if severity == "error" and cb.note else ""
            findings.append(
                Finding(
                    check="physical_bounds",
                    severity=severity,  # type: ignore[arg-type]
                    message=(
                        f"column '{name}': {total} value(s) are {what} for {cb.dimension} "
                        f"({allowed}; {cb.source}), e.g. {hits[0][1]:g} in row {hits[0][0]}."
                        f"{note} Check the source record; nothing was changed."
                    ),
                    column=name,
                    rows=capped,
                    detail={
                        "dimension": cb.dimension,
                        "unit": cb.unit,
                        "bounds_source": cb.source,
                        ("hard_range" if severity == "error" else "typical_range"): [rng[0], rng[1]],
                        "min_seen": min(vals),
                        "max_seen": max(vals),
                        "n_rows_affected": total,
                    },
                )
            )

    return findings


# Base-unit suffix (kalos.normalize.units) -> DIMENSION_BOUNDS key, for cells
# that carry their own unit token.
_SUFFIX_DIMENSION = {
    "_c": "temperature_c", "_pct": "percent", "_g_l": "concentration", "_ml_h": "flow_rate", "_h": "time_h",
}


# --- 3. duplicate rows -------------------------------------------------------- #


def check_duplicate_rows(df: pd.DataFrame, *, outcome_hint: Pattern[str]) -> list[Finding]:
    """Flag exact duplicate rows (WARNING) and outcome-only-differing rows (INFO).

    An exact duplicate (every column equal) is very likely a copy-paste or
    export bug - two "different" measurements cannot be bit-for-bit
    identical across every column including any id/date column, so this is
    a WARNING, not an INFO. Rows equal on every column EXCEPT the ones
    `outcome_hint` matches are a different, benign thing: a genuine
    replicate of the same recipe with a different measured result, which is
    INFO, not a defect. `outcome_hint` is passed in (rather than
    hardcoded) so the caller's `DomainProfile` stays the single source of
    truth for what counts as an outcome column.
    """
    findings: list[Finding] = []
    if df.empty:
        return findings

    full_dup_mask = df.duplicated(keep=False)
    if full_dup_mask.any():
        rows = sorted(int(i) for i in df.index[full_dup_mask])
        capped, total = cap_rows(rows)
        findings.append(
            Finding(
                check="duplicate_rows",
                severity="warning",
                message="exact duplicate rows found (identical across every column)",
                rows=capped,
                detail={"n_rows_affected": total},
            )
        )

    outcome_cols = [c for c in df.columns if outcome_hint.search(str(c))]
    compare_cols = [c for c in df.columns if c not in outcome_cols]
    if compare_cols:
        subset_dup_mask = df.duplicated(subset=compare_cols, keep=False)
        replicate_only_mask = subset_dup_mask & ~full_dup_mask
        if replicate_only_mask.any():
            rows = sorted(int(i) for i in df.index[replicate_only_mask])
            capped, total = cap_rows(rows)
            findings.append(
                Finding(
                    check="duplicate_rows",
                    severity="info",
                    message=(
                        "rows identical on every column except the outcome column(s) - "
                        "these look like replicates of the same condition"
                    ),
                    rows=capped,
                    detail={
                        "outcome_columns": [str(c) for c in outcome_cols],
                        "n_rows_affected": total,
                    },
                )
            )

    return findings


# --- 4. missingness ----------------------------------------------------------- #


def check_missingness(df: pd.DataFrame) -> list[Finding]:
    """Flag sparse columns and sparse rows.

    Per column: >90% blank -> WARNING, the column is effectively unusable;
    50-90% blank -> WARNING, the column is sparse but may still carry
    signal. Per row: rows more than 50% blank across their columns ->
    WARNING (one finding aggregating all such rows, not one per row).
    """
    findings: list[Finding] = []
    n_rows = len(df)
    n_cols = df.shape[1]
    if n_rows == 0 or n_cols == 0:
        return findings

    for col in df.columns:
        name = str(col)
        mask = _nonblank_mask(df[col])
        frac_blank = 1.0 - (int(mask.sum()) / n_rows)
        if frac_blank > 0.90:
            findings.append(
                Finding(
                    check="missingness",
                    severity="warning",
                    message=f"column '{name}' is {frac_blank:.0%} blank - effectively unusable",
                    column=name,
                    detail={"frac_blank": frac_blank},
                )
            )
        elif frac_blank >= 0.50:
            findings.append(
                Finding(
                    check="missingness",
                    severity="warning",
                    message=f"column '{name}' is {frac_blank:.0%} blank - sparse",
                    column=name,
                    detail={"frac_blank": frac_blank},
                )
            )

    nonblank_frame = df.apply(_nonblank_mask)
    row_nonblank_counts = nonblank_frame.sum(axis=1)
    row_frac_blank = 1.0 - (row_nonblank_counts / n_cols)
    bad_rows_mask = row_frac_blank > 0.50
    if bad_rows_mask.any():
        rows = sorted(int(i) for i in df.index[bad_rows_mask])
        capped, total = cap_rows(rows)
        findings.append(
            Finding(
                check="missingness",
                severity="warning",
                message="rows more than 50% blank across their columns",
                rows=capped,
                detail={"n_rows_affected": total},
            )
        )

    return findings


# --- 5. outliers --------------------------------------------------------------- #

# BRE-118. The old rule was one marginal MAD z-score per column, |z| > 5, with a
# switch to log10 when a positive column spanned two decades. On the shipped
# 160-run example it flagged 60 rows of a perfectly clean sheet: the titer is
# zero-inflated (non-producers read 0) and right-skewed (the runs near the
# optimum are the point of the experiment), so its median and MAD describe the
# non-producers and every good run scored |z| > 5; and a uniformly varied
# antifoam setting (0.2-50 mg/L) crossed the two-decade switch and its lowest
# settings were called outliers. A check that cries wolf on the best runs of a
# clean sheet gets switched off, taking the real findings with it.
#
# The rule now asks the question an entry slip actually raises, per column:
#
#   1. Which scale? A column is judged on log10 when its positive values are
#      right-skewed and the log makes them less skewed (titers, a scale ladder,
#      anything multiplicative). Otherwise it stays linear (a uniformly varied
#      setpoint, pH, temperature). Zeros are a floor (a non-producer, a
#      component not added), never part of the spread.
#   2. On a log column: is a value DETACHED from the rest? A x10 or x1000 slip
#      lands a whole decade or three above its true value. A value (or a small
#      cluster, at most a quarter of the column) that sits at least
#      `_jump_threshold(m)` times above the next value below it is flagged. The
#      threshold shrinks as the column grows, because the best of 30 runs can
#      legitimately stand far above the second best while the best of 150 rarely
#      does (see `_jump_threshold`). Values that repeat exactly (designed levels
#      such as a 1/3/10/30 L ladder) are never called slips.
#   3. On a linear column: the robust z-score |z| > 5 (median and MAD), as
#      before. On a designed (roughly uniform) setting the most extreme setting
#      scores about 1.35, so only a value far outside the design is flagged.
#
# Every flagged value carries the factor it stands apart by and, where a x10,
# x100 or x1000 slip would put it back inside the column's range, the value it
# would then be (`suspected_value`). Nothing is changed in the data.

_OUTLIER_MIN_VALUES = 3          # linear robust z needs at least this many values
_JUMP_MIN_POSITIVE = 8           # the jump test needs at least this many positive values
_JUMP_MAX_CLUSTER_FRACTION = 0.25
_LINEAR_Z = 5.0
_SLIP_POWERS = (1, 2, 3)         # x10, x100, x1000


def _jump_threshold(m: int) -> float | None:
    """How many times above the next value a value must sit to be called detached.

    `m` is the number of positive values in the column. Calibrated on clean data,
    not on faults: on 2,000 random subsamples of the repo's clean sheets (a
    zero-inflated, right-skewed synthetic titer, the Kaggle yields and feeds, a
    scale-up ladder's airflow) the largest value exceeded the second largest by
    these factors in fewer than about 1 % of samples of that size. The skewed
    titer is the hardest case: its best-to-second-best ratio passed 3.2 in 4 % of
    30-run samples, passed 3.2 in under 1 % of 45-run samples, and never passed
    2 in 150-run samples. Fewer than `_JUMP_MIN_POSITIVE` values: no test.
    """
    if m < _JUMP_MIN_POSITIVE:
        return None
    if m < 25:
        return 10.0
    if m < 50:
        return 5.0
    if m < 100:
        return 3.0
    return 2.0


def _skew(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    sd = float(values.std())
    if sd == 0.0:
        return 0.0
    return float(np.mean(((values - values.mean()) / sd) ** 3))


def _prefers_log(positive: np.ndarray) -> bool:
    """True when the positive values are right-skewed and log10 makes them more
    symmetric: the signature of a multiplicative quantity (a titer, a ladder)."""
    if len(positive) < _JUMP_MIN_POSITIVE:
        return False
    raw = _skew(positive)
    return raw > 0.5 and abs(_skew(np.log10(positive))) < abs(raw)


def _suspected_value(value: float, rest: np.ndarray, *, upward: bool) -> tuple[float | None, int | None]:
    """The value a x10 / x100 / x1000 slip would have come from: the SMALLEST
    such slip that puts it back inside the range of `rest` (a misplaced decimal
    is more common than a larger slip; a larger one cannot be ruled out, which
    is why the result is called suspected). `(None, None)` when none does."""
    if len(rest) == 0 or value <= 0:
        return None, None
    lo, hi = float(rest.min()), float(rest.max())
    for k in _SLIP_POWERS:
        cand = value / 10**k if upward else value * 10**k
        if lo <= cand <= hi:
            return cand, k
    return None, None


def _ladder_floor(arr: np.ndarray) -> float:
    """The smallest factor that can count as detached given the column's own
    spacing: three of its widest steps between neighbouring distinct values in
    its middle half (25th-75th percentile), on a log scale. A dense column (a
    titer) has tiny steps and this floor is near 1. A designed ladder (1, 3, 10,
    30 L, or an airflow that follows it) steps about 3x per rung, so a missing
    rung or two is the design, not a slip; only a jump far beyond the ladder's
    own step counts there."""
    lo, hi = np.percentile(arr, [25, 75])
    mid = np.unique(arr[(arr >= lo) & (arr <= hi)])
    if len(mid) < 2:
        return 1.0
    step = float(np.max(np.diff(np.log10(mid))))
    return float(10 ** (3.0 * step))


def _detached_clusters(
    values: pd.Series, threshold: float
) -> list[tuple[list[int], float, float, bool]]:
    """Clusters of positive values detached from the rest by >= `threshold`x.

    Returns `(row_labels, factor, neighbour_value, upward)` per cluster: the rows
    in the cluster, how many times apart it sits from the value it is compared
    with, that value, and whether the cluster is above (True) or below.

    A value is compared with the next value toward the bulk and, so that a
    second slip cannot hide the first (two x10 slips, the smaller one sitting
    between the bulk and the larger), also with the value after that, skipping
    one (Dixon's r21 idea). The skipped value is not flagged on that account.
    The skip-one ratio is calibrated by the same clean-data rule as
    `_jump_threshold`. At most a quarter of the positive values can form a
    cluster; more than that is a mode of the data, not a slip.

    The lower tail is only tested when the column holds no zeros or negatives
    and its bulk is well away from zero (10th percentile at least half the
    median): near a detection limit, low values are naturally far apart in
    ratio terms.
    """
    pos = values[values > 0].sort_values(kind="mergesort")
    m = len(pos)
    arr = pos.to_numpy(float)
    labels = [_row_index(i) for i in pos.index]
    k_max = max(1, int(np.floor(_JUMP_MAX_CLUSTER_FRACTION * m)))
    out: list[tuple[list[int], float, float, bool]] = []
    threshold = max(threshold, _ladder_floor(arr))

    def designed_level(cluster: np.ndarray) -> bool:
        # Exact repeats are designed levels (a scale ladder, a two-level DoE),
        # not slips: a slip lands on an arbitrary value.
        return len(cluster) >= 2 and len(np.unique(cluster)) <= len(cluster) / 2

    def scan(a: np.ndarray) -> tuple[int, float, float] | None:
        """On ascending `a` (bulk first), the first top position i (cluster
        a[i:]) detached from the bulk: (i, factor, neighbour)."""
        for skip in (1, 2):
            for i in range(m - k_max, m):
                if i - skip < 0:
                    continue
                factor = float(a[i] / a[i - skip])
                if factor >= threshold and not designed_level(a[i:]):
                    return i, factor, float(a[i - skip])
        return None

    hit = scan(arr)
    if hit is not None:
        i, factor, neighbour = hit
        out.append((labels[i:], factor, neighbour, True))

    no_floor = not bool((values <= 0).any())
    bulk_away_from_zero = float(np.percentile(arr, 10)) >= 0.5 * float(np.median(arr))
    if no_floor and bulk_away_from_zero:
        hit = scan(1.0 / arr[::-1])
        if hit is not None:
            i, factor, neighbour = hit
            out.append((labels[::-1][i:], factor, 1.0 / neighbour, False))
    return out


def significant_digits(value: float) -> int:
    """Significant digits as written (0.942 -> 3, 5.0 -> 1, 1200 -> 2). Trailing
    zeros of an integer are not counted: 1200 may well be a rounded reading."""
    if not np.isfinite(value) or value == 0:
        return 0
    text = repr(abs(float(value))).lower().split("e")[0]
    digits = text.replace(".", "").lstrip("0")
    if text.endswith(".0"):
        digits = digits[:-1]
    return len(digits.rstrip("0")) or 1


def detached_from_reference(value: float, reference: Iterable[float]) -> dict | None:
    """Judge one new value against measured `reference` values with the same
    detached-value rule as `check_outliers` (the reference plus the new value,
    the factor set by the number of positive values). None when the value is
    not detached, not positive, or the reference is too small to judge.

    Returns `factor`, `threshold_factor`, `neighbour` (the measured value it is
    compared with), `direction` ("high"/"low"), `suspected_value` and
    `slip_factor` (the x10/x100/x1000 slip that puts it back inside the
    measured range, or None) and `n_reference`."""
    ref = [float(v) for v in reference if v is not None and np.isfinite(float(v))]
    if not np.isfinite(value) or value <= 0:
        return None
    series = pd.Series([*ref, float(value)])
    new_label = len(series) - 1
    threshold = _jump_threshold(int((series > 0).sum()))
    if threshold is None:
        return None
    for rows, factor, neighbour, upward in _detached_clusters(series, threshold):
        if new_label not in rows:
            continue
        rest = np.asarray([v for v in ref if v > 0], float)
        suspect, power = _suspected_value(float(value), rest, upward=upward)
        return {
            "factor": factor,
            "threshold_factor": max(threshold, _ladder_floor(np.sort(series[series > 0].to_numpy(float)))),
            "neighbour": neighbour,
            "direction": "high" if upward else "low",
            "suspected_value": None if suspect is None else float(f"{suspect:.6g}"),
            "slip_factor": None if power is None else 10**power,
            "n_reference": len(ref),
        }
    return None


def check_outliers(df: pd.DataFrame, *, outcome_columns: Iterable[str] = ()) -> list[Finding]:
    """Flag values that stand apart from the rest of their column (see the
    section comment above for the rule and why it replaced a marginal z-score).

    `outcome_columns` (the target and other measured outputs) are judged by the
    detached-value test only, whatever their shape: an outcome is skewed by the
    very settings the sheet varies (titer falls at the largest scales, peaks
    near the optimum), so a marginal z-score would flag the experiment's own
    effects. A x10 or x1000 slip still stands a decade or more apart.

    Operates on `pd.to_numeric` coercion of each column, so a cell holding
    unit-tagged text ("34.6 C") is excluded rather than compared as a raw
    number. WARNING severity: a flagged value may be real (a breakthrough run, a
    deliberate edge setting); the finding says what to check and changes nothing.
    """
    findings: list[Finding] = []
    outcomes = {str(c) for c in outcome_columns}

    for col in df.columns:
        name = str(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if len(valid) < _OUTLIER_MIN_VALUES:
            continue
        positive = valid[valid > 0].to_numpy(float)
        is_outcome = name in outcomes

        if not (valid < 0).any() and (is_outcome or _prefers_log(positive)):
            threshold = _jump_threshold(len(positive))
            if threshold is None:
                continue
            for rows, factor, neighbour, upward in _detached_clusters(valid, threshold):
                rest = valid[(valid > 0) & ~valid.index.isin(rows)].to_numpy(float)
                flagged = []
                for r in rows[:20]:
                    v = float(valid.loc[r])
                    suspect, power = _suspected_value(v, rest, upward=upward)
                    flagged.append({
                        "row": r, "value": v,
                        "suspected_value": None if suspect is None else float(f"{suspect:.6g}"),
                        "slip_factor": None if power is None else 10**power,
                    })
                capped, total = cap_rows(rows)
                first = flagged[0]
                side = "above the next-highest" if upward else "below the next-lowest"
                hint = ""
                if first["suspected_value"] is not None:
                    hint = (
                        f" A x{first['slip_factor']} entry slip would look like this; the value "
                        f"would then be {first['suspected_value']:g}."
                    )
                findings.append(
                    Finding(
                        check="outliers",
                        severity="warning",
                        message=(
                            f"column '{name}': {total} value(s) stand apart from the rest, e.g. "
                            f"{first['value']:g} (row {first['row']}) is {factor:.1f}x {side} value "
                            f"({neighbour:g}).{hint} Check the source record; nothing was changed."
                        ),
                        column=name,
                        rows=capped,
                        detail={
                            "scale": "ratio",
                            "rule": "detached_value",
                            "factor": factor,
                            "threshold_factor": threshold,
                            "n_positive": len(positive),
                            "direction": "high" if upward else "low",
                            "flagged": flagged,
                            "n_rows_affected": total,
                        },
                    )
                )
            continue

        median = float(valid.median())
        mad = float((valid - median).abs().median())
        if mad == 0.0:
            continue
        z = 0.6745 * (valid - median) / mad
        outlier_mask = z.abs() > _LINEAR_Z
        if outlier_mask.any():
            rows = sorted(int(i) for i in valid.index[outlier_mask])
            capped, total = cap_rows(rows)
            worst = int(z.abs().idxmax())  # type: ignore[arg-type]
            findings.append(
                Finding(
                    check="outliers",
                    severity="warning",
                    message=(
                        f"column '{name}': {total} value(s) lie far outside the rest of the column, "
                        f"e.g. {float(valid.loc[worst]):g} (row {worst}) against a median of "
                        f"{median:g} (robust z {float(z.loc[worst]):.1f}; flagged above 5). "
                        "Check the source record; nothing was changed."
                    ),
                    column=name,
                    rows=capped,
                    detail={
                        "scale": "linear",
                        "rule": "robust_z",
                        "median": median,
                        "mad": mad,
                        "threshold_z": _LINEAR_Z,
                        "n_rows_affected": total,
                    },
                )
            )

    return findings


# --- 6. provenance metadata ---------------------------------------------------- #

_DATE_RE = re.compile(r"date|timestamp", re.I)
_OPERATOR_RE = re.compile(r"operator|instrument|analyst|technician|equipment", re.I)

# Run/batch identifier detection for provenance. This does NOT reuse the
# caller's `DomainProfile.id_hint`, even though that pattern lists the same
# vocabulary, because the two ask different questions and id_hint is anchored
# (`^(...|run|batch|lot)$`). Anchoring is right for id_hint's own job in
# `_analyze` - deciding whether a column is ENTIRELY an identifier and should be
# dropped from modeling, where matching `batch_titer` would silently discard a
# real measurement. It is wrong here, where the question is merely "does any
# traceability column exist", because every real header is compound: `batch_id`,
# `run_number`, `run_id`, `lot_number`. Reusing id_hint made the gate report "no
# run/batch identifier column found" on a sheet whose first two columns were
# `batch_id` and `run_number` - telling a client their data is untraceable when
# it is fully traceable, which is exactly the kind of false alarm that makes a
# validation report get ignored.
#
# The optional suffix group keeps specificity: `batch`/`batch_id`/`run_number`
# match, while `batch_titer` and `run_duration_days` (measurements that merely
# start with an identifier word) do not.
_RUN_ID_RE = re.compile(
    r"^(run|batch|lot|campaign|experiment|expt|sample|culture|vessel|reactor|bioreactor)"
    r"([ _\-.]?(id|ids|no|num|number|code|label|name|key|ref))?$"
    r"|^(id|uuid|index|well|barcode)$",
    re.I,
)


def check_provenance_metadata(df: pd.DataFrame, id_hint: Pattern[str]) -> list[Finding]:
    """Flag missing run-traceability columns: a run/batch id, a date, an operator/instrument.

    Each missing category gets its own WARNING naming exactly what is absent,
    because a client cannot fix what they cannot see named.

    `id_hint` is accepted for interface compatibility with the caller's
    `DomainProfile` but is deliberately NOT used to answer the identifier
    question - see `_RUN_ID_RE` above for why an anchored role-classification
    pattern gives the wrong answer here.
    """
    findings: list[Finding] = []
    names = [str(c) for c in df.columns]

    if not any(_RUN_ID_RE.match(n.strip()) for n in names):
        findings.append(
            Finding(
                check="provenance_metadata",
                severity="warning",
                message=(
                    "no run/batch identifier column found - results cannot be traced "
                    "back to a physical run"
                ),
                detail={"missing": "run_or_batch_id"},
            )
        )
    if not any(_DATE_RE.search(n) for n in names):
        findings.append(
            Finding(
                check="provenance_metadata",
                severity="warning",
                message="no date column found - results cannot be traced to when the run happened",
                detail={"missing": "date"},
            )
        )
    if not any(_OPERATOR_RE.search(n) for n in names):
        findings.append(
            Finding(
                check="provenance_metadata",
                severity="warning",
                message=(
                    "no operator/instrument column found - results cannot be traced "
                    "to who or what equipment ran the assay"
                ),
                detail={"missing": "operator_or_instrument"},
            )
        )

    return findings


# --- 7. replicate adequacy ------------------------------------------------------ #


def check_replicate_adequacy(
    df: pd.DataFrame, target: str | None, features: list[str] | None
) -> list[Finding]:
    """Flag row-count and replicate-structure problems for fitting a GP.

    `n_rows < 6` is the engine's own hard floor (ERROR - the engine will
    refuse to fit at all). `n_rows < 10 * n_features` is a WARNING: fewer
    than ten rows per feature is under-determined for a Gaussian process in
    practice, even though the engine will still attempt a fit. When `target`
    and `features` are both given, `kalos.core.replicates.noise_report` is
    reused to report the distinct-condition count, how many are replicated,
    and the estimated noise floor / ICC (INFO); zero replicated conditions
    is a further WARNING, because with no replication the assay noise
    cannot be separated from a real effect.
    """
    findings: list[Finding] = []
    n_rows = len(df)
    feats = list(features) if features else []
    n_features = len(feats)

    if n_rows < 6:
        findings.append(
            Finding(
                check="replicate_adequacy",
                severity="error",
                message=f"only {n_rows} row(s); the engine requires at least 6 rows to fit a model",
                detail={"n_rows": n_rows},
            )
        )
    if n_features > 0 and n_rows < 10 * n_features:
        findings.append(
            Finding(
                check="replicate_adequacy",
                severity="warning",
                message=(
                    f"{n_rows} rows for {n_features} feature(s) is under-determined for a "
                    f"GP (want at least {10 * n_features})"
                ),
                detail={"n_rows": n_rows, "n_features": n_features},
            )
        )

    if target is not None and feats and n_rows > 0:
        X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_valid, y_valid = X[valid], y[valid]
        if X_valid.shape[0] >= 1:
            summary = noise_report(X_valid, y_valid)
            findings.append(
                Finding(
                    check="replicate_adequacy",
                    severity="info",
                    message=(
                        f"{summary['n_recipes']} distinct condition(s) found, "
                        f"{summary['n_replicated']} replicated"
                    ),
                    detail=summary,
                )
            )
            if summary["n_recipes"] > 0 and summary["n_replicated"] == 0:
                findings.append(
                    Finding(
                        check="replicate_adequacy",
                        severity="warning",
                        message=(
                            "no condition is replicated - assay noise cannot be separated "
                            "from a real effect"
                        ),
                        detail={"n_recipes": summary["n_recipes"]},
                    )
                )

    return findings


# --- 8. controls present --------------------------------------------------------- #

_CONTROL_RE = re.compile(r"control|blank|reference|baseline", re.I)


def check_controls_present(df: pd.DataFrame) -> list[Finding]:
    """Flag a run sheet with no control/blank/reference/baseline label anywhere.

    Looks for the label both in column names (a column literally called
    "control") and in the values of any object-dtype (categorical/text)
    column, since a run sheet more commonly marks a control row with a
    label value ("control", "blank") in an existing condition column than
    with a dedicated column. Absent either way -> WARNING: without an
    internal reference, batch-to-batch or calibration drift is invisible.
    """
    for col in df.columns:
        if _CONTROL_RE.search(str(col)):
            return []

    for col in df.columns:
        series = df[col]
        # Skip genuinely numeric columns (a control label cannot live in
        # them); everything else - legacy `object` string columns and
        # pandas' newer dedicated string dtypes alike - is scanned by value.
        if pd.api.types.is_numeric_dtype(series):
            continue
        for val in series.dropna():
            if _CONTROL_RE.search(str(val)):
                return []

    return [
        Finding(
            check="controls_present",
            severity="warning",
            message=(
                "no control/blank/reference/baseline label found - no internal "
                "reference to detect batch or calibration drift"
            ),
        )
    ]


# --- 9. constant columns ----------------------------------------------------------- #


def check_constant_columns(df: pd.DataFrame) -> list[Finding]:
    """Flag zero-variance numeric columns (INFO): they carry no signal.

    Operates on `pd.to_numeric` coercion, same rationale as `check_outliers`.
    Requires at least 2 parseable values before calling a column "constant" -
    a column with a single non-blank value and the rest genuinely blank is a
    missingness problem (`check_missingness`'s job), not a constant-value one.
    """
    findings: list[Finding] = []
    for col in df.columns:
        name = str(col)
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.dropna()
        if len(valid) < 2:
            continue
        if valid.nunique() == 1:
            findings.append(
                Finding(
                    check="constant_columns",
                    severity="info",
                    message=(
                        f"column '{name}' has zero variance - carries no signal and "
                        "will be dropped downstream"
                    ),
                    column=name,
                    detail={"value": float(valid.iloc[0])},
                )
            )
    return findings


# --- 10. informative missingness (MNAR) ---------------------------------------- #

# A Spearman rho's standard error scales roughly as 1/sqrt(n-3), so below
# about a dozen paired observations even a genuine correlation reads as
# confident nonsense - the number would look decisive while being mostly
# sampling noise. Both the row-level and column-level tests below refuse to
# report anything under this floor rather than manufacture a rho nobody
# should trust.
_MIN_N_FOR_RHO = 12

# How far |rho| has to clear before missingness is called informative rather
# than noise. At ordinary sheet sizes (n ~ 30-100) a null (label-shuffled)
# Spearman rho already wanders past +-0.3 with non-trivial probability by
# chance alone, so 0.35 sits with real margin above that noise floor - while
# staying far below the -0.835 that a real clone-funnel dataset actually
# showed (earlier engine docs/ml-review.md, finding P0.1).
# This check exists to catch that class of signal, not to flag every faint
# wobble in a small sheet.
_MISSINGNESS_RHO_WARN = 0.35

# A column must be at least this blank before its missingness indicator is
# tested against the target at all. Below 5% blank, "missing" is a handful of
# rows out of the whole sheet, and any correlation computed from that few
# positive cases is dominated by which specific rows happen to be missing,
# not a real pattern - the same small-n concern as `_MIN_N_FOR_RHO`, applied
# to the minority class of a binary indicator instead of to the whole sample.
_COL_MISSINGNESS_MIN_BLANK_FRAC = 0.05


def check_informative_missingness(df: pd.DataFrame, target: str | None) -> list[Finding]:
    """Flag missingness that CORRELATES WITH THE TARGET (MNAR), not just blank cells.

    `check_missingness` reports how much of a column or row is blank; this asks
    a different question - whether being blank is itself informative, i.e.
    correlated with the outcome, rather than happening at random. On a
    real clone-funnel dataset, per-row missing-fraction vs titer
    had Spearman rho = -0.835, and non-producers were 66.7% missing vs 37.0%
    for producers: "no measurement" was the human culling decision on low
    performers, not benign absence. `kalos`'s GP path zero-fills blanks before
    fitting, which is safe under a missing-COMPLETELY-at-random assumption and
    quietly encodes the outcome into the features the moment that assumption
    is false.

    Two rank-based tests (Spearman - neither assumes the target or the
    missing-fraction is normally distributed, and Spearman between a 0/1
    indicator and a continuous variable reduces exactly to the rank-biserial
    correlation, so no separate point-biserial formula needs to be
    maintained):

      - ROW-level: each row's fraction of blank cells across every column
        EXCEPT `target` itself, vs. that row's numeric target value. A
        confidently negative rho means rows with more blanks skew toward
        lower outcomes - "not measured" looking like a proxy for "did not
        perform."
      - COLUMN-level: for every column at least `_COL_MISSINGNESS_MIN_BLANK_FRAC`
        blank, its own 0/1 missingness indicator (1 = blank in that row) vs.
        the target.

    Both require at least `_MIN_N_FOR_RHO` rows with a valid (numeric,
    non-blank) target, and both fire only once |rho| clears
    `_MISSINGNESS_RHO_WARN`. Severity is ALWAYS "warning", never "error": this
    check changes nothing about whether a sheet is usable, only whether
    zero-filling it is safe to do silently - flipping a strict-mode upload
    from accepted to rejected on this finding would be a product decision this
    check has no business making on its own.

    Never crashes and never raises severity beyond warning: no `target` given,
    a `target` that is entirely non-numeric, a constant target (rho is
    undefined - nothing to correlate against), or a frame/target pair with no
    blanks at all in the feature columns each produce an empty finding list,
    not an exception.
    """
    findings: list[Finding] = []
    if target is None or target not in df.columns:
        return findings

    y_all = pd.to_numeric(df[target], errors="coerce")
    valid_target = y_all.notna()
    if int(valid_target.sum()) < _MIN_N_FOR_RHO:
        return findings
    if y_all.loc[valid_target].nunique() < 2:
        return findings  # constant target: rho is undefined, nothing to correlate against

    feature_cols = [c for c in df.columns if c != target]
    if not feature_cols:
        return findings

    nonblank = df[feature_cols].apply(_nonblank_mask)
    if bool(nonblank.to_numpy().all()):
        return findings  # no blanks anywhere in the feature columns: nothing to say

    # --- row-level: fraction of blank feature cells vs. the target -------------- #
    row_frac_blank = 1.0 - (nonblank.sum(axis=1) / len(feature_cols))
    frac_valid = row_frac_blank.loc[valid_target]
    y_valid = y_all.loc[valid_target]
    if frac_valid.nunique() >= 2:
        rho, _p = spearmanr(frac_valid, y_valid, nan_policy="omit")
        if not np.isnan(rho) and abs(rho) >= _MISSINGNESS_RHO_WARN:
            findings.append(
                Finding(
                    check="informative_missingness",
                    severity="warning",
                    message=(
                        f"per-row missing-fraction correlates with target '{target}' "
                        f"(Spearman rho={rho:.2f}, n={int(frac_valid.shape[0])}) - "
                        "missingness looks informative, not random; zero-filling "
                        "before the fit may be encoding the outcome into the features"
                    ),
                    detail={
                        "target": target,
                        "rho": float(rho),
                        "n": int(frac_valid.shape[0]),
                    },
                )
            )

    # --- column-level: each sparse column's own missingness vs. the target ------ #
    for col in feature_cols:
        name = str(col)
        col_nonblank = nonblank[col]
        frac_blank_col = 1.0 - (int(col_nonblank.sum()) / len(df))
        if frac_blank_col < _COL_MISSINGNESS_MIN_BLANK_FRAC or frac_blank_col >= 1.0:
            continue
        indicator = (~col_nonblank.loc[valid_target]).astype(int)
        if len(indicator) < _MIN_N_FOR_RHO or indicator.nunique() < 2:
            continue
        y_here = y_all.loc[valid_target]
        rho, _p = spearmanr(indicator, y_here, nan_policy="omit")
        if not np.isnan(rho) and abs(rho) >= _MISSINGNESS_RHO_WARN:
            findings.append(
                Finding(
                    check="informative_missingness",
                    severity="warning",
                    message=(
                        f"column '{name}' missingness correlates with target '{target}' "
                        f"(Spearman rho={rho:.2f}, {frac_blank_col:.0%} blank, "
                        f"n={len(indicator)}) - whether this cell was measured looks "
                        "informative, not random; zero-filling it may be encoding the "
                        "outcome into the features"
                    ),
                    column=name,
                    detail={
                        "target": target,
                        "rho": float(rho),
                        "frac_blank": frac_blank_col,
                        "n": len(indicator),
                    },
                )
            )

    return findings


# --- 11. constant-within-group columns ------------------------------------------ #


def check_constant_within_group(df: pd.DataFrame, *, group_hint: Pattern[str]) -> list[Finding]:
    """Flag a column that varies globally but is a group-identity proxy in disguise.

    `check_constant_columns` catches a column that is constant EVERYWHERE - no
    signal at all. This catches the sneakier variant: a column that varies
    across the whole sheet (so `check_constant_columns` says nothing) but has
    exactly one value inside every group - so within any single group it
    carries zero information, and whatever it appears to predict is actually
    just "which group this row is in." On a real clone-funnel dataset,
    4 of 6 candidate features had exactly one unique value
    within every campaign; they were campaign identity in disguise, and they
    inflated a within-population score to ~0.86 when the campaign-mean anchor
    alone - which uses no per-clone information whatsoever - already scored
    ~0.75 (earlier engine docs/ml-review.md, finding P0.2). A model fit
    on such a column mostly learns "which group," not the thing the group
    label is standing in for.

    The group column is detected the same way `kalos.portal.analysis` infers
    it for the live analyze path: the first column whose name matches
    `group_hint` (a `DomainProfile`'s `group_hint` regex, e.g.
    `medium|strain|recipe|batch|campaign|group|lot` for the bioprocess
    profile). No group column present -> nothing to check, empty list.

    A group must have at least 2 non-blank-labeled rows to say anything about
    "varies within it" at all (a singleton group trivially has exactly one
    value in every column, which is not evidence of anything); if every group
    is a singleton, the check has nothing measurable and returns no findings.
    Blank cells never count as "a value" - a column that is blank throughout
    some group is a missingness problem (`check_missingness`'s job), not
    evidence the column is constant there, so that group correctly fails the
    "exactly one unique value" test and the column is not flagged on its
    account.

    WARNING severity: a proxy feature does not make a sheet unusable, but any
    model fit on it needs the caller to know that measured skill may be group
    memorization, not real signal.
    """
    findings: list[Finding] = []
    gcol = next((c for c in df.columns if group_hint.search(str(c))), None)
    if gcol is None:
        return findings

    nonblank = df.apply(_nonblank_mask)
    group_present = nonblank[gcol]
    if not bool(group_present.any()):
        return findings

    group_labels = df.loc[group_present, gcol]
    group_sizes = group_labels.value_counts()
    measurable_groups = set(group_sizes[group_sizes >= 2].index)
    if not measurable_groups:
        return findings  # every group is a singleton: within-group variation is unmeasurable

    in_measurable = group_present & df[gcol].isin(measurable_groups)
    groups_for_rows = df.loc[in_measurable, gcol]
    blanked = df.where(nonblank)  # blank cells -> NaN, matches `_nonblank_mask` everywhere else

    gcol_name = str(gcol)
    for col in df.columns:
        if col == gcol:
            continue
        name = str(col)
        global_values = blanked[col].dropna()
        if global_values.nunique() < 2:
            continue  # already globally constant - check_constant_columns' job, not this one

        within_group_nunique = blanked.loc[in_measurable, col].groupby(groups_for_rows).nunique(
            dropna=True
        )
        # Every measurable group must show EXACTLY one non-blank value (not "at
        # most one": a group where this column is entirely blank has nunique
        # 0, which correctly fails this test rather than looking constant).
        if len(within_group_nunique) == len(measurable_groups) and bool(
            (within_group_nunique == 1).all()
        ):
            findings.append(
                Finding(
                    check="constant_within_group",
                    severity="warning",
                    message=(
                        f"column '{name}' varies globally but has exactly one value "
                        f"within every group of '{gcol_name}' ({len(measurable_groups)} "
                        "group(s) with 2+ rows checked) - looks like a proxy for group "
                        "identity, not a real feature; any predictive skill attributed "
                        "to it may be group memorization"
                    ),
                    column=name,
                    detail={
                        "group_column": gcol_name,
                        "n_groups_checked": len(measurable_groups),
                    },
                )
            )

    return findings


# --- 12. copy-pasted results ------------------------------------------------------ #

# Expected number of neighbouring matches by chance on the sheet (see
# `check_result_duplicates`). At or below `_PASTE_WARN_EXPECTED` a match between
# neighbours is a warning; up to `_PASTE_MAX_EXPECTED` it is reported as info
# (worth a look, but a coincidence would not be surprising); above that the check
# stays silent, because a long sheet of 3-decimal values repeats by chance.
_PASTE_WARN_EXPECTED = 0.02
_PASTE_MAX_EXPECTED = 0.1


def check_result_duplicates(df: pd.DataFrame, target: str | None) -> list[Finding]:
    """Flag neighbouring rows whose result is exactly the same while their
    settings differ: the signature of a value pasted into the next run.

    Only precise values count (3 or more significant digits, not 0, and not a
    reading the sheet records often, such as a reporting limit). The check
    also asks how often two neighbours would match by chance on THIS sheet:
    with every repeated value placed at random, the expected number of
    neighbouring matches is sum(c * (c - 1)) / n over the repeated values. When
    that exceeds `_PASTE_MAX_EXPECTED`, a match between neighbours is not
    evidence of anything and nothing is reported; up to `_PASTE_WARN_EXPECTED`
    it is a WARNING, in between INFO.

    The suspected correct value is unknown (only the lab record has it), so the
    finding names both rows and asks for the record to be checked.
    """
    if target is None or target not in df.columns or len(df) < 3:
        return []
    y = pd.to_numeric(df[target], errors="coerce")
    counts = y.dropna().value_counts()
    n = int(y.notna().sum())
    common = max(3, int(np.ceil(0.05 * n)))

    def eligible(v: float) -> bool:
        return bool(np.isfinite(v)) and v != 0 and significant_digits(v) >= 3 and int(counts.get(v, 0)) < common

    repeated = [int(c) for v, c in zip(counts.index.to_numpy(float), counts.to_numpy()) if c >= 2 and eligible(float(v))]
    expected = sum(c * (c - 1) for c in repeated) / max(n, 1)
    if not repeated or expected > _PASTE_MAX_EXPECTED:
        return []
    settings = [c for c in df.columns if c != target and not _RUN_ID_RE.match(str(c).strip())]
    as_text = df[settings].astype(str) if settings else None
    labels = list(df.index)
    pairs: list[tuple[int, int, float]] = []
    for a, b in pairwise(labels):
        va, vb = y.loc[a], y.loc[b]
        if pd.isna(va) or va != vb or not eligible(float(va)):
            continue
        if as_text is not None and as_text.loc[a].equals(as_text.loc[b]):
            continue  # the whole row repeats: check_duplicate_rows reports it
        pairs.append((_row_index(a), _row_index(b), float(va)))
    if not pairs:
        return []
    rows = sorted({r for a, b, _ in pairs for r in (a, b)})
    capped, total = cap_rows(rows)
    a, b, v = pairs[0]
    return [
        Finding(
            check="result_duplicate_entry",
            severity="warning" if expected <= _PASTE_WARN_EXPECTED else "info",
            message=(
                f"column '{target}': rows {a} and {b} are neighbours with exactly the same result "
                f"({v:g}) but different settings{f', and {len(pairs) - 1} more such pair(s)' if len(pairs) > 1 else ''}. "
                "Either the value was pasted into the next run or the two results happen to match "
                f"(on this sheet a match between neighbours is expected about {expected:.2f} times by "
                "chance). Check both runs in the lab record; nothing was changed."
            ),
            column=str(target),
            rows=capped,
            detail={
                "pairs": [{"rows": [a_, b_], "value": v_} for a_, b_, v_ in pairs[:20]],
                "expected_by_chance": expected,
                "suspected_value": None,
                "n_rows_affected": total,
            },
        )
    ]

