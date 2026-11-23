"""Cell classification for run-sheet values: what a single cell is, before the
analysis decides what to do with it.

Every cell of a sheet the engine uses (the target, a continuous feature, a
categorical) falls into exactly one kind:

  - ``numeric``      a finite number (``3.5``, ``-2``).
  - ``blank``        an empty cell.
  - ``missing``      a null token written in the sheet (``n/a``, ``NA``, ``nan``,
                     ``null``, ``-``, ``?``). The sheet says "no value".
  - ``censored``     a detection-limit or range bound (``<LOD``, ``>100``,
                     ``<0.5``, ``≤2``, ``BLQ``). The true value is not known.
                     It is excluded from a fit, never imputed.
  - ``non_numeric``  anything else that is not a number (``ERR``, ``pending``).

Upload parsing reads the sheet with null tokens kept as text (see
`kalos.portal.uploads`), so ``n/a`` and a truly blank cell are different kinds
here, and the analysis can report each drop by its real reason instead of
folding them into one silent NaN.

Torch-free and dependency-light (pandas + stdlib) so the parser can use it.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

BLANK = "blank"
MISSING = "missing"
CENSORED = "censored"
NUMERIC = "numeric"
NON_NUMERIC = "non_numeric"

# Reasons a row can be dropped for, as reported in the result and the run record.
REASON_MISSING = "missing"
REASON_NON_NUMERIC = "non_numeric"
REASON_CENSORED = "censored"
REASONS = (REASON_MISSING, REASON_NON_NUMERIC, REASON_CENSORED)

# Null tokens: the sheet's own way of writing "no value". Compared case-insensitively.
NULL_TOKENS = frozenset({"", "n/a", "na", "n.a.", "nan", "null", "none", "-", "--", "?", "#n/a"})

# A value that begins with a bound operator, or is a bare detection-limit word.
_CENSORED_RE = re.compile(r"^\s*(?:<=?|>=?|\u2264|\u2265)")
_CENSORED_WORDS = frozenset({"lod", "loq", "bloq", "bql", "bdl", "blq"})


# A number as a spreadsheet writes it: ASCII digits, an optional sign, decimal
# point and exponent. Deliberately narrower than Python's `float()`, which also
# accepts `1_000`, non-ASCII digits and `inf`: those are text here (reported as
# non-numeric), never silently a number. Classification (`classify_value`) and
# conversion (`parse_number`, `to_numbers`) both use this, so a cell can never be
# "numeric" to one and missing to the other (a cell like that used to be fitted
# as 0.0 in a blank-is-absent column).
_NUMBER_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def parse_number(value: object) -> float | None:
    """The finite number a cell holds, or None. Surrounding whitespace (including
    a non-breaking or thin space) is trimmed; nothing else is cleaned."""
    if value is None or value is pd.NA or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip()
    if not _NUMBER_RE.fullmatch(text):
        return None
    out = float(text)
    return out if math.isfinite(out) else None


def to_numbers(series: pd.Series | pd.DataFrame) -> pd.Series:
    """A column as floats: each cell's `parse_number`, NaN where there is none.
    The single converter for every value the model reads (replaces
    `pd.to_numeric(errors="coerce")`, which disagreed with `classify_value`)."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        out = series.astype(float)
        return out.where(np.isfinite(out.to_numpy(dtype=float, na_value=np.nan)))
    vals = [parse_number(v) for v in series.tolist()]
    return pd.Series([np.nan if v is None else v for v in vals], index=series.index, dtype=float)


def cell_label(value: object) -> str:
    """A cell as a category label, as the sheet shows it: trimmed text; a
    whole-number float (pandas types `1, 2, <blank>` as 1.0, 2.0, NaN) as `1`;
    blank as "". The one formatter for category levels, so a label never reads
    `1.0` for a sheet that says `1`."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (float, np.floating)):
        num = float(value)
        if math.isnan(num):
            return ""
        if math.isfinite(num) and num.is_integer() and abs(num) < 1e15:
            return str(int(num))
        return str(value)
    return str(value).strip()


def cell_labels(series: pd.Series) -> pd.Series:
    """`cell_label` over a column, index kept."""
    return series.map(cell_label).astype(object)


def _kind_of_text(text: str) -> str:
    stripped = text.strip()
    low = stripped.lower()
    if low in NULL_TOKENS:
        return BLANK if low == "" else MISSING
    if _CENSORED_RE.match(stripped) or low in _CENSORED_WORDS:
        return CENSORED
    return NUMERIC if parse_number(stripped) is not None else NON_NUMERIC


def classify_value(value: object) -> str:
    """The kind of one cell value (see the module docstring)."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return BLANK
    if isinstance(value, bool):
        return NON_NUMERIC
    if isinstance(value, (int, float)):
        return NUMERIC if math.isfinite(float(value)) else NON_NUMERIC
    return _kind_of_text(str(value))


def classify_cells(series: pd.Series) -> pd.Series:
    """Per-cell kinds for a column, aligned with the column's index.

    A duplicate-labelled frame column (a 2-D selection) is classified by its
    first column, matching `numeric_columns`."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # Fast path for a column pandas already typed as numbers: no per-cell Python work.
        vals = series.to_numpy(dtype=float, na_value=np.nan)
        kinds = np.where(np.isnan(vals), BLANK, np.where(np.isfinite(vals), NUMERIC, NON_NUMERIC))
        return pd.Series(kinds, index=series.index, dtype=object)
    return series.map(classify_value).astype(object)


def is_null_label(text: str) -> bool:
    """True for a categorical label that means "no value" (blank or a null token)."""
    return text.strip().lower() in NULL_TOKENS


__all__ = [
    "BLANK",
    "CENSORED",
    "MISSING",
    "NON_NUMERIC",
    "NULL_TOKENS",
    "NUMERIC",
    "REASONS",
    "REASON_CENSORED",
    "REASON_MISSING",
    "REASON_NON_NUMERIC",
    "classify_cells",
    "classify_value",
    "is_null_label",
]
