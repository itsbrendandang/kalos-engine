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


def _kind_of_text(text: str) -> str:
    stripped = text.strip()
    low = stripped.lower()
    if low in NULL_TOKENS:
        return BLANK if low == "" else MISSING
    if _CENSORED_RE.match(stripped) or low in _CENSORED_WORDS:
        return CENSORED
    try:
        value = float(stripped)
    except ValueError:
        return NON_NUMERIC
    return NUMERIC if math.isfinite(value) else NON_NUMERIC


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
    "NUMERIC",
    "NULL_TOKENS",
    "REASONS",
    "REASON_CENSORED",
    "REASON_MISSING",
    "REASON_NON_NUMERIC",
    "classify_cells",
    "classify_value",
    "is_null_label",
]
