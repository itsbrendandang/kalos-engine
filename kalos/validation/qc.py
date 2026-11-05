"""Exclude runs flagged as QC failures (contaminated cultures) before any fit.

A run sheet often carries a QC column: `qc` = pass/fail, or `contamination` =
Y/N. A contaminated culture's reading is not a measurement of the recipe; it is
frequently an INFLATED titer (overgrowth reads as product, a mixed-up sample
reads high), so a model fitted on it, and an incumbent picked from it, chase a
number the process can never reproduce. The bench simulator in
`kalos/bench/qc_trap.py` plants exactly this trap.

Policy: a row whose QC cell is a recognised FAIL token is excluded from the fit
and the incumbent, and the count is reported. Anything else (pass, blank,
an unrecognised token) is kept: this module only removes rows the sheet itself
marks as failed, it never guesses. The QC column itself is not a process input
and is removed so it cannot become a categorical feature.

Torch-free and pandas-only, so the portal's preflight can use it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Column names that carry a QC verdict, by meaning of the cell values.
#   status columns: the cell says pass/fail.
#   contamination columns: the cell says whether the run was contaminated.
QC_STATUS_COLUMN = re.compile(r"^\s*qc(?:[\s_-]*(?:status|result|flag|verdict))?\s*$", re.IGNORECASE)
CONTAMINATION_COLUMN = re.compile(r"^\s*contaminat(?:ed|ion)(?:[\s_-]*(?:flag|status))?\s*$", re.IGNORECASE)

QC_FAIL_TOKENS = frozenset({"fail", "failed", "reject", "rejected", "contaminated", "oos", "invalid"})
CONTAMINATION_TRUE_TOKENS = frozenset({"y", "yes", "true", "1", "1.0", "contaminated", "x"})


@dataclass(frozen=True)
class QCExclusion:
    """What `exclude_qc_failed` did. `column` is None when the sheet has no QC
    column (nothing was checked, nothing was excluded)."""

    column: str | None
    n_excluded: int = 0
    excluded_index: tuple = field(default_factory=tuple)
    column_removed: bool = False

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "n_excluded": int(self.n_excluded),
            "column_removed": bool(self.column_removed),
        }


def find_qc_column(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """`(column, kind)` of the first QC column, kind `"status"` or
    `"contamination"`; `(None, None)` when there is none."""
    for c in df.columns:
        name = str(c)
        if QC_STATUS_COLUMN.match(name):
            return c, "status"
        if CONTAMINATION_COLUMN.match(name):
            return c, "contamination"
    return None, None


def qc_fail_mask(values: pd.Series, kind: str) -> np.ndarray:
    """Boolean mask, True where the cell is a recognised fail token for `kind`."""
    tokens = QC_FAIL_TOKENS if kind == "status" else CONTAMINATION_TRUE_TOKENS
    norm = values.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return norm.str.strip().str.lower().isin(tokens).to_numpy(bool)


def exclude_qc_failed(
    df: pd.DataFrame, *, drop_column: bool = True, keep_columns: set[str] | None = None
) -> tuple[pd.DataFrame, QCExclusion]:
    """Return `(df_without_failed_rows, QCExclusion)`.

    The original index labels are kept (no reset), so row numbers reported by
    later checks still point at the uploaded sheet. `drop_column` removes the QC
    column after filtering unless its name is in `keep_columns` (e.g. a caller
    that declared it as a feature on purpose).
    """
    col, kind = find_qc_column(df)
    if col is None or kind is None:
        return df, QCExclusion(None)
    fail = qc_fail_mask(df[col], kind)
    out = df.loc[~fail]
    removed = bool(drop_column and str(col) not in (keep_columns or set()))
    if removed:
        out = out.drop(columns=[col])
    return out, QCExclusion(str(col), int(fail.sum()), tuple(df.index[fail].tolist()), removed)


__all__ = [
    "CONTAMINATION_TRUE_TOKENS",
    "QC_FAIL_TOKENS",
    "QCExclusion",
    "exclude_qc_failed",
    "find_qc_column",
    "qc_fail_mask",
]
