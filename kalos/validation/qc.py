"""Exclude runs flagged as QC failures (contaminated cultures) before any fit.

A run sheet often carries a QC column: `qc` = pass/fail, or `contamination` =
Y/N. A contaminated culture's reading is not a measurement of the recipe; it is
frequently an INFLATED titer (overgrowth reads as product, a mixed-up sample
reads high), so a model fitted on it, and an incumbent picked from it, chase a
number the process can never reproduce. The bench simulator in
`kalos/bench/qc_trap.py` plants exactly this trap.

Policy, conservative on purpose (excluding a customer's row is a decision that
has to be attributable and visible):

- A column is a QC flag only if its NAME matches (`qc`, `qc_status`,
  `contamination`, ...) AND every non-blank cell is in that kind's closed
  vocabulary (pass/fail tokens, or yes/no tokens). A `contamination` column
  holding levels 0/1/2, or a `qc` column with free text, is NOT treated as a
  flag; nothing is excluded and `skipped_reason` says why.
- Numbers are never flag values (a 0/1/2 level column is a measurement).
- The portal applies this only when the caller opts in
  (`exclude_flagged_rows`); otherwise a detected flag column is only reported.
- A row is excluded only when its cell is a FAIL token. Pass and blank are kept.
- The exclusion carries the excluded row labels (same row convention as the
  validation findings) and the token that matched, and the portal reports it
  as a warning finding next to its `qc_excluded` block.

Torch-free and pandas-only, so the portal's preflight can use it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .report import Finding, cap_rows

QC_STATUS_COLUMN = re.compile(r"^\s*qc(?:[\s_-]*(?:status|result|flag|verdict))?\s*$", re.IGNORECASE)
CONTAMINATION_COLUMN = re.compile(r"^\s*contaminat(?:ed|ion)(?:[\s_-]*(?:flag|status))?\s*$", re.IGNORECASE)

# Boolean-like TEXT only. Numbers are never flag values: a `contamination`
# column holding 0/1/2 is a measurement, and 1 must never read as "fail".
QC_FAIL_TOKENS = frozenset({"fail", "failed", "reject", "rejected", "contaminated", "oos", "invalid"})
QC_PASS_TOKENS = frozenset({"pass", "passed", "ok", "accept", "accepted", "valid", "clean"})
CONTAMINATION_TRUE_TOKENS = frozenset({"y", "yes", "true", "contaminated"})
CONTAMINATION_FALSE_TOKENS = frozenset({"n", "no", "false", "clean", "none", "ok"})

MAX_REPORTED_ROWS = 200


@dataclass(frozen=True)
class QCExclusion:
    """What `exclude_qc_failed` did. `column` is None when the sheet has no
    usable QC column; `skipped_reason` is set when a QC-named column was found
    but its values are not a recognised flag vocabulary (nothing excluded)."""

    column: str | None
    n_excluded: int = 0
    rows: tuple[int, ...] = ()
    tokens: dict = field(default_factory=dict)
    column_removed: bool = False
    skipped_reason: str | None = None
    values: tuple[str, ...] = ()  # the raw cell that triggered each excluded row

    def triggers(self) -> list[dict]:
        """One `{row, column, value}` per excluded row (raw cell text)."""
        return [
            {"row": int(r), "column": self.column, "value": v}
            for r, v in zip(self.rows, self.values)
        ][:MAX_REPORTED_ROWS]

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "n_excluded": int(self.n_excluded),
            "rows": [int(r) for r in self.rows[:MAX_REPORTED_ROWS]],
            "rows_truncated": len(self.rows) > MAX_REPORTED_ROWS,
            "matched_tokens": {str(k): int(v) for k, v in self.tokens.items()},
            "excluded": self.triggers(),
            "column_removed": bool(self.column_removed),
            "skipped_reason": self.skipped_reason,
        }


def _norm(values: pd.Series) -> pd.Series:
    def one(v: object) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v).strip().lower()

    return values.map(one)


def _fail_tokens(values: pd.Series, kind: str) -> frozenset[str] | None:
    """The fail-token set for this column, or None when a non-blank cell is
    outside the kind's closed vocabulary (so the column is not a flag)."""
    seen = set(_norm(values)) - {""}
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return None  # numeric / bool dtype: not boolean-like text
    if kind == "status":
        return QC_FAIL_TOKENS if seen <= QC_FAIL_TOKENS | QC_PASS_TOKENS else None
    if seen <= CONTAMINATION_TRUE_TOKENS | CONTAMINATION_FALSE_TOKENS:
        return CONTAMINATION_TRUE_TOKENS
    return None


def find_qc_column(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """`(column, kind)` of the first QC-NAMED column, kind `"status"` or
    `"contamination"`; `(None, None)` when there is none. Name only: whether
    its values are a usable flag is decided by `exclude_qc_failed`."""
    for c in df.columns:
        name = str(c)
        if QC_STATUS_COLUMN.match(name):
            return c, "status"
        if CONTAMINATION_COLUMN.match(name):
            return c, "contamination"
    return None, None


def qc_fail_mask(values: pd.Series, kind: str) -> np.ndarray:
    """True where the cell is a fail token for `kind`; all False when the
    column's values are not a recognised flag vocabulary."""
    tokens = _fail_tokens(values, kind)
    if tokens is None:
        return np.zeros(len(values), dtype=bool)
    return _norm(values).isin(tokens).to_numpy(bool)


def exclude_qc_failed(
    df: pd.DataFrame,
    *,
    column: str | None = None,
    drop_column: bool = True,
    keep_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, QCExclusion]:
    """Return `(df_without_failed_rows, QCExclusion)`.

    `column` names the QC column explicitly (its kind is still read from the
    name, defaulting to a pass/fail status column); `None` looks one up by
    name. Index labels are kept (no reset), so row numbers reported by later
    checks still point at the uploaded sheet. `drop_column` removes the QC
    column after filtering unless its name is in `keep_columns`.
    """
    if column is not None:
        if column not in df.columns:
            return df, QCExclusion(None, skipped_reason=f"declared QC column {column!r} is not in the sheet")
        col: object = column
        kind = "contamination" if CONTAMINATION_COLUMN.match(str(column)) else "status"
    else:
        found, found_kind = find_qc_column(df)
        if found is None or found_kind is None:
            return df, QCExclusion(None)
        col, kind = found, found_kind
    tokens = _fail_tokens(df[col], kind)
    if tokens is None:
        return df, QCExclusion(
            str(col),
            skipped_reason=(
                "values are not a recognised QC flag vocabulary "
                "(pass/fail or yes/no); no rows were excluded"
            ),
        )
    norm = _norm(df[col])
    fail = norm.isin(tokens).to_numpy(bool)
    counts = norm[fail].value_counts().to_dict()
    out = df.loc[~fail]
    removed = bool(drop_column and str(col) not in (keep_columns or set()))
    if removed:
        out = out.drop(columns=[col])
    rows = tuple(int(r) for r in df.index[fail].tolist())
    raw = tuple(str(v).strip() for v in df[col][fail].tolist())
    return out, QCExclusion(str(col), int(fail.sum()), rows, counts, removed, values=raw)


def detect_qc_column(df: pd.DataFrame) -> QCExclusion:
    """Report-only: the QC flag column `exclude_qc_failed` WOULD use and how
    many rows it would exclude, without excluding anything."""
    _, info = exclude_qc_failed(df, drop_column=False)
    return info


def qc_not_applied_findings(detected: QCExclusion) -> list[Finding]:
    """INFO finding when a QC flag column is present but exclusion was not
    requested: the flagged rows stay in the fit, and the caller is told how to
    opt in. Nothing for a sheet with no usable flag column."""
    if detected.column is None or not detected.n_excluded:
        return []
    n = detected.n_excluded
    return [Finding(
        check="qc_flags",
        severity="info",
        column=detected.column,
        message=(
            f"{n} run{'s are' if n != 1 else ' is'} flagged as QC failures in {detected.column!r} "
            "but were KEPT in the model (exclusion is opt-in: declare roles with "
            "exclude_flagged_rows=true)."
        ),
        rows=cap_rows(list(detected.rows))[0],
        detail={"n_rows_affected": n, "matched_tokens": dict(detected.tokens)},
    )]


def qc_findings(exclusion: QCExclusion) -> list[Finding]:
    """Validation findings for an exclusion: a WARNING naming the excluded rows
    when rows were dropped, an INFO when a QC-named column was ignored."""
    if exclusion.column is None:
        if exclusion.skipped_reason:
            return [Finding(check="qc_flags", severity="info", message=exclusion.skipped_reason)]
        return []
    if exclusion.n_excluded:
        n = exclusion.n_excluded
        return [Finding(
            check="qc_flags",
            severity="warning",
            message=(
                f"{n} run{'s' if n != 1 else ''} flagged as QC failures in {exclusion.column!r} "
                "were excluded from the model and from the best-run incumbent."
            ),
            column=exclusion.column,
            rows=cap_rows(list(exclusion.rows))[0],
            detail={
                "n_rows_affected": n,
                "matched_tokens": dict(exclusion.tokens),
                "excluded": exclusion.triggers(),
            },
        )]
    if exclusion.skipped_reason:
        return [Finding(
            check="qc_flags", severity="info", column=exclusion.column,
            message=f"QC-named column {exclusion.column!r} was not used: {exclusion.skipped_reason}.",
        )]
    return []


__all__ = [
    "CONTAMINATION_TRUE_TOKENS",
    "QC_FAIL_TOKENS",
    "QCExclusion",
    "detect_qc_column",
    "exclude_qc_failed",
    "find_qc_column",
    "qc_fail_mask",
    "qc_findings",
    "qc_not_applied_findings",
]
