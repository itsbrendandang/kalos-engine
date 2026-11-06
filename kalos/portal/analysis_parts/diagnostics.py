"""Data-accounting diagnostics: the `data_drops` block and the findings that
explain every excluded row and cell, appended to the validation report.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from kalos.portal.cells import (
    BLANK,
    CENSORED,
    MISSING,
    NON_NUMERIC,
    REASON_CENSORED,
    REASON_MISSING,
    REASON_NON_NUMERIC,
)
from kalos.validation.report import Finding, ValidationReport, build_report

# (check, column) -> (affected row labels, {row label: cell} or None).
RowSets = dict[tuple[str, str | None], tuple[tuple[int, ...], dict[int, Any] | None]]


def data_finding_rows(
    df: pd.DataFrame,
    *,
    active: np.ndarray,
    incomplete: np.ndarray,
    censored: np.ndarray,
    target: str,
    tgt_kinds: np.ndarray,
    cont_feats: list,
    feat_kind: Mapping[Any, np.ndarray],
    absent_zero: set[str],
    kept_cats: list,
    cat_ok: Mapping[Any, np.ndarray],
    fit_pos: np.ndarray,
) -> RowSets:
    """Which rows each data-accounting finding is about, from the same masks the
    tallies use (positions over `df`, reported as its row labels), with the
    offending cells for the per-column findings."""
    labels = np.asarray(df.index)

    def rows(mask: np.ndarray) -> tuple[int, ...]:
        return tuple(int(labels[i]) for i in np.flatnonzero(mask))

    def cells(column: Any, picked: tuple[int, ...]) -> dict[int, Any]:
        col = df[column]
        return {r: col.loc[r] for r in picked}

    out: RowSets = {
        ("dropped_rows", None): (rows(incomplete & active), None),
        ("censored_rows", None): (rows(censored & ~incomplete & active), None),
    }

    def per_column(column: Any, kinds: np.ndarray, blank_is_missing: bool) -> None:
        bad = [MISSING, NON_NUMERIC] + ([BLANK] if blank_is_missing else [])
        dropped = rows(np.isin(kinds, bad) & active)
        if dropped:
            out[("dropped_cells", str(column))] = (dropped, cells(column, dropped))
        cens = rows((kinds == CENSORED) & active)
        if cens:
            out[("censored_values", str(column))] = (cens, cells(column, cens))

    per_column(target, tgt_kinds, True)
    for c in cont_feats:
        per_column(c, feat_kind[c], str(c) not in absent_zero)
        if str(c) in absent_zero:
            fitted_blank = np.zeros(len(df), dtype=bool)
            fitted_blank[fit_pos] = feat_kind[c][fit_pos] == BLANK
            if fitted_blank.any():
                out[("blank_as_absent", str(c))] = (rows(fitted_blank), None)
    for c in kept_cats:
        dropped = rows(~cat_ok[c] & active)
        if dropped:
            out[("dropped_cells", str(c))] = (dropped, cells(c, dropped))
    return out


def sheet_row_mapper(
    header: Mapping[str, Any], row_map: list[int] | None, transposed: bool
) -> Callable[[int], int | None]:
    """Frame row label -> the row number the user sees in the sheet: header on
    row 1 (or 2 when the names came from the second row), first data row next,
    blank lines the parser skipped counted. None for a transposed sheet, where
    runs are columns."""
    if transposed:
        return lambda _r: None
    if row_map:
        lines = list(row_map)
        return lambda r: lines[r] if 0 <= r < len(lines) else None
    offset = int(header.get("header_row") or 0) + 2
    return lambda r: int(r) + offset


def _with_rows(findings: list[Finding], row_sets: RowSets | None) -> list[Finding]:
    """Attach `all_rows`, `values` and `detail.n_rows_affected` from `row_sets`."""
    if not row_sets:
        return findings
    out = []
    for f in findings:
        hit = row_sets.get((f.check, f.column))
        if hit is None:
            out.append(f)
            continue
        picked, values = hit
        out.append(replace(
            f, all_rows=picked, values=values,
            detail={**f.detail, "n_rows_affected": len(picked)},
        ))
    return out


def _data_drops_block(
    by_column: dict[str, dict[str, int]],
    n_rows_dropped: int,
    n_censored_rows: int,
    zero_filled: dict[str, int],
) -> dict:
    """The `data_drops` block of the result: what was excluded from the fit and why.

    `by_column` counts CELLS per column and reason over the rows that carry data;
    `rows_dropped` counts the runs that were excluded. Nothing here is imputed, and the
    per-column keys are the sheet's column names (aliased under `anonymize`)."""
    by_reason = {REASON_MISSING: 0, REASON_NON_NUMERIC: 0, REASON_CENSORED: 0}
    for reasons in by_column.values():
        for reason, n in reasons.items():
            by_reason[reason] = by_reason.get(reason, 0) + n
    return {
        "rows_dropped": int(n_rows_dropped),
        "censored_rows_excluded": int(n_censored_rows),
        "cells_by_reason": by_reason,
        "cells_by_column": {c: dict(r) for c, r in by_column.items()},
        # blank feature cells kept as 0 (absent) in the fitted runs
        "blank_cells_as_absent": dict(zero_filled),
    }


_REASON_TEXT = {
    REASON_MISSING: "missing",
    REASON_NON_NUMERIC: "non-numeric",
    REASON_CENSORED: "censored",
}


def _data_findings(
    by_column: dict[str, dict[str, int]],
    n_rows_dropped: int,
    n_censored_rows: int,
    zero_filled: dict[str, int],
    *,
    auto_cats: list[str],
    header: dict,
    row_sets: RowSets | None = None,
) -> list[Finding]:
    """Findings for everything the data accounting did. Column names appear only in
    `column` and in a message that names that one column, so `anonymize` can alias
    them with the same mechanism as every other finding."""
    out: list[Finding] = []
    if n_rows_dropped:
        out.append(Finding(
            check="dropped_rows", severity="warning",
            message=(
                f"{n_rows_dropped} run{'s were' if n_rows_dropped != 1 else ' was'} excluded from "
                "the fit because the target or a feature had a missing or non-numeric cell. "
                "The cells involved are counted per column in the findings. Nothing was imputed."
            ),
            detail={"rows_dropped": int(n_rows_dropped)},
        ))
    if n_censored_rows:
        out.append(Finding(
            check="censored_rows", severity="warning",
            message=(
                f"{n_censored_rows} run{'s were' if n_censored_rows != 1 else ' was'} excluded from "
                "the fit because a value was censored (<LOD, >100, <x or >x). Censored values "
                "are not imputed and are counted per column in the findings."
            ),
            detail={"censored_rows_excluded": int(n_censored_rows)},
        ))
    for column, reasons in by_column.items():
        parts = [
            f"{reasons[r]} {_REASON_TEXT[r]}"
            for r in (REASON_MISSING, REASON_NON_NUMERIC, REASON_CENSORED)
            if reasons.get(r)
        ]
        if not parts:
            continue
        censored = reasons.get(REASON_CENSORED, 0)
        others = [p for p in parts if not p.endswith("censored")]
        if others:
            out.append(Finding(
                check="dropped_cells", severity="warning", column=column,
                message=(
                    f"Column {column!r}: "
                    + " and ".join(others)
                    + " cell(s) were not usable; the runs containing them were excluded from the "
                    "fit (not imputed)."
                ),
                detail={"cells": {r: reasons[r] for r in reasons if r != REASON_CENSORED}},
            ))
        if censored:
            out.append(Finding(
                check="censored_values", severity="warning", column=column,
                message=(
                    f"Column {column!r} has {censored} censored value(s) (<LOD, >100, <x or >x). "
                    "They were excluded from the fit and not imputed."
                ),
                detail={"censored_cells": int(censored)},
            ))
    for column, n in zero_filled.items():
        out.append(Finding(
            check="blank_as_absent", severity="info", column=column,
            message=(
                f"Column {column!r}: {n} blank cell(s) in the fitted runs were treated as 0 "
                "(absent), as for a sparse component column. The runs were kept."
            ),
            detail={"blank_cells": int(n)},
        ))
    for column in auto_cats:
        out.append(Finding(
            check="auto_categorical", severity="info", column=column,
            message=(
                f"Column {column!r} was treated as a categorical feature: text with a few "
                "repeated values, detected automatically because no roles were declared. "
                "To confirm it, declare it in roles.categoricals. Declaring any roles turns "
                "auto-detection off, so list every column the model should use."
            ),
        ))
    if header.get("header_row") == 1:
        out.append(Finding(
            check="header_row", severity="info",
            message=(
                "The column names were read from the sheet's second row. The first row was "
                "treated as a group-label row (merged headers) and not used as names."
            ),
        ))
    for column in header.get("named_by_position") or []:
        out.append(Finding(
            check="unnamed_column", severity="warning", column=str(column),
            message=(
                f"Column {column!r} has no header in the sheet, so it was named by its position. "
                "Check that it is a real input, or give it a name in the sheet."
            ),
        ))
    return _with_rows(out, row_sets)


def _with_data_findings(report: ValidationReport, extra: list[Finding]) -> ValidationReport:
    """The validation report with the data-accounting findings appended. `status`
    is re-derived by `build_report`, so a warning here makes the report say so."""
    if not extra:
        return report
    return build_report(
        mode=report.mode,
        n_rows=report.n_rows,
        n_columns=report.n_columns,
        checks_run=[*report.checks_run, "data_accounting"],
        findings=[*report.findings, *extra],
        conversions=report.conversions,
    )
