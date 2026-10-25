"""Data-accounting diagnostics: the `data_drops` block and the findings that
explain every excluded row and cell, appended to the validation report.
"""
from __future__ import annotations

from kalos.portal.cells import REASON_CENSORED, REASON_MISSING, REASON_NON_NUMERIC
from kalos.validation.report import Finding, ValidationReport, build_report


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
    return out


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
