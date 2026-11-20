"""Findings from the lab-file parsers in `kalos.ingest`.

The parsers return `kalos.validation.report.Finding`, the one finding type the
validation gate uses (check, severity, message, column, rows, detail), so the
upload path can put parser findings into a validation report as they are.

`rows` are indices into the long table the parser returned (0-based,
positional), capped at `ROW_CAP`; the true count is in
`detail["n_rows_affected"]`. Where a finding is about a place in the source
file (a line, a well, a plate), that place goes in `detail`, so a scientist
can find it in the file they exported.

This file is shared by the plate, instrument and dispense parsers. Keep it
byte-identical across their branches; put parser-specific helpers in the
parser's own module.
"""
from __future__ import annotations

from collections.abc import Iterable

from kalos.validation.report import ROW_CAP, Finding, Severity, cap_rows

__all__ = ["ROW_CAP", "Finding", "Severity", "finding"]


def finding(
    check: str,
    severity: Severity,
    message: str,
    *,
    column: str | None = None,
    rows: Iterable[int] = (),
    **detail: object,
) -> Finding:
    """Build a finding with capped rows and `n_rows_affected` filled in."""
    capped, n = cap_rows(list(rows))
    info = dict(detail)
    if n:
        info["n_rows_affected"] = n
    return Finding(check, severity, message, column, capped, info)
