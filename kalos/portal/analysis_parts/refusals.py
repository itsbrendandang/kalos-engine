"""Refusals raised before or during a fit, each with a stable `code` and an
actionable message: too few rows, no process inputs, an unusable target, and a
malformed constraint.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from kalos.domains import DomainProfile
from kalos.portal.uploads import MIN_FIT_ROWS, UploadRejected, clip_name
from kalos.validation.target import TargetVariation, assess_target_variation

_ERR_NO_PROCESS_INPUTS = (
    "No usable process-input columns: every numeric column other than the target is "
    "an identifier, another measured output, or constant. Add at least one varying "
    "process input (for example pH, temperature, or a feed rate)."
)


def _too_few_rows(
    n: int, target: str | None = None, *, as_written: list[int] | None = None
) -> UploadRejected:
    if target is None and as_written:
        rows, cols = int(as_written[0]), int(as_written[1])
        msg = (
            f"The sheet was read as transposed (one run per column), which gives {n} "
            f"run{'s' if n != 1 else ''}. As written it has {rows} data rows and {cols} "
            f"columns. At least {MIN_FIT_ROWS} runs are needed to fit a model. Put one run "
            "per row with the column names in the first row, and upload again."
        )
    elif target is None:
        msg = (
            f"The sheet has {n} data row{'s' if n != 1 else ''}; at least {MIN_FIT_ROWS} "
            "runs are needed to fit a model. Add more runs and upload again."
        )
    else:
        msg = (
            f"Only {n} row{'s have' if n != 1 else ' has'} a numeric value for the target "
            f"{clip_name(target)!r}; at least {MIN_FIT_ROWS} are needed to fit a model. Add more "
            "measured runs, or choose another target column."
        )
    ctx: dict = {"min_rows": MIN_FIT_ROWS, "n_rows": int(n)}
    if target is not None:
        ctx["target"] = str(target)
    if target is None and as_written:
        ctx["rows_as_written"] = int(as_written[0])
    return UploadRejected(msg, code="too_few_rows", context=ctx)


def _target_error_context(
    df: pd.DataFrame, target: str, num: list, profile: DomainProfile, verdict: TargetVariation
) -> dict:
    """JSON-safe context for a refused target: the verdict plus the OTHER outcome
    columns that do vary, so a client can offer them as alternatives."""
    alternatives = [
        str(c) for c in num
        if str(c) != str(target)
        and profile.outcome_hint.search(str(c))
        and assess_target_variation(df[c]).varies
    ]
    return {
        "target": str(target),
        "target_summary": verdict.to_dict(),
        "available_targets": alternatives,
    }


def _refuse_unusable_target(
    df: pd.DataFrame, target: str, num: list, profile: DomainProfile
) -> None:
    """Raise `UploadRejected` (code `target_constant` / `target_empty` /
    `target_not_numeric`) when `df[target]` cannot support a model fit; see
    `kalos.validation.target` for the tolerance and its justification."""
    verdict = assess_target_variation(df[target])
    if verdict.varies:
        return
    if verdict.status == "constant" and verdict.n_numeric < MIN_FIT_ROWS:
        # Too few measured values to fit anything, whether or not they agree:
        # "add more runs" is the actionable answer, not "the target is constant".
        raise _too_few_rows(verdict.n_numeric, str(target))
    ctx = _target_error_context(df, target, num, profile, verdict)
    tail = "No model can be fit and no proposals were generated."
    if verdict.status == "constant":
        if verdict.n_distinct <= 1:
            what = f"has the same value ({verdict.value_min:g}) in every row"
        else:
            what = (
                f"does not vary: its values ({verdict.value_min:.15g} to "
                f"{verdict.value_max:.15g}) differ only by floating-point rounding"
            )
        raise UploadRejected(
            f"The target column {clip_name(target)!r} {what}, so there is nothing to learn "
            f"from. {tail} Choose a target that varies across runs, or add runs "
            "where it differs.",
            code="target_constant",
            context=ctx,
        )
    if verdict.status == "empty":
        raise UploadRejected(
            f"The target column {clip_name(target)!r} has no values. {tail} "
            "Fill in the measured outcome, or choose another target column.",
            code="target_empty",
            context=ctx,
        )
    raise UploadRejected(
        f"The target column {clip_name(target)!r} has no numeric values. {tail} "
        "The target must be a measured number (e.g. a titer); choose another column.",
        code="target_not_numeric",
        context=ctx,
    )


def _validate_constraint(constraint: Mapping[str, Any] | None) -> None:
    """Refuse a malformed constraint with a clean ValueError (review L1).

    A NaN or infinite floor used to reach botorch's acquisition and raise a bare
    RuntimeError ("probability tensor contains either inf, nan ..."). The floor is
    a number the caller states, so a non-finite one is a caller error and is named as
    one, before any fit runs."""
    if constraint is None:
        return
    column = constraint.get("column")
    if not isinstance(column, str) or not column.strip():
        raise ValueError("constraint.column must be a non-empty column name")
    floor = constraint.get("floor")
    if isinstance(floor, bool) or not isinstance(floor, (int, float)):
        raise ValueError("constraint.floor must be a finite number")
    if not np.isfinite(float(floor)):
        raise ValueError("constraint.floor must be a finite number (got NaN or infinity)")
