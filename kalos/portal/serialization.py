"""Kalos portal — JSON-safe serialization helpers."""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame rows as JSON-safe dicts: NaN/NaT/inf -> None.

    Missing cells arrive from pandas as float NaN; Starlette's JSONResponse
    encodes with allow_nan=False, so an un-sanitized NaN 500s the response.
    Coerce every non-finite float and NaT to JSON null, so both the stored
    payload and the HTTP response are valid JSON.
    """
    safe = df.astype(object).where(pd.notna(df), None)
    # column labels are strings here; to_dict types them as Hashable
    records = cast("list[dict[str, Any]]", safe.to_dict(orient="records"))
    for row in records:
        for key, val in row.items():
            if isinstance(val, float) and not np.isfinite(val):
                row[key] = None
    return records


def is_finite_number(value: object) -> bool:
    """True for a real int/float that is not NaN or infinity. Booleans are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def finite_interval(bounds: object, ndigits: int = 3) -> list[float] | None:
    """A JSON pair of finite bounds, or None when either end is missing or not finite.

    `grouped_cv_report` returns `(nan, nan)` when a bootstrap cannot run (fewer
    than three groups). Rounding those NaNs and putting them in the response
    makes Starlette's `JSONResponse` (`allow_nan=False`) raise, which the upload
    route reports as `parse_error` and a reanalyze reports as HTTP 500.
    """
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    if not is_finite_number(bounds[0]) or not is_finite_number(bounds[1]):
        return None
    return [round(float(bounds[0]), ndigits), round(float(bounds[1]), ndigits)]


def json_safe(value: Any) -> Any:
    """Copy `value` with every non-finite float replaced by None.

    The analysis dict is encoded with `allow_nan=False`. One NaN anywhere in it
    fails the response. Callers still replace a known missing interval with
    None (not `[None, None]`) before this runs; this is the backstop for a
    number that was not handled at its source.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
