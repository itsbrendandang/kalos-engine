"""Last-step guard on an analysis dict: no NaN or infinity reaches the client.

The analysis is serialized straight into a JSON response and into the
campaign store. JSON has no NaN or infinity, so one such float anywhere turns
a finished analysis into an HTTP 500 (replay F1: `cv_ci95 = [nan, nan]` on a
sheet with 2 CV groups). Each block should already emit None for a number it
could not form; this pass catches anything that slipped through, turns it
into None, and logs the key path so the source can be fixed.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

log = logging.getLogger("kalos.portal")


def _clean(value: Any, path: str, hits: list[str]) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return value
        hits.append(path)
        return None
    if isinstance(value, dict):
        return {k: _clean(v, f"{path}.{k}" if path else str(k), hits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, f"{path}[{i}]", hits) for i, v in enumerate(value)]
    return value


def sanitize_non_finite(result: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """`result` with every non-finite float replaced by None (tuples become
    lists, as JSON would make them anyway), plus the key paths that were
    replaced. Logs the top-level keys and the count when any were: only the
    engine's own field names, since nested keys can be the customer's
    column names."""
    hits: list[str] = []
    cleaned = _clean(result, "", hits)
    if hits:
        top = sorted({h.split(".")[0].split("[")[0] for h in hits})
        log.warning(
            "analysis had %d non-finite float(s), sent as null, under keys: %s",
            len(hits), ", ".join(top),
        )
    return cleaned, hits
