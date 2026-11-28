"""Read-only SQL queries and metadata helpers on the `jobs` table (storage layer).

Placed in `kalos.storage` so `campaigns`, `assist`, and `jobs` can all read job
metadata without importing upward into `kalos.jobs`.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

REANALYZE_MODE = "reanalyze"


def job_kind_from_params(params_text: Any) -> str:
    """`'reanalyze'` or `'analysis'`, from the serialized `params.mode` JSON."""
    try:
        data = json.loads(params_text or "{}")
    except (TypeError, ValueError):
        return "analysis"
    if isinstance(data, dict) and data.get("mode") == REANALYZE_MODE:
        return REANALYZE_MODE
    return "analysis"


def latest_job_refusal(store: Any, *, tenant: str, campaign_id: str) -> dict[str, Any] | None:
    """Return `{code, context, at}` if the newest job for `(tenant, campaign_id)` failed."""
    try:
        with store._connect() as conn:
            row = conn.execute(
                "SELECT state, error_code, finished_at FROM jobs "
                "WHERE tenant = ? AND campaign_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (tenant, campaign_id),
            ).fetchone()
    except (sqlite3.Error, AttributeError):
        return None
    if not row:
        return None
    state, error_code, finished_at = row
    if state != "failed" or not error_code:
        return None
    return {
        "code": error_code,
        "context": {},
        "at": float(finished_at) if isinstance(finished_at, (int, float)) else None,
    }


__all__ = ["REANALYZE_MODE", "job_kind_from_params", "latest_job_refusal"]
