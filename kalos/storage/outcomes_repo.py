"""Read-only SQL over the campaign result tables, for the records outcome views.

Reads `run_results` (every measured version, schema v5) and `run_predictions`
(the prediction recorded at start, schema v8). Never writes: a result is
recorded only through `CampaignStore.set_result`, which keeps the campaign state
and the history in one transaction. These tables live with the campaigns
(`portal.db` today; behind `storage/campaign_repo.py` after REORG_PLAN R12, when
these queries move next to it). Every statement carries ``tenant = ?`` and an
explicit campaign id list.
"""
from __future__ import annotations

from typing import Any

from kalos.storage.backend import Conn

_VERSION_COLS = ("campaign_id", "run_id", "version", "value", "previous_value", "reason", "recorded_at")
_PRED_COLS = (
    "campaign_id", "run_id", "round", "available", "unavailable_reason", "pred", "lo_90", "hi_90",
    "method_90", "basis_90", "lo_95", "hi_95", "method_95", "basis_95", "model_version",
    "recorded_at",
)
MAX_CAMPAIGNS_PER_QUERY = 200


def _marks(n: int) -> str:
    return ", ".join("?" for _ in range(n))


def run_keys(
    conn: Conn, *, tenant: str, campaign_ids: list[str], limit: int, offset: int,
) -> tuple[list[tuple[str, str]], bool]:
    """A page of (campaign_id, run_id) pairs that have a recorded prediction or
    a recorded result, ordered by campaign then run; (rows, has_more)."""
    if not campaign_ids:
        return [], False
    ids = campaign_ids[:MAX_CAMPAIGNS_PER_QUERY]
    m = _marks(len(ids))
    rows = conn.execute(
        f"SELECT campaign_id, run_id FROM run_results WHERE tenant = ? AND campaign_id IN ({m}) "  # noqa: S608 - placeholders only
        f"UNION SELECT campaign_id, run_id FROM run_predictions WHERE tenant = ? AND campaign_id IN ({m}) "
        "ORDER BY 1, 2 LIMIT ? OFFSET ?",
        (tenant, *ids, tenant, *ids, limit + 1, offset),
    ).fetchall()
    keys = [(str(c), str(r)) for c, r in rows]
    return keys[:limit], len(keys) > limit


def versions_for(
    conn: Conn, *, tenant: str, keys: list[tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Every result version of the given runs, oldest first."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cid in sorted({c for c, _ in keys}):
        runs = sorted({r for c, r in keys if c == cid})
        rows = conn.execute(
            f"SELECT {', '.join(_VERSION_COLS)} FROM run_results WHERE tenant = ? AND campaign_id = ? "  # noqa: S608 - module-constant columns, placeholders
            f"AND run_id IN ({_marks(len(runs))}) ORDER BY run_id, version",
            (tenant, cid, *runs),
        ).fetchall()
        for r in rows:
            d = dict(zip(_VERSION_COLS, r))
            out.setdefault((str(d["campaign_id"]), str(d["run_id"])), []).append(d)
    return out


def predictions_for(
    conn: Conn, *, tenant: str, keys: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """The prediction recorded at start for each of the given runs (if any)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for cid in sorted({c for c, _ in keys}):
        runs = sorted({r for c, r in keys if c == cid})
        rows = conn.execute(
            f"SELECT {', '.join(_PRED_COLS)} FROM run_predictions WHERE tenant = ? AND campaign_id = ? "  # noqa: S608 - module-constant columns, placeholders
            f"AND run_id IN ({_marks(len(runs))})",
            (tenant, cid, *runs),
        ).fetchall()
        for r in rows:
            d = dict(zip(_PRED_COLS, r))
            out[(str(d["campaign_id"]), str(d["run_id"]))] = d
    return out


def run_known(conn: Conn, *, tenant: str, campaign_id: str, run_id: str) -> bool:
    """Whether the campaign has a recorded prediction or result for `run_id`."""
    return conn.execute(
        "SELECT 1 FROM run_predictions WHERE tenant = ? AND campaign_id = ? AND run_id = ? "
        "UNION SELECT 1 FROM run_results WHERE tenant = ? AND campaign_id = ? AND run_id = ?",
        (tenant, campaign_id, run_id, tenant, campaign_id, run_id),
    ).fetchone() is not None


__all__ = ["predictions_for", "run_keys", "run_known", "versions_for"]
