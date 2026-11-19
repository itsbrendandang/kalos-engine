"""`GET /api/campaigns?summary=true`: the campaigns list with per-campaign summary
fields, so the web home renders every card from ONE request.

Efficiency: a page of campaigns is summarized by four SQL statements in one read
transaction (the page with its state/analysis fields, the setup/stages settings,
the newest gate decision, the active job), not one request or one Python JSON
parse per campaign. The big JSON blobs (campaign state with `base_rows`, the
published analysis) are read with SQLite's JSON functions for the few fields the
card needs; Python only parses a blob when SQLite reports it as invalid JSON
(e.g. a legacy blob holding a bare `NaN`), and then only that campaign's.

Every statement carries `tenant = ?`; joins between tables also match on tenant.

Honesty: `best_measured` is the best MEASURED target value over the campaign's
base data (uploaded rows plus folded results), never a prediction. Its direction
is the campaign setup's goal (`campaign_settings` kind `setup`, `value.goal`):
`minimize` uses min, anything else (no setup, or no recognized goal) uses max and
says so in `goal.source = "default"`. Analysis fields are copied as stored; an
analysis computed before a field existed reads `null` for it.
"""
from __future__ import annotations

import base64
import binascii
import json
import math
import sqlite3
from typing import Any

from kalos.portal.campaign import CampaignStore
from kalos.portal.migrate import CAMPAIGN_ID_PATTERN
from kalos.portal.tenancy import validate_tenant_id

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
_MAX_CURSOR_CHARS = 256
# JSON5 Infinity parses as a real in SQLite; this keeps only finite numbers.
_FINITE = "r.value BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308"
_ANALYSIS_PATHS = (
    "$.engine_version", "$.cv_spearman", "$.conformal_q", "$.conformal_coverage",
    "$.reliability.clears_floor", "$.reliability.ci_excludes_zero",
    "$.reliability.spearman", "$.reliability.ci95", "$.reliability.spearman_floor",
    "$.reliability.producer_clears_floor",
)


class InvalidCursor(ValueError):
    """The pagination cursor is not one this server issued."""


def encode_cursor(updated_at: float, campaign_id: str) -> str:
    raw = json.dumps([updated_at, campaign_id], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[float, str]:
    """`(updated_at, id)` of the last item of the previous page. InvalidCursor for
    anything malformed (the input is never echoed)."""
    if not isinstance(cursor, str) or not cursor or len(cursor) > _MAX_CURSOR_CHARS:
        raise InvalidCursor()
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        data = json.loads(raw)
    except (binascii.Error, ValueError):
        raise InvalidCursor() from None
    if (not isinstance(data, list) or len(data) != 2 or isinstance(data[0], bool)
            or not isinstance(data[0], (int, float)) or not math.isfinite(data[0])
            or not isinstance(data[1], str) or CAMPAIGN_ID_PATTERN.fullmatch(data[1]) is None):
        raise InvalidCursor()
    return float(data[0]), data[1]


# --- SQL ------------------------------------------------------------------------ #

_PAGE_SQL = """
WITH page AS MATERIALIZED (
    SELECT id, name, created_by, created_at, updated_at, archived_at, generation,
           state IS NOT NULL AS seeded
    FROM campaigns
    WHERE tenant = ? {archived} {after}
    ORDER BY updated_at DESC, id
    LIMIT ?
),
cs AS MATERIALIZED (
    SELECT c.id, json_valid(c.state) AS ok,
           CASE WHEN json_valid(c.state) THEN json_extract(c.state, '$.target') END AS target,
           CASE WHEN json_valid(c.state) THEN json_extract(c.state, '$.round') END AS round,
           CASE WHEN json_valid(c.state)
                THEN json_array_length(c.state, '$.base_rows') END AS n_base,
           CASE WHEN json_valid(c.state)
                THEN json_array_length(c.state, '$.pending') END AS n_pending,
           CASE WHEN json_valid(c.state) THEN (
               SELECT count(*) FROM json_each(c.state, '$.pending') pe
               WHERE coalesce(json_type(pe.value, '$.result'), 'null') = 'null'
           ) END AS n_awaiting
    FROM campaigns c JOIN page p ON p.id = c.id
    WHERE c.tenant = ? AND c.state IS NOT NULL
),
best AS (
    SELECT c.id, max(r.value) AS mx, min(r.value) AS mn
    FROM cs JOIN campaigns c ON c.id = cs.id AND c.tenant = ?,
         json_each(CASE WHEN cs.ok THEN c.state ELSE '{{}}' END, '$.base_rows') b,
         json_each(CASE WHEN b.type = 'object' THEN b.value ELSE '{{}}' END) r
    WHERE cs.ok AND r.key = cs.target AND r.type IN ('integer', 'real') AND {finite}
    GROUP BY c.id
),
an AS MATERIALIZED (
    SELECT a.campaign_id, a.updated_at, a.campaign_generation, json_valid(a.state) AS ok,
           CASE WHEN json_valid(a.state) THEN json_extract(a.state, {paths}) END AS ax,
           CASE WHEN json_valid(a.state)
                THEN json_array_length(a.state, '$.proposals') END AS n_proposals
    FROM analyses a JOIN page p ON p.id = a.campaign_id
    WHERE a.tenant = ?
)
SELECT p.id, p.name, p.created_by, p.created_at, p.updated_at, p.archived_at, p.seeded,
       cs.ok, cs.target, cs.round, cs.n_base, cs.n_pending, cs.n_awaiting,
       best.mx, best.mn,
       an.campaign_id IS NOT NULL, an.updated_at,
       an.campaign_generation IS NOT NULL AND an.campaign_generation IS p.generation,
       an.ok, an.ax, an.n_proposals
FROM page p
LEFT JOIN cs ON cs.id = p.id
LEFT JOIN best ON best.id = p.id
LEFT JOIN an ON an.campaign_id = p.id
ORDER BY p.updated_at DESC, p.id
"""


def _in(ids: list[str]) -> str:
    return ", ".join("?" for _ in ids)


def _settings(conn: sqlite3.Connection, tenant: str, ids: list[str]) -> dict[str, dict[str, Any]]:
    """`{campaign_id: {"goal": ..., "current_stage": ...}}` from v6 settings."""
    rows = conn.execute(
        "SELECT campaign_id, kind, CASE WHEN json_valid(value) "  # noqa: S608 - only '?' placeholders are interpolated
        "THEN json_extract(value, '$.goal', '$.current_stage') END "
        f"FROM campaign_settings WHERE tenant = ? AND campaign_id IN ({_in(ids)})",
        (tenant, *ids),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for cid, kind, extracted in rows:
        vals = _loads_list(extracted, 2)
        entry = out.setdefault(cid, {})
        if kind == "setup":
            entry["goal"] = vals[0]
        elif kind == "stages":
            entry["current_stage"] = vals[1]
    return out


def _latest_gates(
    conn: sqlite3.Connection, tenant: str, ids: list[str]
) -> dict[str, dict[str, Any]]:
    """The newest gate decision (by append order) per campaign."""
    rows = conn.execute(
        "SELECT g.campaign_id, g.gate, g.decision, g.decided_at FROM gate_decisions g "  # noqa: S608 - only '?' placeholders are interpolated
        "WHERE g.tenant = ? AND g.id = (SELECT max(g2.id) FROM gate_decisions g2 "
        "WHERE g2.tenant = ? AND g2.campaign_id = g.campaign_id) "
        f"AND g.campaign_id IN ({_in(ids)})",
        (tenant, tenant, *ids),
    ).fetchall()
    return {cid: {"gate": gate, "decision": decision, "decided_at": at}
            for cid, gate, decision, at in rows}


def _active_jobs(
    conn: sqlite3.Connection, tenant: str, ids: list[str]
) -> dict[str, dict[str, Any]]:
    """The newest queued/running job per campaign."""
    rows = conn.execute(
        "SELECT campaign_id, id, params, state, progress, stage FROM jobs "  # noqa: S608 - only '?' placeholders are interpolated
        "WHERE tenant = ? AND state IN ('queued', 'running') "
        f"AND campaign_id IN ({_in(ids)}) ORDER BY created_at, id",
        (tenant, *ids),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    from kalos.portal.jobs import job_kind  # local: jobs imports the campaign store

    for cid, jid, params, state, progress, stage in rows:  # newest last, so it wins
        # `kind` as the jobs API reports it: 'reanalyze' or 'analysis' (params.mode).
        out[cid] = {"id": jid, "kind": job_kind({"params": params}), "state": state,
                    "progress": progress,
                    "stage": stage}
    return out


# --- shaping --------------------------------------------------------------------- #


def _loads_list(text: object, n: int) -> list[Any]:
    if isinstance(text, str):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, list) and len(data) == n:
            return data
    return [None] * n


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _state_fallback(conn: sqlite3.Connection, tenant: str, cid: str) -> dict[str, Any] | None:
    """Python read of ONE campaign state SQLite could not parse (rare: legacy
    NaN). None when Python cannot parse it either."""
    row = conn.execute(
        "SELECT state FROM campaigns WHERE id = ? AND tenant = ?", (cid, tenant)
    ).fetchone()
    try:
        data = json.loads(row[0]) if row and row[0] is not None else None
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    target = data.get("target")
    raw_base, raw_pending = data.get("base_rows"), data.get("pending")
    base: list[Any] = raw_base if isinstance(raw_base, list) else []
    pending: list[Any] = raw_pending if isinstance(raw_pending, list) else []
    vals = [v for r in base if isinstance(r, dict) and (v := _finite(r.get(target))) is not None]
    awaiting = sum(1 for r in pending if isinstance(r, dict) and r.get("result") is None)
    return {"target": target if isinstance(target, str) else None,
            "round": data.get("round"), "n_base": len(base), "n_pending": len(pending),
            "n_awaiting": awaiting, "mx": max(vals) if vals else None,
            "mn": min(vals) if vals else None}


def _analysis_fallback(conn: sqlite3.Connection, tenant: str, cid: str) -> list[Any]:
    row = conn.execute(
        "SELECT state FROM analyses WHERE campaign_id = ? AND tenant = ?", (cid, tenant)
    ).fetchone()
    try:
        data = json.loads(row[0]) if row else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return [None] * (len(_ANALYSIS_PATHS) + 1)
    rel = data.get("reliability") if isinstance(data.get("reliability"), dict) else {}
    vals = []
    for path in _ANALYSIS_PATHS:
        keys = path[2:].split(".")
        src: Any = rel if keys[0] == "reliability" else data
        vals.append(src.get(keys[-1]) if isinstance(src, dict) else None)
    props = data.get("proposals")
    return [*vals, len(props) if isinstance(props, list) else None]


def _goal(raw: Any) -> dict[str, str]:
    if raw in ("maximize", "minimize"):
        return {"direction": raw, "source": "setup"}
    return {"direction": "maximize", "source": "default"}


def _number(value: Any) -> float | int | None:
    return value if _finite(value) is not None else None


def _analysis_block(present: bool, updated: Any, in_sync: Any, vals: list[Any]) -> dict[str, Any]:
    if not present:
        return {"present": False}
    (engine_version, cv_spearman, conformal_q, conformal_coverage, clears_floor,
     ci_excludes_zero, spearman, ci95, spearman_floor, producer_clears_floor,
     n_proposals) = vals
    return {
        "present": True,
        "updated": updated,
        "in_sync": bool(in_sync),
        "engine_version": engine_version if isinstance(engine_version, str) else None,
        "cv_spearman": _number(cv_spearman),
        "conformal_q": _number(conformal_q),
        "conformal_coverage": _number(conformal_coverage),
        "reliability": {
            "clears_floor": _bool_or_none(clears_floor),
            "ci_excludes_zero": _bool_or_none(ci_excludes_zero),
            "spearman": _number(spearman),
            "ci95": ci95 if isinstance(ci95, list) else None,
            "spearman_floor": _number(spearman_floor),
            "producer_clears_floor": _bool_or_none(producer_clears_floor),
        },
        "n_proposals": n_proposals if isinstance(n_proposals, int) else None,
    }


def list_campaign_summaries(
    store: CampaignStore, *, tenant: str, include_archived: bool = False,
    limit: int = DEFAULT_LIMIT, cursor: str | None = None,
) -> dict[str, Any]:
    """One page of `tenant`'s campaigns, newest-updated first, each with its
    summary fields. `{"campaigns": [...], "next_cursor": str | None, "limit": n}`.
    Raises InvalidCursor for a malformed cursor and ValueError for a bad limit."""
    validate_tenant_id(tenant)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}")
    after = decode_cursor(cursor) if cursor is not None else None
    sql = _PAGE_SQL.format(
        archived="" if include_archived else "AND archived_at IS NULL",
        after="AND (updated_at < ? OR (updated_at = ? AND id > ?))" if after else "",
        finite=_FINITE,
        paths=", ".join(f"'{p}'" for p in _ANALYSIS_PATHS),
    )
    params: list[Any] = [tenant]
    if after:
        params += [after[0], after[0], after[1]]
    params += [limit + 1, tenant, tenant, tenant]

    with store._txn(write=False) as conn:
        default = store._default_id(conn, tenant)
        rows = conn.execute(sql, params).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        ids = [r[0] for r in rows]
        settings = _settings(conn, tenant, ids) if ids else {}
        gates = _latest_gates(conn, tenant, ids) if ids else {}
        jobs = _active_jobs(conn, tenant, ids) if ids else {}
        items = [_item(conn, tenant, r, default, settings, gates, jobs) for r in rows]

    next_cursor = encode_cursor(rows[-1][4], rows[-1][0]) if more and rows else None
    return {"campaigns": items, "next_cursor": next_cursor, "limit": limit}


def _item(
    conn: sqlite3.Connection, tenant: str, r: tuple[Any, ...], default: str | None,
    settings: dict[str, dict[str, Any]], gates: dict[str, dict[str, Any]],
    jobs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    (cid, name, created_by, created_at, updated_at, archived_at, seeded,
     s_ok, target, rnd, n_base, n_pending, n_awaiting, mx, mn,
     has_analysis, a_updated, in_sync, a_ok, ax, n_proposals) = r
    if seeded and not s_ok:
        fb = _state_fallback(conn, tenant, cid)
        if fb is not None:
            target, rnd, n_base, n_pending, n_awaiting, mx, mn = (
                fb["target"], fb["round"], fb["n_base"], fb["n_pending"], fb["n_awaiting"],
                fb["mx"], fb["mn"])
    if has_analysis and not a_ok:
        a_vals = _analysis_fallback(conn, tenant, cid)
    else:
        a_vals = [*_loads_list(ax, len(_ANALYSIS_PATHS)), n_proposals]

    cfg = settings.get(cid, {})
    goal = _goal(cfg.get("goal"))
    minimize = goal["direction"] == "minimize"
    best = _finite(mn if minimize else mx)
    if not seeded:
        n_base, n_pending, n_awaiting = 0, 0, 0
    readable = isinstance(n_base, int)
    best_block: dict[str, Any] = {
        "value": best, "basis": "measured", "direction": "min" if minimize else "max",
    }
    if best is None:
        best_block["reason"] = (
            "no data uploaded yet" if not seeded else
            "campaign state unreadable" if not readable else
            "no finite numeric target value in the measured data"
        )
    return {
        "id": cid,
        "name": name,
        "created_by": created_by,
        "created_at": created_at,
        "updated_at": updated_at,
        "archived_at": archived_at,
        "archived": archived_at is not None,
        "seeded": bool(seeded),
        "is_default": cid == default,
        "has_data": readable and n_base > 0,
        "target": target if isinstance(target, str) else None,
        "n_base": n_base if readable else None,
        "round": rnd if isinstance(rnd, int) and not isinstance(rnd, bool) else None,
        "n_awaiting": n_awaiting if isinstance(n_awaiting, int) else None,
        "n_measured": (n_pending - n_awaiting
                       if isinstance(n_pending, int) and isinstance(n_awaiting, int) else None),
        "goal": goal,
        "best_measured": best_block,
        "analysis": _analysis_block(bool(has_analysis), a_updated, in_sync, a_vals),
        "current_stage": cfg.get("current_stage"),
        "latest_gate_decision": gates.get(cid),
        "active_job": jobs.get(cid),
    }


__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "InvalidCursor", "decode_cursor", "encode_cursor",
           "list_campaign_summaries"]
