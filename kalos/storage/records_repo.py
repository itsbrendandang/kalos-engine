"""SQL for the v9 records tables (projects, project campaigns, factors, lineage).

Dialect-neutral (kalos/storage/backend.py): `?` placeholders, `RETURNING id`,
`ON CONFLICT ... DO UPDATE`, no SQLite- or PostgreSQL-only functions. Every
statement carries ``tenant = ?``; a row of another tenant is invisible exactly
like a missing one. Functions take a `Conn` inside the caller's transaction and
return plain dicts; they hold no rules (kalos/records does) and never import the
domain. `insert_audit` writes the audit row into the SAME transaction, so a
change that cannot be audited is not made.
"""
from __future__ import annotations

from typing import Any

from kalos.storage.backend import Conn

_PROJECT_COLS = (
    "id", "name", "objective", "status", "sponsor", "created_at", "created_by", "updated_at",
    "updated_by",
)
_FACTOR_COLS = (
    "name", "description", "lower_bound", "upper_bound", "unit", "dimension", "role",
    "updated_at", "updated_by",
)
_LINK_COLS = (
    "id", "parent_campaign_id", "parent_run_id", "child_campaign_id", "child_run_id", "relation",
    "parent_stage", "child_stage", "note", "created_at", "created_by",
)
_MAX_AUDIT_DETAIL = 500


def _clip(value: str | None, n: int = 256) -> str | None:
    return value[:n] if isinstance(value, str) else None


# --- audit ---------------------------------------------------------------------- #


def insert_audit(
    conn: Conn, *, tenant: str, action: str, outcome: str, subject: str | None,
    auth_method: str | None, now: float, campaign_id: str | None = None,
    request_id: str | None = None, detail: str | None = None,
) -> None:
    """One `audit_log` row (the v2 columns), inside the caller's transaction.
    Identifiers and short codes only: never a token, a credential or a name a
    user typed."""
    conn.execute(
        "INSERT INTO audit_log (ts, tenant, subject, auth_method, action, campaign_id, job_id, "
        "outcome, request_id, detail) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
        (now, tenant, _clip(subject), _clip(auth_method), _clip(action, 64),
         _clip(campaign_id, 64), _clip(outcome, 64), _clip(request_id, 128),
         _clip(detail, _MAX_AUDIT_DETAIL)),
    )


# --- projects ------------------------------------------------------------------- #


def _project(row: tuple[Any, ...] | None) -> dict[str, Any] | None:
    return dict(zip(_PROJECT_COLS, row)) if row is not None else None


def get_project(conn: Conn, *, tenant: str, project_id: str) -> dict[str, Any] | None:
    return _project(conn.execute(
        f"SELECT {', '.join(_PROJECT_COLS)} FROM projects WHERE tenant = ? AND id = ?",  # noqa: S608 - module-constant columns
        (tenant, project_id),
    ).fetchone())


def project_name_taken(conn: Conn, *, tenant: str, name: str, except_id: str | None = None) -> bool:
    row = conn.execute(
        "SELECT id FROM projects WHERE tenant = ? AND name = ?", (tenant, name)
    ).fetchone()
    return row is not None and str(row[0]) != except_id


def count_projects(conn: Conn, *, tenant: str) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM projects WHERE tenant = ?", (tenant,)).fetchone()[0])


def insert_project(
    conn: Conn, *, tenant: str, project_id: str, name: str, objective: str, status: str,
    sponsor: str | None, actor: str | None, now: float,
) -> dict[str, Any]:
    conn.execute(
        "INSERT INTO projects (id, tenant, name, objective, status, sponsor, created_at, "
        "created_by, updated_at, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, tenant, name, objective, status, sponsor, now, _clip(actor), now, _clip(actor)),
    )
    out = get_project(conn, tenant=tenant, project_id=project_id)
    assert out is not None
    return out


def update_project(
    conn: Conn, *, tenant: str, project_id: str, name: str, objective: str, status: str,
    sponsor: str | None, actor: str | None, now: float,
) -> dict[str, Any] | None:
    conn.execute(
        "UPDATE projects SET name = ?, objective = ?, status = ?, sponsor = ?, updated_at = ?, "
        "updated_by = ? WHERE tenant = ? AND id = ?",
        (name, objective, status, sponsor, now, _clip(actor), tenant, project_id),
    )
    return get_project(conn, tenant=tenant, project_id=project_id)


def list_projects(
    conn: Conn, *, tenant: str, status: str | None, limit: int, offset: int,
) -> tuple[list[dict[str, Any]], bool]:
    """A page of projects, newest update first; (rows, has_more)."""
    sql = f"SELECT {', '.join(_PROJECT_COLS)} FROM projects WHERE tenant = ?"  # noqa: S608 - module-constant columns
    params: list[Any] = [tenant]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY updated_at DESC, id LIMIT ? OFFSET ?"
    params += [limit + 1, offset]
    rows = [_project(r) for r in conn.execute(sql, params).fetchall()]
    return [r for r in rows[:limit] if r is not None], len(rows) > limit


# --- project campaigns ----------------------------------------------------------- #


def campaign_project(conn: Conn, *, tenant: str, campaign_id: str) -> str | None:
    row = conn.execute(
        "SELECT project_id FROM project_campaigns WHERE tenant = ? AND campaign_id = ?",
        (tenant, campaign_id),
    ).fetchone()
    return str(row[0]) if row is not None else None


def project_campaign_ids(conn: Conn, *, tenant: str, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT campaign_id, added_at, added_by FROM project_campaigns "
        "WHERE tenant = ? AND project_id = ? ORDER BY added_at, campaign_id",
        (tenant, project_id),
    ).fetchall()
    return [{"campaign_id": r[0], "added_at": r[1], "added_by": r[2]} for r in rows]


def project_campaign_counts(conn: Conn, *, tenant: str, project_ids: list[str]) -> dict[str, int]:
    if not project_ids:
        return {}
    marks = ", ".join("?" for _ in project_ids)
    rows = conn.execute(
        f"SELECT project_id, COUNT(*) FROM project_campaigns WHERE tenant = ? AND project_id IN ({marks}) "  # noqa: S608 - placeholders only
        "GROUP BY project_id",
        (tenant, *project_ids),
    ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


def attach_campaign(
    conn: Conn, *, tenant: str, project_id: str, campaign_id: str, actor: str | None, now: float,
) -> None:
    conn.execute(
        "INSERT INTO project_campaigns (tenant, campaign_id, project_id, added_at, added_by) "
        "VALUES (?, ?, ?, ?, ?)",
        (tenant, campaign_id, project_id, now, _clip(actor)),
    )


def detach_campaign(conn: Conn, *, tenant: str, project_id: str, campaign_id: str) -> bool:
    cur = conn.execute(
        "DELETE FROM project_campaigns WHERE tenant = ? AND project_id = ? AND campaign_id = ?",
        (tenant, project_id, campaign_id),
    )
    return int(cur.rowcount or 0) > 0


# --- factors --------------------------------------------------------------------- #


def list_factors(conn: Conn, *, tenant: str, project_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT {', '.join(_FACTOR_COLS)} FROM project_factors WHERE tenant = ? AND project_id = ? "  # noqa: S608 - module-constant columns
        "ORDER BY name",
        (tenant, project_id),
    ).fetchall()
    return [dict(zip(_FACTOR_COLS, r)) for r in rows]


def factor_exists(conn: Conn, *, tenant: str, project_id: str, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM project_factors WHERE tenant = ? AND project_id = ? AND name = ?",
        (tenant, project_id, name),
    ).fetchone() is not None


def count_factors(conn: Conn, *, tenant: str, project_id: str) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM project_factors WHERE tenant = ? AND project_id = ?", (tenant, project_id)
    ).fetchone()[0])


def upsert_factor(
    conn: Conn, *, tenant: str, project_id: str, name: str, description: str | None, lower: float,
    upper: float, unit: str | None, dimension: str | None, role: str, actor: str | None, now: float,
) -> None:
    conn.execute(
        "INSERT INTO project_factors (tenant, project_id, name, description, lower_bound, "
        "upper_bound, unit, dimension, role, updated_at, updated_by) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (tenant, project_id, name) DO UPDATE SET description = excluded.description, "
        "lower_bound = excluded.lower_bound, upper_bound = excluded.upper_bound, "
        "unit = excluded.unit, dimension = excluded.dimension, role = excluded.role, "
        "updated_at = excluded.updated_at, updated_by = excluded.updated_by",
        (tenant, project_id, name, description, lower, upper, unit, dimension, role, now, _clip(actor)),
    )


def delete_factor(conn: Conn, *, tenant: str, project_id: str, name: str) -> bool:
    cur = conn.execute(
        "DELETE FROM project_factors WHERE tenant = ? AND project_id = ? AND name = ?",
        (tenant, project_id, name),
    )
    return int(cur.rowcount or 0) > 0


# --- lineage --------------------------------------------------------------------- #

_ACTIVE = (
    f"SELECT {', '.join('l.' + c for c in _LINK_COLS)} FROM lineage_links l "  # noqa: S608 - module-constant columns
    "WHERE l.tenant = ? AND NOT EXISTS (SELECT 1 FROM lineage_retractions r "
    "WHERE r.tenant = l.tenant AND r.link_id = l.id)"
)


def active_links(conn: Conn, *, tenant: str, limit: int) -> list[dict[str, Any]]:
    """The tenant's active (not retracted) links, oldest first, at most `limit`."""
    rows = conn.execute(_ACTIVE + " ORDER BY l.id LIMIT ?", (tenant, limit)).fetchall()
    return [dict(zip(_LINK_COLS, r)) for r in rows]


def active_links_of_campaigns(
    conn: Conn, *, tenant: str, campaign_ids: list[str], limit: int,
) -> list[dict[str, Any]]:
    """Active links with BOTH ends among `campaign_ids` (a project's lineage)."""
    if not campaign_ids:
        return []
    marks = ", ".join("?" for _ in campaign_ids)
    rows = conn.execute(
        _ACTIVE + f" AND l.parent_campaign_id IN ({marks}) AND l.child_campaign_id IN ({marks}) "
        "ORDER BY l.id LIMIT ?",
        (tenant, *campaign_ids, *campaign_ids, limit),
    ).fetchall()
    return [dict(zip(_LINK_COLS, r)) for r in rows]


def count_links(conn: Conn, *, tenant: str) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM lineage_links WHERE tenant = ?", (tenant,)).fetchone()[0])


def get_link(conn: Conn, *, tenant: str, link_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        f"SELECT {', '.join(_LINK_COLS)} FROM lineage_links WHERE tenant = ? AND id = ?",  # noqa: S608 - module-constant columns
        (tenant, link_id),
    ).fetchone()
    if row is None:
        return None
    out = dict(zip(_LINK_COLS, row))
    ret = conn.execute(
        "SELECT reason, created_at, created_by FROM lineage_retractions WHERE tenant = ? AND link_id = ?",
        (tenant, link_id),
    ).fetchone()
    out["retraction"] = (
        {"reason": ret[0], "retracted_at": ret[1], "retracted_by": ret[2]} if ret is not None else None
    )
    return out


def insert_link(
    conn: Conn, *, tenant: str, parent_campaign_id: str, parent_run_id: str, child_campaign_id: str,
    child_run_id: str, relation: str, parent_stage: str | None, child_stage: str | None,
    note: str | None, actor: str | None, now: float,
) -> int:
    row = conn.execute(
        "INSERT INTO lineage_links (tenant, parent_campaign_id, parent_run_id, child_campaign_id, "
        "child_run_id, relation, parent_stage, child_stage, note, created_at, created_by) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (tenant, parent_campaign_id, parent_run_id, child_campaign_id, child_run_id, relation,
         parent_stage, child_stage, note, now, _clip(actor)),
    ).fetchone()
    return int(row[0])


def insert_retraction(
    conn: Conn, *, tenant: str, link_id: int, reason: str, actor: str | None, now: float,
) -> None:
    conn.execute(
        "INSERT INTO lineage_retractions (tenant, link_id, reason, created_at, created_by) "
        "VALUES (?, ?, ?, ?, ?)",
        (tenant, link_id, reason, now, _clip(actor)),
    )


__all__ = [
    "active_links", "active_links_of_campaigns", "attach_campaign", "campaign_project",
    "count_factors", "count_links", "count_projects", "delete_factor", "detach_campaign",
    "factor_exists", "get_link", "get_project", "insert_audit", "insert_link", "insert_project",
    "insert_retraction", "list_factors", "list_projects", "project_campaign_counts",
    "project_campaign_ids", "project_name_taken", "update_project", "upsert_factor",
]
