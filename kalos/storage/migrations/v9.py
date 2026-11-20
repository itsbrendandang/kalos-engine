"""Schema v9: records above and between campaigns (ported from Voyager).

Five tables, all tenant-scoped (every statement carries ``tenant = ?``):

- ``projects``: a grouping ABOVE campaigns (Voyager ``Project``): a name, an
  objective, a status (drafting / in_progress / completed / terminated) and an
  optional free-text sponsor (Voyager's ``Customer`` CRM table is not ported).
  Text ids ``prj_<32 hex>``, like campaign ids, so ids cannot be enumerated.
- ``project_campaigns``: which campaigns a project groups. A campaign belongs to
  at most one project (primary key ``(tenant, campaign_id)``).
- ``project_factors``: a project's factor definitions (Voyager ``Factor``): name,
  description, lower/upper bound, unit from the units registry
  (`kalos.normalize.unit_registry`). They map onto a campaign's setup bounds; they
  never change a campaign by themselves.
- ``lineage_links`` (APPEND-ONLY): a parent -> child edge between two campaigns,
  or two runs of them, across rounds and scale stages (Voyager ``ExperimentLink``,
  whose experiments are Kalos campaigns and whose iterations are rounds).
- ``lineage_retractions`` (APPEND-ONLY): a link withdrawn, with a reason. A link
  is never edited or deleted; a retraction is a new row.

There is no foreign key from these tables to ``campaigns``: with
``KALOS_DATABASE_URL`` set they live in PostgreSQL while the campaign tables are
still in ``portal.db`` (docs/MIGRATIONS.md, "PostgreSQL backend"). The HTTP layer
resolves every campaign id under the caller's tenant before it writes one.

The DDL is given for both dialects. SQLite gets it inside `migrate.py`'s
``BEGIN IMMEDIATE`` after the verified backup (`apply_sqlite`); PostgreSQL gets
it in `pg_runner`'s one transaction (`apply_postgres`). `down` on SQLite renames
the tables aside (``<name>_v9_rolledback``), never drops them.

Torch-free and FastAPI-free (the migration CLI imports it).
"""
from __future__ import annotations

import sqlite3
from typing import Any

V9 = 9
V9_NAME = "records: projects, project campaigns, factors, lineage links and retractions"

TABLES = (
    "projects", "project_campaigns", "project_factors", "lineage_links", "lineage_retractions",
)
APPEND_ONLY_TABLES = ("lineage_links", "lineage_retractions")
SQLITE_TRIGGERS = tuple(
    f"{t}_no_{op}" for t in APPEND_ONLY_TABLES for op in ("update", "delete")
)
INDEXES = (
    "projects_tenant_idx", "project_campaigns_project_idx", "lineage_links_parent_idx",
    "lineage_links_child_idx",
)

PROJECT_STATUSES = ("drafting", "in_progress", "completed", "terminated")
LINK_RELATIONS = ("next_round", "scale_up", "scale_down", "follow_up", "replicate", "derived")
FACTOR_ROLES = ("setting", "covariate")


def _status_check() -> str:
    return ", ".join(f"'{s}'" for s in PROJECT_STATUSES)


def _relation_check() -> str:
    return ", ".join(f"'{r}'" for r in LINK_RELATIONS)


def _role_check() -> str:
    return ", ".join(f"'{r}'" for r in FACTOR_ROLES)


def _tables_ddl(*, serial: str, real: str) -> tuple[str, ...]:
    """The table DDL, parameterised by the two type spellings that differ:
    `serial` (an auto id column) and `real` (a float)."""
    return (
        f"""
        CREATE TABLE projects (
            id TEXT PRIMARY KEY,
            tenant TEXT NOT NULL,
            name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 80),
            objective TEXT NOT NULL CHECK (length(objective) BETWEEN 1 AND 500),
            status TEXT NOT NULL CHECK (status IN ({_status_check()})),
            sponsor TEXT CHECK (sponsor IS NULL OR length(sponsor) BETWEEN 1 AND 120),
            created_at {real} NOT NULL,
            created_by TEXT,
            updated_at {real} NOT NULL,
            updated_by TEXT,
            UNIQUE (id, tenant),
            UNIQUE (tenant, name)
        )
        """,
        "CREATE INDEX projects_tenant_idx ON projects (tenant, updated_at)",
        f"""
        CREATE TABLE project_campaigns (
            tenant TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            added_at {real} NOT NULL,
            added_by TEXT,
            PRIMARY KEY (tenant, campaign_id),
            FOREIGN KEY (project_id, tenant) REFERENCES projects (id, tenant)
        )
        """,
        "CREATE INDEX project_campaigns_project_idx ON project_campaigns (tenant, project_id)",
        f"""
        CREATE TABLE project_factors (
            tenant TEXT NOT NULL,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 64),
            description TEXT CHECK (description IS NULL OR length(description) <= 300),
            lower_bound {real} NOT NULL,
            upper_bound {real} NOT NULL,
            unit TEXT,
            dimension TEXT,
            role TEXT NOT NULL DEFAULT 'setting' CHECK (role IN ({_role_check()})),
            updated_at {real} NOT NULL,
            updated_by TEXT,
            PRIMARY KEY (tenant, project_id, name),
            CHECK (lower_bound < upper_bound),
            FOREIGN KEY (project_id, tenant) REFERENCES projects (id, tenant)
        )
        """,
        f"""
        CREATE TABLE lineage_links (
            id {serial},
            tenant TEXT NOT NULL,
            parent_campaign_id TEXT NOT NULL,
            parent_run_id TEXT NOT NULL DEFAULT '',
            child_campaign_id TEXT NOT NULL,
            child_run_id TEXT NOT NULL DEFAULT '',
            relation TEXT NOT NULL CHECK (relation IN ({_relation_check()})),
            parent_stage TEXT,
            child_stage TEXT,
            note TEXT CHECK (note IS NULL OR length(note) <= 500),
            created_at {real} NOT NULL,
            created_by TEXT,
            UNIQUE (id, tenant),
            CHECK (NOT (parent_campaign_id = child_campaign_id AND parent_run_id = child_run_id))
        )
        """,
        "CREATE INDEX lineage_links_parent_idx ON lineage_links (tenant, parent_campaign_id)",
        "CREATE INDEX lineage_links_child_idx ON lineage_links (tenant, child_campaign_id)",
        f"""
        CREATE TABLE lineage_retractions (
            id {serial},
            tenant TEXT NOT NULL,
            link_id BIGINT NOT NULL,
            reason TEXT NOT NULL CHECK (length(trim(reason)) BETWEEN 1 AND 500),
            created_at {real} NOT NULL,
            created_by TEXT,
            UNIQUE (tenant, link_id),
            FOREIGN KEY (link_id, tenant) REFERENCES lineage_links (id, tenant)
        )
        """,
    )


SQLITE_DDL: tuple[str, ...] = _tables_ddl(
    serial="INTEGER PRIMARY KEY AUTOINCREMENT", real="REAL"
) + tuple(
    f"""
    CREATE TRIGGER {table}_no_{op} BEFORE {op.upper()} ON {table}
    BEGIN
        SELECT RAISE(ABORT, '{table} is append-only: {op.upper()} is rejected');
    END
    """
    for table in APPEND_ONLY_TABLES for op in ("update", "delete")
)

# PostgreSQL: the same tables; append-only is a trigger function raising
# SQLSTATE 23000 (psycopg's IntegrityError, the class SQLite raises) on UPDATE
# and DELETE, plus a statement trigger on TRUNCATE, which skips row triggers.
PG_APPEND_ONLY_FUNCTION = "kalos_reject_change"
PG_FUNCTION_DDL = f"""
CREATE OR REPLACE FUNCTION {PG_APPEND_ONLY_FUNCTION}() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only: % is rejected', TG_TABLE_NAME, TG_OP
        USING ERRCODE = '23000';
END
$$
"""


def pg_append_only_ddl(table: str) -> tuple[str, ...]:
    return (
        f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {PG_APPEND_ONLY_FUNCTION}()",
        f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {PG_APPEND_ONLY_FUNCTION}()",
        f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {PG_APPEND_ONLY_FUNCTION}()",
    )


PG_TRIGGERS = tuple(
    f"{t}_no_{op}" for t in APPEND_ONLY_TABLES for op in ("update", "delete", "truncate")
)
PG_DDL: tuple[str, ...] = (
    _tables_ddl(serial="BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY", real="DOUBLE PRECISION")
    + tuple(ddl for t in APPEND_ONLY_TABLES for ddl in pg_append_only_ddl(t))
)


def apply_sqlite(conn: sqlite3.Connection) -> list[str]:
    """v8 -> v9 inside the caller's BEGIN IMMEDIATE: create the five tables, the
    indexes and the append-only triggers. Additive: no existing row changes.
    Returns warnings (none today)."""
    for ddl in SQLITE_DDL:
        conn.execute(ddl)
    return []


def apply_postgres(conn: Any) -> None:
    """The v9 tables on PostgreSQL, inside the caller's transaction (`pg_runner`).
    The append-only trigger function is created by the baseline step first."""
    for ddl in PG_DDL:
        conn.execute(ddl)


def render_up() -> list[str]:
    return [
        "  create: projects, project_campaigns, project_factors, and lineage_links + "
        "lineage_retractions (append-only: UPDATE/DELETE triggers) (schema v9, records); "
        "no existing row changes",
    ]


def _free_name(existing: set[str], base: str) -> str:
    name, n = base, 1
    while name in existing:
        n += 1
        name = f"{base}_{n}"
    return name


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    return {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def down_sqlite(conn: sqlite3.Connection) -> None:
    """v9 -> v8 inside the caller's transaction: drop the triggers and indexes,
    rename the five tables aside (`<name>_v9_rolledback`); no row is deleted."""
    for name in SQLITE_TRIGGERS:
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    for idx in INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {idx}")
    tables = _sqlite_tables(conn)
    # Children first, so a rename never leaves a foreign key pointing at a
    # renamed parent mid-way (SQLite rewrites FK references on rename).
    for table in reversed(TABLES):
        if table in tables:
            new = _free_name(tables, f"{table}_v9_rolledback")
            conn.execute(f'ALTER TABLE "{table}" RENAME TO "{new}"')
            tables.add(new)
    conn.execute("DELETE FROM schema_migrations WHERE version = ?", (V9,))


def render_down(conn: sqlite3.Connection) -> list[str]:
    tables = _sqlite_tables(conn)
    lines = [f"  drop: triggers {', '.join(SQLITE_TRIGGERS)} and indexes {', '.join(INDEXES)}"]
    for table in reversed(TABLES):
        if table in tables:
            (count,) = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()  # noqa: S608 - module-constant table names
            new = _free_name(tables, f"{table}_v9_rolledback")
            lines.append(f"  rename: {table} -> {new} ({count} rows kept, not dropped)")
            tables.add(new)
    return lines


def missing_sqlite(tables: set[str], triggers: set[str]) -> list[str]:
    """What a database recording v9 lacks (for `_assert_v2_shape`)."""
    return [t for t in TABLES if t not in tables] + [
        f"trigger {t}" for t in SQLITE_TRIGGERS if t not in triggers
    ]
