"""Kalos portal - the schema on PostgreSQL (`KALOS_DATABASE_URL=postgresql://...`).

The same schema versions as `kalos.portal.migrate` (the SQLite path), expressed
in PostgreSQL DDL, recorded in the same `schema_migrations` table:

  v2 campaigns, analyses, tenant_defaults, audit_log
  v3 jobs
  v4 audit_log is append-only (triggers)
  v5 run_results, append-only (triggers)

Differences from SQLite, all deliberate (docs/MIGRATIONS.md, "PostgreSQL backend"):

- No v1. PostgreSQL never held a v1 database, so there is no `latest/` import,
  no repair, no stuck shape, and `down` stops at v2.
- An EMPTY database is created at the current version on engine startup, in one
  transaction (PostgreSQL DDL is transactional, so a failure leaves nothing).
  A database at an OLDER version is refused at startup: the operator takes a
  `pg_dump` and runs `python -m kalos.portal.migrate up`. The engine cannot
  take and verify a backup of a server database itself, so it never upgrades
  one silently.
- Append-only is a trigger function raising SQLSTATE 23000 (psycopg's
  IntegrityError, the class SQLite raises) on UPDATE and DELETE, plus a
  statement trigger on TRUNCATE, which bypasses row triggers in PostgreSQL.
- Every migration holds the store's write advisory lock, so it never
  interleaves with an engine write or another migration.

Torch-free and FastAPI-free, like `kalos.portal.migrate`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from kalos.portal.migrate import (
    _V3_TABLES,
    _V5_TABLES,
    JOB_KINDS,
    JOB_STATES,
    SCHEMA_VERSION,
    V2,
    V2_NAME,
    V3,
    V3_NAME,
    V4,
    V4_NAME,
    V5,
    V5_NAME,
    MigrationError,
    SchemaError,
)
from kalos.portal.storage import (
    WRITE_LOCK_KEY,
    PgConnection,
    pg_connect,
    redact_dsn,
)

log = logging.getLogger("kalos.portal")

APPEND_ONLY_FUNCTION = "kalos_reject_change"

_V2_TABLES_PG = ("campaigns", "analyses", "tenant_defaults", "audit_log", "schema_migrations")
_V4_TRIGGERS_PG = ("audit_log_no_update", "audit_log_no_delete", "audit_log_no_truncate")
_V5_TRIGGERS_PG = ("run_results_no_update", "run_results_no_delete", "run_results_no_truncate")

_V2_DDL_PG: tuple[str, ...] = (
    """
    CREATE TABLE schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at DOUBLE PRECISION NOT NULL
    )
    """,
    """
    CREATE TABLE campaigns (
        id TEXT PRIMARY KEY,
        tenant TEXT NOT NULL,
        name TEXT NOT NULL,
        state TEXT,
        generation TEXT,
        created_by TEXT,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        archived_at DOUBLE PRECISION,
        UNIQUE (id, tenant)
    )
    """,
    "CREATE INDEX campaigns_tenant_idx ON campaigns (tenant, updated_at)",
    """
    CREATE TABLE analyses (
        campaign_id TEXT PRIMARY KEY,
        tenant TEXT NOT NULL,
        state TEXT NOT NULL,
        dataset TEXT,
        campaign_generation TEXT,
        updated_at DOUBLE PRECISION NOT NULL,
        FOREIGN KEY (campaign_id, tenant) REFERENCES campaigns (id, tenant)
    )
    """,
    """
    CREATE TABLE tenant_defaults (
        tenant TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL,
        FOREIGN KEY (campaign_id, tenant) REFERENCES campaigns (id, tenant)
    )
    """,
    """
    CREATE TABLE audit_log (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts DOUBLE PRECISION NOT NULL,
        tenant TEXT NOT NULL,
        subject TEXT,
        auth_method TEXT,
        action TEXT NOT NULL,
        campaign_id TEXT,
        job_id TEXT,
        outcome TEXT NOT NULL,
        request_id TEXT,
        detail TEXT
    )
    """,
    "CREATE INDEX audit_log_tenant_ts_idx ON audit_log (tenant, ts)",
)

_V3_DDL_PG: tuple[str, ...] = (
    f"""
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        tenant TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ({", ".join(repr(k) for k in JOB_KINDS)})),
        state TEXT NOT NULL CHECK (state IN ({", ".join(repr(s) for s in JOB_STATES)})),
        created_at DOUBLE PRECISION NOT NULL,
        started_at DOUBLE PRECISION,
        finished_at DOUBLE PRECISION,
        updated_at DOUBLE PRECISION NOT NULL,
        progress DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
        stage TEXT,
        error_code TEXT,
        error_message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        input_path TEXT,
        input_sha256 TEXT,
        input_bytes BIGINT,
        params TEXT,
        base_generation TEXT,
        result_ref TEXT,
        requested_by TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        runner_id TEXT,
        worker_pid INTEGER,
        input_purged_at DOUBLE PRECISION,
        FOREIGN KEY (campaign_id, tenant) REFERENCES campaigns (id, tenant)
    )
    """,
    "CREATE INDEX jobs_tenant_campaign_idx ON jobs (tenant, campaign_id, created_at)",
    "CREATE INDEX jobs_state_idx ON jobs (state, created_at)",
)

# One function for every append-only table. TG_TABLE_NAME / TG_OP name the
# table and the operation in the message, which matches the SQLite text
# ("<table> is append-only: UPDATE is rejected").
_APPEND_ONLY_FUNCTION_DDL = f"""
CREATE OR REPLACE FUNCTION {APPEND_ONLY_FUNCTION}() RETURNS trigger
LANGUAGE plpgsql AS $fn$
BEGIN
    RAISE EXCEPTION '% is append-only: % is rejected', TG_TABLE_NAME, TG_OP
        USING ERRCODE = '23000';
END
$fn$
"""


def _append_only_ddl(table: str) -> tuple[str, ...]:
    return (
        _APPEND_ONLY_FUNCTION_DDL,
        f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {APPEND_ONLY_FUNCTION}()",
        f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
        f"FOR EACH ROW EXECUTE FUNCTION {APPEND_ONLY_FUNCTION}()",
        f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {APPEND_ONLY_FUNCTION}()",
    )


_V4_DDL_PG = _append_only_ddl("audit_log")

_V5_DDL_PG: tuple[str, ...] = (
    """
    CREATE TABLE run_results (
        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        tenant TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 1),
        value DOUBLE PRECISION NOT NULL,
        previous_value DOUBLE PRECISION,
        reason TEXT,
        actor TEXT,
        auth_method TEXT,
        recorded_at DOUBLE PRECISION NOT NULL,
        UNIQUE (campaign_id, tenant, run_id, version),
        CHECK (version = 1 OR (reason IS NOT NULL AND length(trim(reason)) > 0)),
        FOREIGN KEY (campaign_id, tenant) REFERENCES campaigns (id, tenant)
    )
    """,
    *_append_only_ddl("run_results"),
)


# --- introspection (the connection's current schema only) -------------------- #


def _tables(conn: PgConnection) -> set[str]:
    return {
        str(r[0]) for r in conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'"
        ).fetchall()
    }


def _triggers(conn: PgConnection) -> set[str]:
    return {
        str(r[0]) for r in conn.execute(
            "SELECT t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = current_schema() AND NOT t.tgisinternal AND t.tgenabled <> 'D'"
        ).fetchall()
    }


def _columns(conn: PgConnection, table: str) -> set[str]:
    return {
        str(r[0]) for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = ?",
            (table,),
        ).fetchall()
    }


def current_version(conn: PgConnection) -> int:
    """0 = no kalos tables in this schema; N = the highest applied migration.
    A schema with kalos tables but no version row is refused (as on SQLite)."""
    tables = _tables(conn)
    if "schema_migrations" not in tables:
        if tables & set(_V2_TABLES_PG) or "jobs" in tables or "run_results" in tables:
            raise SchemaError(
                "the PostgreSQL schema has kalos tables but no schema_migrations table; "
                "refusing to guess the version. See docs/MIGRATIONS.md."
            )
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    if not row or row[0] is None:
        raise SchemaError(
            "the PostgreSQL schema has an empty schema_migrations table; refusing to "
            "guess the version. See docs/MIGRATIONS.md."
        )
    return int(row[0])


def assert_shape(conn: PgConnection, version: int) -> None:
    """The tables, columns and triggers `version` requires exist (and the
    triggers are enabled). Raises SchemaError otherwise."""
    tables = _tables(conn)
    triggers = _triggers(conn)
    expected = list(_V2_TABLES_PG)
    if version >= V3:
        expected += list(_V3_TABLES)
    if version >= V5:
        expected += list(_V5_TABLES)
    missing = [t for t in expected if t not in tables]
    if version >= V4:
        missing += [f"trigger {t}" for t in _V4_TRIGGERS_PG if t not in triggers]
    if version >= V5:
        missing += [f"trigger {t}" for t in _V5_TRIGGERS_PG if t not in triggers]
    if "campaigns" in tables:
        from kalos.portal.migrate import _V2_CAMPAIGN_COLUMNS

        missing_cols = sorted(_V2_CAMPAIGN_COLUMNS - _columns(conn, "campaigns"))
        missing += [f"campaigns.{c}" for c in missing_cols]
    if missing:
        raise SchemaError(
            f"the PostgreSQL schema records v{version} but lacks: {', '.join(missing)} "
            "(a missing or DISABLED trigger counts as missing); refusing to start. "
            "See docs/MIGRATIONS.md (PostgreSQL backend)."
        )


def _refuse_newer(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise SchemaError(
            f"the PostgreSQL schema is at v{version}, newer than this code "
            f"(v{SCHEMA_VERSION}); refusing to run against it"
        )


# --- up --------------------------------------------------------------------- #


@dataclass
class PgMigrationResult:
    action: str  # "noop" | "created" | "migrated" | "rolled_back" | "dry_run"
    plan_text: str
    warnings: list[str] = field(default_factory=list)


def _record(conn: PgConnection, version: int, name: str) -> None:
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (version, name, time.time()),
    )


_STEPS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (V2, V2_NAME, _V2_DDL_PG),
    (V3, V3_NAME, _V3_DDL_PG),
    (V4, V4_NAME, _V4_DDL_PG),
    (V5, V5_NAME, _V5_DDL_PG),
)


def _check_steps_cover_schema() -> None:
    """Fail closed when a schema version was added to the SQLite path
    (`migrate.SCHEMA_VERSION`) without its PostgreSQL DDL here: the engine would
    otherwise record or expect a version whose tables do not exist."""
    covered = {v for v, _, _ in _STEPS}
    expected = set(range(V2, SCHEMA_VERSION + 1))
    if covered != expected:
        raise SchemaError(
            f"this build has PostgreSQL DDL for schema versions {sorted(covered)} but the "
            f"engine schema is v{SCHEMA_VERSION} (needs {sorted(expected)}); refusing to "
            "use PostgreSQL. Add the missing steps to kalos/portal/migrate_pg.py."
        )


# Tables whose append-only triggers the table OWNER (or a superuser) can switch
# off with ALTER TABLE ... DISABLE TRIGGER. The engine role must not own them.
PROTECTED_TABLES = ("audit_log", "run_results")


def owner_warnings(conn: PgConnection) -> list[str]:
    """Warnings when the connected role can bypass the append-only triggers:
    it owns a protected table, or it is a superuser. Read-only."""
    out: list[str] = []
    row = conn.execute(
        "SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    user, is_super = (str(row[0]), bool(row[1])) if row else ("?", False)
    if is_super:
        out.append(
            f"the engine connects as superuser {user!r}: it can disable the append-only "
            "triggers. Connect as a non-owner, non-superuser role (docs/MIGRATIONS.md, "
            "\"PostgreSQL grant model\")."
        )
    owned = [
        str(r[0]) for r in conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = current_schema() AND c.relkind = 'r' "
            "AND c.relname IN ('audit_log', 'run_results') "
            "AND pg_has_role(current_user, c.relowner, 'USAGE')"
        ).fetchall()
    ]
    if owned:
        out.append(
            f"the engine role {user!r} owns (or is a member of the owner of) "
            f"{', '.join(sorted(owned))}: an owner can ALTER TABLE ... DISABLE TRIGGER. "
            "Run migrations as an owner role and the engine as a separate role with "
            "SELECT/INSERT only on these tables (docs/MIGRATIONS.md, \"PostgreSQL grant model\")."
        )
    return out


def _plan_up(where: str, version: int) -> str:
    if version >= SCHEMA_VERSION:
        return f"{where}: already at schema v{version}; nothing to do"
    lines = [f"{where}: schema v{version} -> v{SCHEMA_VERSION} (one transaction)"]
    if version > 0:
        lines.append("  backup: NOT taken by this tool; take a pg_dump first")
    for v, name, _ in _STEPS:
        if v > version:
            lines.append(f"  v{v}: {name}")
    return "\n".join(lines)


def _begin_locked(conn: PgConnection) -> None:
    conn.execute("BEGIN ISOLATION LEVEL READ COMMITTED")
    conn.execute("SELECT pg_advisory_xact_lock(?)", (WRITE_LOCK_KEY,))


def _rollback(conn: PgConnection) -> None:
    if conn.in_transaction:
        conn.execute("ROLLBACK")


def migrate_up(dsn: str, *, dry_run: bool = False, allow_upgrade: bool = True) -> PgMigrationResult:
    """Bring the schema to v5. Empty -> created; current -> verified no-op;
    older -> upgraded only when `allow_upgrade` (the CLI). Engine startup passes
    `allow_upgrade=False` and gets a SchemaError instead."""
    _check_steps_cover_schema()
    where = redact_dsn(dsn)
    conn = PgConnection(pg_connect(dsn))
    try:
        if dry_run:
            conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                version = current_version(conn)
                _refuse_newer(version)
                if version >= V2:
                    assert_shape(conn, version)
            finally:
                _rollback(conn)
            return PgMigrationResult("dry_run", _plan_up(where, version))
        _begin_locked(conn)
        try:
            version = current_version(conn)
            _refuse_newer(version)
            if version >= V2:
                assert_shape(conn, version)
            if version == SCHEMA_VERSION:
                _rollback(conn)
                return PgMigrationResult("noop", f"{where}: already at schema v{SCHEMA_VERSION}")
            if version > 0 and not allow_upgrade:
                raise SchemaError(
                    f"the PostgreSQL schema is at v{version}; this engine needs "
                    f"v{SCHEMA_VERSION}. The engine does not upgrade a server database "
                    "on startup: take a pg_dump, then run "
                    "`python -m kalos.portal.migrate up` with KALOS_DATABASE_URL set. "
                    "See docs/MIGRATIONS.md (PostgreSQL backend)."
                )
            plan = _plan_up(where, version)
            for v, name, ddl in _STEPS:
                if v > version:
                    for stmt in ddl:
                        conn.execute(stmt)
                    _record(conn, v, name)
            assert_shape(conn, SCHEMA_VERSION)
            conn.execute("COMMIT")
        except BaseException:
            _rollback(conn)
            raise
    finally:
        conn.raw.close()
    return PgMigrationResult("created" if version == 0 else "migrated", plan)


def ensure_schema(dsn: str) -> PgMigrationResult:
    """Startup hook used by `CampaignStore` on PostgreSQL. Also logs a warning
    when the engine role can bypass the append-only triggers (`owner_warnings`)."""
    result = migrate_up(dsn, allow_upgrade=False)
    conn = PgConnection(pg_connect(dsn))
    try:
        result.warnings.extend(owner_warnings(conn))
    finally:
        conn.raw.close()
    for w in result.warnings:
        log.warning("kalos postgres: %s", w)
    return result


# --- down ------------------------------------------------------------------- #


def _free_name(existing: set[str], base: str) -> str:
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def _down_step(conn: PgConnection, step_from: int, *, apply: bool) -> str:
    tables = _tables(conn)
    if step_from == V5:
        n = int(conn.execute("SELECT COUNT(*) FROM run_results").fetchone()[0]) \
            if "run_results" in tables else 0
        new = _free_name(tables, "run_results_v5_rolledback")
        if apply:
            for t in _V5_TRIGGERS_PG:
                conn.execute(f'DROP TRIGGER IF EXISTS "{t}" ON run_results')
            conn.execute(f'ALTER TABLE run_results RENAME TO "{new}"')
            conn.execute("DELETE FROM schema_migrations WHERE version = ?", (V5,))
        return (f"schema v{V5} -> v{V4}\n  drop: run_results UPDATE/DELETE/TRUNCATE triggers\n"
                f"  rename: run_results -> {new} (kept; {n} row(s))")
    if step_from == V4:
        if apply:
            for t in _V4_TRIGGERS_PG:
                conn.execute(f'DROP TRIGGER IF EXISTS "{t}" ON audit_log')
            conn.execute("DELETE FROM schema_migrations WHERE version = ?", (V4,))
        return (f"schema v{V4} -> v{V3}\n  drop: audit_log UPDATE/DELETE/TRUNCATE triggers\n"
                "  audit rows: unchanged")
    if step_from == V3:
        n = int(conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE state IN ('queued', 'running')"
        ).fetchone()[0]) if "jobs" in tables else 0
        new = _free_name(tables, "jobs_v3_rolledback")
        if apply:
            conn.execute("DROP INDEX IF EXISTS jobs_tenant_campaign_idx")
            conn.execute("DROP INDEX IF EXISTS jobs_state_idx")
            conn.execute(f'ALTER TABLE jobs RENAME TO "{new}"')
            conn.execute("DELETE FROM schema_migrations WHERE version = ?", (V3,))
        return (f"schema v{V3} -> v{V2}\n  rename: jobs -> {new} (kept)\n"
                f"  unfinished jobs: {n} (they will not run on v2 code)")
    raise MigrationError("rollback below v2 is not supported on PostgreSQL")  # pragma: no cover


def migrate_down(
    dsn: str, *, dry_run: bool = False, target: int | None = None, confirm_backup: bool = False
) -> PgMigrationResult:
    """Roll back one version (or to `target`, >= 2), in one transaction. Nothing
    is dropped except triggers and indexes; tables are renamed aside. Requires
    `confirm_backup` (the operator states a pg_dump was taken) unless dry-run."""
    where = redact_dsn(dsn)
    conn = PgConnection(pg_connect(dsn))
    try:
        if dry_run:
            conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                version = current_version(conn)
                goal = _down_goal(version, target)
                parts = [_down_step(conn, s, apply=False) for s in range(version, goal, -1)]
            finally:
                _rollback(conn)
            return PgMigrationResult("dry_run", "\n".join([f"{where}: rollback plan", *parts]))
        if not confirm_backup:
            raise MigrationError(
                "`down` on PostgreSQL takes no backup itself: take a pg_dump first, then "
                "re-run with --confirm-backup"
            )
        _begin_locked(conn)
        try:
            version = current_version(conn)
            goal = _down_goal(version, target)
            assert_shape(conn, version)
            parts = [_down_step(conn, s, apply=True) for s in range(version, goal, -1)]
            conn.execute("COMMIT")
        except BaseException:
            _rollback(conn)
            raise
    finally:
        conn.raw.close()
    return PgMigrationResult("rolled_back", "\n".join([f"{where}: rollback", *parts]))


def _down_goal(version: int, target: int | None) -> int:
    if version < V3:
        raise MigrationError(
            f"the PostgreSQL schema is at v{version}; rollback below v2 is not supported "
            "on PostgreSQL (there is no v1 PostgreSQL schema)"
        )
    goal = version - 1 if target is None else target
    if not V2 <= goal < version:
        raise MigrationError(
            f"cannot roll back from schema v{version} to v{goal} on PostgreSQL; "
            f"valid targets are {V2}..{version - 1}"
        )
    return goal


# --- status ----------------------------------------------------------------- #


def status_report(dsn: str) -> tuple[str, bool]:
    """(report, healthy). Read-only. `healthy` is False when the engine would
    refuse to start (shape check, unversioned tables, newer or OLDER schema:
    the engine does not upgrade PostgreSQL on startup)."""
    lines = [f"database: {redact_dsn(dsn)} (postgres)"]
    conn = PgConnection(pg_connect(dsn))
    healthy = True
    try:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        try:
            schema = conn.execute("SELECT current_schema()").fetchone()[0]
            lines.append(f"  schema: {schema}")
            try:
                version = current_version(conn)
            except SchemaError as exc:
                lines.append("SHAPE CHECK FAILED: the engine would refuse to start.")
                lines.append(f"  {exc}")
                return "\n".join(lines), False
            lines.append(f"  schema version: v{version} (this code: v{SCHEMA_VERSION})")
            for table in sorted(_tables(conn)):
                n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]  # noqa: S608 - table names come from information_schema
                lines.append(f"  table {table}: {n} row(s)")
            problem: str | None = None
            try:
                _refuse_newer(version)
                if version >= V2:
                    assert_shape(conn, version)
            except SchemaError as exc:
                problem = str(exc)
            if problem is not None:
                healthy = False
                lines.append("SHAPE CHECK FAILED: the engine would refuse to start.")
                lines.append(f"  {problem}")
            elif version == 0:
                lines.append("  empty: the engine creates the schema on first start")
            elif version < SCHEMA_VERSION:
                healthy = False
                lines.append("pending migration (the engine refuses to start until it runs):")
                lines.append(_plan_up(redact_dsn(dsn), version))
            else:
                lines.append("  shape check: ok")
            if version >= V2:
                for w in owner_warnings(conn):
                    lines.append(f"WARNING: {w}")
        finally:
            _rollback(conn)
    finally:
        conn.raw.close()
    return "\n".join(lines), healthy
