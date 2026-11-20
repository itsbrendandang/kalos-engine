"""Schema migrations on PostgreSQL (`KALOS_DATABASE_URL=postgresql://...`).

The same version numbers as the SQLite path (`kalos.portal.migrate`), recorded in
the same `schema_migrations` table. PostgreSQL support starts at v9, so its first
step is a BASELINE that creates, in one transaction:

- `schema_migrations`;
- `audit_log` (the v2 columns) with the append-only triggers of v4 (UPDATE,
  DELETE and TRUNCATE rejected), so a records write and its audit row commit in
  the same transaction, as they do on SQLite;
- the v9 records tables (`kalos.storage.migrations.v9`).

The campaign, job and result tables join PostgreSQL in a later step, when
`CampaignStore` moves behind `kalos.storage` (REORG_PLAN R06/R12); until then they
stay in `portal.db`. docs/MIGRATIONS.md ("PostgreSQL backend") has the operator
view.

Rules, the same as the SQLite machinery wherever PostgreSQL allows:

- ONE transaction per upgrade. PostgreSQL DDL is transactional, so a failed step
  leaves nothing behind. Every migration holds the store's write advisory lock,
  so it never interleaves with an engine write or another migration.
- An EMPTY database is created at the current version on engine startup (there
  is nothing to back up). A database at an OLDER version is refused at startup:
  `python -m kalos.storage.migrations.pg_runner up` upgrades it after a VERIFIED
  backup: `pg_dump --format=custom` into the backup directory, then
  `pg_restore --list` must read the file back and list every table of the old
  version. Without pg_dump on the PATH the operator takes the dump and passes
  `--confirm-backup`. A NEWER database is refused (run the newer code).
- The DSN is read from `KALOS_DATABASE_URL` only (never from argv, where it would
  show in `ps` and shell history) and is redacted in every message. pg_dump gets
  the password through its environment, not its command line.

Torch-free and FastAPI-free.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from kalos.storage.backend import (
    DATABASE_URL_ENV,
    WRITE_LOCK_KEY,
    Conn,
    StorageConfigError,
    backend_from_env,
    pg_connect,
    redact_dsn,
)
from kalos.storage.migrations import v9

log = logging.getLogger("kalos.storage")

BASELINE_VERSION = v9.V9
BASELINE_NAME = "postgres baseline: audit log + " + v9.V9_NAME

AUDIT_TRIGGERS = ("audit_log_no_update", "audit_log_no_delete", "audit_log_no_truncate")
BASELINE_TABLES = ("schema_migrations", "audit_log", *v9.TABLES)

_BASELINE_DDL: tuple[str, ...] = (
    """
    CREATE TABLE schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at DOUBLE PRECISION NOT NULL
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
    v9.PG_FUNCTION_DDL,
    *v9.pg_append_only_ddl("audit_log"),
)


class PgMigrationError(RuntimeError):
    """The PostgreSQL schema cannot be brought to this code's version safely.
    The message names what to do and never contains the password."""


@dataclass(frozen=True)
class Step:
    version: int
    name: str
    apply: Callable[[Conn], None]
    tables: tuple[str, ...]  # what a database at this version must have


def _baseline(conn: Conn) -> None:
    for ddl in _BASELINE_DDL:
        conn.execute(ddl)
    v9.apply_postgres(conn)


# Ordered. A new schema version that touches PostgreSQL appends a Step here.
STEPS: tuple[Step, ...] = (
    Step(BASELINE_VERSION, BASELINE_NAME, _baseline, BASELINE_TABLES),
)
LATEST = STEPS[-1].version


@dataclass(frozen=True)
class PgMigrationResult:
    action: str  # created | noop | upgraded | dry-run
    message: str
    from_version: int
    to_version: int
    backup: str | None = None


# --- inspection ----------------------------------------------------------------- #


def _tables(conn: Conn) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
    ).fetchall()
    return {str(r[0]) for r in rows}


def _triggers(conn: Conn) -> set[str]:
    rows = conn.execute(
        "SELECT t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = current_schema() AND NOT t.tgisinternal"
    ).fetchall()
    return {str(r[0]) for r in rows}


def current_version(conn: Conn) -> int:
    """0 for an empty schema; the newest recorded version otherwise. A schema
    with tables but no `schema_migrations` is refused: it is not ours."""
    tables = _tables(conn)
    if "schema_migrations" not in tables:
        if tables & set(BASELINE_TABLES):
            raise PgMigrationError(
                "the PostgreSQL schema has Kalos-named tables but no schema_migrations table; "
                "refusing to guess its version (point KALOS_DATABASE_URL at an empty database "
                "or schema)"
            )
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def assert_shape(conn: Conn, version: int) -> None:
    """Refuse a database that records `version` but lacks its tables/triggers."""
    tables, triggers = _tables(conn), _triggers(conn)
    need_tables = {t for s in STEPS if s.version <= version for t in s.tables}
    need_triggers = set(AUDIT_TRIGGERS) | set(v9.PG_TRIGGERS) if version >= BASELINE_VERSION else set()
    missing = sorted(need_tables - tables) + sorted(f"trigger {t}" for t in need_triggers - triggers)
    if missing:
        raise PgMigrationError(
            f"the PostgreSQL database records schema v{version} but lacks {', '.join(missing)}; "
            "refusing to start (restore the backup taken before the upgrade)"
        )


def _record(conn: Conn, version: int, name: str) -> None:
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (version, name, time.time()),
    )


def _begin_locked(conn: Conn) -> None:
    conn.execute("BEGIN ISOLATION LEVEL READ COMMITTED")
    conn.execute("SELECT pg_advisory_xact_lock(?)", (WRITE_LOCK_KEY,))


def _rollback(conn: Conn) -> None:
    try:
        conn.execute("ROLLBACK")
    except Exception:  # noqa: BLE001 - the original error is the one to raise
        log.debug("rollback after a failed migration also failed", exc_info=True)


def _apply_steps(conn: Conn, from_version: int) -> int:
    to = from_version
    for step in STEPS:
        if step.version > from_version:
            step.apply(conn)
            _record(conn, step.version, step.name)
            to = step.version
    return to


def _refuse_newer(version: int) -> None:
    if version > LATEST:
        raise PgMigrationError(
            f"the PostgreSQL database is at schema v{version}, newer than this code "
            f"(PostgreSQL v{LATEST}); run the newer engine, or restore a backup taken "
            "before the upgrade"
        )


# --- startup ---------------------------------------------------------------------- #


def ensure_schema(dsn: str) -> PgMigrationResult:
    """Engine startup hook: create an EMPTY database at the current version (one
    transaction), verify one already current, refuse anything else."""
    conn = pg_connect(dsn)
    try:
        _begin_locked(conn)
        try:
            version = current_version(conn)
            _refuse_newer(version)
            if version == 0:
                to = _apply_steps(conn, 0)
                assert_shape(conn, to)
                conn.execute("COMMIT")
                return PgMigrationResult(
                    "created", f"{redact_dsn(dsn)}: created schema v{to}", 0, to
                )
            if version < LATEST:
                raise PgMigrationError(
                    f"the PostgreSQL database is at schema v{version}; this code needs "
                    f"v{LATEST}. The engine does not upgrade a server database on startup: "
                    f"run `python -m kalos.storage.migrations.pg_runner up` (it takes and "
                    "verifies a pg_dump first)"
                )
            assert_shape(conn, version)
            conn.execute("COMMIT")
            return PgMigrationResult("noop", f"{redact_dsn(dsn)}: at schema v{version}", version, version)
        except BaseException:
            _rollback(conn)
            raise
    finally:
        conn.raw.close()


# --- verified backup --------------------------------------------------------------- #


def _split_password(dsn: str) -> tuple[str, str | None]:
    """The DSN without its password, and the password (for PGPASSWORD)."""
    parts = urlsplit(dsn)
    if parts.password is None:
        return dsn, None
    user = parts.username or ""
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"{user}@{host}", parts.path, parts.query, parts.fragment)), parts.password


def take_verified_backup(dsn: str, backup_dir: Path, *, label: str, expect_tables: tuple[str, ...]) -> Path:
    """`pg_dump --format=custom` of the engine's schema into `backup_dir`, then `pg_restore --list` must
    read the file back and list every table in `expect_tables`. Raises
    `PgMigrationError` (and removes a partial file) otherwise."""
    pg_dump, pg_restore = shutil.which("pg_dump"), shutil.which("pg_restore")
    if not pg_dump or not pg_restore:
        raise PgMigrationError(
            "pg_dump/pg_restore are not on the PATH, so this tool cannot take and verify the "
            "backup itself; take a pg_dump of the database and re-run with --confirm-backup"
        )
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = backup_dir / f"kalos-pg.{label}-{stamp}.dump"
    conn = pg_connect(dsn)
    try:
        (schema,) = conn.execute("SELECT current_schema()").fetchone()
    finally:
        conn.raw.close()
    url, password = _split_password(dsn)
    env = {k: v for k, v in os.environ.items() if k != DATABASE_URL_ENV}
    if password is not None:
        env["PGPASSWORD"] = password
    try:
        subprocess.run(  # noqa: S603 - fixed argv; the URL carries no password
            [pg_dump, "--format=custom", "--no-owner", f"--schema={schema}", f"--file={path}",
             f"--dbname={url}"],
            env=env, check=True, capture_output=True, timeout=3600,
        )
        listing = subprocess.run(  # noqa: S603 - fixed argv
            [pg_restore, "--list", str(path)], env=env, check=True, capture_output=True,
            timeout=600, text=True,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        path.unlink(missing_ok=True)
        raise PgMigrationError(
            f"the backup of {redact_dsn(dsn)} failed or could not be read back "
            f"({type(exc).__name__}); nothing was migrated"
        ) from None
    listed = set(re.findall(r"\bTABLE \S+ (\S+) ", listing))
    missing = [t for t in expect_tables if t not in listed]
    if missing or path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise PgMigrationError(
            f"the backup could not be verified (pg_restore --list does not show {missing or 'any data'}); "
            "nothing was migrated"
        )
    os.chmod(path, 0o600)
    return path


# --- CLI upgrade ---------------------------------------------------------------- #


def migrate_up(
    dsn: str, *, dry_run: bool = False, backup_dir: Path | None = None, confirm_backup: bool = False,
) -> PgMigrationResult:
    """Bring the database to LATEST: verified backup first (unless empty), then
    every missing step in ONE transaction."""
    conn = pg_connect(dsn)
    try:
        version = current_version(conn)
        _refuse_newer(version)
        if version == LATEST:
            return PgMigrationResult("noop", f"{redact_dsn(dsn)}: already at schema v{LATEST}", version, version)
        steps = [s for s in STEPS if s.version > version]
        plan = f"{redact_dsn(dsn)}: schema v{version} -> v{LATEST} (" + "; ".join(s.name for s in steps) + ")"
        if dry_run:
            return PgMigrationResult("dry-run", plan, version, version)
        backup: Path | None = None
        if version > 0 and not confirm_backup:
            expect = tuple(t for s in STEPS if s.version <= version for t in s.tables)
            backup = take_verified_backup(
                dsn, backup_dir or Path.cwd(), label=f"pre-v{steps[0].version}", expect_tables=expect,
            )
        _begin_locked(conn)
        try:
            again = current_version(conn)
            if again != version:
                raise PgMigrationError("the schema changed while the backup was taken; re-run")
            to = _apply_steps(conn, version)
            assert_shape(conn, to)
            conn.execute("COMMIT")
        except BaseException:
            _rollback(conn)
            raise
        return PgMigrationResult(
            "upgraded" if version else "created", plan, version, to,
            backup=str(backup) if backup else None,
        )
    finally:
        conn.raw.close()


def status(dsn: str) -> str:
    conn = pg_connect(dsn)
    try:
        version = current_version(conn)
        state = "current" if version == LATEST else ("newer than this code" if version > LATEST else "needs `up`")
        return f"{redact_dsn(dsn)}: schema v{version} ({state}; this code: v{LATEST})"
    finally:
        conn.raw.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kalos.storage.migrations.pg_runner",
        description=f"Kalos schema on PostgreSQL (v{BASELINE_VERSION}..v{LATEST}). The database "
        f"comes from {DATABASE_URL_ENV}; it is never taken from the command line.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="print the recorded schema version")
    up = sub.add_parser("up", help="upgrade to this code's version (verified pg_dump first)")
    up.add_argument("--dry-run", action="store_true")
    up.add_argument("--backup-dir", type=Path, default=None)
    up.add_argument("--confirm-backup", action="store_true",
                    help="you took and checked a pg_dump yourself (no pg_dump on this box)")
    args = parser.parse_args(argv)
    try:
        cfg = backend_from_env()
        if not cfg.is_postgres:
            raise StorageConfigError(f"{DATABASE_URL_ENV} is not a postgresql:// URL; nothing to do")
        dsn = cfg.dsn or ""
        if args.cmd == "status":
            print(status(dsn))
            return 0
        res = migrate_up(dsn, dry_run=args.dry_run, backup_dir=args.backup_dir,
                         confirm_backup=args.confirm_backup)
        print(res.message)
        if res.backup:
            print(f"backup: {res.backup} (verified with pg_restore --list)")
        return 0
    except (PgMigrationError, StorageConfigError) as exc:
        print(f"kalos pg migrate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
