"""Kalos portal - schema versioning for `portal.db` and the v1 -> v2 migration.

Schema v1 (everything before this module) is one table,
`campaigns(tenant PRIMARY KEY, state, updated_at)`: one campaign per tenant,
plus a per-tenant analysis cache in `<state_dir>/latest/*.json` and an
in-memory map in `kalos.portal.app`.

Schema v2 (this module) stores several campaigns per tenant and the analyses in
SQLite:

  - `schema_migrations(version, name, applied_at)`: which migrations ran.
  - `campaigns(id 'cmp_'+32 hex, tenant, name, state JSON, generation, ...)`,
    `UNIQUE(id, tenant)` so other tables can carry a composite FK that binds a
    campaign id to its tenant.
  - `analyses(campaign_id PK, tenant, state, ...)`: the published analysis of a
    campaign (what `/api/latest` serves), replacing the `latest/` files.
  - `tenant_defaults(tenant PK, campaign_id)`: the campaign the legacy
    single-campaign routes (`/api/run`, `/api/latest`, `/api/campaign/*`) act on.
  - `audit_log(...)`: who did what, never tokens or raw data.

The migration is explicit code, run once: at startup by `CampaignStore` when
`schema_migrations` is missing, or by an operator with
`python -m kalos.portal.migrate up|down|status [--dry-run]`.
Before it changes anything it takes an online SQLite backup to
`<db>.pre-v2-<UTC timestamp>` and aborts if that fails. The change itself is a
single `BEGIN IMMEDIATE` transaction, and the v1 table is renamed (to
`campaigns_v1`), never dropped. `latest/` files are read, never moved or
deleted. See docs/MIGRATIONS.md for the runbook.

Schema v3 (`feat/background-jobs`) is additive: one `jobs` table for the
persistent background-job queue (kalos/portal/jobs.py, docs/JOBS.md). A v2
database is brought to v3 with the same machinery (verified `pre-v3` backup,
one `BEGIN IMMEDIATE` transaction); `down` steps back one version at a time
(v3 -> v2 renames `jobs` to `jobs_v3_rolledback`, never drops it), or to an
explicit `--to` version.

This module is torch-free and FastAPI-free on purpose: the CLI must run on a
box where the engine itself is stopped.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import stat
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kalos.portal.tenancy import is_valid_tenant_id

log = logging.getLogger("kalos.portal")

SCHEMA_VERSION = 3
V2 = 2  # the multi-campaign schema; v1 <-> v2 code below is keyed on it
V2_NAME = "multi-campaign storage"
V3_NAME = "background jobs"
BUSY_TIMEOUT_MS = 5000
DB_FILENAME = "portal.db"

# A campaign id is `cmp_` + uuid4().hex. Validated before any SQL runs, so a
# malformed id is refused without touching the database.
CAMPAIGN_ID_PATTERN = re.compile(r"cmp_[0-9a-f]{32}")

# Tables that hold tenant data. Every statement the store issues against these
# carries `tenant = ?` (tests/test_campaign_schema_v2.py enforces it).
TENANT_TABLES = ("campaigns", "analyses", "tenant_defaults", "audit_log", "jobs")

# A job id is `job_` + uuid4().hex (kalos/portal/jobs.py), validated the same way.
JOB_ID_PATTERN = re.compile(r"job_[0-9a-f]{32}")

IMPORTED_CAMPAIGN_NAME = "Imported campaign"
MIGRATION_ACTOR = "migration:v2"

_V1_REQUIRED_COLUMNS = frozenset({"tenant", "state", "updated_at"})
_V1_CONFLICT_COLUMN = "tenant"
_V2_CAMPAIGN_COLUMNS = frozenset(
    {"id", "tenant", "name", "state", "generation", "created_by", "created_at",
     "updated_at", "archived_at"}
)

V1_CAMPAIGNS_DDL = """
CREATE TABLE IF NOT EXISTS campaigns (
    tenant TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""

# `campaigns.state` is nullable: a NULL state is a campaign that exists (it has
# an id and may own an analysis) but has never been seeded with a dataset. The
# legacy `/api/campaign` reports it as `has_campaign: false`, exactly as v1 did
# for a tenant with an analysis but no campaign row.
#
# `campaigns.generation` mirrors the `generation` token inside `state` as a real
# column so the compare-and-swap is plain SQL (`generation IS ?`) and never
# depends on the JSON blob parsing under SQLite's JSON functions.
_V2_DDL: tuple[str, ...] = (
    """
    CREATE TABLE schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at REAL NOT NULL
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
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        archived_at REAL,
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
        updated_at REAL NOT NULL,
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
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
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
_V2_TABLES = ("campaigns", "analyses", "tenant_defaults", "audit_log", "schema_migrations")
_V2_INDEXES = ("campaigns_tenant_idx", "audit_log_tenant_ts_idx")

# Schema v3: the background-job queue (docs/JOBS.md). Additive: no v2 table is
# touched. `(campaign_id, tenant)` carries the same composite FK as `analyses`,
# so a job can only ever point at a campaign of its own tenant. `params` holds
# the upload form fields needed to (re)run the job (target, roles, sheet, file
# name); `input_path` is RELATIVE to the state directory and is built from
# server-generated parts only (tenant hash + job id), never caller strings.
JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled", "timed_out")
JOB_KINDS = ("analysis",)
_V3_DDL: tuple[str, ...] = (
    f"""
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        tenant TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ({", ".join(repr(k) for k in JOB_KINDS)})),
        state TEXT NOT NULL CHECK (state IN ({", ".join(repr(s) for s in JOB_STATES)})),
        created_at REAL NOT NULL,
        started_at REAL,
        finished_at REAL,
        updated_at REAL NOT NULL,
        progress REAL NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 1),
        stage TEXT,
        error_code TEXT,
        error_message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        input_path TEXT,
        input_sha256 TEXT,
        input_bytes INTEGER,
        params TEXT,
        base_generation TEXT,
        result_ref TEXT,
        requested_by TEXT,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        runner_id TEXT,
        worker_pid INTEGER,
        input_purged_at REAL,
        FOREIGN KEY (campaign_id, tenant) REFERENCES campaigns (id, tenant)
    )
    """,
    "CREATE INDEX jobs_tenant_campaign_idx ON jobs (tenant, campaign_id, created_at)",
    "CREATE INDEX jobs_state_idx ON jobs (state, created_at)",
)
_V3_TABLES = ("jobs",)
_V3_INDEXES = ("jobs_tenant_campaign_idx", "jobs_state_idx")


class MigrationError(RuntimeError):
    """The migration could not run safely and changed nothing (or rolled back)."""


class SchemaError(RuntimeError):
    """`portal.db` is in a shape this code refuses to run against."""


# --- connections ------------------------------------------------------------ #


def connect(db_path: Path | str, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """A connection with the store's per-connection pragmas.

    `isolation_level=None`: transactions are opened explicitly (`BEGIN
    IMMEDIATE`) by the caller, never implicitly by the sqlite3 module.
    `foreign_keys` and `busy_timeout` are per-connection settings, so every
    connection sets them. WAL is persistent in the file and is switched on by
    `enable_wal` once the schema is in place (see `migrate_up` for why not
    earlier).
    """
    conn = sqlite3.connect(
        str(db_path), isolation_level=None, check_same_thread=check_same_thread
    )
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def enable_wal(conn: sqlite3.Connection) -> str:
    """Switch the database to WAL. Returns the resulting mode; a filesystem that
    cannot do WAL (some network mounts) keeps its mode and is logged, not fatal."""
    mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        log.warning("kalos campaign store: could not enable WAL (journal_mode=%s)", mode)
    return mode


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """A read-only connection, for `--dry-run` and `status`: SQLite itself
    refuses any write through it."""
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
    # Table names here are module constants or names read from sqlite_master;
    # never caller input.
    return list(conn.execute(f'PRAGMA table_info("{table}")').fetchall())


def _free_name(existing: set[str], base: str) -> str:
    """`base`, or `base_2`, `base_3`, ... - the first not in `existing`.
    Deterministic, and never reuses a name, so nothing is ever overwritten."""
    if base not in existing:
        return base
    n = 2
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


# --- version detection -------------------------------------------------------- #


def current_version(conn: sqlite3.Connection) -> int:
    """0 = empty database; 1 = legacy (a `campaigns` table but no
    `schema_migrations`); N = the highest applied migration."""
    tables = _tables(conn)
    if "schema_migrations" not in tables:
        return 1 if "campaigns" in tables else 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0]) if row and row[0] is not None else (1 if "campaigns" in tables else 0)


def campaigns_table_is_v1_compatible(conn: sqlite3.Connection) -> bool:
    """Whether an existing `campaigns` table has the v1 shape: the v1 columns,
    with `tenant` as the SOLE primary key (the v1 upsert's `ON
    CONFLICT(tenant)` needs exactly that)."""
    info = _columns(conn, "campaigns")
    if not info:
        return False
    names = {str(row[1]) for row in info}
    pk_columns = [str(row[1]) for row in info if row[5]]
    return _V1_REQUIRED_COLUMNS <= names and pk_columns == [_V1_CONFLICT_COLUMN]


def _assert_v2_shape(conn: sqlite3.Connection, version: int | None = None) -> None:
    """The v2 tables (and, at v3, the `jobs` table) exist. `version` defaults to
    the recorded one."""
    tables = _tables(conn)
    v = current_version(conn) if version is None else version
    expected = _V2_TABLES + (_V3_TABLES if v >= 3 else ())
    missing = [t for t in expected if t not in tables]
    cols = {str(r[1]) for r in _columns(conn, "campaigns")}
    if missing or not _V2_CAMPAIGN_COLUMNS <= cols:
        raise SchemaError(
            f"portal.db records schema v{v} but its tables do not have the v{v} shape "
            f"(missing tables: {missing or 'none'}); refusing to start. "
            "See docs/MIGRATIONS.md (restore the pre-v2 backup or run `down`)."
        )


# --- legacy latest/ files ------------------------------------------------------- #


def legacy_latest_filename(tenant: str) -> str:
    """The `latest/` filename PR #86 introduced: `t-` + lowercase hex of the
    tenant's UTF-8 bytes. Injective, so it is the unambiguous form."""
    return f"t-{tenant.encode('utf-8').hex()}.json"


@dataclass
class LatestImport:
    filename: str
    tenant: str
    state: dict[str, Any]


@dataclass
class LatestSkip:
    filename: str
    reason: str


def classify_latest_files(
    latest_dir: Path | None, known_tenants: set[str]
) -> tuple[list[LatestImport], list[LatestSkip]]:
    """Map each `latest/*.json` file to the tenant it belongs to, or skip it.

    Two naming schemes exist on disk:

    - `t-<hex>.json` (PR #86): decodes to exactly one tenant.
    - `<name>.json` (before PR #86): the tenant with every character outside
      `[A-Za-z0-9_.-]` replaced by `_`. That is not injective (`acme/bio` and
      `acme_bio` both became `acme_bio.json`), so a name containing `_` is
      AMBIGUOUS and skipped. A name without `_` was never rewritten and maps to
      exactly that tenant.

    A `t-<hex>` stem is itself also a legal old-style name; if a known v1 tenant
    is literally called that, the file is ambiguous and skipped. When both a
    `t-<hex>` file and an old-style file map to one tenant, the `t-<hex>` file
    wins: it is the one the running code was serving. Unreadable or non-object
    JSON is skipped. Skipped files are reported, and every file is left on disk.
    """
    imports: dict[str, LatestImport] = {}
    skipped: list[LatestSkip] = []
    if latest_dir is None or not latest_dir.is_dir():
        return [], []
    candidates: list[tuple[Path, str, bool]] = []  # (path, tenant, is_hex)
    for path in sorted(latest_dir.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        stem = path.name[: -len(".json")]
        hex_part = stem[2:] if stem.startswith("t-") else None
        decoded: str | None = None
        if hex_part and len(hex_part) % 2 == 0 and re.fullmatch(r"[0-9a-f]+", hex_part):
            try:
                decoded = bytes.fromhex(hex_part).decode("utf-8")
            except UnicodeDecodeError:
                decoded = None
        if decoded is not None:
            if stem in known_tenants and decoded != stem:
                skipped.append(LatestSkip(path.name, "ambiguous: both a hex-encoded name and an existing tenant id"))
                continue
            candidates.append((path, decoded, True))
            continue
        if not stem:
            skipped.append(LatestSkip(path.name, "empty tenant name"))
        elif "_" in stem:
            skipped.append(LatestSkip(
                path.name,
                "ambiguous: pre-#86 sanitised name ('_' may stand for any replaced character)",
            ))
        else:
            candidates.append((path, stem, False))

    # hex files first, so they win any collision with an old-style file
    for path, tenant, is_hex in sorted(candidates, key=lambda c: not c[2]):
        if tenant in imports:
            skipped.append(LatestSkip(path.name, f"superseded by {imports[tenant].filename}"))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped.append(LatestSkip(path.name, "unreadable or corrupt JSON"))
            continue
        if not isinstance(data, dict):
            skipped.append(LatestSkip(path.name, "not a JSON object"))
            continue
        imports[tenant] = LatestImport(path.name, tenant, data)
    return list(imports.values()), skipped


# --- plan ----------------------------------------------------------------- #


@dataclass
class UpPlan:
    db_path: Path
    from_version: int
    needs_migration: bool
    repair_note: str | None = None
    v1_rows: list[tuple[str, str, float]] = field(default_factory=list)
    corrupt_tenants: list[str] = field(default_factory=list)
    invalid_tenants: list[str] = field(default_factory=list)
    latest_imports: list[LatestImport] = field(default_factory=list)
    latest_skipped: list[LatestSkip] = field(default_factory=list)
    empty_defaults: list[str] = field(default_factory=list)  # tenants with analysis only

    def render(self) -> str:
        if not self.needs_migration:
            return f"{self.db_path}: already at schema v{self.from_version}; nothing to do"
        if self.from_version == V2:
            return "\n".join([
                f"{self.db_path}: schema v{V2} -> v{SCHEMA_VERSION}",
                f"  backup: online copy to {self.db_path.name}.pre-v3-<UTC timestamp>",
                "  create: jobs table (background jobs, docs/JOBS.md); no v2 table is changed",
            ])
        lines = [f"{self.db_path}: schema v{self.from_version} -> v{V2}"]
        if self.from_version > 0:
            lines.append(f"  backup: online copy to {self.db_path.name}.pre-v2-<UTC timestamp>")
        if self.repair_note:
            lines.append(f"  repair: {self.repair_note}")
        if self.from_version > 0:
            lines.append("  rename: campaigns -> campaigns_v1 (kept, not dropped)")
        lines.append(f"  campaigns: {len(self.v1_rows)} v1 row(s) -> '{IMPORTED_CAMPAIGN_NAME}' + tenant default")
        for t in self.corrupt_tenants:
            lines.append(f"  warn: tenant {t!r} state is not valid JSON; copied verbatim")
        for t in self.invalid_tenants:
            lines.append(f"  warn: tenant {t!r} fails the tenant-id pattern; imported unchanged")
        for imp in self.latest_imports:
            lines.append(f"  analysis: latest/{imp.filename} -> tenant {imp.tenant!r}")
        for t in self.empty_defaults:
            lines.append(f"  campaign: tenant {t!r} has an analysis but no campaign; an empty default campaign is created")
        for sk in self.latest_skipped:
            lines.append(f"  skip: latest/{sk.filename} ({sk.reason})")
        lines.append("  latest/ files are left in place")
        lines.append(f"  then: schema v{V2} -> v{SCHEMA_VERSION} (create the jobs table)")
        return "\n".join(lines)


def _plan_up(conn: sqlite3.Connection, db_path: Path, latest_dir: Path | None) -> UpPlan:
    version = current_version(conn)
    if version >= SCHEMA_VERSION:
        return UpPlan(db_path, version, needs_migration=False)
    plan = UpPlan(db_path, version, needs_migration=True)
    if version == V2:
        return plan
    tables = _tables(conn)
    if "campaigns" in tables:
        if campaigns_table_is_v1_compatible(conn):
            raw = conn.execute("SELECT tenant, state, updated_at FROM campaigns").fetchall()
            plan.v1_rows = [(str(t), s, float(u)) for t, s, u in raw]
        else:
            plan.repair_note = _describe_repair(conn)
            plan.v1_rows = _salvageable_rows(conn)
    for tenant, state, _ in plan.v1_rows:
        if _parse_state(state) is None:
            plan.corrupt_tenants.append(tenant)
        if not is_valid_tenant_id(tenant):
            plan.invalid_tenants.append(tenant)
    known = {t for t, _, _ in plan.v1_rows}
    plan.latest_imports, plan.latest_skipped = classify_latest_files(latest_dir, known)
    for imp in plan.latest_imports:
        if imp.tenant not in known:
            plan.empty_defaults.append(imp.tenant)
        if not is_valid_tenant_id(imp.tenant) and imp.tenant not in plan.invalid_tenants:
            plan.invalid_tenants.append(imp.tenant)
    return plan


def _parse_state(state: Any) -> dict[str, Any] | None:
    if not isinstance(state, str):
        return None
    try:
        data = json.loads(state)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# --- repair of an incompatible pre-v2 table ------------------------------------- #
# Kept from the v1 store: a `campaigns` table of the wrong shape (e.g. the
# composite-key table from an unmerged multi-campaign branch) is renamed to
# `campaigns_backup_N`, never dropped, and its newest row per tenant carried over.


def _describe_repair(conn: sqlite3.Connection) -> str:
    old_columns = sorted(str(r[1]) for r in _columns(conn, "campaigns"))
    backup = _free_backup_table(conn)
    salvageable = _V1_REQUIRED_COLUMNS <= set(old_columns)
    return (
        f"campaigns table was incompatible (columns {old_columns}); "
        f"renamed to {backup} and recreated, "
        + (
            "carrying over the newest campaign for each tenant"
            if salvageable
            else "with no state carried over (the old table lacked the required columns)"
        )
    )


def _free_backup_table(conn: sqlite3.Connection) -> str:
    existing = _tables(conn)
    n = 1
    while f"campaigns_backup_{n}" in existing:
        n += 1
    return f"campaigns_backup_{n}"


def _salvageable_rows(conn: sqlite3.Connection) -> list[tuple[str, str, float]]:
    cols = {str(r[1]) for r in _columns(conn, "campaigns")}
    if not _V1_REQUIRED_COLUMNS <= cols:
        return []
    # One row per tenant: the newest. A bare column next to MAX() comes from the
    # row that produced the max in SQLite, which is exactly the row wanted.
    rows = conn.execute(
        "SELECT tenant, state, MAX(updated_at) FROM campaigns GROUP BY tenant"
    ).fetchall()
    return [(str(t), s, float(u)) for t, s, u in rows]


def _repair_in_txn(conn: sqlite3.Connection) -> str:
    """Rename an incompatible `campaigns` to `campaigns_backup_N` and recreate
    it in the v1 shape with the salvageable rows. Runs inside the caller's
    transaction. Returns the operator-facing note."""
    note = _describe_repair(conn)
    rows = _salvageable_rows(conn)
    backup = _free_backup_table(conn)
    conn.execute(f'ALTER TABLE campaigns RENAME TO "{backup}"')
    conn.execute(V1_CAMPAIGNS_DDL)
    conn.executemany(
        "INSERT INTO campaigns (tenant, state, updated_at) VALUES (?, ?, ?)", rows
    )
    return note


# --- backup --------------------------------------------------------------- #


def _online_backup(src: sqlite3.Connection, dest_path: Path) -> None:
    """SQLite's online backup API into `dest_path`. Separate function so a test
    can make it fail."""
    dst = sqlite3.connect(str(dest_path))
    try:
        src.backup(dst)
    finally:
        dst.close()


def take_backup(conn: sqlite3.Connection, db_path: Path, label: str) -> Path:
    """Online-backup `db_path` to `<db>.<label>-<UTC timestamp>` and verify it.

    Written to a `.tmp` name first and renamed only after `PRAGMA quick_check`
    passes and the table set matches, so a half-written file never carries the
    final name. The copy gets the source file's permission bits (it holds the
    same tenant data). Any failure raises `MigrationError` and removes the
    partial file; the source is never written.
    """
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = db_path.with_name(f"{db_path.name}.{label}-{ts}")
    dest = base
    n = 2
    while dest.exists() or dest.with_name(dest.name + ".tmp").exists():
        dest = base.with_name(f"{base.name}-{n}")
        n += 1
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        _online_backup(conn, tmp)
        check = sqlite3.connect(str(tmp))
        try:
            ok = check.execute("PRAGMA quick_check").fetchone()[0]
            copied = _tables(check)
        finally:
            check.close()
        if ok != "ok" or copied != _tables(conn):
            raise MigrationError(f"backup verification failed ({ok})")
        os.chmod(tmp, stat.S_IMODE(os.stat(db_path).st_mode))
        os.replace(tmp, dest)
    except MigrationError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - any backup failure aborts the migration
        tmp.unlink(missing_ok=True)
        raise MigrationError(f"backup to {dest} failed: {exc}") from exc
    return dest


# --- up ------------------------------------------------------------------- #


@dataclass
class MigrationResult:
    action: str  # "noop" | "created" | "migrated" | "rolled_back" | "dry_run"
    plan_text: str
    backup_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _new_campaign_id() -> str:
    return "cmp_" + uuid.uuid4().hex


def _create_v2(conn: sqlite3.Connection) -> None:
    for ddl in _V2_DDL:
        conn.execute(ddl)


def _record_version(conn: sqlite3.Connection, version: int, name: str) -> None:
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
        (version, name, time.time()),
    )


def _apply_v3_in_txn(conn: sqlite3.Connection) -> None:
    """v2 -> v3, inside the caller's BEGIN IMMEDIATE: create the `jobs` table.
    Additive, so no data moves."""
    for ddl in _V3_DDL:
        conn.execute(ddl)
    _record_version(conn, 3, V3_NAME)


def _generation_of(state: str | None) -> str | None:
    parsed = _parse_state(state)
    gen = parsed.get("generation") if parsed else None
    return gen if isinstance(gen, str) else None


def _apply_up_in_txn(conn: sqlite3.Connection, latest_dir: Path | None) -> list[str]:
    """The v1 -> v2 change, inside the caller's BEGIN IMMEDIATE. Returns warnings."""
    warnings: list[str] = []
    tables = _tables(conn)
    if "campaigns" in tables and not campaigns_table_is_v1_compatible(conn):
        warnings.append(_repair_in_txn(conn))
        tables = _tables(conn)
    v1_rows: list[tuple[str, str, float]] = []
    if "campaigns" in tables:
        raw = conn.execute("SELECT tenant, state, updated_at FROM campaigns").fetchall()
        v1_rows = [(str(t), s, float(u)) for t, s, u in raw]
        v1_name = _free_name(tables, "campaigns_v1")
        conn.execute(f'ALTER TABLE campaigns RENAME TO "{v1_name}"')
    _create_v2(conn)

    default_of: dict[str, str] = {}
    for tenant, state, updated_at in v1_rows:
        if not is_valid_tenant_id(tenant):
            warnings.append(
                f"tenant {tenant!r} fails the tenant-id pattern; imported unchanged "
                "(it cannot be reached through the API until renamed by an operator)"
            )
        if _parse_state(state) is None:
            warnings.append(f"tenant {tenant!r}: campaign state is not valid JSON; copied verbatim")
        cid = _new_campaign_id()
        conn.execute(
            "INSERT INTO campaigns (id, tenant, name, state, generation, created_by, "
            "created_at, updated_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (cid, tenant, IMPORTED_CAMPAIGN_NAME, state, _generation_of(state),
             MIGRATION_ACTOR, updated_at, updated_at),
        )
        conn.execute(
            "INSERT INTO tenant_defaults (tenant, campaign_id) VALUES (?, ?)", (tenant, cid)
        )
        default_of[tenant] = cid

    imports, skipped = classify_latest_files(latest_dir, set(default_of))
    for sk in skipped:
        warnings.append(f"latest/{sk.filename} not imported: {sk.reason}")
    for imp in imports:
        cid_or_none = default_of.get(imp.tenant)
        updated = imp.state.get("updated")
        ts = float(updated) if isinstance(updated, (int, float)) and not isinstance(updated, bool) else time.time()
        if cid_or_none is None:
            # An analysis with no campaign row: v1 served it on /api/latest while
            # /api/campaign said has_campaign=false. Keep both facts: an unseeded
            # (NULL-state) default campaign that owns the analysis.
            cid = _new_campaign_id()
            conn.execute(
                "INSERT INTO campaigns (id, tenant, name, state, generation, created_by, "
                "created_at, updated_at, archived_at) VALUES (?, ?, ?, NULL, NULL, ?, ?, ?, NULL)",
                (cid, imp.tenant, IMPORTED_CAMPAIGN_NAME, MIGRATION_ACTOR, ts, ts),
            )
            conn.execute(
                "INSERT INTO tenant_defaults (tenant, campaign_id) VALUES (?, ?)", (imp.tenant, cid)
            )
            default_of[imp.tenant] = cid
            warnings.append(
                f"latest/{imp.filename}: tenant {imp.tenant!r} had an analysis but no campaign; "
                "created an empty default campaign for it"
            )
            if not is_valid_tenant_id(imp.tenant):
                warnings.append(f"tenant {imp.tenant!r} fails the tenant-id pattern; imported unchanged")
        else:
            cid = cid_or_none
        dataset = imp.state.get("dataset")
        gen = imp.state.get("campaign_generation")
        conn.execute(
            "INSERT INTO analyses (campaign_id, tenant, state, dataset, campaign_generation, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, imp.tenant, json.dumps(imp.state),
             dataset if isinstance(dataset, str) else None,
             gen if isinstance(gen, str) else None, ts),
        )
    _record_version(conn, V2, V2_NAME)
    return warnings


def migrate_up(
    db_path: Path | str,
    *,
    latest_dir: Path | str | None = None,
    dry_run: bool = False,
) -> MigrationResult:
    """Bring `db_path` to the current schema (v3). Idempotent: a v3 database is
    a no-op. v0/v1 go through the v1 -> v2 migration and then v2 -> v3 in the
    SAME transaction; v2 only gets v2 -> v3 (backup label `pre-v3`).

    Order matters for safety:
      1. Read the version (no write). v2 -> return. Newer -> refuse.
      2. If the database has any tables, take a verified online backup. A
         failure raises `MigrationError` with the database untouched.
      3. One `BEGIN IMMEDIATE` transaction: re-check the version (another
         process may have migrated meanwhile), repair/rename, create v2, import,
         record the version. Any error rolls the whole thing back.
      4. Only then switch the file to WAL. That pragma rewrites the file header,
         so doing it earlier would modify the database even when step 2 aborts.

    `dry_run` opens the database read-only and returns the plan; nothing is
    written and no backup is taken.
    """
    db = Path(db_path)
    ldir = Path(latest_dir) if latest_dir is not None else db.parent / "latest"
    if dry_run:
        if not db.exists():
            plan = UpPlan(db, 0, needs_migration=True)
            plan.latest_imports, plan.latest_skipped = classify_latest_files(ldir, set())
            plan.empty_defaults = [i.tenant for i in plan.latest_imports]
            return MigrationResult("dry_run", plan.render() + "\n  (database file does not exist yet)")
        ro = _connect_readonly(db)
        try:
            plan = _plan_up(ro, db, ldir)
        finally:
            ro.close()
        _refuse_newer(plan.from_version)
        return MigrationResult("dry_run", plan.render())

    conn = connect(db)
    try:
        version = current_version(conn)
        _refuse_newer(version)
        if version == SCHEMA_VERSION:
            _assert_v2_shape(conn, version)
            enable_wal(conn)
            return MigrationResult("noop", f"{db}: already at schema v{SCHEMA_VERSION}")
        if version == V2:
            _assert_v2_shape(conn, version)
        plan_text = _plan_up(conn, db, ldir).render()
        backup: Path | None = None
        if _tables(conn):
            backup = take_backup(conn, db, "pre-v3" if version == V2 else "pre-v2")
            log.warning("kalos portal.db: pre-migration backup written to %s", backup)
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_version = current_version(conn)
            if now_version >= SCHEMA_VERSION:
                conn.execute("ROLLBACK")
                _assert_v2_shape(conn)
                enable_wal(conn)
                return MigrationResult("noop", f"{db}: migrated concurrently by another process", backup)
            warnings = _apply_up_in_txn(conn, ldir) if now_version < V2 else []
            _apply_v3_in_txn(conn)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        enable_wal(conn)
    finally:
        conn.close()
    for w in warnings:
        log.warning("kalos portal.db migration: %s", w)
    action = "migrated" if version > 0 or backup is not None else "created"
    return MigrationResult(action, plan_text, backup, warnings)


def _refuse_newer(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise SchemaError(
            f"portal.db is at schema v{version}, newer than this code (v{SCHEMA_VERSION}); "
            "refusing to run against it"
        )


def ensure_schema(db_path: Path | str, *, latest_dir: Path | str | None = None) -> MigrationResult:
    """Startup hook used by `CampaignStore`: migrate to v3 when needed, else verify."""
    return migrate_up(db_path, latest_dir=latest_dir)


# --- down ----------------------------------------------------------------- #


@dataclass
class DownPlan:
    """The v2 -> v1 step of a rollback."""

    db_path: Path
    from_version: int
    rows: list[tuple[str, str, str, float]] = field(default_factory=list)  # tenant, cid, state, updated_at
    exports: list[tuple[str, str]] = field(default_factory=list)  # tenant, campaign id of the analysis
    skipped_tenants: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.from_version not in (V2, SCHEMA_VERSION):
            return f"{self.db_path}: at schema v{self.from_version}; nothing to roll back"
        lines = [
            f"{self.db_path}: schema v{V2} -> v1",
            f"  backup: online copy to {self.db_path.name}.pre-down-<UTC timestamp>",
            "  rename: " + ", ".join(f"{t} -> {t}_v2_rolledback" for t in _V2_TABLES),
            f"  campaigns (v1): {len(self.rows)} tenant row(s)",
        ]
        for tenant, cid, _, _ in self.rows:
            lines.append(f"    tenant {tenant!r} <- {cid}")
        for t in self.skipped_tenants:
            lines.append(f"  skip: tenant {t!r} has no seeded campaign; no v1 row")
        for tenant, cid in self.exports:
            lines.append(
                f"  analysis: {cid} -> latest/{legacy_latest_filename(tenant)} "
                "(an existing file is moved aside, not overwritten)"
            )
        return "\n".join(lines)


def _plan_down(conn: sqlite3.Connection, db_path: Path) -> DownPlan:
    """Which campaign becomes each tenant's v1 row, and which analysis becomes
    its `latest/` file. Reads only the v2 tables under their v2 names (which a
    v3 database still has: v3 only adds `jobs`)."""
    version = current_version(conn)
    plan = DownPlan(db_path, version)
    if version not in (V2, SCHEMA_VERSION):
        return plan
    tenants = [str(r[0]) for r in conn.execute("SELECT DISTINCT tenant FROM campaigns ORDER BY tenant")]
    for tenant in tenants:
        row = conn.execute(
            "SELECT c.id, c.state, c.updated_at FROM tenant_defaults d "
            "JOIN campaigns c ON c.id = d.campaign_id AND c.tenant = d.tenant "
            "WHERE d.tenant = ? AND c.state IS NOT NULL",
            (tenant,),
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT id, state, updated_at FROM campaigns WHERE tenant = ? "
                "AND state IS NOT NULL AND archived_at IS NULL "
                "ORDER BY updated_at DESC LIMIT 1",
                (tenant,),
            ).fetchone()
        chosen: str | None = None
        if row is None:
            plan.skipped_tenants.append(tenant)
        else:
            chosen = str(row[0])
            plan.rows.append((tenant, chosen, str(row[1]), float(row[2])))
        # The analysis the v1 code should serve: the default campaign's, else
        # the chosen campaign's.
        arow = conn.execute(
            "SELECT a.campaign_id FROM tenant_defaults d JOIN analyses a "
            "ON a.campaign_id = d.campaign_id AND a.tenant = d.tenant WHERE d.tenant = ?",
            (tenant,),
        ).fetchone()
        if arow is None and chosen is not None:
            arow = conn.execute(
                "SELECT campaign_id FROM analyses WHERE campaign_id = ? AND tenant = ?",
                (chosen, tenant),
            ).fetchone()
        if arow is not None and is_valid_tenant_id(tenant):
            plan.exports.append((tenant, str(arow[0])))
    return plan


def _render_down_v3(db_path: Path, conn: sqlite3.Connection) -> str:
    """The v3 -> v2 step: rename `jobs` aside. Reports how many jobs were still
    unfinished (they will not run on v2 code)."""
    n_active = 0
    if "jobs" in _tables(conn):
        n_active = int(conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE state IN ('queued', 'running')"
        ).fetchone()[0])
    return "\n".join([
        f"{db_path}: schema v{SCHEMA_VERSION} -> v{V2}",
        f"  backup: online copy to {db_path.name}.pre-down-<UTC timestamp>",
        "  rename: jobs -> jobs_v3_rolledback (kept, not dropped)",
        f"  unfinished jobs: {n_active} (queued/running; they will not run on v2 code)",
        "  job input files under <state dir>/jobs/ are left on disk",
    ])


def _down_v3_in_txn(conn: sqlite3.Connection) -> None:
    """v3 -> v2 inside the caller's BEGIN IMMEDIATE. Nothing is dropped except
    the v3 indexes; the `jobs` rows stay readable in `jobs_v3_rolledback`."""
    tables = _tables(conn)
    for idx in _V3_INDEXES:
        conn.execute(f'DROP INDEX IF EXISTS "{idx}"')
    for table in _V3_TABLES:
        if table in tables:
            new = _free_name(tables, f"{table}_v3_rolledback")
            conn.execute(f'ALTER TABLE "{table}" RENAME TO "{new}"')
            tables.add(new)
    conn.execute("DELETE FROM schema_migrations WHERE version = ?", (SCHEMA_VERSION,))


def migrate_down(
    db_path: Path | str,
    *,
    latest_dir: Path | str | None = None,
    dry_run: bool = False,
    target: int | None = None,
) -> MigrationResult:
    """Roll the database back, ONE version by default (v3 -> v2, or v2 -> v1),
    or down to `target` (1 or 2).

    v3 -> v2: renames `jobs` to `jobs_v3_rolledback` and forgets version 3. The
    v2 tables are untouched, so the previous (E4) engine runs on the result.

    v2 -> v1: rebuilds `campaigns(tenant PK, state, updated_at)` from each
    tenant's default campaign (or, without a seeded default, its newest seeded
    non-archived campaign), and renames every v2 table to
    `<name>_v2_rolledback` - nothing is dropped, so the other campaigns stay
    recoverable. After commit, each tenant's analysis is written to
    `latest/t-<hex>.json` (the file the pre-v2 code reads), moving any existing
    file aside to `<name>.pre-down-<ts>` rather than overwriting it.

    One `pre-down` backup is taken first and all steps run in one transaction.
    Stop the engine first: a running process would see its tables vanish.
    """
    db = Path(db_path)
    ldir = Path(latest_dir) if latest_dir is not None else db.parent / "latest"
    if not db.exists():
        return MigrationResult("noop", f"{db}: does not exist; nothing to roll back")

    def _target_for(version: int) -> int | None:
        if version not in (V2, SCHEMA_VERSION):
            return None
        goal = version - 1 if target is None else target
        if goal not in (1, V2) or goal >= version:
            raise MigrationError(
                f"cannot roll back from schema v{version} to v{goal}; "
                f"valid targets are 1..{version - 1}"
            )
        return goal

    if dry_run:
        ro = _connect_readonly(db)
        try:
            version = current_version(ro)
            goal = _target_for(version)
            if goal is None:
                return MigrationResult("dry_run", DownPlan(db, version).render())
            dry_parts: list[str] = []
            if version == SCHEMA_VERSION:
                dry_parts.append(_render_down_v3(db, ro))
            if goal == 1:
                dry_parts.append(_plan_down(ro, db).render())
        finally:
            ro.close()
        return MigrationResult("dry_run", "\n".join(dry_parts))

    conn = connect(db)
    try:
        version = current_version(conn)
        goal = _target_for(version)
        if goal is None:
            return MigrationResult(
                "noop", f"{db}: at schema v{version}, not v{V2} or v{SCHEMA_VERSION}; nothing to roll back"
            )
        _assert_v2_shape(conn, version)
        backup = take_backup(conn, db, "pre-down")
        parts: list[str] = []
        renamed: dict[str, str] = {}
        plan: DownPlan | None = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            if version == SCHEMA_VERSION:
                parts.append(_render_down_v3(db, conn))
                _down_v3_in_txn(conn)
            if goal == 1:
                plan = _plan_down_from_v2(conn, db)
                parts.append(plan.render())
                tables = _tables(conn)
                for idx in _V2_INDEXES:
                    conn.execute(f'DROP INDEX IF EXISTS "{idx}"')
                for table in _V2_TABLES:
                    new = _free_name(tables, f"{table}_v2_rolledback")
                    conn.execute(f'ALTER TABLE "{table}" RENAME TO "{new}"')
                    tables.add(new)
                    renamed[table] = new
                conn.execute(V1_CAMPAIGNS_DDL)
                conn.executemany(
                    "INSERT INTO campaigns (tenant, state, updated_at) VALUES (?, ?, ?)",
                    [(t, s, u) for t, _, s, u in plan.rows],
                )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        warnings = (
            _export_latest(conn, renamed["analyses"], plan.exports, ldir) if plan is not None else []
        )
    finally:
        conn.close()
    for w in warnings:
        log.warning("kalos portal.db rollback: %s", w)
    return MigrationResult("rolled_back", "\n".join(parts), backup, warnings)


def _plan_down_from_v2(conn: sqlite3.Connection, db_path: Path) -> DownPlan:
    """`_plan_down` inside a rollback transaction, after the v3 step has already
    removed version 3 (so the recorded version reads 2)."""
    plan = _plan_down(conn, db_path)
    if plan.from_version not in (V2, SCHEMA_VERSION):  # pragma: no cover - guarded above
        raise MigrationError("rollback planning saw an unexpected schema version")
    return plan


def _export_latest(
    conn: sqlite3.Connection, analyses_table: str, exports: list[tuple[str, str]], ldir: Path
) -> list[str]:
    warnings: list[str] = []
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for tenant, cid in exports:
        row = conn.execute(
            f'SELECT state FROM "{analyses_table}" WHERE campaign_id = ? AND tenant = ?',
            (cid, tenant),
        ).fetchone()
        if row is None:
            continue
        try:
            ldir.mkdir(parents=True, exist_ok=True)
            path = ldir / legacy_latest_filename(tenant)
            if path.exists():
                aside = path.with_name(f"{path.name}.pre-down-{ts}")
                n = 2
                while aside.exists():
                    aside = path.with_name(f"{path.name}.pre-down-{ts}-{n}")
                    n += 1
                os.replace(path, aside)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(str(row[0]), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            warnings.append(f"could not export the analysis for tenant {tenant!r}: {exc}")
    return warnings


# --- status ----------------------------------------------------------------- #


def status(db_path: Path | str, *, latest_dir: Path | str | None = None) -> str:
    db = Path(db_path)
    ldir = Path(latest_dir) if latest_dir is not None else db.parent / "latest"
    lines = [f"database: {db}"]
    if not db.exists():
        lines.append("  does not exist (a fresh v2 database is created on first start)")
        return "\n".join(lines)
    ro = _connect_readonly(db)
    try:
        version = current_version(ro)
        mode = ro.execute("PRAGMA journal_mode").fetchone()[0]
        lines.append(f"  schema version: v{version} (this code: v{SCHEMA_VERSION}); journal_mode={mode}")
        for table in sorted(_tables(ro)):
            n = ro.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            lines.append(f"  table {table}: {n} row(s)")
        if version < SCHEMA_VERSION:
            lines.append("pending migration (dry run):")
            lines.append(_plan_up(ro, db, ldir).render())
    finally:
        ro.close()
    backups = sorted(db.parent.glob(f"{db.name}.pre-*"))
    for b in backups:
        lines.append(f"  backup file: {b.name}")
    return "\n".join(lines)


# --- CLI ------------------------------------------------------------------ #


def _default_state_dir() -> Path:
    return Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kalos.portal.migrate",
        description="Inspect, apply or roll back the kalos portal.db schema migrations (v1 <-> v2 <-> v3).",
    )
    parser.add_argument("command", choices=["up", "down", "status"])
    parser.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
    parser.add_argument("--state-dir", type=Path, default=None,
                        help="state directory (default: $KALOS_STATE_DIR or ~/.kalos)")
    parser.add_argument("--db", type=Path, default=None, help="database path (default: <state-dir>/portal.db)")
    parser.add_argument("--latest-dir", type=Path, default=None,
                        help="legacy analysis cache directory (default: <db dir>/latest)")
    parser.add_argument("--to", type=int, default=None, dest="target",
                        help="`down` only: the schema version to roll back to "
                             "(default: one version below the current one)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    state_dir = args.state_dir or _default_state_dir()
    db = args.db or state_dir / DB_FILENAME
    try:
        if args.command == "status":
            print(status(db, latest_dir=args.latest_dir))
            return 0
        if args.command == "up":
            if not args.dry_run and not db.exists():
                db.parent.mkdir(parents=True, exist_ok=True)
            result = migrate_up(db, latest_dir=args.latest_dir, dry_run=args.dry_run)
        else:
            result = migrate_down(
                db, latest_dir=args.latest_dir, dry_run=args.dry_run, target=args.target
            )
    except (MigrationError, SchemaError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.plan_text)
    if result.backup_path is not None:
        print(f"backup: {result.backup_path}")
    for w in result.warnings:
        print(f"warning: {w}")
    print(f"result: {result.action}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
