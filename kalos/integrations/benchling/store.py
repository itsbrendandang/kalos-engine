"""Connector state in `<state_dir>/integrations.db` (a separate SQLite file, so the
campaign store's schema and migrations are untouched).

Every statement carries `tenant = ?`; no method can read or write across
tenants. Tables:

- `benchling_connections`: one row per Kalos tenant: base URL, app client id,
  the app secret sealed with `KALOS_SECRET_KEY` (kalos.integrations.benchling.
  secrets), the app definition id (for webhook keys), who connected and when.
- `benchling_imports`: one row per import (study or schema id, fetch time,
  filename, sheet sha256, provenance and report). Append-only.
- `benchling_push_audit`: one row per push attempt with the Benchling ids that
  came back, the Kalos subject, and which Benchling identity wrote (`app` or
  `delegated`). Append-only (UPDATE/DELETE rejected by triggers).
- `benchling_pushed_results`: which (campaign, run, version) went to which
  Benchling result.
- `benchling_result_claims`: a (campaign, run, version) is claimed in one
  transaction BEFORE it is sent, so two pushes at once (threads or processes)
  never both write it. A claim is released only when Benchling provably wrote
  nothing (the transaction was aborted, or never opened).
- `benchling_webhook_deliveries`: webhook-id seen per tenant (re-delivery is a no-op).
- `benchling_changes`: change notices from webhooks and polling. Append-only.
- `benchling_cursors`: polling high-water marks.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kalos.integrations.benchling.secrets import open_sealed, seal
from kalos.portal.tenancy import validate_tenant_id

DB_FILENAME = "integrations.db"
MAX_DETAIL_CHARS = 512

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchling_connections (
    tenant TEXT PRIMARY KEY,
    base_url TEXT NOT NULL,
    client_id TEXT NOT NULL,
    secret_sealed TEXT NOT NULL,
    app_definition_id TEXT,
    connected_by TEXT,
    connected_at REAL NOT NULL,
    verified_at REAL
);
CREATE TABLE IF NOT EXISTS benchling_imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    campaign_id TEXT,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    sheet_sha256 TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT NOT NULL,
    created_by TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_benchling_imports ON benchling_imports(tenant, campaign_id, id);
CREATE TABLE IF NOT EXISTS benchling_push_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    tenant TEXT NOT NULL,
    subject TEXT,
    kalos_auth_method TEXT,
    benchling_identity TEXT NOT NULL,
    benchling_creator TEXT,
    action TEXT NOT NULL,
    campaign_id TEXT,
    source_id TEXT,
    outcome TEXT NOT NULL,
    benchling_ids TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_benchling_push_audit ON benchling_push_audit(tenant, id);
CREATE TRIGGER IF NOT EXISTS benchling_push_audit_no_update BEFORE UPDATE ON benchling_push_audit
BEGIN SELECT RAISE(ABORT, 'benchling_push_audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS benchling_push_audit_no_delete BEFORE DELETE ON benchling_push_audit
BEGIN SELECT RAISE(ABORT, 'benchling_push_audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS benchling_imports_no_update BEFORE UPDATE ON benchling_imports
BEGIN SELECT RAISE(ABORT, 'benchling_imports is append-only'); END;
CREATE TRIGGER IF NOT EXISTS benchling_imports_no_delete BEFORE DELETE ON benchling_imports
BEGIN SELECT RAISE(ABORT, 'benchling_imports is append-only'); END;
CREATE TABLE IF NOT EXISTS benchling_pushed_results (
    tenant TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    benchling_result_id TEXT NOT NULL,
    push_audit_id INTEGER NOT NULL,
    PRIMARY KEY (tenant, campaign_id, run_id, version)
);
CREATE TABLE IF NOT EXISTS benchling_result_claims (
    tenant TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    claimed_at REAL NOT NULL,
    PRIMARY KEY (tenant, campaign_id, run_id, version)
);
CREATE TABLE IF NOT EXISTS benchling_webhook_deliveries (
    tenant TEXT NOT NULL,
    webhook_id TEXT NOT NULL,
    received_at REAL NOT NULL,
    event_type TEXT,
    PRIMARY KEY (tenant, webhook_id)
);
CREATE TABLE IF NOT EXISTS benchling_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    origin TEXT NOT NULL,
    event_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    marker TEXT NOT NULL,
    scope_id TEXT,
    observed_at REAL NOT NULL,
    UNIQUE (tenant, origin, event_type, resource_id, marker)
);
CREATE TRIGGER IF NOT EXISTS benchling_changes_no_update BEFORE UPDATE ON benchling_changes
BEGIN SELECT RAISE(ABORT, 'benchling_changes is append-only'); END;
CREATE TRIGGER IF NOT EXISTS benchling_changes_no_delete BEFORE DELETE ON benchling_changes
BEGIN SELECT RAISE(ABORT, 'benchling_changes is append-only'); END;
CREATE TABLE IF NOT EXISTS benchling_cursors (
    tenant TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (tenant, key)
);
"""


def _clip(value: Any, limit: int = MAX_DETAIL_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:limit]


class IntegrationStore:
    def __init__(self, state_dir: Path) -> None:
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / DB_FILENAME
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA)
        try:
            self._path.chmod(0o600)  # holds sealed secrets; owner-only
        except OSError:
            pass

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    # --- connection ------------------------------------------------------- #

    def save_connection(
        self, *, tenant: str, base_url: str, client_id: str, client_secret: str,
        app_definition_id: str | None, connected_by: str | None, verified: bool,
    ) -> None:
        validate_tenant_id(tenant)
        sealed = seal(client_secret, tenant=tenant)
        now = time.time()
        with self._txn() as c:
            c.execute(
                "INSERT INTO benchling_connections (tenant, base_url, client_id, secret_sealed, "
                "app_definition_id, connected_by, connected_at, verified_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(tenant) DO UPDATE SET base_url=excluded.base_url, client_id=excluded.client_id, "
                "secret_sealed=excluded.secret_sealed, app_definition_id=excluded.app_definition_id, "
                "connected_by=excluded.connected_by, connected_at=excluded.connected_at, "
                "verified_at=excluded.verified_at",
                (tenant, base_url, client_id, sealed, app_definition_id, connected_by, now,
                 now if verified else None),
            )

    def connection(self, *, tenant: str) -> dict[str, Any] | None:
        """The tenant's connection WITHOUT the secret."""
        validate_tenant_id(tenant)
        with self._lock:
            row = self._conn.execute(
                "SELECT base_url, client_id, app_definition_id, connected_by, connected_at, verified_at "
                "FROM benchling_connections WHERE tenant = ?", (tenant,),
            ).fetchone()
        if row is None:
            return None
        keys = ("base_url", "client_id", "app_definition_id", "connected_by", "connected_at", "verified_at")
        return dict(zip(keys, row))

    def host_used_by_another_tenant(self, *, tenant: str, base_url: str) -> bool:
        """Whether another Kalos tenant already has a connection to `base_url`."""
        validate_tenant_id(tenant)
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM benchling_connections WHERE base_url = ? AND tenant != ? LIMIT 1", (base_url, tenant),
            ).fetchone()
        return row is not None

    def client_secret(self, *, tenant: str) -> str | None:
        """The opened app secret, for building a client. Never logged or returned
        by a route. Raises SecretUnreadable / SecretKeyUnavailable."""
        validate_tenant_id(tenant)
        with self._lock:
            row = self._conn.execute(
                "SELECT secret_sealed FROM benchling_connections WHERE tenant = ?", (tenant,),
            ).fetchone()
        if row is None:
            return None
        return open_sealed(row[0], tenant=tenant)

    # --- imports ---------------------------------------------------------- #

    def record_import(
        self, *, tenant: str, campaign_id: str | None, summary: dict[str, Any], sheet: bytes,
        outcome: str, created_by: str | None,
    ) -> int:
        validate_tenant_id(tenant)
        with self._txn() as c:
            cur = c.execute(
                "INSERT INTO benchling_imports (tenant, campaign_id, source, source_id, fetched_at, filename, "
                "sheet_sha256, summary, outcome, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (tenant, campaign_id, summary["source"], summary["source_id"], summary["fetched_at"],
                 summary["filename"], hashlib.sha256(sheet).hexdigest(), json.dumps(summary, sort_keys=True),
                 _clip(outcome, 64), _clip(created_by), time.time()),
            )
            return int(cur.lastrowid or 0)

    def latest_import(self, *, tenant: str, campaign_id: str) -> dict[str, Any] | None:
        """The last SUCCESSFUL import into `campaign_id` (its provenance drives pushes)."""
        validate_tenant_id(tenant)
        with self._lock:
            row = self._conn.execute(
                "SELECT id, summary, sheet_sha256 FROM benchling_imports WHERE tenant = ? AND campaign_id = ? "
                "AND outcome = 'ok' ORDER BY id DESC LIMIT 1", (tenant, campaign_id),
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "sheet_sha256": row[2], **json.loads(row[1])}

    def imports(self, *, tenant: str, limit: int = 20) -> list[dict[str, Any]]:
        validate_tenant_id(tenant)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, campaign_id, source, source_id, fetched_at, filename, sheet_sha256, outcome, created_by, "
                "created_at FROM benchling_imports WHERE tenant = ? ORDER BY id DESC LIMIT ?",
                (tenant, max(1, min(limit, 200))),
            ).fetchall()
        keys = ("id", "campaign_id", "source", "source_id", "fetched_at", "filename", "sheet_sha256", "outcome",
                "created_by", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    # --- push audit ------------------------------------------------------- #

    def record_push(
        self, *, tenant: str, subject: str | None, kalos_auth_method: str | None, benchling_identity: str,
        action: str, campaign_id: str | None, source_id: str | None, outcome: str,
        benchling_ids: dict[str, Any], benchling_creator: str | None = None, detail: str | None = None,
        pushed_results: list[tuple[str, int, str]] | None = None,
    ) -> int:
        """One append-only push row (and, for results, the pushed-result index in
        the same transaction)."""
        validate_tenant_id(tenant)
        if benchling_identity not in ("app", "delegated"):
            raise ValueError("benchling_identity must be 'app' or 'delegated'")
        with self._txn() as c:
            cur = c.execute(
                "INSERT INTO benchling_push_audit (ts, tenant, subject, kalos_auth_method, benchling_identity, "
                "benchling_creator, action, campaign_id, source_id, outcome, benchling_ids, detail) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), tenant, _clip(subject), _clip(kalos_auth_method, 32), benchling_identity,
                 _clip(benchling_creator, 128), _clip(action, 64), _clip(campaign_id, 64), _clip(source_id, 128),
                 _clip(outcome, 64), json.dumps(benchling_ids, sort_keys=True), _clip(detail)),
            )
            audit_id = int(cur.lastrowid or 0)
            for run_id, version, result_id in pushed_results or []:
                c.execute(
                    "INSERT INTO benchling_pushed_results (tenant, campaign_id, run_id, version, "
                    "benchling_result_id, push_audit_id) VALUES (?,?,?,?,?,?)",
                    (tenant, campaign_id, run_id, version, result_id, audit_id),
                )
            return audit_id

    def push_audit(self, *, tenant: str, limit: int = 50) -> list[dict[str, Any]]:
        validate_tenant_id(tenant)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, subject, kalos_auth_method, benchling_identity, benchling_creator, action, "
                "campaign_id, source_id, outcome, benchling_ids, detail FROM benchling_push_audit "
                "WHERE tenant = ? ORDER BY id DESC LIMIT ?", (tenant, max(1, min(limit, 500))),
            ).fetchall()
        keys = ("id", "ts", "subject", "kalos_auth_method", "benchling_identity", "benchling_creator", "action",
                "campaign_id", "source_id", "outcome", "benchling_ids", "detail")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["benchling_ids"] = json.loads(d["benchling_ids"])
            out.append(d)
        return out

    def pushed_versions(self, *, tenant: str, campaign_id: str) -> set[tuple[str, int]]:
        """(run_id, version) pairs pushed or claimed by a push in flight."""
        validate_tenant_id(tenant)
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, version FROM benchling_pushed_results WHERE tenant = ? AND campaign_id = ? "
                "UNION SELECT run_id, version FROM benchling_result_claims WHERE tenant = ? AND campaign_id = ?",
                (tenant, campaign_id, tenant, campaign_id),
            ).fetchall()
        return {(r[0], int(r[1])) for r in rows}

    def claim_results(
        self, *, tenant: str, campaign_id: str, items: list[tuple[str, int]],
    ) -> list[tuple[str, int]]:
        """Claim each (run_id, version) not pushed or claimed before. Returns the
        ones THIS call claimed; a concurrent push gets none of them."""
        validate_tenant_id(tenant)
        claimed: list[tuple[str, int]] = []
        now = time.time()
        with self._txn() as c:
            for run_id, version in items:
                pushed = c.execute(
                    "SELECT 1 FROM benchling_pushed_results WHERE tenant = ? AND campaign_id = ? AND run_id = ? "
                    "AND version = ?", (tenant, campaign_id, run_id, version),
                ).fetchone()
                if pushed is not None:
                    continue
                cur = c.execute(
                    "INSERT OR IGNORE INTO benchling_result_claims (tenant, campaign_id, run_id, version, claimed_at) "
                    "VALUES (?,?,?,?,?)", (tenant, campaign_id, run_id, version, now),
                )
                if cur.rowcount == 1:
                    claimed.append((run_id, version))
        return claimed

    def release_claims(self, *, tenant: str, campaign_id: str, items: list[tuple[str, int]]) -> None:
        """Give claims back after a push that provably wrote nothing."""
        validate_tenant_id(tenant)
        with self._txn() as c:
            c.executemany(
                "DELETE FROM benchling_result_claims WHERE tenant = ? AND campaign_id = ? AND run_id = ? "
                "AND version = ?", [(tenant, campaign_id, r, v) for r, v in items],
            )

    # --- webhooks and polling ------------------------------------------------ #

    def claim_delivery(self, *, tenant: str, webhook_id: str, event_type: str | None) -> bool:
        """True the first time `webhook_id` is seen for this tenant, False on a
        re-delivery."""
        validate_tenant_id(tenant)
        with self._txn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO benchling_webhook_deliveries (tenant, webhook_id, received_at, event_type) "
                "VALUES (?,?,?,?)", (tenant, _clip(webhook_id, 128), time.time(), _clip(event_type, 64)),
            )
            return cur.rowcount == 1

    def record_change(
        self, *, tenant: str, origin: str, event_type: str, resource_id: str, marker: str,
        scope_id: str | None = None,
    ) -> bool:
        """Append a change notice; False when the same (origin, type, resource,
        marker) is already recorded."""
        validate_tenant_id(tenant)
        with self._txn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO benchling_changes (tenant, origin, event_type, resource_id, marker, "
                "scope_id, observed_at) VALUES (?,?,?,?,?,?,?)",
                (tenant, _clip(origin, 16), _clip(event_type, 64), _clip(resource_id, 128), _clip(marker, 64),
                 _clip(scope_id, 128), time.time()),
            )
            return cur.rowcount == 1

    def changes(self, *, tenant: str, after_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        validate_tenant_id(tenant)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, origin, event_type, resource_id, marker, scope_id, observed_at FROM benchling_changes "
                "WHERE tenant = ? AND id > ? ORDER BY id LIMIT ?", (tenant, after_id, max(1, min(limit, 1000))),
            ).fetchall()
        keys = ("id", "origin", "event_type", "resource_id", "marker", "scope_id", "observed_at")
        return [dict(zip(keys, r)) for r in rows]

    def cursor(self, *, tenant: str, key: str) -> str | None:
        validate_tenant_id(tenant)
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM benchling_cursors WHERE tenant = ? AND key = ?", (tenant, key),
            ).fetchone()
        return row[0] if row else None

    def set_cursor(self, *, tenant: str, key: str, value: str) -> None:
        validate_tenant_id(tenant)
        with self._txn() as c:
            c.execute(
                "INSERT INTO benchling_cursors (tenant, key, value) VALUES (?,?,?) "
                "ON CONFLICT(tenant, key) DO UPDATE SET value = excluded.value", (tenant, key, value),
            )


_STORES: dict[Path, IntegrationStore] = {}
_STORES_LOCK = threading.Lock()


def integration_store_at(state_dir: Path) -> IntegrationStore:
    """One store per state directory, shared by the process."""
    key = Path(state_dir).resolve()
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = _STORES[key] = IntegrationStore(key)
        return store
