"""The storage backend: SQLite (the default) or PostgreSQL, behind one interface.

`KALOS_DATABASE_URL` selects the backend (read through `kalos.envconfig` rules:
an unknown value is refused, never guessed):

- unset or empty: SQLite, the `portal.db` file in the state directory, as always;
- `postgresql://...` or `postgres://...`: PostgreSQL through psycopg 3
  (`pip install 'kalos[postgres]'`), imported only on this path;
- anything else: `StorageConfigError` at startup.

Scope today (docs/MIGRATIONS.md, "PostgreSQL backend"): the records tables
(projects, factors, lineage; schema v9) use this backend. The campaign, job and
audit tables of `CampaignStore` move behind it in a later change (REORG_PLAN R06
and R12, `storage/campaign_repo.py`), so with a PostgreSQL URL the campaign data
still lives in `portal.db` until then.

SQL is written ONCE, dialect-neutral, with `?` placeholders. `translate_sql`
rewrites a statement for psycopg: `?` becomes `%s` (outside quoted text) and a
literal `%` becomes `%%`. Nothing else is translated, so a SQLite-only construct
in new code fails loudly on PostgreSQL instead of being silently reinterpreted.
The rules repositories follow:

- ids come back with `INSERT ... RETURNING id` (never `cursor.lastrowid`);
- upserts use `ON CONFLICT (...) DO UPDATE` (both dialects);
- JSON goes in through `Conn.json_param` (TEXT on SQLite, JSONB on PostgreSQL)
  and comes out through `json_value`, which accepts both shapes;
- booleans are INTEGER 0/1 and times are epoch seconds (REAL / DOUBLE PRECISION).

Transactions: a write on SQLite is `BEGIN IMMEDIATE` (the database write lock up
front); on PostgreSQL it is READ COMMITTED and takes one database-wide advisory
lock first, so writers serialize exactly as on SQLite, across processes. A read
is one snapshot (SQLite deferred BEGIN; PostgreSQL REPEATABLE READ READ ONLY).

The DSN is a credential: it is never logged or put in an error unredacted
(`redact_dsn`). This module is torch-free and FastAPI-free.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from kalos.envconfig import ConfigError, env_enum, env_int

_log = logging.getLogger("kalos.storage")

DATABASE_URL_ENV = "KALOS_DATABASE_URL"
POOL_MIN_ENV = "KALOS_DB_POOL_MIN"
POOL_MAX_ENV = "KALOS_DB_POOL_MAX"
CONNECT_TIMEOUT_ENV = "KALOS_DB_CONNECT_TIMEOUT"
SQLITE_FILENAME = "portal.db"
SQLITE_BUSY_TIMEOUT_MS = 5000

SQLITE = "sqlite"
POSTGRES = "postgres"

# The key of the database-wide write lock (pg_advisory_xact_lock): "kalos" in
# ASCII. Every write transaction and every migration takes it first.
WRITE_LOCK_KEY = 0x6B616C6F73

_SAFE_SSLMODES = ("require", "verify-ca", "verify-full")
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "")


class StorageConfigError(ConfigError):
    """`KALOS_DATABASE_URL` (or a pool setting) holds a value the engine refuses,
    or the PostgreSQL driver is missing. Raised at startup, before any request.
    The message never contains the password."""


@dataclass(frozen=True)
class BackendConfig:
    kind: str  # SQLITE | POSTGRES
    dsn: str | None = None  # PostgreSQL only; never logged unredacted

    @property
    def is_postgres(self) -> bool:
        return self.kind == POSTGRES

    def describe(self) -> str:
        if self.kind == SQLITE:
            return "sqlite (portal.db in the state directory)"
        return f"postgres ({redact_dsn(self.dsn or '')})"


def backend_from_env(environ: Mapping[str, str] | None = None) -> BackendConfig:
    """The backend `KALOS_DATABASE_URL` selects. Unset or empty means SQLite."""
    env = os.environ if environ is None else environ
    url = (env.get(DATABASE_URL_ENV) or "").strip()
    if not url:
        return BackendConfig(SQLITE)
    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    if scheme in ("postgresql", "postgres"):
        return BackendConfig(POSTGRES, url)
    raise StorageConfigError(
        f"{DATABASE_URL_ENV} must be empty (SQLite in the state directory) or a "
        f"postgresql:// URL; got a value with scheme {scheme[:16] or '(none)'!r}"
    )


REDACTED_DSN = "postgresql://***"
_PORT_RE = re.compile(r"[0-9]+(,[0-9]+)*")


def conninfo(dsn: str) -> dict[str, str] | None:
    """The connection parameters libpq reads from `dsn` (psycopg's own parser,
    so `#`, `?` and `/` in a password are handled exactly as libpq does), or
    None when psycopg is missing or the string does not parse."""
    try:
        from psycopg.conninfo import conninfo_to_dict
    except ImportError:
        return None
    try:
        parsed = conninfo_to_dict(dsn)
    except Exception:  # noqa: BLE001 - any parse failure means "do not show it"
        return None
    return {str(k): str(v) for k, v in parsed.items() if v is not None}


def redact_dsn(dsn: str) -> str:
    """A safe description of `dsn` for logs and errors: only the user, host,
    port and database are kept, rebuilt from the PARSED parameters (never from
    the raw string), so no part of a password or any other parameter can
    appear. Anything that does not parse cleanly gives the fixed
    `postgresql://***`."""
    params = conninfo(dsn)
    if params is None:
        return REDACTED_DSN
    user, host = params.get("user", ""), params.get("host", "")
    port, dbname = params.get("port", ""), params.get("dbname", "")
    password = params.get("password", "")
    shown = (user, host, port, dbname)
    # A password with an unencoded '/' or '@' makes libpq read part of it as a
    # host, port or database: refuse to show anything rather than leak it.
    if (
        any("@" in v for v in shown)
        or (password and any(password in v for v in shown if v))
        or (port and _PORT_RE.fullmatch(port) is None)
        or "@" in dsn and not user
        or any(c in dbname for c in "/?#")
    ):
        return REDACTED_DSN
    netloc = (quote(user, safe="") + "@" if user else "") + quote(host, safe=",")
    if port:
        netloc += ":" + port
    return f"postgresql://{netloc}/{quote(dbname, safe='')}"


def _dsn_host_and_sslmode(dsn: str) -> tuple[str, str | None]:
    parts = urlsplit(dsn)
    query = parse_qs(parts.query)
    host = (parts.hostname or "").lower()
    if not host and "host" in query:  # postgresql:///db?host=/var/run/postgresql
        host = query["host"][0]
    sslmode = query.get("sslmode", [None])[0]
    return host, sslmode


def check_database_settings(environ: Mapping[str, str] | None = None) -> BackendConfig:
    """Validate the storage settings before the engine binds a socket.

    The environment profile (`KALOS_ENV`, kalos/portal/config.py) applies here
    too: in `production`, a PostgreSQL URL to a host off this machine must ask
    for TLS (`sslmode=require`, `verify-ca` or `verify-full`); a unix socket or
    a loopback host needs none. On PostgreSQL the driver and the pool bounds
    are checked too."""
    cfg = backend_from_env(environ)
    if not cfg.is_postgres:
        return cfg
    production = env_enum("KALOS_ENV", ("development", "production"), "development") == "production"
    host, sslmode = _dsn_host_and_sslmode(cfg.dsn or "")
    off_box = not (host in _LOOPBACK_HOSTS or host.startswith("/"))
    if production and off_box and sslmode not in _SAFE_SSLMODES:
        raise StorageConfigError(
            f"KALOS_ENV=production with {DATABASE_URL_ENV} pointing at another host requires "
            "TLS: add sslmode=require (or verify-ca / verify-full) to the URL"
        )
    _require_psycopg()
    lo, hi = _pool_bounds()
    if hi < lo:
        raise StorageConfigError(f"{POOL_MAX_ENV} ({hi}) is below {POOL_MIN_ENV} ({lo})")
    _connect_timeout()
    return cfg


def _pool_bounds() -> tuple[int, int]:
    lo = env_int(POOL_MIN_ENV, 1, minimum=0, maximum=64, error=StorageConfigError)
    hi = env_int(POOL_MAX_ENV, 10, minimum=1, maximum=256, error=StorageConfigError)
    return lo, hi


def _connect_timeout() -> int:
    return env_int(CONNECT_TIMEOUT_ENV, 10, minimum=1, maximum=300, error=StorageConfigError)


# --- SQL translation (neutral `?` dialect -> psycopg) ------------------------- #


def translate_sql(sql: str) -> str:
    """Rewrite neutral SQL for psycopg, for a statement that HAS parameters.
    Every `%` becomes `%%` (psycopg scans quoted text too); outside quoted text
    ('...' and "...") `?` becomes `%s`; a `?` inside quotes is left alone."""
    out: list[str] = []
    i, n, seg_start = 0, len(sql), 0

    def flush(end: int) -> None:
        out.append(sql[seg_start:end].replace("%", "%%").replace("?", "%s"))

    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            flush(i)
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled quote = escaped
                        j += 2
                        continue
                    break
                j += 1
            out.append(sql[i : j + 1].replace("%", "%%"))
            i = j + 1
            seg_start = i
            continue
        i += 1
    flush(n)
    return "".join(out)


_TRANSLATED: dict[str, str] = {}
_TRANSLATED_LOCK = threading.Lock()


def _translated(sql: str) -> str:
    hit = _TRANSLATED.get(sql)
    if hit is None:
        hit = translate_sql(sql)
        with _TRANSLATED_LOCK:
            if len(_TRANSLATED) < 4096:
                _TRANSLATED[sql] = hit
    return hit


# --- errors ------------------------------------------------------------------- #


def _pg_error_classes() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    try:
        import psycopg
    except ImportError:
        return (), ()
    return (psycopg.Error,), (psycopg.IntegrityError,)


_PG_DB, _PG_INTEGRITY = _pg_error_classes()
# For `except` sites that must catch a database error from either backend.
DB_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, *_PG_DB)
INTEGRITY_ERRORS: tuple[type[BaseException], ...] = (sqlite3.IntegrityError, *_PG_INTEGRITY)


def _require_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise StorageConfigError(
            f"{DATABASE_URL_ENV} selects PostgreSQL but psycopg 3 is not installed; "
            "install the extra: pip install 'kalos[postgres]'"
        ) from exc
    return psycopg


# --- one connection, either dialect ------------------------------------------- #


class Conn:
    """The connection a repository receives inside a transaction. `execute`
    takes neutral SQL with `?` placeholders and returns a DB-API cursor whose
    rows are tuples on both dialects."""

    def __init__(self, raw: Any, dialect: str) -> None:
        self.raw = raw
        self.dialect = dialect

    @property
    def is_postgres(self) -> bool:
        return self.dialect == POSTGRES

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        if self.dialect == SQLITE:
            return self.raw.execute(sql, tuple(params))
        cur = self.raw.cursor()
        if params:
            cur.execute(_translated(sql), tuple(params))
        else:
            cur.execute(sql)
        return cur

    def json_param(self, obj: Any) -> Any:
        """A JSON value to bind: TEXT on SQLite, JSONB on PostgreSQL. NaN and
        infinity are refused (they are not JSON)."""
        if self.dialect == SQLITE:
            return json.dumps(obj, sort_keys=True, allow_nan=False, ensure_ascii=False)
        from psycopg.types.json import Jsonb

        json.dumps(obj, allow_nan=False)  # same refusal as SQLite
        return Jsonb(obj)


def json_value(value: Any) -> Any:
    """A JSON column read back: SQLite gives TEXT, PostgreSQL JSONB gives the
    parsed value already. None stays None; unparseable text reads as None."""
    if value is None or not isinstance(value, (str, bytes, bytearray)):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


class Backend:
    """One database: `txn(write=...)` yields a `Conn` inside one transaction.

    SQLite: one shared connection to `<state_dir>/portal.db` (WAL, busy timeout),
    serialized by a thread lock, like `CampaignStore`. PostgreSQL: a psycopg_pool
    pool of autocommit connections whose transactions are opened explicitly."""

    def __init__(self, config: BackendConfig, *, sqlite_path: Path | None = None) -> None:
        self.config = config
        self.dialect = config.kind
        self._lock = threading.Lock()
        self._sqlite: sqlite3.Connection | None = None
        self._pool: Any = None
        if config.kind == SQLITE:
            if sqlite_path is None:
                raise ValueError("a SQLite backend needs the database path")
            self.sqlite_path = Path(sqlite_path)
            conn = sqlite3.connect(
                str(self.sqlite_path), isolation_level=None, check_same_thread=False,
                timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
            )
            conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys = ON")
            self._sqlite = conn
        else:
            self._pool = _open_pool(config.dsn or "")

    def describe(self) -> str:
        return self.config.describe()

    @contextmanager
    def txn(self, *, write: bool) -> Iterator[Conn]:
        if self._sqlite is not None:
            with self._lock:
                raw = self._sqlite
                raw.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                conn = Conn(raw, SQLITE)
                try:
                    yield conn
                except BaseException:
                    if raw.in_transaction:
                        raw.execute("ROLLBACK")
                    raise
                try:
                    raw.execute("COMMIT")
                except BaseException:
                    if raw.in_transaction:
                        raw.execute("ROLLBACK")
                    raise
            return
        with self._pool.connection() as raw:
            conn = Conn(raw, POSTGRES)
            with pg_transaction(conn, write=write):
                yield conn

    def close(self) -> None:
        if self._sqlite is not None:
            with self._lock:
                self._sqlite.close()
        if self._pool is not None:
            self._pool.close()


def _open_pool(dsn: str) -> Any:
    _require_psycopg()
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise StorageConfigError(
            "psycopg_pool is not installed; install the extra: pip install 'kalos[postgres]'"
        ) from exc
    lo, hi = _pool_bounds()
    if hi < lo:
        raise StorageConfigError(f"{POOL_MAX_ENV} ({hi}) is below {POOL_MIN_ENV} ({lo})")
    timeout = _connect_timeout()
    pool = ConnectionPool(
        dsn, min_size=lo, max_size=hi, timeout=float(timeout), open=False,
        kwargs={"autocommit": True, "connect_timeout": timeout},
        check=ConnectionPool.check_connection, name="kalos-storage",
    )
    try:
        pool.open(wait=lo > 0, timeout=float(timeout))
    except Exception as exc:  # noqa: BLE001 - re-raised without the DSN
        pool.close()
        raise StorageConfigError(
            f"could not connect to the PostgreSQL database {redact_dsn(dsn)} "
            f"({type(exc).__name__})"
        ) from None
    return pool


def pg_connect(dsn: str) -> Conn:
    """One autocommit PostgreSQL connection (the migration CLI and tests)."""
    psycopg = _require_psycopg()
    try:
        raw = psycopg.connect(dsn, autocommit=True, connect_timeout=_connect_timeout())
    except psycopg.Error as exc:
        raise StorageConfigError(
            f"could not connect to the PostgreSQL database {redact_dsn(dsn)} "
            f"({type(exc).__name__})"
        ) from None
    return Conn(raw, POSTGRES)


@contextmanager
def pg_transaction(conn: Conn, *, write: bool) -> Iterator[Conn]:
    """One PostgreSQL transaction with the store's semantics: a write takes the
    database-wide advisory lock first (serialized writers); a read is one
    REPEATABLE READ READ ONLY snapshot. Any error rolls back."""
    if write:
        conn.execute("BEGIN ISOLATION LEVEL READ COMMITTED")
        try:
            conn.execute("SELECT pg_advisory_xact_lock(?)", (WRITE_LOCK_KEY,))
        except BaseException:
            _rollback_quietly(conn)
            raise
    else:
        conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
    try:
        yield conn
    except BaseException:
        _rollback_quietly(conn)
        raise
    try:
        conn.execute("COMMIT")
    except BaseException:
        _rollback_quietly(conn)
        raise


def _rollback_quietly(conn: Conn) -> None:
    try:
        conn.execute("ROLLBACK")
    except Exception:  # noqa: BLE001 - the original error is the one to raise
        _log.debug("rollback after a failed transaction also failed", exc_info=True)


def open_backend(state_dir: Path, environ: Mapping[str, str] | None = None) -> Backend:
    """The records backend for this engine: PostgreSQL when `KALOS_DATABASE_URL`
    is set, else SQLite at `<state_dir>/portal.db` (the file `CampaignStore`
    migrates, so the v9 tables already exist there)."""
    cfg = backend_from_env(environ)
    if cfg.is_postgres:
        from kalos.storage.migrations.pg_runner import ensure_schema

        ensure_schema(cfg.dsn or "")
        return Backend(cfg)
    return Backend(cfg, sqlite_path=Path(state_dir) / SQLITE_FILENAME)


__all__ = [
    "DATABASE_URL_ENV", "DB_ERRORS", "INTEGRITY_ERRORS", "POSTGRES", "SQLITE",
    "Backend", "BackendConfig", "Conn", "StorageConfigError", "backend_from_env",
    "check_database_settings", "json_value", "open_backend", "pg_connect",
    "pg_transaction", "redact_dsn", "translate_sql",
]
