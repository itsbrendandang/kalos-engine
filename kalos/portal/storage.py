"""Kalos portal - the storage backend: SQLite (default) or PostgreSQL.

`KALOS_DATABASE_URL` picks the backend:

  - unset or empty: SQLite at `<KALOS_STATE_DIR>/portal.db`, as before.
  - `postgresql://...` or `postgres://...`: PostgreSQL through psycopg 3
    (`pip install 'kalos[postgres]'`), imported only on this path.
  - anything else: refused (`StorageConfigError`), never guessed.

The store's SQL is written once, in the SQLite dialect it has always used.
`PgConnection` adapts a statement with parameters for psycopg: `?`
placeholders become `%s`, a literal `%` becomes `%%`, and `IS ?` (the
generation compare-and-swap, which must match NULL) becomes `IS NOT DISTINCT
FROM %s`. A `?` inside quoted text is not a placeholder. A statement without
parameters is sent as written. Nothing else is translated, so a SQLite-only construct in new code
fails loudly on PostgreSQL instead of being silently reinterpreted.

Transactions (docs/MIGRATIONS.md, "PostgreSQL backend"): a write transaction takes one database-wide advisory
lock as its first statement, so writers are serialized exactly as SQLite's
`BEGIN IMMEDIATE` serializes them, across every engine process and job worker
on the same database. A read transaction is REPEATABLE READ READ ONLY: one
snapshot, like a deferred SQLite transaction.

This module is torch-free and FastAPI-free (the migration CLI imports it).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from kalos.envconfig import ConfigError

_log = logging.getLogger("kalos.portal")

DATABASE_URL_ENV = "KALOS_DATABASE_URL"
POOL_MIN_ENV = "KALOS_DB_POOL_MIN"
POOL_MAX_ENV = "KALOS_DB_POOL_MAX"
CONNECT_TIMEOUT_ENV = "KALOS_DB_CONNECT_TIMEOUT"

SQLITE = "sqlite"
POSTGRES = "postgres"

# The key of the database-wide write lock (pg_advisory_xact_lock). An arbitrary
# fixed 64-bit value: "kalos" in ASCII. Every write transaction and every
# migration takes it first.
WRITE_LOCK_KEY = 0x6B616C6F73
# The key of the engine's session-level lock on a PostgreSQL database (the
# counterpart of the state-directory flock, kalos/portal/engine_lock.py).
ENGINE_LOCK_KEY = 0x6B616C6F7345


class StorageConfigError(ConfigError):
    """`KALOS_DATABASE_URL` is set to something this engine refuses to use, or
    the PostgreSQL driver is missing. Raised at startup, before any request."""


@dataclass(frozen=True)
class BackendConfig:
    kind: str  # SQLITE | POSTGRES
    dsn: str | None = None  # PostgreSQL only; never logged unredacted

    @property
    def is_postgres(self) -> bool:
        return self.kind == POSTGRES

    def describe(self) -> str:
        return "sqlite (portal.db in the state directory)" if self.kind == SQLITE else (
            f"postgres ({redact_dsn(self.dsn or '')})"
        )


def backend_from_env(environ: dict[str, str] | None = None) -> BackendConfig:
    """The backend `KALOS_DATABASE_URL` selects. Unset/empty = SQLite."""
    env = os.environ if environ is None else environ
    url = (env.get(DATABASE_URL_ENV) or "").strip()
    if not url:
        return BackendConfig(SQLITE)
    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    if scheme in ("postgresql", "postgres"):
        return BackendConfig(POSTGRES, url)
    raise StorageConfigError(
        f"{DATABASE_URL_ENV} must be empty (SQLite in the state directory) or a "
        f"postgresql:// URL; got a value with scheme {scheme or '(none)'!r}"
    )


def redact_dsn(dsn: str) -> str:
    """The DSN with any password replaced by `***` (URL userinfo and a
    `password=` query or keyword parameter). Safe to log."""
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return "<unparseable database url>"
    netloc = parts.netloc
    if "@" in netloc:
        userinfo, host = netloc.rsplit("@", 1)
        if ":" in userinfo:
            userinfo = userinfo.split(":", 1)[0] + ":***"
        netloc = f"{userinfo}@{host}"
    query = re.sub(r"(?i)(^|&)(password)=[^&]*", r"\1\2=***", parts.query)
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def check_database_settings() -> BackendConfig:
    """Validate KALOS_DATABASE_URL and, on PostgreSQL, the pool settings and the
    driver, before the engine binds a socket (`check_env_settings`)."""
    cfg = backend_from_env()
    if cfg.is_postgres:
        _require_psycopg()
        lo = _int_env(POOL_MIN_ENV, 1, minimum=0)
        hi = _int_env(POOL_MAX_ENV, 10, minimum=1)
        if hi < lo:
            raise StorageConfigError(f"{POOL_MAX_ENV} ({hi}) is below {POOL_MIN_ENV} ({lo})")
        _int_env(CONNECT_TIMEOUT_ENV, 10, minimum=1)
    return cfg


# --- SQL translation (SQLite dialect -> psycopg) ------------------------------ #

_IS_PARAM = re.compile(r"\bIS\s+\?", re.IGNORECASE)


def translate_sql(sql: str) -> str:
    """Rewrite the store's SQLite-dialect SQL for psycopg, for a statement that
    HAS parameters. Every `%` becomes `%%` (psycopg scans the whole string,
    quoted text included). Outside quoted text ('...' and "..."), `IS ?`
    becomes `IS NOT DISTINCT FROM %s` and `?` becomes `%s`; a `?` inside quotes
    is left as it is. A statement without parameters is sent unchanged (psycopg
    does no placeholder processing then), see `PgConnection.execute`."""
    out: list[str] = []
    i, n = 0, len(sql)
    seg_start = 0

    def flush(end: int) -> None:
        seg = sql[seg_start:end].replace("%", "%%")
        seg = _IS_PARAM.sub("IS NOT DISTINCT FROM ?", seg)
        out.append(seg.replace("?", "%s"))

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


class PgConnection:
    """The subset of `sqlite3.Connection` the store uses, over a psycopg 3
    connection, with the SQL translated (see the module docstring)."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = self.raw.cursor()
        if params:
            cur.execute(_translated(sql), tuple(params))
        else:
            cur.execute(sql)
        return cur

    def executemany(self, sql: str, seq: Sequence[Sequence[Any]]) -> Any:
        cur = self.raw.cursor()
        cur.executemany(_translated(sql), [tuple(p) for p in seq])
        return cur

    @property
    def in_transaction(self) -> bool:
        from psycopg.pq import TransactionStatus

        return self.raw.info.transaction_status != TransactionStatus.IDLE


# --- errors ----------------------------------------------------------------- #


def _pg_errors() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    try:
        import psycopg
    except ImportError:
        return (), ()
    return (psycopg.Error,), (psycopg.IntegrityError,)


_PG_DB, _PG_INTEGRITY = _pg_errors()
# For `except` sites that must catch a database error from either backend.
DB_ERRORS: tuple[type[BaseException], ...] = (sqlite3.Error, *_PG_DB)
INTEGRITY_ERRORS: tuple[type[BaseException], ...] = (sqlite3.IntegrityError, *_PG_INTEGRITY)


# --- PostgreSQL connections and pool ------------------------------------------ #


def _require_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise StorageConfigError(
            f"{DATABASE_URL_ENV} selects PostgreSQL but psycopg 3 is not installed; "
            "install the extra: pip install 'kalos[postgres]'"
        ) from exc
    return psycopg


def _int_env(name: str, default: int, *, minimum: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise StorageConfigError(f"{name} must be an integer; got {raw!r}") from None
    if value < minimum:
        raise StorageConfigError(f"{name} must be at least {minimum}; got {value}")
    return value


def pg_connect(dsn: str) -> Any:
    """One autocommit psycopg connection (transactions are opened explicitly,
    like the SQLite path's `isolation_level=None`)."""
    psycopg = _require_psycopg()
    timeout = _int_env(CONNECT_TIMEOUT_ENV, 10, minimum=1)
    return psycopg.connect(dsn, autocommit=True, connect_timeout=timeout)


class PgPool:
    """A psycopg_pool.ConnectionPool of autocommit connections, sized by
    KALOS_DB_POOL_MIN / KALOS_DB_POOL_MAX (defaults 1 / 10)."""

    def __init__(self, dsn: str) -> None:
        _require_psycopg()
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise StorageConfigError(
                "psycopg_pool is not installed; install the extra: pip install 'kalos[postgres]'"
            ) from exc
        lo = _int_env(POOL_MIN_ENV, 1, minimum=0)
        hi = _int_env(POOL_MAX_ENV, 10, minimum=1)
        if hi < lo:
            raise StorageConfigError(f"{POOL_MAX_ENV} ({hi}) is below {POOL_MIN_ENV} ({lo})")
        timeout = _int_env(CONNECT_TIMEOUT_ENV, 10, minimum=1)
        self._pool = ConnectionPool(
            dsn, min_size=lo, max_size=hi, timeout=float(timeout), open=False,
            kwargs={"autocommit": True, "connect_timeout": timeout},
            check=ConnectionPool.check_connection,
            name="kalos-portal",
        )
        # wait=True: fail at startup (StorageConfigError-like PoolTimeout) when
        # the server is unreachable, rather than on the first request.
        self._pool.open(wait=lo > 0, timeout=float(timeout))

    @contextmanager
    def connection(self) -> Iterator[PgConnection]:
        with self._pool.connection() as raw:
            yield PgConnection(raw)

    def close(self) -> None:
        self._pool.close()


class EngineDatabaseLockHeld(RuntimeError):
    """Another engine process holds the PostgreSQL engine lock."""


def acquire_pg_engine_lock(dsn: str) -> Any:
    """One engine per PostgreSQL database (schema): take a session-level
    advisory lock on a dedicated connection and return that connection. The
    lock lives as long as the connection; the server releases it when the
    process dies. A second engine refuses to start, as with the state-directory
    flock: two engines would each recover the other's running jobs at startup."""
    raw = pg_connect(dsn)
    try:
        got = raw.execute("SELECT pg_try_advisory_lock(%s)", (ENGINE_LOCK_KEY,)).fetchone()[0]
    except BaseException:
        raw.close()
        raise
    if not got:
        raw.close()
        raise EngineDatabaseLockHeld(
            f"another kalos engine already holds the database {redact_dsn(dsn)}; "
            "run one engine per database (see docs/HARDENING.md, \"One engine per state directory\")"
        )
    return raw


@contextmanager
def pg_transaction(conn: PgConnection, *, write: bool) -> Iterator[PgConnection]:
    """One PostgreSQL transaction with the store's semantics: a write takes the
    database-wide advisory lock first (serialized writers, READ COMMITTED, so
    every statement after the lock sees every earlier commit); a read is one
    REPEATABLE READ READ ONLY snapshot. Any error rolls back; a failed COMMIT
    rolls back and re-raises."""
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


def _rollback_quietly(conn: PgConnection) -> None:
    try:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
    except Exception:  # noqa: BLE001 - the original error is the one to raise
        _log.debug("rollback after a failed transaction also failed", exc_info=True)


# What the store's in-transaction helpers receive: either driver's connection.
DbConnection = sqlite3.Connection | PgConnection
