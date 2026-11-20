"""The storage backend switch (kalos/storage/backend.py) and the PostgreSQL schema
runner (kalos/storage/migrations/pg_runner.py).

The configuration and SQL-translation tests always run. The `postgres` tests run
only when KALOS_TEST_DATABASE_URL points at a PostgreSQL server (CI: the postgres
job's service container); each uses a fresh schema.
"""
from __future__ import annotations

import pytest
import records_helpers

from kalos.envconfig import ConfigError
from kalos.storage import backend as sb
from kalos.storage.migrations import pg_runner, v9

SECRET = "s3cr3t-Pa55"
pg_schema_dsn = records_helpers.pg_schema_dsn  # the shared fixture

# --- configuration (no server needed) ----------------------------------------------- #


def test_sqlite_is_the_default():
    assert sb.backend_from_env({}).kind == sb.SQLITE
    assert sb.backend_from_env({"KALOS_DATABASE_URL": "   "}).kind == sb.SQLITE


@pytest.mark.parametrize("url", ["postgresql://u@h/db", "postgres://u@h/db", "POSTGRESQL://u@h/db"])
def test_postgres_urls_select_postgres(url):
    assert sb.backend_from_env({"KALOS_DATABASE_URL": url}).is_postgres


@pytest.mark.parametrize("url", ["sqlite:///x.db", "mysql://u:p@h/db", "/var/lib/kalos.db", "file.db"])
def test_other_urls_are_refused_not_guessed(url):
    with pytest.raises(sb.StorageConfigError) as exc:
        sb.backend_from_env({"KALOS_DATABASE_URL": url})
    assert isinstance(exc.value, ConfigError)


@pytest.mark.parametrize(("dsn", "shown"), [
    (f"postgresql://kalos:{SECRET}@db.internal:5432/kalos", "postgresql://kalos:***@db.internal:5432/kalos"),
    (f"postgresql://db/kalos?user=k&password={SECRET}&sslmode=require",
     "postgresql://db/kalos?user=k&password=***&sslmode=require"),
    ("postgresql://kalos@db/kalos", "postgresql://kalos@db/kalos"),
])
def test_redact_dsn_never_shows_the_password(dsn, shown):
    assert sb.redact_dsn(dsn) == shown
    assert SECRET not in sb.BackendConfig(sb.POSTGRES, dsn).describe()


@pytest.mark.parametrize(("sql", "expected"), [
    ("SELECT * FROM t WHERE a = ? AND b = ?", "SELECT * FROM t WHERE a = %s AND b = %s"),
    ("SELECT '?' , \"a?\" FROM t WHERE x = ?", "SELECT '?' , \"a?\" FROM t WHERE x = %s"),
    ("SELECT 'it''s ?' WHERE y LIKE '%a' AND z = ?", "SELECT 'it''s ?' WHERE y LIKE '%%a' AND z = %s"),
    ("SELECT 5 % 2 WHERE q = ?", "SELECT 5 %% 2 WHERE q = %s"),
])
def test_translate_sql(sql, expected):
    assert sb.translate_sql(sql) == expected


@pytest.mark.parametrize(("env", "dsn", "ok"), [
    ("production", f"postgresql://k:{SECRET}@db.example.com/kalos", False),
    ("production", f"postgresql://k:{SECRET}@db.example.com/kalos?sslmode=require", True),
    ("production", f"postgresql://k:{SECRET}@db.example.com/kalos?sslmode=prefer", False),
    ("production", "postgresql://k@127.0.0.1/kalos", True),
    ("production", "postgresql:///kalos?host=/var/run/postgresql", True),
    ("development", f"postgresql://k:{SECRET}@db.example.com/kalos", True),
])
def test_production_profile_requires_tls_off_box(monkeypatch, env, dsn, ok):
    pytest.importorskip("psycopg")
    monkeypatch.setenv("KALOS_ENV", env)
    environ = {"KALOS_DATABASE_URL": dsn}
    if ok:
        assert sb.check_database_settings(environ).is_postgres
    else:
        with pytest.raises(sb.StorageConfigError) as exc:
            sb.check_database_settings(environ)
        assert SECRET not in str(exc.value) and "sslmode=require" in str(exc.value)


@pytest.mark.parametrize(("name", "value"), [
    ("KALOS_DB_POOL_MAX", "0"), ("KALOS_DB_POOL_MIN", "-1"), ("KALOS_DB_POOL_MAX", "ten"),
    ("KALOS_DB_CONNECT_TIMEOUT", "0"),
])
def test_pool_settings_are_checked_at_startup(monkeypatch, name, value):
    pytest.importorskip("psycopg")
    monkeypatch.setenv(name, value)
    with pytest.raises(sb.StorageConfigError):
        sb.check_database_settings({"KALOS_DATABASE_URL": "postgresql://k@127.0.0.1/kalos"})


def test_startup_check_reads_the_storage_settings(monkeypatch):
    from kalos.portal.config import check_env_settings

    monkeypatch.setenv("KALOS_DATABASE_URL", "mysql://x@y/z")
    with pytest.raises(ConfigError):
        check_env_settings()
    monkeypatch.delenv("KALOS_DATABASE_URL")
    monkeypatch.setenv("KALOS_LOG_FORMAT", "yaml")
    with pytest.raises(ConfigError):
        check_env_settings()


def test_an_unreachable_server_is_refused_without_the_password(monkeypatch):
    pytest.importorskip("psycopg")
    monkeypatch.setenv("KALOS_DB_CONNECT_TIMEOUT", "1")
    with pytest.raises(sb.StorageConfigError) as exc:
        sb.pg_connect(f"postgresql://kalos:{SECRET}@127.0.0.1:1/kalos")
    assert SECRET not in str(exc.value) and "***" in str(exc.value)


def test_sqlite_backend_dialect_neutral_round_trip(tmp_path):
    b = sb.Backend(sb.BackendConfig(sb.SQLITE), sqlite_path=tmp_path / "x.db")
    try:
        with b.txn(write=True) as conn:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, doc TEXT)")
            rid = conn.execute("INSERT INTO t (doc) VALUES (?) RETURNING id",
                               (conn.json_param({"a": [1, 2]}),)).fetchone()[0]
        with b.txn(write=False) as conn:
            (doc,) = conn.execute("SELECT doc FROM t WHERE id = ?", (rid,)).fetchone()
        assert sb.json_value(doc) == {"a": [1, 2]}
        with pytest.raises(ValueError), b.txn(write=True) as conn:
            conn.json_param({"x": float("nan")})
    finally:
        b.close()


# --- PostgreSQL (needs KALOS_TEST_DATABASE_URL) ---------------------------------------- #

pg = pytest.mark.postgres


def _tables(dsn: str) -> set[str]:
    conn = sb.pg_connect(dsn)
    try:
        return pg_runner._tables(conn)
    finally:
        conn.raw.close()


@pg
def test_empty_database_is_created_at_v9_in_one_step(pg_schema_dsn):
    res = pg_runner.ensure_schema(pg_schema_dsn)
    assert (res.action, res.to_version) == ("created", 9)
    assert _tables(pg_schema_dsn) >= set(pg_runner.BASELINE_TABLES)
    conn = sb.pg_connect(pg_schema_dsn)
    try:
        assert pg_runner._triggers(conn) >= set(v9.PG_TRIGGERS) | set(pg_runner.AUDIT_TRIGGERS)
        assert conn.execute("SELECT version, name FROM schema_migrations").fetchall() == [
            (9, pg_runner.BASELINE_NAME)]
    finally:
        conn.raw.close()
    assert pg_runner.ensure_schema(pg_schema_dsn).action == "noop"
    assert "schema v9 (current" in pg_runner.status(pg_schema_dsn)


@pg
@pytest.mark.parametrize("table", ["audit_log", "lineage_links", "lineage_retractions"])
def test_append_only_on_postgres_rejects_update_delete_truncate(pg_schema_dsn, table):
    import psycopg

    pg_runner.ensure_schema(pg_schema_dsn)
    b = sb.Backend(sb.BackendConfig(sb.POSTGRES, pg_schema_dsn))
    try:
        with b.txn(write=True) as conn:
            if table == "audit_log":
                conn.execute("INSERT INTO audit_log (ts, tenant, action, outcome) VALUES (1, 'acme', 'a', 'ok')")
            else:
                conn.execute("INSERT INTO lineage_links (tenant, parent_campaign_id, child_campaign_id, relation, "
                             "created_at) VALUES ('acme', 'cmp_a', 'cmp_b', 'scale_up', 1)")
                if table == "lineage_retractions":
                    (lid,) = conn.execute("SELECT id FROM lineage_links").fetchone()
                    conn.execute("INSERT INTO lineage_retractions (tenant, link_id, reason, created_at) "
                                 "VALUES ('acme', ?, 'r', 1)", (lid,))
        for sql in (f"UPDATE {table} SET tenant = 'x'", f"DELETE FROM {table}", f"TRUNCATE {table} CASCADE"):
            with pytest.raises(psycopg.IntegrityError, match="append-only"), b.txn(write=True) as conn:
                conn.execute(sql)
    finally:
        b.close()


@pg
def test_newer_and_foreign_and_damaged_databases_are_refused(pg_schema_dsn):
    import psycopg

    raw = psycopg.connect(pg_schema_dsn, autocommit=True)
    try:
        raw.execute("CREATE TABLE audit_log (id int)")
        with pytest.raises(pg_runner.PgMigrationError, match="no schema_migrations"):
            pg_runner.ensure_schema(pg_schema_dsn)
        raw.execute("DROP TABLE audit_log")
        pg_runner.ensure_schema(pg_schema_dsn)
        raw.execute("DROP TRIGGER lineage_links_no_delete ON lineage_links")
        with pytest.raises(pg_runner.PgMigrationError, match="lineage_links_no_delete"):
            pg_runner.ensure_schema(pg_schema_dsn)
        raw.execute("INSERT INTO schema_migrations VALUES (10, 'future', 0)")
        with pytest.raises(pg_runner.PgMigrationError, match="newer than this code"):
            pg_runner.ensure_schema(pg_schema_dsn)
    finally:
        raw.close()


@pg
def test_an_older_database_is_not_upgraded_on_startup(pg_schema_dsn, monkeypatch):
    """A future step appended to STEPS: startup refuses, the CLI upgrades after
    a backup (here: --confirm-backup), in one transaction."""
    pg_runner.ensure_schema(pg_schema_dsn)
    created: list[str] = []

    def step10(conn):
        conn.execute("CREATE TABLE future_table (id int)")
        created.append("future_table")

    monkeypatch.setattr(pg_runner, "STEPS", (*pg_runner.STEPS, pg_runner.Step(10, "future", step10, ("future_table",))))
    monkeypatch.setattr(pg_runner, "LATEST", 10)
    with pytest.raises(pg_runner.PgMigrationError, match="does not upgrade a server database on startup"):
        pg_runner.ensure_schema(pg_schema_dsn)
    assert pg_runner.migrate_up(pg_schema_dsn, dry_run=True).action == "dry-run"
    assert created == []
    res = pg_runner.migrate_up(pg_schema_dsn, confirm_backup=True)
    assert (res.from_version, res.to_version) == (9, 10) and "future_table" in _tables(pg_schema_dsn)


@pg
def test_a_failing_step_leaves_nothing_behind(pg_schema_dsn, monkeypatch):
    def broken(conn):
        conn.execute("CREATE TABLE half_done (id int)")
        raise RuntimeError("step failed midway")

    monkeypatch.setattr(pg_runner, "STEPS", (pg_runner.Step(9, "broken", broken, ("half_done",)),))
    monkeypatch.setattr(pg_runner, "LATEST", 9)
    with pytest.raises(RuntimeError):
        pg_runner.ensure_schema(pg_schema_dsn)
    assert _tables(pg_schema_dsn) == set()  # one transaction: nothing was created


@pg
def test_upgrade_needs_a_verified_backup(pg_schema_dsn, monkeypatch, tmp_path):
    pg_runner.ensure_schema(pg_schema_dsn)
    monkeypatch.setattr(pg_runner, "STEPS", (*pg_runner.STEPS, pg_runner.Step(10, "f", lambda c: None, ())))
    monkeypatch.setattr(pg_runner, "LATEST", 10)
    monkeypatch.setattr(pg_runner.shutil, "which", lambda name: None)
    with pytest.raises(pg_runner.PgMigrationError, match="--confirm-backup"):
        pg_runner.migrate_up(pg_schema_dsn, backup_dir=tmp_path)
    assert "schema v9" in pg_runner.status(pg_schema_dsn)


@pg
def test_verified_pg_dump_backup(pg_schema_dsn, tmp_path):
    import shutil

    if not (shutil.which("pg_dump") and shutil.which("pg_restore")):
        pytest.skip("pg_dump/pg_restore not on PATH")
    pg_runner.ensure_schema(pg_schema_dsn)
    path = pg_runner.take_verified_backup(
        pg_schema_dsn, tmp_path, label="pre-v10", expect_tables=pg_runner.BASELINE_TABLES)
    assert path.exists() and path.stat().st_size > 0 and oct(path.stat().st_mode)[-3:] == "600"
    with pytest.raises(pg_runner.PgMigrationError, match="could not be verified"):
        pg_runner.take_verified_backup(pg_schema_dsn, tmp_path, label="x", expect_tables=("not_a_table",))


@pg
def test_cli_reads_the_url_from_the_environment_only(pg_schema_dsn, monkeypatch, capsys):
    monkeypatch.setenv("KALOS_DATABASE_URL", pg_schema_dsn)
    assert pg_runner.main(["up"]) == 0
    assert pg_runner.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "schema v9" in out
    from urllib.parse import urlsplit

    password = urlsplit(pg_schema_dsn).password
    if password:
        assert password not in out
    monkeypatch.delenv("KALOS_DATABASE_URL")
    assert pg_runner.main(["status"]) == 2
