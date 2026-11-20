"""Records storage at startup and when the database is down (review S1).

With KALOS_DATABASE_URL set, the engine opens the PostgreSQL backend in its
lifespan: an unreachable server, or a foreign, damaged, older or newer schema,
fails the boot instead of the first request. A failure while serving is cached
for a short, growing backoff (one connection attempt per window), and the cache
lock is never held while connecting.
"""
from __future__ import annotations

import json
import threading
import time

import pytest
import records_helpers
from records_helpers import TOKENS, H

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.api.routers import _records_common as rc  # noqa: E402
from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.storage.backend import StorageConfigError  # noqa: E402
from kalos.storage.migrations import pg_runner  # noqa: E402

pg_schema_dsn = records_helpers.pg_schema_dsn
UNREACHABLE = "postgresql://kalos:s3cr3t-startup@127.0.0.1:1/kalos"


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """Point the app at a fresh store; the test sets KALOS_DATABASE_URL."""
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    monkeypatch.setenv("KALOS_DB_CONNECT_TIMEOUT", "1")
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    rc.reset_backends()
    try:
        yield store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        rc.reset_backends()


def test_sqlite_boot_needs_nothing_extra(boot, monkeypatch):
    monkeypatch.delenv("KALOS_DATABASE_URL", raising=False)
    with TestClient(app) as tc:
        assert tc.get("/api/projects", headers=H("a-read")).status_code == 200


def test_an_unreachable_database_fails_the_boot_without_the_password(boot, monkeypatch):
    pytest.importorskip("psycopg")
    monkeypatch.setenv("KALOS_DATABASE_URL", UNREACHABLE)
    with pytest.raises(StorageConfigError) as exc, TestClient(app):
        pass
    assert "s3cr3t" not in str(exc.value)


def test_a_failure_while_serving_is_cached_with_backoff(boot, monkeypatch):
    monkeypatch.delenv("KALOS_DATABASE_URL", raising=False)
    with TestClient(app) as tc:
        monkeypatch.setenv("KALOS_DATABASE_URL", UNREACHABLE)  # the database "goes away"
        calls = []

        def failing_open(state_dir):
            calls.append(time.monotonic())
            raise StorageConfigError("could not connect to the PostgreSQL database postgresql://kalos@127.0.0.1:1/kalos")

        monkeypatch.setattr(rc, "_open", failing_open)
        first = tc.get("/api/projects", headers=H("a-read"))
        second = tc.get("/api/projects", headers=H("a-read"))
        assert first.status_code == second.status_code == 503
        assert first.json()["error"] == "records storage is unavailable; try again later"
        assert int(first.headers["retry-after"]) >= 1 and "s3cr3t" not in first.text
        assert len(calls) == 1  # the second request did not try to connect again
        failure = rc._FAILURES[rc._key(boot.state_dir)]
        failure.until = 0.0  # the window passes: one more attempt, a longer backoff
        tc.get("/api/projects", headers=H("a-read"))
        assert len(calls) == 2 and rc._FAILURES[rc._key(boot.state_dir)].attempts == 2


def test_the_cache_lock_is_not_held_while_connecting(tmp_path, monkeypatch):
    rc.reset_backends()
    other = tmp_path / "other"
    other.mkdir()
    CampaignStore(other).close()
    monkeypatch.delenv("KALOS_DATABASE_URL", raising=False)
    rc._backends(other)  # cached
    entered, release = threading.Event(), threading.Event()
    real_open = rc._open

    def slow_open(state_dir):
        if state_dir == tmp_path:
            entered.set()
            release.wait(5)
        return real_open(state_dir)

    monkeypatch.setattr(rc, "_open", slow_open)
    CampaignStore(tmp_path).close()
    t = threading.Thread(target=rc._backends, args=(tmp_path,))
    t.start()
    try:
        assert entered.wait(5)
        started = time.monotonic()
        rc._backends(other)  # a cache hit while the other key is still connecting
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        t.join(5)
        rc.reset_backends()


pg = pytest.mark.postgres


@pg
def test_a_foreign_schema_stops_startup(boot, pg_schema_dsn, monkeypatch):
    import psycopg

    with psycopg.connect(pg_schema_dsn, autocommit=True) as c:
        c.execute("CREATE TABLE projects (id text)")  # somebody else's table, no schema_migrations
    monkeypatch.setenv("KALOS_DATABASE_URL", pg_schema_dsn)
    with pytest.raises(pg_runner.PgMigrationError, match="no schema_migrations"), TestClient(app):
        pass


@pg
def test_a_stale_schema_stops_startup(boot, pg_schema_dsn, monkeypatch):
    pg_runner.ensure_schema(pg_schema_dsn)  # v9
    monkeypatch.setattr(pg_runner, "STEPS", (*pg_runner.STEPS, pg_runner.Step(10, "f", lambda c: None, ())))
    monkeypatch.setattr(pg_runner, "LATEST", 10)
    monkeypatch.setenv("KALOS_DATABASE_URL", pg_schema_dsn)
    with pytest.raises(pg_runner.PgMigrationError, match="does not upgrade a server database on startup"), \
            TestClient(app):
        pass


@pg
def test_an_empty_schema_is_created_at_startup(boot, pg_schema_dsn, monkeypatch):
    monkeypatch.setenv("KALOS_DATABASE_URL", pg_schema_dsn)
    with TestClient(app) as tc:
        assert "schema v9 (current" in pg_runner.status(pg_schema_dsn)
        assert tc.get("/api/projects", headers=H("a-read")).status_code == 200
