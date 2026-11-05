"""The PostgreSQL storage backend (kalos/portal/storage.py, kalos/portal/migrate_pg.py).

Two layers:

- Unit tests (always run): SQL translation, DSN parsing and redaction, the
  backend switch, setting validation. No server needed.
- Server tests: run against a REAL PostgreSQL when `KALOS_TEST_DATABASE_URL` is
  set (for example `postgresql://postgres@/kalos_test?host=/tmp/pg&port=5499`);
  otherwise they SKIP with that reason. Each test gets a fresh schema, selected
  through `search_path`, and drops it afterwards.
"""
from __future__ import annotations

import os
import threading
import uuid
from urllib.parse import quote

import pandas as pd
import pytest

from kalos.portal import storage
from kalos.portal.migrate import SCHEMA_VERSION, MigrationError, SchemaError
from kalos.portal.storage import (
    POSTGRES,
    SQLITE,
    StorageConfigError,
    backend_from_env,
    redact_dsn,
    translate_sql,
)

# --- unit: translation -------------------------------------------------------- #


def test_placeholders_become_psycopg_style():
    assert translate_sql("SELECT a FROM t WHERE x = ? AND y = ?") == (
        "SELECT a FROM t WHERE x = %s AND y = %s"
    )


def test_is_param_becomes_null_safe_comparison():
    sql = "UPDATE c SET s = ? WHERE id = ? AND generation IS ?"
    assert translate_sql(sql).endswith("generation IS NOT DISTINCT FROM %s")


def test_is_null_is_left_alone():
    assert translate_sql("SELECT 1 FROM t WHERE a IS NULL AND b = ?") == (
        "SELECT 1 FROM t WHERE a IS NULL AND b = %s"
    )


def test_question_marks_in_quotes_are_not_placeholders_and_percent_is_escaped():
    sql = "SELECT '?' || \"a?b\" FROM t WHERE n LIKE 'x%' AND m = ? AND k % 2 = 0"
    assert translate_sql(sql) == (
        "SELECT '?' || \"a?b\" FROM t WHERE n LIKE 'x%%' AND m = %s AND k %% 2 = 0"
    )


def test_doubled_quotes_inside_literals():
    assert translate_sql("SELECT 'it''s ?' , ?") == "SELECT 'it''s ?' , %s"


def test_upsert_and_limit_pass_through():
    sql = ("INSERT INTO d (tenant, campaign_id) VALUES (?, ?) ON CONFLICT(tenant) DO UPDATE "
           "SET campaign_id = excluded.campaign_id LIMIT ?")
    assert translate_sql(sql).count("%s") == 3


# --- unit: configuration ------------------------------------------------------- #


def test_unset_or_empty_url_is_sqlite():
    assert backend_from_env({}).kind == SQLITE
    assert backend_from_env({"KALOS_DATABASE_URL": "  "}).kind == SQLITE


@pytest.mark.parametrize("url", ["postgresql://u@h/db", "postgres://u:p@h:5432/db", "POSTGRESQL://h/db"])
def test_postgres_urls_select_postgres(url):
    cfg = backend_from_env({"KALOS_DATABASE_URL": url})
    assert cfg.kind == POSTGRES and cfg.dsn == url


@pytest.mark.parametrize("url", ["sqlite:///x.db", "mysql://h/db", "/tmp/portal.db", "host=h dbname=x"])
def test_other_values_are_refused(url):
    with pytest.raises(StorageConfigError):
        backend_from_env({"KALOS_DATABASE_URL": url})


def test_redaction_hides_passwords():
    assert redact_dsn("postgresql://kalos:s3cret@db:5432/kalos") == "postgresql://kalos:***@db:5432/kalos"
    assert "s3cret" not in redact_dsn("postgresql://db/kalos?password=s3cret&sslmode=require")
    assert redact_dsn("postgresql://db/kalos") == "postgresql://db/kalos"


def test_pool_settings_are_validated(monkeypatch):
    monkeypatch.setenv("KALOS_DATABASE_URL", "postgresql://h/db")
    monkeypatch.setenv("KALOS_DB_POOL_MAX", "0")
    with pytest.raises(StorageConfigError):
        storage.check_database_settings()
    monkeypatch.setenv("KALOS_DB_POOL_MAX", "2")
    monkeypatch.setenv("KALOS_DB_POOL_MIN", "3")
    with pytest.raises(StorageConfigError):
        storage.check_database_settings()
    monkeypatch.setenv("KALOS_DB_POOL_MIN", "x")
    with pytest.raises(StorageConfigError):
        storage.check_database_settings()


def test_check_env_settings_refuses_a_bad_database_url(monkeypatch):
    from kalos.envconfig import ConfigError
    from kalos.portal.config import check_env_settings

    monkeypatch.setenv("KALOS_DATABASE_URL", "mysql://h/db")
    with pytest.raises(ConfigError):
        check_env_settings()


def test_sqlite_store_is_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KALOS_DATABASE_URL", raising=False)
    from kalos.portal.campaign import CampaignStore

    s = CampaignStore(tmp_path)
    assert s.backend == SQLITE
    assert (tmp_path / "portal.db").exists()
    s.close()


# --- server tests ------------------------------------------------------------- #

BASE_DSN = os.environ.get("KALOS_TEST_DATABASE_URL", "").strip()
needs_pg = pytest.mark.skipif(
    not BASE_DSN,
    reason="KALOS_TEST_DATABASE_URL is not set: no PostgreSQL server for this run",
)


def _with_search_path(dsn: str, schema: str) -> str:
    opt = quote(f"-csearch_path={schema}", safe="")
    return f"{dsn}{'&' if '?' in dsn else '?'}options={opt}"


@pytest.fixture
def pg_dsn(monkeypatch):
    if not BASE_DSN:
        pytest.skip("KALOS_TEST_DATABASE_URL is not set")
    import psycopg

    schema = "kt_" + uuid.uuid4().hex[:16]
    with psycopg.connect(BASE_DSN, autocommit=True) as c:
        c.execute(f'CREATE SCHEMA "{schema}"')
    dsn = _with_search_path(BASE_DSN, schema)
    monkeypatch.setenv("KALOS_DATABASE_URL", dsn)
    monkeypatch.setenv("KALOS_DB_POOL_MAX", "8")
    yield dsn
    with psycopg.connect(BASE_DSN, autocommit=True) as c:
        c.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def pg_store(pg_dsn, tmp_path):
    from kalos.portal.campaign import CampaignStore

    s = CampaignStore(tmp_path)
    assert s.backend == POSTGRES
    yield s
    s.close()


def _raw(dsn: str):
    import psycopg

    return psycopg.connect(dsn, autocommit=True)


@needs_pg
def test_empty_database_is_created_at_the_current_version(pg_dsn):
    from kalos.portal import migrate_pg

    res = migrate_pg.migrate_up(pg_dsn)
    assert res.action == "created"
    assert migrate_pg.migrate_up(pg_dsn).action == "noop"
    report, healthy = migrate_pg.status_report(pg_dsn)
    assert healthy and "shape check: ok" in report
    with _raw(pg_dsn) as c:
        versions = [r[0] for r in c.execute("SELECT version FROM schema_migrations ORDER BY 1")]
        trig = {r[0] for r in c.execute(
            "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal")}
    assert versions == [2, 3, 4, 5]
    assert {"audit_log_no_update", "audit_log_no_delete", "audit_log_no_truncate",
            "run_results_no_update", "run_results_no_delete", "run_results_no_truncate"} <= trig


@needs_pg
@pytest.mark.parametrize("table", ["audit_log", "run_results"])
def test_append_only_tables_reject_update_delete_truncate(pg_store, pg_dsn, table):
    import psycopg

    cid = pg_store.create_campaign(tenant="acme", name="c")["id"]
    pg_store.record_audit(tenant="acme", action="campaign.create", outcome="ok", campaign_id=cid)
    with _raw(pg_dsn) as c:
        if table == "run_results":
            c.execute(
                "INSERT INTO run_results (tenant, campaign_id, run_id, version, value, recorded_at) "
                "VALUES ('acme', %s, 'r1', 1, 1.5, 1.0)", (cid,))
        before = c.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
        for stmt in (f"UPDATE {table} SET tenant = 'evil'", f"DELETE FROM {table}",
                     f"TRUNCATE {table}"):
            with pytest.raises(psycopg.IntegrityError, match="append-only"):
                c.execute(stmt)
        assert c.execute(f"SELECT * FROM {table} ORDER BY id").fetchall() == before


@needs_pg
def test_store_loop_end_to_end(pg_store):
    df = pd.DataFrame({"x": [0.1, 0.5, 0.9], "y": [1.0, 2.0, 3.0]})
    cid = pg_store.create_campaign(tenant="acme", name="Loop", make_default=True)["id"]
    gen = pg_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid,
                        analysis={"ok": True}, dataset="d.csv")
    assert pg_store.load_analysis(tenant="acme", campaign_id=cid)["campaign_generation"] == gen
    runs = pg_store.start([{"recipe": {"x": 0.3}}], tenant="acme", campaign_id=cid)
    rid = runs[0]["id"]
    pg_store.set_result(rid, 4.0, tenant="acme", campaign_id=cid, actor="alice")
    from kalos.portal.campaign import CampaignReasonRequired

    with pytest.raises(CampaignReasonRequired):
        pg_store.set_result(rid, 5.0, tenant="acme", campaign_id=cid)
    pg_store.set_result(rid, 5.0, tenant="acme", campaign_id=cid, reason="typo", actor="bob")
    hist = pg_store.result_history(rid, tenant="acme", campaign_id=cid)
    assert [(h["version"], h["value"], h["previous_value"]) for h in hist] == [
        (1, 4.0, None), (2, 5.0, 4.0)]
    _df, _target, planned = pg_store.plan_fold(tenant="acme", campaign_id=cid)
    state = pg_store.commit_fold(planned, tenant="acme", campaign_id=cid, analysis={"r": 1})
    assert state["round"] == 1 and len(state["base_rows"]) == 4
    summary = pg_store.summary(tenant="acme", campaign_id=cid)
    assert summary["best"] == 5.0 and summary["n_base"] == 4


@needs_pg
def test_null_generation_compare_and_swap_and_conflict(pg_store):
    from kalos.portal.campaign import CampaignConflict

    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    cid = pg_store.create_campaign(tenant="acme", name="c")["id"]
    # expected_generation=None must match the NULL generation of a fresh campaign
    gen = pg_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid, expected_generation=None)
    with pytest.raises(CampaignConflict):
        pg_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid, expected_generation=None)
    pg_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid, expected_generation=gen)


@needs_pg
def test_tenant_isolation_and_archive(pg_store):
    from kalos.portal.campaign import CampaignArchived, CampaignNotFound

    cid = pg_store.create_campaign(tenant="acme", name="c")["id"]
    with pytest.raises(CampaignNotFound):
        pg_store.campaign_info(tenant="globex", campaign_id=cid)
    assert pg_store.list_campaigns(tenant="globex") == []
    pg_store.archive_campaign(tenant="acme", campaign_id=cid)
    with pytest.raises(CampaignArchived):
        pg_store.seed(pd.DataFrame({"x": [1.0], "y": [1.0]}), "y", ["x"], tenant="acme",
                      campaign_id=cid)


@needs_pg
def test_a_failed_audit_write_rolls_the_result_back(pg_store, monkeypatch):
    df = pd.DataFrame({"x": [0.1, 0.2], "y": [1.0, 2.0]})
    cid = pg_store.create_campaign(tenant="acme", name="c")["id"]
    pg_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid)
    rid = pg_store.start([{"recipe": {"x": 0.3}}], tenant="acme", campaign_id=cid)[0]["id"]
    before = pg_store.get(tenant="acme", campaign_id=cid)

    def boom(*_a, **_k):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(type(pg_store), "_insert_audit", staticmethod(boom))
    with pytest.raises(RuntimeError):
        pg_store.set_result(rid, 3.0, tenant="acme", campaign_id=cid, audit=True)
    assert pg_store.get(tenant="acme", campaign_id=cid) == before
    assert pg_store.result_history(rid, tenant="acme", campaign_id=cid) == []


@needs_pg
def test_audit_row_is_in_the_same_transaction(pg_store):
    df = pd.DataFrame({"x": [0.1], "y": [1.0]})
    cid = pg_store.create_campaign(tenant="acme", name="c")["id"]
    pg_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid)
    rid = pg_store.start([{"recipe": {"x": 0.3}}], tenant="acme", campaign_id=cid)[0]["id"]
    pg_store.set_result(rid, 3.0, tenant="acme", campaign_id=cid, audit=True, actor="alice")
    rows = pg_store.audit_entries(tenant="acme")
    assert rows[0]["action"] == "campaign.result" and '"new_value": 3.0' in rows[0]["detail"]
    assert pg_store.audit_entries(tenant="globex") == []


@needs_pg
def test_read_transactions_refuse_writes(pg_store):
    import psycopg

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction), pg_store._txn(write=False) as conn:
        conn.execute("INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                     (99, "x", 0.0))


@needs_pg
def test_concurrent_writers_from_two_stores_lose_nothing(pg_dsn, tmp_path):
    """Two stores (two pools: stands in for two processes) and several threads
    each start runs on ONE campaign. Serialized writers mean every start reads
    the latest state, so no write is lost and no CAS conflict is raised."""
    from kalos.portal.campaign import CampaignStore

    a = CampaignStore(tmp_path / "a")
    b = CampaignStore(tmp_path / "b")
    try:
        cid = a.create_campaign(tenant="acme", name="c")["id"]
        a.seed(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}), "y", ["x"],
               tenant="acme", campaign_id=cid)
        errors: list[BaseException] = []

        def work(store, k):
            try:
                for i in range(5):
                    store.start([{"recipe": {"x": (k * 10 + i) / 100}}], tenant="acme",
                                campaign_id=cid)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(s, k))
                   for k, s in enumerate([a, b, a, b, a, b])]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(a.get(tenant="acme", campaign_id=cid)["pending"]) == 30
    finally:
        a.close()
        b.close()


@needs_pg
def test_engine_starts_refusing_an_older_schema_and_cli_upgrades_it(pg_dsn, tmp_path, capsys):
    from kalos.portal import migrate, migrate_pg
    from kalos.portal.campaign import CampaignStore

    migrate_pg.migrate_up(pg_dsn)
    with pytest.raises(MigrationError):
        migrate_pg.migrate_down(pg_dsn)  # no --confirm-backup
    res = migrate_pg.migrate_down(pg_dsn, target=3, confirm_backup=True)
    assert "run_results_v5_rolledback" in res.plan_text
    with pytest.raises(SchemaError, match="pg_dump"):
        CampaignStore(tmp_path)
    assert migrate.main(["status"]) == 2
    assert migrate.main(["up"]) == 0
    out = capsys.readouterr().out
    assert "result: migrated" in out
    assert migrate.main(["status"]) == 0
    with pytest.raises(MigrationError):
        migrate_pg.migrate_down(pg_dsn, target=1, confirm_backup=True)


@needs_pg
def test_a_disabled_trigger_fails_the_shape_check(pg_store, pg_dsn, tmp_path):
    from kalos.portal import migrate_pg
    from kalos.portal.campaign import CampaignStore

    with _raw(pg_dsn) as c:
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_delete")
    report, healthy = migrate_pg.status_report(pg_dsn)
    assert not healthy and "audit_log_no_delete" in report
    with pytest.raises(SchemaError):
        CampaignStore(tmp_path / "other")


@needs_pg
def test_tables_without_a_version_row_are_refused(pg_dsn, tmp_path):
    from kalos.portal.campaign import CampaignStore

    with _raw(pg_dsn) as c:
        c.execute("CREATE TABLE campaigns (id TEXT)")
    with pytest.raises(SchemaError, match="no schema_migrations"):
        CampaignStore(tmp_path)


@needs_pg
def test_cli_never_prints_the_password(pg_dsn, monkeypatch, capsys):
    from kalos.portal import migrate

    bad = "postgresql://kalos:hunter2@127.0.0.1:1/nope?connect_timeout=1"
    monkeypatch.setenv("KALOS_DATABASE_URL", bad)
    monkeypatch.setenv("KALOS_DB_CONNECT_TIMEOUT", "1")
    assert migrate.main(["status"]) == 1
    captured = capsys.readouterr()
    assert "hunter2" not in captured.out + captured.err


@needs_pg
def test_one_engine_per_database(pg_dsn):
    held = storage.acquire_pg_engine_lock(pg_dsn)
    try:
        with pytest.raises(storage.EngineDatabaseLockHeld):
            storage.acquire_pg_engine_lock(pg_dsn)
    finally:
        held.close()
    storage.acquire_pg_engine_lock(pg_dsn).close()


@needs_pg
def test_jobs_queue_on_postgres(pg_store):
    from kalos.portal import jobs

    cid = pg_store.create_campaign(tenant="acme", name="c")["id"]
    csv = pd.DataFrame({"x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]}).to_csv(index=False).encode()
    job = jobs.enqueue(pg_store, tenant="acme", campaign_id=cid, raw=csv, filename="r.csv",
                       target="y", anonymize=False, roles="", sheet=None, requested_by="alice")
    assert job["state"] == "queued"
    with pytest.raises(jobs.UploadPending):
        jobs.enqueue(pg_store, tenant="acme", campaign_id=cid, raw=csv, filename="r.csv",
                     target="y", anonymize=False, roles="", sheet=None, requested_by="alice")
    assert jobs.queued_tenants(pg_store) == ["acme"]
    row = jobs.claim(pg_store, tenant="acme", runner_id="t")
    assert row is not None and row["state"] == "running"
    jobs.finish(pg_store, tenant="acme", job_id=row["id"], state="failed",
                code="x", message="y")
    assert jobs.get_job(pg_store, tenant="acme", job_id=row["id"])["state"] == "failed"


def test_postgres_steps_cover_every_schema_version():
    """A schema version added to the SQLite path (kalos/portal/migrate.py) without
    its PostgreSQL DDL fails here, and the engine refuses PostgreSQL at startup."""
    from kalos.portal import migrate_pg

    assert {v for v, _, _ in migrate_pg._STEPS} == set(range(2, SCHEMA_VERSION + 1))
    migrate_pg._check_steps_cover_schema()


def test_a_sqlite_only_version_bump_is_refused(monkeypatch):
    from kalos.portal import migrate_pg

    monkeypatch.setattr(migrate_pg, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
    with pytest.raises(SchemaError, match="PostgreSQL DDL"):
        migrate_pg._check_steps_cover_schema()
    with pytest.raises(SchemaError, match="PostgreSQL DDL"):
        migrate_pg.migrate_up("postgresql://unused/never-connected")


def _grants_sql(role: str, schema: str, database: str) -> list[str]:
    """deploy/postgres/grants.sql with its names swapped for this test's."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "deploy/postgres/grants.sql").read_text()
    lines = [ln.split("--", 1)[0] for ln in text.splitlines()]
    body = "\n".join(lines)
    body = body.replace("kalos_engine", f'"{role}"').replace("SCHEMA public", f'SCHEMA "{schema}"')
    body = body.replace("DATABASE kalos ", f'DATABASE "{database}" ')
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


@needs_pg
def test_owner_and_superuser_are_warned_and_the_documented_grants_work(pg_store, pg_dsn, tmp_path,
                                                                       monkeypatch):
    """The test server connects as a superuser that owns the tables: status warns.
    A role set up with deploy/postgres/grants.sql gets no warning, runs the store
    (result + audit in one transaction), and cannot UPDATE, DELETE, TRUNCATE or
    disable a trigger on the append-only tables."""
    import psycopg

    from kalos.portal import migrate_pg
    from kalos.portal.campaign import CampaignStore

    report, healthy = migrate_pg.status_report(pg_dsn)
    assert healthy and "WARNING" in report and "superuser" in report and "owns" in report
    role = "kalos_app_" + uuid.uuid4().hex[:8]
    schema = pg_dsn.split("search_path%3D")[1]
    with _raw(pg_dsn) as c:
        database = c.execute("SELECT current_database()").fetchone()[0]
        for stmt in _grants_sql(role, schema, database):
            c.execute(stmt)
    app_dsn = pg_dsn.replace("postgresql://postgres@", f"postgresql://{role}@")
    try:
        report, healthy = migrate_pg.status_report(app_dsn)
        assert healthy and "WARNING" not in report
        monkeypatch.setenv("KALOS_DATABASE_URL", app_dsn)
        app_store = CampaignStore(tmp_path / "app")
        try:
            df = pd.DataFrame({"x": [0.1, 0.2], "y": [1.0, 2.0]})
            cid = app_store.create_campaign(tenant="acme", name="c")["id"]
            app_store.seed(df, "y", ["x"], tenant="acme", campaign_id=cid)
            rid = app_store.start([{"recipe": {"x": 0.3}}], tenant="acme", campaign_id=cid)[0]["id"]
            app_store.set_result(rid, 3.0, tenant="acme", campaign_id=cid, audit=True, actor="a")
            assert app_store.audit_entries(tenant="acme")[0]["action"] == "campaign.result"
        finally:
            app_store.close()
        with _raw(app_dsn) as c:
            for table in ("audit_log", "run_results"):
                for stmt in (f"UPDATE {table} SET tenant = 'b'", f"DELETE FROM {table}",
                             f"TRUNCATE {table}",
                             f"ALTER TABLE {table} DISABLE TRIGGER {table}_no_delete"):
                    with pytest.raises(psycopg.Error):
                        c.execute(stmt)
    finally:
        with _raw(pg_dsn) as c:
            c.execute(f'DROP OWNED BY "{role}"')
            c.execute(f'REVOKE CONNECT ON DATABASE "{database}" FROM "{role}"')
            c.execute(f'DROP ROLE "{role}"')


@needs_pg
def test_audit_archive_copies_and_deletes_nothing_on_postgres(pg_store, tmp_path):
    import time

    for i in range(3):
        pg_store.record_audit(tenant="acme", action=f"a{i}", outcome="ok")
    later = time.time() + 10 * 86_400
    assert pg_store.archive_audit_log(1, now=later) == 3
    assert pg_store.archive_audit_log(1, now=later) == 0  # never copied twice
    assert len(pg_store.audit_entries(tenant="acme")) == 3
    files = list((tmp_path / "audit-archive").glob("*.jsonl"))
    assert len(files) == 1 and len(files[0].read_text().splitlines()) == 3
