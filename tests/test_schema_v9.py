"""Schema v9: records ported from Voyager (projects, factors, lineage).

Covers a fresh database at v9; the five tables, their CHECKs and the append-only
lineage triggers; the v8 -> v9 upgrade on a COPY of a real v8 database
(`tests/fixtures/portal_v8.db`, built by running the v8 engine over HTTP, see
`tests/fixtures/make_portal_v8_db.py`): a verified `pre-v9` backup, every v8 row
kept, nothing but the new tables added; the engine then serves the records
routes on the migrated file, with the outcome view reading the REAL v8 results
and predictions; the v9 -> v8 rollback renames aside, a v8 engine's shape check
accepts it, and a re-upgrade is clean. The fixture file is never opened for
writing.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from kalos.portal import migrate
from kalos.portal.campaign import CampaignStore
from kalos.storage.migrations import v9

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "portal_v8.db"
_REPO = Path(__file__).resolve().parents[1]
V9_TABLES = set(v9.TABLES)
V9_TRIGGERS = set(v9.SQLITE_TRIGGERS)


def _q(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _names(db: Path, kind: str) -> set[str]:
    return {r[0] for r in _q(db, "SELECT name FROM sqlite_master WHERE type = ? "
                                 "AND name NOT LIKE 'sqlite_%'", (kind,))}


def _snapshot(db: Path) -> dict[str, list[tuple]]:
    return {t: sorted(_q(db, f'SELECT * FROM "{t}"'), key=repr)  # noqa: S608 - sqlite_master names
            for t in sorted(_names(db, "table"))}


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def v8_copy(tmp_path) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    db = state / "portal.db"
    shutil.copyfile(FIXTURE, db)
    return db


def test_fresh_store_is_created_at_v9(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    assert migrate.SCHEMA_VERSION == 9
    assert _q(db, "SELECT version, name FROM schema_migrations WHERE version = 9") == [(9, migrate.V9_NAME)]
    assert _names(db, "table") >= V9_TABLES
    assert _names(db, "trigger") >= V9_TRIGGERS
    assert set(v9.INDEXES) <= _names(db, "index")
    assert set(migrate.TENANT_TABLES) >= V9_TABLES


def test_the_fixture_is_a_real_v8_database():
    assert _q(FIXTURE, "SELECT MAX(version) FROM schema_migrations") == [(8,)]
    assert not (_names(FIXTURE, "table") & V9_TABLES)
    # real engine history: recorded predictions, corrected results and folds
    assert _q(FIXTURE, "SELECT COUNT(*) FROM run_predictions")[0][0] > 0
    assert _q(FIXTURE, "SELECT COUNT(*) FROM run_results WHERE version > 1")[0][0] >= 1
    assert _q(FIXTURE, "SELECT COUNT(*) FROM run_folds")[0][0] > 0


@pytest.fixture
def v9db(tmp_path):
    CampaignStore(tmp_path).close()
    conn = sqlite3.connect(str(tmp_path / "portal.db"))
    conn.execute(
        "INSERT INTO projects (id, tenant, name, objective, status, created_at, updated_at) "
        "VALUES ('prj_1', 'acme', 'P', 'o', 'drafting', 1, 1)")
    conn.execute(
        "INSERT INTO lineage_links (tenant, parent_campaign_id, child_campaign_id, relation, created_at) "
        "VALUES ('acme', 'cmp_a', 'cmp_b', 'scale_up', 1)")
    conn.execute("INSERT INTO lineage_retractions (tenant, link_id, reason, created_at) VALUES ('acme', 1, 'x', 1)")
    conn.commit()
    yield conn
    conn.close()


@pytest.mark.parametrize("table", ["lineage_links", "lineage_retractions"])
def test_lineage_tables_are_append_only(v9db, table):
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        v9db.execute(f"UPDATE {table} SET created_at = 2")  # noqa: S608 - parametrized constant
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        v9db.execute(f"DELETE FROM {table}")  # noqa: S608 - parametrized constant


@pytest.mark.parametrize("sql", [
    "INSERT INTO projects (id, tenant, name, objective, status, created_at, updated_at) "
    "VALUES ('prj_2', 'acme', 'Q', 'o', 'running', 1, 1)",                       # unknown status
    "INSERT INTO projects (id, tenant, name, objective, status, created_at, updated_at) "
    "VALUES ('prj_3', 'acme', 'P', 'o', 'drafting', 1, 1)",                       # name taken in tenant
    "INSERT INTO project_factors (tenant, project_id, name, lower_bound, upper_bound, updated_at) "
    "VALUES ('acme', 'prj_1', 'f', 5, 5, 1)",                                     # lower == upper
    "INSERT INTO project_factors (tenant, project_id, name, lower_bound, upper_bound, role, updated_at) "
    "VALUES ('acme', 'prj_1', 'f', 1, 5, 'outcome', 1)",                          # unknown role
    "INSERT INTO lineage_links (tenant, parent_campaign_id, child_campaign_id, relation, created_at) "
    "VALUES ('acme', 'cmp_a', 'cmp_a', 'scale_up', 1)",                           # self link
    "INSERT INTO lineage_links (tenant, parent_campaign_id, child_campaign_id, relation, created_at) "
    "VALUES ('acme', 'cmp_a', 'cmp_c', 'cousin', 1)",                             # unknown relation
    "INSERT INTO lineage_retractions (tenant, link_id, reason, created_at) VALUES ('acme', 1, 'again', 1)",
    "INSERT INTO lineage_retractions (tenant, link_id, reason, created_at) VALUES ('acme', 1, '  ', 1)",
])
def test_v9_checks(v9db, sql):
    with pytest.raises(sqlite3.IntegrityError):
        v9db.execute(sql)


def test_a_project_name_may_repeat_across_tenants(v9db):
    v9db.execute(
        "INSERT INTO projects (id, tenant, name, objective, status, created_at, updated_at) "
        "VALUES ('prj_9', 'globex', 'P', 'o', 'drafting', 1, 1)")


# --- v8 -> v9 on a copy of the real v8 database --------------------------------------- #


def test_v8_copy_migrates_with_a_verified_backup_and_keeps_every_row(v8_copy):
    fixture_sha = _sha_file(FIXTURE)
    db = v8_copy
    before = _snapshot(db)

    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert "schema v8 -> v9" in plan and "pre-v9" in plan and "lineage_links" in plan
    assert "analysis_snapshots" not in plan  # only the remaining step is listed
    assert _snapshot(db) == before

    result = migrate.migrate_up(db)
    assert result.action == "migrated" and result.warnings == []
    assert result.backup_path is not None and result.backup_path.name.startswith("portal.db.pre-v9-")
    assert _snapshot(result.backup_path) == before  # the backup is the v8 database

    after = _snapshot(db)
    assert [r[0] for r in after.pop("schema_migrations")] == [2, 3, 4, 5, 6, 7, 8, 9]
    before.pop("schema_migrations")
    for t in V9_TABLES:
        assert after.pop(t) == []
    assert after == before  # every v8 row, byte for byte
    assert _names(db, "trigger") >= V9_TRIGGERS
    assert _sha_file(FIXTURE) == fixture_sha  # the fixture itself was never written

    assert migrate.migrate_up(db).action == "noop"


def test_records_routes_work_on_the_migrated_database_and_read_real_history(v8_copy, monkeypatch):
    """The upgraded file serves the records routes; a project over the fixture's
    real campaigns shows each run's newest measured value and its prediction at
    start exactly as the v8 engine recorded them."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.api.routers._records_common import reset_backends
    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import get_campaign_store

    state = v8_copy.parent
    monkeypatch.delenv("KALOS_DATABASE_URL", raising=False)
    store = CampaignStore(state)  # startup migrates v8 -> v9
    tok = [{"token_sha256": hashlib.sha256(b"w").hexdigest(), "subject": "s", "tenant": "acme",
            "scopes": ["read", "write"]}]
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(tok))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", state)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        tc, h = TestClient(app), {"Authorization": "Bearer w"}
        (cid,) = [r[0] for r in _q(v8_copy, "SELECT id FROM campaigns WHERE tenant = 'acme' AND state IS NOT NULL")]
        pid = tc.post("/api/projects", headers=h, json={"name": "P", "objective": "o"}).json()["project"]["id"]
        assert tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=h).status_code == 200
        runs = tc.get(f"/api/projects/{pid}/outcomes?limit=500", headers=h).json()["runs"]
        latest = {(r[0]): r[1] for r in _q(
            v8_copy, "SELECT run_id, value FROM run_results r WHERE tenant = 'acme' AND campaign_id = ? "
                     "AND version = (SELECT MAX(version) FROM run_results x WHERE x.tenant = r.tenant "
                     "AND x.campaign_id = r.campaign_id AND x.run_id = r.run_id)", (cid,))}
        preds = {r[0]: r[1] for r in _q(
            v8_copy, "SELECT run_id, pred FROM run_predictions WHERE tenant = 'acme' AND campaign_id = ? "
                     "AND available = 1", (cid,))}
        assert {r["run_id"] for r in runs} == set(latest) | set(preds)
        for r in runs:
            if r["run_id"] in latest:
                assert r["measured"]["value"] == latest[r["run_id"]]
            if r["run_id"] in preds:
                assert r["prediction_at_start"]["mean"] == preds[r["run_id"]]
        # a run-level link between two real runs of the fixture
        a, b = sorted(latest)[:2]
        link = tc.post("/api/lineage/links", headers=h, json={
            "parent_campaign_id": cid, "child_campaign_id": cid, "relation": "replicate",
            "parent_run_id": a, "child_run_id": b})
        assert link.status_code == 201, link.text
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        reset_backends()
        store.close()


# --- down and back ------------------------------------------------------------------ #


def test_down_renames_aside_and_a_v8_engine_accepts_it_and_re_up_is_clean(v8_copy):
    db = v8_copy
    migrate.migrate_up(db)
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO projects (id, tenant, name, objective, status, created_at, updated_at) "
                 "VALUES ('prj_1', 'acme', 'P', 'o', 'drafting', 1, 1)")
    conn.commit()
    conn.close()
    dry = migrate.migrate_down(db, dry_run=True).plan_text
    assert "schema v9 -> v8" in dry and "schema v8 -> v7" not in dry
    assert "rename: projects -> projects_v9_rolledback (1 rows kept, not dropped)" in dry

    result = migrate.migrate_down(db)
    assert result.action == "rolled_back" and result.backup_path is not None
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(8,)]
    tables = _names(db, "table")
    assert not (tables & V9_TABLES)
    assert {f"{t}_v9_rolledback" for t in V9_TABLES} <= tables
    assert _q(db, 'SELECT COUNT(*) FROM "projects_v9_rolledback"') == [(1,)]
    assert not (V9_TRIGGERS & _names(db, "trigger"))
    conn2 = migrate.connect(db)
    try:
        migrate._assert_v2_shape(conn2)  # what a v8 engine checks at startup
    finally:
        conn2.close()

    assert migrate.migrate_up(db).action == "migrated"
    assert _q(db, "SELECT COUNT(*) FROM projects") == [(0,)]
    migrate.migrate_down(db)
    assert {f"{t}_v9_rolledback_2" for t in V9_TABLES} <= _names(db, "table")


def test_status_shows_v9_and_pending_from_v8(v8_copy):
    report, healthy = migrate.status_report(v8_copy)
    assert healthy and "schema version: v8 (this code: v9)" in report and "schema v8 -> v9" in report
    migrate.migrate_up(v8_copy)
    report, healthy = migrate.status_report(v8_copy)
    assert healthy and "schema version: v9" in report and "shape check: ok" in report


@pytest.mark.parametrize(("ddl", "missing"), [
    ("DROP TRIGGER lineage_links_no_update", "trigger lineage_links_no_update"),
    ("DROP TRIGGER lineage_retractions_no_delete", "trigger lineage_retractions_no_delete"),
    ("ALTER TABLE project_factors RENAME TO elsewhere", "project_factors"),
])
def test_shape_check_catches_a_missing_v9_table_or_trigger(tmp_path, ddl, missing):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(ddl)
    conn.commit()
    conn.close()
    with pytest.raises(migrate.SchemaError):
        CampaignStore(tmp_path)
    report, healthy = migrate.status_report(db)
    assert not healthy and missing in report


def test_a_failed_backup_stops_the_v9_upgrade(v8_copy, monkeypatch):
    before = _snapshot(v8_copy)

    def boom(src, dest):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(migrate, "_online_backup", boom)
    with pytest.raises(migrate.MigrationError):
        migrate.migrate_up(v8_copy)
    assert _snapshot(v8_copy) == before


def test_docs_describe_v9_and_the_postgres_backend():
    text = (_REPO / "docs" / "MIGRATIONS.md").read_text()
    assert "## Schema v9: records (projects, factors, lineage)" in text
    assert "## PostgreSQL backend" in text
    assert "current (`SCHEMA_VERSION = 9`" in text
