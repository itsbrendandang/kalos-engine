"""Schema v6: `virtual_runs` is append-only and can never hold measured rows
(kalos/portal/migrate.py, docs/VIRTUAL_RUNS.md). Every database is in tmp_path."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from kalos.portal import migrate
from kalos.portal.campaign import CampaignStore


def _q(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _row(tenant: str, cid: str, **over) -> dict:
    row = {
        "tenant": tenant, "campaign_id": cid, "batch_id": "vjb_" + "0" * 32, "k": 0,
        "origin": "synthetic", "generator_id": "bench.gaussian_bump",
        "generator_version": "1+abc", "params_json": "{}", "params_sha256": "a" * 64,
        "seed_entropy": "[1,2,0,3]", "pcg64_state": "{}", "solver_json": "{}",
        "library_versions": "{}", "twin_version_id": None, "design_json": "{}",
        "design_sha256": "b" * 64, "outcome_json": "{}", "requested_by": "alice",
        "auth_method": "bearer", "created_at": time.time(),
    }
    row.update(over)
    return row


def _insert(conn: sqlite3.Connection, row: dict) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO virtual_runs ({cols}) VALUES ({marks})", tuple(row.values()))  # noqa: S608 - test-only column names


@pytest.fixture
def db(tmp_path):
    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="acme", name="A")["id"]
    other = store.create_campaign(tenant="other", name="B")["id"]
    store.close()
    return tmp_path / "portal.db", cid, other


def test_fresh_database_has_virtual_runs_and_its_triggers(db):
    path, _, _ = db
    assert migrate.SCHEMA_VERSION == 6
    assert (6, migrate.V6_NAME) in _q(path, "SELECT version, name FROM schema_migrations")
    names = {r[0] for r in _q(path, "SELECT name FROM sqlite_master")}
    assert {"virtual_runs", "virtual_runs_no_update", "virtual_runs_no_delete"} <= names
    assert "virtual_runs" in migrate.TENANT_TABLES


def test_insert_ok_update_and_delete_rejected(db):
    path, cid, _ = db
    conn = migrate.connect(path)
    try:
        _insert(conn, _row("acme", cid))
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE virtual_runs SET outcome_json = '{\"y\": 1}'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM virtual_runs")
        conn.rollback()
        assert conn.execute("SELECT COUNT(*) FROM virtual_runs").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize("over", [
    {"origin": "measured"},
    {"origin": "MEASURED"},
    {"origin": "real"},
    {"origin": "synthetic", "twin_version_id": "tv_1"},
    {"origin": "simulated", "twin_version_id": None},
    {"params_sha256": "short"},
    {"requested_by": "   "},
    {"k": -1},
])
def test_check_constraints(db, over):
    path, cid, _ = db
    conn = migrate.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            _insert(conn, _row("acme", cid, **over))
    finally:
        conn.close()


def test_simulated_with_twin_version_is_accepted_and_duplicates_are_refused(db):
    path, cid, _ = db
    conn = migrate.connect(path)
    try:
        _insert(conn, _row("acme", cid, origin="simulated", twin_version_id="tv_1"))
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            _insert(conn, _row("acme", cid, origin="simulated", twin_version_id="tv_1"))
        conn.commit()
    finally:
        conn.close()


def test_campaign_of_another_tenant_is_refused_by_the_foreign_key(db):
    path, _, other = db
    conn = migrate.connect(path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            _insert(conn, _row("acme", other))
    finally:
        conn.close()


def _down_to_v5_with_rows(path: Path, cid: str) -> None:
    conn = migrate.connect(path)
    try:
        _insert(conn, _row("acme", cid))
        conn.commit()
    finally:
        conn.close()
    result = migrate.migrate_down(path)
    assert "schema v6 -> v5" in result.plan_text and "1 row(s)" in result.plan_text


def test_down_keeps_rows_and_up_recreates_the_table_with_a_backup(db):
    path, cid, _ = db
    _down_to_v5_with_rows(path, cid)
    assert migrate.current_version(sqlite3.connect(str(path))) == 5
    tables = {r[0] for r in _q(path, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "virtual_runs" not in tables and "virtual_runs_v6_rolledback" in tables
    assert _q(path, "SELECT COUNT(*) FROM virtual_runs_v6_rolledback") == [(1,)]
    triggers = {r[0] for r in _q(path, "SELECT name FROM sqlite_master WHERE type = 'trigger'")}
    assert not triggers & set(migrate._V6_TRIGGERS)
    # the v5 shape check passes (a v5 engine runs on it)
    report, healthy = migrate.status_report(path)
    assert healthy and "pending migration" in report and "virtual_runs" in report

    plan = migrate.migrate_up(path, dry_run=True).plan_text
    assert "schema v5 -> v6" in plan and "create: virtual_runs" in plan
    result = migrate.migrate_up(path)
    assert result.action == "migrated" and result.backup_path is not None
    assert result.backup_path.name.startswith("portal.db.pre-v6-")
    assert _q(path, "SELECT COUNT(*) FROM virtual_runs") == [(0,)]
    assert _q(path, "SELECT COUNT(*) FROM virtual_runs_v6_rolledback") == [(1,)]
    assert migrate.migrate_up(path).action == "noop"


def test_v6_recorded_without_the_table_is_refused(db):
    path, _, _ = db
    conn = sqlite3.connect(str(path))
    conn.execute("DROP TRIGGER virtual_runs_no_delete")
    conn.commit()
    conn.close()
    with pytest.raises(migrate.SchemaError, match="virtual_runs_no_delete"):
        CampaignStore(path.parent)
