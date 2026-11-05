"""`audit_log` is append-only at the database level (schema v4, docs/MIGRATIONS.md).

Covers: the two triggers reject UPDATE and DELETE and leave every row as it was;
the engine's store has no delete path; the v4 migration and its rollback run on a
COPY of a real v3 database (backup first, rows unchanged, up and down both ways,
the original file never touched); a rolled-back v3 database has the shape the v3
engine checks.

Every database here lives in a pytest tmp dir.
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest

from kalos.portal import migrate
from kalos.portal.campaign import CampaignStore

APPEND_ONLY = "append-only"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _triggers(db: Path) -> set[str]:
    return {r[0] for r in _q(db, "SELECT name FROM sqlite_master WHERE type = 'trigger'")}


def _audit_rows(db: Path) -> list[tuple]:
    return _q(db, "SELECT * FROM audit_log ORDER BY id")


def _make_v3_database(db: Path) -> None:
    """A v3 database built with the v3 schema code (v2 tables + `jobs`), holding a
    campaign row and audit rows, as a v3 engine would have left it."""
    conn = migrate.connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        migrate._create_v2(conn)
        migrate._record_version(conn, migrate.V2, migrate.V2_NAME)
        migrate._apply_v3_in_txn(conn)
        conn.execute("COMMIT")
        conn.execute(
            "INSERT INTO campaigns (id, tenant, name, state, generation, created_by, created_at, "
            "updated_at, archived_at) VALUES (?, 'acme', 'Campaign A', NULL, NULL, 'legacy-route', "
            "1.0, 1.0, NULL)",
            ("cmp_" + uuid.uuid4().hex,),
        )
        for i, action in enumerate(["campaign.start", "campaign.result", "run.upload"]):
            conn.execute(
                "INSERT INTO audit_log (ts, tenant, subject, auth_method, action, outcome, detail) "
                "VALUES (?, 'acme', 'alice', 'bearer', ?, 'ok', ?)",
                (1000.0 + i, action, f"row {i}"),
            )
        conn.execute(
            "INSERT INTO audit_log (ts, tenant, subject, auth_method, action, outcome) "
            "VALUES (2000.0, 'globex', 'bob', 'bearer', 'campaign.start', 'http_400')"
        )
    finally:
        conn.close()


# --- the triggers, on a fresh database ---------------------------------------- #


def test_fresh_database_rejects_update_and_delete_and_keeps_rows(tmp_path):
    store = CampaignStore(tmp_path)
    store.record_audit(tenant="acme", action="campaign.start", outcome="ok", subject="alice")
    store.record_audit(tenant="acme", action="run.upload", outcome="ok", subject="alice")
    store.close()
    db = tmp_path / "portal.db"
    before = _audit_rows(db)
    assert len(before) == 2
    assert {"audit_log_no_update", "audit_log_no_delete"} <= _triggers(db)

    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError, match=f"audit_log is {APPEND_ONLY}: UPDATE is rejected"):
            conn.execute("UPDATE audit_log SET outcome = 'forged' WHERE id = 1")
        with pytest.raises(sqlite3.IntegrityError, match=f"audit_log is {APPEND_ONLY}: DELETE is rejected"):
            conn.execute("DELETE FROM audit_log WHERE id = 1")
        with pytest.raises(sqlite3.IntegrityError, match="DELETE is rejected"):
            conn.execute("DELETE FROM audit_log")  # every row, no WHERE
    finally:
        conn.close()
    assert _audit_rows(db) == before


def test_engine_store_has_no_delete_path_and_still_appends(tmp_path):
    store = CampaignStore(tmp_path)
    assert not hasattr(CampaignStore, "prune_audit_log")
    store.record_audit(tenant="acme", action="a", outcome="ok")
    store.record_audit(tenant="acme", action="b", outcome="ok")
    assert [e["action"] for e in store.audit_entries(tenant="acme")] == ["b", "a"]
    store.close()


def test_v4_is_recorded_and_the_store_refuses_a_v3_shape_without_triggers(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    assert (migrate.V4, migrate.V4_NAME) in _q(
        db, "SELECT version, name FROM schema_migrations ORDER BY version")
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TRIGGER audit_log_no_update")
    conn.commit()
    conn.close()
    with pytest.raises(migrate.SchemaError, match="trigger audit_log_no_update"):
        CampaignStore(tmp_path)


# --- the v3 -> v4 migration, on a copy of a real v3 database ------------------ #


def test_v3_copy_migrates_up_with_a_backup_and_keeps_every_audit_row(tmp_path):
    source = tmp_path / "source" / "portal.db"
    source.parent.mkdir()
    _make_v3_database(source)
    source_hash = _sha256(source)
    before_rows = _audit_rows(source)
    before_campaigns = _q(source, "SELECT id, tenant, name FROM campaigns ORDER BY id")

    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    copy = copy_dir / "portal.db"
    shutil.copy2(source, copy)

    plan = migrate.migrate_up(copy, dry_run=True).plan_text
    assert "schema v3 -> v5" in plan and "triggers" in plan
    assert _triggers(copy) == set()  # a dry run changes nothing

    result = migrate.migrate_up(copy)
    assert result.action == "migrated"
    assert result.backup_path is not None
    assert result.backup_path.name.startswith("portal.db.pre-v4-")
    assert result.backup_path.exists()
    assert _audit_rows(result.backup_path) == before_rows  # the pre-v4 backup is a real copy

    assert _q(copy, "SELECT version FROM schema_migrations ORDER BY version") == [(2,), (3,), (4,), (5,)]
    assert {"audit_log_no_update", "audit_log_no_delete"} <= _triggers(copy)
    assert _audit_rows(copy) == before_rows
    assert _q(copy, "SELECT id, tenant, name FROM campaigns ORDER BY id") == before_campaigns
    assert migrate.migrate_up(copy).action == "noop"

    conn = sqlite3.connect(str(copy))
    try:
        with pytest.raises(sqlite3.IntegrityError, match=APPEND_ONLY):
            conn.execute("UPDATE audit_log SET subject = 'mallory'")
        with pytest.raises(sqlite3.IntegrityError, match=APPEND_ONLY):
            conn.execute("DELETE FROM audit_log")
    finally:
        conn.close()
    assert _audit_rows(copy) == before_rows

    # the original v3 file was only ever read
    assert _sha256(source) == source_hash


def test_v4_copy_rolls_back_to_v3_and_forward_again(tmp_path):
    source = tmp_path / "source" / "portal.db"
    source.parent.mkdir()
    _make_v3_database(source)
    before_rows = _audit_rows(source)
    copy = tmp_path / "copy" / "portal.db"
    copy.parent.mkdir()
    shutil.copy2(source, copy)
    migrate.migrate_up(copy)

    plan = migrate.migrate_down(copy, dry_run=True, target=3).plan_text
    assert "schema v4 -> v3" in plan and "drop: audit_log" in plan
    assert {"audit_log_no_update", "audit_log_no_delete"} <= _triggers(copy)

    result = migrate.migrate_down(copy, target=3)
    assert result.action == "rolled_back"
    assert result.backup_path is not None and result.backup_path.name.startswith("portal.db.pre-down-")
    assert _triggers(copy) == set()
    assert _q(copy, "SELECT version FROM schema_migrations ORDER BY version") == [(2,), (3,)]
    assert _audit_rows(copy) == before_rows  # rows survive the rollback
    # the v3 shape is what the v3 engine checks
    conn = sqlite3.connect(str(copy))
    try:
        migrate._assert_v2_shape(conn, 3)
    finally:
        conn.close()
    # on v3 the old delete path works again (this is why rolling back is an operator decision)
    conn = sqlite3.connect(str(copy))
    try:
        conn.execute("DELETE FROM audit_log WHERE tenant = 'globex'")
        conn.commit()
    finally:
        conn.close()
    # a row written while on v3, then up again: the triggers come back and the row stays
    conn = sqlite3.connect(str(copy))
    conn.execute("INSERT INTO audit_log (ts, tenant, action, outcome) VALUES (3000, 'acme', 'x', 'ok')")
    conn.commit()
    conn.close()
    rows_after_v3 = _audit_rows(copy)
    assert migrate.migrate_up(copy).action == "migrated"
    assert {"audit_log_no_update", "audit_log_no_delete"} <= _triggers(copy)
    assert _audit_rows(copy) == rows_after_v3


def test_v3_copy_rolls_back_all_the_way_to_v1(tmp_path):
    """`down --to 1` from v4 passes through the v4 and v3 steps, then v2 -> v1."""
    db = tmp_path / "portal.db"
    _make_v3_database(db)
    migrate.migrate_up(db)
    result = migrate.migrate_down(db, target=1)
    assert result.action == "rolled_back"
    conn = sqlite3.connect(str(db))
    try:
        assert migrate.campaigns_table_is_v1_compatible(conn)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()
    assert {"audit_log_v2_rolledback", "jobs_v3_rolledback", "campaigns_v2_rolledback"} <= tables
    assert _triggers(db) == set()


def test_down_from_v4_with_a_newer_schema_is_refused(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO schema_migrations VALUES (6, 'future', 0)")
    conn.commit()
    conn.close()
    with pytest.raises(migrate.SchemaError, match="newer"):
        CampaignStore(tmp_path)


def test_invalid_down_target_is_refused(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    with pytest.raises(migrate.MigrationError, match=r"valid targets are 1\.\.4"):
        migrate.migrate_down(db, target=5)
