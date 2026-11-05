"""Migration safety on three database shapes, each built in a pytest tmp dir:

(a) a clean v3 database: migrates to v4 in one transaction, idempotent;
(b) a database whose `schema_migrations` row is missing or empty but whose v2
    tables exist: REFUSED before any backup or write (the v1 path would rename
    `campaigns` and break the foreign keys that name it);
(c) the "stuck" shape seen on a real state directory: version row 3, live
    `campaigns` in the v1 shape, a `campaigns_backup_N` holding the v2 shape
    that the foreign keys still name: REFUSED, nothing renamed or backed up.

In every refusal the audit triggers, the audit rows and every table are left as
they were, and the rename-based repair (`campaigns` -> `campaigns_backup_N`) is
never reached. The fixtures are synthetic: no copy of any real state directory
is used.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from kalos.portal import migrate
from kalos.portal.campaign import CampaignStore


def _snapshot(db: Path) -> dict:
    conn = sqlite3.connect(str(db))
    try:
        tables = sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))
        triggers = sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'"))
        audit = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
        versions = (
            conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
            if "schema_migrations" in tables else None
        )
        campaigns = conn.execute("SELECT * FROM campaigns ORDER BY 1").fetchall()
    finally:
        conn.close()
    return {"tables": tables, "triggers": triggers, "audit": audit,
            "versions": versions, "campaigns": campaigns}


def _backups(db: Path) -> list[str]:
    return sorted(p.name for p in db.parent.glob(f"{db.name}.pre-*"))


def _add_audit_row(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO audit_log (ts, tenant, subject, auth_method, action, outcome) "
        "VALUES (1.0, 'acme', 'alice', 'bearer', 'campaign.start', 'ok')"
    )
    conn.commit()
    conn.close()


def _clean_v4(state: Path) -> Path:
    store = CampaignStore(state)
    store.record_audit(tenant="acme", action="campaign.start", outcome="ok", subject="alice")
    store.close()
    return state / "portal.db"


def _clean_v3(state: Path) -> Path:
    """A v3 database as the v3 code wrote it: v2 tables, `jobs`, version 3, no triggers."""
    db = state / "portal.db"
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
    finally:
        conn.close()
    _add_audit_row(db)
    return db


def _stuck_shape(state: Path) -> Path:
    """The real-world stuck state, reproduced: version row says 3, the live
    `campaigns` table has the v1 shape, and the v2 table lives on as
    `campaigns_backup_2`, which `analyses` and `tenant_defaults` still reference."""
    db = _clean_v3(state)
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE campaigns RENAME TO campaigns_backup_2")
    conn.execute(migrate.V1_CAMPAIGNS_DDL)
    conn.execute(
        "INSERT INTO campaigns (tenant, state, updated_at) VALUES ('acme', '{}', 2.0)"
    )
    conn.commit()
    conn.close()
    return db


# --- (a) clean v3 -------------------------------------------------------------- #


def test_clean_v3_migrates_once_and_is_idempotent(tmp_path):
    db = _clean_v3(tmp_path)
    before = _snapshot(db)

    result = migrate.migrate_up(db)

    assert result.action == "migrated"
    assert result.backup_path is not None and result.backup_path.name.startswith("portal.db.pre-v4-")
    after = _snapshot(db)
    assert after["audit"] == before["audit"]
    assert after["campaigns"] == before["campaigns"]
    assert after["versions"] == [(2,), (3,), (4,), (5,)]
    assert {"audit_log_no_update", "audit_log_no_delete"} <= set(after["triggers"])
    assert not any("backup" in t for t in after["tables"])  # no repair table appeared
    assert migrate.migrate_up(db).action == "noop"
    assert _snapshot(db) == after
    assert _backups(db) == [result.backup_path.name]  # no second backup on the no-op


def test_clean_v3_dry_run_changes_nothing(tmp_path):
    db = _clean_v3(tmp_path)
    before = _snapshot(db)
    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert "schema v3 -> v5" in plan
    assert _snapshot(db) == before
    assert _backups(db) == []


# --- (b) schema_migrations missing or empty ------------------------------------ #


@pytest.mark.parametrize("how", ["emptied", "dropped"])
def test_v2_tables_without_a_version_row_are_refused_before_any_change(tmp_path, how):
    db = _clean_v4(tmp_path)
    conn = sqlite3.connect(str(db))
    if how == "emptied":
        conn.execute("DELETE FROM schema_migrations")
    else:
        conn.execute("DROP TABLE schema_migrations")
    conn.commit()
    conn.close()
    before = _snapshot(db)

    with pytest.raises(migrate.SchemaError, match="no schema version row"):
        migrate.migrate_up(db)
    with pytest.raises(migrate.SchemaError, match="no schema version row"):
        migrate.migrate_up(db, dry_run=True)
    with pytest.raises(migrate.SchemaError, match="no schema version row"):
        CampaignStore(tmp_path)  # the engine's startup path

    assert _snapshot(db) == before  # no rename, no table, no trigger change
    assert _backups(db) == []  # refused before the backup, so none was written
    assert not any("campaigns_backup" in t for t in before["tables"])


# --- (c) the stuck shape -------------------------------------------------------- #


def test_stuck_shape_is_refused_and_nothing_is_renamed(tmp_path):
    db = _stuck_shape(tmp_path)
    before = _snapshot(db)
    assert "campaigns_backup_2" in before["tables"]
    assert before["versions"] == [(2,), (3,)]

    with pytest.raises(migrate.SchemaError):
        migrate.migrate_up(db)
    with pytest.raises(migrate.SchemaError):
        migrate.migrate_up(db, dry_run=True)

    after = _snapshot(db)
    assert after == before
    assert "campaigns_backup_3" not in after["tables"]
    assert _backups(db) == []  # the refusal happens before any backup
    with pytest.raises(migrate.SchemaError):
        CampaignStore(tmp_path)


def test_stuck_shape_refusal_names_the_missing_columns(tmp_path):
    db = _stuck_shape(tmp_path)
    with pytest.raises(migrate.SchemaError, match="v3 shape"):
        migrate.migrate_up(db)


# --- the new v4 tables and triggers survive the rename path -------------------- #


def test_audit_triggers_and_rows_survive_a_down_and_up_round_trip(tmp_path):
    """The rename that a rollback or repair performs moves tables; the audit
    triggers must come back exactly, and every audit row with them."""
    db = _clean_v4(tmp_path)
    before_audit = _snapshot(db)["audit"]
    migrate.migrate_down(db, target=3)
    assert _snapshot(db)["triggers"] == []
    migrate.migrate_up(db)
    after = _snapshot(db)
    assert {"audit_log_no_update", "audit_log_no_delete"} <= set(after["triggers"])
    assert after["audit"] == before_audit
    conn = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM audit_log")
    finally:
        conn.close()
