"""Schema v7: label-only `campaign_clones` + append-only `clone_titer_history`.

Covers a fresh database at v7, the v6 -> v7 upgrade on a populated v6 database
(every existing row kept, a verified `pre-v7` backup taken), the table
constraints and the append-only triggers, the v7 -> v6 rollback (both tables
renamed aside with their rows; a re-upgrade is clean and never reuses a name),
the plan/status text and the shape check. Since schema v8 the current schema is
newer than v7: the fixtures roll back with `--to 6` and the upgrade goes on to
v8 in the same transaction (tests/test_schema_v8.py covers the v8 step).
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

from kalos.portal import campaign_settings, clone_registry, migrate
from kalos.portal.campaign import CampaignStore

RECIPE = {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1, "mode": "explore", "reason": "x"}
_V7_TABLES = {"campaign_clones", "clone_titer_history"}
_V7_TRIGGERS = {"clone_titer_history_no_update", "clone_titer_history_no_delete"}
_REPO = Path(__file__).resolve().parents[1]


def _sheet(n: int = 8, seed: int = 0) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2).round(3)})


def _q(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _names(db: Path, kind: str) -> set[str]:
    return {r[0] for r in _q(db, "SELECT name FROM sqlite_master WHERE type = ? AND name NOT LIKE 'sqlite_%'", (kind,))}


def _snapshot(db: Path) -> dict[str, list[tuple]]:
    out = {}
    for t in sorted(_names(db, "table")):
        out[t] = sorted(_q(db, f'SELECT * FROM "{t}"'), key=repr)  # noqa: S608 - sqlite_master names
    return out


def _make_v6(state: Path) -> tuple[Path, dict[str, str]]:
    """A populated v6 database as a v6 engine leaves it (no v7 trace): seeded
    campaigns, analyses, a correction in run_results, a setting, a gate decision."""
    store = CampaignStore(state)
    ids: dict[str, str] = {}
    for i, tenant in enumerate(("acme", "globex")):
        store.seed(_sheet(seed=i), "lipase_titer", ["Methanol", "pH"], tenant=tenant,
                   analysis={"target": "lipase_titer", "tenant": tenant}, dataset=f"{tenant}.csv")
        ids[tenant] = store.default_campaign_id(tenant=tenant) or ""
        run = store.start([RECIPE], tenant=tenant)[0]
        store.set_result(run["id"], 3.0 + i, tenant=tenant)
        store.set_result(run["id"], 3.5 + i, tenant=tenant, reason="re-read the plate")
        campaign_settings.put_setting(store, tenant=tenant, campaign_id=ids[tenant], kind="setup",
                                      value={"goal": "maximize"}, updated_by="u")
    store._conn.execute(
        "INSERT INTO gate_decisions (tenant, campaign_id, gate, decision, note, evidence, decided_by, "
        "auth_method, decided_at) VALUES ('acme', ?, 'S1', 'go', 'ok', '[]', 'a', 'static', ?)",
        (ids["acme"], time.time()),
    )
    store.close()
    db = state / "portal.db"
    migrate.migrate_down(db, target=6)
    for b in state.glob("portal.db.pre-*"):
        b.unlink()
    conn = sqlite3.connect(str(db))
    for t in ("campaign_clones_v7_rolledback", "clone_titer_history_v7_rolledback",
              "analysis_snapshots_v8_rolledback", "run_predictions_v8_rolledback",
              "run_folds_v8_rolledback"):
        conn.execute(f'DROP TABLE "{t}"')
    conn.commit()
    conn.close()
    return db, ids


# --- fresh + upgrade ----------------------------------------------------------- #


def test_fresh_store_is_created_at_v7(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    assert migrate.SCHEMA_VERSION >= 7
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(migrate.SCHEMA_VERSION,)]
    assert _q(db, "SELECT name FROM schema_migrations WHERE version = 7") == [(migrate.V7_NAME,)]
    assert _names(db, "table") >= _V7_TABLES
    assert _names(db, "trigger") >= _V7_TRIGGERS
    assert "clone_titer_history_label_idx" in _names(db, "index")
    idx_cols = [r[2] for r in _q(db, "PRAGMA index_info('clone_titer_history_label_idx')")]
    assert idx_cols == ["tenant", "campaign_id", "label", "id"]
    assert {"campaign_clones", "clone_titer_history"} <= set(migrate.TENANT_TABLES)


def test_the_v6_fixture_has_no_v7_trace(tmp_path):
    db, _ = _make_v6(tmp_path / "s")
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(6,)]
    assert not (_names(db, "table") & _V7_TABLES)
    assert not any("v7" in t or "v8" in t for t in _names(db, "table"))


def test_v6_to_v7_keeps_every_row_and_takes_a_verified_backup(tmp_path):
    state = tmp_path / "s"
    db, ids = _make_v6(state)
    before = _snapshot(db)
    assert len(before["run_results"]) == 4 and len(before["campaign_settings"]) == 2
    assert len(before["gate_decisions"]) == 1

    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert f"schema v6 -> v{migrate.SCHEMA_VERSION}" in plan and "pre-v7" in plan
    assert "campaign_clones" in plan and "clone_titer_history" in plan
    assert "gate_decisions" not in plan  # only the remaining step is listed
    assert _snapshot(db) == before

    result = migrate.migrate_up(db)
    assert result.action == "migrated"
    assert result.backup_path is not None and result.backup_path.name.startswith("portal.db.pre-v7-")
    assert _snapshot(result.backup_path) == before

    after = _snapshot(db)
    assert [r[0] for r in after.pop("schema_migrations")] == list(range(2, migrate.SCHEMA_VERSION + 1))
    before.pop("schema_migrations")
    assert after.pop("campaign_clones") == [] and after.pop("clone_titer_history") == []
    for t in ("analysis_snapshots", "run_predictions", "run_folds"):  # v8, tests/test_schema_v8.py
        after.pop(t)
    after.pop("sqlite_sequence", None)
    before.pop("sqlite_sequence", None)
    assert after == before
    assert _names(db, "trigger") >= _V7_TRIGGERS

    store = CampaignStore(state)
    assert store.get(tenant="acme")["pending"][0]["result"] == 3.5
    clone_registry.put_clone(store, tenant="acme", campaign_id=ids["acme"], label="CHO-A12",
                             status="lead", fields={}, updated_by="u")
    assert clone_registry.lead_clone_label(store, tenant="acme", campaign_id=ids["acme"]) == "CHO-A12"
    store.close()
    assert migrate.migrate_up(db).action == "noop"


def test_plan_text_from_v1_mentions_the_v7_tables(tmp_path):
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE campaigns (tenant TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL)")
    conn.commit()
    conn.close()
    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert f"v2 -> v{migrate.SCHEMA_VERSION}" in plan and "campaign_clones and clone_titer_history" in plan


# --- constraints and triggers ---------------------------------------------------- #


_INS_CLONE = ("INSERT INTO campaign_clones (tenant, campaign_id, label, fields, status, updated_at, "
              "updated_by) VALUES (?, ?, ?, '{}', ?, 1.0, NULL)")
_INS_TITER = ("INSERT INTO clone_titer_history (tenant, campaign_id, label, recorded_at, basis, value, "
              "unit, age_kind, previous_id, reason) VALUES (?, ?, 'CHO-A12', 1.0, ?, 2.5, 'g/L', ?, ?, ?)")


@pytest.fixture
def v7(tmp_path):
    store = CampaignStore(tmp_path)
    c = store.create_campaign(tenant="acme", name="a")
    store.close()
    conn = migrate.connect(tmp_path / "portal.db")
    yield conn, c["id"]
    conn.close()


def test_clone_titer_history_is_append_only(v7):
    conn, cid = v7
    conn.execute(_INS_TITER, ("acme", cid, "harvest_titer", None, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="append-only: UPDATE"):
        conn.execute("UPDATE clone_titer_history SET value = 9")
    with pytest.raises(sqlite3.IntegrityError, match="append-only: DELETE"):
        conn.execute("DELETE FROM clone_titer_history")
    assert conn.execute("SELECT value FROM clone_titer_history").fetchall() == [(2.5,)]


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_a_correction_row_needs_a_reason(v7, reason):
    conn, cid = v7
    conn.execute(_INS_TITER, ("acme", cid, "harvest_titer", None, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_TITER, ("acme", cid, "harvest_titer", None, 1, reason))
    conn.execute(_INS_TITER, ("acme", cid, "harvest_titer", None, 1, "re-assayed"))


@pytest.mark.parametrize(("basis", "age_kind"), [("titer", None), ("QP", None), ("qP", "days")])
def test_titer_enums_are_checked(v7, basis, age_kind):
    conn, cid = v7
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_TITER, ("acme", cid, basis, age_kind, None, None))


def test_clone_status_and_label_checks_and_tenant_fk(v7):
    conn, cid = v7
    for status in ("lead", "backup", "retired", "candidate"):
        conn.execute(_INS_CLONE, ("acme", cid, f"C-{status}", status))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_CLONE, ("acme", cid, "C-x", "primary"))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_CLONE, ("acme", cid, "", "lead"))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_CLONE, ("acme", cid, "x" * 65, "lead"))
    conn.execute(_INS_CLONE, ("acme", cid, "x" * 64, "lead"))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        conn.execute(_INS_CLONE, ("acme", cid, "C-lead", "backup"))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(_INS_CLONE, ("globex", cid, "C-g", "lead"))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(_INS_TITER, ("globex", cid, "harvest_titer", None, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(_INS_CLONE, ("acme", "cmp_" + uuid.uuid4().hex, "C-m", "lead"))


# --- down and back ------------------------------------------------------------------ #


def test_down_renames_both_tables_aside_and_re_up_is_clean(tmp_path):
    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="acme", name="a")["id"]
    clone_registry.put_clone(store, tenant="acme", campaign_id=cid, label="CHO-A12", status="lead",
                             fields={"host": "CHO-K1"}, updated_by="u")
    clone_registry.append_titer(store, tenant="acme", campaign_id=cid, label="CHO-A12",
                                row={"basis": "qP", "value": 22.0, "unit": "pg/cell/day"}, actor="u")
    store.close()
    db = tmp_path / "portal.db"

    dry = migrate.migrate_down(db, dry_run=True, target=6).plan_text
    assert "schema v7 -> v6" in dry and "schema v6 -> v5" not in dry
    assert "campaign_clones -> campaign_clones_v7_rolledback (kept, not dropped; 1 row(s))" in dry
    assert "clone_titer_history -> clone_titer_history_v7_rolledback (kept, not dropped; 1 row(s))" in dry

    result = migrate.migrate_down(db, target=6)
    assert result.action == "rolled_back" and result.backup_path is not None
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(6,)]
    tables = _names(db, "table")
    assert not (tables & _V7_TABLES)
    assert {"campaign_clones_v7_rolledback", "clone_titer_history_v7_rolledback"} <= tables
    assert _q(db, "SELECT label, status FROM campaign_clones_v7_rolledback") == [("CHO-A12", "lead")]
    assert _q(db, "SELECT basis, value FROM clone_titer_history_v7_rolledback") == [("qP", 22.0)]
    assert not (_V7_TRIGGERS & _names(db, "trigger"))
    assert "clone_titer_history_label_idx" not in _names(db, "index")
    conn = migrate.connect(db)
    try:
        migrate._assert_v2_shape(conn)  # a v6 engine accepts it
    finally:
        conn.close()

    assert migrate.migrate_up(db).action == "migrated"
    assert _q(db, "SELECT COUNT(*) FROM campaign_clones") == [(0,)]
    assert _q(db, "SELECT COUNT(*) FROM clone_titer_history_v7_rolledback") == [(1,)]
    store = CampaignStore(tmp_path)
    assert clone_registry.list_clones(store, tenant="acme", campaign_id=cid) == []
    clone_registry.put_clone(store, tenant="acme", campaign_id=cid, label="CHO-B07", status="backup",
                             fields={}, updated_by="u")
    store.close()
    dry2 = migrate.migrate_down(db, dry_run=True, target=6).plan_text
    assert "campaign_clones -> campaign_clones_v7_rolledback_2 (kept, not dropped; 1 row(s))" in dry2
    migrate.migrate_down(db, target=6)
    assert {"campaign_clones_v7_rolledback_2", "clone_titer_history_v7_rolledback_2"} <= _names(db, "table")


def test_down_to_v5_passes_through_v7_and_v6(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    plan = migrate.migrate_down(db, dry_run=True, target=5).plan_text
    assert "schema v7 -> v6" in plan and "schema v6 -> v5" in plan
    migrate.migrate_down(db, target=5)
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(5,)]
    assert migrate.migrate_up(db).action == "migrated"
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(migrate.SCHEMA_VERSION,)]


def test_status_shows_v7_and_pending_from_v6(tmp_path):
    state = tmp_path / "s"
    db, _ = _make_v6(state)
    report, healthy = migrate.status_report(db)
    v = migrate.SCHEMA_VERSION
    assert healthy and f"schema version: v6 (this code: v{v})" in report and f"schema v6 -> v{v}" in report
    migrate.migrate_up(db)
    report, healthy = migrate.status_report(db)
    assert healthy and f"schema version: v{v}" in report and "shape check: ok" in report


@pytest.mark.parametrize(
    ("ddl", "missing"),
    [
        ("DROP TRIGGER clone_titer_history_no_update", "trigger clone_titer_history_no_update"),
        ("DROP TRIGGER clone_titer_history_no_delete", "trigger clone_titer_history_no_delete"),
        ("ALTER TABLE campaign_clones RENAME TO elsewhere", "campaign_clones"),
        ("DROP TABLE clone_titer_history", "clone_titer_history"),
    ],
)
def test_shape_check_catches_a_missing_v7_table_or_trigger(tmp_path, ddl, missing):
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
