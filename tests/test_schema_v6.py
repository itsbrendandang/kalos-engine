"""Schema v6: `campaign_settings` + append-only `gate_decisions` (docs/MIGRATIONS.md).

Covers the v5 -> v6 upgrade on a realistic v5 database (every existing row kept,
a verified `pre-v6` backup taken), the table constraints and append-only triggers,
the v6 -> v5 rollback (both tables renamed aside with their rows; a re-upgrade
creates fresh tables and never collides with the renamed ones), the plan text,
the shape check, and the tenant isolation of `kalos.portal.campaign_settings`.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

import pandas as pd
import pytest

from kalos.portal import campaign_settings, migrate
from kalos.portal.campaign import CampaignArchived, CampaignNotFound, CampaignStore

RECIPE = {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1, "mode": "explore", "reason": "x"}
_V6_TABLES = {"campaign_settings", "gate_decisions"}


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
    """Every row of every table (ordered), for before/after comparisons."""
    out = {}
    for t in sorted(_names(db, "table")):
        out[t] = sorted(_q(db, f'SELECT * FROM "{t}"'), key=repr)  # noqa: S608 - sqlite_master names
    return out


def _populate(state: Path) -> tuple[CampaignStore, dict[str, str]]:
    """Two tenants with real work: seeded campaigns, analyses, measured runs and a
    correction (run_results), an archived campaign, audit rows and a job row."""
    store = CampaignStore(state)
    ids: dict[str, str] = {}
    for i, tenant in enumerate(("acme", "globex")):
        store.seed(_sheet(seed=i), "lipase_titer", ["Methanol", "pH"], tenant=tenant,
                   analysis={"target": "lipase_titer", "tenant": tenant}, dataset=f"{tenant}.csv")
        ids[tenant] = store.default_campaign_id(tenant=tenant) or ""
        run = store.start([RECIPE], tenant=tenant)[0]
        store.set_result(run["id"], 3.0 + i, tenant=tenant)
        store.set_result(run["id"], 3.5 + i, tenant=tenant, reason="re-read the plate")
        store.record_audit(tenant=tenant, action="run.upload", outcome="ok", campaign_id=ids[tenant])
    old = store.create_campaign(tenant="acme", name="old")
    store.archive_campaign(tenant="acme", campaign_id=old["id"])
    ids["acme_archived"] = old["id"]
    now = time.time()
    store._conn.execute(
        "INSERT INTO jobs (id, tenant, campaign_id, kind, state, created_at, updated_at) "
        "VALUES (?, 'acme', ?, 'analysis', 'succeeded', ?, ?)",
        ("job_" + uuid.uuid4().hex, ids["acme"], now, now),
    )
    return store, ids


def _make_v5(state: Path) -> tuple[Path, dict[str, str]]:
    """A v5 database as a v5 engine leaves it: populated through the store, rolled
    back to v5 (through v7 -> v6), and the (empty) renamed-aside v6 and v7 tables
    dropped so no v6/v7 trace remains."""
    store, ids = _populate(state)
    store.close()
    db = state / "portal.db"
    migrate.migrate_down(db, target=5)
    for b in state.glob("portal.db.pre-*"):
        b.unlink()
    conn = sqlite3.connect(str(db))
    for t in ("campaign_settings_v6_rolledback", "gate_decisions_v6_rolledback",
              "campaign_clones_v7_rolledback", "clone_titer_history_v7_rolledback"):
        conn.execute(f'DROP TABLE "{t}"')
    conn.commit()
    conn.close()
    return db, ids


def _built_v5_tables(tmp: Path) -> set[str]:
    """The tables the v5 code itself creates on an empty database."""
    db = tmp / "ref.db"
    conn = migrate.connect(db)
    conn.execute("BEGIN IMMEDIATE")
    migrate._apply_up_in_txn(conn, None)
    migrate._apply_v3_in_txn(conn)
    migrate._apply_v4_in_txn(conn)
    migrate._apply_v5_in_txn(conn)
    conn.execute("COMMIT")
    conn.close()
    return _names(db, "table")


def _gate_row(cid: str, *, tenant: str = "acme", decision: str = "go", note: str = "pilot ok") -> tuple:
    return (tenant, cid, "pilot", decision, note, json.dumps({"run_ids": []}), "alice", "token", time.time())


_INSERT_GATE = (
    "INSERT INTO gate_decisions (tenant, campaign_id, gate, decision, note, evidence, "
    "decided_by, auth_method, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


# --- v5 -> v6 ------------------------------------------------------------------ #


def test_the_v5_fixture_has_exactly_the_v5_shape(tmp_path):
    db, _ = _make_v5(tmp_path / "s")
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(migrate.V5,)]
    assert _names(db, "table") == _built_v5_tables(tmp_path)
    assert not (_names(db, "table") & _V6_TABLES)


def test_v5_to_v6_keeps_every_row_and_takes_a_verified_backup(tmp_path):
    state = tmp_path / "s"
    db, ids = _make_v5(state)
    before = _snapshot(db)
    assert len(before["run_results"]) == 4 and len(before["audit_log"]) >= 2 and before["jobs"]

    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert "schema v5 -> v7" in plan and "campaign_settings" in plan and "gate_decisions" in plan
    assert "pre-v6" in plan and "jobs table" not in plan  # only the remaining step is listed
    assert _snapshot(db) == before  # dry run changed nothing

    result = migrate.migrate_up(db)
    assert result.action == "migrated"
    assert result.backup_path is not None and result.backup_path.name.startswith("portal.db.pre-v6-")
    assert _snapshot(result.backup_path) == before  # the backup is a real copy of the v5 db

    after = _snapshot(db)
    schema_rows = after.pop("schema_migrations")
    before_schema = before.pop("schema_migrations")
    assert [r[0] for r in schema_rows] == [2, 3, 4, 5, 6, 7]
    assert schema_rows[:4] == before_schema
    assert after.pop("campaign_settings") == [] and after.pop("gate_decisions") == []
    assert after.pop("campaign_clones") == [] and after.pop("clone_titer_history") == []
    after.pop("sqlite_sequence", None)
    before.pop("sqlite_sequence", None)
    assert after == before
    assert {"gate_decisions_no_update", "gate_decisions_no_delete"} <= _names(db, "trigger")
    assert "gate_decisions_campaign_idx" in _names(db, "index")

    store = CampaignStore(state)  # the engine starts on it
    assert store.get(tenant="acme")["pending"][0]["result"] == 3.5
    assert store.load_analysis(tenant="globex")["tenant"] == "globex"
    listed = {c["id"] for c in store.list_campaigns(tenant="acme", include_archived=True)}
    assert {ids["acme"], ids["acme_archived"]} <= listed
    assert migrate.migrate_up(db).action == "noop"


def test_fresh_store_is_created_at_v6(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    assert _q(db, "SELECT version, name FROM schema_migrations WHERE version = 6") == [(6, migrate.V6_NAME)]
    assert _names(db, "table") >= _V6_TABLES


def test_plan_text_from_v1_mentions_the_v6_tables(tmp_path):
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE campaigns (tenant TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL)")
    conn.commit()
    conn.close()
    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert "v2 -> v7" in plan and "campaign_settings and gate_decisions" in plan


# --- constraints --------------------------------------------------------------- #


@pytest.fixture
def v6(tmp_path):
    store = CampaignStore(tmp_path)
    c = store.create_campaign(tenant="acme", name="a")
    store.close()
    conn = migrate.connect(tmp_path / "portal.db")
    yield conn, c["id"]
    conn.close()


def test_gate_decisions_is_append_only(v6):
    conn, cid = v6
    conn.execute(_INSERT_GATE, _gate_row(cid))
    with pytest.raises(sqlite3.IntegrityError, match="append-only: UPDATE"):
        conn.execute("UPDATE gate_decisions SET decision = 'no_go'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only: DELETE"):
        conn.execute("DELETE FROM gate_decisions")
    assert conn.execute("SELECT decision, note FROM gate_decisions").fetchall() == [("go", "pilot ok")]


@pytest.mark.parametrize("note", ["", "   ", None])
def test_gate_decisions_requires_a_non_empty_note(v6, note):
    conn, cid = v6
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(_INSERT_GATE, _gate_row(cid, note=note))  # type: ignore[arg-type]


@pytest.mark.parametrize("decision", ["GO", "approved", "", "no-go"])
def test_gate_decisions_rejects_unknown_decisions(v6, decision):
    conn, cid = v6
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(_INSERT_GATE, _gate_row(cid, decision=decision))


@pytest.mark.parametrize("decision", ["go", "no_go", "pending"])
def test_gate_decisions_accepts_the_three_decisions(v6, decision):
    conn, cid = v6
    conn.execute(_INSERT_GATE, _gate_row(cid, decision=decision))


def test_gate_decisions_foreign_key_binds_campaign_to_tenant(v6):
    conn, cid = v6
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(_INSERT_GATE, _gate_row(cid, tenant="globex"))


def test_campaign_settings_foreign_key_and_kind_check(v6):
    conn, cid = v6
    ins = ("INSERT INTO campaign_settings (campaign_id, tenant, kind, value, updated_at, updated_by) "
           "VALUES (?, ?, ?, '{}', 1.0, NULL)")
    conn.execute(ins, (cid, "acme", "setup"))
    conn.execute(ins, (cid, "acme", "stages"))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(ins, (cid, "acme", "goal"))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(ins, (cid, "globex", "setup"))  # the id exists, but for another tenant
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(ins, ("cmp_" + uuid.uuid4().hex, "acme", "setup"))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        conn.execute(ins, (cid, "acme", "setup"))


# --- v6 -> v5 and back ------------------------------------------------------------ #


def test_down_renames_both_tables_aside_and_re_up_is_clean(tmp_path):
    store = CampaignStore(tmp_path)
    store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant="acme")
    cid = store.default_campaign_id(tenant="acme")
    assert cid is not None
    campaign_settings.put_setting(store, tenant="acme", campaign_id=cid, kind="setup",
                                  value={"goal": "maximize"}, updated_by="alice")
    store._conn.execute(_INSERT_GATE, _gate_row(cid))
    store.close()
    db = tmp_path / "portal.db"
    blob_before = _q(db, "SELECT state FROM campaigns WHERE id = ?", (cid,))

    dry = migrate.migrate_down(db, dry_run=True, target=5).plan_text
    assert "schema v7 -> v6" in dry and "schema v6 -> v5" in dry
    assert "campaign_settings -> campaign_settings_v6_rolledback (kept, not dropped; 1 row(s))" in dry
    assert "gate_decisions -> gate_decisions_v6_rolledback (kept, not dropped; 1 row(s))" in dry
    assert "schema v5 -> v4" not in dry  # stops at the target

    result = migrate.migrate_down(db, target=5)
    assert result.action == "rolled_back" and result.backup_path is not None
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(5,)]
    tables = _names(db, "table")
    assert not (tables & _V6_TABLES)
    assert {"campaign_settings_v6_rolledback", "gate_decisions_v6_rolledback"} <= tables
    assert _q(db, "SELECT kind, value FROM campaign_settings_v6_rolledback") == [
        ("setup", json.dumps({"goal": "maximize"}, sort_keys=True))]
    assert _q(db, "SELECT decision, note FROM gate_decisions_v6_rolledback") == [("go", "pilot ok")]
    assert not ({"gate_decisions_no_update", "gate_decisions_no_delete"} & _names(db, "trigger"))
    assert "gate_decisions_campaign_idx" not in _names(db, "index")
    assert _q(db, "SELECT state FROM campaigns WHERE id = ?", (cid,)) == blob_before
    conn = migrate.connect(db)
    try:
        migrate._assert_v2_shape(conn)  # a v5 engine accepts it
    finally:
        conn.close()

    # re-upgrade: fresh, empty v6 tables; the renamed-aside rows are untouched
    up = migrate.migrate_up(db)
    assert up.action == "migrated" and up.backup_path is not None
    assert _q(db, "SELECT COUNT(*) FROM campaign_settings") == [(0,)]
    assert _q(db, "SELECT COUNT(*) FROM gate_decisions") == [(0,)]
    assert _q(db, "SELECT COUNT(*) FROM gate_decisions_v6_rolledback") == [(1,)]
    store = CampaignStore(tmp_path)
    assert campaign_settings.get_setting(store, tenant="acme", campaign_id=cid, kind="setup") is None
    campaign_settings.put_setting(store, tenant="acme", campaign_id=cid, kind="setup",
                                  value={"goal": "minimize"}, updated_by="bob")
    store.close()

    # a second rollback never reuses the first one's names
    dry2 = migrate.migrate_down(db, dry_run=True, target=5).plan_text
    assert "campaign_settings -> campaign_settings_v6_rolledback_2 (kept, not dropped; 1 row(s))" in dry2
    migrate.migrate_down(db, target=5)
    assert {"campaign_settings_v6_rolledback_2", "gate_decisions_v6_rolledback_2"} <= _names(db, "table")
    assert _q(db, "SELECT COUNT(*) FROM campaign_settings_v6_rolledback") == [(1,)]
    assert migrate.migrate_up(db).action == "migrated"


def test_down_to_v4_passes_through_v6_and_v5(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    plan = migrate.migrate_down(db, dry_run=True, target=4).plan_text
    assert "schema v6 -> v5" in plan and "schema v5 -> v4" in plan
    migrate.migrate_down(db, target=4)
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(4,)]
    assert {"campaign_settings_v6_rolledback", "gate_decisions_v6_rolledback",
            "run_results_v5_rolledback"} <= _names(db, "table")
    assert migrate.migrate_up(db).action == "migrated"
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(migrate.SCHEMA_VERSION,)]


def test_status_shows_v6_and_pending_from_v5(tmp_path):
    state = tmp_path / "s"
    db, _ = _make_v5(state)
    report, healthy = migrate.status_report(db)
    assert healthy and "schema version: v5 (this code: v7)" in report and "schema v5 -> v7" in report
    migrate.migrate_up(db)
    report, healthy = migrate.status_report(db)
    assert healthy and "schema version: v7" in report and "shape check: ok" in report


# --- shape check ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ddl", "missing"),
    [
        ("DROP TRIGGER gate_decisions_no_update", "trigger gate_decisions_no_update"),
        ("DROP TRIGGER gate_decisions_no_delete", "trigger gate_decisions_no_delete"),
        ("ALTER TABLE campaign_settings RENAME TO elsewhere", "campaign_settings"),
        ("DROP TABLE gate_decisions", "gate_decisions"),
    ],
)
def test_shape_check_catches_a_missing_v6_table_or_trigger(tmp_path, ddl, missing):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(ddl)
    conn.commit()
    conn.close()
    conn = migrate.connect(db)
    try:
        with pytest.raises(migrate.SchemaError, match=missing):
            migrate._assert_v2_shape(conn)
    finally:
        conn.close()
    with pytest.raises(migrate.SchemaError):
        CampaignStore(tmp_path)  # the engine refuses to start
    report, healthy = migrate.status_report(db)
    assert not healthy and missing in report


# --- campaign_settings accessor: tenant isolation ----------------------------------- #


@pytest.fixture
def two_tenants(tmp_path):
    store = CampaignStore(tmp_path)
    a = store.create_campaign(tenant="acme", name="a")["id"]
    g = store.create_campaign(tenant="globex", name="g")["id"]
    yield store, a, g
    store.close()


def test_settings_round_trip_and_audit_in_the_same_transaction(two_tenants):
    store, a, _g = two_tenants
    assert campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="stages") is None
    out = campaign_settings.put_setting(
        store, tenant="acme", campaign_id=a, kind="stages", value={"stages": [{"name": "pilot"}]},
        updated_by="alice", audit_action="settings.stages.put", auth_method="token",
    )
    got = campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="stages")
    assert got == out and got["value"] == {"stages": [{"name": "pilot"}]} and got["updated_by"] == "alice"
    assert [e["action"] for e in store.audit_entries(tenant="acme")] == ["settings.stages.put"]
    campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="stages",
                                  value={"stages": []}, updated_by=None)
    got2 = campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="stages")
    assert got2 is not None and got2["value"] == {"stages": []} and got2["updated_by"] is None
    assert campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="setup") is None


def test_settings_foreign_campaign_is_not_found(two_tenants):
    store, a, g = two_tenants
    campaign_settings.put_setting(store, tenant="globex", campaign_id=g, kind="setup",
                                  value={"secret": 1}, updated_by="g")
    with pytest.raises(CampaignNotFound):
        campaign_settings.get_setting(store, tenant="acme", campaign_id=g, kind="setup")
    with pytest.raises(CampaignNotFound):
        campaign_settings.put_setting(store, tenant="acme", campaign_id=g, kind="setup",
                                      value={"x": 1}, updated_by="a")
    with pytest.raises(CampaignNotFound):
        campaign_settings.get_setting(store, tenant="acme", campaign_id="cmp_" + uuid.uuid4().hex, kind="setup")
    with pytest.raises(CampaignNotFound):
        campaign_settings.get_setting(store, tenant="acme", campaign_id="not-an-id", kind="setup")
    # globex's document is unchanged, and acme has nothing stored
    got = campaign_settings.get_setting(store, tenant="globex", campaign_id=g, kind="setup")
    assert got is not None and got["value"] == {"secret": 1}
    assert _q(store._path, "SELECT COUNT(*) FROM campaign_settings WHERE tenant = 'acme'") == [(0,)]
    assert campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="setup") is None


def test_settings_archived_campaign_refuses_writes_but_reads(two_tenants):
    store, a, _g = two_tenants
    campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="setup",
                                  value={"goal": "max"}, updated_by="a")
    store.archive_campaign(tenant="acme", campaign_id=a)
    with pytest.raises(CampaignArchived):
        campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="setup",
                                      value={"goal": "min"}, updated_by="a", audit_action="settings.setup.put")
    got = campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="setup")
    assert got is not None and got["value"] == {"goal": "max"}
    assert store.audit_entries(tenant="acme") == []  # nothing audited for the refused write


def test_settings_too_large_and_bad_input_are_refused(two_tenants):
    store, a, _g = two_tenants
    big = {"x": "a" * campaign_settings.MAX_SETTING_JSON_CHARS}
    with pytest.raises(campaign_settings.SettingTooLarge):
        campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="setup",
                                      value=big, updated_by="a")
    with pytest.raises(ValueError, match="unknown settings kind"):
        campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="goal",
                                      value={}, updated_by="a")
    with pytest.raises(ValueError):
        campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="setup",
                                      value={"x": float("nan")}, updated_by="a")
    with pytest.raises(ValueError, match="JSON object"):
        campaign_settings.put_setting(store, tenant="acme", campaign_id=a, kind="setup",
                                      value=[1, 2], updated_by="a")  # type: ignore[arg-type]
    assert campaign_settings.get_setting(store, tenant="acme", campaign_id=a, kind="setup") is None
