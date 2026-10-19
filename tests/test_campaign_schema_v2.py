"""portal.db schema v2: multi-campaign storage and the v1 -> v2 migration.

Covers the migration (backup first, one transaction, nothing lost, legacy
`latest/` import rules, dry run, idempotence, rollback to the v1 shape), the
store's isolation rules (tenant on every statement, cross-tenant ids read as
missing, archived campaigns refuse writes), its concurrency rules (BEGIN
IMMEDIATE + compare-and-swap across threads and connections), and that the
legacy single-campaign routes map onto the tenant's default campaign.

Every database here lives in a pytest tmp dir.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import sqlite3
import threading
from pathlib import Path

import pandas as pd
import pytest

from kalos.portal import migrate
from kalos.portal.campaign import (
    CampaignArchived,
    CampaignConflict,
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    _campaigns_table_is_compatible,
)

# --- fixtures / helpers ------------------------------------------------------ #

V1_DDL = "CREATE TABLE campaigns (tenant TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at REAL NOT NULL)"


def _v1_state(seed: int, *, generation: str | None = "gen-seed") -> dict:
    """A realistic v1 state blob, mid-campaign: one awaiting and one measured
    pending run, two rounds of history."""
    base = [{"Methanol": 1.0 + i + seed, "pH": 6.0, "lipase_titer": 2.0 + i} for i in range(5)]
    state = {
        "target": "lipase_titer",
        "features": ["Methanol", "pH"],
        "base_rows": base,
        "pending": [
            {"id": f"run-awaiting-{seed}", "recipe": {"Methanol": 2.5, "pH": 6.5}, "pred": 4.0,
             "std": 0.2, "mode": "explore", "reason": "gap", "result": None,
             "created_at": 1000.0, "measured_at": None},
            {"id": f"run-measured-{seed}", "recipe": {"Methanol": 3.5, "pH": 6.1}, "pred": 5.0,
             "std": 0.3, "mode": "exploit", "reason": "best", "result": 5.5,
             "created_at": 1001.0, "measured_at": 1002.0},
        ],
        "round": 2,
        "history": [{"round": 0, "best": 6.0, "n_base": 3}, {"round": 1, "best": 6.0, "n_base": 4},
                    {"round": 2, "best": 6.0, "n_base": 5}],
        "updated_at": 1002.0,
    }
    if generation is not None:
        state["generation"] = generation
    return state


def _make_v1(state_dir: Path, rows: list[tuple[str, str, float]]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    db = state_dir / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute(V1_DDL)
    conn.executemany("INSERT INTO campaigns (tenant, state, updated_at) VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return db


def _standard_v1(state_dir: Path) -> tuple[Path, dict[str, str]]:
    blobs = {
        "acme": json.dumps(_v1_state(1, generation="gen-acme")),
        "globex": json.dumps(_v1_state(2, generation=None)),  # pre-generation legacy row
    }
    db = _make_v1(state_dir, [("acme", blobs["acme"], 1500.0), ("globex", blobs["globex"], 1600.0)])
    return db, blobs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _tables(db: Path) -> set[str]:
    return {r[0] for r in _q(db, "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}


def _sheet(n: int = 8, seed: int = 0) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2).round(3)})


def _seeded(store: CampaignStore, tenant: str = "acme", **kw) -> str:
    return store.seed(_sheet(), "lipase_titer", ["Methanol", "pH"], tenant=tenant, **kw)


RECIPE = {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1, "mode": "explore", "reason": "x"}


# --- fresh database ---------------------------------------------------------- #


def test_fresh_database_is_created_at_v2_without_a_backup(tmp_path):
    store = CampaignStore(tmp_path)
    db = tmp_path / "portal.db"
    assert {"schema_migrations", "campaigns", "analyses", "tenant_defaults", "audit_log"} <= _tables(db)
    # schema v3 (background jobs) is applied on top of v2 in the same transaction
    assert _q(db, "SELECT version, name FROM schema_migrations ORDER BY version") == [
        (2, migrate.V2_NAME), (3, migrate.V3_NAME)]
    assert not list(tmp_path.glob("portal.db.pre-*"))
    assert store.get(tenant="acme") is None
    assert store.summary(tenant="acme") == {"has_campaign": False}


def test_wal_and_connection_pragmas(tmp_path):
    store = CampaignStore(tmp_path)
    conn = store._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == migrate.BUSY_TIMEOUT_MS
    # the composite FK really binds an analysis to its campaign's tenant
    cid = _seeded(store) and store.default_campaign_id(tenant="acme")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO analyses (campaign_id, tenant, state, updated_at) VALUES (?, 'globex', '{}', 0)",
            (cid,),
        )


# --- v1 -> v2 migration -------------------------------------------------------- #


def test_v1_migration_keeps_every_byte_of_campaign_state(tmp_path, caplog):
    db, blobs = _standard_v1(tmp_path)
    with caplog.at_level("WARNING"):
        store = CampaignStore(tmp_path)

    rows = _q(db, "SELECT tenant, name, state, created_by, created_at, updated_at, archived_at, id, generation "
                  "FROM campaigns ORDER BY tenant")
    assert [r[0] for r in rows] == ["acme", "globex"]
    for tenant, name, state, created_by, created_at, updated_at, archived_at, cid, generation in rows:
        assert state == blobs[tenant]  # verbatim, not re-serialized
        assert name == migrate.IMPORTED_CAMPAIGN_NAME
        assert created_by == migrate.MIGRATION_ACTOR
        assert created_at == updated_at == {"acme": 1500.0, "globex": 1600.0}[tenant]
        assert archived_at is None
        assert re.fullmatch(r"cmp_[0-9a-f]{32}", cid)
        assert generation == {"acme": "gen-acme", "globex": None}[tenant]
    defaults = dict(_q(db, "SELECT tenant, campaign_id FROM tenant_defaults"))
    assert defaults == {r[0]: r[7] for r in rows}

    # the v1 table is kept, untouched, under a new name
    assert sorted(_q(db, "SELECT tenant, state, updated_at FROM campaigns_v1")) == [
        ("acme", blobs["acme"], 1500.0), ("globex", blobs["globex"], 1600.0)]

    # the store serves exactly the old state, pending runs included
    assert store.get(tenant="acme") == json.loads(blobs["acme"])
    assert store.get(tenant="globex") == json.loads(blobs["globex"])
    summary = store.summary(tenant="acme")
    assert summary["n_awaiting"] == 1 and summary["n_measured"] == 1 and summary["round"] == 2
    assert store.awaiting_recipes(tenant="acme") == [{"Methanol": 2.5, "pH": 6.5}]

    # and the loop continues on the migrated state: a legacy (generation-less)
    # campaign folds too, minting a real generation
    _df, _t, gen = store.plan_fold(tenant="globex")
    assert gen is None
    state = store.commit_fold(gen, tenant="globex")
    assert state["round"] == 3 and len(state["base_rows"]) == 6
    assert store.get(tenant="globex")["generation"] == state["generation"]


def test_backup_is_written_and_restorable(tmp_path):
    db, blobs = _standard_v1(tmp_path / "state")
    CampaignStore(tmp_path / "state")
    backups = list((tmp_path / "state").glob("portal.db.pre-v2-*"))
    assert len(backups) == 1
    assert not list((tmp_path / "state").glob("*.tmp"))

    # restore into a NEW path and read it as the v1 code would
    restored = tmp_path / "restored" / "portal.db"
    restored.parent.mkdir()
    shutil.copy2(backups[0], restored)
    conn = sqlite3.connect(str(restored))
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert _campaigns_table_is_compatible(conn)
    assert dict(conn.execute("SELECT tenant, state FROM campaigns").fetchall()) == blobs
    assert "schema_migrations" not in {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()
    # and the restored copy migrates again cleanly
    assert CampaignStore(restored.parent).get(tenant="acme") == json.loads(blobs["acme"])


def test_failed_backup_aborts_and_leaves_the_database_untouched(tmp_path, monkeypatch):
    db, _ = _standard_v1(tmp_path)
    before = _sha(db)

    def _boom(src, dest):
        Path(dest).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(migrate, "_online_backup", _boom)
    with pytest.raises(migrate.MigrationError, match="backup"):
        CampaignStore(tmp_path)
    assert _sha(db) == before
    assert _tables(db) == {"campaigns"}
    assert not list(tmp_path.glob("portal.db.pre-*"))  # the partial file was removed
    assert not (tmp_path / "portal.db-wal").exists()


def test_corrupt_v1_row_is_kept_verbatim(tmp_path, caplog):
    good = json.dumps(_v1_state(1))
    db = _make_v1(tmp_path, [("acme", "{not json", 10.0), ("globex", good, 20.0)])
    with caplog.at_level("WARNING"):
        store = CampaignStore(tmp_path)
    assert any("not valid JSON" in r.getMessage() for r in caplog.records)
    assert _q(db, "SELECT state FROM campaigns WHERE tenant = 'acme'") == [("{not json",)]
    # a corrupt row reads as "no campaign" instead of breaking the API
    assert store.get(tenant="acme") is None
    assert store.summary(tenant="acme") == {"has_campaign": False}
    assert store.get(tenant="globex") == json.loads(good)
    # ... and survives a rollback byte-for-byte
    migrate.migrate_down(db, target=1)
    assert _q(db, "SELECT state FROM campaigns WHERE tenant = 'acme'") == [("{not json",)]


def test_regex_failing_legacy_tenant_is_imported_unchanged(tmp_path, caplog):
    odd = "acme/bio.lab"  # fails ^[A-Za-z0-9_-]{1,64}$
    blob = json.dumps(_v1_state(3))
    db = _make_v1(tmp_path, [(odd, blob, 5.0), ("acme_bio", json.dumps(_v1_state(4)), 6.0)])
    with caplog.at_level("WARNING"):
        store = CampaignStore(tmp_path)
    assert any("fails the tenant-id pattern" in r.getMessage() for r in caplog.records)
    # stored under its exact original id, never renamed or merged into acme_bio
    assert _q(db, "SELECT state FROM campaigns WHERE tenant = ?", (odd,)) == [(blob,)]
    assert len(_q(db, "SELECT 1 FROM tenant_defaults WHERE tenant = ?", (odd,))) == 1
    assert store.get(tenant="acme_bio")["base_rows"][0]["Methanol"] == 5.0
    # unreachable through the API (auth and the store reject the id)...
    with pytest.raises(ValueError):
        store.get(tenant=odd)
    # ...and restored as-is by a rollback
    migrate.migrate_down(db, target=1)
    assert _q(db, "SELECT state FROM campaigns WHERE tenant = ?", (odd,)) == [(blob,)]


# --- legacy latest/ files ------------------------------------------------------ #


def _write_latest(latest: Path, name: str, payload) -> None:
    latest.mkdir(parents=True, exist_ok=True)
    (latest / name).write_text(payload if isinstance(payload, str) else json.dumps(payload))


def test_latest_files_import_only_when_unambiguous(tmp_path, caplog):
    db = _make_v1(tmp_path, [
        ("acme", json.dumps(_v1_state(1, generation="gen-acme")), 10.0),
        ("acme_bio", json.dumps(_v1_state(2)), 11.0),
    ])
    latest = tmp_path / "latest"
    hexname = migrate.legacy_latest_filename
    _write_latest(latest, hexname("acme"), {"target": "lipase_titer", "dataset": "acme.csv",
                                            "updated": 42.0, "campaign_generation": "gen-acme"})
    _write_latest(latest, "acme.json", {"target": "stale", "dataset": "old.csv"})  # superseded
    _write_latest(latest, "acme_bio.json", {"target": "?", "dataset": "who.csv"})  # ambiguous
    _write_latest(latest, hexname("orphan"), {"target": "t", "dataset": "o.csv"})  # no campaign
    _write_latest(latest, "legacy.json", {"target": "t", "dataset": "l.csv"})  # pre-#86, no '_'
    _write_latest(latest, hexname("broken"), "{nope")  # corrupt
    files_before = sorted(p.name for p in latest.iterdir())

    with caplog.at_level("WARNING"):
        store = CampaignStore(tmp_path)

    # unambiguous hex file -> the tenant's default campaign, stamp preserved
    acme = store.load_analysis(tenant="acme")
    assert acme["dataset"] == "acme.csv" and acme["updated"] == 42.0
    assert store.summary(tenant="acme", analysis_generation=acme["campaign_generation"])["analysis_in_sync"]
    # ambiguous pre-#86 name -> skipped with a warning, never guessed
    assert store.load_analysis(tenant="acme_bio") is None
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "acme_bio.json not imported: ambiguous" in msgs
    assert "acme.json not imported: superseded" in msgs
    assert "not imported: unreadable or corrupt JSON" in msgs
    # missing campaign -> an UNSEEDED default campaign owns the analysis, so
    # /api/latest keeps serving it and /api/campaign still says no campaign
    assert store.load_analysis(tenant="orphan")["dataset"] == "o.csv"
    assert store.summary(tenant="orphan") == {"has_campaign": False}
    assert store.load_analysis(tenant="legacy")["dataset"] == "l.csv"
    assert store.load_analysis(tenant="broken") is None
    # nothing on disk was moved, renamed or deleted
    assert sorted(p.name for p in latest.iterdir()) == files_before
    assert _q(db, "SELECT COUNT(*) FROM analyses") == [(3,)]


def test_classify_treats_a_hex_name_that_is_also_a_tenant_as_ambiguous(tmp_path):
    latest = tmp_path / "latest"
    _write_latest(latest, "t-61.json", {"x": 1})  # hex("a") == "61", but "t-61" is a real tenant
    imports, skipped = migrate.classify_latest_files(latest, {"t-61"})
    assert imports == [] and "ambiguous" in skipped[0].reason
    imports, skipped = migrate.classify_latest_files(latest, set())
    assert [i.tenant for i in imports] == ["a"]


# --- dry run, idempotence, status ------------------------------------------------ #


def test_dry_run_changes_nothing_and_prints_a_plan(tmp_path, capsys):
    db, _ = _standard_v1(tmp_path)
    _write_latest(tmp_path / "latest", "acme_x.json", {"a": 1})
    before = _sha(db)
    result = migrate.migrate_up(db, dry_run=True)
    assert result.action == "dry_run"
    assert "schema v1 -> v2" in result.plan_text
    assert "2 v1 row(s)" in result.plan_text
    assert "skip: latest/acme_x.json" in result.plan_text
    assert migrate.main(["up", "--dry-run", "--db", str(db)]) == 0
    assert "schema v1 -> v2" in capsys.readouterr().out
    assert migrate.main(["status", "--db", str(db)]) == 0
    assert "pending migration" in capsys.readouterr().out
    assert _sha(db) == before
    assert _tables(db) == {"campaigns"}
    assert not list(tmp_path.glob("portal.db.pre-*"))
    assert not (tmp_path / "portal.db-wal").exists()


def test_migration_is_idempotent(tmp_path):
    db, blobs = _standard_v1(tmp_path)
    first = migrate.migrate_up(db)
    assert first.action == "migrated" and first.backup_path is not None
    ids = _q(db, "SELECT id FROM campaigns ORDER BY id")
    assert migrate.migrate_up(db).action == "noop"
    CampaignStore(tmp_path)
    CampaignStore(tmp_path)
    assert _q(db, "SELECT id FROM campaigns ORDER BY id") == ids
    assert _q(db, "SELECT version FROM schema_migrations ORDER BY version") == [(2,), (3,)]
    assert len(list(tmp_path.glob("portal.db.pre-v2-*"))) == 1
    assert migrate.migrate_up(db, dry_run=True).plan_text.endswith("nothing to do")


def test_newer_schema_is_refused(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO schema_migrations VALUES (4, 'future', 0)")
    conn.commit()
    conn.close()
    with pytest.raises(migrate.SchemaError, match="newer"):
        CampaignStore(tmp_path)


# --- down ------------------------------------------------------------------- #


def test_down_restores_the_v1_shape_the_old_code_reads(tmp_path):
    db, blobs = _standard_v1(tmp_path)
    store = CampaignStore(tmp_path)
    # work done on v2 after the migration must survive the rollback
    started = store.start([RECIPE], tenant="acme")
    store.set_result(started[0]["id"], 7.0, tenant="acme")
    acme_now = store.get(tenant="acme")
    _seeded(store, tenant="initech", analysis={"target": "lipase_titer"}, dataset="i.csv")
    # a second, newer non-default campaign must NOT replace the default
    other = store.create_campaign(tenant="acme", name="second")
    store.seed(_sheet(seed=9), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=other["id"])
    store.close()

    plan = migrate.migrate_down(db, dry_run=True, target=1)
    assert "schema v3 -> v2" in plan.plan_text and "schema v2 -> v1" in plan.plan_text
    result = migrate.migrate_down(db, target=1)
    assert result.action == "rolled_back" and result.backup_path is not None

    conn = sqlite3.connect(str(db))
    assert _campaigns_table_is_compatible(conn)
    rows = {t: json.loads(s) for t, s in conn.execute("SELECT tenant, state FROM campaigns")}
    assert set(rows) == {"acme", "globex", "initech"}
    assert rows["acme"] == acme_now  # the default campaign, with the v2-era result
    assert rows["globex"] == json.loads(blobs["globex"])
    # the v1 code's own statements work against it
    conn.execute(
        "INSERT INTO campaigns (tenant, state, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(tenant) DO UPDATE SET state = excluded.state, updated_at = excluded.updated_at",
        ("acme", "{}", 1.0),
    )
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"campaigns_v2_rolledback", "analyses_v2_rolledback", "tenant_defaults_v2_rolledback",
            "audit_log_v2_rolledback", "schema_migrations_v2_rolledback", "campaigns_v1"} <= tables
    assert "schema_migrations" not in tables
    # the analysis is handed back as the v1 latest/ file
    exported = json.loads((tmp_path / "latest" / migrate.legacy_latest_filename("initech")).read_text())
    assert exported["dataset"] == "i.csv"
    # and the next v2 start migrates again without clobbering anything
    again = CampaignStore(tmp_path)
    assert again.get(tenant="globex") == json.loads(blobs["globex"])
    assert "campaigns_v1_2" in _tables(db)
    assert migrate.migrate_down(db, dry_run=True).plan_text.startswith(str(db))


def test_down_moves_an_existing_latest_file_aside(tmp_path):
    store = CampaignStore(tmp_path)
    _seeded(store, analysis={"target": "new"}, dataset="new.csv")
    store.close()
    path = tmp_path / "latest" / migrate.legacy_latest_filename("acme")
    _write_latest(path.parent, path.name, {"target": "old"})
    migrate.migrate_down(tmp_path / "portal.db", target=1)
    assert json.loads(path.read_text())["target"] == "new"
    aside = list(path.parent.glob(path.name + ".pre-down-*"))
    assert len(aside) == 1 and json.loads(aside[0].read_text())["target"] == "old"


def test_cli_up_down_status(tmp_path, capsys):
    _standard_v1(tmp_path)
    assert migrate.main(["up", "--state-dir", str(tmp_path)]) == 0
    assert "result: migrated" in capsys.readouterr().out
    assert migrate.main(["status", "--state-dir", str(tmp_path)]) == 0
    assert "schema version: v3" in capsys.readouterr().out
    assert migrate.main(["down", "--to", "1", "--state-dir", str(tmp_path)]) == 0
    assert "result: rolled_back" in capsys.readouterr().out


# --- isolation --------------------------------------------------------------- #


def test_cross_tenant_campaign_id_is_indistinguishable_from_a_missing_one(tmp_path):
    store = CampaignStore(tmp_path)
    acme = store.create_campaign(tenant="acme", name="secret project")
    _seeded(store, campaign_id=acme["id"])
    before = store.get(tenant="acme", campaign_id=acme["id"])
    missing = "cmp_" + "0" * 32

    errors = []
    for cid in (acme["id"], missing, "not-an-id", "cmp_" + "A" * 32):
        for call in (
            lambda c: store.get(tenant="globex", campaign_id=c),
            lambda c: store.campaign_info(tenant="globex", campaign_id=c),
            lambda c: store.start([RECIPE], tenant="globex", campaign_id=c),
            lambda c: store.archive_campaign(tenant="globex", campaign_id=c),
            lambda c: store.load_analysis(tenant="globex", campaign_id=c),
            lambda c: store.save_analysis({}, "x", tenant="globex", campaign_id=c),
        ):
            with pytest.raises(CampaignNotFound) as exc:
                call(cid)
            errors.append(str(exc.value))
    assert set(errors) == {"campaign not found"}
    assert store.get(tenant="acme", campaign_id=acme["id"]) == before
    assert store.list_campaigns(tenant="globex") == []


def test_malformed_campaign_id_never_reaches_sql(tmp_path):
    store = CampaignStore(tmp_path)
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    with pytest.raises(CampaignNotFound):
        store.get(tenant="acme", campaign_id="cmp_x' OR 1=1 --")
    assert seen == []


def test_archived_campaign_rejects_writes_with_409(tmp_path):
    store = CampaignStore(tmp_path)
    c = store.create_campaign(tenant="acme", name="done")
    _seeded(store, campaign_id=c["id"])
    run = store.start([RECIPE], tenant="acme", campaign_id=c["id"])[0]
    store.set_result(run["id"], 3.0, tenant="acme", campaign_id=c["id"])
    _df, _t, gen = store.plan_fold(tenant="acme", campaign_id=c["id"])
    info = store.archive_campaign(tenant="acme", campaign_id=c["id"])
    assert info["archived"] is True
    snapshot = store.get(tenant="acme", campaign_id=c["id"])

    cid = c["id"]
    for write in (
        lambda: _seeded(store, campaign_id=cid),
        lambda: store.start([RECIPE], tenant="acme", campaign_id=cid),
        lambda: store.set_result(run["id"], 4.0, tenant="acme", campaign_id=cid),
        lambda: store.plan_fold(tenant="acme", campaign_id=cid),
        lambda: store.commit_fold(gen, tenant="acme", campaign_id=cid),
        lambda: store.save_analysis({}, "x", tenant="acme", campaign_id=cid),
    ):
        with pytest.raises(CampaignArchived):
            write()
    assert store.get(tenant="acme", campaign_id=cid) == snapshot  # still readable, unchanged
    assert store.list_campaigns(tenant="acme") == []
    assert [x["id"] for x in store.list_campaigns(tenant="acme", include_archived=True)] == [cid]


def test_archiving_the_default_lets_the_next_upload_start_a_new_default(tmp_path):
    store = CampaignStore(tmp_path)
    _seeded(store)
    old = store.default_campaign_id(tenant="acme")
    store.archive_campaign(tenant="acme", campaign_id=old)
    assert store.summary(tenant="acme") == {"has_campaign": False}
    _seeded(store)
    new = store.default_campaign_id(tenant="acme")
    assert new != old and store.summary(tenant="acme")["has_campaign"] is True


def test_for_tenant_view_binds_the_tenant(tmp_path):
    store = CampaignStore(tmp_path)
    acme = store.for_tenant("acme")
    acme.seed(_sheet(), "lipase_titer", ["Methanol", "pH"])
    assert acme.summary()["has_campaign"] is True
    assert store.summary(tenant="globex") == {"has_campaign": False}
    with pytest.raises(AttributeError):
        acme._conn  # noqa: B018 - private store internals are not exposed
    with pytest.raises(ValueError):
        store.for_tenant("../etc")


def test_audit_log_is_tenant_scoped(tmp_path):
    store = CampaignStore(tmp_path)
    store.record_audit(tenant="acme", action="campaign.start", outcome="ok", subject="alice",
                       auth_method="bearer", detail="runs=1")
    store.record_audit(tenant="globex", action="run.upload", outcome="http_400")
    (entry,) = store.audit_entries(tenant="acme")
    assert entry["subject"] == "alice" and entry["action"] == "campaign.start"
    assert [e["action"] for e in store.audit_entries(tenant="globex")] == ["run.upload"]


# --- every statement on a tenant table names the tenant ------------------------ #

_TENANT_TABLE_RE = re.compile(r"\b(" + "|".join(migrate.TENANT_TABLES) + r")\b")


def _strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def test_every_traced_statement_on_a_tenant_table_mentions_tenant(tmp_path):
    store = CampaignStore(tmp_path)
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)

    gen = _seeded(store, analysis={"target": "lipase_titer"}, dataset="a.csv")
    store.summary(tenant="acme", analysis_generation=gen)
    run = store.start([RECIPE], tenant="acme")[0]
    store.set_result(run["id"], 3.0, tenant="acme")
    store.awaiting_recipes(tenant="acme")
    _df, _t, g = store.plan_fold(tenant="acme")
    store.commit_fold(g, tenant="acme", analysis={"target": "lipase_titer"})
    store.save_analysis({"target": "x"}, "b.csv", tenant="acme", campaign_generation=None)
    store.load_analysis(tenant="acme")
    c = store.create_campaign(tenant="acme", name="two", make_default=True)
    store.campaign_info(tenant="acme", campaign_id=c["id"])
    store.list_campaigns(tenant="acme", include_archived=True)
    store.default_campaign_id(tenant="acme")
    store.archive_campaign(tenant="acme", campaign_id=c["id"])
    store.record_audit(tenant="acme", action="a", outcome="ok")
    store.audit_entries(tenant="acme")
    store.ping()
    store._conn.set_trace_callback(None)

    touching = [s for s in seen if _TENANT_TABLE_RE.search(_strip_literals(s))]
    assert len(touching) > 20  # the trace really captured the store's traffic
    offenders = [s for s in touching if "tenant" not in _strip_literals(s)]
    assert offenders == []


def _sql_strings(source: str) -> list[str]:
    """Every string literal (f-string constant parts joined) in a module."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            out.append("".join(v.value for v in node.values
                               if isinstance(v, ast.Constant) and isinstance(v.value, str)))
    return out


def test_store_source_never_queries_a_tenant_table_without_tenant():
    import kalos.portal.campaign as campaign_module

    source = Path(campaign_module.__file__).read_text()
    sql = [s for s in _sql_strings(source)
           if re.match(r"\s*(SELECT|INSERT|UPDATE|DELETE)\b", s) and _TENANT_TABLE_RE.search(s)]
    assert len(sql) >= 8
    assert [s for s in sql if "tenant" not in s] == []


# --- concurrency ------------------------------------------------------------- #


def test_concurrent_writers_across_threads_and_connections_lose_nothing(tmp_path):
    a = CampaignStore(tmp_path)
    b = CampaignStore(tmp_path)  # a second connection to the same file
    _seeded(a)
    per_thread, threads = 10, 8
    errors: list[BaseException] = []

    def worker(store: CampaignStore) -> None:
        try:
            for _ in range(per_thread):
                store.start([RECIPE], tenant="acme")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(a if i % 2 else b,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []
    pending = a.get(tenant="acme")["pending"]
    assert len(pending) == per_thread * threads
    assert len({r["id"] for r in pending}) == per_thread * threads


def test_compare_and_swap_rejects_a_stale_fold_from_another_connection(tmp_path):
    a = CampaignStore(tmp_path)
    b = CampaignStore(tmp_path)
    _seeded(a)
    run = a.start([RECIPE], tenant="acme")[0]
    a.set_result(run["id"], 3.0, tenant="acme")
    _df, _t, gen = a.plan_fold(tenant="acme")
    b.start([RECIPE], tenant="acme")  # a write through the other connection
    with pytest.raises(CampaignConflict):
        a.commit_fold(gen, tenant="acme", analysis={"target": "stale"})
    assert a.load_analysis(tenant="acme") is None  # the analysis did not land either
    assert a.get(tenant="acme")["round"] == 0


def test_racing_commits_of_one_plan_commit_exactly_once(tmp_path):
    a = CampaignStore(tmp_path)
    b = CampaignStore(tmp_path)
    _seeded(a)
    run = a.start([RECIPE], tenant="acme")[0]
    a.set_result(run["id"], 3.0, tenant="acme")
    _df, _t, gen = a.plan_fold(tenant="acme")
    outcomes: list[str] = []
    barrier = threading.Barrier(4)

    def commit(store: CampaignStore) -> None:
        barrier.wait()
        try:
            store.commit_fold(gen, tenant="acme")
            outcomes.append("ok")
        except CampaignError:
            outcomes.append("conflict")

    ts = [threading.Thread(target=commit, args=(s,)) for s in (a, b, a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(outcomes) == ["conflict", "conflict", "conflict", "ok"]
    state = a.get(tenant="acme")
    assert state["round"] == 1 and len(state["base_rows"]) == len(_sheet()) + 1


# --- legacy routes map to the default campaign ---------------------------------- #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.campaign import get_campaign_store  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app), store
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


def test_legacy_routes_map_to_the_tenant_default_campaign(client):
    tc, store = client
    assert tc.get("/api/latest").json() == {"has_data": False}
    assert store.default_campaign_id(tenant="default") is None

    r = tc.post("/api/run", files={"file": ("runs.csv", _sheet(n=10).to_csv(index=False), "text/csv")},
                data={"target": "lipase_titer"})
    assert r.status_code == 200, r.text
    cid = store.default_campaign_id(tenant="default")
    assert cid is not None  # created on first upload
    (info,) = store.list_campaigns(tenant="default")
    assert info["id"] == cid and info["is_default"] and info["seeded"]

    latest = tc.get("/api/latest").json()
    assert latest["has_data"] is True and latest["dataset"] == "runs.csv"
    assert store.load_analysis(tenant="default", campaign_id=cid)["dataset"] == "runs.csv"
    assert tc.get("/api/campaign").json()["analysis_in_sync"] is True

    started = tc.post("/api/campaign/start", json={"recipes": [RECIPE]}).json()["started"]
    assert tc.post("/api/campaign/result", json={"id": started[0]["id"], "value": 6.0}).status_code == 200
    assert store.get(tenant="default", campaign_id=cid)["pending"][0]["result"] == 6.0
    # a second upload reuses the same default campaign
    tc.post("/api/run", files={"file": ("again.csv", _sheet(n=10, seed=3).to_csv(index=False), "text/csv")},
            data={"target": "lipase_titer"})
    assert store.default_campaign_id(tenant="default") == cid
    assert len(store.list_campaigns(tenant="default")) == 1

    actions = [(e["action"], e["outcome"]) for e in store.audit_entries(tenant="default")]
    assert ("run.upload", "ok") in actions and ("campaign.start", "ok") in actions
    assert all(e["campaign_id"] in (None, cid) for e in store.audit_entries(tenant="default"))


def test_archived_campaign_maps_to_409_on_legacy_routes(client, monkeypatch):
    tc, store = client

    def _archived(*_a, **_k):
        raise CampaignArchived()

    monkeypatch.setattr(store, "start", _archived)
    monkeypatch.setattr(store, "set_result", _archived)
    r = tc.post("/api/campaign/start", json={"recipes": [RECIPE]})
    assert r.status_code == 409 and "archived" in r.json()["error"]
    assert tc.post("/api/campaign/result", json={"id": "x", "value": 1.0}).status_code == 409


def test_readyz_does_not_read_tenant_tables(client):
    tc, store = client
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    assert tc.get("/readyz").json() == {"status": "ok"}
    store._conn.set_trace_callback(None)
    assert seen and not any(_TENANT_TABLE_RE.search(s) for s in seen)


def test_startup_migrates_and_a_failed_backup_stops_the_app(tmp_path, monkeypatch):
    """The lifespan hook opens the store, so a v1 database is migrated before the
    first request, and a failed backup stops startup instead of serving 500s."""
    state = tmp_path / "state"
    db, blobs = _standard_v1(state)
    monkeypatch.setenv("KALOS_STATE_DIR", str(state))
    monkeypatch.setattr(campaign_module, "_STORE", None)
    before = _sha(db)

    def _boom(src, dest):
        raise OSError("read-only filesystem")

    with monkeypatch.context() as m:
        m.setattr(migrate, "_online_backup", _boom)
        with pytest.raises(migrate.MigrationError):
            with TestClient(portal_module.app):
                pass
    assert _sha(db) == before and campaign_module._STORE is None

    with TestClient(portal_module.app) as tc:
        assert tc.get("/readyz").json() == {"status": "ok"}
    assert _q(db, "SELECT version FROM schema_migrations ORDER BY version") == [(2,), (3,)]
    assert campaign_module._STORE.get(tenant="acme") == json.loads(blobs["acme"])
