"""Schema v8: append-only prediction history (BRE-32).

Covers a fresh database at v8; the three tables, their constraints and the
append-only triggers; the v7 -> v8 upgrade on a COPY of a real v7 database
(`tests/fixtures/portal_v7.db`, built by running the v7 engine over HTTP, see
`tests/fixtures/make_portal_v7_db.py`): a verified `pre-v8` backup, every v7 row
kept byte for byte, the published analyses and the in-flight runs' recorded
predictions copied (with THEIR timestamps), nothing invented for runs v7 already
folded; the engine then keeps going on the migrated file and stops losing
history at the next fold; the v8 -> v7 rollback renames aside, and a re-upgrade
is clean. The fixture file itself is never opened for writing.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest

from kalos.portal import migrate
from kalos.portal.campaign import CampaignStore
from kalos.portal.prediction_history import prediction_columns, sha256_text

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "portal_v7.db"
_REPO = Path(__file__).resolve().parents[1]
V8_TABLES = {"analysis_snapshots", "run_predictions", "run_folds"}
V8_TRIGGERS = {f"{t}_no_{op}" for t in V8_TABLES for op in ("update", "delete")}


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
def v7_copy(tmp_path) -> Path:
    state = tmp_path / "state"
    state.mkdir()
    db = state / "portal.db"
    shutil.copyfile(FIXTURE, db)
    return db


# --- fresh ---------------------------------------------------------------------- #


def test_fresh_store_is_created_at_v8(tmp_path):
    CampaignStore(tmp_path).close()
    db = tmp_path / "portal.db"
    assert migrate.SCHEMA_VERSION == 8
    assert _q(db, "SELECT version, name FROM schema_migrations WHERE version = 8") == [(8, migrate.V8_NAME)]
    assert _names(db, "table") >= V8_TABLES
    assert _names(db, "trigger") >= V8_TRIGGERS
    assert {"analysis_snapshots_campaign_idx", "run_predictions_campaign_idx"} <= _names(db, "index")
    assert set(migrate.TENANT_TABLES) >= V8_TABLES


def test_the_fixture_is_a_real_v7_database():
    assert _q(FIXTURE, "SELECT MAX(version) FROM schema_migrations") == [(7,)]
    assert not (_names(FIXTURE, "table") & V8_TABLES)
    rounds = dict(_q(FIXTURE, "SELECT tenant, json_extract(state, '$.round') FROM campaigns"))
    assert rounds == {"acme": 2, "globex": 1}
    # v7 already lost the predictions of the folded runs: they are not in pending.
    n_pending = dict(_q(FIXTURE, "SELECT tenant, json_array_length(json_extract(state, '$.pending')) "
                                 "FROM campaigns"))
    assert n_pending == {"acme": 3, "globex": 2}
    assert _q(FIXTURE, "SELECT COUNT(DISTINCT run_id) FROM run_results") == [(13,)]


# --- constraints and triggers ------------------------------------------------------ #


@pytest.fixture
def v8(tmp_path):
    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="acme", name="a")["id"]
    store.close()
    conn = migrate.connect(tmp_path / "portal.db")
    yield conn, cid
    conn.close()


_INS_PRED = ("INSERT INTO run_predictions (tenant, campaign_id, run_id, source, pred_source, recipe, "
             "available, unavailable_reason, pred, lo_90, hi_90, recorded_at) "
             "VALUES (?, ?, ?, 'start', 'engine', '{}', ?, ?, ?, ?, ?, 1.0)")
_INS_SNAP = ("INSERT INTO analysis_snapshots (tenant, campaign_id, kind, analysis_sha256, analysis_bytes, "
             "analysis, created_at) VALUES (?, ?, ?, ?, 2, '{}', 1.0)")
_INS_FOLD = ("INSERT INTO run_folds (tenant, campaign_id, run_id, folded_in_round, value, folded_at) "
             "VALUES (?, ?, ?, 1, 2.0, 1.0)")


@pytest.mark.parametrize("table", sorted(V8_TABLES))
def test_every_v8_table_is_append_only(v8, table):
    conn, cid = v8
    conn.execute(_INS_PRED, ("acme", cid, "r1", 1, None, 2.0, 1.0, 3.0))
    conn.execute(_INS_SNAP, ("acme", cid, "upload", "a" * 64))
    conn.execute(_INS_FOLD, ("acme", cid, "r1"))
    with pytest.raises(sqlite3.IntegrityError, match=f"{table} is append-only: UPDATE"):
        conn.execute(f"UPDATE {table} SET tenant = 'acme'")  # noqa: S608 - parametrized constant
    with pytest.raises(sqlite3.IntegrityError, match=f"{table} is append-only: DELETE"):
        conn.execute(f"DELETE FROM {table}")  # noqa: S608 - parametrized constant
    assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (1,)  # noqa: S608


def test_prediction_checks(v8):
    conn, cid = v8
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):  # available without a mean
        conn.execute(_INS_PRED, ("acme", cid, "r1", 1, None, None, None, None))
    for reason in (None, "", "  "):  # unavailable without a reason
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(_INS_PRED, ("acme", cid, "r2", 0, reason, None, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):  # half an interval
        conn.execute(_INS_PRED, ("acme", cid, "r3", 1, None, 2.0, 1.0, None))
    conn.execute(_INS_PRED, ("acme", cid, "r4", 0, "not a proposal", None, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):  # one prediction per run
        conn.execute(_INS_PRED, ("acme", cid, "r4", 1, None, 2.0, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):  # another tenant's campaign
        conn.execute(_INS_PRED, ("globex", cid, "r5", 1, None, 2.0, None, None))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(_INS_SNAP, ("globex", cid, "upload", "a" * 64))
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute(_INS_FOLD, ("acme", "cmp_" + uuid.uuid4().hex, "r6"))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_SNAP, ("acme", cid, "guess", "a" * 64))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(_INS_SNAP, ("acme", cid, "upload", "short"))


# --- v7 -> v8 on a copy of the real v7 database --------------------------------------- #


def test_v7_copy_migrates_with_a_verified_backup_and_keeps_every_row(v7_copy):
    fixture_sha = _sha_file(FIXTURE)
    db = v7_copy
    before = _snapshot(db)

    plan = migrate.migrate_up(db, dry_run=True).plan_text
    assert "schema v7 -> v8" in plan and "pre-v8" in plan
    assert "analysis_snapshots, run_predictions and run_folds" in plan
    assert "campaign_clones" not in plan  # only the remaining step is listed
    assert _snapshot(db) == before  # the dry run wrote nothing

    result = migrate.migrate_up(db)
    assert result.action == "migrated" and result.warnings == []
    assert result.backup_path is not None and result.backup_path.name.startswith("portal.db.pre-v8-")
    assert _snapshot(result.backup_path) == before  # the backup is the v7 database

    after = _snapshot(db)
    assert [r[0] for r in after.pop("schema_migrations")] == [2, 3, 4, 5, 6, 7, 8]
    before.pop("schema_migrations")
    for t in V8_TABLES:
        after.pop(t)
    after.pop("sqlite_sequence", None)
    before.pop("sqlite_sequence", None)
    assert after == before  # every v7 row, byte for byte
    assert _names(db, "trigger") >= V8_TRIGGERS
    assert _sha_file(FIXTURE) == fixture_sha  # the fixture was never written
    assert migrate.migrate_up(db).action == "noop"


def test_v7_copy_backfill_copies_what_v7_still_held_and_invents_nothing(v7_copy):
    db = v7_copy
    migrate.migrate_up(db)
    # One snapshot per published analysis, byte for byte, linked to its round.
    snaps = _q(db, "SELECT s.tenant, s.campaign_id, s.kind, s.round, s.analysis_sha256, s.analysis, "
                   "a.state, json_extract(c.state, '$.round') FROM analysis_snapshots s "
                   "JOIN analyses a ON a.campaign_id = s.campaign_id AND a.tenant = s.tenant "
                   "JOIN campaigns c ON c.id = s.campaign_id AND c.tenant = s.tenant ORDER BY s.tenant")
    assert [s[0] for s in snaps] == ["acme", "globex"]
    for _t, _cid, kind, rnd, sha, blob, published, cur_round in snaps:
        assert kind == "migrated_v7" and blob == published and sha == sha256_text(published)
        assert rnd == cur_round  # the in-flight runs were started from it in this round
    # One prediction row per in-flight run, from the block v7 recorded at start.
    rows = _q(db, "SELECT p.tenant, p.run_id, p.source, p.recorded_at, p.pred, p.lo_90, p.hi_90, "
                  "p.lo_95, p.band_lo, p.round, p.analysis_snapshot_id, p.seed_id FROM run_predictions p "
                  "ORDER BY p.tenant, p.id")
    assert len(rows) == 5
    for tenant, run_id, source, recorded_at, pred, lo90, hi90, lo95, band_lo, rnd, sid, seed in rows:
        (state_text,) = _q(db, "SELECT state FROM campaigns WHERE tenant = ?", (tenant,))[0]
        run = next(r for r in json.loads(state_text)["pending"] if r["id"] == run_id)
        block = run["prediction_at_start"]
        assert source == "migrated_v7_state" and seed is None
        assert recorded_at == block["recorded_at"]  # ITS time, never the migration time
        assert pred == block["pred"] and band_lo == block["interval"]["lo"] and rnd == run["round"]
        iv90 = next(i for i in block["intervals"] if i["coverage"] == 0.9)
        iv95 = next(i for i in block["intervals"] if i["coverage"] == 0.95)
        assert (lo90, hi90, lo95) == (iv90["lo"], iv90["hi"], iv95["lo"])
        assert sid == _q(db, "SELECT id FROM analysis_snapshots WHERE tenant = ?", (tenant,))[0][0]
        # the caveats the engine served, copied verbatim (rule: every band keeps its caveat)
        cav = _q(db, "SELECT caveat_code_90, caveat_90, caveat_code_95, caveat_95, band_basis, "
                     "band_caveat_code, band_caveat FROM run_predictions WHERE tenant = ? AND run_id = ?",
                 (tenant, run_id))[0]
        (published,) = _q(db, "SELECT state FROM analyses WHERE tenant = ?", (tenant,))[0]
        cc = json.loads(published)["conformal_caveat"]
        ok95 = iv95.get("available") is not False  # a refused level is not stored, nor its caveat
        assert cav == (iv90["caveat"]["code"], iv90["caveat"]["message"],
                       iv95["caveat"]["code"] if ok95 else None, iv95["caveat"]["message"] if ok95 else None,
                       cc["coverage_basis"], cc["code"], cc["message"])
    # Runs v7 already folded lost their predictions: nothing is invented for them.
    folded_ids = {r[0] for r in _q(db, "SELECT DISTINCT run_id FROM run_results")} - {r[1] for r in rows}
    assert len(folded_ids) == 9  # acme 3 + 3, globex 3
    assert _q(db, "SELECT COUNT(*) FROM run_folds") == [(0,)]


_CAVEAT_90 = {"code": "coverage_lower_at_proposals", "message": "as served (90)", "coverage_at_proposals_estimate": None}
_BLOCKS = [
    {"available": True, "pred": 2.0, "std": 0.1, "proposal_index": 0,
     "interval": {"lo": 1.0, "hi": 3.0, "coverage": 0.9, "method": "conformal_global"},
     "analysis_provenance": {"engine_version": "0.1.0", "provenance_sha256": "p" * 64},
     "analysis_generation": "g" * 32,
     "intervals": [{"coverage": 0.9, "available": True, "lo": 1.5, "hi": 2.5, "caveat": _CAVEAT_90,
                    "method": "conformal_selection_aware", "coverage_basis": "x"},
                   {"coverage": 0.95, "available": False, "lo": None, "hi": None}]},
    {"available": True, "pred": 2.0, "intervals_omitted": "too large"},
    {"available": False, "reason": "not a proposal"},
    {"available": True, "pred": float("nan")},
    None,
]
_ANALYSIS = {"conformal_caveat": {"code": "coverage_not_guaranteed", "coverage_basis": "held_out_past_runs",
                                  "message": "as served (global)"}}


def test_live_extraction_and_the_frozen_migration_copy_agree():
    """`CampaignStore.start` (prediction_history) and the frozen v7 -> v8 copy in
    migrate.py extract the same columns today."""
    for block in _BLOCKS:
        for analysis in (None, _ANALYSIS):
            assert migrate._v8_prediction_columns(block, None, analysis) == prediction_columns(block, None, analysis)
    proposal = {"intervals": [{"coverage": 0.95, "available": True, "lo": 0.5, "hi": 3.5, "method": "m",
                               "coverage_basis": "b", "caveat": {"code": "c", "message": "as served (95)"}}]}
    assert migrate._v8_prediction_columns(_BLOCKS[0], proposal, None) == prediction_columns(_BLOCKS[0], proposal)
    assert set(migrate._V8_PRED_COLUMNS) == set(prediction_columns(None))


def test_extraction_copies_values_and_caveats_verbatim():
    cols = prediction_columns(_BLOCKS[0], None, _ANALYSIS)
    assert (cols["lo_90"], cols["hi_90"], cols["method_90"], cols["basis_90"]) == (1.5, 2.5, "conformal_selection_aware", "x")
    assert (cols["caveat_code_90"], cols["caveat_90"]) == ("coverage_lower_at_proposals", "as served (90)")
    assert cols["lo_95"] is None and cols["hi_95"] is None and cols["caveat_95"] is None  # refused level
    assert (cols["band_lo"], cols["band_coverage"], cols["model_version"]) == (1.0, 0.9, "0.1.0")
    assert (cols["band_basis"], cols["band_caveat_code"], cols["band_caveat"]) == (
        "held_out_past_runs", "coverage_not_guaranteed", "as served (global)")
    assert prediction_columns(_BLOCKS[0])["band_caveat"] is None  # no analysis given: nothing invented
    assert prediction_columns(None)["available"] == 0 and prediction_columns(None)["unavailable_reason"]
    off = prediction_columns({"available": False, "reason": "not a proposal"})
    assert off == {**off, "available": 0, "unavailable_reason": "not a proposal", "pred": None}


def test_the_migration_does_not_import_campaign_code():
    """R03: storage must not import kalos.portal.prediction_history (campaigns)."""
    source = (_REPO / "kalos" / "portal" / "migrate.py").read_text()
    assert "prediction_history" not in source.replace("prediction_history.py", "")


def test_a_migrated_run_without_prediction_at_start_is_copied_as_unavailable(v7_copy):
    conn = sqlite3.connect(str(v7_copy))
    (cid, text) = conn.execute("SELECT id, state FROM campaigns WHERE tenant = 'globex'").fetchone()
    state = json.loads(text)
    legacy = state["pending"][0]
    legacy.pop("prediction_at_start")  # a run started before predictions at start existed
    no_time = state["pending"][1]
    no_time["prediction_at_start"].pop("recorded_at")
    conn.execute("UPDATE campaigns SET state = ? WHERE id = ?", (json.dumps(state), cid))
    conn.commit()
    conn.close()
    migrate.migrate_up(v7_copy)
    row = _q(v7_copy, "SELECT available, unavailable_reason, pred, lo_90, recorded_at, source "
                      "FROM run_predictions WHERE run_id = ?", (legacy["id"],))[0]
    assert row[:4] == (0, migrate._V8_PRE_R5_REASON, None, None)
    assert row[4] == legacy["created_at"] and row[5] == "migrated_v7_state"
    (recorded,) = _q(v7_copy, "SELECT recorded_at FROM run_predictions WHERE run_id = ?", (no_time["id"],))[0]
    assert recorded == no_time["created_at"]  # the run's start time, not the migration time


def test_the_engine_keeps_history_on_the_migrated_v7_database(v7_copy):
    """After the upgrade the v8 store folds the migrated in-flight runs and keeps
    their prediction, band and run id (v7 dropped them at this point)."""
    state = v7_copy.parent
    store = CampaignStore(state)  # migrates on open, as the engine does
    try:
        before = {r[1]: r for r in _q(v7_copy, "SELECT tenant, run_id, pred, lo_90 FROM run_predictions "
                                               "WHERE tenant = 'acme'")}
        (cid,) = _q(v7_copy, "SELECT id FROM campaigns WHERE tenant = 'acme'")[0]
        cur = store.get(tenant="acme", campaign_id=cid)
        awaiting = [r for r in cur["pending"] if r["result"] is None]
        assert len(awaiting) == 1
        store.set_result(awaiting[0]["id"], 6.5, tenant="acme", campaign_id=cid)
        _df, _t, gen = store.plan_fold(tenant="acme", campaign_id=cid)
        store.commit_fold(gen, tenant="acme", campaign_id=cid,
                          analysis={"target": "lipase_titer", "proposals": []})
        assert store.get(tenant="acme", campaign_id=cid)["pending"] == []
        folds = _q(v7_copy, "SELECT run_id, folded_in_round, result_version FROM run_folds "
                            "WHERE tenant = 'acme'")
        assert {f[0] for f in folds} == set(before) and {f[1] for f in folds} == {3}
        after = {r[1]: r for r in _q(v7_copy, "SELECT tenant, run_id, pred, lo_90 FROM run_predictions "
                                              "WHERE tenant = 'acme'")}
        assert after == before  # the predictions survived the fold
        kinds = [r[0] for r in _q(v7_copy, "SELECT kind FROM analysis_snapshots WHERE tenant = 'acme' "
                                           "ORDER BY id")]
        assert kinds == ["migrated_v7", "reanalyze"]  # the v7 analysis was not overwritten
    finally:
        store.close()


# --- down and back ------------------------------------------------------------------ #


def test_down_renames_aside_and_a_v7_engine_accepts_it_and_re_up_is_clean(v7_copy):
    db = v7_copy
    migrate.migrate_up(db)
    counts = {t: _q(db, f"SELECT COUNT(*) FROM {t}")[0][0] for t in V8_TABLES}  # noqa: S608
    dry = migrate.migrate_down(db, dry_run=True).plan_text
    assert "schema v8 -> v7" in dry and "schema v7 -> v6" not in dry
    for t in V8_TABLES:
        assert f"rename: {t} -> {t}_v8_rolledback (kept, not dropped; {counts[t]} row(s))" in dry

    result = migrate.migrate_down(db)
    assert result.action == "rolled_back" and result.backup_path is not None
    assert _q(db, "SELECT MAX(version) FROM schema_migrations") == [(7,)]
    tables = _names(db, "table")
    assert not (tables & V8_TABLES)
    assert {f"{t}_v8_rolledback" for t in V8_TABLES} <= tables
    for t in V8_TABLES:
        assert _q(db, f'SELECT COUNT(*) FROM "{t}_v8_rolledback"')[0][0] == counts[t]  # noqa: S608
    assert not (V8_TRIGGERS & _names(db, "trigger"))
    conn = migrate.connect(db)
    try:
        migrate._assert_v2_shape(conn)  # what a v7 engine checks at startup
    finally:
        conn.close()
    # the v7 rows are what the fixture had
    assert _q(db, "SELECT COUNT(*) FROM run_results") == _q(FIXTURE, "SELECT COUNT(*) FROM run_results")

    assert migrate.migrate_up(db).action == "migrated"
    assert _q(db, "SELECT COUNT(*) FROM run_predictions") == [(counts["run_predictions"],)]
    migrate.migrate_down(db)
    assert {f"{t}_v8_rolledback_2" for t in V8_TABLES} <= _names(db, "table")


def test_status_shows_v8_and_pending_from_v7(v7_copy):
    report, healthy = migrate.status_report(v7_copy)
    assert healthy and "schema version: v7 (this code: v8)" in report and "schema v7 -> v8" in report
    migrate.migrate_up(v7_copy)
    report, healthy = migrate.status_report(v7_copy)
    assert healthy and "schema version: v8" in report and "shape check: ok" in report


@pytest.mark.parametrize(("ddl", "missing"), [
    ("DROP TRIGGER run_predictions_no_update", "trigger run_predictions_no_update"),
    ("DROP TRIGGER analysis_snapshots_no_delete", "trigger analysis_snapshots_no_delete"),
    ("DROP TRIGGER run_folds_no_update", "trigger run_folds_no_update"),
    ("ALTER TABLE run_folds RENAME TO elsewhere", "run_folds"),
])
def test_shape_check_catches_a_missing_v8_table_or_trigger(tmp_path, ddl, missing):
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


def test_a_corrupt_campaign_state_is_skipped_with_a_warning(v7_copy):
    conn = sqlite3.connect(str(v7_copy))
    conn.execute("UPDATE campaigns SET state = '{not json' WHERE tenant = 'globex'")
    conn.commit()
    conn.close()
    result = migrate.migrate_up(v7_copy)
    assert result.action == "migrated"
    assert any("state is not valid JSON" in w for w in result.warnings)
    assert _q(v7_copy, "SELECT COUNT(*) FROM run_predictions WHERE tenant = 'globex'") == [(0,)]
    assert _q(v7_copy, "SELECT COUNT(*) FROM run_predictions WHERE tenant = 'acme'") == [(3,)]


def test_docs_describe_v8_and_move_the_other_crux_items_on():
    text = (_REPO / "docs" / "MIGRATIONS.md").read_text()
    assert "## Schema v8: prediction history" in text
    assert "The remaining Crux items (result check, decision ledger) take v9 or later." in text
    assert "reserved" not in text.split("| v8 |")[1].split("\n")[0]
