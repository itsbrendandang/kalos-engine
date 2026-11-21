"""Prediction history (schema v8, BRE-32): the append-only record of what the
engine published, what it predicted for each run BEFORE the run was measured,
and which recorded result each fold put into the model.

Before v8 the campaign state blob was the only place a started run's
`prediction_at_start` lived, and `commit_fold` drops measured runs from
`pending`: after a fold a run lost its prediction, its band and its run id, and
`analyses` kept only the newest analysis. Three append-only tables (the same
two-trigger UPDATE/DELETE guard as `gate_decisions`) keep that history:

- `analysis_snapshots`: every published analysis, byte for byte, with its
  sha256, the campaign round it describes and its provenance hashes. Written in
  the SAME transaction as the `analyses` upsert (`CampaignStore._write_analysis`),
  so a rolled-back publish leaves no snapshot.
- `run_predictions`: one row per started run, written in the SAME transaction as
  `CampaignStore.start`: the engine's mean and sd, the per-proposal 90% and 95%
  intervals with their method and coverage basis, the global conformal band, the
  model (engine) version, the round, and the sha256 + snapshot id of the
  analysis the prediction was read from. A run whose recipe is not a published
  proposal gets a row with `available = 0` and the reason: the engine never
  predicted it, and a caller-supplied `pred` is never stored as an engine one.
- `run_folds`: one row per run folded into the base dataset by `commit_fold`:
  the round it was folded in, the value written into `base_rows` and the
  `run_results` version that value came from. Result versions and corrections
  themselves stay in the v5 `run_results` table (append-only, a correction needs
  a reason); with `run_predictions` and `run_folds` every version can be tied
  back to its run, round and prediction after the fold.

`seed_id` is a random id `CampaignStore.seed` writes into the campaign state on
every upload: a re-upload replaces the base dataset and restarts the round
count, so `(seed_id, round)` identifies a round. Campaigns seeded before v8 have
no `seed_id` (NULL) until their next upload.

Every statement carries `tenant = ?`. The functions taking a `conn` run inside
the caller's transaction and never commit. Torch-free and FastAPI-free:
`kalos.portal.migrate` imports `prediction_columns` for the v7 -> v8 backfill.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping
from typing import Any

SNAPSHOT_KINDS = ("upload", "reanalyze", "publish", "migrated_v7")
PREDICTION_SOURCES = ("start", "migrated_v7_state")
PRED_SOURCES = ("engine", "caller")
# The per-proposal interval levels stored as columns (`lo_90`, `hi_95`, ...).
LEVELS: tuple[float, ...] = (0.9, 0.95)
MAX_RECIPE_JSON_CHARS = 16_384

PRE_R5_REASON = (
    "the run was started before the engine recorded predictions at start; no prediction "
    "was recorded for it"
)
UNKNOWN_REASON = "no engine prediction was recorded for this run"

# Columns of `run_predictions` filled from a `prediction_at_start` block.
PREDICTION_FIELDS: tuple[str, ...] = (
    "available", "unavailable_reason", "pred", "std",
    "lo_90", "hi_90", "method_90", "basis_90", "caveat_code_90", "caveat_90",
    "lo_95", "hi_95", "method_95", "basis_95", "caveat_code_95", "caveat_95",
    "band_lo", "band_hi", "band_coverage", "band_method", "band_basis",
    "band_caveat_code", "band_caveat",
    "model_version", "provenance_sha256", "analysis_generation", "proposal_index",
)
MAX_CAVEAT_CHARS = 2048


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str(value: Any, limit: int = 128) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def level_key(coverage: float) -> str | None:
    """'90' / '95' for a stored level, else None."""
    for lv in LEVELS:
        if abs(float(coverage) - lv) < 1e-9:
            return str(int(round(lv * 100)))
    return None


def _caveat(value: Any) -> tuple[str | None, str | None]:
    """(code, message) of a caveat block as the engine served it, else (None, None)."""
    if not isinstance(value, Mapping):
        return None, None
    msg = value.get("message")
    return _str(value.get("code"), 64), (msg[:MAX_CAVEAT_CHARS] if isinstance(msg, str) and msg else None)


def prediction_columns(
    block: Any, proposal: Mapping[str, Any] | None = None,
    analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The flat `run_predictions` columns for a run's `prediction_at_start` block.

    The per-proposal intervals are read from `proposal` (the matched published
    proposal) when given, else from the block's own size-capped copy
    (`kalos.portal.run_prediction.COPIED_PROPOSAL_KEYS`). Only AVAILABLE
    intervals with finite ends are stored; a level the engine refused stays NULL.
    Each stored interval keeps the caveat the engine served with it (`caveat`:
    code + message, copied verbatim). The global band keeps the published
    analysis's `conformal_caveat` (code, message, coverage_basis) when
    `analysis` is given. Nothing is computed: every number and every word is
    copied from what the engine published.
    """
    out: dict[str, Any] = dict.fromkeys(PREDICTION_FIELDS)
    out["available"] = 0
    if not isinstance(block, Mapping):
        out["unavailable_reason"] = PRE_R5_REASON
        return out
    pred = _num(block.get("pred"))
    if not block.get("available") or pred is None:
        reason = block.get("reason")
        out["unavailable_reason"] = reason[:512] if isinstance(reason, str) and reason else UNKNOWN_REASON
        return out
    out.update(available=1, pred=pred, std=_num(block.get("std")),
               proposal_index=_int(block.get("proposal_index")),
               analysis_generation=_str(block.get("analysis_generation")))
    source = proposal if isinstance(proposal, Mapping) else block
    intervals = source.get("intervals")
    for entry in intervals if isinstance(intervals, list) else []:
        if not isinstance(entry, Mapping) or entry.get("available") is False:
            continue
        cov = _num(entry.get("coverage"))
        key = level_key(cov) if cov is not None else None
        lo, hi = _num(entry.get("lo")), _num(entry.get("hi"))
        if key is None or lo is None or hi is None or out[f"lo_{key}"] is not None:
            continue
        out[f"lo_{key}"], out[f"hi_{key}"] = lo, hi
        out[f"method_{key}"] = _str(entry.get("method"), 64)
        out[f"basis_{key}"] = _str(entry.get("coverage_basis"), 64)
        out[f"caveat_code_{key}"], out[f"caveat_{key}"] = _caveat(entry.get("caveat"))
    band = block.get("interval")
    if isinstance(band, Mapping):
        lo, hi, cov = _num(band.get("lo")), _num(band.get("hi")), _num(band.get("coverage"))
        if lo is not None and hi is not None and cov is not None:
            out.update(band_lo=lo, band_hi=hi, band_coverage=cov,
                       band_method=_str(band.get("method"), 64))
            cc = (analysis or {}).get("conformal_caveat") if isinstance(analysis, Mapping) else None
            if isinstance(cc, Mapping):
                out["band_basis"] = _str(cc.get("coverage_basis"), 64)
                out["band_caveat_code"], out["band_caveat"] = _caveat(cc)
    prov = block.get("analysis_provenance")
    if isinstance(prov, Mapping):
        out["model_version"] = _str(prov.get("engine_version"), 64)
        out["provenance_sha256"] = _str(prov.get("provenance_sha256"), 64)
    return out


# --- writes (inside the caller's transaction) ---------------------------------- #


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _snapshot_fields(analysis: Mapping[str, Any] | None) -> dict[str, Any]:
    a = _mapping(analysis)
    prov = _mapping(a.get("proposal_provenance"))
    run = _mapping(a.get("run"))
    proposals = a.get("proposals")
    engine_version = prov.get("engine_version") or a.get("engine_version")
    n_rows = _int(run.get("n_rows"))
    if n_rows is None:
        n_rows = _int(a.get("n"))
    return {
        "provenance_sha256": _str(prov.get("provenance_sha256"), 64),
        "input_sha256": _str(prov.get("input_sha256") or run.get("input_sha256"), 64),
        "engine_version": _str(engine_version, 64),
        "n_rows": n_rows,
        "n_proposals": len(proposals) if isinstance(proposals, list) else None,
    }


def record_snapshot(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, blob: str,
    analysis: Mapping[str, Any] | None, kind: str, round_: int | None,
    campaign_generation: str | None, dataset: str | None, created_at: float,
) -> int:
    """Append the published analysis `blob` (stored byte for byte). Returns its id."""
    if kind not in SNAPSHOT_KINDS:
        raise ValueError(f"unknown snapshot kind {kind!r}")
    if analysis is None:
        try:
            parsed = json.loads(blob)
        except ValueError:
            parsed = None
        analysis = parsed if isinstance(parsed, dict) else None
    f = _snapshot_fields(analysis)
    cur = conn.execute(
        "INSERT INTO analysis_snapshots (tenant, campaign_id, round, kind, campaign_generation, "
        "analysis_sha256, provenance_sha256, input_sha256, engine_version, dataset, n_rows, "
        "n_proposals, analysis_bytes, analysis, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tenant, campaign_id, round_, kind, campaign_generation, sha256_text(blob),
         f["provenance_sha256"], f["input_sha256"], f["engine_version"],
         dataset[:256] if isinstance(dataset, str) else None, f["n_rows"], f["n_proposals"],
         len(blob.encode("utf-8")), blob, created_at),
    )
    return int(cur.lastrowid or 0)


def snapshot_id_for(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, analysis_sha256: str,
) -> int | None:
    """The newest snapshot of this campaign whose blob has this sha256."""
    row = conn.execute(
        "SELECT id FROM analysis_snapshots WHERE tenant = ? AND campaign_id = ? "
        "AND analysis_sha256 = ? ORDER BY id DESC LIMIT 1",
        (tenant, campaign_id, analysis_sha256),
    ).fetchone()
    return int(row[0]) if row is not None else None


def record_prediction(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, run_id: str,
    round_: int | None, seed_id: str | None, recipe: Mapping[str, Any], block: Any,
    proposal: Mapping[str, Any] | None, pred_source: str, analysis_sha256: str | None,
    analysis_snapshot_id: int | None, source: str, recorded_at: float,
    analysis: Mapping[str, Any] | None = None,
) -> None:
    """Append the prediction recorded for one started run."""
    if source not in PREDICTION_SOURCES:
        raise ValueError(f"unknown prediction source {source!r}")
    cols = prediction_columns(block, proposal, analysis)
    try:
        recipe_json = json.dumps(dict(recipe), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        recipe_json = json.dumps({"omitted": "recipe is not JSON-safe; not stored in the history"})
    if len(recipe_json) > MAX_RECIPE_JSON_CHARS:
        recipe_json = json.dumps({"omitted": "recipe too large to store in the history"})
    names = ("tenant", "campaign_id", "run_id", "round", "seed_id", "source", "pred_source",
             "recipe", *PREDICTION_FIELDS, "analysis_sha256", "analysis_snapshot_id", "recorded_at")
    values = (tenant, campaign_id, run_id, round_, seed_id, source,
              pred_source if pred_source in PRED_SOURCES else "caller", recipe_json,
              *(cols[k] for k in PREDICTION_FIELDS), analysis_sha256, analysis_snapshot_id,
              recorded_at)
    conn.execute(
        f"INSERT INTO run_predictions ({', '.join(names)}) "  # noqa: S608 - module-constant column names; values are bound
        f"VALUES ({', '.join('?' * len(names))})",
        values,
    )


def record_fold(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, run: Mapping[str, Any],
    folded_in_round: int, analysis_snapshot_id: int | None, folded_at: float,
) -> None:
    """Append which value of `run` the fold wrote into `base_rows`, and its version."""
    value = _num(run.get("result"))
    if value is None:  # pragma: no cover - the fold only folds finite results
        return
    conn.execute(
        "INSERT INTO run_folds (tenant, campaign_id, run_id, round_started, folded_in_round, "
        "result_version, value, measured_at, analysis_snapshot_id, folded_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tenant, campaign_id, str(run.get("id")), _int(run.get("round")), folded_in_round,
         _int(run.get("result_version")), value, _num(run.get("measured_at")),
         analysis_snapshot_id, folded_at),
    )


# --- reads ----------------------------------------------------------------------- #

SNAPSHOT_SUMMARY_COLUMNS = (
    "id", "round", "kind", "campaign_generation", "analysis_sha256", "provenance_sha256",
    "input_sha256", "engine_version", "dataset", "n_rows", "n_proposals", "analysis_bytes",
    "created_at",
)
_TRACK_COLUMNS = (
    "id", "run_id", "round", "seed_id", "source", "pred_source", "available",
    "unavailable_reason", "pred", "std", "lo_90", "hi_90", "method_90", "basis_90",
    "caveat_code_90", "caveat_90", "lo_95", "hi_95", "method_95", "basis_95",
    "caveat_code_95", "caveat_95", "band_lo", "band_hi", "band_coverage",
    "band_method", "band_basis", "band_caveat_code", "band_caveat", "model_version",
    "provenance_sha256", "analysis_sha256", "analysis_snapshot_id", "recorded_at",
)
# The latest result version, the first result time and the version count come
# from `run_results`; the fold (if any) from `run_folds`. Every subquery names
# the tenant.
_TRACK_SQL = (
    "SELECT " + ", ".join(f"p.{c}" for c in _TRACK_COLUMNS) + ", "  # noqa: S608 - module-constant columns; values are bound
    "(SELECT COUNT(*) FROM run_results r WHERE r.tenant = p.tenant "
    " AND r.campaign_id = p.campaign_id AND r.run_id = p.run_id) AS n_versions, "
    "(SELECT MIN(r.recorded_at) FROM run_results r WHERE r.tenant = p.tenant "
    " AND r.campaign_id = p.campaign_id AND r.run_id = p.run_id) AS first_result_at, "
    "(SELECT r.value FROM run_results r WHERE r.tenant = p.tenant "
    " AND r.campaign_id = p.campaign_id AND r.run_id = p.run_id ORDER BY r.version DESC LIMIT 1) AS value, "
    "(SELECT r.version FROM run_results r WHERE r.tenant = p.tenant "
    " AND r.campaign_id = p.campaign_id AND r.run_id = p.run_id ORDER BY r.version DESC LIMIT 1) AS version, "
    "(SELECT r.recorded_at FROM run_results r WHERE r.tenant = p.tenant "
    " AND r.campaign_id = p.campaign_id AND r.run_id = p.run_id ORDER BY r.version DESC LIMIT 1) AS value_recorded_at, "
    "f.folded_in_round, f.result_version AS folded_version, f.value AS folded_value "
    "FROM run_predictions p LEFT JOIN run_folds f ON f.tenant = p.tenant "
    "AND f.campaign_id = p.campaign_id AND f.run_id = p.run_id "
    "WHERE p.tenant = ? AND p.campaign_id = ?"
)
TRACK_ROW_KEYS = (*_TRACK_COLUMNS, "n_versions", "first_result_at", "value", "version",
                  "value_recorded_at", "folded_in_round", "folded_version", "folded_value")


def track_rows(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str,
    after_id: int | None = None, round_: int | None = None, limit: int | None = None,
) -> list[dict[str, Any]]:
    """Every recorded prediction of the campaign joined with its results and
    fold, oldest first (keyset on `run_predictions.id`)."""
    sql = _TRACK_SQL
    params: list[Any] = [tenant, campaign_id]
    if after_id is not None:
        sql += " AND p.id > ?"
        params.append(after_id)
    if round_ is not None:
        sql += " AND p.round = ?"
        params.append(round_)
    sql += " ORDER BY p.id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [dict(zip(TRACK_ROW_KEYS, r)) for r in conn.execute(sql, params).fetchall()]


def recipes_for(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, ids: list[int],
) -> dict[int, Any]:
    """`run_predictions.recipe` (parsed) for the given row ids."""
    if not ids:
        return {}
    marks = ", ".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, recipe FROM run_predictions WHERE tenant = ? AND campaign_id = ? "  # noqa: S608 - placeholders only; values are bound
        f"AND id IN ({marks})",
        (tenant, campaign_id, *ids),
    ).fetchall()
    out: dict[int, Any] = {}
    for rid, text in rows:
        try:
            out[int(rid)] = json.loads(text)
        except (TypeError, ValueError):
            out[int(rid)] = None
    return out


def prediction_for_run(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, run_id: str,
) -> dict[str, Any] | None:
    rows = conn.execute(_TRACK_SQL + " AND p.run_id = ?", (tenant, campaign_id, run_id)).fetchall()
    return dict(zip(TRACK_ROW_KEYS, rows[0])) if rows else None


def result_versions(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, run_id: str, limit: int,
) -> list[dict[str, Any]]:
    cols = ("id", "version", "value", "previous_value", "reason", "actor", "recorded_at")
    rows = conn.execute(
        f"SELECT {', '.join(cols)} FROM run_results WHERE tenant = ? AND campaign_id = ? "  # noqa: S608 - module-constant columns; values are bound
        "AND run_id = ? ORDER BY version LIMIT ?",
        (tenant, campaign_id, run_id, limit),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def fold_for_run(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, run_id: str,
) -> dict[str, Any] | None:
    cols = ("round_started", "folded_in_round", "result_version", "value", "measured_at",
            "analysis_snapshot_id", "folded_at")
    row = conn.execute(
        f"SELECT {', '.join(cols)} FROM run_folds WHERE tenant = ? AND campaign_id = ? "  # noqa: S608 - module-constant columns; values are bound
        "AND run_id = ?",
        (tenant, campaign_id, run_id),
    ).fetchone()
    return dict(zip(cols, row)) if row is not None else None


def snapshot_page(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, before_id: int | None, limit: int,
) -> dict[str, Any]:
    """One newest-first page of snapshot SUMMARIES (never the blobs)."""
    sql = (
        f"SELECT {', '.join(SNAPSHOT_SUMMARY_COLUMNS)} FROM analysis_snapshots "  # noqa: S608 - module-constant columns; values are bound
        "WHERE tenant = ? AND campaign_id = ?"
        + (" AND id < ?" if before_id is not None else "")
        + " ORDER BY id DESC LIMIT ?"
    )
    params: tuple[Any, ...] = (tenant, campaign_id, *((before_id,) if before_id is not None else ()),
                               limit + 1)
    fetched = conn.execute(sql, params).fetchall()
    items = [dict(zip(SNAPSHOT_SUMMARY_COLUMNS, r)) for r in fetched[:limit]]
    has_more = len(fetched) > limit
    return {"items": items, "has_more": has_more,
            "next_before_id": items[-1]["id"] if has_more and items else None}


def snapshot_blob(
    conn: sqlite3.Connection, *, tenant: str, campaign_id: str, snapshot_id: int,
) -> tuple[dict[str, Any], str] | None:
    """(summary, stored blob) of one snapshot, or None."""
    row = conn.execute(
        f"SELECT {', '.join(SNAPSHOT_SUMMARY_COLUMNS)}, analysis FROM analysis_snapshots "  # noqa: S608 - module-constant columns; values are bound
        "WHERE tenant = ? AND campaign_id = ? AND id = ?",
        (tenant, campaign_id, snapshot_id),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(SNAPSHOT_SUMMARY_COLUMNS, row[:-1])), str(row[-1])
