"""Kalos portal - prediction history and the prospective track record (schema v8,
BRE-32). Read-only routes, `read` scope:

- ``GET /api/campaigns/{id}/track-record``: per round (newest first, paginated
  with `limit`/`offset`), the runs' predictions recorded at start vs the results
  measured: inside/outside the band recorded at start, coverage of the round and
  coverage so far (counts + Wilson 95% interval, `judged` from
  `MIN_JUDGED_N` runs), and the error (measured - predicted). Plus the pooled
  score and the rounds of the current dataset that were folded with no recorded
  prediction (before v8). Query: `coverage` (0.9 | 0.95), `band` (proposal |
  global). Rules: kalos/portal/track_record.py.
- ``GET /api/campaigns/{id}/track-record/runs``: the same, one row per run
  (keyset-paginated oldest first with `after_id`; optional `round`): recipe,
  the prediction and band recorded at start, the latest measured value, the
  number of result versions, the fold, and why a run is not scored.
- ``GET /api/campaigns/{id}/runs/{run_id}/history``: one run's full record: the
  prediction recorded at start, every result version (corrections with their
  reasons, capped at `MAX_RESULT_VERSIONS`) and the fold.
- ``GET /api/campaigns/{id}/analyses``: summaries of every analysis the campaign
  published (newest first, keyset `before_id`); never the blobs.
- ``GET /api/campaigns/{id}/analyses/{snapshot_id}``: one stored analysis, byte
  for byte as published, with its sha256.

Same security rules as `kalos.portal.campaigns_api`: the scope is checked before
the id is looked at; the tenant is always `principal.tenant`; malformed, missing
and foreign campaign ids get the byte-identical `campaign_not_found_response()`;
every SQL statement carries `tenant = ?`; caller input is never echoed. Every
list is bounded (`limit` caps), so no response grows with the campaign's age
except the snapshot detail, which is one published analysis (the `/analysis`
size).
"""
from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from kalos.portal.auth import READ, Principal, require_scope
from kalos.portal.campaign import CampaignNotFound, CampaignStore, get_campaign_store
from kalos.portal.campaign_routes import campaign_not_found_response
from kalos.portal.campaigns_api import RUN_ID_PATTERN, _valid_campaign_id
from kalos.portal.prediction_history import (
    LEVELS,
    fold_for_run,
    prediction_for_run,
    recipes_for,
    result_versions,
    snapshot_blob,
    snapshot_page,
    track_rows,
)
from kalos.portal.track_record import (
    EXCLUDED_REASONS,
    RULES,
    band_for,
    build_track_record,
    rounds_without_record,
    score_run,
)

router = APIRouter()

ROUNDS_DEFAULT_LIMIT = 20
ROUNDS_MAX_LIMIT = 100
RUNS_DEFAULT_LIMIT = 100
RUNS_MAX_LIMIT = 500
SNAPSHOTS_DEFAULT_LIMIT = 50
SNAPSHOTS_MAX_LIMIT = 200
MAX_RESULT_VERSIONS = 200
MAX_UNRECORDED_ROUNDS_LISTED = 200
MAX_OFFSET = 100_000

NOT_RECORDED_REASON = (
    "no prediction was recorded for this round's runs: they were folded before the engine kept "
    "prediction history (schema v8), or none was started from a published proposal"
)
_RUN_NOT_FOUND = {"error": "run not found", "code": "run_not_found"}
_SNAPSHOT_NOT_FOUND = {"error": "analysis snapshot not found", "code": "snapshot_not_found"}


def _coverage_error() -> JSONResponse:
    return JSONResponse(
        {"error": "unsupported coverage", "code": "unsupported_coverage", "supported": list(LEVELS)},
        status_code=400,
    )


def _level(coverage: float) -> float | None:
    for lv in LEVELS:
        if abs(coverage - lv) < 1e-9:
            return lv
    return None


def _prediction_view(row: dict[str, Any]) -> dict[str, Any]:
    """The prediction recorded at start, as stored (basis predicted)."""
    intervals = []
    for lv in LEVELS:
        b = band_for(row, lv, "proposal")
        if b is not None:
            intervals.append({k: b[k] for k in ("coverage", "lo", "hi", "method", "coverage_basis")})
    global_band = None
    if row.get("band_lo") is not None:
        global_band = {"lo": row["band_lo"], "hi": row["band_hi"], "coverage": row["band_coverage"],
                       "method": row.get("band_method")}
    return {
        "available": bool(row.get("available")),
        "unavailable_reason": row.get("unavailable_reason"),
        "basis": "predicted",
        "pred": row.get("pred"),
        "std": row.get("std"),
        "intervals": intervals,
        "global_band": global_band,
        "pred_source": row.get("pred_source"),
        "model_version": row.get("model_version"),
        "provenance_sha256": row.get("provenance_sha256"),
        "analysis_sha256": row.get("analysis_sha256"),
        "analysis_snapshot_id": row.get("analysis_snapshot_id"),
        "recorded_at": row.get("recorded_at"),
        "source": row.get("source"),
    }


def _run_item(row: dict[str, Any], score: dict[str, Any], recipe: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "round": row.get("round"),
        "recipe": recipe,
        "status": score["status"],
        "prediction": _prediction_view(row),
        "band": score["band"],
        "measured": {
            "value": score["measured"],
            "basis": "measured",
            "version": row.get("version") if score["measured"] is not None else None,
            "n_versions": row.get("n_versions") or 0,
            "first_recorded_at": row.get("first_result_at"),
            "latest_recorded_at": row.get("value_recorded_at"),
            "corrected": score["corrected"],
        },
        "folded_in_round": row.get("folded_in_round"),
        "folded_version": row.get("folded_version"),
        "scored": score["scored"],
        "inside": score["inside"],
        "error": score["error"],
        "excluded": score["excluded"],
    }


@router.get("/api/campaigns/{campaign_id}/track-record")
def track_record(
    campaign_id: str,
    coverage: float = Query(default=0.9),
    band: Literal["proposal", "global"] = Query(default="proposal"),
    limit: int = Query(default=ROUNDS_DEFAULT_LIMIT, ge=1, le=ROUNDS_MAX_LIMIT),
    offset: int = Query(default=0, ge=0, le=MAX_OFFSET),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Per round (newest first): recorded predictions vs measured results."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    level = _level(coverage)
    if level is None:
        return _coverage_error()
    tenant = principal.tenant
    try:
        store._check(tenant, campaign_id)
        with store._txn(write=False) as conn:
            row = store._row(conn, tenant, campaign_id)
            if row is None:
                raise CampaignNotFound()
            state = row.state()
            rows = track_rows(conn, tenant=tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    record = build_track_record(rows, coverage=level, band=band, state=state)
    newest_first = list(reversed(record["rounds"]))
    page = newest_first[offset: offset + limit]
    unrecorded = rounds_without_record(state, record["rounds"])
    body = {
        "campaign_id": campaign_id,
        "has_campaign": state is not None,
        "basis": record["basis"],
        "coverage": level,
        "band": band,
        "current_round": state.get("round") if state else None,
        "rounds": page,
        "page": {
            "limit": limit, "offset": offset, "total_rounds": len(newest_first),
            "has_more": offset + limit < len(newest_first),
            "next_offset": offset + limit if offset + limit < len(newest_first) else None,
        },
        "pooled": record["pooled"],
        "rounds_without_record": {
            "rounds": unrecorded[-MAX_UNRECORDED_ROUNDS_LISTED:],
            "count": len(unrecorded),
            "reason": NOT_RECORDED_REASON if unrecorded else None,
        },
        "rules": record["rules"],
        "excluded_reasons": record["excluded_reasons"],
    }
    return JSONResponse(body)


@router.get("/api/campaigns/{campaign_id}/track-record/runs")
def track_record_runs(
    campaign_id: str,
    coverage: float = Query(default=0.9),
    band: Literal["proposal", "global"] = Query(default="proposal"),
    round: int | None = Query(default=None, ge=0, le=1_000_000),  # noqa: A002 - the API's field name
    limit: int = Query(default=RUNS_DEFAULT_LIMIT, ge=1, le=RUNS_MAX_LIMIT),
    after_id: int | None = Query(default=None, ge=0),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """One row per recorded run, oldest first (keyset `after_id`)."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    level = _level(coverage)
    if level is None:
        return _coverage_error()
    tenant = principal.tenant
    try:
        store._check(tenant, campaign_id)
        with store._txn(write=False) as conn:
            row = store._row(conn, tenant, campaign_id)
            if row is None:
                raise CampaignNotFound()
            state = row.state()
            fetched = track_rows(conn, tenant=tenant, campaign_id=campaign_id, after_id=after_id,
                                 round_=round, limit=limit + 1)
            recipes = recipes_for(conn, tenant=tenant, campaign_id=campaign_id,
                                  ids=[int(r["id"]) for r in fetched[:limit]])
    except CampaignNotFound:
        return campaign_not_found_response()
    pending = (state or {}).get("pending")
    in_flight = {str(r.get("id")) for r in pending if isinstance(r, dict)} if isinstance(pending, list) else set()
    items = [
        _run_item(r, score_run(r, coverage=level, band=band, in_flight=in_flight), recipes.get(int(r["id"])))
        for r in fetched[:limit]
    ]
    has_more = len(fetched) > limit
    return JSONResponse({
        "campaign_id": campaign_id,
        "basis": "prospective",
        "coverage": level,
        "band": band,
        "items": items,
        "has_more": has_more,
        "next_after_id": items[-1]["id"] if has_more and items else None,
        "rules": RULES,
        "excluded_reasons": EXCLUDED_REASONS,
    })


@router.get("/api/campaigns/{campaign_id}/runs/{run_id}/history")
def run_history(
    campaign_id: str,
    run_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """One run: its prediction recorded at start, every result version
    (corrections and reasons), and the fold that put it into the model."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        return JSONResponse(_RUN_NOT_FOUND, status_code=404)
    tenant = principal.tenant
    try:
        store._check(tenant, campaign_id)
        with store._txn(write=False) as conn:
            row = store._row(conn, tenant, campaign_id)
            if row is None:
                raise CampaignNotFound()
            state = row.state()
            pred = prediction_for_run(conn, tenant=tenant, campaign_id=campaign_id, run_id=run_id)
            versions = result_versions(conn, tenant=tenant, campaign_id=campaign_id, run_id=run_id,
                                       limit=MAX_RESULT_VERSIONS + 1)
            fold = fold_for_run(conn, tenant=tenant, campaign_id=campaign_id, run_id=run_id)
            recipe = (recipes_for(conn, tenant=tenant, campaign_id=campaign_id, ids=[int(pred["id"])])
                      .get(int(pred["id"])) if pred is not None else None)
    except CampaignNotFound:
        return campaign_not_found_response()
    pending = (state or {}).get("pending")
    in_flight = {str(r.get("id")) for r in pending if isinstance(r, dict)} if isinstance(pending, list) else set()
    if pred is None and not versions and run_id not in in_flight:
        return JSONResponse(_RUN_NOT_FOUND, status_code=404)
    for v in versions:
        v["basis"] = "measured"
        v["is_correction"] = v["previous_value"] is not None
    return JSONResponse({
        "campaign_id": campaign_id,
        "run_id": run_id,
        "round": pred.get("round") if pred else None,
        "recipe": recipe,
        "status": ("folded" if fold else "measured" if versions else "awaiting")
        if (fold or run_id in in_flight) else "discarded",
        "prediction": _prediction_view(pred) if pred is not None else None,
        "prediction_unavailable_reason": None if pred is not None else (
            "no prediction was recorded for this run (it was started before the engine kept "
            "prediction history)"),
        "result_versions": versions[:MAX_RESULT_VERSIONS],
        "result_versions_truncated": len(versions) > MAX_RESULT_VERSIONS,
        "fold": fold,
    })


@router.get("/api/campaigns/{campaign_id}/analyses")
def analysis_history(
    campaign_id: str,
    limit: int = Query(default=SNAPSHOTS_DEFAULT_LIMIT, ge=1, le=SNAPSHOTS_MAX_LIMIT),
    before_id: int | None = Query(default=None, ge=1),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Every analysis this campaign published, newest first (summaries only)."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    tenant = principal.tenant
    try:
        store._check(tenant, campaign_id)
        with store._txn(write=False) as conn:
            if store._row(conn, tenant, campaign_id) is None:
                raise CampaignNotFound()
            page = snapshot_page(conn, tenant=tenant, campaign_id=campaign_id,
                                 before_id=before_id, limit=limit)
    except CampaignNotFound:
        return campaign_not_found_response()
    return JSONResponse({
        "campaign_id": campaign_id, "limit": limit, **page,
        "note": "append-only analysis_snapshots (schema v8); each item is one published analysis; "
                "fetch one with /analyses/{id}. Analyses published before v8 kept only the newest "
                "(kind migrated_v7).",
    })


@router.get("/api/campaigns/{campaign_id}/analyses/{snapshot_id}")
def analysis_snapshot(
    campaign_id: str,
    snapshot_id: int,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """One stored analysis, as published, with its summary and sha256."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    tenant = principal.tenant
    try:
        store._check(tenant, campaign_id)
        with store._txn(write=False) as conn:
            if store._row(conn, tenant, campaign_id) is None:
                raise CampaignNotFound()
            found = snapshot_blob(conn, tenant=tenant, campaign_id=campaign_id,
                                  snapshot_id=snapshot_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    if found is None:
        return JSONResponse(_SNAPSHOT_NOT_FOUND, status_code=404)
    summary, blob = found
    try:
        analysis = json.loads(blob)
    except ValueError:
        analysis = None
    return JSONResponse({"campaign_id": campaign_id, "snapshot": summary, "analysis": analysis})
