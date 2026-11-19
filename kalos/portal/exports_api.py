"""Kalos portal - report exports for one campaign ("you own your data").

Routes (all campaign-scoped, `read` scope except `POST .../operating-ranges`,
which is `write`; see docs in the route docstrings):

- `GET  /api/campaigns/{id}/export/rows?format=csv|json`: the campaign's
  ORIGINAL uploaded rows, and, as a separate, labelled section, the rows folded
  in later from measured runs.
- `GET  /api/campaigns/{id}/results/history`: every recorded result version of
  every run (the append-only `run_results` table, corrections included) and
  the campaign's audit-log entries, paginated newest first.
- `GET|POST /api/campaigns/{id}/operating-ranges`: a model-based operating
  range, computed only when defensible (kalos/portal/operating_ranges.py).
  GET is `read`; POST is `write` (it occupies the analysis slot, and the web
  mints `write` for every non-GET request; docs/route-scopes.json).

Security rules are those of `kalos.portal.campaigns_api`: scope is checked
before the id is looked at; the tenant is always `principal.tenant`; malformed,
missing and foreign ids give the byte-identical `campaign_not_found_response()`;
every SQL statement carries `tenant = ?`; caller input is never echoed.

FOLD ORDERING (verified in `CampaignStore.seed` / `commit_fold`): `seed` writes
`base_rows` = the uploaded rows and `history = [{round: 0, n_base: len(rows)}]`.
Each `commit_fold` APPENDS every measured run (`{**recipe, target: result}`) to
the end of `base_rows`, increments `round` and appends `{round, n_base:
len(base_rows)}` to `history`. Rows are never reordered or removed, so rows
`[0, history[0].n_base)` are the upload and rows `[history[k-1].n_base,
history[k].n_base)` were folded in at round `history[k].round`. The split is
reported only when the history is internally consistent with the rows
(`_partition_rows`); otherwise `original_rows_identifiable: false` with a
reason and every row is exported as `origin: "unknown"`. Nothing is guessed.

ANONYMIZE: when the campaign's analysis was computed with `anonymize=True`
(`analysis.run.anonymize`, carried forward by re-analysis), the stored rows
still hold the uploader's raw column names and cells. The export then applies
the analysis's OWN aliases (`kalos.portal.analysis_parts.anonymize`): an
identifier-type column name becomes `col_<hash8>`, every value of an
identifier-type column becomes `level_<hash8>` (the analysis's categorical
level alias, extended to every value of those columns), and every TEXT cell of
a column the analysis does not expose as a non-identifier model input or as the
target becomes `level_<hash8>` too (the analysis never shows those labels).
Numbers in non-identifier columns are kept. The alias mapping is first
verified against the stored analysis (every column name it shows, and every
aliased categorical level in its proposals, must be reproducible); if it is
not, or the anonymize setting is not recorded, the export is refused with 409
`anonymized_export_unavailable`. The uploader's filename is never included.

CSV FORMULA INJECTION: a TEXT cell or header starting with `=`, `+`, `-`, `@`,
TAB or CR gets a leading apostrophe (`'`), so a spreadsheet shows it as text
instead of evaluating it. JSON numbers are written as numbers and are never
prefixed (a negative number is data, not a formula). The JSON format returns
values unmodified.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import logging
import math
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from pydantic import Strict as _Strict

from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.busy import RETRY_AFTER_SECONDS, AnalysisBusy, run_in_slot
from kalos.portal.campaign import CampaignNotFound, CampaignStore, get_campaign_store
from kalos.portal.campaign_routes import campaign_not_found_response
from kalos.portal.campaigns_api import _audit, _valid_campaign_id

log = logging.getLogger("kalos.portal")

router = APIRouter()

# Size cap of one export (rows x columns). Base rows are bounded by the upload
# caps plus folded runs; this refuses a pathological campaign instead of building
# an unbounded response.
EXPORT_MAX_CELLS = 2_000_000
HISTORY_DEFAULT_LIMIT = 100
HISTORY_MAX_LIMIT = 500
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
ORIGIN_UPLOAD = "upload"
ORIGIN_MEASURED = "measured_run"
ORIGIN_UNKNOWN = "unknown"


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, **extra}, status_code=status)


def _load(store: CampaignStore, tenant: str, campaign_id: str) -> tuple[dict | None, dict | None]:
    """(state, analysis) of one of `tenant`'s campaigns. CampaignNotFound otherwise."""
    state = store.get(tenant=tenant, campaign_id=campaign_id)
    analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id)
    return state, analysis


# --- (a) row export -------------------------------------------------------------- #


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _partition_rows(state: dict[str, Any]) -> dict[str, Any]:
    """Split `base_rows` into the upload and the fold rounds, or say why not."""
    rows = state.get("base_rows")
    rows = rows if isinstance(rows, list) else []
    history = state.get("history")

    def unknown(reason: str) -> dict[str, Any]:
        return {"identifiable": False, "reason": reason, "segments": [(ORIGIN_UNKNOWN, None, rows)]}

    if not isinstance(history, list) or not history:
        return unknown("the campaign has no recorded fold history (it predates it)")
    points = [(_int(h.get("round")), _int(h.get("n_base"))) if isinstance(h, dict) else (None, None)
              for h in history]
    if any(r is None or n is None for r, n in points):
        return unknown("the campaign's fold history is malformed")
    if [r for r, _ in points] != list(range(len(points))):
        return unknown("the campaign's fold history does not start at the upload (round 0) "
                       "or skips a round")
    sizes = [int(n) for _, n in points if n is not None]
    if any(b < a for a, b in itertools.pairwise(sizes)) or sizes[0] < 0:
        return unknown("the campaign's fold history is not monotone")
    if sizes[-1] != len(rows):
        return unknown("the campaign's fold history does not match its row count")
    if _int(state.get("round")) != len(points) - 1:
        return unknown("the campaign's round does not match its fold history")
    segments: list[tuple[str, int | None, list]] = [(ORIGIN_UPLOAD, 0, rows[: sizes[0]])]
    for k in range(1, len(sizes)):
        segments.append((ORIGIN_MEASURED, k, rows[sizes[k - 1]: sizes[k]]))
    return {"identifiable": True, "reason": None, "segments": segments}


def _column_alias(name: str) -> str:
    """`kalos.portal.analysis_parts.anonymize._anonymize_result`'s `alias`."""
    from kalos.data.anonymizer import _hash
    from kalos.portal.analysis_parts.anonymize import _is_identifier_name

    return f"col_{_hash(name)[:8]}" if _is_identifier_name(name) else name


def _level_alias(column: str, value: object) -> str:
    """The analysis's `alias_level` for a categorical label of `column`."""
    from kalos.data.anonymizer import _hash

    text = value.strip() if isinstance(value, str) else str(value)
    return f"level_{_hash(column + chr(31) + text)[:8]}"


class AnonymizedExportUnavailable(Exception):
    """The analysis's alias mapping cannot be reproduced exactly: refuse, never leak."""


def _anonymize_plan(raw_columns: list[str], analysis: dict[str, Any]) -> dict[str, Any]:
    """Verify and return the alias mapping for an anonymized campaign."""
    from kalos.portal.analysis_parts.anonymize import _is_identifier_name

    provenance = analysis.get("provenance")
    shown = {str(r["name"]) for r in provenance if isinstance(r, dict) and "name" in r} \
        if isinstance(provenance, list) else set()
    if not shown:
        raise AnonymizedExportUnavailable(
            "the analysis carries no column provenance to verify its aliases against")
    col_map = {c: _column_alias(c) for c in raw_columns}
    produced = set(col_map.values())
    if not shown <= produced:
        raise AnonymizedExportUnavailable(
            "the analysis shows column names this export cannot reproduce (the alias salt or "
            "the column set changed since it was computed)")
    by_alias = {a: c for c, a in col_map.items()}
    features = [str(f) for f in analysis.get("features") or []]
    cats = {str(c) for c in analysis.get("categorical_features") or []}
    target = analysis.get("target")
    # Columns whose text labels the analysis shows as-is: non-identifier inputs and the target.
    exposed = {
        by_alias[f] for f in [*features, *([target] if isinstance(target, str) else [])]
        if f in by_alias and not _is_identifier_name(by_alias[f])
    }
    # Aliased categorical levels in the proposals must be reproducible too.
    for alias in cats:
        raw = by_alias.get(alias)
        if raw is None or not _is_identifier_name(raw):
            continue
        for prop in analysis.get("proposals") or []:
            recipe = prop.get("recipe") if isinstance(prop, dict) else None
            if isinstance(recipe, dict) and alias in recipe:
                value = recipe[alias]
                if isinstance(value, str) and not value.startswith("level_"):
                    raise AnonymizedExportUnavailable(
                        "the analysis shows an identifier label this export cannot reproduce")
    return {"col_map": col_map, "aliases": produced, "exposed": exposed}


def _anon_cell(column: str, value: Any, plan: dict[str, Any]) -> Any:
    from kalos.portal.analysis_parts.anonymize import _is_identifier_name

    if value is None:
        return None
    if _is_identifier_name(column):
        return _level_alias(column, value)
    if isinstance(value, str) and column not in plan["exposed"]:
        return _level_alias(column, value)
    return value


def _transform_row(row: dict[str, Any], plan: dict[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return dict(row)
    out: dict[str, Any] = {}
    for key, value in row.items():
        name = str(key)
        if name in plan["aliases"] and name not in plan["col_map"]:
            out[name] = value  # already an alias (a folded key written under it)
            continue
        out[plan["col_map"].get(name, _column_alias(name))] = _anon_cell(name, value, plan)
    return out


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(str(key), None)
    return list(seen)


def _anonymize_flag(analysis: dict[str, Any] | None) -> bool | None:
    run = (analysis or {}).get("run")
    if isinstance(run, dict) and isinstance(run.get("anonymize"), bool):
        return bool(run["anonymize"])
    return None


def _provenance(state: dict[str, Any], analysis: dict[str, Any] | None, anonymized: bool) -> dict:
    from kalos import __version__ as engine_version

    analysis = analysis or {}
    run_block = analysis.get("run")
    run: dict[str, Any] = run_block if isinstance(run_block, dict) else {}
    prov = analysis.get("proposal_provenance")
    kind = run.get("kind")
    return {
        "campaign_generation": state.get("generation"),
        "campaign_round": state.get("round"),
        "analysis_generation": analysis.get("campaign_generation"),
        "analysis_in_sync": analysis.get("campaign_generation") is not None
        and analysis.get("campaign_generation") == state.get("generation"),
        "analysis_engine_version": analysis.get("engine_version"),
        "export_engine_version": engine_version,
        "analysis_run_kind": kind,
        "analysis_input_sha256": run.get("input_sha256"),
        "analysis_input_describes": (
            "the uploaded file bytes" if kind == "upload"
            else "the folded dataset as CSV (run_record.fold_bytes)" if kind == "reanalyze"
            else None
        ),
        "provenance_sha256": prov.get("provenance_sha256") if isinstance(prov, dict) else None,
        # Under anonymize the stored label is never echoed (it could predate the alias rule).
        "dataset": None if anonymized else analysis.get("dataset"),
        "anonymized": anonymized,
        "exported_at": time.time(),
    }


def _neutralize(text: str) -> str:
    return "'" + text if text.startswith(CSV_FORMULA_PREFIXES) else text


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return _neutralize(str(value))
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) and math.isfinite(value) else str(value)
    return _neutralize(str(value))


def _build_export(state: dict[str, Any], analysis: dict[str, Any] | None) -> dict[str, Any]:
    """The JSON export body. Raises AnonymizedExportUnavailable."""
    anonymize = _anonymize_flag(analysis)
    if anonymize is None:
        raise AnonymizedExportUnavailable(
            "the campaign's analysis does not record whether it was anonymized")
    parts = _partition_rows(state)
    all_rows = [r for _, _, seg in parts["segments"] for r in seg if isinstance(r, dict)]
    plan = _anonymize_plan(_columns(all_rows), analysis or {}) if anonymize else None
    target = state.get("target")
    if plan is not None and isinstance(target, str):
        target = target if target in plan["aliases"] else plan["col_map"].get(target, _column_alias(target))
    original: list[dict] | None = None
    folded: list[dict] | None = None
    unattributed: list[dict] | None = None
    for origin, rnd, seg in parts["segments"]:
        out = [_transform_row(r, plan) for r in seg if isinstance(r, dict)]
        if origin == ORIGIN_UPLOAD:
            original = out
        elif origin == ORIGIN_MEASURED:
            folded = (folded or []) + [
                {"origin": ORIGIN_MEASURED, "folded_in_round": rnd, "values": r} for r in out
            ]
        else:
            unattributed = out
    if parts["identifiable"] and folded is None:
        folded = []
    columns = _columns([*(original or []), *[f["values"] for f in folded or []], *(unattributed or [])])
    return {
        "target": target,
        "columns": columns,
        "original_rows_identifiable": parts["identifiable"],
        "original_rows_reason": parts["reason"],
        "original_rows": (
            {"origin": ORIGIN_UPLOAD, "round": 0, "n": len(original), "rows": original}
            if original is not None else None
        ),
        "folded_rows": (
            {"origin": ORIGIN_MEASURED, "n": len(folded), "rows": folded}
            if folded is not None else None
        ),
        "unattributed_rows": (
            {"origin": ORIGIN_UNKNOWN, "n": len(unattributed), "rows": unattributed}
            if unattributed is not None else None
        ),
        "anonymization": (
            {
                "applied": True,
                "column_alias": "identifier-type column names -> col_<hash8> (same as the analysis)",
                "value_alias": (
                    "identifier-type column values, and text cells of columns the analysis does not "
                    "show as a non-identifier input or target -> level_<hash8>"
                ),
            } if plan is not None else {"applied": False}
        ),
        "csv_formula_neutralization": (
            "CSV only: text cells and headers starting with = + - @ TAB or CR get a leading apostrophe"
        ),
        "provenance": _provenance(state, analysis, plan is not None),
    }


def _csv_body(export: dict[str, Any]) -> str:
    columns = export["columns"]
    origin_col, round_col = "kalos_origin", "kalos_folded_in_round"
    while origin_col in columns or round_col in columns:
        origin_col, round_col = "_" + origin_col, "_" + round_col
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow([origin_col, round_col, *(_neutralize(c) for c in columns)])

    def emit(origin: str, rnd: int | None, row: dict[str, Any]) -> None:
        writer.writerow([origin, "" if rnd is None else str(rnd),
                         *(_csv_cell(row.get(c)) for c in columns)])

    for row in (export["original_rows"] or {}).get("rows", []):
        emit(ORIGIN_UPLOAD, 0, row)
    for item in (export["folded_rows"] or {}).get("rows", []):
        emit(ORIGIN_MEASURED, item["folded_in_round"], item["values"])
    for row in (export["unattributed_rows"] or {}).get("rows", []):
        emit(ORIGIN_UNKNOWN, None, row)
    return buf.getvalue()


@router.get("/api/campaigns/{campaign_id}/export/rows")
def export_rows(
    campaign_id: str,
    format: Literal["csv", "json"] = Query(default="json"),  # noqa: A002 - the documented query name
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> Response:
    """The campaign's uploaded rows and, separately labelled, the rows folded in
    from measured runs. See the module docstring for the split, anonymize and
    CSV rules. Each export (success or refusal) writes an audit row."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        state, analysis = _load(store, principal.tenant, campaign_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    response = _export_response(campaign_id, state, analysis, format)
    _audit(store, principal, "campaign.export_rows", response.status_code, campaign_id,
           detail=f"format={format}")
    return response


def _export_response(
    campaign_id: str, state: dict | None, analysis: dict | None, fmt: str,
) -> Response:
    if state is None:
        if fmt == "json":
            return JSONResponse({"campaign_id": campaign_id, "has_data": False})
        return _error(409, "no_data", "the campaign has no data to export")
    rows = state.get("base_rows") or []
    n_cols = len(_columns([r for r in rows if isinstance(r, dict)])) if rows else 0
    if len(rows) * max(n_cols, 1) > EXPORT_MAX_CELLS:
        return _error(413, "export_too_large", "the campaign is too large to export in one response",
                      max_cells=EXPORT_MAX_CELLS)
    try:
        export = _build_export(state, analysis)
    except AnonymizedExportUnavailable as exc:
        return _error(409, "anonymized_export_unavailable",
                      f"the export is refused so no hidden identifier can leak: {exc}",
                      what_would_fix_it="re-analyze the campaign so its analysis records the "
                                        "anonymize setting with reproducible aliases")
    if fmt == "json":
        return JSONResponse({"campaign_id": campaign_id, "has_data": True, "format": "json", **export})
    body = _csv_body(export).encode("utf-8")
    prov = export["provenance"]
    headers = {
        "Content-Disposition": f'attachment; filename="{campaign_id}-rows.csv"',
        "X-Kalos-Content-Sha256": hashlib.sha256(body).hexdigest(),
        "X-Kalos-Original-Rows-Identifiable": "true" if export["original_rows_identifiable"] else "false",
        "X-Kalos-Anonymized": "true" if prov["anonymized"] else "false",
        "X-Kalos-Campaign-Generation": str(prov["campaign_generation"] or ""),
        "X-Kalos-Engine-Version": str(prov["export_engine_version"]),
        "X-Kalos-Analysis-Input-Sha256": str(prov["analysis_input_sha256"] or ""),
        "X-Kalos-Csv-Formula-Neutralized": "apostrophe-prefix",
        "Cache-Control": "no-store",
    }
    return Response(body, media_type="text/csv; charset=utf-8", headers=headers)


# --- (b) results history + audit -------------------------------------------------- #

_RESULT_COLUMNS = ("id", "run_id", "version", "value", "previous_value", "reason", "actor",
                   "auth_method", "recorded_at")
_AUDIT_COLUMNS = ("id", "ts", "subject", "auth_method", "action", "campaign_id", "job_id",
                  "outcome", "request_id", "detail")


def _page(
    store: CampaignStore, *, tenant: str, campaign_id: str, table: str, columns: tuple[str, ...],
    before_id: int | None, limit: int,
) -> dict[str, Any]:
    """One newest-first page of `table` for one of `tenant`'s campaigns."""
    sql = (
        f"SELECT {', '.join(columns)} FROM {table} WHERE tenant = ? AND campaign_id = ?"  # noqa: S608 - identifiers are module constants; values are bound parameters
        + (" AND id < ?" if before_id is not None else "")
        + " ORDER BY id DESC LIMIT ?"
    )
    params: tuple[Any, ...] = (tenant, campaign_id, *((before_id,) if before_id is not None else ()),
                               limit + 1)
    with store._txn(write=False) as conn:
        if store._row(conn, tenant, campaign_id) is None:
            raise CampaignNotFound()
        fetched = conn.execute(sql, params).fetchall()
    items = [dict(zip(columns, r)) for r in fetched[:limit]]
    has_more = len(fetched) > limit
    return {
        "items": items,
        "has_more": has_more,
        "next_before_id": items[-1]["id"] if has_more and items else None,
    }


@router.get("/api/campaigns/{campaign_id}/results/history")
def results_history(
    campaign_id: str,
    section: Literal["all", "results", "audit"] = Query(default="all"),
    limit: int = Query(default=HISTORY_DEFAULT_LIMIT, ge=1, le=HISTORY_MAX_LIMIT),
    results_before_id: int | None = Query(default=None, ge=1),
    audit_before_id: int | None = Query(default=None, ge=1),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Every recorded result version (corrections included) and the campaign's
    audit entries, each newest first with its own `before_id` cursor."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    tenant = principal.tenant
    body: dict[str, Any] = {"campaign_id": campaign_id, "limit": limit}
    try:
        store._check(tenant, campaign_id)
        if section in ("all", "results"):
            page = _page(store, tenant=tenant, campaign_id=campaign_id, table="run_results",
                         columns=_RESULT_COLUMNS, before_id=results_before_id, limit=limit)
            for item in page["items"]:
                item["is_correction"] = item["previous_value"] is not None
                item["basis"] = "measured"
            body["results"] = page
        if section in ("all", "audit"):
            body["audit"] = _page(store, tenant=tenant, campaign_id=campaign_id, table="audit_log",
                                  columns=_AUDIT_COLUMNS, before_id=audit_before_id, limit=limit)
    except CampaignNotFound:
        return campaign_not_found_response()
    body["notes"] = {
        "results": "append-only run_results rows; version 1 is the first value recorded, a row with "
                   "previous_value is a correction; actor is shown as stored",
        "audit": "audit_log rows for this campaign; subject and detail are shown as stored",
    }
    return JSONResponse(body)


# --- (c) operating ranges --------------------------------------------------------- #

_Probability = Annotated[float, _Strict(), Field(ge=0.5, le=0.99)]
_Threshold = Annotated[float, _Strict()]


class _RangesBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: _Threshold | None = None
    probability: _Probability = 0.9
    basis: Literal["target"] = "target"


async def _operating_ranges(
    campaign_id: str, body: _RangesBody, principal: Principal, store: CampaignStore,
) -> JSONResponse:
    from kalos.portal import operating_ranges as orng
    from kalos.portal.campaign_settings import get_setting

    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    tenant = principal.tenant
    try:
        state, analysis = _load(store, tenant, campaign_id)
        setup = get_setting(store, tenant=tenant, campaign_id=campaign_id, kind="setup")
    except CampaignNotFound:
        return campaign_not_found_response()
    threshold = body.threshold
    if threshold is not None and not math.isfinite(threshold):
        return _error(400, "invalid_body", "invalid request body: threshold must be a finite number")
    try:
        rel = orng.check_preconditions(state, analysis, threshold)
        if state is None or analysis is None or threshold is None:  # narrowed by the check above
            raise orng.NotComputed("no_data", "the campaign has no data yet", "upload a run sheet")
        goal, goal_source = orng.resolve_goal(setup, analysis)
        column_for, display_level = _ranges_mapping(state, analysis)
        plan = orng.build_plan(
            state, analysis, threshold=threshold, probability=body.probability,
            goal=goal, goal_source=goal_source, column_for=column_for, display_level=display_level,
        )
        try:
            result = await run_in_slot(lambda: orng.fit_and_score(plan), tenant=tenant)
        except AnalysisBusy as busy:
            return JSONResponse({"error": str(busy), "code": "engine_busy"}, status_code=503,
                                headers={"Retry-After": str(RETRY_AFTER_SECONDS)})
    except orng.NotComputed as nc:
        return JSONResponse({"campaign_id": campaign_id, **nc.body()})
    calibration = rel.get("calibration") if isinstance(rel.get("calibration"), dict) else None
    promotion_block = analysis.get("promotion")
    promotion: dict[str, Any] = promotion_block if isinstance(promotion_block, dict) else {}
    caveats = [
        "model-based: verify the range experimentally before relying on it",
        "categorical inputs are held at the best measured recipe's levels",
    ] if plan.cat_names else ["model-based: verify the range experimentally before relying on it"]
    z_std = (calibration or {}).get("z_std")
    if isinstance(z_std, (int, float)) and z_std > 1.0:
        caveats.append(
            f"held-out calibration z_std={z_std} > 1: the GP's bands were too narrow on held-out runs, "
            "so these probabilities may be overstated")
    return JSONResponse({
        "campaign_id": campaign_id,
        **result,
        "reliability": {
            "spearman": rel.get("spearman"), "ci95": rel.get("ci95"),
            "spearman_floor": rel.get("spearman_floor"), "clears_floor": rel.get("clears_floor"),
            "ci_excludes_zero": rel.get("ci_excludes_zero"),
            "calibration": calibration,
            "promotion_passed": promotion.get("passed"),
        },
        "analysis": {
            "campaign_generation": analysis.get("campaign_generation"),
            "engine_version": analysis.get("engine_version"),
            "provenance_sha256": (analysis.get("proposal_provenance") or {}).get("provenance_sha256")
            if isinstance(analysis.get("proposal_provenance"), dict) else None,
        },
        "caveats": caveats,
    })


def _ranges_mapping(state: dict[str, Any], analysis: dict[str, Any]) -> tuple[dict[str, str], Any]:
    """Map analysis names -> raw row keys (identity unless anonymized) and the
    label display function. Refuses (not_computed) when it cannot be reproduced."""
    from kalos.portal.analysis_parts.anonymize import _is_identifier_name
    from kalos.portal.operating_ranges import NotComputed

    rows = [r for r in state.get("base_rows") or [] if isinstance(r, dict)]
    raw_cols = _columns(rows)
    anonymize = _anonymize_flag(analysis)
    if anonymize is None:
        raise NotComputed("anonymized_mapping_unavailable",
                          "the analysis does not record whether it was anonymized, so labels "
                          "could not be shown safely",
                          "re-analyze the campaign")
    if not anonymize:
        return {c: c for c in raw_cols} | {str(analysis.get("target")): str(state.get("target"))}, \
            (lambda _col, label: label)
    try:
        plan = _anonymize_plan(raw_cols, analysis)
    except AnonymizedExportUnavailable as exc:
        raise NotComputed("anonymized_mapping_unavailable", str(exc), "re-analyze the campaign") from exc
    mapping = {a: c for c, a in plan["col_map"].items()}

    def display(col: str, label: str) -> str:
        return _level_alias(col, label) if _is_identifier_name(col) else label

    return mapping, display


@router.post("/api/campaigns/{campaign_id}/operating-ranges")
async def operating_ranges_post(
    campaign_id: str,
    body: _RangesBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Compute (never store) a model-based operating range. `write` scope: it
    changes nothing, but it occupies the analysis slot, and the web mints
    `write` for non-GET requests. Read-only callers use the GET form. CPU is
    bounded by the analysis slot and the sample cap."""
    return await _operating_ranges(campaign_id, body, principal, store)


@router.get("/api/campaigns/{campaign_id}/operating-ranges")
async def operating_ranges_get(
    campaign_id: str,
    threshold: float | None = Query(default=None),
    probability: float = Query(default=0.9, ge=0.5, le=0.99),
    basis: Literal["target"] = Query(default="target"),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """The GET form of the operating-ranges route (query parameters); `read` scope."""
    body = _RangesBody(threshold=threshold, probability=probability, basis=basis)
    return await _operating_ranges(campaign_id, body, principal, store)


__all__ = ["router"]
