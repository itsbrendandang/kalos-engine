"""Kalos portal - the label-only clone registry (docs/CLONES.md).

Routes:

- ``GET  /api/campaigns/{id}/clones``                          (read)  registry + labels seen in data
- ``GET  /api/campaigns/{id}/clones/summary``                  (read)  analysis `strain_summary` joined with the registry
- ``PUT  /api/campaigns/{id}/clones/{label}``                  (write) create/replace one clone (labels only)
- ``POST /api/campaigns/{id}/clones/{label}/titer-history``    (write) append one measured row
- ``GET  /api/campaigns/{id}/clones/{label}/titer-history``    (read)  newest-first page

Same security rules as `kalos.portal.gates_api`: the scope is checked before the
id is looked at; the tenant is always `principal.tenant`; malformed, missing and
foreign campaign ids get the byte-identical `campaign_not_found_response()`;
errors never echo caller input (no label, key or value); archived campaigns
answer 409 on writes; every successful write has its audit row in the SAME
transaction, refused writes a best-effort one.

Labels only: the PUT body is an allowlist (`extra="forbid"`); `sequence`,
`fasta`, `smiles`, `pdb`, `uniprot` and any other key are refused with
`invalid_body`. The summary never orders clones by a computed score:
`registry_order` is lead, backup, candidate, retired, then unregistered, by name.
"""
from __future__ import annotations

import copy
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import (
    AllowInfNan,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)
from pydantic import Strict as _Strict

from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign import (
    CampaignArchived,
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    get_campaign_store,
)
from kalos.portal.campaign_routes import (
    campaign_error_response,
    campaign_not_found_response,
)
from kalos.portal.campaigns_api import _audit, _auth_method, _valid_campaign_id
from kalos.portal.clone_registry import (
    CLONE_FIELD_KEYS,
    FORBIDDEN_KEYS,
    HISTORY_DEFAULT_LIMIT,
    HISTORY_MAX_LIMIT,
    LABELS_ONLY_REASON,
    MAX_CLONES_PER_CAMPAIGN,
    MAX_FIELD_CHARS,
    MAX_NOTE_CHARS,
    MAX_REASON_CHARS,
    MAX_SOURCE_NOTE_CHARS,
    MAX_TITER_ROWS_PER_CAMPAIGN,
    STATUS_ORDER,
    CloneNotRegistered,
    CloneValidationError,
    TiterHistoryLimit,
    TooManyClones,
    append_titer,
    check_clone_label,
    list_clones,
    list_titer_history,
    put_clone,
)

router = APIRouter()

_ARCHIVED_BODY = {
    "error": "the campaign is archived; it can be read but not changed", "code": "campaign_archived",
}
_NOT_REGISTERED_BODY = {
    "error": "this clone is not registered in the campaign",
    "code": "clone_not_registered",
    "what_would_fix_it": "register the clone first with PUT /api/campaigns/{id}/clones/{label}",
}
_ANONYMIZED_REASON = "anonymized analysis: labels are aliases"

_Num = Annotated[float, _Strict(), AllowInfNan(False)]
_Short = Annotated[str, _Strict(), StringConstraints(max_length=MAX_FIELD_CHARS)]


class _CloneBody(BaseModel):
    """PUT body: the WHOLE registry row (a replace). Labels only."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["lead", "backup", "retired", "candidate"]
    host: _Short | None = None
    parent_cell_bank: _Short | None = None
    construct_id: _Short | None = None
    phenotype_label: _Short | None = None
    product: _Short | None = None
    passage_at_banking: _Num | _Short | None = None
    note: Annotated[str, _Strict(), StringConstraints(max_length=MAX_NOTE_CHARS)] | None = None


class _TiterBody(BaseModel):
    """POST body: one measured value. Who recorded it and when come from the
    principal and the server clock."""

    model_config = ConfigDict(extra="forbid")

    measured_on: Annotated[str, _Strict(), StringConstraints(max_length=10)] | None = None
    basis: Literal["harvest_titer", "qP", "space_time_yield"]
    value: _Num
    unit: Literal["g/L", "mg/L", "pg/cell/day", "g/L/day"]
    scale_label: _Short | None = None
    age_kind: Literal["passage", "generation", "pdl"] | None = None
    age_value: _Num | None = None
    source_note: Annotated[str, _Strict(), StringConstraints(max_length=MAX_SOURCE_NOTE_CHARS)] | None = None
    previous_id: Annotated[int, _Strict(), Field(ge=1)] | None = None
    reason: Annotated[str, _Strict(), StringConstraints(max_length=MAX_REASON_CHARS)] | None = None


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, **extra}, status_code=status)


def _invalid_body(message: str) -> JSONResponse:
    return _error(400, "invalid_body", message)


def _parse(model: type[BaseModel], payload: Any) -> BaseModel | JSONResponse:
    """Validate `payload` against `model`. A refusal names only allowlisted field
    names and pydantic's input-free message; never a submitted key or value."""
    if not isinstance(payload, dict):
        return _invalid_body("invalid request body: the body must be a JSON object")
    if any(k in FORBIDDEN_KEYS or k not in model.model_fields for k in payload):
        return _invalid_body(f"invalid request body: {LABELS_ONLY_REASON}")
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        parts = []
        for err in exc.errors(include_input=False, include_url=False)[:5]:
            loc = err.get("loc", ())
            name = str(loc[0]) if loc and loc[0] in model.model_fields else "body"
            parts.append(f"{name}: {err.get('msg', 'invalid')}"[:200])
        return _invalid_body("invalid request body: " + "; ".join(parts))


def _store_error(exc: CampaignError) -> JSONResponse:
    if isinstance(exc, CampaignArchived):
        return JSONResponse(_ARCHIVED_BODY, status_code=409)
    return campaign_error_response(exc)


def _valid_label(label: str) -> bool:
    try:
        check_clone_label(label)
    except CloneValidationError:
        return False
    return True


def _bad_label() -> JSONResponse:
    return _error(400, "invalid_label",
                  "the clone label must be 1-64 characters, not blank, without control characters")


def _is_anonymized(analysis: dict[str, Any]) -> bool:
    run = analysis.get("run")
    return isinstance(run, dict) and run.get("anonymize") is True


def _strain_block(analysis: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """(the available strain_summary, None) or (None, why it is unavailable)."""
    block = analysis.get("strain_summary")
    if not isinstance(block, dict):
        return None, "the latest analysis has no strain_summary (no strain/clone role was set)"
    if block.get("available") is not True:
        reason = block.get("reason")
        return None, (reason if isinstance(reason, str) and reason
                      else "the latest analysis reports the strain summary as unavailable")
    return block, None


def _level_labels(block: dict[str, Any]) -> list[str]:
    levels = block.get("levels")
    if not isinstance(levels, list):
        return []
    out: list[str] = []
    for lvl in levels:
        if isinstance(lvl, dict) and isinstance(lvl.get("label"), str) and lvl["label"] not in out:
            out.append(lvl["label"])
    return out


def _clones_view(store: CampaignStore, tenant: str, campaign_id: str) -> dict[str, Any]:
    clones = list_clones(store, tenant=tenant, campaign_id=campaign_id)
    analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id)
    seen: list[str] = []
    reason: str | None = None
    if not analysis:
        reason = "the campaign has no analysis yet"
    elif _is_anonymized(analysis):
        reason = _ANONYMIZED_REASON
    else:
        block, reason = _strain_block(analysis)
        if block is not None:
            seen = _level_labels(block)
    registered = {c["label"] for c in clones}
    body: dict[str, Any] = {
        "campaign_id": campaign_id,
        "clones": clones,
        "seen_in_data": seen,
        "unregistered_labels": sorted(lbl for lbl in seen if lbl not in registered),
        "max_clones": MAX_CLONES_PER_CAMPAIGN,
    }
    if reason is not None:
        body["seen_in_data_reason"] = reason
    return body


# --- routes: the literal `/clones/summary` path is registered BEFORE the
# `/clones/{label}` routes. Only GET is defined on it, and no GET exists on
# `/clones/{label}`, so `summary` is never captured as a label on GET. A PUT to
# `/clones/summary` registers a clone literally labelled "summary" (harmless).


@router.get("/api/campaigns/{campaign_id}/clones")
def get_clones(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """The registry, plus the clone labels in the latest analysis's
    strain_summary (`seen_in_data`) and those not yet registered."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        return JSONResponse(_clones_view(store, principal.tenant, campaign_id))
    except CampaignNotFound:
        return campaign_not_found_response()


@router.get("/api/campaigns/{campaign_id}/clones/summary")
def get_clones_summary(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """The latest analysis's `strain_summary`, each level joined with its
    registry row (`registry: {status, fields} | null`). Never joined on an
    anonymized analysis (its labels are aliases)."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    tenant = principal.tenant
    try:
        clones = list_clones(store, tenant=tenant, campaign_id=campaign_id)
        analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    if not analysis:
        return _error(409, "no_analysis", "the campaign has no analysis yet",
                      what_would_fix_it="upload a run sheet with a clone column and analyze it")
    block, reason = _strain_block(analysis)
    if block is None:
        return JSONResponse({"campaign_id": campaign_id, "available": False, "reason": reason})
    out = copy.deepcopy(block)
    out["campaign_id"] = campaign_id
    if _is_anonymized(analysis):
        out["registry_join"] = {"applied": False, "reason": _ANONYMIZED_REASON}
        return JSONResponse(out)
    by_label = {c["label"]: c for c in clones}
    for lvl in out.get("levels") or []:
        if isinstance(lvl, dict):
            reg = by_label.get(lvl.get("label")) if isinstance(lvl.get("label"), str) else None
            lvl["registry"] = {"status": reg["status"], "fields": reg["fields"]} if reg else None
    labels = _level_labels(out)
    unregistered = sorted(lbl for lbl in labels if lbl not in by_label)
    registered = sorted(
        (lbl for lbl in labels if lbl in by_label),
        key=lambda lbl: (STATUS_ORDER.get(by_label[lbl]["status"], len(STATUS_ORDER)), lbl),
    )
    out["unregistered_labels"] = unregistered
    out["registry_order"] = registered + unregistered
    out["registry_join"] = {"applied": True, "reason": None}
    return JSONResponse(out)


@router.put("/api/campaigns/{campaign_id}/clones/{label}")
def put_clone_route(
    campaign_id: str,
    label: str,
    payload: Any = Body(default=None),
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Create or replace one clone's registry row (labels only). The audit row
    records the label, status and the keys sent, never field values."""
    action = "campaign.clone.put"
    if not _valid_campaign_id(campaign_id):
        _audit(store, principal, action, 404, None)
        return campaign_not_found_response()
    if not _valid_label(label):
        _audit(store, principal, action, 400, campaign_id, detail="code=invalid_label")
        return _bad_label()
    parsed = _parse(_CloneBody, payload)
    if isinstance(parsed, JSONResponse):
        _audit(store, principal, action, 400, campaign_id, detail="code=invalid_body")
        return parsed
    assert isinstance(parsed, _CloneBody)
    data = parsed.model_dump(exclude_none=True)
    status = data.pop("status")
    try:
        row = put_clone(
            store, tenant=principal.tenant, campaign_id=campaign_id, label=label, status=status,
            fields={k: v for k, v in data.items() if k in CLONE_FIELD_KEYS},
            updated_by=principal.subject, auth_method=_auth_method(principal), audit_action=action,
        )
    except CloneValidationError as exc:
        _audit(store, principal, action, 400, campaign_id, detail=f"code={exc.code}")
        return _error(400, exc.code, exc.reason)
    except CampaignError as exc:
        response = _store_error(exc)
        _audit(store, principal, action, response.status_code, campaign_id)
        return response
    except TooManyClones:
        _audit(store, principal, action, 409, campaign_id, detail="code=too_many_clones")
        return _error(409, "too_many_clones",
                      f"a campaign can register at most {MAX_CLONES_PER_CAMPAIGN} clones",
                      what_would_fix_it="edit an existing clone (for example mark it retired) "
                                        "instead of registering a new one",
                      max_clones=MAX_CLONES_PER_CAMPAIGN)
    return JSONResponse({"campaign_id": campaign_id, "clone": row})


@router.post("/api/campaigns/{campaign_id}/clones/{label}/titer-history")
def post_titer_history(
    campaign_id: str,
    label: str,
    payload: Any = Body(default=None),
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Append one measured value to a registered clone's history (append-only).
    A correction carries `previous_id` and a non-empty `reason`."""
    action = "campaign.clone.titer_append"
    if not _valid_campaign_id(campaign_id):
        _audit(store, principal, action, 404, None)
        return campaign_not_found_response()
    if not _valid_label(label):
        _audit(store, principal, action, 400, campaign_id, detail="code=invalid_label")
        return _bad_label()
    parsed = _parse(_TiterBody, payload)
    if isinstance(parsed, JSONResponse):
        _audit(store, principal, action, 400, campaign_id, detail="code=invalid_body")
        return parsed
    try:
        row = append_titer(
            store, tenant=principal.tenant, campaign_id=campaign_id, label=label,
            row=parsed.model_dump(exclude_none=True), actor=principal.subject,
            auth_method=_auth_method(principal), audit_action=action,
        )
    except CloneValidationError as exc:
        _audit(store, principal, action, 400, campaign_id, detail=f"code={exc.code}")
        return _error(400, exc.code, exc.reason)
    except CampaignError as exc:
        response = _store_error(exc)
        _audit(store, principal, action, response.status_code, campaign_id)
        return response
    except CloneNotRegistered:
        _audit(store, principal, action, 404, campaign_id, detail="code=clone_not_registered")
        return JSONResponse(_NOT_REGISTERED_BODY, status_code=404)
    except TiterHistoryLimit:
        _audit(store, principal, action, 409, campaign_id, detail="code=titer_history_limit")
        return _error(409, "titer_history_limit",
                      f"this campaign has reached its limit of {MAX_TITER_ROWS_PER_CAMPAIGN} "
                      "titer history rows")
    return JSONResponse({"campaign_id": campaign_id, "row": row})


@router.get("/api/campaigns/{campaign_id}/clones/{label}/titer-history")
def get_titer_history(
    campaign_id: str,
    label: str,
    limit: int = Query(default=HISTORY_DEFAULT_LIMIT, ge=1, le=HISTORY_MAX_LIMIT),
    before_id: int | None = Query(default=None, ge=1),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """A registered clone's measured history, newest first, `before_id` cursor."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    if not _valid_label(label):
        return _bad_label()
    try:
        page = list_titer_history(store, tenant=principal.tenant, campaign_id=campaign_id,
                                  label=label, before_id=before_id, limit=limit)
    except CampaignNotFound:
        return campaign_not_found_response()
    except CloneNotRegistered:
        return JSONResponse(_NOT_REGISTERED_BODY, status_code=404)
    return JSONResponse({
        "campaign_id": campaign_id, "label": label, "limit": limit, "history": page,
        "notes": "append-only measured rows; a row with previous_id is a correction of that row",
    })
