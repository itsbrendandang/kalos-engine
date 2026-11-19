"""Kalos portal - per-campaign scale-up stage track and go/no-go gates.

Routes (API reference: the module docstring of `kalos.portal.stages` and the
contract in the web-redesign API doc):

- ``GET  /api/campaigns/{id}/stages``        (read)  stage definitions + gate status
- ``PUT  /api/campaigns/{id}/stages``        (write) replace definitions + current_stage
- ``GET  /api/campaigns/{id}/gates``         (read)  every gate: current decision + history
- ``PUT  /api/campaigns/{id}/gates/{gate}``  (write) record (append) a decision

Same security rules as `kalos.portal.campaigns_api`: scope checked before the
id is looked at; tenant is always `principal.tenant`; malformed, missing and
foreign campaign ids get the byte-identical `campaign_not_found_response()`;
caller input is never echoed in an error; archived campaigns answer 409 on
writes; every successful mutation writes its audit row in the SAME
transaction as the change (`campaign.stages.update`, `campaign.gate.decide`),
refused writes a best-effort one.

Honesty: a gate's decision is ONLY ever the recorded human decision. The
`computed_evidence` block next to it is copied from the stored analysis,
labeled `basis: "computed"`, and never turned into a go/no-go.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
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
from kalos.portal.campaign_settings import SettingTooLarge, put_setting
from kalos.portal.campaigns_api import _audit, _auth_method, _valid_campaign_id
from kalos.portal.stages import (
    EVIDENCE_ID_PATTERN,
    MAX_EVIDENCE_ITEMS,
    MAX_EVIDENCE_LABEL_CHARS,
    MAX_NOMINAL_SCALES,
    MAX_NOTE_CHARS,
    MAX_STAGES,
    STAGES_KIND,
    DecisionLimitReached,
    GateNotFound,
    StageValidationError,
    check_label,
    computed_evidence,
    list_decisions,
    load_stages,
    record_decision,
    validate_stages,
)

router = APIRouter()

_GATE_NOT_FOUND_BODY = {"error": "gate not found", "code": "gate_not_found"}
_ARCHIVED_BODY = {
    "error": "the campaign is archived; it can be read but not changed",
    "code": "campaign_archived",
}

_Str = Annotated[str, _Strict()]
_Volume = Annotated[float, _Strict()]


class _StageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: Annotated[str, _Strict(), StringConstraints(max_length=32)]
    label: Annotated[str, _Strict(), StringConstraints(max_length=256)]
    min_L: _Volume | None = None
    max_L: _Volume | None = None
    nominal_L: Annotated[list[_Volume], Field(max_length=MAX_NOMINAL_SCALES)] | None = None


class _StagesBody(BaseModel):
    """`current_stage` is REQUIRED (send null explicitly to clear it), so a
    client that only meant to edit stages cannot clear it by omission."""

    model_config = ConfigDict(extra="forbid")

    stages: Annotated[list[_StageIn], Field(min_length=1, max_length=MAX_STAGES)]
    current_stage: Annotated[str, _Strict(), StringConstraints(max_length=32)] | None


class _EvidenceIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["analysis", "scale_readout", "job", "run"]
    id: Annotated[str, _Strict(), StringConstraints(max_length=128)]
    label: Annotated[str, _Strict(), StringConstraints(max_length=MAX_EVIDENCE_LABEL_CHARS * 2)] | None = None

    @field_validator("id")
    @classmethod
    def _id_ok(cls, v: str) -> str:
        if EVIDENCE_ID_PATTERN.fullmatch(v) is None:
            raise ValueError("evidence id must be 1-128 of A-Z a-z 0-9 _ . : - (starting alphanumeric)")
        return v

    @field_validator("label")
    @classmethod
    def _label_ok(cls, v: str | None) -> str | None:
        if v is None:
            return None
        try:
            return check_label(v, max_chars=MAX_EVIDENCE_LABEL_CHARS, what="evidence label")
        except StageValidationError as exc:
            raise ValueError(exc.reason) from None


class _DecisionBody(BaseModel):
    """Who decided and when are taken from the authenticated principal and the
    server clock; a body carrying them is refused (`extra="forbid"`)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["go", "no_go", "pending"]
    note: Annotated[str, _Strict(), StringConstraints(max_length=MAX_NOTE_CHARS)]
    evidence: Annotated[list[_EvidenceIn], Field(max_length=MAX_EVIDENCE_ITEMS)] = []


# --- views -------------------------------------------------------------------- #


def _gate_summary(key: str, history: list[dict[str, Any]], *, with_history: bool) -> dict[str, Any]:
    current = history[0] if history else None
    view: dict[str, Any] = {
        "key": key,
        # The recorded decision only (null = nothing recorded yet).
        "decision": current["decision"] if current else None,
        "state": "recorded" if current else "open",
        "basis": "recorded",
        "current": current,
        "history_count": len(history),
    }
    if with_history:
        view["history"] = history
    return view


def _stages_view(store: CampaignStore, tenant: str, campaign_id: str) -> dict[str, Any]:
    doc = load_stages(store, tenant=tenant, campaign_id=campaign_id)
    decisions = list_decisions(store, tenant=tenant, campaign_id=campaign_id)
    keys = [s["key"] for s in doc["stages"]]
    stages = [
        {**s, "volume_unit": "L", "gate": _gate_summary(s["key"], decisions.get(s["key"], []),
                                                         with_history=False)}
        for s in doc["stages"]
    ]
    orphaned = [
        {**_gate_summary(k, decisions[k], with_history=False), "in_stage_definitions": False}
        for k in sorted(decisions) if k not in keys
    ]
    return {
        "campaign_id": campaign_id,
        "source": doc["source"],
        "updated_at": doc["updated_at"],
        "updated_by": doc["updated_by"],
        "current_stage": doc["current_stage"],
        "current_stage_basis": "recorded",
        "stages": stages,
        "orphaned_gates": orphaned,
    }


def _gates_view(store: CampaignStore, tenant: str, campaign_id: str) -> dict[str, Any]:
    doc = load_stages(store, tenant=tenant, campaign_id=campaign_id)
    decisions = list_decisions(store, tenant=tenant, campaign_id=campaign_id)
    evidence = computed_evidence(store.load_analysis(tenant=tenant, campaign_id=campaign_id))
    keys = [s["key"] for s in doc["stages"]]
    gates = [
        {**_gate_summary(s["key"], decisions.get(s["key"], []), with_history=True),
         "stage_label": s["label"], "in_stage_definitions": True,
         "computed_evidence": evidence}
        for s in doc["stages"]
    ]
    orphaned = [
        {**_gate_summary(k, decisions[k], with_history=True), "stage_label": None,
         "in_stage_definitions": False, "computed_evidence": evidence}
        for k in sorted(decisions) if k not in keys
    ]
    return {
        "campaign_id": campaign_id,
        "stages_source": doc["source"],
        "current_stage": doc["current_stage"],
        "gates": gates,
        "orphaned_gates": orphaned,
    }


def _error(exc: StageValidationError) -> JSONResponse:
    return JSONResponse({"error": exc.reason, "code": exc.code}, status_code=400)


def _store_error(exc: CampaignError) -> JSONResponse:
    if isinstance(exc, CampaignArchived):
        return JSONResponse(_ARCHIVED_BODY, status_code=409)
    return campaign_error_response(exc)


# --- routes ------------------------------------------------------------------- #


@router.get("/api/campaigns/{campaign_id}/stages")
def get_stages(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Stage definitions (defaults until edited), the recorded current stage,
    and each stage's gate status with its decision count."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        return JSONResponse(_stages_view(store, principal.tenant, campaign_id))
    except CampaignNotFound:
        return campaign_not_found_response()


@router.put("/api/campaigns/{campaign_id}/stages")
def put_stages(
    campaign_id: str,
    body: _StagesBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Replace the stage definitions and the recorded current stage. Recorded
    gate decisions are never touched: a gate whose stage disappears keeps its
    history and is listed under `orphaned_gates`."""
    action = "campaign.stages.update"
    if not _valid_campaign_id(campaign_id):
        _audit(store, principal, action, 404, None)
        return campaign_not_found_response()
    try:
        doc = validate_stages([s.model_dump() for s in body.stages], body.current_stage)
    except StageValidationError as exc:
        _audit(store, principal, action, 400, campaign_id)
        return _error(exc)
    try:
        put_setting(
            store, tenant=principal.tenant, campaign_id=campaign_id, kind=STAGES_KIND,
            value=doc, updated_by=principal.subject, audit_action=action,
            auth_method=_auth_method(principal),
            audit_detail=f"n_stages={len(doc['stages'])}",
        )
    except CampaignError as exc:
        response = _store_error(exc)
        _audit(store, principal, action, response.status_code, campaign_id)
        return response
    except SettingTooLarge:
        _audit(store, principal, action, 400, campaign_id)
        return JSONResponse(
            {"error": "the stage document is too large", "code": "stages_too_large"}, status_code=400
        )
    return JSONResponse(_stages_view(store, principal.tenant, campaign_id))


@router.get("/api/campaigns/{campaign_id}/gates")
def get_gates(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Every gate: its current (newest) recorded decision, full history newest
    first, and the read-only computed evidence from the stored analysis."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        return JSONResponse(_gates_view(store, principal.tenant, campaign_id))
    except CampaignNotFound:
        return campaign_not_found_response()


@router.put("/api/campaigns/{campaign_id}/gates/{gate}")
def put_gate_decision(
    campaign_id: str,
    gate: str,
    body: _DecisionBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Record a go / no_go / pending decision for one gate. APPEND-ONLY: each
    call adds a row that supersedes the previous one; nothing is overwritten."""
    action = "campaign.gate.decide"
    if not _valid_campaign_id(campaign_id):
        _audit(store, principal, action, 404, None)
        return campaign_not_found_response()
    try:
        recorded = record_decision(
            store, tenant=principal.tenant, campaign_id=campaign_id, gate=gate,
            decision=body.decision, note=body.note,
            evidence=[e.model_dump(exclude_none=True) for e in body.evidence],
            decided_by=principal.subject, auth_method=_auth_method(principal),
        )
    except StageValidationError as exc:
        _audit(store, principal, action, 400, campaign_id)
        return _error(exc)
    except CampaignError as exc:
        response = _store_error(exc)
        _audit(store, principal, action, response.status_code, campaign_id)
        return response
    except GateNotFound:
        _audit(store, principal, action, 404, campaign_id)
        return JSONResponse(_GATE_NOT_FOUND_BODY, status_code=404)
    except DecisionLimitReached:
        _audit(store, principal, action, 409, campaign_id)
        return JSONResponse(
            {"error": "this campaign has reached its limit of recorded gate decisions",
             "code": "gate_decision_limit"},
            status_code=409,
        )
    view = _gates_view(store, principal.tenant, campaign_id)
    gate_view = next((g for g in view["gates"] if g["key"] == gate), None)
    return JSONResponse({"campaign_id": campaign_id, "recorded": recorded, "gate": gate_view})
