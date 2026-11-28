"""Kalos portal - `/api/campaigns/{id}/setup`: the campaign's goal and operating
boundaries (goal direction, user bounds, linear constraints, target unit).

Stored in `campaign_settings` (kind ``setup``); validation, hashing and the
conversion into optimizer arguments live in `kalos.portal.setup_settings`. The
setup takes effect at the NEXT analysis (upload or re-analyze), which records
the setup hash it used; `settings_in_sync` says whether that is the stored one.

Security rules (the `kalos.portal.campaigns_api` pattern): scope is checked
before the id is looked at (`read` for GET, `write` for PUT); the tenant is
always `principal.tenant`; malformed, missing and foreign ids give the
byte-identical `campaign_not_found_response()`; errors never echo caller input
(features are referred to by position); archived campaigns answer 409 on PUT;
every PUT writes an audit row (ids, goal and counts only, never the body).
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, StringConstraints
from pydantic import Strict as _Strict

from kalos.campaigns.setup import load_setup
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
from kalos.portal.setup_settings import (
    DEFAULT_SETUP,
    DEFAULT_SETUP_HASH,
    MAX_BOUND_FEATURES,
    MAX_CONSTRAINT_TERMS,
    MAX_FEATURE_NAME_CHARS,
    MAX_LINEAR_CONSTRAINTS,
    SetupInvalid,
    bounds_report,
    dataset_context,
    normalize_setup,
    setup_hash,
    validate_against_dataset,
)

router = APIRouter()

_ARCHIVED_BODY = {
    "error": "the campaign is archived; it can be read but not changed", "code": "campaign_archived",
}

_Num = Annotated[float, _Strict(), AllowInfNan(False)]
_Name = Annotated[
    str, _Strict(), StringConstraints(min_length=1, max_length=MAX_FEATURE_NAME_CHARS)
]


class _Bound(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lo: _Num
    hi: _Num


class _Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coefficients: dict[_Name, _Num] = Field(min_length=1, max_length=MAX_CONSTRAINT_TERMS)
    op: Literal["<=", ">="]
    rhs: _Num


class _SetupBody(BaseModel):
    """PUT body: the WHOLE setup (a replace, not a patch). Omitted keys take
    their defaults: goal maximize, no bounds, no constraints, no unit."""

    model_config = ConfigDict(extra="forbid")

    goal: Literal["maximize", "minimize"] = "maximize"
    bounds: dict[_Name, _Bound] = Field(default_factory=dict, max_length=MAX_BOUND_FEATURES)
    linear_constraints: list[_Constraint] = Field(
        default_factory=list, max_length=MAX_LINEAR_CONSTRAINTS
    )
    # Checked again after trimming (<= 32 characters, no control characters).
    target_unit: Annotated[str, _Strict(), StringConstraints(max_length=64)] | None = None
    # Clone / strain (docs/CLONES.md). Unset keys are not stored, so a setup
    # without them keeps its hash. `strain_policy` defaults to "fixed" in the
    # analysis once `strain_column` is set.
    strain_column: _Name | None = None
    strain_policy: Literal["fixed", "factor"] | None = None
    strain_level: Annotated[str, _Strict(), StringConstraints(min_length=1, max_length=64)] | None = None
    covariates: list[_Name] | None = Field(default=None, max_length=32)


def _view(store: CampaignStore, tenant: str, campaign_id: str) -> dict[str, Any]:
    """The GET body. Raises `CampaignNotFound`."""
    source = "default"
    updated_at: float | None = None
    updated_by: str | None = None
    doc: dict[str, Any] | None
    try:
        row = load_setup(store, tenant=tenant, campaign_id=campaign_id)
    except SetupInvalid:
        row, doc, source = None, None, "unreadable"
    else:
        doc = dict(DEFAULT_SETUP) if row is None else row["value"]
        if row is not None:
            source = "stored"
            updated_at, updated_by = row.get("updated_at"), row.get("updated_by")
    analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id)
    ctx = dataset_context(analysis)
    recorded: str | None = None
    if analysis:
        block = analysis.get("setup")
        if isinstance(block, dict) and isinstance(block.get("hash"), str):
            recorded = block["hash"]
    current = setup_hash(doc) if doc is not None else None
    in_sync: bool | None
    if not analysis:
        in_sync = None  # nothing analyzed yet: nothing to be in or out of sync with
    elif current is None:
        in_sync = False
    else:
        # An analysis without a `setup` block was computed with the default setup.
        in_sync = current == (recorded if recorded is not None else DEFAULT_SETUP_HASH)
    return {
        "campaign_id": campaign_id,
        "setup": doc,
        "source": source,
        "hash": current,
        "updated_at": updated_at,
        "updated_by": updated_by,
        "bounds_report": bounds_report(doc, ctx) if doc is not None else {},
        "has_analysis": bool(analysis),
        "analysis_setup_hash": recorded,
        "settings_in_sync": in_sync,
    }


@router.get("/api/campaigns/{campaign_id}/setup")
def get_setup(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """The campaign's setup (defaults when none was saved), per-bound
    extrapolation flags against the latest analysis, and `settings_in_sync`."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        return JSONResponse(_view(store, principal.tenant, campaign_id))
    except CampaignNotFound:
        return campaign_not_found_response()


@router.put("/api/campaigns/{campaign_id}/setup")
def put_setup(
    campaign_id: str,
    body: _SetupBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Replace the campaign's setup after validating it against the campaign's
    latest analysis and checking the bounds + constraints are feasible (LP)."""
    action = "campaign.setup"
    if not _valid_campaign_id(campaign_id):
        _audit(store, principal, action, 404, None)
        return campaign_not_found_response()
    tenant = principal.tenant
    try:
        info = store.campaign_info(tenant=tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        response = campaign_error_response(exc)
        _audit(store, principal, action, response.status_code, campaign_id)
        return response
    if info.get("archived"):
        _audit(store, principal, action, 409, campaign_id)
        return JSONResponse(_ARCHIVED_BODY, status_code=409)
    try:
        doc = normalize_setup(body.model_dump())
        ctx = dataset_context(store.load_analysis(tenant=tenant, campaign_id=campaign_id))
        validate_against_dataset(doc, ctx)
    except SetupInvalid as exc:
        _audit(store, principal, action, 400, campaign_id, detail=f"code={exc.code}")
        return JSONResponse(exc.to_body(), status_code=400)
    except CampaignNotFound:
        _audit(store, principal, action, 404, campaign_id)
        return campaign_not_found_response()
    detail = (
        f"goal={doc['goal']} bounds={len(doc['bounds'])} "
        f"constraints={len(doc['linear_constraints'])} unit={'set' if doc['target_unit'] else 'none'}"
    )
    try:
        # The audit row is written in the same transaction as the change.
        put_setting(
            store, tenant=tenant, campaign_id=campaign_id, kind="setup", value=doc,
            updated_by=principal.subject, audit_action=action,
            auth_method=_auth_method(principal), audit_detail=detail,
        )
    except SettingTooLarge:
        _audit(store, principal, action, 400, campaign_id, detail="code=setup_too_large")
        return JSONResponse(
            {"error": "the setup document is too large", "code": "setup_too_large"},
            status_code=400,
        )
    except CampaignArchived:
        _audit(store, principal, action, 409, campaign_id)
        return JSONResponse(_ARCHIVED_BODY, status_code=409)
    except CampaignNotFound:
        _audit(store, principal, action, 404, campaign_id)
        return campaign_not_found_response()
    try:
        return JSONResponse(_view(store, tenant, campaign_id))
    except CampaignNotFound:
        return campaign_not_found_response()
