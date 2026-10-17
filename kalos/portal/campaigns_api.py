"""Kalos portal - `/api/campaigns*`: several named campaigns per tenant (E4).

The schema v2 store (`kalos.portal.campaign`, docs/MIGRATIONS.md) holds any
number of campaigns per tenant; this router exposes them over HTTP. The legacy
single-campaign routes (`/api/run`, `/api/latest`, `/api/campaign/*`) are
unchanged and keep acting on the caller's DEFAULT campaign. API reference:
docs/CAMPAIGNS_API.md.

Security rules, enforced here for every route:

- Authentication + scope via `require_scope`: `read` for GET, `write` for POST
  and PATCH, and additionally `admin` for archiving/unarchiving. Scope is
  checked BEFORE the campaign id is looked at, so a 403 never depends on
  whether the id exists.
- The tenant is ALWAYS `principal.tenant`. No route reads a tenant from the
  path, query, body or headers.
- A campaign id is validated against `^cmp_[0-9a-f]{32}$` before the store is
  touched. A malformed id, a missing id and another tenant's id all produce the
  byte-identical response of `campaign_not_found_response()` (same status,
  body and headers). Missing and cross-tenant ids also take the same code path:
  the store's `WHERE id = ? AND tenant = ?` lookup returns no row for both.
  Caller input is never echoed in an error.
- Writes to an archived campaign answer 409.
- Mutations write a best-effort `audit_log` row (subject, auth method, action,
  campaign id, outcome). Never a token, a body, a filename, or uploaded data.

Uploads reuse the legacy `/api/run` pipeline (`kalos.portal.app._run_uploaded`)
synchronously, with the same size caps, single analysis slot, sheet parameter
and error-code contract. Background jobs are a later slice.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import Strict as _Strict

from kalos.portal.auth import ADMIN, READ, WRITE, Principal, require_scope
from kalos.portal.campaign import (
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    get_campaign_store,
)
from kalos.portal.campaign_routes import (
    _StartBody,
    _reanalyze,
    campaign_error_response,
    campaign_not_found_response,
)
from kalos.portal.limits import RequestSizeLimitMiddleware
from kalos.portal.migrate import CAMPAIGN_ID_PATTERN

log = logging.getLogger("kalos.portal")

router = APIRouter()

MAX_NAME_CHARS = 120
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
# Multipart uploads to a campaign get the upload-sized body cap. Only a
# well-formed campaign id qualifies; anything else keeps the small default cap.
CAMPAIGN_UPLOAD_PATH = re.compile(r"/api/campaigns/cmp_[0-9a-f]{32}/upload/?")

_RUN_NOT_FOUND_BODY = {"error": "run not found"}


class CampaignUploadSizeLimit(RequestSizeLimitMiddleware):
    """`RequestSizeLimitMiddleware` that also gives `POST
    /api/campaigns/{id}/upload` the upload cap. Kept here (not in
    `kalos.portal.limits`) so the generic middleware stays path-agnostic."""

    def cap_for(self, path: str) -> int:
        if CAMPAIGN_UPLOAD_PATH.fullmatch(path) is not None:
            return self.upload_cap
        return super().cap_for(path)


# --- helpers ------------------------------------------------------------------ #


def _valid_campaign_id(campaign_id: str) -> bool:
    return CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is not None


def _auth_method(principal: Principal) -> str:
    # E2 (JWT) adds `auth_method` to Principal; until then static bearer tokens.
    method = getattr(principal, "auth_method", None)
    if isinstance(method, str) and method:
        return method
    return "anonymous" if principal.anonymous else "bearer"


def _audit(
    store: CampaignStore, principal: Principal, action: str, status_code: int,
    campaign_id: str | None, *, detail: str | None = None,
) -> None:
    """Best-effort audit row. `campaign_id` is recorded only when the request
    resolved to one of the caller's campaigns (never for a 404, so probe input
    is not persisted). A failure to audit never fails the request."""
    try:
        store.record_audit(
            tenant=principal.tenant,
            subject=principal.subject,
            auth_method=_auth_method(principal),
            action=action,
            campaign_id=campaign_id if status_code != 404 else None,
            outcome="ok" if status_code < 400 else f"http_{status_code}",
            detail=detail,
        )
    except Exception:  # noqa: BLE001 - auditing must not break the audited request
        log.exception("failed to write an audit entry for %s", action)


def _require_admin(principal: Principal) -> None:
    if not principal.has(ADMIN):
        # Same shape and wording as `require_scope`'s 403.
        raise HTTPException(status_code=403, detail=f"requires '{ADMIN}' scope")


def _check_name(value: str) -> str:
    """1-120 characters after trimming surrounding whitespace, and no control,
    surrogate, or line/paragraph-separator characters. The message never
    echoes the input."""
    name = value.strip()
    if not 1 <= len(name) <= MAX_NAME_CHARS:
        raise ValueError(f"name must be 1-{MAX_NAME_CHARS} characters")
    if any(unicodedata.category(ch) in {"Cc", "Cs", "Zl", "Zp"} for ch in name):
        raise ValueError("name must not contain control characters")
    return name


_Name = Annotated[str, _Strict()]


class _CreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: _Name

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        return _check_name(v)


class _PatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: _Name | None = None
    archived: Annotated[bool, _Strict()] | None = None

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str | None) -> str | None:
        return None if v is None else _check_name(v)


class _RunResultBody(BaseModel):
    """Body for `POST /api/campaigns/{id}/runs/{run_id}/result`: the measured
    outcome. A JSON number only (bools and numeric strings are refused)."""

    model_config = ConfigDict(extra="forbid")

    value: Annotated[float, _Strict()] = Field()


# --- campaign management ------------------------------------------------------- #


@router.get("/api/campaigns")
def list_campaigns(
    include_archived: bool = Query(default=False),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> dict[str, Any]:
    """The caller's tenant's campaigns, newest first."""
    return {
        "campaigns": store.list_campaigns(
            tenant=principal.tenant, include_archived=include_archived
        )
    }


@router.post("/api/campaigns")
def create_campaign(
    body: _CreateBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Create an empty campaign in the caller's tenant. 201 with its info."""
    try:
        info = store.create_campaign(
            tenant=principal.tenant, name=body.name, created_by=principal.subject
        )
    except CampaignError as exc:
        _audit(store, principal, "campaign.create", 400, None)
        return campaign_error_response(exc)
    _audit(store, principal, "campaign.create", 201, info["id"])
    return JSONResponse(info, status_code=201)


@router.get("/api/campaigns/{campaign_id}")
def get_campaign(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """One campaign: its info plus the loop summary (the `GET /api/campaign`
    shape, scoped to this campaign)."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        info = store.campaign_info(tenant=principal.tenant, campaign_id=campaign_id)
        analysis = store.load_analysis(tenant=principal.tenant, campaign_id=campaign_id) or {}
        summary = store.summary(
            tenant=principal.tenant, campaign_id=campaign_id,
            analysis_generation=analysis.get("campaign_generation"),
        )
    except CampaignNotFound:
        return campaign_not_found_response()
    return JSONResponse({"campaign": info, "summary": summary})


@router.patch("/api/campaigns/{campaign_id}")
def update_campaign(
    campaign_id: str,
    body: _PatchBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Rename (`write`) and/or archive/unarchive (`admin`) a campaign."""
    if body.archived is not None:
        action = "campaign.archive" if body.archived else "campaign.unarchive"
        if not principal.has(ADMIN):
            _audit(store, principal, action, 403, None)
        _require_admin(principal)  # before the id is looked at
    if body.name is None and body.archived is None:
        return JSONResponse(
            {"error": "invalid request body: provide 'name' and/or 'archived'"}, status_code=400
        )
    if not _valid_campaign_id(campaign_id):
        status, response = 404, campaign_not_found_response()
        info = None
    else:
        try:
            info = store.update_campaign(
                tenant=principal.tenant, campaign_id=campaign_id,
                name=body.name, archived=body.archived,
            )
            status, response = 200, JSONResponse(info)
        except CampaignError as exc:
            response = campaign_error_response(exc)
            status = response.status_code
    if body.name is not None:
        _audit(store, principal, "campaign.rename", status, campaign_id)
    if body.archived is not None:
        _audit(store, principal, "campaign.archive" if body.archived else "campaign.unarchive",
               status, campaign_id)
    return response


# --- the loop, campaign-scoped ------------------------------------------------- #


@router.post("/api/campaigns/{campaign_id}/upload")
async def upload_to_campaign(
    campaign_id: str,
    file: UploadFile = File(...),
    target: str = Form(default=""),
    anonymize: bool = Form(default=False),
    roles: str = Form(default=""),
    sheet: str = Form(default=""),
    sheet_query: str | None = Query(default=None, alias="sheet"),
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Upload a run sheet into this campaign: the `/api/run` pipeline and
    contract (caps, `sheet`, error codes, 503 when the analysis slot is
    busy), seeding THIS campaign and publishing its analysis. Synchronous."""
    from kalos.portal.app import _run_uploaded  # deferred: app imports this module

    if not _valid_campaign_id(campaign_id):
        response = campaign_not_found_response()
    else:
        # Refuse a missing/foreign/archived campaign before spending a GP fit on it.
        try:
            info = store.campaign_info(tenant=principal.tenant, campaign_id=campaign_id)
        except CampaignError as exc:
            response = campaign_error_response(exc)
        else:
            if info["archived"]:
                response = JSONResponse(
                    {"error": "the campaign is archived; it can be read but not changed"},
                    status_code=409,
                )
            else:
                response = await _run_uploaded(
                    file, target, anonymize, roles, sheet, sheet_query, principal, store,
                    campaign_id=campaign_id,
                )
    _audit(store, principal, "campaign.upload", response.status_code, campaign_id)
    return response


@router.get("/api/campaigns/{campaign_id}/analysis")
def campaign_analysis(
    campaign_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """This campaign's published analysis (the `/api/latest` shape)."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        data = store.load_analysis(tenant=principal.tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    if not data:
        return JSONResponse({"has_data": False})
    return JSONResponse({"has_data": True, **data})


@router.post("/api/campaigns/{campaign_id}/runs")
def start_runs(
    campaign_id: str,
    body: _StartBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Start a proposed batch: same body and bounds as `/api/campaign/start`."""
    if not _valid_campaign_id(campaign_id):
        response = campaign_not_found_response()
        detail = None
    else:
        try:
            started = store.start(
                [p.model_dump() for p in body.recipes],
                tenant=principal.tenant, campaign_id=campaign_id,
            )
            response, detail = JSONResponse({"started": started}), f"runs={len(started)}"
        except CampaignError as exc:
            response, detail = campaign_error_response(exc), None
    _audit(store, principal, "campaign.start", response.status_code, campaign_id, detail=detail)
    return response


@router.post("/api/campaigns/{campaign_id}/runs/{run_id}/result")
def log_run_result(
    campaign_id: str,
    run_id: str,
    body: _RunResultBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Set the measured outcome on a pending run of this campaign."""
    if not _valid_campaign_id(campaign_id):
        response = campaign_not_found_response()
    elif not math.isfinite(body.value):
        response = JSONResponse({"error": "result value must be a finite number"}, status_code=400)
    elif RUN_ID_PATTERN.fullmatch(run_id) is None:
        response = JSONResponse(_RUN_NOT_FOUND_BODY, status_code=404)
    else:
        try:
            run = store.set_result(
                run_id, body.value, tenant=principal.tenant, campaign_id=campaign_id
            )
            response = JSONResponse(run)
        except CampaignError as exc:
            if type(exc) is CampaignError:
                # Unknown run id / unseeded campaign. The store's message echoes
                # the run id, so it is replaced, not forwarded.
                response = JSONResponse(_RUN_NOT_FOUND_BODY, status_code=404)
            else:
                response = campaign_error_response(exc)
    _audit(store, principal, "campaign.result", response.status_code, campaign_id)
    return response


@router.post("/api/campaigns/{campaign_id}/reanalyze")
async def reanalyze(
    campaign_id: str,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Fold measured runs, re-run the engine, publish: `/api/campaign/reanalyze`
    for this campaign (same transaction, slot, and error contract)."""
    if not _valid_campaign_id(campaign_id):
        response = campaign_not_found_response()
    else:
        response = await _reanalyze(store, principal, campaign_id=campaign_id)
    _audit(store, principal, "campaign.reanalyze", response.status_code, campaign_id)
    return response
