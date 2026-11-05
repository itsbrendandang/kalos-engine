"""`/api/campaigns/{id}/virtual-runs*` and `/api/virtual-runs/generators`:
simulated and synthetic runs over HTTP (docs/VIRTUAL_RUNS.md).

Same security rules as `kalos.portal.campaigns_api`:

- `require_scope`: `write` to create a batch, `read` to list, export, or list
  generators. Scope is checked before the campaign id is looked at.
- The tenant is always `principal.tenant`. A malformed, missing or other-tenant
  campaign id gets the byte-identical `campaign_not_found_response()`.
- Caller input is never echoed in an error.

Generation is synchronous and bounded (`MAX_ROWS_PER_BATCH` rows; the built-in
generators take well under a second for that), so it does not use the analysis
job queue. No route here feeds virtual rows to an analysis.
"""
from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic import Strict as _Strict

from kalos.portal import virtual_runs as vr
from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign import (
    CampaignArchived,
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    get_campaign_store,
)
from kalos.portal.campaign_routes import campaign_not_found_response
from kalos.portal.migrate import CAMPAIGN_ID_PATTERN
from kalos.twin import generators as gen

log = logging.getLogger("kalos.portal")

router = APIRouter()

BATCH_ID_PATTERN = re.compile(r"vjb_[0-9a-f]{32}")
MAX_PARAMS = 32

_GeneratorId = Annotated[str, StringConstraints(min_length=1, max_length=gen.GENERATOR_ID_CHARS)]
_ParamValue = Annotated[float, _Strict()] | Annotated[int, _Strict()] | Annotated[bool, _Strict()] | Annotated[str, StringConstraints(max_length=128)] | None


class _CreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator: _GeneratorId
    params: dict[Annotated[str, StringConstraints(max_length=64)], _ParamValue] | None = Field(
        default=None, max_length=MAX_PARAMS
    )
    n: Annotated[int, _Strict()] = Field(ge=1, le=gen.MAX_ROWS_PER_BATCH)


def _auth_method(principal: Principal) -> str:
    method = getattr(principal, "auth_method", None)
    if isinstance(method, str) and method:
        return method
    return "anonymous" if principal.anonymous else "bearer"


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": message, "code": code}, status_code=status)


def _archived() -> JSONResponse:
    return _error(409, "campaign_archived", "the campaign is archived; it can be read but not changed")


@router.get("/api/virtual-runs/generators")
def list_generators(principal: Principal = Depends(require_scope(READ))) -> dict[str, Any]:
    """The registered generators (id, version, origin, defaults)."""
    return {"generators": [s.info() for s in gen.list_generators()]}


@router.post("/api/campaigns/{campaign_id}/virtual-runs")
def create_virtual_runs(
    campaign_id: str,
    body: _CreateBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Generate and store one batch of virtual runs for this campaign. 201."""
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        return campaign_not_found_response()
    try:
        out = vr.create_batch(
            store, tenant=principal.tenant, campaign_id=campaign_id,
            generator_id=body.generator, params=body.params, n=body.n,
            requested_by=principal.subject, auth_method=_auth_method(principal),
        )
    except CampaignNotFound:
        return campaign_not_found_response()
    except CampaignArchived:
        return _archived()
    except gen.GeneratorError as exc:
        return _error(400, "invalid_generator_request", str(exc))
    log.info("virtual runs: batch created (n=%d)", out["n"])
    return JSONResponse(out, status_code=201)


@router.get("/api/campaigns/{campaign_id}/virtual-runs")
def get_virtual_runs(
    campaign_id: str,
    origin: Literal["simulated", "synthetic"] | None = Query(default=None),
    batch_id: str | None = Query(default=None, max_length=64),
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=vr.MAX_LIST_ROWS),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """This campaign's virtual runs, oldest first; page with `after=<next_after>`.
    Every row carries its `origin` and `label`."""
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        return campaign_not_found_response()
    if batch_id is not None and BATCH_ID_PATTERN.fullmatch(batch_id) is None:
        return _error(400, "invalid_batch_id", "batch_id must be 'vjb_' followed by 32 hex digits")
    try:
        rows = vr.list_runs(store, tenant=principal.tenant, campaign_id=campaign_id,
                            origin=origin, batch_id=batch_id, after=after, limit=limit)
    except CampaignNotFound:
        return campaign_not_found_response()
    runs = [
        {
            "id": r["id"], "batch_id": r["batch_id"], "k": r["k"], "origin": r["origin"],
            "label": vr.origin_label(r["origin"], r["twin_version_id"]),
            "generator_id": r["generator_id"], "generator_version": r["generator_version"],
            "params": r["params_json"], "params_sha256": r["params_sha256"],
            "seed_entropy": r["seed_entropy"], "pcg64_state": r["pcg64_state"],
            "solver": r["solver_json"], "library_versions": r["library_versions"],
            "twin_version_id": r["twin_version_id"], "design": r["design_json"],
            "design_sha256": r["design_sha256"], "outcome": r["outcome_json"],
            "requested_by": r["requested_by"], "created_at": r["created_at"],
        }
        for r in rows
    ]
    next_after = runs[-1]["id"] if len(runs) == limit else None
    return JSONResponse({"campaign_id": campaign_id, "runs": runs, "next_after": next_after})


@router.get("/api/campaigns/{campaign_id}/virtual-runs/export")
def export_virtual_runs(
    campaign_id: str,
    origin: Literal["simulated", "synthetic"] | None = Query(default=None),
    batch_id: str | None = Query(default=None, max_length=64),
    include_measured: bool = Query(default=False),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> Response:
    """One origin of this campaign's virtual runs as CSV. Mixed exports (two
    origins, or measured plus virtual) are refused with 409 `mixed_export_refused`."""
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        return campaign_not_found_response()
    if batch_id is not None and BATCH_ID_PATTERN.fullmatch(batch_id) is None:
        return _error(400, "invalid_batch_id", "batch_id must be 'vjb_' followed by 32 hex digits")
    try:
        filename, text, the_origin = vr.export_csv(
            store, tenant=principal.tenant, campaign_id=campaign_id, origin=origin,
            batch_id=batch_id, include_measured=include_measured,
        )
    except CampaignNotFound:
        return campaign_not_found_response()
    except vr.MixedExportRefused as exc:
        return _error(409, exc.code, str(exc))
    except vr.NoVirtualRuns as exc:
        return _error(404, exc.code, str(exc))
    except CampaignError as exc:
        return _error(400, "invalid_request", str(exc))
    return Response(
        content=text, media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Kalos-Origin": str(the_origin),
            "Cache-Control": "no-store",
        },
    )
