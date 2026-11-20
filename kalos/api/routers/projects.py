"""Projects, their campaigns, factors, outcomes and lineage (docs/RECORDS.md).

Routes (every one scoped; the tenant is always the verified token's):

- ``GET    /api/project-statuses``                               (read)
- ``GET    /api/projects``                                       (read)  paged
- ``POST   /api/projects``                                       (write)
- ``GET    /api/projects/{project_id}``                          (read)
- ``PATCH  /api/projects/{project_id}``                          (write)
- ``PUT    /api/projects/{project_id}/campaigns/{campaign_id}``  (write) attach
- ``DELETE /api/projects/{project_id}/campaigns/{campaign_id}``  (write) detach
- ``GET    /api/projects/{project_id}/factors``                  (read)
- ``PUT    /api/projects/{project_id}/factors``                  (write) upsert one, by name
- ``DELETE /api/projects/{project_id}/factors?name=``            (write)
- ``GET    /api/projects/{project_id}/factors/setup-bounds``     (read)  preview only
- ``GET    /api/projects/{project_id}/outcomes``                 (read)  paged
- ``GET    /api/projects/{project_id}/lineage``                  (read)  bounded

Ported from Voyager's ``/api/project*`` routes (NOTICE) with Kalos's rules: the
project-status list needed no token in Voyager and needs `read` here; the
factor routes were public writes there and need `write` here.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from kalos.api.routers._records_common import (
    Records,
    actor,
    audit_ok,
    audit_refused,
    campaign_cards,
    campaign_info,
    error,
    get_records,
    not_found_project,
    physical_bounds,
)
from kalos.api.schemas.records import FactorPut, ProjectCreate, ProjectUpdate
from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.records import factors as factors_rules
from kalos.records import lineage as lineage_rules
from kalos.records import outcomes as outcome_rules
from kalos.records import projects as project_rules
from kalos.records.projects import RecordsValidationError
from kalos.storage import outcomes_repo, records_repo

router = APIRouter()


def _project_or_none(rec: Records, tenant: str, project_id: str) -> dict[str, Any] | None:
    if not project_rules.valid_project_id(project_id):
        return None
    with rec.records.txn(write=False) as conn:
        return records_repo.get_project(conn, tenant=tenant, project_id=project_id)


# --- statuses and projects --------------------------------------------------------- #


@router.get("/api/project-statuses")
def get_project_statuses(principal: Principal = Depends(require_scope(READ))) -> JSONResponse:
    """The project status vocabulary, in lifecycle order."""
    return JSONResponse({"statuses": project_rules.status_catalogue()})


@router.get("/api/projects")
def list_projects(
    status: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0, le=10_000),
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """The caller's projects, most recently updated first, with campaign counts."""
    if status is not None and status not in project_rules.STATUSES:
        return error(400, "invalid_status", f"status must be one of: {', '.join(project_rules.STATUSES)}")
    with rec.records.txn(write=False) as conn:
        rows, has_more = records_repo.list_projects(
            conn, tenant=principal.tenant, status=status, limit=limit, offset=offset)
        counts = records_repo.project_campaign_counts(
            conn, tenant=principal.tenant, project_ids=[r["id"] for r in rows])
    for r in rows:
        r["n_campaigns"] = counts.get(r["id"], 0)
    return JSONResponse({
        "projects": rows, "limit": limit, "offset": offset, "has_more": has_more,
        "next_offset": offset + len(rows) if has_more else None,
    })


@router.post("/api/projects", status_code=201)
def create_project(
    body: ProjectCreate,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Create a project. 409 when the name is taken or the tenant is at its limit."""
    action = "project.create"
    try:
        fields = project_rules.project_fields(
            name=body.name, objective=body.objective, status=body.status, sponsor=body.sponsor)
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 400, code=exc.code)
        return error(400, exc.code, exc.reason)
    now, pid = time.time(), project_rules.new_project_id()
    try:
        with rec.records.txn(write=True) as conn:
            if records_repo.count_projects(conn, tenant=principal.tenant) >= project_rules.MAX_PROJECTS_PER_TENANT:
                raise RecordsValidationError("too_many_projects", "")
            if records_repo.project_name_taken(conn, tenant=principal.tenant, name=fields.name):
                raise RecordsValidationError("name_taken", "")
            row = records_repo.insert_project(
                conn, tenant=principal.tenant, project_id=pid, name=fields.name,
                objective=fields.objective, status=fields.status, sponsor=fields.sponsor,
                actor=actor(principal), now=now)
            audit_ok(conn, principal, action, detail=f"project={pid};status={fields.status}", now=now)
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 409, code=exc.code)
        if exc.code == "name_taken":
            return error(409, "name_taken", "a project with this name already exists")
        return error(409, "too_many_projects",
                     f"a tenant can have at most {project_rules.MAX_PROJECTS_PER_TENANT} projects")
    row["n_campaigns"] = 0
    return JSONResponse({"project": row}, status_code=201)


@router.get("/api/projects/{project_id}")
def get_project(
    project_id: str,
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """One project with its campaigns (name, recorded stage) and factors."""
    project = _project_or_none(rec, principal.tenant, project_id)
    if project is None:
        return not_found_project()
    with rec.records.txn(write=False) as conn:
        members = records_repo.project_campaign_ids(conn, tenant=principal.tenant, project_id=project_id)
        factor_rows = records_repo.list_factors(conn, tenant=principal.tenant, project_id=project_id)
    cards = campaign_cards(rec, principal.tenant, [m["campaign_id"] for m in members])
    campaigns = [
        {"campaign_id": m["campaign_id"], **cards[m["campaign_id"]],
         "added_at": m["added_at"], "added_by": m["added_by"]}
        for m in members
    ]
    project["n_campaigns"] = len(members)
    return JSONResponse({"project": project, "campaigns": campaigns, "factors": factor_rows})


@router.patch("/api/projects/{project_id}")
def patch_project(
    project_id: str,
    body: ProjectUpdate,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Change a project's name, objective, status or sponsor (keys sent only)."""
    action = "project.update"
    if not project_rules.valid_project_id(project_id):
        return not_found_project()
    patch = body.model_dump(exclude_unset=True)
    now = time.time()
    try:
        with rec.records.txn(write=True) as conn:
            current = records_repo.get_project(conn, tenant=principal.tenant, project_id=project_id)
            if current is None:
                return not_found_project()
            fields = project_rules.merge_update(current, patch)
            if fields.name != current["name"] and records_repo.project_name_taken(
                    conn, tenant=principal.tenant, name=fields.name, except_id=project_id):
                raise RecordsValidationError("name_taken", "a project with this name already exists")
            row = records_repo.update_project(
                conn, tenant=principal.tenant, project_id=project_id, name=fields.name,
                objective=fields.objective, status=fields.status, sponsor=fields.sponsor,
                actor=actor(principal), now=now)
            audit_ok(conn, principal, action, now=now, detail=(
                f"project={project_id};keys={','.join(sorted(patch))};"
                f"status={current['status']}->{fields.status}"))
    except RecordsValidationError as exc:
        status = 409 if exc.code == "name_taken" else 400
        audit_refused(rec, principal, action, status, code=exc.code)
        return error(status, exc.code, exc.reason)
    return JSONResponse({"project": row})


# --- project campaigns ------------------------------------------------------------- #


@router.put("/api/projects/{project_id}/campaigns/{campaign_id}")
def attach_campaign(
    project_id: str,
    campaign_id: str,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Add one of the caller's campaigns to the project (idempotent). A campaign
    belongs to at most one project: 409 if it is in another."""
    action = "project.campaign.attach"
    if not project_rules.valid_project_id(project_id):
        return not_found_project()
    if campaign_info(rec.store, principal.tenant, campaign_id) is None:
        from kalos.portal.campaign_routes import campaign_not_found_response

        return campaign_not_found_response()
    now = time.time()
    try:
        with rec.records.txn(write=True) as conn:
            if records_repo.get_project(conn, tenant=principal.tenant, project_id=project_id) is None:
                return not_found_project()
            owner = records_repo.campaign_project(conn, tenant=principal.tenant, campaign_id=campaign_id)
            if owner == project_id:
                return JSONResponse({"project_id": project_id, "campaign_id": campaign_id, "attached": True,
                                     "created": False})
            if owner is not None:
                raise RecordsValidationError("campaign_in_other_project", "")
            members = records_repo.project_campaign_ids(conn, tenant=principal.tenant, project_id=project_id)
            if len(members) >= project_rules.MAX_CAMPAIGNS_PER_PROJECT:
                raise RecordsValidationError("too_many_campaigns", "")
            records_repo.attach_campaign(conn, tenant=principal.tenant, project_id=project_id,
                                         campaign_id=campaign_id, actor=actor(principal), now=now)
            audit_ok(conn, principal, action, campaign_id=campaign_id, now=now, detail=f"project={project_id}")
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 409, code=exc.code, campaign_id=campaign_id)
        if exc.code == "campaign_in_other_project":
            return error(409, exc.code, "the campaign already belongs to another project",
                         what_would_fix_it="detach it from that project first")
        return error(409, exc.code, f"a project can group at most {project_rules.MAX_CAMPAIGNS_PER_PROJECT} campaigns")
    return JSONResponse({"project_id": project_id, "campaign_id": campaign_id, "attached": True, "created": True})


@router.delete("/api/projects/{project_id}/campaigns/{campaign_id}")
def detach_campaign(
    project_id: str,
    campaign_id: str,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Remove a campaign from the project. The campaign and its data are
    untouched; lineage links that involve it are kept."""
    action = "project.campaign.detach"
    if not project_rules.valid_project_id(project_id):
        return not_found_project()
    if campaign_info(rec.store, principal.tenant, campaign_id) is None:
        from kalos.portal.campaign_routes import campaign_not_found_response

        return campaign_not_found_response()
    now = time.time()
    with rec.records.txn(write=True) as conn:
        if records_repo.get_project(conn, tenant=principal.tenant, project_id=project_id) is None:
            return not_found_project()
        removed = records_repo.detach_campaign(conn, tenant=principal.tenant, project_id=project_id,
                                               campaign_id=campaign_id)
        if removed:
            audit_ok(conn, principal, action, campaign_id=campaign_id, now=now, detail=f"project={project_id}")
    if not removed:
        return error(404, "not_attached", "the campaign is not part of this project")
    return JSONResponse({"project_id": project_id, "campaign_id": campaign_id, "attached": False})


# --- factors ---------------------------------------------------------------------- #


@router.get("/api/projects/{project_id}/factors")
def get_factors(
    project_id: str,
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    if _project_or_none(rec, principal.tenant, project_id) is None:
        return not_found_project()
    with rec.records.txn(write=False) as conn:
        rows = records_repo.list_factors(conn, tenant=principal.tenant, project_id=project_id)
    return JSONResponse({"project_id": project_id, "factors": rows,
                         "max_factors": factors_rules.MAX_FACTORS_PER_PROJECT})


@router.put("/api/projects/{project_id}/factors")
def put_factor(
    project_id: str,
    body: FactorPut,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Create or replace one factor (keyed by `name`). The range is refused when
    the unit is unknown or it is physically impossible for the unit or name."""
    action = "project.factor.put"
    if not project_rules.valid_project_id(project_id):
        return not_found_project()
    try:
        f = factors_rules.factor_fields(
            name=body.name, lower=body.lower_bound, upper=body.upper_bound, unit=body.unit,
            description=body.description, role=body.role, physical_bounds=physical_bounds)
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 400, code=exc.code)
        return error(400, exc.code, exc.reason)
    now = time.time()
    try:
        with rec.records.txn(write=True) as conn:
            if records_repo.get_project(conn, tenant=principal.tenant, project_id=project_id) is None:
                return not_found_project()
            exists = records_repo.factor_exists(conn, tenant=principal.tenant, project_id=project_id, name=f.name)
            if not exists and records_repo.count_factors(
                    conn, tenant=principal.tenant, project_id=project_id) >= factors_rules.MAX_FACTORS_PER_PROJECT:
                raise RecordsValidationError("too_many_factors", "")
            records_repo.upsert_factor(
                conn, tenant=principal.tenant, project_id=project_id, name=f.name, description=f.description,
                lower=f.lower, upper=f.upper, unit=f.unit, dimension=f.dimension, role=f.role,
                actor=actor(principal), now=now)
            audit_ok(conn, principal, action, now=now,
                     detail=f"project={project_id};created={0 if exists else 1};role={f.role}")
            rows = records_repo.list_factors(conn, tenant=principal.tenant, project_id=project_id)
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 409, code=exc.code)
        return error(409, exc.code, f"a project can define at most {factors_rules.MAX_FACTORS_PER_PROJECT} factors")
    factor = next(r for r in rows if r["name"] == f.name)
    return JSONResponse({"project_id": project_id, "factor": factor, "created": not exists})


@router.delete("/api/projects/{project_id}/factors")
def delete_factor(
    project_id: str,
    name: str = Query(min_length=1, max_length=factors_rules.MAX_FACTOR_NAME_CHARS),
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Remove one factor definition. Campaign setups already filled from it are
    not changed."""
    action = "project.factor.delete"
    if not project_rules.valid_project_id(project_id):
        return not_found_project()
    try:
        clean = factors_rules.check_factor_name(name)
    except RecordsValidationError as exc:
        return error(400, exc.code, exc.reason)
    now = time.time()
    with rec.records.txn(write=True) as conn:
        if records_repo.get_project(conn, tenant=principal.tenant, project_id=project_id) is None:
            return not_found_project()
        removed = records_repo.delete_factor(conn, tenant=principal.tenant, project_id=project_id, name=clean)
        if removed:
            audit_ok(conn, principal, action, now=now, detail=f"project={project_id}")
    if not removed:
        return error(404, "factor_not_found", "the project has no factor with this name")
    return JSONResponse({"project_id": project_id, "deleted": True})


@router.get("/api/projects/{project_id}/factors/setup-bounds")
def get_setup_bounds(
    project_id: str,
    campaign_id: str | None = Query(default=None, max_length=36),
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """The project's `setting` factors as a campaign setup `bounds` object.
    With `campaign_id` (a campaign of this project), factors that are not columns
    of its latest analysis are listed as skipped. Nothing is written."""
    if _project_or_none(rec, principal.tenant, project_id) is None:
        return not_found_project()
    columns: list[str] | None = None
    with rec.records.txn(write=False) as conn:
        rows = records_repo.list_factors(conn, tenant=principal.tenant, project_id=project_id)
        owner = (records_repo.campaign_project(conn, tenant=principal.tenant, campaign_id=campaign_id)
                 if campaign_id is not None else None)
    if campaign_id is not None:
        if campaign_info(rec.store, principal.tenant, campaign_id) is None or owner != project_id:
            return error(404, "campaign_not_in_project", "the campaign is not part of this project")
        from kalos.portal.setup_settings import dataset_context

        ctx = dataset_context(rec.store.load_analysis(tenant=principal.tenant, campaign_id=campaign_id))
        # Bounds apply to the continuous settings the analysis kept.
        columns = list(ctx.continuous) if ctx is not None else None
    out = factors_rules.setup_bounds(rows, columns=columns)
    out.update({"project_id": project_id, "campaign_id": campaign_id})
    if campaign_id is not None and columns is None:
        out["columns_reason"] = "the campaign has no analysis yet; factors were not checked against its columns"
    return JSONResponse(out)


# --- outcomes and lineage ----------------------------------------------------------- #


@router.get("/api/projects/{project_id}/outcomes")
def get_outcomes(
    project_id: str,
    limit: int = Query(default=outcome_rules.DEFAULT_RUNS_PER_PAGE, ge=1, le=outcome_rules.MAX_RUNS_PER_PAGE),
    offset: int = Query(default=0, ge=0, le=100_000),
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Every run of the project's campaigns that has a recorded prediction or
    result: the newest measured value (and its version count) beside the
    prediction recorded at start. Measured and predicted are never merged."""
    if _project_or_none(rec, principal.tenant, project_id) is None:
        return not_found_project()
    with rec.records.txn(write=False) as conn:
        members = records_repo.project_campaign_ids(conn, tenant=principal.tenant, project_id=project_id)
    cards = campaign_cards(rec, principal.tenant, [m["campaign_id"] for m in members])
    ids = [cid for cid, card in cards.items() if card["available"]]
    with rec.campaigns.txn(write=False) as conn:
        keys, has_more = outcomes_repo.run_keys(conn, tenant=principal.tenant, campaign_ids=ids,
                                                limit=limit, offset=offset)
        versions = outcomes_repo.versions_for(conn, tenant=principal.tenant, keys=keys)
        predictions = outcomes_repo.predictions_for(conn, tenant=principal.tenant, keys=keys)
    return JSONResponse({
        "project_id": project_id,
        "campaign_ids": ids,
        "runs": outcome_rules.outcomes_page(keys, versions, predictions),
        "limit": limit, "offset": offset, "has_more": has_more,
        "next_offset": offset + len(keys) if has_more else None,
        "notes": "measured values are the newest recorded version (corrections included); "
                 "prediction_at_start is the band recorded when the run started",
    })


@router.get("/api/projects/{project_id}/lineage")
def get_project_lineage(
    project_id: str,
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """The project's campaigns and every active lineage link among them
    (Voyager's project-wide experiment-link view)."""
    if _project_or_none(rec, principal.tenant, project_id) is None:
        return not_found_project()
    with rec.records.txn(write=False) as conn:
        members = records_repo.project_campaign_ids(conn, tenant=principal.tenant, project_id=project_id)
        ids = [m["campaign_id"] for m in members]
        links = records_repo.active_links_of_campaigns(
            conn, tenant=principal.tenant, campaign_ids=ids,
            limit=lineage_rules.MAX_LINKS_RETURNED + 1)
    truncated = len(links) > lineage_rules.MAX_LINKS_RETURNED
    return JSONResponse({
        "project_id": project_id,
        "campaigns": [{"campaign_id": cid, **card}
                      for cid, card in campaign_cards(rec, principal.tenant, ids).items()],
        "links": links[:lineage_rules.MAX_LINKS_RETURNED],
        "truncated": ["links"] if truncated else [],
    })


__all__ = ["router"]

