"""Lineage links between campaigns and runs, across rounds and scale stages
(docs/RECORDS.md). Ported from Voyager's ``/api/experiment-link*`` (NOTICE).

Routes:

- ``GET  /api/lineage/relations``                 (read)
- ``POST /api/lineage/links``                     (write) one parent -> child link
- ``GET  /api/lineage/links/{link_id}``           (read)
- ``POST /api/lineage/links/{link_id}/retract``   (write) withdraw with a reason
- ``GET  /api/campaigns/{campaign_id}/lineage``   (read)  ancestors + descendants, bounded

Both ends of a link are resolved under the caller's tenant before anything is
written, so a link can never reach another tenant's campaign (a foreign id gets
the same 404 as a missing one). A run end must be a run the campaign recorded (a
prediction at start or a result). Stages default to each campaign's RECORDED
current stage. Links are append-only.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from kalos.api.routers._records_common import (
    LINK_NOT_FOUND,
    Records,
    actor,
    audit_ok,
    audit_refused,
    campaign_cards,
    campaign_info,
    current_stage,
    error,
    get_records,
)
from kalos.api.schemas.records import LinkCreate, LinkRetract
from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign_routes import campaign_not_found_response
from kalos.records import lineage as rules
from kalos.records import outcomes as outcome_rules
from kalos.records.projects import RecordsValidationError
from kalos.storage import outcomes_repo, records_repo

router = APIRouter()

MAX_LINK_ID = 2**62


def _link_not_found() -> JSONResponse:
    return JSONResponse(LINK_NOT_FOUND, status_code=404)


@router.get("/api/lineage/relations")
def get_relations(principal: Principal = Depends(require_scope(READ))) -> JSONResponse:
    return JSONResponse({"relations": rules.relation_catalogue()})


def _run_known(rec: Records, tenant: str, campaign_id: str, run_id: str) -> bool:
    if not run_id:
        return True
    with rec.campaigns.txn(write=False) as conn:
        if outcomes_repo.run_known(conn, tenant=tenant, campaign_id=campaign_id, run_id=run_id):
            return True
    state = rec.store.get(tenant=tenant, campaign_id=campaign_id) or {}
    return any(isinstance(r, dict) and r.get("id") == run_id for r in state.get("pending") or [])


@router.post("/api/lineage/links", status_code=201)
def create_link(
    body: LinkCreate,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Record that `child` came from `parent`. Refused: a campaign or run the
    caller cannot see (404), a self-link, a duplicate, or a cycle (409)."""
    action = "lineage.link.create"
    tenant = principal.tenant
    parent = campaign_info(rec.store, tenant, body.parent_campaign_id)
    child = campaign_info(rec.store, tenant, body.child_campaign_id)
    if parent is None or child is None:
        audit_refused(rec, principal, action, 404, code="campaign_not_found")
        return campaign_not_found_response()
    try:
        link = rules.link_fields(
            parent_campaign_id=body.parent_campaign_id, child_campaign_id=body.child_campaign_id,
            relation=body.relation, parent_run_id=body.parent_run_id, child_run_id=body.child_run_id,
            parent_stage=body.parent_stage if body.parent_stage is not None
            else current_stage(rec.store, tenant, body.parent_campaign_id),
            child_stage=body.child_stage if body.child_stage is not None
            else current_stage(rec.store, tenant, body.child_campaign_id),
            note=body.note,
        )
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 400, code=exc.code)
        return error(400, exc.code, exc.reason)
    for cid, run in ((link.parent_campaign_id, link.parent_run_id), (link.child_campaign_id, link.child_run_id)):
        if not _run_known(rec, tenant, cid, run):
            audit_refused(rec, principal, action, 404, code="run_not_found", campaign_id=cid)
            return error(404, "run_not_found", "the campaign has no run with this id",
                         what_would_fix_it="use a run id the campaign started or recorded a result for")
    now = time.time()
    try:
        with rec.records.txn(write=True) as conn:
            if records_repo.count_links(conn, tenant=tenant) >= rules.MAX_LINKS_PER_TENANT:
                raise RecordsValidationError(
                    "too_many_links", f"a tenant can record at most {rules.MAX_LINKS_PER_TENANT} links")
            hop_limit = rules.MAX_LINKS_PER_TENANT + 1
            rules.check_new_link(
                link,
                duplicate=records_repo.active_duplicate_exists(
                    conn, tenant=tenant, parent_campaign_id=link.parent_campaign_id,
                    parent_run_id=link.parent_run_id, child_campaign_id=link.child_campaign_id,
                    child_run_id=link.child_run_id, relation=link.relation),
                campaign_children=lambda ids: records_repo.cross_links_from(
                    conn, tenant=tenant, campaign_ids=ids, direction="down", limit=hop_limit),
                run_children=lambda runs: records_repo.run_links_within(
                    conn, tenant=tenant, campaign_id=link.parent_campaign_id, parent_run_ids=runs,
                    limit=hop_limit),
            )
            link_id = records_repo.insert_link(
                conn, tenant=tenant, parent_campaign_id=link.parent_campaign_id,
                parent_run_id=link.parent_run_id, child_campaign_id=link.child_campaign_id,
                child_run_id=link.child_run_id, relation=link.relation, parent_stage=link.parent_stage,
                child_stage=link.child_stage, note=link.note, actor=actor(principal), now=now)
            audit_ok(conn, principal, action, campaign_id=link.child_campaign_id, now=now,
                     detail=f"link={link_id};parent={link.parent_campaign_id};relation={link.relation}")
            row = records_repo.get_link(conn, tenant=tenant, link_id=link_id)
    except RecordsValidationError as exc:
        audit_refused(rec, principal, action, 409, code=exc.code)
        return error(409, exc.code, exc.reason)
    return JSONResponse({"link": row}, status_code=201)


@router.get("/api/lineage/links/{link_id}")
def get_link(
    link_id: int,
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    if not 1 <= link_id <= MAX_LINK_ID:
        return _link_not_found()
    with rec.records.txn(write=False) as conn:
        row = records_repo.get_link(conn, tenant=principal.tenant, link_id=link_id)
    if row is None:
        return _link_not_found()
    return JSONResponse({"link": row})


@router.post("/api/lineage/links/{link_id}/retract")
def retract_link(
    link_id: int,
    body: LinkRetract,
    principal: Principal = Depends(require_scope(WRITE)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Withdraw a link. The link row and the retraction (with its reason) are
    both kept; a retracted link no longer counts in lineage or cycle checks."""
    action = "lineage.link.retract"
    if not 1 <= link_id <= MAX_LINK_ID:
        return _link_not_found()
    try:
        reason = rules.check_reason(body.reason)
    except RecordsValidationError as exc:
        return error(400, exc.code, exc.reason)
    now = time.time()
    with rec.records.txn(write=True) as conn:
        row = records_repo.get_link(conn, tenant=principal.tenant, link_id=link_id)
        if row is None:
            return _link_not_found()
        if row["retraction"] is not None:
            already = True
        else:
            already = False
            records_repo.insert_retraction(conn, tenant=principal.tenant, link_id=link_id, reason=reason,
                                           actor=actor(principal), now=now)
            audit_ok(conn, principal, action, campaign_id=row["child_campaign_id"], now=now,
                     detail=f"link={link_id}")
            row = records_repo.get_link(conn, tenant=principal.tenant, link_id=link_id)
    if already:
        audit_refused(rec, principal, action, 409, code="already_retracted")
        return error(409, "already_retracted", "this link was already retracted")
    return JSONResponse({"link": row})


@router.get("/api/campaigns/{campaign_id}/lineage")
def get_campaign_lineage(
    campaign_id: str,
    depth: int = Query(default=rules.DEFAULT_DEPTH, ge=1, le=rules.MAX_DEPTH),
    include_outcomes: bool = Query(default=False),
    principal: Principal = Depends(require_scope(READ)),
    rec: Records = Depends(get_records),
) -> JSONResponse:
    """Where this campaign came from and what came of it: ancestors and
    descendants over active links, up to `depth` hops each way, with each
    campaign's name and recorded stage. `include_outcomes` adds, for each link
    end that names a run, its measured result and its prediction at start."""
    tenant = principal.tenant
    if campaign_info(rec.store, tenant, campaign_id) is None:
        return campaign_not_found_response()
    hop_limit = rules.MAX_LINKS_RETURNED + 1
    with rec.records.txn(write=False) as conn:
        view = rules.lineage_view(
            campaign_id,
            parents_of=lambda ids: records_repo.cross_links_from(
                conn, tenant=tenant, campaign_ids=ids, direction="up", limit=hop_limit),
            children_of=lambda ids: records_repo.cross_links_from(
                conn, tenant=tenant, campaign_ids=ids, direction="down", limit=hop_limit),
            links_among=lambda ids: records_repo.active_links_of_campaigns(
                conn, tenant=tenant, campaign_ids=ids, limit=hop_limit),
            depth=depth,
        )
    cards = campaign_cards(rec, tenant, [n["campaign_id"] for n in view["nodes"]])
    view["nodes"] = [{**n, **cards[n["campaign_id"]]} for n in view["nodes"]]
    if include_outcomes:
        keys = sorted({(link["parent_campaign_id"], link["parent_run_id"]) for link in view["links"]
                       if link["parent_run_id"]}
                      | {(link["child_campaign_id"], link["child_run_id"]) for link in view["links"]
                         if link["child_run_id"]})[:outcome_rules.MAX_RUNS_PER_PAGE]
        readable = {n["campaign_id"] for n in view["nodes"] if n.get("available")}
        keys = [k for k in keys if k[0] in readable]
        with rec.campaigns.txn(write=False) as conn:
            versions = outcomes_repo.versions_for(conn, tenant=tenant, keys=keys)
            predictions = outcomes_repo.predictions_for(conn, tenant=tenant, keys=keys)
        view["run_outcomes"] = outcome_rules.outcomes_page(keys, versions, predictions)
    return JSONResponse(view)


__all__ = ["router"]
