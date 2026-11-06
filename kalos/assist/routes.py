"""`/api/assist/*`: the guide, the question box, and the index (docs/ASSISTANT.md).

- GET  /api/assist/next?campaign_id=     read   the guide's one next step
- POST /api/assist/ask                   read   a sourced answer, or the guide
- GET  /api/assist/index/status          read   provider, model, where it runs, index size
- POST /api/assist/index/rebuild         write  rebuild the index (audited)

The tenant is always the caller's (`principal.tenant`). A campaign id that is
malformed, missing, or another tenant's gets the shared campaign 404. With
`KALOS_LLM_PROVIDER=none`, `/ask` answers 404 `assist_disabled`; the other
routes keep working. No route here writes campaign state.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints

from kalos.assist.chat import MAX_QUESTION_CHARS, ask
from kalos.assist.guide import next_step
from kalos.assist.http import (
    assist_disabled,
    assist_disabled_response,
    provider_unavailable_response,
)
from kalos.assist.index import AssistIndex
from kalos.assist.llm import ChatModel, ModelUnavailable, get_chat_model, model_info
from kalos.assist.state import campaign_inputs
from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign import CampaignNotFound, CampaignStore, get_campaign_store
from kalos.portal.campaign_routes import campaign_not_found_response
from kalos.portal.campaigns_api import _audit, _valid_campaign_id

log = logging.getLogger("kalos.assist")

router = APIRouter()


class _AskBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(min_length=1, max_length=MAX_QUESTION_CHARS)]
    campaign_id: str | None = None


def _guide(store: CampaignStore, tenant: str, campaign_id: str | None) -> dict[str, Any]:
    cid, summary, analysis, refusal = campaign_inputs(store, tenant, campaign_id)
    return next_step(summary, analysis, refusal, campaign_id=cid)


@router.get("/api/assist/next")
def assist_next(
    campaign_id: str | None = Query(default=None),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """The guide output for one campaign (the tenant default when omitted)."""
    if campaign_id is not None and not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        return JSONResponse(_guide(store, principal.tenant, campaign_id))
    except CampaignNotFound:
        return campaign_not_found_response()


@router.post("/api/assist/ask")
def assist_ask(
    body: _AskBody,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
    model: ChatModel | None = Depends(get_chat_model),
) -> JSONResponse:
    """A sourced answer from the tenant's own data and the docs, or the guide
    step when nothing relevant is found or the reply fails the checks."""
    if assist_disabled() or model is None:
        return assist_disabled_response()
    if body.campaign_id is not None and not _valid_campaign_id(body.campaign_id):
        return campaign_not_found_response()
    try:
        guide = _guide(store, principal.tenant, body.campaign_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    index = AssistIndex(store, principal.tenant)
    cid = guide.get("campaign_id")
    if index.path.exists():
        index.refresh(campaign_ids=[cid] if cid else [])
    else:
        index.refresh()
    hits = index.search(body.question, campaign_id=cid)
    try:
        result = ask(body.question, hits, guide, model)
    except ModelUnavailable:
        log.warning("assist: the model did not answer")
        return provider_unavailable_response()
    # Server log: counts only, never the question or the answer.
    log.info("assist ask: hits=%d sources=%d abstained=%s", len(hits), len(result["sources"]),
             result["abstained"])
    return JSONResponse(result)


def _status(store: CampaignStore, tenant: str, *, build: bool = False) -> dict[str, Any]:
    index = AssistIndex(store, tenant)
    if build and not index.path.exists():
        index.refresh()  # first look: build it, so the count is real (cheap with BM25)
    return {**model_info(), **index.status()}


@router.get("/api/assist/index/status")
def assist_index_status(
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """`{provider, model, local, chunks, built_at, building, retriever}`. The
    index is built on the first call if it does not exist yet."""
    return JSONResponse(_status(store, principal.tenant, build=True))


@router.post("/api/assist/index/rebuild")
def assist_index_rebuild(
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Rebuild the caller's index from scratch, synchronously. Audited."""
    try:
        AssistIndex(store, principal.tenant).rebuild()
    except Exception:
        log.exception("assist: index rebuild failed")
        _audit(store, principal, "assist.index.rebuild", 500, None)
        raise
    _audit(store, principal, "assist.index.rebuild", 200, None)
    return JSONResponse(_status(store, principal.tenant))


__all__ = ["router"]
