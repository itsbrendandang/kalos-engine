"""Kalos portal - `/api/campaigns/{id}/jobs` and `/api/jobs/{id}`: background
analysis jobs (docs/JOBS.md, "HTTP API").

- `POST /api/campaigns/{id}/jobs` (write): upload a run sheet and queue its
  analysis. The same pre-fit refusals as `/api/run` run BEFORE the job is
  queued (`preflight_upload`), so a bad file is a 400 right away, not a failed
  job. 202 with the job; `Location` points at its status.
- `GET /api/campaigns/{id}/jobs` (read): the campaign's recent jobs.
- `GET /api/jobs/{id}` (read): one job's status: state, stage, progress, the
  error code and message if it did not succeed, and the published campaign
  generation if it did.
- `POST /api/jobs/{id}/cancel` (write): cancel a queued or running job.

Security rules, the same as the campaign routes (kalos/portal/campaigns_api.py):
the tenant is always `principal.tenant`; scope is checked before the id is
looked at; a malformed, missing and another tenant's job id produce the
byte-identical 404; caller input is never echoed in an error; every mutation
writes a best-effort audit row (job id, outcome; never the file or its name).

Limits answer honestly: 413 for an upload over the tenant's job size limit,
429 (with Retry-After) when the tenant's queue is full, 503 when jobs are
disabled or when no preflight slot is free (Retry-After).
"""
from __future__ import annotations

import functools
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from kalos.portal import jobs
from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.busy import RETRY_AFTER_SECONDS
from kalos.portal.campaign import (
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    get_campaign_store,
)
from kalos.portal.campaign_routes import campaign_error_response, campaign_not_found_response
from kalos.portal.campaigns_api import CAMPAIGN_UPLOAD_PATH
from kalos.portal.limits import RequestSizeLimitMiddleware, UploadAuthPrecheckMiddleware, route_path
from kalos.portal.migrate import CAMPAIGN_ID_PATTERN
from kalos.portal.pipeline import preflight_upload

log = logging.getLogger("kalos.portal")

router = APIRouter()

# The submit route parses the upload before it is queued (`preflight_upload`, the
# sync route's refusals). That parse is CPU and memory work, so it runs on a thread
# under one of these process-wide slots, separate from the analysis slots of
# busy.py: a tenant's own running job must not stop it from submitting the next one.
# A submit that finds no free slot is refused (503, Retry-After), never queued.
PREFLIGHT_CONCURRENCY = 2
_preflight_slots = threading.BoundedSemaphore(PREFLIGHT_CONCURRENCY)


class PreflightBusy(Exception):
    """Every preflight slot is in use right now."""


# An upload being received holds its whole body in memory (`file.read` below), so the
# number of submissions in flight is capped BEFORE the read: per tenant, and engine-wide.
# A submission counts from the gate to its answer, cancelled or not.
SUBMIT_PER_TENANT = 2
SUBMIT_TOTAL = 4


class SubmitGate:
    """Counts the submissions in flight. `enter` never blocks: it returns the scope that
    is full (`"tenant"` or `"engine"`), or None when the caller is admitted."""

    def __init__(self, per_tenant: int, total: int) -> None:
        self._per_tenant = per_tenant
        self._total = total
        self._lock = threading.Lock()
        self._by_tenant: Counter[str] = Counter()
        self._n = 0

    def enter(self, tenant: str) -> str | None:
        with self._lock:
            if self._by_tenant[tenant] >= self._per_tenant:
                return "tenant"
            if self._n >= self._total:
                return "engine"
            self._by_tenant[tenant] += 1
            self._n += 1
            return None

    def leave(self, tenant: str) -> None:
        with self._lock:
            if self._by_tenant[tenant] > 0:
                self._by_tenant[tenant] -= 1
                self._n -= 1
            if self._by_tenant[tenant] == 0:
                del self._by_tenant[tenant]

    def in_flight(self) -> int:
        with self._lock:
            return self._n


_submit_gate = SubmitGate(SUBMIT_PER_TENANT, SUBMIT_TOTAL)


async def _preflight_gated(call: Callable[[], Any]) -> Any:
    """Run `call` on a thread under a preflight slot. The slot is taken without
    blocking, in the event loop, before a thread is spawned. It is released by the
    thread when the work finishes, so an awaiting request that is cancelled cannot
    free a slot while the parse still runs. If the cancel lands before the thread
    has started the work, the work is skipped and the slot is released here."""
    if not _preflight_slots.acquire(blocking=False):
        raise PreflightBusy()
    guard = threading.Lock()
    state = {"started": False, "abandoned": False, "released": False}

    def release() -> None:
        with guard:
            if state["released"]:
                return
            state["released"] = True
        _preflight_slots.release()

    def work() -> Any:
        with guard:
            if state["abandoned"]:
                return None
            state["started"] = True
        try:
            return call()
        finally:
            release()

    try:
        return await run_in_threadpool(work)
    except BaseException:
        with guard:
            abandon = not state["started"]
            state["abandoned"] = abandon
        if abandon:
            release()
        raise


JOB_UPLOAD_PATH = re.compile(r"/api/campaigns/cmp_[0-9a-f]{32}/jobs/?")
_JOB_NOT_FOUND = {"error": "job not found"}


def _job_not_found() -> JSONResponse:
    return JSONResponse(_JOB_NOT_FOUND, status_code=404)


def _is_campaign_upload(path: str) -> bool:
    """Multipart routes under `/api/campaigns/{id}/`: a campaign upload or a job."""
    return CAMPAIGN_UPLOAD_PATH.fullmatch(path) is not None or JOB_UPLOAD_PATH.fullmatch(path) is not None


class JobUploadSizeLimit(RequestSizeLimitMiddleware):
    """`RequestSizeLimitMiddleware` that gives the multipart campaign routes the
    upload cap: `POST /api/campaigns/{id}/upload` and `POST /api/campaigns/{id}/jobs`.
    (The per-tenant job limit is applied in the route, after the body is read.)"""

    def cap_for(self, path: str) -> int:
        if _is_campaign_upload(path):
            return self.upload_cap
        return super().cap_for(path)


class JobUploadAuthPrecheck(UploadAuthPrecheckMiddleware):
    """Refuse an anonymous multipart campaign upload or job with 401 before its
    body is read, when auth is enforced (see `UploadAuthPrecheckMiddleware`)."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            path = route_path(scope)
            if _is_campaign_upload(path):
                inner = UploadAuthPrecheckMiddleware(
                    self.app, upload_paths=(path,), auth_enforced=self.auth_enforced
                )
                await inner(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


def _audit(
    store: CampaignStore, principal: Principal, action: str, status_code: int,
    *, campaign_id: str | None = None, job_id: str | None = None,
) -> None:
    """Best-effort audit row (never a token, a file, or a file name)."""
    outcome = "ok" if status_code < 400 else f"http_{status_code}"
    log.info("jobs api: %s -> %s", action, outcome)
    try:
        store.record_audit(
            tenant=principal.tenant,
            subject=principal.subject,
            auth_method="anonymous" if principal.anonymous else getattr(principal, "auth_method", "bearer"),
            action=action,
            campaign_id=campaign_id if status_code != 404 else None,
            job_id=job_id if status_code != 404 else None,
            outcome=outcome,
        )
    except Exception:  # noqa: BLE001 - auditing must not break the audited request
        log.exception("failed to write an audit entry for %s", action)


def _valid_campaign_id(campaign_id: str) -> bool:
    return CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is not None


def _job_error_response(exc: CampaignError) -> JSONResponse:
    """The HTTP answer for a job-API error (campaign errors keep their contract)."""
    if isinstance(exc, jobs.JobNotFound):
        return _job_not_found()
    if isinstance(exc, jobs.JobsDisabled):
        return JSONResponse({"error": str(exc)}, status_code=503)
    if isinstance(exc, (jobs.UploadPending, jobs.CampaignHasRuns)):
        return JSONResponse({"error": str(exc), "code": exc.code}, status_code=409)
    if isinstance(exc, (jobs.QueueFull, jobs.EngineQueueFull)):
        return JSONResponse(
            {"error": str(exc)}, status_code=429,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    if isinstance(exc, jobs.JobTooLarge):
        return JSONResponse({"error": str(exc)}, status_code=413)
    return campaign_error_response(exc)


@router.post("/api/campaigns/{campaign_id}/jobs", status_code=202)
async def submit_job(
    campaign_id: str,
    file: UploadFile = File(...),
    target: str = Form(default=""),
    anonymize: bool = Form(default=False),
    roles: str = Form(default=""),
    sheet: str = Form(default=""),
    sheet_query: str | None = Query(default=None, alias="sheet"),
    replace: bool = Form(default=False),
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Queue an analysis of this run sheet into this campaign. 202 + the job."""
    from kalos.portal.app import upload_error_response  # deferred: app imports this module

    job_id: str | None = None
    if not _valid_campaign_id(campaign_id):
        response = campaign_not_found_response()
    else:
        response, job_id = await _submit(
            campaign_id, file, target, anonymize, roles, sheet or sheet_query or None,
            principal, store, upload_error_response, replace=replace,
        )
    _audit(store, principal, "job.submit", response.status_code,
           campaign_id=campaign_id, job_id=job_id)
    return response


async def _submit(
    campaign_id: str, file: UploadFile, target: str, anonymize: bool, roles: str,
    sheet: str | None, principal: Principal, store: CampaignStore, upload_error: Any,
    *, replace: bool = False,
) -> tuple[JSONResponse, str | None]:
    """The submit steps in order; returns the response and the new job id (if any)."""
    try:
        settings = jobs.job_settings()
    except ValueError:
        log.exception("background job settings are invalid")
        return JSONResponse({"error": "background jobs are misconfigured"}, status_code=503), None
    if not settings.enabled:
        return _job_error_response(jobs.JobsDisabled()), None
    # Refuse a full queue before the body is read or parsed (cheap; re-checked in enqueue).
    try:
        jobs.check_queue_capacity(store, tenant=principal.tenant)
    except CampaignError as exc:
        return _job_error_response(exc), None
    try:
        jobs.check_roles(roles)
        jobs.check_no_pending_upload(store, tenant=principal.tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        return _job_error_response(exc), None
    # The upload is about to be held in memory: take a submission slot first.
    refused = _submit_gate.enter(principal.tenant)
    if refused is not None:
        if refused == "tenant":
            message = "too many uploads are being received for this tenant; retry shortly"
            status = 429
        else:
            message = "the engine is receiving too many uploads; retry shortly"
            status = 503
        return JSONResponse({"error": message}, status_code=status,
                            headers={"Retry-After": str(RETRY_AFTER_SECONDS)}), None
    try:
        return await _read_and_queue(
            settings, campaign_id, file, target, anonymize, roles, sheet, principal, store,
            upload_error, replace=replace,
        )
    finally:
        _submit_gate.leave(principal.tenant)


async def _read_and_queue(
    settings: jobs.JobSettings, campaign_id: str, file: UploadFile, target: str,
    anonymize: bool, roles: str, sheet: str | None, principal: Principal, store: CampaignStore,
    upload_error: Any, *, replace: bool = False,
) -> tuple[JSONResponse, str | None]:
    # One byte past the tenant's limit is enough to refuse; never hold more.
    raw = await file.read(settings.max_upload_bytes + 1)
    if len(raw) > settings.max_upload_bytes:
        return _job_error_response(jobs.JobTooLarge(settings.max_upload_bytes)), None
    filename = file.filename or "uploaded dataset"
    try:
        info = store.campaign_info(tenant=principal.tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        return campaign_error_response(exc), None
    if info["archived"]:
        response = JSONResponse(
            {"error": "the campaign is archived; it can be read but not changed"}, status_code=409,
        )
        return response, None
    if not replace:
        # The publish would replace the campaign's runs: only a confirmed upload does that.
        awaiting, measured = jobs.run_counts(store, tenant=principal.tenant, campaign_id=campaign_id)
        if awaiting or measured:
            return _job_error_response(jobs.CampaignHasRuns(awaiting, measured)), None
    # The sync route's refusals, before a job is spent on the file. Nothing is written yet.
    try:
        await _preflight_gated(functools.partial(
            preflight_upload, raw, target or None, anonymize, filename, roles, sheet=sheet,
        ))
    except PreflightBusy:
        return JSONResponse(
            {"error": "the engine is checking other uploads; retry shortly"}, status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        ), None
    except Exception as exc:  # noqa: BLE001 - mapped to the documented envelopes
        return upload_error(exc), None
    try:
        job = jobs.enqueue(
            store, tenant=principal.tenant, campaign_id=campaign_id, raw=raw, filename=filename,
            target=target, anonymize=anonymize, roles=roles, sheet=sheet,
            requested_by=principal.subject, replace=replace,
        )
    except CampaignError as exc:
        return _job_error_response(exc), None
    response = JSONResponse(
        {"job": job}, status_code=202, headers={"Location": f"/api/jobs/{job['id']}"},
    )
    return response, str(job["id"])


@router.get("/api/campaigns/{campaign_id}/jobs")
def list_campaign_jobs(
    campaign_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """This campaign's jobs, newest first."""
    if not _valid_campaign_id(campaign_id):
        return campaign_not_found_response()
    try:
        return JSONResponse({"jobs": jobs.list_jobs(
            store, tenant=principal.tenant, campaign_id=campaign_id, limit=limit,
        )})
    except CampaignNotFound:
        return campaign_not_found_response()


@router.get("/api/jobs/{job_id}")
def get_job(
    job_id: str,
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """One job's status (see the module docstring for the shape)."""
    job = jobs.get_job(store, tenant=principal.tenant, job_id=job_id)
    if job is None:
        return _job_not_found()
    return JSONResponse({"job": job})


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Cancel a queued or running job (a finished job is returned unchanged)."""
    try:
        job = jobs.request_cancel(store, tenant=principal.tenant, job_id=job_id)
    except jobs.JobNotFound:
        _audit(store, principal, "job.cancel", 404)
        return _job_not_found()
    _audit(store, principal, "job.cancel", 200, campaign_id=job["campaign_id"], job_id=job["id"])
    return JSONResponse({"job": job})
