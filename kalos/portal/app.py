"""Kalos Engine portal — the JSON API over the real engine (the UI is kalos-web).

A FastAPI app that, on request, runs the BoTorch optimization (single- and
multi-objective) on a synthetic bioprocess surface or an uploaded run sheet and
returns the results. Not a mock: every number comes from an actual BoTorch fit +
acquisition.

Run:  python -m kalos.portal   (serves http://127.0.0.1:8050)
Needs the portal extra:  pip install -e ".[portal]"

This module wires up the FastAPI app, CORS, and the legacy
`/api/run|latest|single|multi` routes. The rest of the portal lives in
sibling modules:
  - `kalos.portal.uploads` - the untrusted-input boundary (`_parse_upload`,
    `UploadRejected`, the upload size/shape caps).
  - `kalos.portal.analysis` - the science (`_analyze` and its helpers).
  - `kalos.portal.serialization` - JSON-safe serialization helpers.
Names historically imported `from kalos.portal.app import ...` are re-exported
below so existing call sites and tests keep working unchanged.
"""
from __future__ import annotations

import functools
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from kalos.portal.busy import RETRY_AFTER_SECONDS, AnalysisBusy, run_exclusively

from kalos.portal.analysis import _analyze, _annotate
from kalos.portal.auth import (
    READ,
    WRITE,
    AuthConfigError,
    Principal,
    get_authenticator,
    require_scope,
)
from kalos.portal.config import (
    assert_production_posture,
    cors_config,
    is_production,
    is_local_client,
    log_security_posture,
    open_access_explicitly_allowed,
)
from kalos.portal.campaign import CampaignError, CampaignStore, get_campaign_store
from kalos.portal.limits import (
    MULTIPART_OVERHEAD_BYTES,
    max_request_bytes,
)
from kalos.portal.migrate import legacy_latest_filename
from kalos.portal.tenancy import validate_tenant_id
from kalos.portal.campaign_routes import audit as _audit
from kalos.portal.campaign_routes import campaign_error_response
from kalos.portal.campaign_routes import router as _campaign_router
from kalos.portal.campaigns_api import router as _campaigns_router
from kalos.portal.jobs_api import JobUploadAuthPrecheck, JobUploadSizeLimit
from kalos.portal.jobs_api import router as _jobs_router
from kalos.portal.pipeline import compute_upload_analysis, publish_upload_analysis
from kalos.portal.scale_routes import router as _scale_router
from kalos.portal.uploads import (
    MAX_COLUMNS,
    MAX_CSV_ROWS,
    MAX_UPLOAD_BYTES,
    MAX_XLSX_CELLS,
    UploadRejected,
    _ERR_FIT,
    _ERR_PARSE,
    _ERR_TOO_MANY_FIT_ROWS,
    _parse_upload,
    parse_upload,
    rejection_counts as _rejection_counts,
    safe_context,
)
from kalos.providers import provider_status

log = logging.getLogger("kalos.portal")

# Routes that accept a multipart file upload, and so get the upload-sized body
# cap rather than the small JSON one (kalos/portal/limits.py).
UPLOAD_PATHS = frozenset({"/api/run", "/api/scale/readout"})

# --- CPU thread budget ------------------------------------------------------- #
# The upload handler offloads its CPU-bound body (pandas parse + ~7 GP fits +
# optimize_acqf) to a worker thread so concurrent uploads do not serialize on the
# event loop. Cap the torch intra-op thread count so several concurrent fits do
# not oversubscribe the CPU and thrash. Override with KALOS_TORCH_THREADS.
# Empty-string-tolerant like the sizing knobs in uploads.py (see the comment
# there): a compose/k8s template passing `KALOS_TORCH_THREADS=` must mean
# "use the default", not crash the import.
#
# KALOS_TORCH_THREADS=0 means DO NOT PIN - leave torch's own default. Not
# hypothetical tuning surface: on the linux-aarch64 torch build inside the
# reference container, the 4-thread pin took the demo analyze from 47s
# (unpinned) to over 600s - a >12x pathological slowdown from the pin
# interacting with that build's OpenBLAS threading, measured 2026-09-10 on
# the first real compose deployment. The 4-thread default is kept for bare
# metal (where it was tuned and behaves); containers, which already isolate
# CPU, should set 0 - deploy/compose.yaml now does.
# The value and the pin live in `kalos.portal.pipeline` (shared with the
# background-job worker process); re-exported under the historical names.
from kalos.portal.pipeline import TORCH_THREADS as _TORCH_THREADS  # noqa: E402,F401
from kalos.portal.pipeline import ensure_torch_threads as _ensure_torch_threads  # noqa: E402

# --- deployment environment (KALOS_ENV) --------------------------------------- #
# Decided once, at startup. KALOS_ENV=production (docs/HARDENING.md):
#   - refuses to start (import fails) unless authentication is enforced; the
#     KALOS_ALLOW_OPEN_ACCESS escape hatch does not apply;
#   - disables the interactive docs and the schema (/docs, /redoc, /openapi.json);
#   - does not register the demo routes /api/single and /api/multi (CPU-heavy GP
#     loops), so they are 404. Outside production they now run under the same
#     admission limiter as every analysis, with a wall-clock timeout.
_PRODUCTION = is_production()
assert_production_posture(auth_enforced=get_authenticator().enforces())

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Open the campaign store before serving the first request.

    Opening it applies the schema migration when `portal.db` is still v1
    (kalos/portal/migrate.py, docs/MIGRATIONS.md): backup first, then one
    transaction. A failed backup or an unsupported schema (`MigrationError`,
    `SchemaError`) stops the process at startup: serving on top of a database
    that could not be migrated safely is never right. Any other failure to open
    the store (e.g. an unwritable state dir) is logged and the process starts
    anyway, so `/readyz` reports it as a generic 503, as it did before.
    """
    from kalos.portal.job_runner import start_job_runner, stop_job_runner
    from kalos.portal.jobs import job_settings
    from kalos.portal.migrate import MigrationError, SchemaError

    job_settings()  # a malformed KALOS_JOB_* value fails the boot, not the first job

    store = None
    try:
        store = app.dependency_overrides.get(get_campaign_store, get_campaign_store)()
    except (MigrationError, SchemaError):
        raise
    except Exception:  # noqa: BLE001 - surfaced by /readyz instead
        log.exception("campaign store could not be opened at startup")
    # The background-job runner (docs/JOBS.md): recovers jobs orphaned by a
    # previous process, then schedules worker processes. KALOS_JOBS_ENABLED=0
    # leaves it off and the job routes answer 503.
    runner = None
    if store is not None:
        try:
            runner = start_job_runner(store)
        except Exception:  # noqa: BLE001 - jobs are optional; the sync routes still work
            log.exception("background job runner could not be started")
    try:
        yield
    finally:
        if runner is not None:
            stop_job_runner(runner)


app = FastAPI(
    title="Kalos Engine API",
    lifespan=_lifespan,
    docs_url=None if _PRODUCTION else "/docs",
    redoc_url=None if _PRODUCTION else "/redoc",
    openapi_url=None if _PRODUCTION else "/openapi.json",
)

# Probes stay reachable in every posture: they carry no tenant data.
_PROBE_PATHS = frozenset({"/healthz", "/readyz"})

# CORS policy: an explicit allowlist when `KALOS_CORS_ORIGINS` is set (production),
# else the permissive localhost default (dev/pilot). See kalos.portal.config.
# Refuse REMOTE callers while running unauthenticated.
#
# With no tokens provisioned every request resolves to an anonymous read+write
# principal on the `default` tenant (kalos.portal.auth), so an open portal that is
# reachable off-box publishes uploads, analyses and campaign mutation to anyone who
# can route to it. Until now that was guarded by a startup warning only.
#
# This is enforced on the PEER ADDRESS rather than on the bind host on purpose. A
# bind-time check lives in `kalos/portal/__main__.py`, but it can be sidestepped
# entirely by launching `uvicorn kalos.portal.app:app --host 0.0.0.0`, which is
# exactly what a real deployment does. The peer address is the actual threat model
# and cannot be avoided by choosing a different entrypoint.
#
# Local development is untouched: loopback callers are always served, so open mode
# still works on a bench machine and in tests. Auth is re-read per request
# (deliberately, for rotation without a restart), so provisioning tokens lifts this
# immediately with no restart either.
@app.middleware("http")
async def _refuse_open_remote_access(request, call_next):
    if _PRODUCTION and request.url.path not in _PROBE_PATHS:
        # Production refuses to START without tokens, but tokens are re-read per
        # request (rotation without restart), so an emptied tokens file would
        # silently reopen the API. Fail closed instead, local callers included.
        try:
            enforced = get_authenticator().enforces()
        except AuthConfigError:
            log.exception("kalos auth: token configuration is invalid")
            return JSONResponse({"error": "authentication is misconfigured"}, status_code=500)
        if not enforced:
            log.error("refused a request: KALOS_ENV=production but authentication is not enforced")
            return JSONResponse(
                {"error": "This kalos portal is in production mode without authentication "
                          "configured and is refusing all requests."},
                status_code=503,
            )
    if not is_local_client(request.client.host if request.client else None):
        if not get_authenticator().enforces() and not open_access_explicitly_allowed():
            log.warning(
                "refused a remote request to an unauthenticated portal from %s %s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            # 503 rather than 401: the caller has no credential to supply and
            # nothing they can send will help. This is a server posture problem,
            # and saying so points the operator at the fix instead of sending the
            # client hunting for a token that does not exist.
            return JSONResponse(
                {
                    "error": (
                        "This kalos portal is running without authentication and "
                        "therefore only serves local requests. The operator should "
                        "set KALOS_AUTH_TOKENS_FILE to enable remote access."
                    )
                },
                status_code=503,
            )
    return await call_next(request)


# Bound every request body before any route parses it (kalos/portal/limits.py):
# a Content-Length over the cap is a 413 without reading the body; a chunked
# body is counted as it streams. Added before CORS so CORS stays outermost and
# a 413 still carries the CORS headers a browser client needs to read it.
app.add_middleware(
    JobUploadSizeLimit,  # CampaignUploadSizeLimit + POST /api/campaigns/{id}/jobs
    default_cap=max_request_bytes(),
    upload_cap=MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES,
    upload_paths=UPLOAD_PATHS,
)
# With auth enforced, an upload carrying no bearer token is a 401 before its
# (up to ~25 MB) multipart body is read. Presence check only; the token is
# verified by `require_principal` as before (kalos/portal/limits.py).
app.add_middleware(
    JobUploadAuthPrecheck,  # CampaignUploadAuthPrecheck + POST /api/campaigns/{id}/jobs
    upload_paths=UPLOAD_PATHS,
    auth_enforced=lambda: get_authenticator().enforces(),
)
app.add_middleware(CORSMiddleware, **cors_config())

# Log the effective security posture (auth + CORS) once at import/startup.
log_security_posture(auth_enforced=get_authenticator().enforces())

BOUNDS = np.array([[0, 0, 0], [1, 1, 1]], float)
TITER_OPT = np.array([0.7, 0.3, 0.5])
PURITY_OPT = np.array([0.2, 0.8, 0.4])  # different recipe -> titer/purity trade off


# Synthetic but honest demo objectives in real units: titer 0..22 mg/L (peaks at
# TITER_OPT), purity 0..100 % (peaks at a different recipe, so they trade off).
def _titer(X: np.ndarray) -> np.ndarray:
    d = np.sum((np.atleast_2d(X) - TITER_OPT) ** 2, axis=1)
    return 22.0 * np.exp(-3.0 * d)


def _purity(X: np.ndarray) -> np.ndarray:
    d = np.sum((np.atleast_2d(X) - PURITY_OPT) ** 2, axis=1)
    return 100.0 * np.exp(-3.0 * d)


def _objectives(X: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(X)
    return np.stack([_titer(x), _purity(x)], axis=-1)


# --- the most-recently analyzed real dataset, PER TENANT ---------------------- #
# The Overview reads /api/latest so the landing page reflects the LAST dataset a
# tenant actually uploaded. Since schema v2 (kalos/portal/migrate.py) that is the
# `analyses` row of the tenant's DEFAULT campaign in portal.db: one durable
# source of truth, written in the same transaction as the campaign it describes
# (`CampaignStore.seed`/`commit_fold`). The former in-memory map and the
# per-tenant `latest/*.json` files are gone; the files of an existing install
# are imported once by the migration and left on disk untouched.
_STATE_DIR = Path(os.environ.get("KALOS_STATE_DIR", Path.home() / ".kalos"))
_LATEST_DIR = _STATE_DIR / "latest"


def _latest_path(tenant: str) -> Path:
    """The LEGACY (schema v1) per-tenant cache file. Nothing writes these any
    more; kept because the v1 -> v2 migration imports them and `migrate down`
    writes them back, and both must agree on the name.

    The filename is an INJECTIVE encoding of the tenant id - the lowercase hex of
    its UTF-8 bytes - so two distinct tenants can never share a file. The id is
    validated (defense in depth: auth already rejects malformed ids at config
    load), and the output alphabet is `[0-9a-f]`, so it can never traverse the
    path.
    """
    validate_tenant_id(tenant)
    return _LATEST_DIR / legacy_latest_filename(tenant)


def _load_latest(tenant: str, *, store: CampaignStore | None = None) -> dict | None:
    """The tenant's published analysis (its default campaign's), or None."""
    validate_tenant_id(tenant)
    return (store or get_campaign_store()).load_analysis(tenant=tenant)


def _save_latest(
    result: dict, dataset: str, *, tenant: str, campaign_generation: str | None = None,
    store: CampaignStore | None = None, campaign_id: str | None = None,
) -> None:
    """Publish `result` as this tenant's `/api/latest`, outside any campaign write.

    The normal paths publish atomically with the campaign instead
    (`CampaignStore.seed(..., analysis=)`, `commit_fold(..., analysis=)`). This
    is the fallback when that could not happen, e.g. seeding failed: the
    analysis is still published, stamped with `campaign_generation` (None =
    not tied to any campaign state, which `GET /api/campaign` reports as
    `analysis_in_sync: false` rather than assuming fine).

    `tenant` is required (no `default` fallback) so a caller can never publish
    into the open-mode tenant by omission; a malformed id raises `ValueError`.
    """
    validate_tenant_id(tenant)
    (store or get_campaign_store()).save_analysis(
        result, dataset, tenant=tenant, campaign_id=campaign_id,
        campaign_generation=campaign_generation,
    )


@app.get("/api/latest")
def latest(
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> dict:
    """The most recent real (uploaded) analysis for the caller's tenant, for the
    Overview. `has_data` is False until the first successful /api/run so the home
    can show an upload prompt instead of pretending there is data."""
    data = _load_latest(principal.tenant, store=store)
    if not data:
        return {"has_data": False}
    return {"has_data": True, **data}


@app.get("/api/providers")
def providers(principal: Principal = Depends(require_scope(READ))) -> dict:
    """Credential status for every external-provider slot (kalos.providers) -
    which ones are live, which are keyless fallbacks, and what each unlocks.
    Never leaks a credential value, only env var names."""
    return {"providers": provider_status()}


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe for a deploy healthcheck (deploy/engine/Containerfile).

    Unauthenticated BY DESIGN, and safe to leave that way: no
    `Depends(require_scope(...))`.
    The response is fixed to exactly `{"status": "ok"}` - no tenant data, no
    config echo, no store contents, nothing beyond "this process is up and
    answering HTTP requests". `engine_version` is deliberately NOT included:
    verified by reading every route in this module that `kalos.__version__`
    is not exposed anywhere unauthenticated today (no public route returns
    it), so this stays the minimal liveness fact rather
    than becoming the first place the version leaves the process without a
    token.

    Must never touch the database - that is what `/readyz` is for. A
    liveness probe that can block on a locked/slow SQLite file would make an
    orchestrator restart a perfectly healthy process because the DB, not the
    process, was briefly unavailable.

    `async def` on purpose: a sync handler runs on the bounded worker
    threadpool, so a liveness probe could queue behind threads busy with
    upload/reanalyze work and time out on a healthy process. On the event loop
    it answers as long as the loop is alive, which is what "live" means.
    """
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness probe: unlike `/healthz`, this DOES touch the database - on
    purpose, to answer "can this instance actually serve requests that need
    the store", not just "is the process up".

    Read-only and tenant-agnostic: `store.ping()` reads the schema version
    (`schema_migrations`), never a write and never any tenant's data, so a
    readiness check can neither corrupt nor contend for the store it checks.
    Degrades to a 503 on any failure (missing/corrupt/locked database file)
    instead of raising, so an orchestrator always gets a normal HTTP response.

    The 503 body carries a FIXED generic reason. This endpoint is
    unauthenticated, and exception text (sqlite messages, filesystem paths)
    is server detail an anonymous caller must not see; the full exception is
    logged server-side instead.

    The store is obtained INSIDE the try (not as a `Depends`), so a failure
    to even construct it (unwritable state dir, corrupt file at open) is the
    same generic 503 rather than an unhandled 500. Resolved through
    `app.dependency_overrides` so tests can still point it at a temp store.
    """
    try:
        store = app.dependency_overrides.get(get_campaign_store, get_campaign_store)()
        store.ping()
    except Exception:  # noqa: BLE001 - any store failure means "not ready", not a 500
        log.exception("readiness check failed: campaign store unavailable")
        return JSONResponse(
            {"status": "unavailable", "reason": "campaign store unavailable"}, status_code=503
        )
    return JSONResponse({"status": "ok"})


# The demo routes take their work size from the URL, so it is capped: each round
# is a GP fit + acquisition, and an unbounded `rounds`/`q` let one request queue
# hours of CPU outside the single analysis slot.
def _single_compute(rounds: int, q: int) -> dict:
    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    _ensure_torch_threads()
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (6, 3))
    y = _titer(X)
    traj = [{"round": 0, "best": float(y.max()), "n": int(len(y))}]
    proposals: list = []
    for r in range(rounds):
        s = Surrogate().fit(X, y, bounds=BOUNDS)
        nxt = propose(s, BOUNDS, q=q)
        X = np.vstack([X, nxt])
        y = np.concatenate([y, _titer(nxt)])
        traj.append({"round": r + 1, "best": float(y.max()), "n": int(len(y))})
    # annotate the last proposed batch with predicted value + uncertainty + why
    # best before this batch was measured; len(nxt), not q: a batch can come
    # back short when the space has fewer than q distinct candidates
    incumbent = float(y[: -len(nxt)].max())
    mean, std = s.posterior(nxt)
    proposals = _annotate(nxt, mean, std, incumbent)
    bi = int(y.argmax())
    return {
        "trajectory": traj,
        "proposals": proposals,
        "best_recipe": np.round(X[bi], 3).tolist(),
        "best_value": float(y[bi]),
        "optimum": TITER_OPT.tolist(),
    }


def _multi_compute(rounds: int, q: int) -> dict:
    from kalos.core.multiobjective import MultiObjectiveSurrogate, propose_multiobjective
    from kalos.core.surrogate import DEVICE, DTYPE

    _ensure_torch_threads()
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (8, 3))
    Y = _objectives(X)
    traj: list = []
    proposals: list = []
    last_batch = np.empty((0, X.shape[1]))
    for r in range(rounds):
        s = MultiObjectiveSurrogate().fit(X, Y, bounds=BOUNDS)
        nxt = propose_multiobjective(s, BOUNDS, q=q)
        X = np.vstack([X, nxt])
        Y = np.vstack([Y, _objectives(nxt)])
        _, py = s.pareto()
        traj.append(
            {"round": r + 1, "pareto": int(len(py)), "best_titer": float(Y[:, 0].max()), "best_purity": float(Y[:, 1].max())}
        )
        last_batch = nxt
    s = MultiObjectiveSurrogate().fit(X, Y, bounds=BOUNDS)
    assert s.model is not None  # fit() always sets the ModelListGP
    # predicted titer + purity (with uncertainty) for each proposed experiment
    import torch
    post = s.model.posterior(torch.as_tensor(last_batch, dtype=DTYPE, device=DEVICE))
    mY = post.mean.detach().cpu().numpy().reshape(len(last_batch), -1)
    sdY = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy().reshape(len(last_batch), -1)
    sd_mean = sdY.mean(axis=1)
    sd_thr = float(np.quantile(sd_mean, 2 / 3)) if len(sd_mean) > 2 else float(sd_mean.max())
    proposals = [{
        "vals": np.round(last_batch[i], 3).tolist(),
        "pred_titer": round(float(mY[i, 0]), 1), "pred_purity": round(float(mY[i, 1]), 0),
        "mode": "explore" if sd_mean[i] >= sd_thr else "exploit",
        "reason": "fills a gap on the titer/purity frontier",
    } for i in range(len(last_batch))]
    _, py = s.pareto()
    order = np.argsort(py[:, 0])
    return {
        "trajectory": traj,
        "pareto": [{"titer": float(py[i, 0]), "purity": float(py[i, 1])} for i in order],
        "all": [{"titer": float(Y[i, 0]), "purity": float(Y[i, 1])} for i in range(len(Y))],
        "proposals": proposals,
    }


# Demo-route wall clock. The thread cannot be killed, so on timeout the client
# gets a 504 while the computation finishes in the background, STILL holding
# its admission slot (released by the worker thread, kalos/portal/busy.py).
DEMO_TIMEOUT_SECONDS = float(os.environ.get("KALOS_DEMO_TIMEOUT_S", "").strip() or "120")


async def _run_demo(fn, principal: Principal, *args) -> JSONResponse:
    """Run a demo GP loop in the threadpool under the shared admission limiter
    (busy.py) and a wall-clock timeout, so a demo call can neither starve the
    analysis paths nor block the event loop (health probes stay live)."""
    import asyncio

    try:
        locked = run_exclusively(functools.partial(fn, *args), tenant=principal.tenant)
    except AnalysisBusy as busy:
        return JSONResponse(
            {"error": str(busy)}, status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    task = asyncio.ensure_future(run_in_threadpool(locked))
    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=DEMO_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        log.warning("demo route exceeded %ss; answered 504", DEMO_TIMEOUT_SECONDS)
        return JSONResponse({"error": "the demo computation timed out"}, status_code=504)
    return JSONResponse(result)


async def run_single(
    rounds: int = Query(6, ge=1, le=12),
    q: int = Query(2, ge=1, le=4),
    _principal: Principal = Depends(require_scope(READ)),
) -> JSONResponse:
    return await _run_demo(_single_compute, _principal, rounds, q)


async def run_multi(
    rounds: int = Query(5, ge=1, le=12),  # >= 1, so last_batch is always bound
    q: int = Query(2, ge=1, le=4),
    _principal: Principal = Depends(require_scope(READ)),
) -> JSONResponse:
    return await _run_demo(_multi_compute, _principal, rounds, q)


# Demo routes are development conveniences: never served in production.
if not _PRODUCTION:
    app.get("/api/single")(run_single)
    app.get("/api/multi")(run_multi)


def _run_uploaded_sync(
    raw: bytes, target: str | None, anonymize: bool, filename: str, roles_json: str = "",
    *, tenant: str, sheet: str | None = None, store: CampaignStore | None = None,
    campaign_id: str | None = None,
) -> dict:
    """The CPU-bound body of an upload: parse -> analyze -> persist -> result.

    This is the heavy, blocking work (pandas parse, ~7 GP fits, optimize_acqf) and
    runs in a worker thread (see `run_uploaded`), NOT on the asyncio event loop, so
    concurrent uploads do not serialize and GET /api/latest never hangs behind a
    fit. It raises `UploadRejected` / `ValueError` / `FitError` / parser errors;
    the async wrapper maps each to the right 400 envelope. Kept fully synchronous
    so it is trivially unit-testable in isolation.

    When `roles_json` is a non-empty JSON object it is parsed into a `ColumnRoles`
    schema and the analysis runs domain-neutrally (the generic profile) instead of
    inferring bioprocess roles from column names. Omitted, the bioprocess profile
    infers roles exactly as before.
    """
    # parse -> analyze (the CPU-heavy part), then seed + publish: the same two
    # functions the background-job worker uses (kalos/portal/pipeline.py).
    # `_analyze` is resolved from THIS module at call time so tests can stub it.
    validate_tenant_id(tenant)
    df, result = compute_upload_analysis(
        raw, target, anonymize, filename, roles_json, sheet=sheet, analyze=_analyze,
    )
    # A fresh upload starts a fresh campaign state (docs/CAMPAIGN_LOOP.md,
    # "Seeding") in THIS tenant's default campaign (or the explicit
    # `campaign_id`), and publishes the analysis in the same transaction,
    # stamped with the generation it was seeded with. Best-effort for the
    # default campaign; an explicit campaign that vanished or was archived
    # mid-analysis raises (404/409) instead of returning an unstored analysis.
    publish_upload_analysis(
        store if store is not None else get_campaign_store(), result, filename,
        tenant=tenant, campaign_id=campaign_id, df=df,
    )
    return result


@app.post("/api/run")
async def run_uploaded(
    file: UploadFile = File(...),
    target: str = Form(default=""),
    anonymize: bool = Form(default=False),
    roles: str = Form(default=""),
    sheet: str = Form(default=""),
    sheet_query: str | None = Query(default=None, alias="sheet"),
    _principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Upload a run sheet and analyze it.

    `sheet` (multipart form field, or `?sheet=` query parameter; the form field
    wins) picks a workbook sheet by name. Omitted, the first sheet in workbook
    order that holds a usable table is analyzed (`kalos.portal.uploads.
    _parse_excel`). Ignored for CSV/TSV uploads. Excel responses, success or
    error, carry `sheets` and `sheet_used`.

    The outcome is written to the audit log (tenant, subject, status; never the
    file, its name, or its contents).
    """
    response = await _run_uploaded(
        file, target, anonymize, roles, sheet, sheet_query, _principal, store
    )
    _audit(store, _principal, "run.upload", response.status_code)
    return response


async def _run_uploaded(
    file: UploadFile, target: str, anonymize: bool, roles: str, sheet: str,
    sheet_query: str | None, _principal: Principal, store: CampaignStore,
    *, campaign_id: str | None = None,
) -> JSONResponse:
    # One byte past the cap is enough for `_parse_upload` to reject it; never
    # load a multi-GB body into this single process just to measure it.
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    filename = file.filename or "uploaded dataset"
    tenant = _principal.tenant
    try:
        # Offload the CPU-bound parse + fit + save to a worker thread so this
        # single-worker service does not block the event loop (and every other
        # request, including GET /api/latest) while a GP fit runs - under the
        # single analysis slot (kalos/portal/busy.py): a second upload while
        # one is fitting gets an honest 503, and a disconnected client's
        # orphaned fit keeps the slot until it truly finishes.
        result = await run_in_threadpool(
            run_exclusively(
                functools.partial(
                    _run_uploaded_sync, raw, target or None, anonymize, filename, roles,
                    tenant=tenant, sheet=(sheet or sheet_query or None), store=store,
                    campaign_id=campaign_id,
                ),
                tenant=tenant,
            )
        )
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - mapped to the documented envelopes
        return upload_error_response(exc)


def upload_error_response(error: BaseException) -> JSONResponse:
    """THE error contract of an upload analysis (`/api/run`, the campaign
    upload, and the pre-queue check of `POST /api/campaigns/{id}/jobs`)."""
    from kalos.core.surrogate import FitError  # deferred: only needed to match below

    try:
        raise error
    except AnalysisBusy as busy:
        return JSONResponse(
            {"error": str(busy)},
            status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    except CampaignError as exc:
        # Only raised for an explicit campaign id (see _run_uploaded_sync).
        log.warning("campaign upload not stored: %s", exc)
        return campaign_error_response(exc)
    except UploadRejected as rej:
        # A guard tripped: the message is already generic and safe to return.
        # `to_body()` is `{"error": ..., "code": ..., **context}`; `error` is
        # always present, so a client that only reads `d.error` is unaffected.
        # Logged by code and counts only: the message can name the customer's
        # columns and sheets, which do not belong in server logs.
        log.warning("upload rejected: code=%s%s", rej.code or "guard", _rejection_counts(rej))
        return JSONResponse(rej.to_body(), status_code=400)
    except FitError as exc:
        # A numerically-hard-but-valid file that defeated the jittered fit retry.
        # Give it a DISTINCT message so it does not masquerade as a parse failure;
        # the exception text itself is already generic, but we do not echo it.
        log.warning("upload rejected: model could not be fit (ill-conditioned rows)")
        return JSONResponse(
            {"error": _ERR_FIT, "code": "fit_failed", **safe_context(getattr(exc, "upload_context", {}))},
            status_code=400,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeError) as exc:
        # Log the full traceback server-side for debugging; return a generic
        # message so no parser detail, column name, cell value, or stack trace
        # ever reaches the (unauthenticated) client.
        log.exception("failed to parse or analyze an uploaded file")
        return JSONResponse(
            {"error": _ERR_PARSE, "code": "parse_error", **safe_context(getattr(exc, "upload_context", {}))},
            status_code=400,
        )
    except Exception as exc:  # noqa: BLE001 - normalized to a generic 400 below
        # Catch-all so a fit-time failure that is NOT a plain ValueError still
        # returns the {error} JSON envelope the frontend parses, never FastAPI's
        # default text/plain HTTP 500. This covers torch.linalg.LinAlgError from a
        # GP fit and an AssertionError from the leakage guard, among others. As
        # above, the full traceback is logged server-side and only the generic
        # message reaches the (unauthenticated) client.
        log.exception("failed to parse or analyze an uploaded file")
        return JSONResponse(
            {"error": _ERR_PARSE, "code": "parse_error", **safe_context(getattr(exc, "upload_context", {}))},
            status_code=400,
        )


# --- campaign loop: /api/campaign* (docs/CAMPAIGN_LOOP.md) - the closed
# optimization loop: propose -> run -> log outcome -> re-propose. Routes live
# in `kalos.portal.campaign_routes`; mounted here so they are served by this
# app.
app.include_router(_campaign_router)
app.include_router(_campaigns_router)  # /api/campaigns* (kalos.portal.campaigns_api)
app.include_router(_jobs_router)  # background jobs (kalos.portal.jobs_api, docs/JOBS.md)


_CAMPAIGN_BODY_FIELDS = frozenset(
    {"recipes", "recipe", "pred", "std", "mode", "reason", "id", "value", "[key]",
     "name", "archived", "include_archived"}
)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Body validation failures on `/api/campaign/*` answer 400 with the
    `{"error": ...}` envelope those routes already use for every other bad
    input (so the client parses one shape). The message names WHERE and WHAT
    (field location + pydantic's message) but never echoes the input value.
    Every other route keeps FastAPI's default 422."""
    if not request.url.path.startswith("/api/campaign"):
        return await request_validation_exception_handler(request, exc)
    parts = []
    for err in exc.errors()[:5]:
        # Dict keys show up in `loc` (a recipe's feature names): those are caller
        # input, so anything that is not a list index or a known field name is
        # masked rather than echoed.
        loc = ".".join(
            str(p) if isinstance(p, int) or p in _CAMPAIGN_BODY_FIELDS else "<key>"
            for p in err.get("loc", ())
            if p != "body"
        )
        parts.append(f"{loc or 'body'}: {err.get('msg', 'invalid')}"[:200])
    return JSONResponse({"error": "invalid request body: " + "; ".join(parts)}, status_code=400)

# --- Scale-Up Readout: /api/scale/readout (stateless, D6) - a one-page HTML
# readout over an uploaded multi-scale run sheet. Routes live in
# `kalos.portal.scale_routes`; mounted here, same pattern as the router
# above.
app.include_router(_scale_router)

__all__ = [
    "app",
    "get_campaign_store",
    "_analyze",
    "UploadRejected",
    "MAX_COLUMNS",
    "MAX_CSV_ROWS",
    "MAX_UPLOAD_BYTES",
    "MAX_XLSX_CELLS",
    "_ERR_FIT",
    "_ERR_PARSE",
    "_ERR_TOO_MANY_FIT_ROWS",
    "_parse_upload",
    "parse_upload",
]
