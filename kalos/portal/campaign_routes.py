"""Kalos portal — `/api/campaign*`: the closed optimization loop (M3,
docs/CAMPAIGN_LOOP.md). Thin wrappers over `kalos.portal.campaign.CampaignStore`,
mirroring the router pattern in `kalos.portal.experiments`.

Auth posture: the mutating endpoints (`start`, `result`, `reanalyze`) require
the `write` scope via `kalos.portal.auth` (docs/HARDENING.md, Phase 1). In open
mode (no tokens configured) the anonymous principal holds read+write, so the
local dev/pilot loop is unchanged; once tokens are provisioned these endpoints
enforce them. `GET /api/campaign` stays open for now (read gating is a
follow-up). Per-tenant isolation of the campaign store is the next slice.
"""
from __future__ import annotations

import functools
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, StringConstraints
from pydantic import Strict as _Strict
from starlette.concurrency import run_in_threadpool

from kalos.portal.busy import RETRY_AFTER_SECONDS, AnalysisBusy, run_exclusively

from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign import (
    CampaignArchived,
    CampaignConflict,
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    get_campaign_store,
)
from kalos.portal.uploads import MAX_COLUMNS, UploadRejected

log = logging.getLogger("kalos.portal")

router = APIRouter()


def audit(
    store: CampaignStore, principal: Principal, action: str, status_code: int,
    *, detail: str | None = None,
) -> None:
    """Best-effort audit entry for a legacy mutating route: who (subject + auth
    method), which tenant and campaign (the tenant default), what, and the HTTP
    outcome. Never the token, the body, or any uploaded data. A failure to
    audit is logged and never fails the request it describes."""
    try:
        store.record_audit(
            tenant=principal.tenant,
            subject=principal.subject,
            auth_method="anonymous" if principal.anonymous else "bearer",
            action=action,
            campaign_id=store.default_campaign_id(tenant=principal.tenant),
            outcome="ok" if status_code < 400 else f"http_{status_code}",
            detail=detail,
        )
    except Exception:  # noqa: BLE001 - auditing must not break the audited request
        log.exception("failed to write an audit entry for %s", action)


def _archived(exc: CampaignArchived) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=409)


_NOT_FOUND_BODY = {"error": "campaign not found"}


def campaign_not_found_response() -> JSONResponse:
    """THE 404 for a campaign id the caller may not see: malformed, missing,
    and another tenant's all get this byte-identical response (status, body,
    headers), so existence can never be probed. Used by
    `kalos.portal.campaigns_api` and by `_reanalyze` below."""
    return JSONResponse(_NOT_FOUND_BODY, status_code=404)


def campaign_error_response(exc: CampaignError) -> JSONResponse:
    """Map a store error on an EXPLICIT campaign id to its HTTP response:
    not found -> the shared 404, archived/conflict -> 409, anything else -> 400
    with the store-authored (input-free) message."""
    if isinstance(exc, CampaignNotFound):
        return campaign_not_found_response()
    if isinstance(exc, (CampaignArchived, CampaignConflict)):
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"error": str(exc)}, status_code=400)


# --- request-body bounds (P0 hardening) ------------------------------------- #
# These bodies are untrusted and land in the tenant's persisted campaign, so
# their size and shape are bounded here, before the store sees them. The total
# body size is bounded separately (kalos/portal/limits.py). Validation failures
# on these routes answer 400 with the usual `{"error": ...}` envelope (see the
# RequestValidationError handler in kalos/portal/app.py), not FastAPI's 422.
MAX_RECIPES_PER_START = 64
MAX_STRING_CHARS = 512

_BoundedStr = Annotated[str, StringConstraints(max_length=MAX_STRING_CHARS)]
_FeatureName = Annotated[str, StringConstraints(min_length=1, max_length=MAX_STRING_CHARS)]
# Strict: a JSON number only - no bools, no numeric strings. Ints are accepted.
_FiniteNumber = Annotated[float, _Strict(), AllowInfNan(False)]
# A recipe value is a finite number, or a string for a categorical level (the
# engine's own proposals decode categorical dimensions back to their labels,
# see `DesignSpace.decode_row`), or null for a categorical with no levels.
_RecipeValue = _FiniteNumber | _BoundedStr | None


class _Proposal(BaseModel):
    """One proposed experiment, as `_annotate` emits it. Unknown extra fields
    (`vals`, `p_feasible`, `pred_<constraint>`) are ignored, not stored."""

    model_config = ConfigDict(extra="ignore")

    # Keys are further restricted to the campaign's own dataset columns by
    # `CampaignStore.start`, which is the only place that knows them.
    recipe: dict[_FeatureName, _RecipeValue] = Field(min_length=1, max_length=MAX_COLUMNS)
    pred: _FiniteNumber | None = None
    std: _FiniteNumber | None = None
    mode: _BoundedStr | None = None
    reason: _BoundedStr | None = None


class _StartBody(BaseModel):
    """Body for `POST /api/campaign/start` — the proposed batch the caller
    wants to start running, one entry per `{recipe, pred, std, mode, reason}`
    (the shape `_annotate`'s proposals already carry), at most
    `MAX_RECIPES_PER_START` per call."""

    recipes: list[_Proposal] = Field(max_length=MAX_RECIPES_PER_START)


class _ResultBody(BaseModel):
    """Body for `POST /api/campaign/result`. `value` is a JSON number (bools
    and numeric strings are refused); a non-finite value is refused by the
    store with its own message."""

    id: Annotated[str, StringConstraints(min_length=1, max_length=MAX_STRING_CHARS)]
    value: Annotated[float, _Strict()]


@router.get("/api/campaign")
def get_campaign(
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(READ)),
) -> dict[str, Any]:
    """The caller's tenant's campaign summary for the `/decide` view, or
    `{"has_campaign": false}` before any upload has ever seeded one.

    Joins in the campaign stamp `/api/latest` was published with so the summary
    can report `analysis_in_sync` — whether the analysis `/results` renders and
    this campaign provably describe the same run sheet. The join lives here
    rather than in `CampaignStore` so the store stays free of any `_LATEST`
    knowledge; `_load_latest` is imported inside the function for the same
    circular-import reason as `reanalyze_campaign` below.
    """
    from kalos.portal.app import _load_latest

    latest = _load_latest(principal.tenant, store=store) or {}
    return store.summary(
        tenant=principal.tenant,
        analysis_generation=latest.get("campaign_generation"),
    )


@router.post("/api/campaign/start")
def start_campaign(
    body: _StartBody,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Append each proposed recipe as a pending, awaiting run. Requires `write`."""
    try:
        started = store.start(
            [p.model_dump() for p in body.recipes], tenant=principal.tenant
        )
    except CampaignArchived as exc:
        audit(store, principal, "campaign.start", 409)
        return _archived(exc)
    except CampaignError as exc:
        # Authored by CampaignStore, not a raw library error — safe to echo.
        log.warning("campaign start rejected: %s", exc)
        audit(store, principal, "campaign.start", 400)
        return JSONResponse({"error": str(exc)}, status_code=400)
    audit(store, principal, "campaign.start", 200, detail=f"runs={len(started)}")
    return JSONResponse({"started": started})


@router.post("/api/campaign/result")
def log_campaign_result(
    body: _ResultBody,
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Set the measured outcome on a pending run. Requires `write`. 400 on an
    unknown id or a non-finite value — an unmeasured or bad-value run must never
    silently become foldable into the base dataset."""
    try:
        run = store.set_result(body.id, body.value, tenant=principal.tenant)
    except CampaignArchived as exc:
        audit(store, principal, "campaign.result", 409)
        return _archived(exc)
    except CampaignError as exc:
        log.warning("campaign result rejected: %s", exc)
        audit(store, principal, "campaign.result", 400)
        return JSONResponse({"error": str(exc)}, status_code=400)
    audit(store, principal, "campaign.result", 200)
    return JSONResponse(run)


@router.post("/api/campaign/reanalyze")
async def reanalyze_campaign(
    store: CampaignStore = Depends(get_campaign_store),
    principal: Principal = Depends(require_scope(WRITE)),
) -> JSONResponse:
    """Fold every measured pending run into the base dataset, re-run the
    engine (`_analyze`) on the grown dataset, and publish it as the new
    `/api/latest` — the loop closing (docs/CAMPAIGN_LOOP.md, "Re-analyze =
    the loop closing").

    Transactional: `plan_fold` -> `_analyze` -> `commit_fold`. The fold is
    committed only after `_analyze` succeeds, and only if the campaign's
    `generation` is unchanged (a compare-and-swap in SQL). The new analysis is
    written in that SAME transaction (schema v2: analyses live in portal.db),
    so the fold and the analysis describing it land together or not at all. A
    failed analysis or a concurrent upload that reseeds mid-analysis leaves
    both untouched and returns a retry (409). This closes the sub-millisecond
    window the separate `/api/latest` file cache used to leave open.

    `_analyze` is imported from `kalos.portal.app` INSIDE this function, not at
    module load: `app.py` imports and mounts this router, so a module-level
    import here would be circular. `_analyze` is CPU-bound (a GP fit +
    acquisition optimization), so it runs via `run_in_threadpool`, matching how
    `/api/run` offloads it in `app.py`.
    """
    response = await _reanalyze(store, principal)
    audit(store, principal, "campaign.reanalyze", response.status_code)
    return response


async def _reanalyze(
    store: CampaignStore, principal: Principal, campaign_id: str | None = None
) -> JSONResponse:
    from kalos.domains import BIOPROCESS_PROFILE
    from kalos.portal.app import _analyze

    tenant = principal.tenant
    # Transactional re-analyze (docs/CAMPAIGN_LOOP.md): plan the fold in memory,
    # run the (slow, thread-offloaded) analysis, and only THEN commit. All store
    # access is scoped to this tenant and to `campaign_id` (None = the tenant's
    # default campaign, the legacy route; an explicit id comes from
    # `kalos.portal.campaigns_api`).
    try:
        df, target, generation = store.plan_fold(tenant=tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        return campaign_not_found_response()
    except CampaignArchived as exc:
        return _archived(exc)
    except CampaignError as exc:
        log.warning("reanalyze rejected: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Runs still in the incubator. They have no outcome to fold, so they never
    # reach `df`; passed separately they keep the next batch from re-proposing an
    # experiment already underway.
    awaiting = store.awaiting_recipes(tenant=tenant, campaign_id=campaign_id)

    if df.empty:
        log.warning("reanalyze rejected: campaign has no base rows to analyze")
        return JSONResponse({"error": "the campaign has no data to analyze"}, status_code=400)

    try:
        # Same single analysis slot /api/run holds (kalos/portal/busy.py): a
        # reanalyze racing an upload - or another reanalyze - gets an honest
        # 503 instead of contending for the CPU with an unfinished fit.
        result = await run_in_threadpool(
            run_exclusively(
                functools.partial(
                    _analyze, df, target, profile=BIOPROCESS_PROFILE, pending=awaiting
                )
            )
        )
    except AnalysisBusy as busy:
        return JSONResponse(
            {"error": str(busy)},
            status_code=503,
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    except UploadRejected as rej:
        # An engine-authored, safe refusal (e.g. `target_constant`: every
        # measured value is identical, so no model can be fit and no batch is
        # proposed). Nothing was committed, same as the generic branch below.
        log.warning("reanalyze rejected (%s): %s", rej.code or "guard", rej)
        return JSONResponse(rej.to_body(), status_code=400)
    except Exception:  # noqa: BLE001 - normalized to a generic 400, same as /api/run
        # Full traceback logged server-side; the client only ever sees a
        # generic message, matching /api/run's catch-all in kalos/portal/app.py.
        # Nothing was committed (plan_fold does not mutate), so the campaign is
        # untouched and a retry is meaningful.
        log.exception("failed to re-analyze the campaign")
        return JSONResponse({"error": "could not analyze the campaign data"}, status_code=400)

    try:
        # The fold and the analysis that describes it, in one transaction.
        store.commit_fold(generation, tenant=tenant, campaign_id=campaign_id, analysis=result)
    except CampaignNotFound:
        return campaign_not_found_response()
    except CampaignArchived as exc:
        return _archived(exc)
    except CampaignError as exc:
        # The campaign was reseeded/rewritten while _analyze ran: `result`
        # describes a base dataset that no longer exists. Nothing was written,
        # neither the fold nor the analysis. Ask the caller to retry.
        log.warning("reanalyze not committed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=409)

    # Return the SAME shape GET /api/latest returns (the stored analysis carries
    # `dataset` and `updated`, which the frontend's PopulatedResult type expects).
    saved = store.load_analysis(tenant=tenant, campaign_id=campaign_id) or result
    return JSONResponse(
        {
            "analysis": {"has_data": True, **saved},
            "campaign": store.summary(
                tenant=tenant, campaign_id=campaign_id,
                analysis_generation=saved.get("campaign_generation"),
            ) if campaign_id is not None else store.summary(tenant=tenant),
        }
    )
