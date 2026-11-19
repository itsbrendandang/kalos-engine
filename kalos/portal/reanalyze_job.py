"""Kalos portal - campaign re-analysis as a background job (docs/JOBS.md,
"Re-analysis jobs").

The same work as the sync `POST /api/campaigns/{id}/reanalyze`
(kalos.portal.campaign_routes._reanalyze): plan the fold of every measured run
into the base dataset, run `_analyze` on the folded dataset, then commit the
fold and the new analysis in ONE transaction with a compare-and-swap on the
plan's generation. Here it runs in the job worker (kalos.portal.job_worker),
with progress stages and cancel, and the commit also marks the job succeeded
in the same transaction (`commit_fold(on_commit=...)`).

`reanalyze_analysis_kwargs` is THE place the `_analyze` keyword arguments of a
re-analysis are built, for both the sync route and the job, so a campaign
setting added later (goal, bounds) reaches both paths.

Torch-free at import: the profile and `_analyze` are imported inside functions.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kalos.portal.campaign import CampaignError
from kalos.portal.jobs import JobError

if TYPE_CHECKING:
    import pandas as pd

    from kalos.portal.campaign import CampaignStore

ProgressFn = Callable[[str, float], None]

# Stages a re-analysis job reports (in order). `fitting`, `calibrating` and
# `proposing` come from `_analyze` itself (kalos/portal/analysis.py).
STAGES = ("planning", "fitting", "calibrating", "proposing", "publishing")

NO_DATA_MESSAGE = "the campaign has no data to analyze; upload a run sheet first"


class ReanalyzeRefused(JobError):
    """A re-analysis that cannot run on the campaign as it is (400). The message
    is authored here or by the store, never echoes caller input or customer data."""

    code = "invalid_request"


class NothingToFold(ReanalyzeRefused):
    """No measured run to fold (the store's message, which names no data)."""

    code = "nothing_to_fold"


class NoData(ReanalyzeRefused):
    """The campaign was never seeded, or has no base rows."""

    code = "no_data"

    def __init__(self) -> None:
        super().__init__(NO_DATA_MESSAGE)


def reanalyze_analysis_kwargs(store: CampaignStore, tenant: str, campaign_id: str | None) -> dict[str, Any]:
    """The `_analyze(df, target, **kwargs)` keyword arguments of a re-analysis of
    `campaign_id` (None = the tenant's default campaign, the legacy route):

    - `profile`: the bioprocess profile (the campaign does not store roles);
    - `pending`: the recipes still awaiting a result, so the next batch does not
      re-propose an experiment already underway (they never reach the folded data);
    - `anonymize`: carried over from the analysis being replaced (the campaign does
      not store the flag); without this a re-analysis of an anonymized upload would
      publish the real column names.

    Campaign setup settings (goal, bounds) belong here when they are added."""
    from kalos.domains import BIOPROCESS_PROFILE

    awaiting = store.awaiting_recipes(tenant=tenant, campaign_id=campaign_id)
    previous = store.load_analysis(tenant=tenant, campaign_id=campaign_id) or {}
    previous_run = previous.get("run")
    anonymize = bool(isinstance(previous_run, dict) and previous_run.get("anonymize"))
    return {"profile": BIOPROCESS_PROFILE, "pending": awaiting, "anonymize": anonymize}


def plan_reanalysis(
    store: CampaignStore, *, tenant: str, campaign_id: str,
) -> tuple[pd.DataFrame, str, str | None]:
    """`store.plan_fold`, with its refusals made specific: `NoData` (never seeded, or
    no base rows), `NothingToFold` (no measured run). CampaignNotFound and
    CampaignArchived pass through. Writes nothing."""
    try:
        df, target, generation = store.plan_fold(tenant=tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        if type(exc) is not CampaignError:
            raise  # not found, archived: their own contract
        if store.get(tenant=tenant, campaign_id=campaign_id) is None:
            raise NoData() from None
        raise NothingToFold(str(exc)) from None
    if df.empty:
        raise NoData()
    return df, target, generation


def run_reanalysis(
    store: CampaignStore, *, tenant: str, campaign_id: str, progress: ProgressFn,
    on_commit: Callable[[Any, str], None], analyze: Callable[..., dict] | None = None,
) -> None:
    """The worker's job body: plan -> analyze (with progress) -> publish. Raises on
    any refusal or failure; nothing is written unless the final commit succeeds."""
    from kalos.portal.pipeline import attach_reanalyze_run, ensure_torch_threads

    if analyze is None:
        from kalos.portal.analysis import _analyze

        analyze = _analyze
    progress("planning", 0.02)
    df, target, generation = plan_reanalysis(store, tenant=tenant, campaign_id=campaign_id)
    kwargs = reanalyze_analysis_kwargs(store, tenant, campaign_id)
    ensure_torch_threads()
    progress("fitting", 0.05)
    result = analyze(df, target, progress=progress, **kwargs)
    result = attach_reanalyze_run(
        df, result, anonymize=kwargs["anonymize"], pending_runs=len(kwargs["pending"]),
    )
    progress("publishing", 0.95)
    store.commit_fold(
        generation, tenant=tenant, campaign_id=campaign_id, analysis=result,
        on_commit=on_commit,
    )
