"""Kalos portal - the background-job worker: runs ONE queued analysis
(docs/JOBS.md, "The worker").

The runner (kalos.portal.job_runner) starts this module in its own process
(`python -m kalos.portal.job_worker <job_id>`), so a GP fit that runs for
minutes never holds the engine's interpreter, and a timed-out or cancelled
job is stopped by killing the process. The work itself is the SAME code the
sync `/api/run` route runs (kalos.portal.pipeline), and the publish is the
same store transaction with a compare-and-swap on the generation the job was
submitted under, plus a commit hook that marks the job succeeded inside it.

Outcomes, each written once by the first writer (kalos.portal.jobs.finish /
mark_succeeded are guarded):

- succeeded: the analysis is published and the job says so, in one commit;
- failed: a refusal or error, with a stable `error_code` and a fixed message
  (never the raw exception text, which can name customer columns);
- cancelled: a cancel was requested before the publish committed;
- timed_out / worker_exited / interrupted: written by the runner, not here.

Torch-free at import (kalos.portal.pipeline is): torch loads only when the
analysis runs.
"""
from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable
from typing import Any

import pandas as pd

from kalos.portal import jobs
from kalos.portal.campaign import CampaignArchived, CampaignConflict, CampaignError, CampaignNotFound
from kalos.portal.migrate import JOB_ID_PATTERN
from kalos.portal.pipeline import compute_upload_analysis, publish_upload_analysis
from kalos.portal.uploads import UploadRejected

log = logging.getLogger("kalos.portal")

# Generic texts for the two parser/fit failures: the same wording the sync
# route returns (kalos/portal/uploads.py), so a job and an upload agree.
_ERR_PARSE_TEXT = "Could not parse the uploaded file. Check it is a CSV or Excel run-sheet."
_ERR_FIT_TEXT = (
    "The model could not be fit on this data - likely near-duplicate or ill-conditioned rows."
)


def classify(exc: BaseException) -> tuple[str, str]:
    """(error_code, safe message) for a job that failed with `exc`. Mirrors
    `kalos.portal.app.upload_error_response`, without importing FastAPI."""
    if isinstance(exc, CampaignConflict):
        return "campaign_changed", jobs.ERROR_MESSAGES["campaign_changed"]
    if isinstance(exc, CampaignNotFound):
        return "campaign_not_found", "campaign not found"
    if isinstance(exc, CampaignArchived):
        return "campaign_archived", str(exc)
    if isinstance(exc, jobs.JobInputError):
        return "input_unavailable", jobs.ERROR_MESSAGES["input_unavailable"]
    if isinstance(exc, CampaignError):
        return "campaign_error", str(exc)
    if isinstance(exc, UploadRejected):
        return exc.code or "upload_rejected", exc.message
    from kalos.core.surrogate import FitError  # deferred: pulls in torch

    if isinstance(exc, FitError):
        return "fit_failed", _ERR_FIT_TEXT
    if isinstance(exc, (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, UnicodeError)):
        return "parse_error", _ERR_PARSE_TEXT
    return "analysis_failed", "could not analyze the uploaded data"


def run_job(
    store: Any,
    job_id: str,
    *,
    analyze: Callable[..., dict] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Run job `job_id` to a terminal state and return it (`succeeded`,
    `failed`, `cancelled`), or the current state if the job was not runnable.

    `analyze` is the engine's `_analyze` (tests pass a stub); `should_stop` lets
    an in-process runner (tests) ask the worker to stop without a process kill.
    """
    row = jobs.load_for_worker(store, job_id)
    if row is None or JOB_ID_PATTERN.fullmatch(job_id) is None:
        return "missing"
    if row["state"] != "running" or row["cancel_requested"]:
        return str(row["state"])
    tenant = str(row["tenant"])
    campaign_id = str(row["campaign_id"])
    stop = should_stop or (lambda: False)

    def progress(stage: str, fraction: float) -> None:
        if stop():
            raise jobs.JobStopped()
        if jobs.cancel_requested(store, tenant=tenant, job_id=job_id):
            raise jobs.JobCancelled()
        jobs.set_progress(store, tenant=tenant, job_id=job_id, stage=stage, progress=fraction)

    try:
        params = jobs.job_params(row)
        raw = jobs.read_input(store, row)
        df, result = compute_upload_analysis(
            raw, params.get("target") or None, bool(params.get("anonymize")),
            str(params.get("filename") or "uploaded dataset"), str(params.get("roles") or ""),
            sheet=params.get("sheet"), progress=progress, analyze=analyze,
        )
        progress("publishing", 0.95)
        publish_upload_analysis(
            store, result, str(params.get("filename") or "uploaded dataset"),
            tenant=tenant, campaign_id=campaign_id, df=df,
            expected_generation=row["base_generation"],
            on_commit=functools.partial(_commit, tenant=tenant, job_id=job_id),
            fallback_unstamped=False,
        )
    except jobs.JobStopped:
        return "stopped"  # the runner that stopped us records the outcome
    except jobs.JobCancelled:
        jobs.finish(store, tenant=tenant, job_id=job_id, state="cancelled")
        return "cancelled"
    except Exception as exc:  # noqa: BLE001 - every failure becomes the job's recorded outcome
        code, message = classify(exc)
        if code in {"analysis_failed", "parse_error", "fit_failed"}:
            log.exception("background job failed during analysis (code=%s)", code)
        else:
            log.warning("background job failed: code=%s", code)
        jobs.finish(store, tenant=tenant, job_id=job_id, state="failed", code=code, message=message)
        return "failed"
    jobs.purge_input(store, row["input_path"])
    return "succeeded"


def _commit(conn: Any, new_generation: str, *, tenant: str, job_id: str) -> None:
    jobs.mark_succeeded(conn, tenant=tenant, job_id=job_id, generation=new_generation)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the worker process: `python -m kalos.portal.job_worker <job_id>`.
    The state directory is `KALOS_STATE_DIR`, as for the engine."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or JOB_ID_PATTERN.fullmatch(args[0]) is None:
        print("usage: python -m kalos.portal.job_worker <job_id>", file=sys.stderr)
        return 2
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore()
    run_job(store, args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
