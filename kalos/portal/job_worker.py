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
import math
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import pandas as pd

from kalos.portal import jobs
from kalos.portal.campaign import (
    CampaignArchived,
    CampaignConflict,
    CampaignError,
    CampaignNotFound,
)
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
    if isinstance(exc, jobs.CampaignHasRuns):
        return "campaign_has_runs", jobs.ERROR_MESSAGES["campaign_has_runs"]
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


def _engine_is_gone() -> bool:
    """True when the engine that started this worker is no longer its parent. Compares
    with the engine pid the runner passed in (`ENGINE_PID_ENV`), never with a pid read
    late in startup: a parent that died while this process was importing is reparented
    before any such read, and would then look like the parent."""
    raw = os.environ.get(jobs.ENGINE_PID_ENV, "").strip()
    return raw.isdigit() and os.getppid() != int(raw)


def run_job(
    store: Any,
    job_id: str,
    *,
    analyze: Callable[..., dict] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """Run job `job_id` to a terminal state and return it (`succeeded`,
    `failed`, `cancelled`), or the current state if the job was not runnable,
    `missing` for an unknown id, `unavailable` if the job could not be loaded.

    `analyze` is the engine's `_analyze` (tests pass a stub); `should_stop` lets
    an in-process runner (tests) ask the worker to stop without a process kill.
    """
    try:
        row = jobs.load_for_worker(store, job_id)
    except Exception as exc:  # noqa: BLE001 - reported as a code; the message may quote customer data
        log.error("background job could not be loaded: type=%s", type(exc).__qualname__)
        return "unavailable"
    if row is None or JOB_ID_PATTERN.fullmatch(job_id) is None:
        return "missing"
    if row["state"] != "running" or row["cancel_requested"]:
        return str(row["state"])
    tenant = str(row["tenant"])
    campaign_id = str(row["campaign_id"])
    stop = should_stop or (lambda: False)

    def progress(stage: str, fraction: float) -> None:
        if stop() or _engine_is_gone():
            raise jobs.JobStopped()
        if jobs.cancel_requested(store, tenant=tenant, job_id=job_id):
            raise jobs.JobCancelled()
        jobs.set_progress(store, tenant=tenant, job_id=job_id, stage=stage, progress=fraction)

    try:
        params = jobs.job_params(row)
        if not params.get("replace"):
            # The publish replaces the campaign's runs: refuse before any work, unless the
            # submit confirmed it (replace=true). Nothing is changed on refusal.
            awaiting, measured = jobs.run_counts(store, tenant=tenant, campaign_id=campaign_id)
            if awaiting or measured:
                raise jobs.CampaignHasRuns(awaiting, measured)
        raw = jobs.read_input(store, row)
        filename = jobs.job_filename(params)
        df, result = compute_upload_analysis(
            raw, params.get("target") or None, bool(params.get("anonymize")),
            filename, str(params.get("roles") or ""),
            sheet=params.get("sheet"), progress=progress, analyze=analyze, tenant=tenant,
        )
        progress("publishing", 0.95)
        publish_upload_analysis(
            store, result, filename,
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
            _log_analysis_failure(code, exc)
        else:
            log.warning("background job failed: code=%s", code)
        jobs.finish(store, tenant=tenant, job_id=job_id, state="failed", code=code, message=message)
        return "failed"
    jobs.purge_input(store, row["input_path"])
    return "succeeded"


def _log_analysis_failure(code: str, exc: BaseException) -> None:
    """Log an unexpected analysis failure as its type, code and innermost
    location only (file:line in function), never the message or the source
    lines. A parser or KeyError message quotes customer column names and cell
    values, and worker stderr reaches the engine's log stream (docs/JOBS.md,
    "Logs"). The traceback is dropped on purpose; the location is what a
    developer needs to find the line."""
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        last = frames[-1]
        where = f"{os.path.basename(last.filename)}:{last.lineno} in {last.name}"
    else:
        where = "unknown location"
    log.error(
        "background job failed during analysis: code=%s type=%s at %s",
        code, type(exc).__qualname__, where,
    )


def _commit(conn: Any, new_generation: str, *, tenant: str, job_id: str) -> None:
    # Publish guard, inside the publish transaction (a raise rolls it back): an orphan
    # publishes nothing. mark_succeeded also requires the job to still be `running`,
    # which recovery changes when it fails a job its engine left behind.
    if _engine_is_gone():
        raise jobs.JobStopped()
    jobs.mark_succeeded(conn, tenant=tenant, job_id=job_id, generation=new_generation)


def start_lifetime_watchdog(
    deadline_s: float, *, parent_pid: int, interval_s: float = 1.0,
    exit_fn: Callable[[int], None] = os._exit, kill_group: bool = False,
) -> threading.Event:
    """Stop this worker when nobody is left to read its result: when `deadline_s`
    seconds have passed since the watchdog started, or when its parent (the engine
    that started it) is no longer `parent_pid`, i.e. it died and the worker was
    re-parented. The deadline is measured on the monotonic clock, so a wall-clock
    step neither kills a live worker nor extends one. Runs in a daemon thread.
    Exits with status 3 and writes no job state: the job is recorded by the next
    engine start (`interrupted`). Returns an Event that stops the watchdog (tests;
    the worker process just exits).

    `kill_group` (the worker process only, see `main`): before exiting, SIGKILL the
    worker's own process group, so the children it started die with it. Children
    are not otherwise stopped when their parent exits. Only pass it when this
    process leads its group (`ProcessWorker` makes it so): the group id is then
    this process's own, and no other group is touched."""
    stop = threading.Event()
    end = time.monotonic() + deadline_s

    def _watch() -> None:
        while not stop.wait(interval_s):
            if os.getppid() != parent_pid:
                log.error("job worker: the engine that started it is gone; exiting")
            elif time.monotonic() > end:
                log.error("job worker: past its deadline; exiting")
            else:
                continue
            if kill_group:
                _kill_own_group()
            exit_fn(3)
            return

    threading.Thread(target=_watch, name="kalos-job-watchdog", daemon=True).start()
    return stop


def _may_kill_own_group() -> bool:
    """Only a worker process started by ProcessWorker, and still its own session and
    group leader, may SIGKILL its group. Anything else (an in-process caller, a process
    inheriting the variable) would hit a group it does not own."""
    return (
        os.environ.get(jobs.WORKER_SESSION_ENV) == "1"
        and hasattr(os, "getsid")
        and os.getsid(0) == os.getpid()
        and os.getpgid(0) == os.getpid()
    )


def _kill_own_group() -> None:
    """SIGKILL this process's group. That includes this process, which dies at once
    (SIGKILL cannot be caught), so the caller's `exit_fn` runs only if the signal
    fails, and is then the fallback."""
    if not _may_kill_own_group():
        log.error("job worker: not a session leader of its own; not killing a process group")
        return
    try:
        os.killpg(os.getpid(), signal.SIGKILL)
    except OSError:
        log.error("job worker: could not kill its process group")


def main(argv: list[str] | None = None) -> int:
    """Entry point of the worker process: `python -m kalos.portal.job_worker <job_id>`.
    The state directory is `KALOS_STATE_DIR`, as for the engine. When the runner
    sets `KALOS_JOB_DEADLINE_S`, a watchdog ends the process once it passes, or
    if the engine is gone (see `start_lifetime_watchdog`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or JOB_ID_PATTERN.fullmatch(args[0]) is None:
        print("usage: python -m kalos.portal.job_worker <job_id>", file=sys.stderr)
        return 2
    if _engine_is_gone():
        log.error("job worker: the engine that started it is gone before it began; exiting")
        return 3
    raw_deadline = os.environ.get("KALOS_JOB_DEADLINE_S", "").strip()
    raw_engine = os.environ.get(jobs.ENGINE_PID_ENV, "").strip()
    if raw_deadline or raw_engine.isdigit():
        try:
            seconds = float(raw_deadline) if raw_deadline else math.inf
        except ValueError:
            print("KALOS_JOB_DEADLINE_S must be a number of seconds", file=sys.stderr)
            return 2
        # Only a ProcessWorker-started session leader gets the group kill (see _may_kill_own_group).
        leads_group = _may_kill_own_group()
        parent = int(raw_engine) if raw_engine.isdigit() else os.getppid()
        start_lifetime_watchdog(seconds, parent_pid=parent, kill_group=leads_group)
    from kalos.portal.campaign import CampaignStore

    try:
        store = CampaignStore()
    except Exception as exc:  # noqa: BLE001 - logged by type: the message can name a path or a column
        log.error("job worker could not open the campaign store: type=%s", type(exc).__qualname__)
        return 1
    run_job(store, args[0])
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
