"""Kalos portal - background jobs: the persistent queue behind
`POST /api/campaigns/{id}/jobs` (docs/JOBS.md).

A job is one upload analysis, queued instead of run inside the request. This
module owns the `jobs` table (schema v3, kalos/portal/migrate.py) and every
state transition on it; the scheduler is `kalos.portal.job_runner`, the work
itself is `kalos.portal.job_worker`, and the HTTP surface is
`kalos.portal.jobs_api`.

Torch-free and FastAPI-free on purpose: the engine imports it on every
request, and the worker process imports it before it does any fitting.

Lifecycle::

    queued --claim--> running --worker publishes--> succeeded
       |                 |  \\--analysis/parse/guard error--> failed
       |                 |  \\--runner: past its time limit--> timed_out
       |                 |  \\--cancel requested + worker stopped--> cancelled
       \\--cancel--------> cancelled

Every transition is a guarded UPDATE (`... AND state = <from>`), so a late
writer (a worker that finished just after its job timed out, a cancel that
raced a commit) can never move a job out of a state it already reached.

Isolation: every API-facing query carries `tenant = ?`; a job id of another
tenant reads exactly like a missing one (`JobNotFound`). The one
cross-tenant read is the scheduler's queue scan (`queued_tenants`), which
never returns job contents to a caller.

Limits (per tenant, env, docs/JOBS.md "Limits"): queue depth (queued +
running jobs), upload size, and the wall-clock time limit. Concurrency (how
many run at once, globally and per tenant) lives in `kalos.portal.busy`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kalos.portal.campaign import (
    CampaignArchived,
    CampaignError,
    CampaignNotFound,
    validate_campaign_id,
)
from kalos.portal.migrate import JOB_ID_PATTERN, JOB_STATES
from kalos.portal.tenancy import validate_tenant_id
from kalos.portal.uploads import MAX_UPLOAD_BYTES

if TYPE_CHECKING:
    from kalos.portal.campaign import CampaignStore

log = logging.getLogger("kalos.portal")

ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "timed_out"})
assert set(JOB_STATES) == set(ACTIVE_STATES) | TERMINAL_STATES

DEFAULT_QUEUE_DEPTH = 8
DEFAULT_TIMEOUT_S = 1800.0
JOBS_DIRNAME = "jobs"
# Shown to the caller for a job that did not finish. The code is the stable
# part of the contract (docs/JOBS.md, "Error codes"); messages are fixed text.
ERROR_MESSAGES = {
    "timeout": "the job exceeded its time limit and was stopped",
    "worker_exited": "the analysis process stopped unexpectedly",
    "worker_unavailable": "the analysis process could not be started",
    "interrupted": "the engine restarted while this job was running; resubmit the upload",
    "engine_stopped": "the engine stopped while this job was running; resubmit the upload",
    "input_unavailable": "the job's input file is missing or changed; resubmit the upload",
    "campaign_changed": (
        "the campaign changed while this job ran, so nothing was published; resubmit the upload"
    ),
}


class JobError(CampaignError):
    """A job-API error the route answers as-is (the message is authored here)."""


class JobNotFound(JobError):
    """No such job FOR THIS TENANT (malformed, missing and foreign ids alike)."""

    def __init__(self) -> None:
        super().__init__("job not found")


class JobsDisabled(JobError):
    def __init__(self) -> None:
        super().__init__("background jobs are disabled on this engine (KALOS_JOBS_ENABLED=0)")


class QueueFull(JobError):
    def __init__(self, depth: int) -> None:
        super().__init__(
            f"this tenant already has {depth} queued or running jobs; wait for one to finish"
        )


class JobTooLarge(JobError):
    def __init__(self, limit: int) -> None:
        super().__init__(f"the upload is larger than this tenant's job upload limit ({limit} bytes)")


class JobCancelled(Exception):
    """Raised inside a worker when its job must stop (cancel requested, or the
    runner stopped it). Not a `CampaignError`: it must never be mistaken for a
    client-facing campaign error."""


class JobStopped(Exception):
    """The worker was told to stop by its runner (engine shutdown, timeout). Not
    a user cancel: the runner records that outcome, so the worker writes none."""


class JobInputError(Exception):
    """The stored input is missing, or its checksum/size no longer matches."""


@dataclass(frozen=True)
class JobSettings:
    enabled: bool
    queue_depth: int
    max_upload_bytes: int
    timeout_s: float


def _env_number(name: str, default: str, cast: type, minimum: float) -> Any:
    raw = os.environ.get(name, "").strip() or default
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number; got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {raw!r}")
    return value


def job_settings() -> JobSettings:
    """The job limits from the environment. Raises ValueError on a malformed or
    out-of-range value; the engine calls this at startup so that fails loudly."""
    enabled = os.environ.get("KALOS_JOBS_ENABLED", "").strip().lower() not in {"0", "false", "no", "off"}
    max_upload = int(_env_number("KALOS_JOB_MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES), int, 1))
    if max_upload > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"KALOS_JOB_MAX_UPLOAD_BYTES must be <= {MAX_UPLOAD_BYTES} (the engine's upload cap)"
        )
    return JobSettings(
        enabled=enabled,
        queue_depth=int(_env_number("KALOS_JOB_QUEUE_DEPTH", str(DEFAULT_QUEUE_DEPTH), int, 1)),
        max_upload_bytes=max_upload,
        timeout_s=float(_env_number("KALOS_JOB_TIMEOUT_S", str(DEFAULT_TIMEOUT_S), float, 1)),
    )


# --- paths -------------------------------------------------------------------- #


def _tenant_key(tenant: str) -> str:
    """A filesystem-safe, non-reversible name for a tenant's input directory
    (tenant ids are restricted, but the hash keeps the path server-chosen)."""
    return hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:16]


def _relative_input_path(tenant: str, job_id: str) -> str:
    return f"{JOBS_DIRNAME}/{_tenant_key(tenant)}/{job_id}.bin"


def _resolve_input(store: CampaignStore, input_path: str) -> Path:
    """The absolute path of a stored input, refusing anything outside the jobs
    directory (the stored value is server-built, this is defense in depth)."""
    base = (store.state_dir / JOBS_DIRNAME).resolve()
    path = (store.state_dir / input_path).resolve()
    if base not in path.parents:
        raise JobInputError("input path escapes the jobs directory")
    return path


def purge_input(store: CampaignStore, input_path: str | None) -> None:
    """Delete a job's stored upload (best-effort: a missing file is fine)."""
    if not input_path:
        return
    try:
        _resolve_input(store, input_path).unlink(missing_ok=True)
    except (OSError, JobInputError):
        log.exception("could not delete a finished job's input file")


# --- rows --------------------------------------------------------------------- #

_COLUMNS = (
    "id, tenant, campaign_id, kind, state, stage, progress, attempts, created_at, "
    "started_at, finished_at, updated_at, input_path, input_sha256, input_bytes, params, "
    "base_generation, result_ref, requested_by, cancel_requested, runner_id, worker_pid, "
    "error_code, error_message, input_purged_at"
)


def _rows(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
    cur = conn.execute(sql, args)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def _one(conn: sqlite3.Connection, tenant: str, job_id: str) -> dict[str, Any] | None:
    found = _rows(conn, f"SELECT {_COLUMNS} FROM jobs WHERE id = ? AND tenant = ?", (job_id, tenant))
    return found[0] if found else None


def job_view(row: dict[str, Any]) -> dict[str, Any]:
    """The public shape of a job (GET /api/jobs/{id} and friends). No input
    path, parameters, runner or pid: those are internal."""
    state = row["state"]
    error = None
    if row["error_code"]:
        error = {"code": row["error_code"], "message": row["error_message"]}
    return {
        "id": row["id"],
        "campaign_id": row["campaign_id"],
        "kind": row["kind"],
        "state": state,
        "stage": row["stage"],
        "progress": row["progress"],
        "attempts": row["attempts"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "updated_at": row["updated_at"],
        "cancel_requested": bool(row["cancel_requested"]),
        "input_bytes": row["input_bytes"],
        "error": error,
        # The campaign generation the analysis was published under (succeeded only).
        "campaign_generation": row["result_ref"] if state == "succeeded" else None,
    }


# --- submission ----------------------------------------------------------------- #


def _write_input(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def enqueue(
    store: CampaignStore, *, tenant: str, campaign_id: str, raw: bytes, filename: str,
    target: str, anonymize: bool, roles: str, sheet: str | None, requested_by: str | None,
) -> dict[str, Any]:
    """Queue one upload analysis for `campaign_id` (which must be `tenant`'s and
    not archived). The upload is written to the state directory first, then the
    row is inserted with the queue-depth check in the same transaction; a
    refused or failed insert removes the file again. Returns the job view.

    `base_generation` records the campaign generation at submission: the
    worker publishes with a compare-and-swap on it, so a campaign that changed
    while the job waited is not silently overwritten (the job fails with
    `campaign_changed` instead).
    """
    settings = job_settings()
    if not settings.enabled:
        raise JobsDisabled()
    validate_tenant_id(tenant)
    validate_campaign_id(campaign_id)
    if len(raw) > settings.max_upload_bytes:
        raise JobTooLarge(settings.max_upload_bytes)
    job_id = "job_" + uuid.uuid4().hex
    rel = _relative_input_path(tenant, job_id)
    path = store.state_dir / rel
    _write_input(path, raw)
    params = {
        "filename": (filename or "uploaded dataset")[:256],
        "target": target[:256],
        "anonymize": bool(anonymize),
        "roles": roles,
        "sheet": sheet[:256] if sheet else None,
    }
    try:
        with store._txn(write=True) as conn:
            campaign = store._resolve(conn, tenant, campaign_id, create=False)
            if campaign is None:
                raise CampaignNotFound()
            if campaign.archived_at is not None:
                raise CampaignArchived()
            depth = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE tenant = ? AND state IN ('queued', 'running')",
                (tenant,),
            ).fetchone()[0]
            if depth >= settings.queue_depth:
                raise QueueFull(settings.queue_depth)
            now = time.time()
            conn.execute(
                "INSERT INTO jobs (id, tenant, campaign_id, kind, state, created_at, updated_at, "
                "progress, stage, attempts, input_path, input_sha256, input_bytes, params, "
                "base_generation, requested_by, cancel_requested) "
                "VALUES (?, ?, ?, 'analysis', 'queued', ?, ?, 0, 'queued', 0, ?, ?, ?, ?, ?, ?, 0)",
                (job_id, tenant, campaign_id, now, now, rel,
                 hashlib.sha256(raw).hexdigest(), len(raw), json.dumps(params),
                 campaign.generation, _clip(requested_by)),
            )
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    job = get_job(store, tenant=tenant, job_id=job_id)
    assert job is not None
    return job


def _clip(value: str | None, limit: int = 512) -> str | None:
    return None if value is None else str(value)[:limit]


# --- reads (tenant-scoped) ------------------------------------------------------ #


def get_job(store: CampaignStore, *, tenant: str, job_id: str) -> dict[str, Any] | None:
    """`tenant`'s job as a public view, or None (malformed ids included)."""
    validate_tenant_id(tenant)
    if not isinstance(job_id, str) or JOB_ID_PATTERN.fullmatch(job_id) is None:
        return None
    with store._txn(write=False) as conn:
        row = _one(conn, tenant, job_id)
    return job_view(row) if row is not None else None


def list_jobs(
    store: CampaignStore, *, tenant: str, campaign_id: str, limit: int = 50,
) -> list[dict[str, Any]]:
    """`campaign_id`'s jobs (newest first). CampaignNotFound unless it is `tenant`'s."""
    validate_tenant_id(tenant)
    validate_campaign_id(campaign_id)
    with store._txn(write=False) as conn:
        if store._row(conn, tenant, campaign_id) is None:
            raise CampaignNotFound()
        rows = _rows(
            conn,
            f"SELECT {_COLUMNS} FROM jobs WHERE tenant = ? AND campaign_id = ? "
            "ORDER BY created_at DESC, id LIMIT ?",
            (tenant, campaign_id, max(1, min(int(limit), 200))),
        )
    return [job_view(r) for r in rows]


def request_cancel(store: CampaignStore, *, tenant: str, job_id: str) -> dict[str, Any]:
    """Cancel `tenant`'s job. A queued job is cancelled at once; a running one
    gets `cancel_requested` and its worker is stopped by the runner within a
    poll interval (the worker also checks at each progress point). A job that
    already finished is returned unchanged. Idempotent. JobNotFound otherwise."""
    validate_tenant_id(tenant)
    if not isinstance(job_id, str) or JOB_ID_PATTERN.fullmatch(job_id) is None:
        raise JobNotFound()
    purge: str | None = None
    with store._txn(write=True) as conn:
        row = _one(conn, tenant, job_id)
        if row is None:
            raise JobNotFound()
        now = time.time()
        if row["state"] == "queued":
            conn.execute(
                "UPDATE jobs SET state = 'cancelled', stage = 'cancelled', cancel_requested = 1, "
                "finished_at = ?, updated_at = ?, input_purged_at = ? "
                "WHERE id = ? AND tenant = ? AND state = 'queued'",
                (now, now, now, job_id, tenant),
            )
            purge = row["input_path"]
        elif row["state"] == "running":
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1, updated_at = ? "
                "WHERE id = ? AND tenant = ? AND state = 'running'",
                (now, job_id, tenant),
            )
        view_row = _one(conn, tenant, job_id)
        assert view_row is not None
    purge_input(store, purge)
    return job_view(view_row)


# --- scheduler and worker side (tenant-scoped where a tenant is known) ---------- #


def queued_tenants(store: CampaignStore) -> list[str]:
    """Tenants with a queued job, oldest queued job first. The scheduler's one
    cross-tenant read: it returns tenant ids only, never job contents."""
    with store._txn(write=False) as conn:
        rows = conn.execute(
            "SELECT tenant FROM jobs WHERE state = 'queued' GROUP BY tenant ORDER BY MIN(created_at)"
        ).fetchall()
    return [str(r[0]) for r in rows]


def claim(store: CampaignStore, *, tenant: str, runner_id: str) -> dict[str, Any] | None:
    """Move `tenant`'s oldest queued job to running. Returns its full row (the
    worker needs the input and params), or None if there is none. The guarded
    UPDATE loses cleanly to a concurrent cancel."""
    validate_tenant_id(tenant)
    with store._txn(write=True) as conn:
        head = conn.execute(
            "SELECT id FROM jobs WHERE tenant = ? AND state = 'queued' "
            "ORDER BY created_at, id LIMIT 1",
            (tenant,),
        ).fetchone()
        if head is None:
            return None
        now = time.time()
        cur = conn.execute(
            "UPDATE jobs SET state = 'running', stage = 'starting', progress = 0, started_at = ?, "
            "updated_at = ?, attempts = attempts + 1, runner_id = ? "
            "WHERE id = ? AND tenant = ? AND state = 'queued'",
            (now, now, runner_id, head[0], tenant),
        )
        if cur.rowcount != 1:
            return None
        return _one(conn, tenant, str(head[0]))


def running_jobs(store: CampaignStore) -> list[dict[str, Any]]:
    """Every running job, any tenant (startup recovery only)."""
    with store._txn(write=False) as conn:
        return _rows(conn, f"SELECT {_COLUMNS} FROM jobs WHERE state = 'running'", ())


def live_input_paths(store: CampaignStore) -> set[str]:
    with store._txn(write=False) as conn:
        return {str(r[0]) for r in conn.execute(
            "SELECT input_path FROM jobs WHERE state IN ('queued', 'running') AND input_path IS NOT NULL"
        ).fetchall()}


def attach_worker(store: CampaignStore, *, tenant: str, job_id: str, pid: int) -> None:
    with store._txn(write=True) as conn:
        conn.execute(
            "UPDATE jobs SET worker_pid = ? WHERE id = ? AND tenant = ? AND state = 'running'",
            (pid, job_id, tenant),
        )


def set_progress(store: CampaignStore, *, tenant: str, job_id: str, stage: str, progress: float) -> None:
    fraction = min(1.0, max(0.0, float(progress)))
    with store._txn(write=True) as conn:
        conn.execute(
            "UPDATE jobs SET stage = ?, progress = ?, updated_at = ? "
            "WHERE id = ? AND tenant = ? AND state = 'running'",
            (stage[:64], fraction, time.time(), job_id, tenant),
        )


def cancel_requested(store: CampaignStore, *, tenant: str, job_id: str) -> bool:
    with store._txn(write=False) as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ? AND tenant = ?", (job_id, tenant)
        ).fetchone()
    return row is not None and bool(row[0])


def finish(
    store: CampaignStore, *, tenant: str, job_id: str, state: str,
    code: str | None = None, message: str | None = None,
) -> bool:
    """Move a QUEUED or RUNNING job to a terminal `state`, and delete its input.
    Returns False (and changes nothing) if the job had already finished: the
    first terminal writer wins."""
    if state not in TERMINAL_STATES:
        raise ValueError(f"not a terminal job state: {state!r}")
    now = time.time()
    with store._txn(write=True) as conn:
        row = _one(conn, tenant, job_id)
        if row is None or row["state"] not in ACTIVE_STATES:
            return False
        cur = conn.execute(
            "UPDATE jobs SET state = ?, stage = ?, error_code = ?, error_message = ?, "
            "finished_at = ?, updated_at = ?, worker_pid = NULL, input_purged_at = ? "
            "WHERE id = ? AND tenant = ? AND state IN ('queued', 'running')",
            (state, state, code, message, now, now, now, job_id, tenant),
        )
        changed = cur.rowcount == 1
    if changed:
        purge_input(store, row["input_path"])
    return changed


def mark_succeeded(conn: sqlite3.Connection, *, tenant: str, job_id: str, generation: str) -> None:
    """The publish commit hook (`CampaignStore.seed(on_commit=...)`): runs INSIDE
    the transaction that writes the campaign and its analysis, so "succeeded"
    and "published" land together or not at all. Raises `JobCancelled` (which
    rolls the publish back) if a cancel was requested or the job is no longer
    running, e.g. it timed out while the analysis was finishing."""
    now = time.time()
    cur = conn.execute(
        "UPDATE jobs SET state = 'succeeded', stage = 'succeeded', progress = 1, "
        "result_ref = ?, finished_at = ?, updated_at = ?, worker_pid = NULL, "
        "error_code = NULL, error_message = NULL, input_purged_at = ? "
        "WHERE id = ? AND tenant = ? AND state = 'running' AND cancel_requested = 0",
        (generation, now, now, now, job_id, tenant),
    )
    if cur.rowcount != 1:
        raise JobCancelled()


def load_for_worker(store: CampaignStore, job_id: str) -> dict[str, Any] | None:
    """The full row of a job by id, for the worker process only. The worker is
    started by the runner with an id the runner just claimed; it never serves a
    caller, and it refuses any job that is not running."""
    if JOB_ID_PATTERN.fullmatch(job_id) is None:
        return None
    with store._txn(write=False) as conn:
        found = _rows(conn, f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
    return found[0] if found else None


def read_input(store: CampaignStore, row: dict[str, Any]) -> bytes:
    """The stored upload, verified against its recorded size and SHA-256."""
    if not row["input_path"]:
        raise JobInputError("job has no input")
    try:
        raw = _resolve_input(store, row["input_path"]).read_bytes()
    except OSError:
        raise JobInputError("input file missing") from None
    if len(raw) != row["input_bytes"] or hashlib.sha256(raw).hexdigest() != row["input_sha256"]:
        raise JobInputError("input file does not match its recorded checksum")
    return raw


def job_params(row: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(row["params"] or "{}")
    return data if isinstance(data, dict) else {}
