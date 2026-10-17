"""Kalos portal - the background-job runner: schedules queued jobs onto worker
processes and supervises them (docs/JOBS.md, "The runner").

One runner per engine process, started from the app lifespan
(`start_job_runner`) and stopped on shutdown (`stop_job_runner`). Each pass
(`tick`) does two things:

1. Supervise the children it started: a child that exited is reconciled (a
   worker that exited without reaching a terminal state is `failed` as
   `worker_exited`); a child whose job was cancelled is killed and the job is
   `cancelled`; a child past its wall-clock limit is killed and the job is
   `timed_out`. Each child holds one analysis slot until it exits.
2. Admit: while a slot is free (kalos.portal.busy: a global cap plus a
   per-tenant cap), claim the oldest queued job of the tenant that has been
   waiting longest, and start its worker.

Fairness: tenants are visited oldest-queued-job first, and a tenant at its
per-tenant cap is skipped, so one tenant's backlog cannot hold every slot.
Sync routes yield to queued work (busy.py `yield_to_queue`).

Crash safety: a job left `running` by a previous engine process is failed
(`interrupted`) at startup, after its worker process is killed if the pid is
still alive and is provably the job's worker (its command line names the job
id). Input files of jobs that are no longer active are swept.

`spawn` is injectable: production starts `python -m kalos.portal.job_worker`;
tests run the same worker code in a thread (`ThreadWorker`) with a stop flag.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from kalos.portal import jobs
from kalos.portal.busy import AnalysisLimiter, Lease, limiter

if TYPE_CHECKING:
    from kalos.portal.campaign import CampaignStore

log = logging.getLogger("kalos.portal")

KILL_WAIT_S = 5.0


class Worker(Protocol):
    @property
    def pid(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def kill(self) -> None: ...


Spawn = Callable[[str, str, Path], Worker]


class ProcessWorker:
    """A worker in its own session, so a kill takes its whole process group."""

    def __init__(self, job_id: str, state_dir: Path) -> None:
        env = {**os.environ, "KALOS_STATE_DIR": str(state_dir)}
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "kalos.portal.job_worker", job_id],
            env=env, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self._pid = self._proc.pid

    @property
    def pid(self) -> int | None:
        return self._pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def kill(self) -> None:
        _kill_group(self._pid)
        try:
            self._proc.wait(timeout=KILL_WAIT_S)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL is not ignorable
            log.error("a job worker did not exit after SIGKILL")


class ThreadWorker:
    """The worker code run in a thread (tests). `kill()` only asks it to stop
    at its next progress point; a thread cannot be killed."""

    def __init__(self, run: Callable[[Callable[[], bool]], None]) -> None:
        self._stop = threading.Event()
        self._error: BaseException | None = None

        def _target() -> None:
            try:
                run(self._stop.is_set)
            except BaseException as exc:  # noqa: BLE001 - reported as the worker's exit
                self._error = exc

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()

    @property
    def pid(self) -> int | None:
        return None

    def poll(self) -> int | None:
        if self._thread.is_alive():
            return None
        return 1 if self._error is not None else 0

    def kill(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)  # a thread blocked in a fit is not waited for


def _kill_group(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def _is_job_worker(pid: int, job_id: str) -> bool:
    """Whether `pid` is alive and its command line names `job_id` (so a reused
    pid is never signalled)."""
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return job_id in out and "job_worker" in out


def recover(store: CampaignStore) -> int:
    """Startup recovery: fail the jobs an earlier engine process left running
    (killing their workers if they are still alive), sweep orphaned input
    files, and return how many jobs were failed."""
    failed = 0
    for row in jobs.running_jobs(store):
        pid = row.get("worker_pid")
        if isinstance(pid, int) and _is_job_worker(pid, str(row["id"])):
            _kill_group(pid)
        if jobs.finish(store, tenant=str(row["tenant"]), job_id=str(row["id"]),
                       state="failed", code="interrupted",
                       message=jobs.ERROR_MESSAGES["interrupted"]):
            failed += 1
    _sweep_orphan_inputs(store)
    if failed:
        log.warning("background jobs: %d job(s) left running by a previous process were failed", failed)
    return failed


def _sweep_orphan_inputs(store: CampaignStore) -> None:
    base = store.state_dir / jobs.JOBS_DIRNAME
    if not base.is_dir():
        return
    live = jobs.live_input_paths(store)
    for path in base.glob("*/*.bin"):
        rel = f"{jobs.JOBS_DIRNAME}/{path.parent.name}/{path.name}"
        if rel not in live:
            path.unlink(missing_ok=True)


class _Child:
    __slots__ = ("job_id", "tenant", "worker", "lease", "deadline")

    def __init__(self, job_id: str, tenant: str, worker: Worker, lease: Lease, deadline: float) -> None:
        self.job_id = job_id
        self.tenant = tenant
        self.worker = worker
        self.lease = lease
        self.deadline = deadline


class JobRunner:
    def __init__(
        self,
        store: CampaignStore,
        *,
        spawn: Spawn | None = None,
        limit: AnalysisLimiter | None = None,
        timeout_s: float | None = None,
        poll_s: float = 0.5,
    ) -> None:
        self.store = store
        self.runner_id = uuid.uuid4().hex
        self._spawn = spawn or (lambda job_id, _tenant, state_dir: ProcessWorker(job_id, state_dir))
        self._limit = limit
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._children: dict[str, _Child] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- wiring -------------------------------------------------------------- #

    @property
    def limit(self) -> AnalysisLimiter:
        return self._limit if self._limit is not None else limiter()

    @property
    def timeout_s(self) -> float:
        return self._timeout_s if self._timeout_s is not None else jobs.job_settings().timeout_s

    def start(self) -> None:
        recover(self.store)
        self.limit.add_listener(self._wake.set)
        self._thread = threading.Thread(target=self._loop, name="kalos-job-runner", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Stop scheduling and stop every running worker. Their jobs are failed
        as `engine_stopped`, so nothing is left `running` across a restart."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.limit.remove_listener(self._wake.set)
        for child in list(self._children.values()):
            child.worker.kill()
            self._finish(child, "failed", "engine_stopped")
            self._drop(child)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the scheduler must survive one bad pass
                log.exception("background job scheduler pass failed")
            self._wake.wait(self._poll_s)
            self._wake.clear()

    # --- one pass ------------------------------------------------------------ #

    def tick(self) -> None:
        self._supervise()
        self._admit()

    def running_count(self) -> int:
        return len(self._children)

    def _finish(self, child: _Child, state: str, code: str | None) -> None:
        message = jobs.ERROR_MESSAGES.get(code or "")
        jobs.finish(self.store, tenant=child.tenant, job_id=child.job_id, state=state,
                    code=code, message=message)

    def _drop(self, child: _Child) -> None:
        self._children.pop(child.job_id, None)
        child.lease.release()

    def _supervise(self) -> None:
        now = time.monotonic()
        for child in list(self._children.values()):
            if child.worker.poll() is None:
                if jobs.cancel_requested(self.store, tenant=child.tenant, job_id=child.job_id):
                    child.worker.kill()
                    self._finish(child, "cancelled", None)
                elif now > child.deadline:
                    log.warning("background job exceeded its time limit; stopping it")
                    child.worker.kill()
                    self._finish(child, "timed_out", "timeout")
                else:
                    continue
            else:
                # Exited. A clean exit has already reached a terminal state (the
                # worker wrote it); anything still active is a crash.
                self._finish(child, "failed", "worker_exited")
            self._drop(child)

    def _admit(self) -> None:
        self.limit.set_waiting(set(jobs.queued_tenants(self.store)))
        while not self._stop.is_set():
            started = False
            for tenant in jobs.queued_tenants(self.store):
                lease = self.limit.try_acquire(tenant, yield_to_queue=False)
                if lease is None:
                    continue
                job = jobs.claim(self.store, tenant=tenant, runner_id=self.runner_id)
                if job is None:
                    lease.release()
                    continue
                self._start(job, lease)
                started = True
                break
            if not started:
                return

    def _start(self, job: dict[str, Any], lease: Lease) -> None:
        tenant, job_id = str(job["tenant"]), str(job["id"])
        deadline = time.monotonic() + self.timeout_s
        try:
            worker = self._spawn(job_id, tenant, self.store.state_dir)
        except Exception:  # noqa: BLE001 - reported on the job, never raised to the scheduler
            log.exception("could not start a job worker")
            jobs.finish(self.store, tenant=tenant, job_id=job_id, state="failed",
                        code="worker_unavailable", message=jobs.ERROR_MESSAGES["worker_unavailable"])
            lease.release()
            return
        if worker.pid:
            jobs.attach_worker(self.store, tenant=tenant, job_id=job_id, pid=int(worker.pid))
        self._children[job_id] = _Child(job_id, tenant, worker, lease, deadline)


# --- module-level lifecycle (app lifespan) -------------------------------------- #

_runner: JobRunner | None = None


def start_job_runner(store: CampaignStore) -> JobRunner | None:
    """Start the process's runner (unless `KALOS_JOBS_ENABLED=0`). Recovers
    orphaned jobs first. Returns None when disabled."""
    global _runner
    if not jobs.job_settings().enabled:
        log.info("background jobs disabled (KALOS_JOBS_ENABLED=0)")
        return None
    runner = JobRunner(store)
    runner.start()
    _runner = runner
    return runner


def stop_job_runner(runner: JobRunner | None) -> None:
    global _runner
    if runner is not None:
        runner.stop()
    if _runner is runner:
        _runner = None
