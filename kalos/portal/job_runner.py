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
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from kalos.portal import busy, jobs
from kalos.portal.busy import AnalysisLimiter, Lease, limiter
from kalos.portal.limits import upload_read_timeout_s

if TYPE_CHECKING:
    from kalos.portal.campaign import CampaignStore

log = logging.getLogger("kalos.portal")

KILL_WAIT_S = 5.0
# How long stop() waits for a scheduling pass that is under way. A pass is bounded (a
# few database statements, a spawn, at most KILL_WAIT_S per kill), so this is a backstop.
PASS_WAIT_S = 30.0
RUNNER_LOCK_NAME = "jobs.runner.lock"


class RunnerLocked(RuntimeError):
    """This engine cannot run the job scheduler: another process (or another
    runner) already schedules the jobs in this state directory, or the engine is
    configured to run several processes. Two schedulers on one `portal.db` would
    each admit and recover the same jobs."""


def _refuse_several_processes() -> None:
    raw_workers = os.environ.get("WEB_CONCURRENCY", "").strip()
    if raw_workers.isdigit() and int(raw_workers) > 1:
        raise RunnerLocked(
            "background jobs need a single engine process, and WEB_CONCURRENCY is "
            f"{raw_workers}; run one worker, or set KALOS_JOBS_ENABLED=0"
        )


def check_startup_config() -> None:
    """Refuse, before the engine serves a request, a configuration that cannot run:
    a malformed `KALOS_JOB*` value (ValueError naming the variable), always; and,
    when jobs are enabled, a platform or process layout that cannot take the runner
    lock (RunnerLocked). `python -m kalos.portal` calls it before uvicorn starts,
    and the app lifespan calls it too."""
    settings = jobs.job_settings()  # ValueError, naming the variable
    busy.validate_env()
    upload_read_timeout_s()  # ValueError, naming KALOS_UPLOAD_READ_TIMEOUT_S
    if not settings.enabled:
        return
    try:
        import fcntl  # noqa: F401 - availability only; the lock is taken at start
    except ImportError:
        raise RunnerLocked(
            "background jobs need a POSIX host: the runner lock uses flock, which this "
            "platform does not have; set KALOS_JOBS_ENABLED=0"
        ) from None
    _refuse_several_processes()


_expected = False  # set by the app lifespan: this process serves jobs, so it must run the scheduler


def expect_runner(flag: bool) -> None:
    global _expected
    _expected = flag


def runner_down() -> bool:
    """True when this process should run the job scheduler (its lifespan declared
    it) but no scheduler thread is alive. `/readyz` answers 503 then."""
    if not _expected:
        return False
    runner = _runner
    return runner is None or runner._thread is None or not runner._thread.is_alive()


def _acquire_runner_lock(state_dir: Path) -> int:
    """Take the state directory's runner lock (an exclusive, non-blocking flock)
    and return its file descriptor. The OS releases it when the process exits,
    however it exits, so a crashed engine never leaves it held.

    POSIX only (`fcntl`). The import is here, not at module level, so the module
    loads everywhere and only an engine that runs jobs needs the lock. Not
    reliable on network filesystems (NFS, SMB): see docs/JOBS.md."""
    try:
        import fcntl
    except ImportError:
        raise RunnerLocked(
            "background jobs need a POSIX host: the runner lock uses flock, which this "
            "platform does not have; set KALOS_JOBS_ENABLED=0"
        ) from None
    _refuse_several_processes()
    state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(state_dir / RUNNER_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise RunnerLocked(
            f"another engine process already runs the background jobs in {state_dir}; "
            "run one engine per state directory, or set KALOS_JOBS_ENABLED=0"
        ) from None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd
# A worker gets its job time limit plus this margin as a deadline (seconds). The
# runner normally kills it at the limit; the deadline is for the case where the
# runner is gone (the worker then exits by itself, see job_worker).
WORKER_DEADLINE_MARGIN_S = 60.0


class Worker(Protocol):
    @property
    def pid(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def kill(self) -> None: ...


Spawn = Callable[[str, str, Path], Worker]


class ProcessWorker:
    """A worker in its own session, so a kill takes its whole process group."""

    def __init__(
        self, job_id: str, state_dir: Path, *, argv: list[str] | None = None,
        deadline_s: float | None = None,
    ) -> None:
        env = {**os.environ, "KALOS_STATE_DIR": str(state_dir), jobs.ENGINE_PID_ENV: str(os.getpid()),
               jobs.WORKER_SESSION_ENV: "1"}
        if deadline_s is not None:  # seconds from now: the worker exits itself past this
            env["KALOS_JOB_DEADLINE_S"] = repr(float(deadline_s))
        # `argv` replaces the worker command (tests only: a fake worker process).
        command = argv if argv is not None else [sys.executable, "-m", "kalos.portal.job_worker", job_id]
        self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command, env=env, stdin=subprocess.DEVNULL, start_new_session=True,
        )
        self._pid = self._proc.pid

    @property
    def pid(self) -> int | None:
        return self._pid

    def poll(self) -> int | None:
        """The worker's exit status, or None while it runs. On Linux the exit is observed
        WITHOUT reaping (waitid WNOWAIT): an unreaped leader keeps its pid, and so its
        process-group id, reserved until `kill()` has signalled the group. (macOS has no
        waitid; there poll reaps, and the group kill after it is best-effort.)"""
        if self._proc.returncode is not None or not hasattr(os, "waitid"):
            return self._proc.poll()
        info = os.waitid(os.P_PID, self._pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if info is None:
            return None
        return int(info.si_status) if info.si_code == os.CLD_EXITED else -int(info.si_status)

    def kill(self) -> None:
        _kill_group(self._pid)  # before reaping: the leader's pid still names the group
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


def _command_line(pid: int) -> str | None:
    """The command line of the live process `pid`, or None when it cannot be read
    (it exited, or the platform gives no way to read it). On Linux it comes from
    /proc, which needs no helper program, so it works in slim images without `ps`.
    Where /proc does not exist (macOS), it comes from `ps`, if it is installed."""
    if os.path.isdir("/proc/self"):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return None
        return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip() or None
    if shutil.which("ps") is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["ps", "-o", "command=", "-p", str(pid)],  # noqa: S607 - fixed argv, executable resolved from PATH (ps/git)
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip() or None


def _is_job_worker(pid: int, job_id: str) -> bool:
    """Whether `pid` is alive and its command line names `job_id`. A job id is unique
    (`job_` + 32 hex), so a reused pid running something else never matches. When the
    command line cannot be read, the answer is no: an unverified pid is never signalled."""
    cmd = _command_line(pid)
    return cmd is not None and job_id in cmd and "job_worker" in cmd


def _kill_recorded_group(pid: int) -> None:
    """Kill the process group the job's worker leads. The worker is a session leader
    (`ProcessWorker`), so its group id is its pid; a pid that no longer leads a group
    is not signalled."""
    try:
        if os.getpgid(pid) != pid:
            return
    except OSError:
        return
    _kill_group(pid)


def recover(store: CampaignStore) -> int:
    """Startup recovery: fail the jobs an earlier engine process left running
    (killing their workers if they are still alive), sweep orphaned input
    files, and return how many jobs were failed."""
    failed = 0
    for row in jobs.running_jobs(store):
        pid = row.get("worker_pid")
        if isinstance(pid, int) and _is_job_worker(pid, str(row["id"])):
            _kill_recorded_group(pid)
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
    # `.tmp` is a half-written upload (a crash between write and rename): never live.
    for path in [*base.glob("*/*.bin"), *base.glob("*/*.tmp")]:
        rel = f"{jobs.JOBS_DIRNAME}/{path.parent.name}/{path.name}"
        if rel not in live:
            path.unlink(missing_ok=True)


class _Child:
    """A job the runner owns: its worker (None if the worker never started), the
    slot it holds, and the outcome to record once the worker is gone. The child
    stays registered, and keeps its slot, until that outcome is recorded."""

    __slots__ = ("deadline", "job_id", "lease", "outcome", "tenant", "worker")

    def __init__(self, job_id: str, tenant: str, worker: Worker | None, lease: Lease, deadline: float) -> None:
        self.job_id = job_id
        self.tenant = tenant
        self.worker = worker
        self.lease = lease
        self.deadline = deadline
        # (state, error code) decided by the runner (cancel, timeout, a failed start).
        # None means not decided yet: a worker that exits on its own is a crash.
        self.outcome: tuple[str, str | None] | None = None


class JobRunner:
    def __init__(
        self,
        store: CampaignStore,
        *,
        spawn: Spawn | None = None,
        limit: AnalysisLimiter | None = None,
        timeout_s: float | None = None,
        poll_s: float = 0.5,
        pass_wait_s: float = PASS_WAIT_S,
    ) -> None:
        self.store = store
        self.runner_id = uuid.uuid4().hex
        self._spawn = spawn or self._default_spawn
        self._limit = limit
        self._timeout_s = timeout_s
        self._poll_s = poll_s
        self._children: dict[str, _Child] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock_fd: int | None = None
        # One scheduling pass at a time. stop() takes it before it kills the children and
        # sets `_closed`, so a pass that was under way finishes first and no worker can be
        # started after the children were recorded.
        self._pass = threading.Lock()
        self._pass_wait_s = pass_wait_s
        self._closed = False

    # --- wiring -------------------------------------------------------------- #

    @property
    def limit(self) -> AnalysisLimiter:
        return self._limit if self._limit is not None else limiter()

    @property
    def timeout_s(self) -> float:
        return self._timeout_s if self._timeout_s is not None else jobs.job_settings().timeout_s

    def _default_spawn(self, job_id: str, _tenant: str, state_dir: Path) -> Worker:
        return ProcessWorker(job_id, state_dir,
                             deadline_s=self.timeout_s + WORKER_DEADLINE_MARGIN_S)

    def start(self) -> None:
        """Take the state directory's runner lock (RunnerLocked if another process
        holds it), recover orphaned jobs, and start the scheduler thread."""
        self._lock_fd = _acquire_runner_lock(self.store.state_dir)
        try:
            recover(self.store)
        except BaseException:
            self._release_lock()
            raise
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
            if self._thread.is_alive():
                log.error("the job scheduler is still in a pass; it starts no more workers")
        self.limit.remove_listener(self._wake.set)
        if not self._pass.acquire(timeout=self._pass_wait_s):
            # A pass is still running and may start a worker. Keep the lock: the process
            # releases it when it exits, and no second runner may recover these jobs.
            log.error("background jobs could not be stopped cleanly; the runner lock is kept")
            return
        try:
            self._closed = True
            for child in list(self._children.values()):
                try:
                    if child.worker is not None:
                        child.worker.kill()
                    self._finish(child, "failed", "engine_stopped")
                except Exception:  # noqa: BLE001 - shutting down: recovery fails it at the next start
                    log.exception("could not record that background job %s was stopped", child.job_id)
                self._drop(child)
        finally:
            self._pass.release()
        # Last: until every running job is recorded, another runner must not recover them.
        self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)  # closing the descriptor releases the flock
            self._lock_fd = None

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
        with self._pass:
            if self._closed:
                return
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

    def _reconcile(self, child: _Child, now: float) -> None:
        """Bring one child to its recorded outcome. Any failure (the database,
        say) leaves it registered with its slot, and the next pass retries."""
        try:
            self._supervise_one(child, now)
        except Exception:  # noqa: BLE001 - one child must not stall the others
            log.exception("could not record the outcome of background job %s; retrying", child.job_id)

    def _supervise(self) -> None:
        now = time.monotonic()
        for child in list(self._children.values()):
            self._reconcile(child, now)

    def _supervise_one(self, child: _Child, now: float) -> None:
        worker = child.worker
        if worker is not None and worker.poll() is None:
            # Still running: decide whether it must be stopped, or leave it be.
            if child.outcome is None:
                if jobs.cancel_requested(self.store, tenant=child.tenant, job_id=child.job_id):
                    child.outcome = ("cancelled", None)
                elif now > child.deadline:
                    log.warning("background job exceeded its time limit; stopping it")
                    child.outcome = ("timed_out", "timeout")
                else:
                    return
        elif child.outcome is None:
            # Exited. A clean exit has already reached a terminal state (the worker
            # wrote it); anything still active is a crash.
            child.outcome = ("failed", "worker_exited")
        # Every exit path ends here: stop the worker's whole process group, which
        # also catches children it left behind (a no-op if it is already gone),
        # then record the outcome. The first terminal writer wins (jobs.finish).
        if worker is not None:
            worker.kill()
        outcome = child.outcome
        assert outcome is not None  # every path above decides one
        state, code = outcome
        self._finish(child, state, code)
        self._drop(child)

    def _admit(self) -> None:
        self.limit.set_waiting(set(jobs.queued_tenants(self.store)))
        while not self._stop.is_set():
            started = False
            for tenant in jobs.queued_tenants(self.store):
                lease = self.limit.try_acquire(tenant, yield_to_queue=False)
                if lease is None:
                    continue
                try:
                    job = jobs.claim(self.store, tenant=tenant, runner_id=self.runner_id)
                except Exception:  # noqa: BLE001 - the claim is one transaction: the job stays queued
                    lease.release()
                    log.exception("could not claim a queued background job; retrying next pass")
                    return
                if job is None:
                    lease.release()
                    continue
                self._start(job, lease)
                started = True
                break
            if not started:
                return

    def _start(self, job: dict[str, Any], lease: Lease) -> None:
        """Start the worker for a claimed job. The child is registered before ANY
        fallible step: from here on the scheduler owns the job and the slot whatever
        fails. A failed start, a failed pid write, or any BaseException kills the
        worker (if one exists) and records the job as `worker_unavailable`; the slot
        is released once that outcome is recorded (`_reconcile`)."""
        tenant, job_id = str(job["tenant"]), str(job["id"])
        child = _Child(job_id, tenant, None, lease, deadline=0.0)  # nothing here can raise
        self._children[job_id] = child
        try:
            child.deadline = time.monotonic() + self.timeout_s
            child.worker = self._spawn(job_id, tenant, self.store.state_dir)
            if child.worker.pid:
                jobs.attach_worker(self.store, tenant=tenant, job_id=job_id, pid=int(child.worker.pid))
        except BaseException:  # noqa: BLE001 - reported on the job; a worker must never outlive us
            log.exception("could not start a job worker, or record its pid; stopping it")
            child.outcome = ("failed", "worker_unavailable")
            if child.worker is not None:
                try:
                    child.worker.kill()
                except Exception:  # noqa: BLE001 - the reconcile below kills it again
                    log.exception("could not stop a job worker that failed to start")
        self._reconcile(child, time.monotonic())


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
