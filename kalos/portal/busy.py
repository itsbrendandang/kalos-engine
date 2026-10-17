"""Heavy-work admission: a global cap plus a per-tenant cap, shared by every
analysis path (sync routes, background jobs, demo routes).

WHY THIS EXISTS, from the first real containerized run (2026-09-10): a full
`_analyze` is CPU-bound for tens of seconds, `run_in_threadpool` cannot be
cancelled, and a client that gives up (a proxy timeout, a closed tab) leaves
an ORPHAN thread computing to nowhere. With no admission control, each retry
then contended with the ghosts of its predecessors - measured as 598% engine
CPU with zero connected clients, and every request slower than the proxy
timeout that spawned the previous orphan. A one-user thundering herd.

The first gate was ONE global slot. That fixed the herd but let one tenant
starve every other one: whoever held the slot (or re-grabbed it the moment it
freed) answered every other tenant's upload with 503 (review finding V1). The
gate is now a small counting limiter (docs/JOBS.md, "Admission"):

- at most `KALOS_JOB_WORKERS` analyses run at once in this process (default
  2), whether they are sync requests in a thread or background-job worker
  processes;
- at most `KALOS_JOB_PER_TENANT` of them belong to one tenant (default 1), so
  with the defaults one tenant can never hold all the capacity;
- a sync request YIELDS to the job queue: it is admitted only if, after it,
  there is still a free slot for every other tenant that has a runnable job
  waiting. Background jobs are the fair path; sync routes are best effort.

A request that is not admitted gets an honest HTTP 503 with Retry-After,
never a silent queue. The slot is released BY THE WORKER THREAD when the
analysis actually finishes - not when the request's await unwinds - so a
disconnected client's orphan keeps holding it and later requests are told
"busy" instead of being invited to pile on.
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Any, Callable

_ERR_BUSY = (
    "An analysis is already running. This engine runs a limited number of analyses "
    "at a time; retry after the current one finishes, or submit a background job "
    "(POST /api/campaigns/{id}/jobs)."
)
# Advisory client hint, roughly one containerized cold analyze (measured 47s
# on the 160-run fixture on the reference deployment).
RETRY_AFTER_SECONDS = 60

DEFAULT_WORKERS = 2
DEFAULT_PER_TENANT = 1
# The tenant a caller that does not name one is charged to. Only tests and the
# old single-slot call sites use it; every route passes the principal's tenant.
UNSCOPED_TENANT = "default"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


class AnalysisBusy(Exception):
    """No analysis slot is available for this caller right now."""

    def __init__(self) -> None:
        super().__init__(_ERR_BUSY)


class Lease:
    """One held slot. `release()` is idempotent."""

    __slots__ = ("_limiter", "tenant", "_released", "_lock")

    def __init__(self, limiter: AnalysisLimiter, tenant: str) -> None:
        self._limiter = limiter
        self.tenant = tenant
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._limiter._release(self.tenant)


class AnalysisLimiter:
    """Global + per-tenant counting limiter. Thread-safe; in-process only (the
    engine is one uvicorn worker, docs/JOBS.md)."""

    def __init__(self, total: int, per_tenant: int) -> None:
        if total < 1 or per_tenant < 1:
            raise ValueError("limiter capacities must be >= 1")
        self.total = total
        self.per_tenant = per_tenant
        self._lock = threading.Lock()
        self._running: Counter[str] = Counter()
        self._n = 0
        self._waiting: frozenset[str] = frozenset()
        self._listeners: list[Callable[[], None]] = []

    def try_acquire(self, tenant: str, *, yield_to_queue: bool = True) -> Lease | None:
        """A lease, or None when the global or the tenant cap is reached.

        `yield_to_queue` (sync callers): also refuse when taking a slot would
        leave fewer free slots than there are OTHER tenants with a runnable
        job waiting, so a stream of sync requests cannot starve the queue.
        The job runner itself passes False: it is the queue."""
        with self._lock:
            if self._n >= self.total or self._running[tenant] >= self.per_tenant:
                return None
            if yield_to_queue:
                others = len(self._waiting - {tenant})
                if self.total - self._n <= others:
                    return None
            self._running[tenant] += 1
            self._n += 1
            return Lease(self, tenant)

    def _release(self, tenant: str) -> None:
        with self._lock:
            if self._running[tenant] > 0:
                self._running[tenant] -= 1
                self._n -= 1
            if self._running[tenant] == 0:
                del self._running[tenant]
            listeners = list(self._listeners)
        for fn in listeners:  # wake the job scheduler: a slot just freed
            try:
                fn()
            except Exception:  # noqa: BLE001 - a listener must never break a release
                pass

    def set_waiting(self, tenants: frozenset[str] | set[str]) -> None:
        """Tenants that currently have a runnable queued job (the job runner
        refreshes this every scheduling pass)."""
        with self._lock:
            self._waiting = frozenset(tenants)

    def add_listener(self, fn: Callable[[], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[], None]) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def running(self, tenant: str | None = None) -> int:
        with self._lock:
            return self._n if tenant is None else self._running[tenant]

    def free(self) -> int:
        with self._lock:
            return self.total - self._n


_limiter: AnalysisLimiter | None = None
_limiter_lock = threading.Lock()


def limiter() -> AnalysisLimiter:
    """The process-wide limiter, sized from `KALOS_JOB_WORKERS` and
    `KALOS_JOB_PER_TENANT` on first use."""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = AnalysisLimiter(
                _env_int("KALOS_JOB_WORKERS", DEFAULT_WORKERS),
                _env_int("KALOS_JOB_PER_TENANT", DEFAULT_PER_TENANT),
            )
        return _limiter


def reset_limiter(total: int | None = None, per_tenant: int | None = None) -> AnalysisLimiter:
    """Replace the process-wide limiter (tests, or a config reload). Leases on
    the old limiter release into the old one and do not affect the new one."""
    global _limiter
    with _limiter_lock:
        _limiter = AnalysisLimiter(
            total if total is not None else _env_int("KALOS_JOB_WORKERS", DEFAULT_WORKERS),
            per_tenant if per_tenant is not None
            else _env_int("KALOS_JOB_PER_TENANT", DEFAULT_PER_TENANT),
        )
        return _limiter


def run_exclusively(fn: Callable[[], Any], *, tenant: str = UNSCOPED_TENANT) -> Callable[[], Any]:
    """Wrap `fn` for threadpool dispatch under one of `tenant`'s analysis slots.

    Acquires non-blocking HERE (in the event loop, before any thread is
    spawned) so a busy engine answers instantly; raises `AnalysisBusy` when no
    slot is available for this tenant. The returned callable releases the
    slot in ITS OWN finally - inside the worker thread - so cancellation of
    the awaiting request cannot free the slot while the computation is still
    running.
    """
    lease = limiter().try_acquire(tenant)
    if lease is None:
        raise AnalysisBusy()

    def _locked() -> Any:
        try:
            return fn()
        finally:
            lease.release()

    return _locked
