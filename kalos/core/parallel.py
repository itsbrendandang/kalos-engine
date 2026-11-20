"""Process-parallel execution for the analysis hot path, with results identical to serial.

Three parts of an analysis are independent pieces of CPU work that used to run one
after the other: the grouped-CV fold fits, the extra starting points of the
proposal fit, and the per-level L-BFGS solves of the mixed (categorical)
acquisition. `run_ordered` runs such a list of tasks in a pool of worker processes
and returns exactly what running them in order in this process would return.

How "exactly" is kept:

- Worker PROCESSES, never threads. gpytorch/linear_operator settings and
  `warnings.catch_warnings` are process-global, and a thread pool measurably
  changed the batch (docs/PERFORMANCE.md).
- One torch thread in every worker, and the pool is used only when this process
  also runs at one torch thread (`KALOS_TORCH_THREADS=1`, the default), so a
  task's float arithmetic is the same wherever it runs.
- The global RNGs. The tasks are RNG-free in practice, but a library fallback can
  draw from the global torch/numpy RNG (botorch's fit retries). Serially, task i
  starts from the RNG state task i-1 left. Every task here is started from the
  state at the moment the serial loop would have started task 0 and reports
  whether it moved the state. Results are taken in order up to and including the
  first task that moved it; this process then adopts that task's end state and
  computes the remaining tasks itself, serially. If the state already moved
  between `submit` and `collect`, every task is recomputed here.
- Errors. A task that raises (in the worker, or while pickling) is recomputed
  here, serially, from the right RNG state, so the caller sees the exception the
  serial code would have raised at that point.
- Warnings and log records a task emits in a worker are re-emitted here, in task
  order. The worker applies this process's warning filters.

Configuration: `KALOS_ANALYSIS_WORKERS`
  - unset, empty or `auto`: `cpus // KALOS_JOB_WORKERS` (the number of analyses the
    engine runs at once, default 2), at most `MAX_AUTO_WORKERS` (5: the 10 CV fits
    then take 2 rounds, and the mixed acquisition rarely has more than 5 levels;
    more workers cost ~300 MB each for little gain). On a 4-vCPU host that is 2
    workers per analysis, so two concurrent analyses use the 4 CPUs.
  - `0` or `1`: serial (the previous behavior, same code path as before).
  - `N > 1`: N worker processes.
Serial is also used, whatever the setting, inside a multiprocessing child (a pool
worker, or a benchmark harness's own process pool: no nested pools), when this
process runs more than one torch thread, and when the pool cannot be started or
breaks. Every fallback gives the same result.

The pool is created on first use (spawn context, so it is safe from any thread)
and kept for the life of the process. Workers exit when their parent does. As
with any spawn-based pool, a script that runs analyses must keep its top-level
code under `if __name__ == "__main__":` (a spawned worker imports the main
script; every script in this repo does).

Torch-free at import time.
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import pickle
import random
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("kalos.core.parallel")

ENV_WORKERS = "KALOS_ANALYSIS_WORKERS"
# Set in every pool worker so an analysis started inside one never starts a pool.
_IN_WORKER_ENV = "KALOS_ANALYSIS_POOL_WORKER"
MAX_AUTO_WORKERS = 5
_DEFAULT_JOB_WORKERS = 2  # kalos.portal.busy.DEFAULT_WORKERS (not imported: torch-free core)


def _cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:  # pragma: no cover - platform dependent
            pass
    return max(1, os.cpu_count() or 1)


def _job_workers() -> int:
    raw = os.environ.get("KALOS_JOB_WORKERS", "").strip()
    try:
        return max(1, int(raw)) if raw else _DEFAULT_JOB_WORKERS
    except ValueError:
        return _DEFAULT_JOB_WORKERS


def configured_workers() -> int:
    """Worker processes per analysis from `KALOS_ANALYSIS_WORKERS` (see the module
    docstring). A malformed value means serial (1), logged once."""
    raw = os.environ.get(ENV_WORKERS, "").strip().lower()
    if raw in ("", "auto"):
        return max(1, min(MAX_AUTO_WORKERS, _cpu_count() // _job_workers()))
    try:
        n = int(raw)
    except ValueError:
        _warn_once(f"{ENV_WORKERS}={raw!r} is not an integer or 'auto'; running serially")
        return 1
    return max(1, n)


def validate_env() -> None:
    """Raise ValueError for a malformed `KALOS_ANALYSIS_WORKERS` (engine startup)."""
    raw = os.environ.get(ENV_WORKERS, "").strip().lower()
    if raw in ("", "auto"):
        return
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_WORKERS} must be an integer >= 0 or 'auto'") from None
    if n < 0:
        raise ValueError(f"{ENV_WORKERS} must be an integer >= 0 or 'auto'")


_warned: set[str] = set()

# Counters for tests and benchmarks: tasks whose result came from a pool worker,
# tasks computed in this process, and the reasons for falling back.
stats: dict[str, int] = {"pool_tasks": 0, "local_tasks": 0, "rng_redo": 0, "error_redo": 0,
                         "broken_pool": 0}


# Pool tasks per task function name (which parts of the analysis used the pool).
pool_tasks_by_fn: dict[str, int] = {}


def reset_stats() -> None:
    for k in stats:
        stats[k] = 0
    pool_tasks_by_fn.clear()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log.warning(msg)


def serial_reason() -> str | None:
    """Why the next `run_ordered` would run serially, or None if it would use the pool."""
    if os.environ.get(_IN_WORKER_ENV) == "1":
        return "inside an analysis pool worker"
    if configured_workers() <= 1:
        return f"{ENV_WORKERS} is 1"
    if mp.parent_process() is not None or getattr(mp.current_process(), "_inheriting", False):
        # A multiprocessing child (a benchmark harness's own process pool, a
        # daemonic worker, or a spawned child still importing the parent's main
        # script): its parent already spreads work over the CPUs.
        return "inside a multiprocessing child process"
    import sys

    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() != 1:
        return (f"this process runs {torch.get_num_threads()} torch threads; the pool is "
                "used only at 1 (KALOS_TORCH_THREADS=1) so results match the serial path")
    return None


def active_workers() -> int:
    """How many worker processes the next `run_ordered` would use (1 = serial)."""
    return 1 if serial_reason() is not None else configured_workers()


# --- the pool ------------------------------------------------------------------ #

_pool: ProcessPoolExecutor | None = None
_pool_size = 0
_pool_lock = threading.Lock()
# Idle pools are stopped, so a long-running engine does not keep N interpreters
# with torch loaded (~300 MB each) between analyses. The next analysis pays the
# start-up again (about 2-4 s, overlapping its first tasks).
POOL_IDLE_S = 300.0
_inflight = 0
_last_used = 0.0


def _watch_parent(parent_pid: int) -> None:
    while True:
        time.sleep(1.0)
        if os.getppid() != parent_pid:
            os._exit(0)


def _worker_init(parent_pid: int) -> None:
    os.environ[_IN_WORKER_ENV] = "1"
    os.environ[ENV_WORKERS] = "1"
    os.environ["KALOS_TORCH_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    # A worker left behind by a parent that died (kill -9, crash) exits on its own.
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True,
                     name="kalos-pool-parent-watch").start()


def _reap_idle(pool: ProcessPoolExecutor) -> None:
    """Stop `pool` once it has had no task for `POOL_IDLE_S` seconds."""
    global _pool
    while True:
        time.sleep(min(30.0, POOL_IDLE_S))
        with _pool_lock:
            if _pool is not pool:
                return  # replaced or stopped already
            if _inflight == 0 and time.monotonic() - _last_used > POOL_IDLE_S:
                _pool = None
                break
    pool.shutdown(wait=False, cancel_futures=True)


def _mark_used(delta: int) -> None:
    global _inflight, _last_used
    with _pool_lock:
        _inflight += delta
        _last_used = time.monotonic()


def _get_pool(n: int) -> ProcessPoolExecutor | None:
    global _pool, _pool_size
    with _pool_lock:
        if _pool is not None and _pool_size == n:
            return _pool
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
        try:
            _pool = ProcessPoolExecutor(
                max_workers=n, mp_context=mp.get_context("spawn"),
                initializer=_worker_init, initargs=(os.getpid(),),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            _warn_once(f"could not start the analysis worker pool ({exc!r}); running serially")
            return None
        _pool_size = n
        threading.Thread(target=_reap_idle, args=(_pool,), daemon=True,
                         name="kalos-pool-idle-reaper").start()
        return _pool


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


def shutdown() -> None:
    """Stop the pool (tests; also registered at exit)."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


atexit.register(shutdown)


# --- RNG state ------------------------------------------------------------------ #

@dataclass(frozen=True)
class RngState:
    torch: bytes | None
    numpy: Any  # np.random.get_state(): the legacy tuple
    python: tuple

    def same_as(self, other: RngState) -> bool:
        return (self.torch == other.torch and self.python == other.python
                and _np_state_equal(self.numpy, other.numpy))


def _np_state_equal(a: Any, b: Any) -> bool:
    import numpy as np

    return len(a) == len(b) and all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(a, b)
    )


def capture_rng() -> RngState:
    import sys

    import numpy as np

    torch = sys.modules.get("torch")
    t = bytes(torch.get_rng_state().numpy().tobytes()) if torch is not None else None
    return RngState(t, np.random.get_state(), random.getstate())


def restore_rng(state: RngState) -> None:
    import numpy as np

    if state.torch is not None:
        import torch

        torch.set_rng_state(torch.frombuffer(bytearray(state.torch), dtype=torch.uint8))
    np.random.set_state(state.numpy)
    random.setstate(state.python)


# --- running tasks -------------------------------------------------------------- #

class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        record.msg = record.getMessage()
        record.args = None
        record.exc_info = None
        self.records.append(record)


def _run_task(payload: bytes) -> tuple:
    """Worker side: run one task from the parent's RNG state, under the parent's
    warning filters; return (result, warnings, log records, end RNG state, moved)."""
    # The payload was pickled by the parent process for this pool, never by anyone else.
    fn, args, start, filters = pickle.loads(payload)  # noqa: S301
    restore_rng(start)
    handler = _Records()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            _apply_filters(filters)
            result = fn(*args)
    finally:
        root.removeHandler(handler)
    end = capture_rng()
    ws = [(str(w.message), w.category, w.filename, w.lineno) for w in caught]
    return result, ws, handler.records, end, not end.same_as(start)


def _apply_filters(filters: list) -> None:
    """Install the parent's warning filters (same order) through the public API."""
    warnings.resetwarnings()
    for action, message, category, module, lineno in reversed(filters):
        warnings.filterwarnings(
            action,
            message=_pattern(message),
            category=category,
            module=_pattern(module),
            lineno=lineno,
        )


def _pattern(p: Any) -> str:
    """A filter's message/module: a compiled regex, a plain string or None."""
    if p is None:
        return ""
    return str(getattr(p, "pattern", p))


def _replay(ws: list, records: list) -> None:
    for message, category, filename, lineno in ws:
        warnings.warn_explicit(message, category, filename, lineno)
    for rec in records:
        logging.getLogger(rec.name).handle(rec)


class Pending:
    """Tasks submitted to the pool; `collect()` returns the serial-order results."""

    def __init__(self, fn: Callable[..., Any], tasks: list[tuple], start: RngState | None,
                 pool: ProcessPoolExecutor | None, futures: list[Future] | None) -> None:
        self._fn = fn
        self._tasks = tasks
        self._start = start
        self._pool = pool
        self._futures = futures
        self.parallel = futures is not None

    def collect(self) -> list[Any]:
        """Call where the serial code would have started task 0."""
        if self._futures is None:
            return self._collect()
        try:
            return self._collect()
        finally:
            _mark_used(-1)

    def _collect(self) -> list[Any]:
        fn, tasks = self._fn, self._tasks
        futures = self._futures
        if futures is None or self._start is None:
            return _local(fn, tasks)
        out: list[Any] = []
        if not capture_rng().same_as(self._start):
            # Something between submit and collect drew from the global RNG; the
            # worker results started from the wrong state.
            _cancel(futures)
            stats["rng_redo"] += 1
            return _local(fn, tasks)
        for i, fut in enumerate(futures):
            try:
                result, ws, records, end, moved = fut.result()
            except Exception as exc:  # a task error, a pickling error or a broken pool
                if _is_broken(exc) and self._pool is not None:
                    _warn_once(f"analysis worker pool failed ({exc!r}); running serially")
                    stats["broken_pool"] += 1
                    _discard_pool(self._pool)
                else:
                    stats["error_redo"] += 1
                _cancel(futures[i + 1:])
                return out + _local(fn, tasks[i:])
            _replay(ws, records)
            out.append(result)
            stats["pool_tasks"] += 1
            name = getattr(fn, "__name__", "?")
            pool_tasks_by_fn[name] = pool_tasks_by_fn.get(name, 0) + 1
            if moved:
                # Task i drew from the global RNG: the next serial task starts from
                # where it left the state, which no worker did.
                restore_rng(end)
                _cancel(futures[i + 1:])
                stats["rng_redo"] += 1
                return out + _local(fn, tasks[i + 1:])
        return out


def _local(fn: Callable[..., Any], tasks: Sequence[tuple]) -> list[Any]:
    stats["local_tasks"] += len(tasks)
    return [fn(*a) for a in tasks]


def _is_broken(exc: BaseException) -> bool:
    from concurrent.futures.process import BrokenProcessPool

    return isinstance(exc, BrokenProcessPool)


def _cancel(futures: Sequence[Future]) -> None:
    for f in futures:
        f.cancel()


def submit(fn: Callable[..., Any], tasks: Sequence[tuple], *, min_tasks: int = 2) -> Pending:
    """Start `fn(*args)` for every args tuple in `tasks` in the pool (or nothing, if
    the pool is not used) and return a `Pending`. `fn` must be a module-level
    function. With fewer than `min_tasks` tasks the pool is not used."""
    task_list = [tuple(a) for a in tasks]
    n = active_workers()
    if n <= 1 or len(task_list) < min_tasks:
        return Pending(fn, task_list, None, None, None)
    start = capture_rng()
    filters = list(warnings.filters)
    try:
        payloads = [pickle.dumps((fn, a, start, filters), protocol=pickle.HIGHEST_PROTOCOL)
                    for a in task_list]
    except Exception as exc:  # not picklable (a lambda in the task): serial
        log.debug("analysis task not picklable (%r); running serially", exc)
        return Pending(fn, task_list, None, None, None)
    pool = _get_pool(n)
    if pool is None:
        return Pending(fn, task_list, None, None, None)
    _mark_used(+1)
    try:
        futures = [pool.submit(_run_task, p) for p in payloads]
    except Exception as exc:  # broken or shut-down pool
        _mark_used(-1)
        _warn_once(f"analysis worker pool unavailable ({exc!r}); running serially")
        _discard_pool(pool)
        return Pending(fn, task_list, None, None, None)
    return Pending(fn, task_list, start, pool, futures)


def run_ordered(fn: Callable[..., Any], tasks: Sequence[tuple], *, min_tasks: int = 2) -> list[Any]:
    """`[fn(*a) for a in tasks]`, computed in the worker pool when it is in use, with
    the identical result (see the module docstring)."""
    return submit(fn, tasks, min_tasks=min_tasks).collect()


__all__ = [
    "ENV_WORKERS",
    "MAX_AUTO_WORKERS",
    "Pending",
    "RngState",
    "active_workers",
    "capture_rng",
    "configured_workers",
    "pool_tasks_by_fn",
    "reset_stats",
    "restore_rng",
    "run_ordered",
    "serial_reason",
    "shutdown",
    "stats",
    "submit",
    "validate_env",
]
