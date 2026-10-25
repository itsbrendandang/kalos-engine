"""The single analysis slot: one heavy fit at a time, honest 503 for the rest.

WHY. Found on the first real containerized run (2026-09-10): `_analyze` is
CPU-bound for tens of seconds, `run_in_threadpool` work cannot be cancelled,
and a client that gives up (a proxy timeout) orphans a thread that keeps
computing. With no admission control every retry contended with the ghosts of
its predecessors - measured as 598% engine CPU with zero connected clients.
The slot makes the failure mode impossible: a second analysis gets HTTP 503
with Retry-After, and the slot is released by the WORKER THREAD when the fit
actually finishes, so a disconnected client's orphan keeps holding it and the
503 stays truthful.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from kalos.portal import busy
from kalos.portal.busy import AnalysisBusy, run_exclusively


def test_slot_serializes_and_releases_in_the_worker():
    """The unit contract: second acquire while held raises; release happens
    when the wrapped callable finishes, not before."""
    started = threading.Event()
    release = threading.Event()

    def slow() -> str:
        started.set()
        release.wait(timeout=5)
        return "done"

    locked = run_exclusively(slow)
    t = threading.Thread(target=locked)
    t.start()
    assert started.wait(timeout=5)
    with pytest.raises(AnalysisBusy):
        run_exclusively(lambda: "never")
    release.set()
    t.join(timeout=5)
    # slot free again once the worker actually finished
    run_exclusively(lambda: None)()


def test_slot_survives_worker_exception():
    """A failing analysis must free the slot - a crash that leaks the slot
    would turn every later request into a permanent 503."""
    def boom() -> None:
        raise RuntimeError("fit exploded")

    with pytest.raises(RuntimeError):
        run_exclusively(boom)()
    run_exclusively(lambda: None)()  # acquires cleanly


def _tiny_sheet() -> bytes:
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame(
        {
            "Methanol": rng.uniform(0, 4, n).round(3),
            "pH": rng.uniform(5, 7, n).round(2),
            "lipase_titer": rng.uniform(1, 5, n).round(3),
        }
    )
    return df.to_csv(index=False).encode()


def test_api_run_returns_503_with_retry_after_while_slot_held(tmp_path, monkeypatch):
    """The endpoint contract, through the real route: while an analysis holds
    the slot, POST /api/run answers 503 + Retry-After + the JSON error
    envelope the frontend parses - it does not queue, block, or 500."""
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    from kalos.portal.app import app

    client = TestClient(app)
    started = threading.Event()
    release = threading.Event()

    def hold() -> None:
        started.set()
        release.wait(timeout=10)

    locked = run_exclusively(hold)
    t = threading.Thread(target=locked)
    t.start()
    try:
        assert started.wait(timeout=5)
        resp = client.post(
            "/api/run", files={"file": ("t.csv", _tiny_sheet(), "text/csv")}
        )
        assert resp.status_code == 503
        assert "Retry-After" in resp.headers
        assert int(resp.headers["Retry-After"]) == busy.RETRY_AFTER_SECONDS
        assert "already running" in resp.json()["error"]
    finally:
        release.set()
        t.join(timeout=10)

    # and once free, the same upload analyzes normally end to end
    resp = client.post("/api/run", files={"file": ("t.csv", _tiny_sheet(), "text/csv")})
    assert resp.status_code == 200
    assert "proposals" in resp.json()


# --- cancellation before the worker thread starts --------------------------- #
# A dispatch task can be cancelled before its first step (or while it waits
# for a thread). The thread-side release never runs then, so the dispatcher
# must release the slot itself. Each test below leaks the slot on the code
# before that fix: the next acquire would answer 503 for good.


def test_demo_route_cancelled_before_its_task_starts_releases_slot():
    import asyncio
    from types import SimpleNamespace

    from kalos.portal import app as appmod

    busy.reset_limiter(total=1, per_tenant=1)
    ran: list[int] = []

    async def scenario() -> None:
        outer = asyncio.ensure_future(
            appmod._run_demo(lambda: ran.append(1), SimpleNamespace(tenant="default"))
        )
        # One step: the handler takes the slot and schedules its dispatch task.
        await asyncio.sleep(0)
        dispatch = [t for t in asyncio.all_tasks() if t is not outer and t is not asyncio.current_task()]
        assert len(dispatch) == 1
        dispatch[0].cancel()  # cancelled before its first step: the thread never runs
        with pytest.raises(asyncio.CancelledError):
            await outer

    asyncio.run(scenario())
    assert ran == []
    assert busy.limiter().running() == 0
    run_exclusively(lambda: None)()  # the slot is free again
    busy.reset_limiter()


def test_run_in_slot_cancelled_while_waiting_for_a_thread_releases_slot(monkeypatch):
    import asyncio

    import anyio

    busy.reset_limiter(total=2, per_tenant=2)
    ran: list[int] = []
    release = threading.Event()

    async def scenario() -> None:
        # Occupy the only worker thread so the dispatch waits for one.
        limiter = anyio.to_thread.current_default_thread_limiter()
        monkeypatch.setattr(limiter, "total_tokens", 1)
        blocker = asyncio.ensure_future(anyio.to_thread.run_sync(release.wait))
        await asyncio.sleep(0.05)
        task = asyncio.ensure_future(
            busy.run_in_slot(lambda: ran.append(1), tenant="default")
        )
        await asyncio.sleep(0.05)
        assert busy.limiter().running() == 1  # slot taken, thread not yet
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await blocker

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert ran == []
    assert busy.limiter().running() == 0
    busy.reset_limiter()


def test_a_native_cancel_keeps_the_slot_until_the_thread_finishes():
    """N1: `asyncio.wait_for` (native cancellation) returns from `run_in_slot` while
    its thread is still computing. The thread owns the slot until it returns, so
    a second request must still be refused while the first one runs."""
    import asyncio

    busy.reset_limiter(total=1, per_tenant=1)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow() -> str:
        started.set()
        release.wait(10)
        finished.set()
        return "done"

    async def scenario() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(busy.run_in_slot(slow, tenant="a"), timeout=1.0)
        assert started.wait(5)
        assert not finished.is_set()
        assert busy.limiter().running() == 1  # the thread still holds the slot
        assert busy.limiter().try_acquire("b") is None  # a second request is refused
        release.set()
        for _ in range(500):
            if busy.limiter().running() == 0:
                break
            await asyncio.sleep(0.01)

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert finished.is_set()
    assert busy.limiter().running() == 0
    busy.reset_limiter()
