"""Background jobs (docs/JOBS.md): the queue, the worker and the runner, end to end
at the store level. The analysis is a stub (`_stub_analyze`) and the worker runs
in a thread (`ThreadWorker`), so these are fast and deterministic; the real
subprocess path is covered by `test_real_worker_process_publishes`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.portal import busy, jobs, job_runner, job_worker
from kalos.portal.campaign import CampaignConflict, CampaignStore
from kalos.portal.job_runner import JobRunner, ThreadWorker
from kalos.portal.uploads import MAX_UPLOAD_BYTES


def _df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })


def _csv(df: pd.DataFrame | None = None) -> bytes:
    return (df if df is not None else _df()).to_csv(index=False).encode()


def _stub_analyze(df, target, *, profile=None, anonymize=False, progress=None, roles=None, **_):
    """Stands in for the GP analysis: reports the stage boundaries, returns a
    result with the keys the publish path reads."""
    if progress is not None:
        progress("fitting", 0.15)
        progress("calibrating", 0.55)
        progress("proposing", 0.8)
    return {"target": target, "proposal_features": ["Methanol", "pH"],
            "proposals": [], "marker": "stub-result"}


@pytest.fixture(autouse=True)
def _fresh_limiter():
    busy.reset_limiter(total=2, per_tenant=1)
    yield
    busy.reset_limiter()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("KALOS_JOB_QUEUE_DEPTH", raising=False)
    monkeypatch.delenv("KALOS_JOB_MAX_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("KALOS_JOB_TIMEOUT_S", raising=False)
    monkeypatch.delenv("KALOS_JOBS_ENABLED", raising=False)
    s = CampaignStore(tmp_path)
    yield s
    s.close()


def _campaign(store: CampaignStore, tenant: str = "acme", name: str = "c") -> str:
    return store.create_campaign(tenant=tenant, name=name)["id"]


def _submit(store, tenant="acme", cid=None, raw=None) -> dict:
    cid = cid or _campaign(store, tenant)
    return jobs.enqueue(
        store, tenant=tenant, campaign_id=cid, raw=raw if raw is not None else _csv(),
        filename="runs.csv", target="lipase_titer", anonymize=False, roles="", sheet=None,
        requested_by="tester",
    )


def _settle(runner: JobRunner, timeout: float = 10.0) -> None:
    """Tick until every worker the runner started has been reaped."""
    deadline = time.monotonic() + timeout
    while runner.running_count() and time.monotonic() < deadline:
        runner.tick()
        time.sleep(0.01)
    assert runner.running_count() == 0


def _runner(store, spawn_analyze=_stub_analyze, *, timeout_s=60.0) -> JobRunner:
    """A runner whose workers run the real worker code in threads, with `analyze`."""

    def spawn(job_id, _tenant, _state_dir):
        return ThreadWorker(lambda should_stop: job_worker.run_job(
            store, job_id, analyze=spawn_analyze, should_stop=should_stop))

    return JobRunner(store, spawn=spawn, timeout_s=timeout_s, poll_s=0.01)


def _only_job_id(store, tenant: str) -> str:
    with store._txn(write=False) as conn:
        return str(conn.execute("SELECT id FROM jobs WHERE tenant = ?", (tenant,)).fetchone()[0])


def _drive(runner: JobRunner, store, tenant: str, job_id: str, *, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runner.tick()
        job = jobs.get_job(store, tenant=tenant, job_id=job_id)
        assert job is not None
        if job["state"] in jobs.TERMINAL_STATES:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not finish: {job}")


# --- settings ------------------------------------------------------------------ #


def test_settings_have_safe_defaults_and_reject_bad_values(monkeypatch):
    for name in ("KALOS_JOBS_ENABLED", "KALOS_JOB_QUEUE_DEPTH", "KALOS_JOB_MAX_UPLOAD_BYTES",
                 "KALOS_JOB_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    s = jobs.job_settings()
    assert s.enabled and s.queue_depth == jobs.DEFAULT_QUEUE_DEPTH
    assert s.max_upload_bytes == MAX_UPLOAD_BYTES and s.timeout_s == jobs.DEFAULT_TIMEOUT_S
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "0")
    with pytest.raises(ValueError, match="KALOS_JOB_QUEUE_DEPTH"):
        jobs.job_settings()
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "3")
    monkeypatch.setenv("KALOS_JOB_MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES + 1))
    with pytest.raises(ValueError, match="KALOS_JOB_MAX_UPLOAD_BYTES"):
        jobs.job_settings()
    monkeypatch.setenv("KALOS_JOB_MAX_UPLOAD_BYTES", "abc")
    with pytest.raises(ValueError, match="must be a number"):
        jobs.job_settings()


@pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
def test_kalos_jobs_enabled_zero_disables_the_queue(monkeypatch, raw):
    monkeypatch.setenv("KALOS_JOBS_ENABLED", raw)
    assert jobs.job_settings().enabled is False


# --- the happy path ------------------------------------------------------------- #


def test_queued_job_is_run_by_the_worker_and_publishes_atomically(store):
    cid = _campaign(store)
    job = _submit(store, cid=cid)
    assert job["state"] == "queued" and job["campaign_generation"] is None
    assert job["error"] is None and job["cancel_requested"] is False

    done = _drive(_runner(store), store, "acme", job["id"])

    assert done["state"] == "succeeded"
    assert done["progress"] == 1 and done["stage"] == "succeeded"
    assert done["attempts"] == 1
    published = store.load_analysis(tenant="acme", campaign_id=cid)
    assert published["marker"] == "stub-result"
    # the job says which generation it published, and the analysis carries that stamp
    assert done["campaign_generation"] == published["campaign_generation"]
    assert store.get(tenant="acme", campaign_id=cid)["generation"] == done["campaign_generation"]
    # the uploaded file is deleted once the job is finished
    assert not list(store.state_dir.glob("jobs/*/*.bin"))
    assert done["input_bytes"] == len(_csv())


def test_a_refusal_is_a_coded_failure_not_a_publish(store):
    def refuse(df, target, **kw):
        from kalos.portal.uploads import UploadRejected

        raise UploadRejected("the sheet has too few rows to analyze", code="too_few_rows")

    job = _submit(store)
    done = _drive(_runner(store, refuse), store, "acme", job["id"])
    assert done["state"] == "failed"
    assert done["error"] == {"code": "too_few_rows",
                             "message": "the sheet has too few rows to analyze"}
    assert store.load_analysis(tenant="acme") is None


def test_error_messages_are_fixed_text_not_exception_text(store):
    class Secret(Exception):
        pass

    def boom(*_a, **_k):
        raise Secret("column 'PROJECT-ZEUS-SECRET' has bad values")

    job = _submit(store)
    done = _drive(_runner(store, boom), store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "analysis_failed"
    assert "ZEUS" not in json.dumps(done)


def test_worker_logs_type_code_and_location_not_customer_text(store, caplog):
    """LOW10: an unexpected analysis failure is logged as type, code and the
    innermost file:line, never the exception message or the traceback's
    source lines (a KeyError or ValueError quotes customer column names)."""

    def boom(*_a, **_k):
        raise KeyError("column 'PROJECT-ZEUS-SECRET' has bad values")

    job = _submit(store)
    with caplog.at_level("DEBUG"):
        done = _drive(_runner(store, boom), store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "analysis_failed"
    failures = [r for r in caplog.records if "background job failed during analysis" in r.getMessage()]
    assert len(failures) == 1
    rec = failures[0]
    assert rec.exc_info is None  # no traceback attached
    assert "code=analysis_failed" in rec.getMessage()
    assert "type=KeyError" in rec.getMessage()
    assert re.search(r"at test_jobs\.py:\d+ in boom$", rec.getMessage())  # innermost frame
    assert "ZEUS" not in caplog.text  # neither the message nor a traceback line


# --- queue depth, upload size ---------------------------------------------------- #


def test_queue_depth_is_per_tenant_and_frees_on_cancel(store, monkeypatch):
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "2")
    # One campaign per upload: a second pending upload to one campaign is refused (UploadPending).
    a = _submit(store, cid=_campaign(store, name="one"))
    _submit(store, cid=_campaign(store, name="two"))
    with pytest.raises(jobs.QueueFull):
        _submit(store, cid=_campaign(store, name="three"))
    # another tenant is not affected by acme's full queue
    _submit(store, tenant="globex", cid=_campaign(store, "globex"))
    jobs.request_cancel(store, tenant="acme", job_id=a["id"])
    _submit(store, cid=_campaign(store, name="four"))  # room again
    # refused submissions leave no file behind
    assert len(list(store.state_dir.glob("jobs/*/*.bin"))) == 3


def test_the_engine_wide_queue_cap_applies_across_tenants(store, monkeypatch):
    """MED3: a global cap on queued + running jobs, whichever tenants hold them."""
    monkeypatch.setenv("KALOS_JOB_GLOBAL_QUEUE_DEPTH", "2")
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "8")
    _submit(store, tenant="acme", cid=_campaign(store, "acme", "a"))
    _submit(store, tenant="globex", cid=_campaign(store, "globex", "g"))
    before = sorted(store.state_dir.glob("jobs/*/*"))
    with pytest.raises(jobs.EngineQueueFull):
        _submit(store, tenant="initech", cid=_campaign(store, "initech", "i"))
    assert sorted(store.state_dir.glob("jobs/*/*")) == before  # refused, nothing left behind


def test_the_capacity_precheck_sees_the_engine_wide_cap(store, monkeypatch):
    monkeypatch.setenv("KALOS_JOB_GLOBAL_QUEUE_DEPTH", "1")
    _submit(store, tenant="acme", cid=_campaign(store, "acme", "a"))
    with pytest.raises(jobs.EngineQueueFull):
        jobs.check_queue_capacity(store, tenant="globex")


def test_the_global_queue_depth_is_validated_like_the_others(monkeypatch):
    monkeypatch.setenv("KALOS_JOB_GLOBAL_QUEUE_DEPTH", "0")
    with pytest.raises(ValueError, match="KALOS_JOB_GLOBAL_QUEUE_DEPTH"):
        jobs.job_settings()
    monkeypatch.delenv("KALOS_JOB_GLOBAL_QUEUE_DEPTH")
    assert jobs.job_settings().global_queue_depth == jobs.DEFAULT_GLOBAL_QUEUE_DEPTH


def test_the_store_refuses_an_over_long_roles_value_too(store):
    cid = _campaign(store)
    with pytest.raises(jobs.JobRolesTooLarge):
        jobs.enqueue(store, tenant="acme", campaign_id=cid, raw=_csv(), filename="runs.csv",
                     target="lipase_titer", anonymize=False, roles="x" * (jobs.MAX_ROLES_CHARS + 1),
                     sheet=None, requested_by=None)
    assert not list(store.state_dir.glob("jobs/*/*"))


def test_upload_over_the_tenant_limit_is_refused_before_anything_is_written(store, monkeypatch):
    monkeypatch.setenv("KALOS_JOB_MAX_UPLOAD_BYTES", "100")
    cid = _campaign(store)
    with pytest.raises(jobs.JobTooLarge):
        _submit(store, cid=cid, raw=b"x" * 101)
    assert not list(store.state_dir.glob("jobs/*/*.bin"))
    assert jobs.list_jobs(store, tenant="acme", campaign_id=cid) == []


def test_submission_to_a_foreign_or_archived_campaign_is_refused(store):
    globex_cid = _campaign(store, "globex")
    from kalos.portal.campaign import CampaignNotFound

    with pytest.raises(CampaignNotFound):
        _submit(store, tenant="acme", cid=globex_cid)
    cid = _campaign(store)
    store.archive_campaign(tenant="acme", campaign_id=cid)
    from kalos.portal.campaign import CampaignArchived

    with pytest.raises(CampaignArchived):
        _submit(store, cid=cid)
    assert not list(store.state_dir.glob("jobs/*/*.bin"))


# --- cancellation ----------------------------------------------------------------- #


def test_cancelling_a_queued_job_is_immediate_and_never_spawns_a_worker(store):
    spawned: list[str] = []
    runner = JobRunner(store, spawn=lambda j, t, d: spawned.append(j) or pytest.fail("spawned"),
                       timeout_s=60, poll_s=0.01)
    job = _submit(store)
    cancelled = jobs.request_cancel(store, tenant="acme", job_id=job["id"])
    assert cancelled["state"] == "cancelled" and cancelled["cancel_requested"] is True
    runner.tick()
    assert spawned == []
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "cancelled"
    assert not list(store.state_dir.glob("jobs/*/*.bin"))


def test_cancelling_a_running_job_stops_its_worker_and_publishes_nothing(store):
    cid = _campaign(store)
    started, release = threading.Event(), threading.Event()

    def gated(df, target, **kw):
        started.set()
        release.wait(10)
        return _stub_analyze(df, target, **kw)  # first progress() call sees the stop flag

    job = _submit(store, cid=cid)
    runner = _runner(store, gated)
    runner.tick()
    assert started.wait(5)
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "running"

    requested = jobs.request_cancel(store, tenant="acme", job_id=job["id"])
    assert requested["state"] == "running" and requested["cancel_requested"] is True
    release.set()  # the worker resumes, meets its progress checkpoint, and stops
    _settle(runner)
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "cancelled"
    assert store.load_analysis(tenant="acme", campaign_id=cid) is None  # nothing published
    assert store.get(tenant="acme", campaign_id=cid) is None


def test_cancel_after_success_changes_nothing(store):
    job = _submit(store)
    done = _drive(_runner(store), store, "acme", job["id"])
    again = jobs.request_cancel(store, tenant="acme", job_id=job["id"])
    assert again["state"] == "succeeded" and again["error"] is None
    assert again["campaign_generation"] == done["campaign_generation"]


def test_cancel_is_idempotent(store):
    job = _submit(store)
    first = jobs.request_cancel(store, tenant="acme", job_id=job["id"])
    second = jobs.request_cancel(store, tenant="acme", job_id=job["id"])
    assert first["state"] == second["state"] == "cancelled"


def test_a_cancel_requested_before_the_commit_rolls_the_publish_back(store):
    """The commit hook is the last check: a cancel that lands while the analysis
    finishes must stop the publish, not race it."""
    job = _submit(store)
    jobs.claim(store, tenant="acme", runner_id="t")  # now running
    jobs.request_cancel(store, tenant="acme", job_id=job["id"])
    with pytest.raises(jobs.JobCancelled):
        with store._txn(write=True) as conn:
            jobs.mark_succeeded(conn, tenant="acme", job_id=job["id"], generation="g")
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "running"


# --- timeouts and crashes --------------------------------------------------------- #


def test_a_job_past_its_time_limit_is_stopped_and_marked_timed_out(store):
    release = threading.Event()

    def stuck(df, target, **kw):
        release.wait(10)
        return _stub_analyze(df, target, **kw)

    job = _submit(store)
    runner = _runner(store, stuck, timeout_s=0.05)
    runner.tick()
    time.sleep(0.15)
    runner.tick()  # past its limit: the worker is stopped and the job says so
    release.set()
    _settle(runner)
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "timed_out"
    assert done["error"]["code"] == "timeout"
    assert done["campaign_generation"] is None
    assert runner.running_count() == 0 and runner.limit.running() == 0


def test_a_worker_that_crashes_is_recorded_as_failed(store):
    def crash(*_a, **_k):
        raise SystemExit("worker died")  # escapes run_job's Exception handler

    job = _submit(store)
    done = _drive(_runner(store, crash), store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_exited"


def test_stopping_the_runner_fails_its_running_jobs(store):
    release = threading.Event()

    def stuck(df, target, **kw):
        release.wait(10)
        return _stub_analyze(df, target, **kw)

    job = _submit(store)
    runner = _runner(store, stuck)
    runner.tick()
    runner.stop(timeout=1)
    release.set()
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "engine_stopped"
    assert runner.limit.running() == 0


def test_a_failed_worker_start_fails_the_job_and_frees_the_slot(store):
    def no_spawn(*_a):
        raise OSError("no such executable")

    job = _submit(store)
    runner = JobRunner(store, spawn=no_spawn, timeout_s=60, poll_s=0.01)
    runner.tick()
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_unavailable"
    assert runner.limit.running() == 0


class _FakeWorker:
    """A worker that runs until it is killed (or exits with `exit_code`). Its pid
    is a stand-in: nothing is ever signalled."""

    def __init__(self, exit_code: int | None = None) -> None:
        self.pid = 424242
        self.kills = 0
        self._exit_code = exit_code

    def poll(self) -> int | None:
        if self._exit_code is not None:
            return self._exit_code
        return 0 if self.kills else None

    def kill(self) -> None:
        self.kills += 1


def test_a_worker_whose_pid_cannot_be_recorded_is_killed_and_its_slot_freed(store, monkeypatch):
    """The regression from the review (HIGH1): a DB write that fails after the
    spawn must not orphan the worker or leak its slot."""
    job = _submit(store)
    workers: list[_FakeWorker] = []

    def spawn(_job_id, _tenant, _state_dir):
        workers.append(_FakeWorker())
        return workers[-1]

    def locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(jobs, "attach_worker", locked)
    runner = JobRunner(store, spawn=spawn, timeout_s=60, poll_s=0.01)
    runner.tick()  # must not raise
    assert workers and workers[0].kills >= 1
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_unavailable"
    assert runner.running_count() == 0 and runner.limit.running() == 0


def test_a_worker_that_exits_on_its_own_is_failed_and_its_group_is_killed(store):
    """MED5: a worker that exits without recording an outcome is a crash; the
    runner still kills its process group, so a child it left running is stopped."""
    job = _submit(store)
    runner = JobRunner(store, spawn=lambda *_a: _FakeWorker(exit_code=0), timeout_s=60, poll_s=0.01)
    done = _drive(runner, store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_exited"
    assert runner.running_count() == 0 and runner.limit.running() == 0


def _gone(pid: int) -> bool:
    """Exited (or a zombie). Reads /proc where it exists, so it needs no `ps`."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    if os.path.isdir("/proc/self"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return True
        return stat.rsplit(")", 1)[-1].split()[0] == "Z"
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],  # noqa: S603 - fixed argv
                           capture_output=True, text=True, check=False).stdout.strip()
    return state == "" or state.startswith("Z")


_CHILD_IGNORING_TERM = (
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
)


def test_a_worker_that_exits_leaves_no_process_group_members_behind(store, tmp_path):
    """Real processes: the worker starts a child in its own group, then exits
    without publishing. The runner's group kill must stop that child too."""
    job = _submit(store)
    pidfile = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
    )
    runner = JobRunner(
        store,
        spawn=lambda j, _t, d: job_runner.ProcessWorker(j, d, argv=[sys.executable, "-c", script]),
        timeout_s=60, poll_s=0.01,
    )
    done = _drive(runner, store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_exited"
    child_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while not _gone(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child_pid), "the worker's child survived the runner's group kill"


def test_a_failed_claim_leaves_the_job_queued_and_leaks_no_slot(store, monkeypatch):
    job = _submit(store)
    spawned: list[str] = []
    real_claim = jobs.claim

    def locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(jobs, "claim", locked)
    runner = JobRunner(store, spawn=lambda j, _t, _d: spawned.append(j) or _FakeWorker(),
                       timeout_s=60, poll_s=0.01)
    runner.tick()  # must not raise
    assert spawned == [] and runner.running_count() == 0 and runner.limit.running() == 0
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "queued"
    monkeypatch.setattr(jobs, "claim", real_claim)  # the next pass admits it
    runner.tick()
    assert spawned == [job["id"]]


def test_an_outcome_that_cannot_be_recorded_is_retried_and_keeps_its_slot(store, monkeypatch):
    job = _submit(store)
    real_finish = jobs.finish
    failures = {"left": 1}

    def flaky_finish(*args, **kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(jobs, "finish", flaky_finish)
    runner = JobRunner(store, spawn=lambda *_a: _FakeWorker(exit_code=1), timeout_s=60, poll_s=0.01)
    runner.tick()  # the worker exited; recording that failed, so the child stays
    assert runner.running_count() == 1 and runner.limit.running() == 1
    runner.tick()  # the retry records it and frees the slot
    assert runner.running_count() == 0 and runner.limit.running() == 0
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_exited"


# --- concurrency and fairness -------------------------------------------------------- #


def test_per_tenant_cap_stops_one_tenant_from_taking_every_slot(store):
    """Two slots total, one per tenant: acme's second job waits behind its first
    while globex's job gets the other slot."""
    busy.reset_limiter(total=2, per_tenant=1)
    release = threading.Event()

    def gated(df, target, **kw):
        release.wait(10)
        return _stub_analyze(df, target, **kw)

    a1 = _submit(store, cid=_campaign(store, "acme", "a"))
    a2 = _submit(store, cid=_campaign(store, "acme", "b"))
    g1 = _submit(store, tenant="globex", cid=_campaign(store, "globex"))
    runner = _runner(store, gated)
    runner.tick()
    states = {
        j: jobs.get_job(store, tenant=t, job_id=j)["state"]
        for j, t in ((a1["id"], "acme"), (a2["id"], "acme"), (g1["id"], "globex"))
    }
    assert states == {a1["id"]: "running", a2["id"]: "queued", g1["id"]: "running"}
    release.set()
    assert _drive(runner, store, "acme", a1["id"])["state"] == "succeeded"
    assert _drive(runner, store, "acme", a2["id"])["state"] == "succeeded"
    assert _drive(runner, store, "globex", g1["id"])["state"] == "succeeded"
    _settle(runner)


def test_global_cap_bounds_concurrent_workers(store):
    busy.reset_limiter(total=1, per_tenant=1)
    release = threading.Event()

    def gated(df, target, **kw):
        release.wait(10)
        return _stub_analyze(df, target, **kw)

    a = _submit(store, cid=_campaign(store, "acme", "a"))
    g = _submit(store, tenant="globex", cid=_campaign(store, "globex"))
    runner = _runner(store, gated)
    runner.tick()
    assert runner.running_count() == 1
    assert jobs.get_job(store, tenant="globex", job_id=g["id"])["state"] == "queued"
    release.set()
    assert _drive(runner, store, "acme", a["id"])["state"] == "succeeded"
    assert _drive(runner, store, "globex", g["id"])["state"] == "succeeded"


def test_a_sync_request_yields_to_a_queued_job_of_another_tenant(store):
    """With one slot and a queued job for globex, acme's synchronous request is
    refused (503) instead of taking the slot the job is waiting for."""
    busy.reset_limiter(total=1, per_tenant=1)
    _submit(store, tenant="globex", cid=_campaign(store, "globex"))
    runner = _runner(store)
    runner.limit.set_waiting({"globex"})
    assert busy.limiter().try_acquire("acme") is None
    globex_job = jobs.queued_tenants(store)
    assert globex_job == ["globex"]
    _drive(runner, store, "globex", _only_job_id(store, "globex"))
    _settle(runner)  # the worker has exited and its slot is back
    assert busy.limiter().try_acquire("acme") is not None


# --- worker lifetime (MED4) -------------------------------------------------------------- #


def test_the_runner_gives_each_worker_its_time_limit_plus_a_margin(store, monkeypatch):
    seen: dict = {}

    class Recorder(_FakeWorker):
        def __init__(self, job_id, state_dir, **kwargs):
            seen.update(kwargs)
            super().__init__(exit_code=0)

    monkeypatch.setattr(job_runner, "ProcessWorker", Recorder)
    _submit(store)
    JobRunner(store, timeout_s=100, poll_s=0.01).tick()
    assert seen["deadline_s"] == 100 + job_runner.WORKER_DEADLINE_MARGIN_S


def test_a_worker_process_receives_its_deadline_in_its_environment(tmp_path):
    out = tmp_path / "deadline.txt"
    argv = [sys.executable, "-c",
            f"import os, pathlib; pathlib.Path({str(out)!r}).write_text(os.environ['KALOS_JOB_DEADLINE_S'])"]
    worker = job_runner.ProcessWorker("job_" + "a" * 32, tmp_path, argv=argv, deadline_s=1234.5)
    deadline = time.monotonic() + 30
    while worker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert out.read_text() == "1234.5"


def test_the_worker_watchdog_exits_once_its_deadline_has_passed():
    fired: list[int] = []
    stop = job_worker.start_lifetime_watchdog(
        -1.0, parent_pid=os.getppid(), interval_s=0.01, exit_fn=fired.append)
    try:
        deadline = time.monotonic() + 5
        while not fired and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fired == [3]
    finally:
        stop.set()


def test_the_worker_watchdog_exits_when_its_engine_is_gone():
    fired: list[int] = []
    stop = job_worker.start_lifetime_watchdog(
        3600.0, parent_pid=os.getppid() + 100_000, interval_s=0.01, exit_fn=fired.append)
    try:
        deadline = time.monotonic() + 5
        while not fired and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fired == [3]
    finally:
        stop.set()


def test_the_worker_watchdog_stays_quiet_while_the_engine_is_alive_and_on_time():
    fired: list[int] = []
    stop = job_worker.start_lifetime_watchdog(
        3600.0, parent_pid=os.getppid(), interval_s=0.01, exit_fn=fired.append)
    time.sleep(0.2)
    stop.set()
    assert fired == []


def test_a_worker_that_ignores_sigterm_is_killed_with_its_group_at_its_time_limit(store, tmp_path):
    """Real processes: a worker and a child it started both ignore SIGTERM and
    sleep forever. At the time limit the runner kills the whole group (SIGKILL),
    so neither survives and the job is recorded as timed out."""
    job = _submit(store)
    pidfile = tmp_path / "child.pid"
    child_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"
    script = (
        "import pathlib, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "time.sleep(600)\n"
    )
    workers: list[job_runner.ProcessWorker] = []

    def spawn(job_id, _tenant, state_dir):
        workers.append(job_runner.ProcessWorker(job_id, state_dir, argv=[sys.executable, "-c", script]))
        return workers[-1]

    runner = JobRunner(store, spawn=spawn, timeout_s=4, poll_s=0.01)
    runner.tick()
    wait_until = time.monotonic() + 2.5
    while not pidfile.exists() and time.monotonic() < wait_until:
        time.sleep(0.02)
    assert pidfile.exists(), "the worker never started its child"
    done = _drive(runner, store, "acme", job["id"], timeout=30)
    assert done["state"] == "timed_out" and done["error"]["code"] == "timeout"
    assert _gone(workers[0].pid) and _gone(int(pidfile.read_text()))
    assert runner.running_count() == 0 and runner.limit.running() == 0


# --- one scheduler per state directory (MED6) --------------------------------------------- #


def test_a_second_runner_on_the_same_state_dir_is_refused_until_the_first_stops(store):
    first = JobRunner(store, poll_s=0.01)
    first.start()
    try:
        with pytest.raises(job_runner.RunnerLocked):
            JobRunner(store, poll_s=0.01).start()
    finally:
        first.stop(timeout=1)
    second = JobRunner(store, poll_s=0.01)
    second.start()  # the lock was released with the first runner
    second.stop(timeout=1)


def test_a_runner_refused_for_web_concurrency_takes_no_lock(store, monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(job_runner.RunnerLocked, match="single engine process"):
        JobRunner(store).start()
    monkeypatch.setenv("WEB_CONCURRENCY", "1")  # one worker is fine
    runner = JobRunner(store, poll_s=0.01)
    runner.start()
    runner.stop(timeout=1)


def test_the_runner_lock_is_released_after_its_stopped_jobs_are_recorded(store):
    """Another runner's recovery must not see this runner's jobs still `running`."""
    job = _submit(store)
    runner = JobRunner(store, spawn=lambda *_a: _FakeWorker(), timeout_s=60, poll_s=0.01)
    runner._lock_fd = job_runner._acquire_runner_lock(store.state_dir)  # as start() does
    runner.tick()  # the job is running, its worker alive
    held_while_recording: list[bool] = []
    real_finish = jobs.finish

    def probing_finish(*args, **kwargs):
        try:
            os.close(job_runner._acquire_runner_lock(store.state_dir))
            held_while_recording.append(False)
        except job_runner.RunnerLocked:
            held_while_recording.append(True)
        return real_finish(*args, **kwargs)

    jobs_module = jobs
    original = jobs_module.finish
    jobs_module.finish = probing_finish
    try:
        runner.stop(timeout=1)
    finally:
        jobs_module.finish = original
    assert held_while_recording == [True]
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "failed"
    assert runner._lock_fd is None


def test_another_process_holding_the_runner_lock_refuses_this_runner(store, tmp_path):
    """Real processes: a second engine process holds the state directory's lock."""
    lock = store.state_dir / job_runner.RUNNER_LOCK_NAME
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys, time\n"
         f"fd = os.open({str(lock)!r}, os.O_RDWR | os.O_CREAT, 0o600)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('locked', flush=True)\n"
         "time.sleep(600)\n"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(job_runner.RunnerLocked, match="another engine process"):
            JobRunner(store, poll_s=0.01).start()
    finally:
        holder.kill()
        holder.wait(timeout=10)
    runner = JobRunner(store, poll_s=0.01)
    runner.start()  # the other process is gone: the lock is free
    runner.stop(timeout=1)


# --- crash recovery ------------------------------------------------------------------- #


def test_recovery_fails_jobs_a_previous_process_left_running(store):
    job = _submit(store)
    jobs.claim(store, tenant="acme", runner_id="previous-process")
    assert job_runner.recover(store) == 1
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "interrupted"
    assert not list(store.state_dir.glob("jobs/*/*.bin"))


def test_recovery_sweeps_input_files_no_live_job_refers_to(store):
    job = _submit(store)
    orphan = store.state_dir / "jobs" / "deadbeefdeadbeef" / "job_orphan.bin"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"left behind")
    job_runner.recover(store)
    assert orphan.exists() is False
    live = store.state_dir / jobs._relative_input_path("acme", job["id"])
    assert live.exists()  # still queued, so its input is kept


def test_recovery_sweeps_half_written_uploads_too(store):
    """LOW7: a crash between writing an upload and renaming it leaves a `.tmp`."""
    job = _submit(store)
    stale = store.state_dir / "jobs" / "deadbeefdeadbeef" / "job_stale.tmp"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"half a file")
    job_runner.recover(store)
    assert not stale.exists()
    assert (store.state_dir / jobs._relative_input_path("acme", job["id"])).exists()


def test_input_that_no_longer_matches_its_checksum_is_refused(store):
    job = _submit(store)
    path = store.state_dir / jobs._relative_input_path("acme", job["id"])
    path.write_bytes(_csv(_df(20, seed=9)) + b"tampered\n")
    done = _drive(_runner(store), store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "input_unavailable"


def test_input_path_cannot_escape_the_jobs_directory(store):
    with pytest.raises(jobs.JobInputError):
        jobs._resolve_input(store, "../portal.db")


# --- publish compare-and-swap ------------------------------------------------------------ #


def test_campaign_changed_while_the_job_ran_fails_it_and_keeps_the_newer_state(store):
    cid = _campaign(store)
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid,
               analysis={"target": "lipase_titer", "marker": "older"}, dataset="old.csv")
    job = _submit(store, cid=cid)

    def meddling(df, target, **kw):
        result = _stub_analyze(df, target, **kw)
        store.start([{"recipe": {"Methanol": 1.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
                      "mode": "explore", "reason": "x"}], tenant="acme", campaign_id=cid)
        return result

    done = _drive(_runner(store, meddling), store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "campaign_changed"
    assert store.load_analysis(tenant="acme", campaign_id=cid)["marker"] == "older"


def test_publish_without_a_job_still_refuses_a_stale_generation(store):
    """The CAS the job relies on is the store's own: a stale expected generation
    raises (regression guard for the seed() extension)."""
    cid = _campaign(store)
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    with pytest.raises(CampaignConflict):
        store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid,
                   expected_generation=None)


# --- the real worker process ------------------------------------------------------------ #


def test_real_worker_process_publishes(tmp_path):
    """`python -m kalos.portal.job_worker <id>` in its own process, against a real
    state directory, running the REAL analysis on a small linear sheet."""
    import os

    store = CampaignStore(tmp_path)
    cid = _campaign(store)
    job = _submit(store, cid=cid)
    assert jobs.claim(store, tenant="acme", runner_id="test")["id"] == job["id"]  # as the runner does
    store.close()
    env = {**os.environ, "KALOS_STATE_DIR": str(tmp_path), "KALOS_TORCH_THREADS": "0"}
    out = subprocess.run(
        [sys.executable, "-m", "kalos.portal.job_worker", job["id"]],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    store = CampaignStore(tmp_path)
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done is not None
    assert done["state"] == "succeeded", done
    assert done["campaign_generation"] == store.load_analysis(
        tenant="acme", campaign_id=cid)["campaign_generation"]
    store.close()


def test_input_paths_are_server_built_and_tenant_hashed():
    key = jobs._tenant_key("acme")
    assert len(key) == 16 and key.isalnum()
    assert hashlib.sha256(b"acme").hexdigest().startswith(key)
    rel = jobs._relative_input_path("acme", "job_" + "a" * 32)
    assert rel == f"jobs/{key}/job_{'a' * 32}.bin"


@pytest.mark.parametrize("failure", ["timeout_setting", "keyboard_interrupt"])
def test_a_failed_start_cannot_leak_the_claimed_job_or_its_slot(store, failure):
    """N4: the child was built from the runner's timeout setting before the start's
    try block, so a setting that raises left the job `running` with its slot held
    forever. Any failure of the start (even a BaseException) now releases the slot
    and records the job `worker_unavailable`."""
    job = _submit(store)
    spawned: list[str] = []

    if failure == "timeout_setting":
        class Runner(JobRunner):
            @property
            def timeout_s(self) -> float:
                raise ValueError("KALOS_JOB_TIMEOUT_S must be a number")

        runner = Runner(store, spawn=lambda *a: spawned.append("spawned"), timeout_s=None, poll_s=0.01)
    else:
        def spawn(*_a):
            raise KeyboardInterrupt()

        runner = JobRunner(store, spawn=spawn, timeout_s=60, poll_s=0.01)

    runner.tick()  # must not raise: the failure is the job's outcome, not the scheduler's
    assert spawned == []  # the worker is never started when the setup fails
    assert runner.running_count() == 0
    assert runner.limit.running() == 0
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "worker_unavailable"


def test_the_runner_module_imports_without_fcntl_and_refuses_only_when_enabled(tmp_path):
    """N5: `fcntl` (POSIX only) is imported where the lock is taken, not at import time.
    With jobs disabled the runner module loads and starts nothing on a platform
    without fcntl; with jobs enabled the boot fails with a message that says why."""
    code = (
        "import os, sys\n"
        "sys.modules['fcntl'] = None  # what a non-POSIX platform looks like to the import\n"
        "from kalos.portal import job_runner\n"
        "from kalos.portal.campaign import CampaignStore\n"
        "store = CampaignStore(sys.argv[1])\n"
        "os.environ['KALOS_JOBS_ENABLED'] = '0'\n"
        "print('disabled:', job_runner.start_job_runner(store))\n"
        "os.environ['KALOS_JOBS_ENABLED'] = '1'\n"
        "try:\n"
        "    job_runner.start_job_runner(store)\n"
        "except job_runner.RunnerLocked as exc:\n"
        "    print('enabled: refused:', exc)\n"
        "store.close()\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(job_runner.__file__)))
    env = {**os.environ, "PYTHONPATH": repo}
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "state")],
                         capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    assert "disabled: None" in out.stdout
    assert "enabled: refused:" in out.stdout and "POSIX" in out.stdout, out.stdout


class _ProcesslessWorker:
    """A worker that runs until it is killed (the runner's contract, no process)."""

    pid = None

    def __init__(self, killed: list[str], job_id: str) -> None:
        self._alive = True
        self._killed = killed
        self._job_id = job_id

    def poll(self) -> int | None:
        return None if self._alive else -9

    def kill(self) -> None:
        self._alive = False
        self._killed.append(self._job_id)


def test_a_claim_that_races_a_stop_is_never_started_unsupervised(store, monkeypatch):
    """N6: stop() ran between the scheduler's stop check and its claim. It killed the
    children it knew about and released the runner lock; the claim then succeeded
    and a worker started after the lock was gone, with nothing left to stop it. A
    pass that is under way now holds the pass lock, and stop() waits for it, so the
    claimed job is stopped and recorded as engine_stopped like any other."""
    job = _submit(store)
    killed: list[str] = []
    runner = JobRunner(store, spawn=lambda j, _t, _d: _ProcesslessWorker(killed, j), timeout_s=60, poll_s=0.01)
    real_claim = jobs.claim
    helper = threading.Thread(target=lambda: None)

    def claim_while_stop_starts(*args, **kwargs):
        nonlocal helper
        helper = threading.Thread(target=runner.stop, kwargs={"timeout": 0.05})
        helper.start()
        helper.join(timeout=0.5)  # old code: stop finishes here; new code: it waits for this pass
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(jobs, "claim", claim_while_stop_starts)
    runner.tick()
    helper.join(timeout=10)
    assert not helper.is_alive()
    assert killed == [job["id"]]
    assert runner.running_count() == 0 and runner.limit.running() == 0
    done = jobs.get_job(store, tenant="acme", job_id=job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "engine_stopped"


def test_stop_keeps_the_runner_lock_while_a_pass_is_still_running(store, monkeypatch):
    """N6: when the scheduler cannot be stopped in time, the lock is kept (the OS
    releases it at exit), so a second runner still cannot start on this directory."""
    _submit(store)
    runner = JobRunner(store, spawn=lambda j, _t, _d: _ProcesslessWorker([], j), timeout_s=60,
                       poll_s=0.01, pass_wait_s=0.1)
    runner.start()
    entered, let_go = threading.Event(), threading.Event()
    real_claim = jobs.claim

    def slow_claim(*args, **kwargs):
        entered.set()
        let_go.wait(10)
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(jobs, "claim", slow_claim)
    assert entered.wait(10)  # the scheduler thread is inside a pass
    runner.stop(timeout=0.1)
    try:
        assert runner._lock_fd is not None
        with pytest.raises(job_runner.RunnerLocked):
            job_runner._acquire_runner_lock(store.state_dir)
    finally:
        let_go.set()


def test_a_refused_submit_writes_no_upload_to_disk(store, monkeypatch):
    """N7: a submit refused for depth (or size) never writes its bytes to the state
    directory; the depth check runs before the write, and the in-transaction check
    stays authoritative for races."""
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "1")
    _submit(store, tenant="acme")  # the tenant is now at its depth
    files_before = sorted((store.state_dir / jobs.JOBS_DIRNAME).glob("*/*"))
    writes: list[Path] = []
    real_write = jobs._write_input

    def spy(path, raw):
        writes.append(path)
        return real_write(path, raw)

    monkeypatch.setattr(jobs, "_write_input", spy)
    with pytest.raises(jobs.QueueFull):
        _submit(store, tenant="acme")
    assert writes == []
    assert sorted((store.state_dir / jobs.JOBS_DIRNAME).glob("*/*")) == files_before


def test_a_job_that_cannot_be_loaded_is_reported_without_a_traceback(store, monkeypatch, caplog):
    """N8: a failure to load the job (the database, say) must not escape run_job as a
    traceback on the worker's stderr. Its message can quote a customer column; only
    the exception type is logged."""
    def broken_load(_store, _job_id):
        raise ValueError("column 'SSN-4411' is not numeric")

    monkeypatch.setattr(jobs, "load_for_worker", broken_load)
    with caplog.at_level("DEBUG", logger="kalos.portal"):
        result = job_worker.run_job(store, "job_" + "a" * 32)
    assert result == "unavailable"
    assert "SSN-4411" not in caplog.text
    assert "ValueError" in caplog.text


def test_the_worker_reports_a_store_it_cannot_open_without_a_traceback(monkeypatch, caplog):
    """N8: `main` opens the store inside the process; a failure there exits 1 with
    the exception type logged, not a traceback that carries its message."""
    from kalos.portal import campaign as campaign_module

    class Unopenable:
        def __init__(self, *_a, **_k):
            raise OSError("cannot open /customers/acme/portal.db")

    monkeypatch.setattr(campaign_module, "CampaignStore", Unopenable)
    with caplog.at_level("DEBUG", logger="kalos.portal"):
        assert job_worker.main(["job_" + "a" * 32]) == 1
    assert "/customers/acme" not in caplog.text
    assert "OSError" in caplog.text


def test_the_worker_watchdog_is_not_moved_by_a_wall_clock_jump(monkeypatch):
    """The worker deadline is a monotonic duration, so a wall-clock step (NTP, a manual
    change) neither kills a live worker nor extends one. A 3600 s deadline must not
    fire when the wall clock reads year 2096."""
    fired: list[int] = []
    monkeypatch.setattr(job_worker.time, "time", lambda: 4_000_000_000.0)
    stop = job_worker.start_lifetime_watchdog(
        3600.0, parent_pid=os.getppid(), interval_s=0.01, exit_fn=fired.append)
    time.sleep(0.2)
    stop.set()
    assert fired == []


def _wait_for_file(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"{path} was never written"


def test_an_engine_less_worker_and_its_child_are_killed_together(tmp_path, monkeypatch):
    """N3, real processes: a worker whose engine is gone, with a child that ignores
    SIGTERM. The watchdog SIGKILLs the worker's whole process group, so the worker
    and the child both die. Before, the watchdog ended only the worker."""
    monkeypatch.setenv("PYTHONPATH", str(Path(job_worker.__file__).resolve().parents[2]))
    pidfile = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time\n"
        "from kalos.portal.job_worker import start_lifetime_watchdog\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_CHILD_IGNORING_TERM!r}])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "start_lifetime_watchdog(600.0, parent_pid=-1, interval_s=0.05, kill_group=True)\n"
        "time.sleep(600)\n"
    )
    worker = job_runner.ProcessWorker("job_" + "a" * 32, tmp_path, argv=[sys.executable, "-c", script])
    _wait_for_file(pidfile)
    child_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 30
    while worker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert worker.poll() is not None, "the engine-less worker did not exit"
    deadline = time.monotonic() + 10
    while not _gone(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child_pid), "the worker's child survived the worker's own exit"


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="the /proc check is Linux-only")
def test_recover_kills_a_live_worker_group_without_running_ps(store, tmp_path, monkeypatch):
    """N3: after an engine crash the next start kills the worker group the job
    recorded, identifying the worker from /proc, not from `ps` (absent in slim images).
    Before, the check ran `ps` and never killed anything without it."""
    monkeypatch.setenv("PYTHONPATH", str(Path(job_worker.__file__).resolve().parents[2]))
    job = _submit(store)
    job_id = str(job["id"])
    jobs.claim(store, tenant="acme", runner_id="crashed-engine")
    pidfile = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_CHILD_IGNORING_TERM!r}])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "time.sleep(600)\n"
    )
    # The argv names the job, as the real worker's does (`python -m kalos.portal.job_worker <id>`).
    worker = job_runner.ProcessWorker(job_id, tmp_path,
                                      argv=[sys.executable, "-c", script, "job_worker", job_id])
    jobs.attach_worker(store, tenant="acme", job_id=job_id, pid=int(worker.pid))
    _wait_for_file(pidfile)
    child_pid = int(pidfile.read_text())

    def no_ps(*_a, **_k):
        raise AssertionError("recover must not need ps")

    monkeypatch.setattr(job_runner.subprocess, "run", no_ps)
    assert job_runner.recover(store) == 1
    deadline = time.monotonic() + 10
    while worker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert worker.poll() is not None, "the crashed engine's worker survived recovery"
    deadline = time.monotonic() + 10
    while not _gone(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child_pid), "the worker's child survived recovery"
    done = jobs.get_job(store, tenant="acme", job_id=job_id)
    assert done["state"] == "failed" and done["error"]["code"] == "interrupted"


def test_recover_never_signals_a_live_process_that_is_not_the_job_worker(store, tmp_path):
    """N3 guard: a pid recorded for a job that now belongs to another process (a
    reused pid) is left alone: its command line does not name the job."""
    job = _submit(store)
    job_id = str(job["id"])
    jobs.claim(store, tenant="acme", runner_id="crashed-engine")
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        jobs.attach_worker(store, tenant="acme", job_id=job_id, pid=bystander.pid)
        assert job_runner.recover(store) == 1
        assert bystander.poll() is None, "recovery signalled an unrelated process"
    finally:
        bystander.kill()
        bystander.wait()



def test_the_worker_main_kills_its_group_when_its_deadline_passes(tmp_path, monkeypatch):
    """N3, end to end through `job_worker.main`: a worker past its deadline (its job
    stands in as a long `run_job`) takes its child down with it, the child ignoring
    SIGTERM. Covers the wiring: the group-leading worker enables the group kill."""
    monkeypatch.setenv("PYTHONPATH", str(Path(job_worker.__file__).resolve().parents[2]))
    job_id = "job_" + "b" * 32
    pidfile = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time\n"
        "from kalos.portal import job_worker\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_CHILD_IGNORING_TERM!r}])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "job_worker.run_job = lambda *a, **k: time.sleep(600)\n"
        f"sys.exit(job_worker.main([{job_id!r}]))\n"
    )
    worker = job_runner.ProcessWorker(job_id, tmp_path, argv=[sys.executable, "-c", script],
                                      deadline_s=1.0)
    _wait_for_file(pidfile)
    child_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 30
    while worker.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert worker.poll() is not None, "the worker did not stop at its deadline"
    deadline = time.monotonic() + 10
    while not _gone(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child_pid), "the worker's child survived its deadline"


_ORPHAN_WORKER = """
import sys, time
from kalos.portal import job_worker

_real_run_job = job_worker.run_job

def _slow_analyze(df, target, *, profile=None, anonymize=False, progress=None, roles=None, **_):
    time.sleep(1.0)  # a publish that happens after the engine has died
    return {"target": target, "proposal_features": ["Methanol", "pH"], "proposals": [],
            "marker": "orphan"}

job_worker.run_job = lambda store, job_id: _real_run_job(store, job_id, analyze=_slow_analyze)
sys.exit(job_worker.main(sys.argv[1:]))
"""


def test_an_orphaned_worker_never_publishes(store, tmp_path, monkeypatch):
    """D1, real processes: the engine that started a worker dies 0.2 s after the spawn,
    while the worker is still importing. Before, the watchdog took its 'engine' pid
    after the imports, so the orphan ran to completion and published over the campaign
    (job succeeded, pending runs gone). Now the worker compares against the engine pid
    it was given, checks it before any analysis, and refuses to publish."""
    monkeypatch.setenv("PYTHONPATH", str(Path(job_worker.__file__).resolve().parents[2]))
    cid = _campaign(store, "acme")
    job = _submit(store, cid=cid)
    job_id = str(job["id"])
    jobs.claim(store, tenant="acme", runner_id="engine-that-dies")  # the job is running
    pidfile = tmp_path / "worker.pid"
    engine = (
        "import os, pathlib, sys, time\n"
        "from kalos.portal.job_runner import ProcessWorker\n"
        f"w = ProcessWorker({job_id!r}, pathlib.Path({str(tmp_path)!r}), "
        f"argv=[sys.executable, '-c', {_ORPHAN_WORKER!r}, {job_id!r}], deadline_s=60)\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(w.pid))\n"
        "time.sleep(0.2)\n"
        "os._exit(0)\n"  # the engine dies while its worker is still importing
    )
    env = {**os.environ, "KALOS_STATE_DIR": str(tmp_path)}
    subprocess.run([sys.executable, "-c", engine], env=env, timeout=60, check=True)
    _wait_for_file(pidfile)
    worker_pid = int(pidfile.read_text())
    deadline = time.monotonic() + 30
    while not _gone(worker_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(worker_pid), "the orphaned worker is still running"
    done = jobs.get_job(store, tenant="acme", job_id=job_id)
    assert done["state"] != "succeeded", "an orphaned worker published"
    assert store.get(tenant="acme", campaign_id=cid) is None  # the campaign was not touched


def test_a_worker_whose_engine_is_not_its_parent_publishes_nothing(store, monkeypatch):
    """D1, the publish guard on its own: a worker told its engine is another process
    (not its parent) stops before publishing. The job stays running and the campaign
    is untouched. Deterministic, no process."""
    cid = _campaign(store, "acme")
    job = _submit(store, cid=cid)
    job_id = str(job["id"])
    jobs.claim(store, tenant="acme", runner_id="engine")
    monkeypatch.setenv(jobs.ENGINE_PID_ENV, str(os.getppid() + 100_000))
    assert job_worker.run_job(store, job_id, analyze=_stub_analyze) == "stopped"
    assert jobs.get_job(store, tenant="acme", job_id=job_id)["state"] == "running"
    assert store.get(tenant="acme", campaign_id=cid) is None


def _campaign_with_a_pending_run(store, tenant: str = "acme") -> str:
    """A seeded campaign with one awaiting run (what the audit's lost measurement was)."""
    cid = _campaign(store, tenant)
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant=tenant, campaign_id=cid)
    store.start([{"recipe": {"Methanol": 1.0, "pH": 6.0}}], tenant=tenant, campaign_id=cid)
    return cid


def test_a_job_refuses_to_replace_a_campaign_that_gained_runs_after_submission(store):
    """D3: an upload job seeds the campaign, which discards its pending runs. Runs
    added after the submit must stop the job (campaign_has_runs), not be discarded:
    nothing is published and the campaign is unchanged."""
    cid = _campaign(store)
    job = _submit(store, cid=cid)
    job_id = str(job["id"])
    jobs.claim(store, tenant="acme", runner_id="engine")
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    store.start([{"recipe": {"Methanol": 1.0, "pH": 6.0}}], tenant="acme", campaign_id=cid)
    before = store.get(tenant="acme", campaign_id=cid)
    assert job_worker.run_job(store, job_id, analyze=_stub_analyze) == "failed"
    done = jobs.get_job(store, tenant="acme", job_id=job_id)
    assert done["state"] == "failed" and done["error"]["code"] == "campaign_has_runs"
    assert store.get(tenant="acme", campaign_id=cid) == before


def test_a_confirmed_replacement_publishes_over_the_runs(store):
    """D3: with replace=true the job is the explicit confirmation and publishes."""
    cid = _campaign_with_a_pending_run(store)
    job = jobs.enqueue(
        store, tenant="acme", campaign_id=cid, raw=_csv(), filename="runs.csv",
        target="lipase_titer", anonymize=False, roles="", sheet=None, requested_by="tester",
        replace=True,
    )
    jobs.claim(store, tenant="acme", runner_id="engine")
    assert job_worker.run_job(store, str(job["id"]), analyze=_stub_analyze) == "succeeded"
    assert store.get(tenant="acme", campaign_id=cid)["pending"] == []


def test_a_second_pending_upload_to_the_same_campaign_is_refused(store):
    """D15: a second queued (or running) upload to one campaign would fail campaign_changed
    after its full fit, once the first publishes. It is refused at submit instead."""
    cid = _campaign(store)
    _submit(store, cid=cid)  # queued
    with pytest.raises(jobs.UploadPending):
        _submit(store, cid=cid)
    other = _campaign(store, name="other")
    _submit(store, cid=other)  # another campaign is unaffected


@pytest.mark.parametrize("name", ["KALOS_JOB_TIMEOUT_S", "KALOS_JOB_QUEUE_DEPTH",
                                  "KALOS_JOB_GLOBAL_QUEUE_DEPTH", "KALOS_JOB_MAX_UPLOAD_BYTES"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e400", "NaN"])
def test_a_non_finite_job_setting_is_refused_not_ignored(monkeypatch, name, value):
    """Round 3, 1: `nan` passed the minimum check (nan < 1 is False), so a NaN time limit
    never fired and no job ever timed out. Non-finite numbers are refused for every
    numeric KALOS_JOB* variable, and for the limiter's."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        jobs.job_settings()


@pytest.mark.parametrize("name", ["KALOS_JOB_WORKERS", "KALOS_JOB_PER_TENANT"])
def test_a_non_finite_limiter_setting_is_refused_at_startup(monkeypatch, name):
    from kalos.portal import busy

    # Scoped: the autouse fixture resets the limiter from the environment on teardown.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(name, "1e400")
        with pytest.raises(ValueError, match=name):
            busy.validate_env()
