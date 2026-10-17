"""Background jobs (docs/JOBS.md): the queue, the worker and the runner, end to end
at the store level. The analysis is a stub (`_stub_analyze`) and the worker runs
in a thread (`ThreadWorker`), so these are fast and deterministic; the real
subprocess path is covered by `test_real_worker_process_publishes`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time

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


# --- queue depth, upload size ---------------------------------------------------- #


def test_queue_depth_is_per_tenant_and_frees_on_cancel(store, monkeypatch):
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "2")
    cid = _campaign(store)
    a = _submit(store, cid=cid)
    _submit(store, cid=cid)
    with pytest.raises(jobs.QueueFull):
        _submit(store, cid=cid)
    # another tenant is not affected by acme's full queue
    _submit(store, tenant="globex", cid=_campaign(store, "globex"))
    jobs.request_cancel(store, tenant="acme", job_id=a["id"])
    _submit(store, cid=cid)  # room again
    # refused submissions leave no file behind
    assert len(list(store.state_dir.glob("jobs/*/*.bin"))) == 3


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
