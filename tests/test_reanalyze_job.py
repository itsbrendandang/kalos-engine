"""Campaign re-analysis as a background job (docs/JOBS.md, "Re-analysis jobs"):
`POST /api/campaigns/{id}/reanalyze/jobs`, then the shared job status / cancel routes.

Same harness as tests/test_jobs_api.py: two tenants with static bearer tokens, the
worker runs the real `job_worker.run_job` in a thread (`ThreadWorker`) with a stub
analysis, driven by `JobRunner.tick()`.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import busy, job_worker, jobs
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.job_runner import JobRunner, ThreadWorker
from kalos.portal.reanalyze_job import reanalyze_analysis_kwargs


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-write"), "subject": "hank@globex", "tenant": "globex",
     "scopes": ["read", "write"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _df(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2).round(3),
    })


SEEN: list[dict] = []


def _stub_analyze(df, target, *, profile=None, anonymize=False, progress=None, pending=None, **_):
    SEEN.append({"rows": len(df), "pending": pending, "anonymize": anonymize})
    if progress is not None:
        progress("fitting", 0.15)
        progress("calibrating", 0.55)
        progress("proposing", 0.8)
    return {"target": target, "proposal_features": ["Methanol", "pH"], "d": 2,
            "proposals": [], "marker": "reanalyze-stub"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in ("KALOS_JOB_QUEUE_DEPTH", "KALOS_JOB_MAX_UPLOAD_BYTES", "KALOS_JOB_TIMEOUT_S",
                 "KALOS_JOBS_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    busy.reset_limiter(total=2, per_tenant=1)
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    SEEN.clear()
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        busy.reset_limiter()
        store.close()


def _runner(store: CampaignStore, analyze=_stub_analyze) -> JobRunner:
    def spawn(job_id, _tenant, _state_dir):
        return ThreadWorker(lambda should_stop: job_worker.run_job(
            store, job_id, analyze=analyze, should_stop=should_stop))

    return JobRunner(store, spawn=spawn, timeout_s=60, poll_s=0.01)


def _drain(runner: JobRunner, store: CampaignStore, tenant: str, job_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        runner.tick()
        job = jobs.get_job(store, tenant=tenant, job_id=job_id)
        if job and job["state"] in jobs.TERMINAL_STATES:
            while runner.running_count():
                runner.tick()
                time.sleep(0.005)
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _seeded(store: CampaignStore, tenant: str = "acme", *, measured: int = 2, awaiting: int = 1,
            anonymize: bool = False) -> str:
    """A seeded campaign with `measured` measured and `awaiting` awaiting runs."""
    cid = store.create_campaign(tenant=tenant, name="c")["id"]
    df = _df()
    analysis = {"target": "lipase_titer", "d": 2, "proposals": [],
                "run": {"anonymize": anonymize}}
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant=tenant, campaign_id=cid,
               analysis=analysis)
    recipes = [{"recipe": {"Methanol": 1.0 + i, "pH": 6.0}} for i in range(measured + awaiting)]
    if recipes:
        runs = store.start(recipes, tenant=tenant, campaign_id=cid)
        for i, run in enumerate(runs[:measured]):
            store.set_result(run["id"], 2.0 + i, tenant=tenant, campaign_id=cid)
    return cid


def _post(tc: TestClient, token: str, cid: str):
    return tc.post(f"/api/campaigns/{cid}/reanalyze/jobs", headers=H(token))


# --- submit + success --------------------------------------------------------------- #


def test_submit_202_then_success_folds_and_publishes(env):
    tc, store = env
    cid = _seeded(store, measured=2, awaiting=1)
    before = store.get(tenant="acme", campaign_id=cid)
    r = _post(tc, "a-write", cid)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert r.headers["location"] == f"/api/jobs/{job['id']}"
    assert job["kind"] == "reanalyze" and job["state"] == "queued" and job["campaign_id"] == cid
    assert job["input_bytes"] is None and "params" not in job
    # Nothing is folded at submit.
    assert store.get(tenant="acme", campaign_id=cid)["round"] == before["round"]

    done = _drain(_runner(store), store, "acme", job["id"])
    assert done["state"] == "succeeded" and done["error"] is None
    assert done["kind"] == "reanalyze"  # kept after the terminal params clear
    after = store.get(tenant="acme", campaign_id=cid)
    assert after["round"] == before["round"] + 1
    assert len(after["base_rows"]) == len(before["base_rows"]) + 2
    assert [r["result"] for r in after["pending"]] == [None]  # the awaiting run stays
    analysis = tc.get(f"/api/campaigns/{cid}/analysis", headers=H("a-read")).json()
    assert analysis["marker"] == "reanalyze-stub"
    assert analysis["campaign_generation"] == done["campaign_generation"]
    assert analysis["run"]["kind"] == "reanalyze"
    # The analysis saw the folded data and the awaiting recipe.
    assert SEEN[-1]["rows"] == len(before["base_rows"]) + 2
    assert SEEN[-1]["pending"] == [{"Methanol": 3.0, "pH": 6.0}]
    listed = tc.get(f"/api/campaigns/{cid}/jobs", headers=H("a-read")).json()["jobs"]
    assert listed[0]["id"] == job["id"] and listed[0]["kind"] == "reanalyze"
    audit = store.audit_entries(tenant="acme", limit=50)
    assert any(e["action"] == "campaign.reanalyze.submit" and e["job_id"] == job["id"]
               and e["outcome"] == "ok" for e in audit)


def test_upload_jobs_still_read_as_analysis(env):
    _tc, store = env
    cid = store.create_campaign(tenant="acme", name="u")["id"]
    job = jobs.enqueue(store, tenant="acme", campaign_id=cid, raw=_df().to_csv(index=False).encode(),
                       filename="r.csv", target="lipase_titer", anonymize=False, roles="",
                       sheet=None, requested_by="t")
    assert job["kind"] == "analysis"
    assert jobs.request_cancel(store, tenant="acme", job_id=job["id"])["kind"] == "analysis"


def test_progress_stages_are_reported_in_order(env):
    tc, store = env
    cid = _seeded(store)
    seen: list[str] = []
    orig = jobs.set_progress

    def spy(store_, *, tenant, job_id, stage, progress):
        seen.append(stage)
        orig(store_, tenant=tenant, job_id=job_id, stage=stage, progress=progress)

    jobs.set_progress = spy
    try:
        job = _post(tc, "a-write", cid).json()["job"]
        assert _drain(_runner(store), store, "acme", job["id"])["state"] == "succeeded"
    finally:
        jobs.set_progress = orig
    assert seen == ["planning", "fitting", "fitting", "calibrating", "proposing", "publishing"]


def test_anonymize_is_carried_over(env):
    _tc, store = env
    cid = _seeded(store, anonymize=True)
    kwargs = reanalyze_analysis_kwargs(store, "acme", cid)
    assert kwargs["anonymize"] is True and kwargs["pending"] == [{"Methanol": 3.0, "pH": 6.0}]
    assert kwargs["profile"].name


# --- cancel and conflict ------------------------------------------------------------- #


def test_cancel_a_running_reanalysis_writes_nothing(env):
    tc, store = env
    cid = _seeded(store)
    before = store.get(tenant="acme", campaign_id=cid)
    started, release = threading.Event(), threading.Event()

    def gated(df, target, **kw):
        started.set()
        release.wait(10)
        return _stub_analyze(df, target, **kw)

    job = _post(tc, "a-write", cid).json()["job"]
    runner = _runner(store, gated)
    runner.tick()
    assert started.wait(5)
    r = tc.post(f"/api/jobs/{job['id']}/cancel", headers=H("a-write"))
    assert r.status_code == 200 and r.json()["job"]["cancel_requested"] is True
    release.set()
    done = _drain(runner, store, "acme", job["id"])
    assert done["state"] == "cancelled" and done["kind"] == "reanalyze"
    after = store.get(tenant="acme", campaign_id=cid)
    assert after["round"] == before["round"] and after["pending"] == before["pending"]


def test_cancel_a_queued_reanalysis(env):
    tc, store = env
    cid = _seeded(store)
    job = _post(tc, "a-write", cid).json()["job"]
    r = tc.post(f"/api/jobs/{job['id']}/cancel", headers=H("a-write"))
    assert r.json()["job"]["state"] == "cancelled"
    # The campaign is free again.
    assert _post(tc, "a-write", cid).status_code == 202


def test_a_concurrent_write_fails_the_job_with_conflict_and_writes_nothing(env):
    tc, store = env
    cid = _seeded(store, measured=1, awaiting=1)
    started, release = threading.Event(), threading.Event()

    def gated(df, target, **kw):
        started.set()
        release.wait(10)
        return _stub_analyze(df, target, **kw)

    job = _post(tc, "a-write", cid).json()["job"]
    runner = _runner(store, gated)
    runner.tick()
    assert started.wait(5)
    # A result lands while the analysis runs: the planned fold is stale.
    awaiting = [r for r in store.get(tenant="acme", campaign_id=cid)["pending"] if r["result"] is None]
    store.set_result(awaiting[0]["id"], 9.0, tenant="acme", campaign_id=cid)
    mid = store.get(tenant="acme", campaign_id=cid)
    release.set()
    done = _drain(runner, store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "conflict"
    assert done["campaign_generation"] is None
    after = store.get(tenant="acme", campaign_id=cid)
    assert after["round"] == mid["round"] and after["pending"] == mid["pending"]
    assert after["base_rows"] == mid["base_rows"]
    assert store.load_analysis(tenant="acme", campaign_id=cid).get("marker") is None


def test_nothing_to_fold_at_run_time_fails_the_job(env):
    tc, store = env
    cid = _seeded(store, measured=1, awaiting=0)
    job = _post(tc, "a-write", cid).json()["job"]
    # Folded synchronously by someone else before the job ran.
    _frame, _target, gen = store.plan_fold(tenant="acme", campaign_id=cid)
    store.commit_fold(gen, tenant="acme", campaign_id=cid)
    done = _drain(_runner(store), store, "acme", job["id"])
    assert done["state"] == "failed" and done["error"]["code"] == "nothing_to_fold"


# --- refusals before queueing ---------------------------------------------------------- #


def test_nothing_to_fold_is_refused_before_queueing(env):
    tc, store = env
    cid = _seeded(store, measured=0, awaiting=1)
    r = _post(tc, "a-write", cid)
    assert r.status_code == 400 and r.json()["code"] == "nothing_to_fold"
    assert "measured" in r.json()["error"]
    assert jobs.list_jobs(store, tenant="acme", campaign_id=cid) == []


def test_an_unseeded_campaign_is_refused_no_data(env):
    tc, store = env
    cid = store.create_campaign(tenant="acme", name="empty")["id"]
    r = _post(tc, "a-write", cid)
    assert r.status_code == 400 and r.json()["code"] == "no_data"


def test_archived_is_409(env):
    tc, store = env
    cid = _seeded(store)
    store.archive_campaign(tenant="acme", campaign_id=cid)
    r = _post(tc, "a-write", cid)
    assert r.status_code == 409 and r.json()["code"] == "campaign_archived"


def test_one_active_job_per_campaign(env):
    tc, store = env
    cid = _seeded(store)
    assert _post(tc, "a-write", cid).status_code == 202
    second = _post(tc, "a-write", cid)
    assert second.status_code == 409 and second.json()["code"] == "job_pending"
    # An upload job on the same campaign is refused too (its historical code).
    up = tc.post(f"/api/campaigns/{cid}/jobs", headers=H("a-write"),
                 files={"file": ("r.csv", _df().to_csv(index=False).encode(), "text/csv")},
                 data={"target": "lipase_titer", "replace": "true"})
    assert up.status_code == 409 and up.json()["code"] == "upload_pending"
    # And a reanalyze is refused while an upload job is pending.
    other = _seeded(store)
    jobs.enqueue(store, tenant="acme", campaign_id=other, raw=b"a,b\n1,2\n", filename="r.csv",
                 target="b", anonymize=False, roles="", sheet=None, requested_by="t", replace=True)
    r = _post(tc, "a-write", other)
    assert r.status_code == 409 and r.json()["code"] == "job_pending"


def test_queue_full_is_429_and_disabled_is_503(env, monkeypatch):
    tc, store = env
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "1")
    a, b = _seeded(store), _seeded(store)
    assert _post(tc, "a-write", a).status_code == 202
    r = _post(tc, "a-write", b)
    assert r.status_code == 429 and r.json()["code"] == "rate_limited" and "retry-after" in r.headers
    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    r = _post(tc, "a-write", b)
    assert r.status_code == 503 and r.json()["code"] == "service_unavailable"


# --- scope and isolation ----------------------------------------------------------------- #


def test_read_scope_is_refused(env):
    tc, store = env
    cid = _seeded(store)
    assert _post(tc, "a-read", cid).status_code == 403
    assert tc.post(f"/api/campaigns/{cid}/reanalyze/jobs").status_code == 401
    assert jobs.list_jobs(store, tenant="acme", campaign_id=cid) == []


def test_cross_tenant_is_the_same_404_as_missing(env):
    tc, store = env
    globex = _seeded(store, tenant="globex")
    foreign = _post(tc, "a-write", globex)
    missing = _post(tc, "a-write", "cmp_" + "0" * 32)
    malformed = _post(tc, "a-write", "not-an-id")
    assert foreign.status_code == missing.status_code == malformed.status_code == 404
    assert foreign.content == missing.content == malformed.content
    assert jobs.list_jobs(store, tenant="globex", campaign_id=globex) == []
    # Globex's own job is invisible and uncancellable for acme.
    job = _post(tc, "b-write", globex).json()["job"]
    assert tc.get(f"/api/jobs/{job['id']}", headers=H("a-read")).status_code == 404
    assert tc.post(f"/api/jobs/{job['id']}/cancel", headers=H("a-write")).status_code == 404
    assert jobs.get_job(store, tenant="globex", job_id=job["id"])["state"] == "queued"
