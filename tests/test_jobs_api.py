"""`/api/campaigns/{id}/jobs` and `/api/jobs/{id}` (docs/JOBS.md): the HTTP contract,
and the cross-tenant isolation of jobs (tenant A cannot see, list, or cancel
tenant B's jobs, and gets the same bytes for "not yours" as for "not there").

Two tenants with STATIC bearer tokens, as in tests/test_cross_tenant_isolation.py.
The worker runs in a thread with a stub analysis (tests/test_jobs.py), driven by
`JobRunner.tick()`; the TestClient does not run the app lifespan, so nothing
schedules jobs behind the test's back.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import busy, job_worker, jobs  # noqa: E402
from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.portal.job_runner import JobRunner, ThreadWorker  # noqa: E402
from kalos.portal.uploads import MAX_UPLOAD_BYTES  # noqa: E402


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("b-write"), "subject": "hank@globex", "tenant": "globex",
     "scopes": ["read", "write"]},
]

GLOBEX_SECRET = "GLOBEX-SECRET-Project-Hera"
CID_RE = re.compile(r"cmp_[0-9a-f]{32}")


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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
    if progress is not None:
        progress("fitting", 0.15)
        progress("proposing", 0.8)
    return {"target": target, "proposal_features": ["Methanol", "pH"], "proposals": [],
            "marker": "api-stub"}


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
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        busy.reset_limiter()
        store.close()


def _runner(store: CampaignStore) -> JobRunner:
    def spawn(job_id, _tenant, _state_dir):
        return ThreadWorker(lambda should_stop: job_worker.run_job(
            store, job_id, analyze=_stub_analyze, should_stop=should_stop))

    return JobRunner(store, spawn=spawn, timeout_s=60, poll_s=0.01)


def _drain(runner: JobRunner, store: CampaignStore, tenant: str, job_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        runner.tick()
        job = jobs.get_job(store, tenant=tenant, job_id=job_id)
        if job and job["state"] in jobs.TERMINAL_STATES:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _submit(tc: TestClient, token: str, cid: str, *, data: bytes | None = None,
            target: str = "lipase_titer", name: str = "runs.csv"):
    return tc.post(
        f"/api/campaigns/{cid}/jobs",
        files={"file": (name, data if data is not None else _csv(), "text/csv")},
        data={"target": target},
        headers=H(token),
    )


def _campaign(tc: TestClient, token: str, name: str = "c") -> str:
    return tc.post("/api/campaigns", json={"name": name}, headers=H(token)).json()["id"]


# --- the happy path over HTTP ----------------------------------------------------- #


def test_submit_202_then_poll_status_until_published(env):
    tc, store = env
    cid = _campaign(tc, "a-write")
    r = _submit(tc, "a-write", cid)
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert r.headers["location"] == f"/api/jobs/{job['id']}"
    assert job["state"] == "queued" and job["error"] is None and job["campaign_id"] == cid
    assert "input_path" not in job and "params" not in job and "runner_id" not in job

    status = tc.get(f"/api/jobs/{job['id']}", headers=H("a-read"))
    assert status.status_code == 200 and status.json()["job"]["state"] == "queued"

    done = _drain(_runner(store), store, "acme", job["id"])
    body = tc.get(f"/api/jobs/{job['id']}", headers=H("a-read")).json()["job"]
    assert body["state"] == done["state"] == "succeeded"
    assert body["campaign_generation"] is not None
    analysis = tc.get(f"/api/campaigns/{cid}/analysis", headers=H("a-read")).json()
    assert analysis["has_data"] is True and analysis["marker"] == "api-stub"
    assert analysis["campaign_generation"] == body["campaign_generation"]


def test_job_list_is_newest_first_and_limited(env):
    tc, _store = env
    cid = _campaign(tc, "a-write")
    ids = [_submit(tc, "a-write", cid).json()["job"]["id"] for _ in range(3)]
    listed = tc.get(f"/api/campaigns/{cid}/jobs?limit=2", headers=H("a-read")).json()["jobs"]
    assert [j["id"] for j in listed] == list(reversed(ids))[:2]
    assert tc.get(f"/api/campaigns/{cid}/jobs?limit=0", headers=H("a-read")).status_code == 400


def test_a_bad_sheet_is_refused_synchronously_and_queues_nothing(env):
    tc, store = env
    cid = _campaign(tc, "a-write")
    r = _submit(tc, "a-write", cid, target="no_such_column")
    assert r.status_code == 400, r.text
    assert "code" in r.json() or "error" in r.json()
    assert jobs.list_jobs(store, tenant="acme", campaign_id=cid) == []
    assert not list(store.state_dir.glob("jobs/*/*.bin"))


def test_cancel_a_queued_job_over_http_is_immediate_and_idempotent(env):
    tc, store = env
    cid = _campaign(tc, "a-write")
    job = _submit(tc, "a-write", cid).json()["job"]
    r = tc.post(f"/api/jobs/{job['id']}/cancel", headers=H("a-write"))
    assert r.status_code == 200 and r.json()["job"]["state"] == "cancelled"
    again = tc.post(f"/api/jobs/{job['id']}/cancel", headers=H("a-write"))
    assert again.status_code == 200 and again.json()["job"]["state"] == "cancelled"
    assert tc.get(f"/api/jobs/{job['id']}", headers=H("a-read")).json()["job"]["error"] is None


# --- limits ------------------------------------------------------------------------ #


def test_queue_depth_answers_429_with_retry_after(env, monkeypatch):
    tc, _store = env
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "1")
    cid = _campaign(tc, "a-write")
    assert _submit(tc, "a-write", cid).status_code == 202
    r = _submit(tc, "a-write", cid)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) == busy.RETRY_AFTER_SECONDS
    assert "queued or running" in r.json()["error"]


def test_upload_over_the_tenant_limit_is_413(env, monkeypatch):
    tc, store = env
    monkeypatch.setenv("KALOS_JOB_MAX_UPLOAD_BYTES", "1000")
    cid = _campaign(tc, "a-write")
    r = _submit(tc, "a-write", cid, data=_csv(_df(200)))
    assert r.status_code == 413
    assert jobs.list_jobs(store, tenant="acme", campaign_id=cid) == []


def test_upload_over_the_engine_cap_is_413_before_the_route(env):
    tc, _store = env
    cid = _campaign(tc, "a-write")
    r = _submit(tc, "a-write", cid, data=b"0" * (MAX_UPLOAD_BYTES + 2 * 1024 * 1024))
    assert r.status_code == 413


def test_jobs_disabled_answers_503_and_status_still_reads(env, monkeypatch):
    tc, store = env
    cid = _campaign(tc, "a-write")
    job = _submit(tc, "a-write", cid).json()["job"]
    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    r = _submit(tc, "a-write", cid)
    assert r.status_code == 503 and "KALOS_JOBS_ENABLED=0" in r.json()["error"]
    assert tc.get(f"/api/jobs/{job['id']}", headers=H("a-read")).status_code == 200


def test_a_malformed_job_setting_is_a_503_not_a_crash(env, monkeypatch):
    tc, _store = env
    cid = _campaign(tc, "a-write")
    monkeypatch.setenv("KALOS_JOB_TIMEOUT_S", "soon")
    r = _submit(tc, "a-write", cid)
    assert r.status_code == 503


# --- scopes and auth ------------------------------------------------------------------ #


def test_read_scope_cannot_submit_or_cancel(env):
    tc, store = env
    cid = _campaign(tc, "a-write")
    assert _submit(tc, "a-read", cid).status_code == 403
    job = _submit(tc, "a-write", cid).json()["job"]
    assert tc.post(f"/api/jobs/{job['id']}/cancel", headers=H("a-read")).status_code == 403
    assert jobs.get_job(store, tenant="acme", job_id=job["id"])["state"] == "queued"


def test_anonymous_job_upload_is_401_before_the_body_is_read(env):
    tc, _store = env
    cid = _campaign(tc, "a-write")
    r = tc.post(f"/api/campaigns/{cid}/jobs", files={"file": ("x.csv", b"x" * 64, "text/csv")})
    assert r.status_code == 401
    assert r.headers.get("www-authenticate") == "Bearer"


def test_unknown_token_is_401(env):
    tc, _store = env
    cid = _campaign(tc, "a-write")
    assert _submit(tc, "not-a-token", cid).status_code == 401


# --- cross-tenant isolation ------------------------------------------------------------- #


def _globex_job(tc: TestClient, store: CampaignStore) -> tuple[str, str]:
    cid = _campaign(tc, "b-admin", GLOBEX_SECRET)
    job = _submit(tc, "b-write", cid, name=f"{GLOBEX_SECRET}.csv").json()["job"]
    return cid, job["id"]


def test_another_tenants_job_is_not_found_and_byte_identical_to_a_missing_one(env):
    tc, store = env
    _cid, job_id = _globex_job(tc, store)
    missing = "job_" + uuid.uuid4().hex
    foreign = tc.get(f"/api/jobs/{job_id}", headers=H("a-admin"))
    absent = tc.get(f"/api/jobs/{missing}", headers=H("a-admin"))
    malformed = tc.get("/api/jobs/job_NOT-A-JOB", headers=H("a-admin"))
    for r in (foreign, absent, malformed):
        assert r.status_code == 404
        assert r.json() == {"error": "job not found"}
        assert r.content == foreign.content
    assert GLOBEX_SECRET not in foreign.text


def test_another_tenants_job_cannot_be_cancelled_and_is_left_untouched(env):
    tc, store = env
    _cid, job_id = _globex_job(tc, store)
    before = jobs.get_job(store, tenant="globex", job_id=job_id)
    r_foreign = tc.post(f"/api/jobs/{job_id}/cancel", headers=H("a-admin"))
    r_missing = tc.post(f"/api/jobs/job_{uuid.uuid4().hex}/cancel", headers=H("a-admin"))
    assert r_foreign.status_code == r_missing.status_code == 404
    assert r_foreign.content == r_missing.content
    assert jobs.get_job(store, tenant="globex", job_id=job_id) == before
    assert before["state"] == "queued" and before["cancel_requested"] is False


def test_a_running_foreign_job_cannot_be_cancelled_either(env):
    tc, store = env
    _cid, job_id = _globex_job(tc, store)
    jobs.claim(store, tenant="globex", runner_id="t")  # now running
    r = tc.post(f"/api/jobs/{job_id}/cancel", headers=H("a-admin"))
    assert r.status_code == 404
    assert jobs.get_job(store, tenant="globex", job_id=job_id)["cancel_requested"] is False


def test_jobs_cannot_be_submitted_to_or_listed_from_another_tenants_campaign(env):
    tc, store = env
    gcid, job_id = _globex_job(tc, store)
    submit = _submit(tc, "a-admin", gcid)
    missing = _submit(tc, "a-admin", "cmp_" + uuid.uuid4().hex)
    assert submit.status_code == 404 and submit.content == missing.content
    listed = tc.get(f"/api/campaigns/{gcid}/jobs", headers=H("a-admin"))
    absent = tc.get(f"/api/campaigns/cmp_{uuid.uuid4().hex}/jobs", headers=H("a-admin"))
    assert listed.status_code == 404 and listed.content == absent.content
    assert GLOBEX_SECRET not in listed.text
    # and nothing was written under globex
    assert [j["id"] for j in jobs.list_jobs(store, tenant="globex", campaign_id=gcid)] == [job_id]


def test_job_listings_and_audit_rows_are_scoped_to_the_caller(env):
    tc, store = env
    acme_cid = _campaign(tc, "a-write")
    acme_job = _submit(tc, "a-write", acme_cid).json()["job"]["id"]
    gcid, globex_job = _globex_job(tc, store)

    acme_list = tc.get(f"/api/campaigns/{acme_cid}/jobs", headers=H("a-read")).json()["jobs"]
    assert [j["id"] for j in acme_list] == [acme_job]
    assert globex_job not in json.dumps(acme_list)
    acme_audit = store.audit_entries(tenant="acme", limit=500)
    globex_audit = store.audit_entries(tenant="globex", limit=500)
    assert any(e["job_id"] == acme_job and e["action"] == "job.submit" for e in acme_audit)
    assert all(e["job_id"] != acme_job for e in globex_audit)
    assert all(e["job_id"] != globex_job for e in acme_audit)
    assert GLOBEX_SECRET not in json.dumps(acme_audit)


def test_each_tenants_job_publishes_only_into_its_own_campaign(env):
    """The scheduler serves every tenant, but a job runs as its own row: its
    publish is scoped by that row's tenant and campaign, so two tenants' jobs
    running side by side never write into each other's campaign."""
    tc, store = env
    gcid, globex_job = _globex_job(tc, store)
    acme_cid = _campaign(tc, "a-write")
    acme_job = _submit(tc, "a-write", acme_cid, name="acme-runs.csv").json()["job"]["id"]
    runner = _runner(store)
    assert _drain(runner, store, "acme", acme_job)["state"] == "succeeded"
    assert _drain(runner, store, "globex", globex_job)["state"] == "succeeded"
    acme_analysis = store.load_analysis(tenant="acme", campaign_id=acme_cid)
    globex_analysis = store.load_analysis(tenant="globex", campaign_id=gcid)
    assert acme_analysis["dataset"] == "acme-runs.csv"
    assert globex_analysis["dataset"].startswith(GLOBEX_SECRET)
    assert store.load_analysis(tenant="acme", campaign_id=acme_cid) is not None
    assert GLOBEX_SECRET not in json.dumps(acme_analysis)
    assert store.get(tenant="acme", campaign_id=acme_cid)["base_rows"]


# --- the app lifespan starts and stops the runner -------------------------------------- #


def test_lifespan_starts_and_stops_the_runner_unless_disabled(env, monkeypatch):
    from kalos.portal import job_runner

    tc, _store = env
    with TestClient(app):
        assert job_runner._runner is not None
        assert job_runner._runner.runner_id
    assert job_runner._runner is None  # stopped on shutdown

    monkeypatch.setenv("KALOS_JOBS_ENABLED", "0")
    with TestClient(app):
        assert job_runner._runner is None


def test_lifespan_refuses_to_boot_on_a_malformed_job_setting(env, monkeypatch):
    tc, _store = env
    monkeypatch.setenv("KALOS_JOB_QUEUE_DEPTH", "-1")
    with pytest.raises(ValueError, match="KALOS_JOB_QUEUE_DEPTH"):
        with TestClient(app):
            pass
