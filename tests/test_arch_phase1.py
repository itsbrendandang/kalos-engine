"""Tests for Phase 1 architecture seams: unified providers, Cascade routing, and job hooks."""
from __future__ import annotations

from pathlib import Path

from kalos.assist import hooks
from kalos.portal import job_worker, jobs
from kalos.portal.campaign import CampaignStore
from kalos.portal.job_runner import JobRunner, ThreadWorker
from kalos.providers.grounding import raw_number_tokens
from kalos.providers.ollama_provider import generate, health_check


def test_raw_number_tokens_ignores_hex_record_ids() -> None:
    text = "Run run:2f30d1c7a9b8e4f1234567890abcdef1 measured 42.5 g/L at pH 7.0"
    assert raw_number_tokens(text) == {"42.5", "7.0"}


def test_ollama_provider_rejects_non_http_schemes() -> None:
    assert health_check("file:///etc/passwd") is False
    assert generate("ftp://localhost:11434", "qwen2.5:7b-instruct", "hi") is None


def test_job_runner_invokes_on_job_done_callback(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="acme", name="c", created_by="tester")["id"]
    csv_bytes = b"pH,Temp,Titer\n6.8,30,1.2\n7.0,32,2.4\n7.2,34,3.1\n6.9,31,1.8\n7.1,33,2.7\n7.3,35,3.5\n"
    job = jobs.enqueue(
        store,
        tenant="acme",
        campaign_id=cid,
        raw=csv_bytes,
        filename="runs.csv",
        target="Titer",
        anonymize=False,
        roles="",
        sheet=None,
        requested_by="tester",
        replace=True,
    )

    events: list[tuple[str, str | None, str, str]] = []

    def _on_done(_store: object, tenant: str, campaign_id: str | None, kind: str, state: str) -> None:
        events.append((tenant, campaign_id, kind, state))

    def _stub_analyze(*_a: object, **_k: object) -> dict[str, object]:
        return {
            "target": "Titer",
            "proposal_features": ["pH", "Temp"],
            "proposals": [],
            "n": 6,
            "d": 2,
        }

    runner = JobRunner(
        store,
        spawn=lambda jid, _t, _sd: ThreadWorker(
            lambda stop: job_worker.run_job(store, jid, analyze=_stub_analyze, should_stop=stop)
        ),
        poll_s=0.05,
        on_job_done=_on_done,
    )
    runner.tick()
    if job["id"] in runner._children:
        worker = runner._children[job["id"]].worker
        assert isinstance(worker, ThreadWorker)
        worker._thread.join(timeout=10.0)
        runner.tick()

    assert events == [("acme", cid, "analysis", "succeeded")]


def test_after_job_clears_refusal_on_succeeded_analysis(tmp_path: Path) -> None:
    from kalos.assist.state import latest_refusal, record_refusal

    store = CampaignStore(tmp_path)
    cid = store.create_campaign(tenant="acme", name="c", created_by="tester")["id"]
    record_refusal(store, "acme", cid, {"code": "too_few_rows"})
    assert latest_refusal(store, "acme", cid) is not None

    hooks.after_job(store, "acme", cid, kind="analysis", state="succeeded")
    assert latest_refusal(store, "acme", cid) is None
