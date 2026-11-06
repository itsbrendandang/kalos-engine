"""`/api/assist/*` over HTTP with a fake model: citation enforcement,
abstention, provider=none, scopes, audit, hooks, and tenant isolation."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.assist.index import AssistIndex
from kalos.assist.llm import ModelUnavailable, get_chat_model
from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store

A_SECRET = "AcmeSecretZeus"
B_SECRET = "GlobexSecretHera"


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme", "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
    {"token_sha256": _sha("b-write"), "subject": "gina@globex", "tenant": "globex", "scopes": ["read", "write"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


ANALYSIS = {
    "target": "lipase_titer",
    "features": ["Methanol", "pH"],
    "n": 8,
    "timestamp": 100,
    "provenance": {"input_sha256": "cd" * 32},
    "validation": {"findings": []},
    "promotion": {"passed": True, "failures": []},
    "drivers": [{"name": "Methanol", "rho": 0.62, "ci95": [0.2, 0.9]}],
    "proposals": [{"recipe": {"Methanol": 2.5, "pH": 6.1}, "pred": 3.3, "std": 0.4}],
}


class FakeModel:
    provider = "fake"
    model = "fake-1"
    local = True

    def __init__(self) -> None:
        self.reply = ""
        self.prompts: list[str] = []
        self.fail = False

    def label_of(self, prompt: str, needle: str) -> str:
        for block in prompt.split("\n\n"):
            m = re.match(r"\[(S\d+)\]", block)
            if m and needle in block:
                return m.group(1)
        raise AssertionError(f"no source with {needle!r} in the prompt")

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise ModelUnavailable("down")
        return self.reply(prompt) if callable(self.reply) else self.reply


def _df(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"Methanol": rng.uniform(0, 4, n).round(3), "pH": rng.uniform(5, 7, n).round(2),
                         "lipase_titer": rng.uniform(1, 3, n).round(3)})


def _seed(store: CampaignStore, tenant: str, name: str, *, start: bool = False) -> str:
    cid = store.create_campaign(tenant=tenant, name=name, created_by="t")["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant=tenant, campaign_id=cid,
               analysis={**ANALYSIS, "marker": name}, dataset="runs.csv")
    if start:
        store.start([{"recipe": {"Methanol": 2.5, "pH": 6.1}}], tenant=tenant, campaign_id=cid)
    return cid


@pytest.fixture
def env(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CAMPAIGN_LOOP.md").write_text("# Loop\n\nRecord a result for every started run.\n")
    monkeypatch.setenv("KALOS_ASSIST_DOCS_DIR", str(docs))
    monkeypatch.setenv("KALOS_ASSIST_EMBED", "0")
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    store = CampaignStore(tmp_path / "state")
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    fake = FakeModel()
    app.dependency_overrides[get_campaign_store] = lambda: store
    app.dependency_overrides[get_chat_model] = lambda: fake
    try:
        yield TestClient(app), store, fake
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_chat_model, None)


# --- /next ----------------------------------------------------------------------- #


def test_next_without_any_campaign_is_upload(env):
    tc, _store, _fake = env
    r = tc.get("/api/assist/next", headers=H("a-read"))
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "no_sheet" and body["action_route"] == "/campaign/setup"
    assert body["campaign_id"] is None and body["blockers"] == []
    assert body["why_it_matters"]


def test_next_for_a_campaign_and_foreign_ids_are_the_shared_404(env):
    tc, store, _fake = env
    cid = _seed(store, "acme", A_SECRET, start=True)
    b_cid = _seed(store, "globex", B_SECRET)
    r = tc.get(f"/api/assist/next?campaign_id={cid}", headers=H("a-read"))
    assert r.status_code == 200 and r.json()["state"] == "runs_awaiting"
    assert r.json()["action_route"] == "/campaign/record"
    foreign = tc.get(f"/api/assist/next?campaign_id={b_cid}", headers=H("a-read"))
    missing = tc.get("/api/assist/next?campaign_id=cmp_" + "0" * 32, headers=H("a-read"))
    malformed = tc.get("/api/assist/next?campaign_id=../../etc", headers=H("a-read"))
    assert foreign.status_code == missing.status_code == malformed.status_code == 404
    assert foreign.content == missing.content == malformed.content
    assert B_SECRET not in foreign.text


def test_next_needs_a_token(env):
    tc, _store, _fake = env
    assert tc.get("/api/assist/next").status_code == 401


# --- /ask ------------------------------------------------------------------------ #


def test_ask_keeps_only_cited_sentences_and_returns_sources(env):
    tc, store, fake = env
    cid = _seed(store, "acme", A_SECRET)

    def reply(prompt: str) -> str:
        s = fake.label_of(prompt, "moved with lipase_titer")
        return (f"Methanol moved with lipase_titer in this campaign [{s}]. "
                "Methanol is the setting to push hardest. "
                f"Its rank association is 0.62 [{s}].")

    fake.reply = reply
    r = tc.post("/api/assist/ask", json={"question": "Which inputs moved with lipase titer?",
                                         "campaign_id": cid}, headers=H("a-read"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abstained"] is False
    assert "push hardest" not in body["answer"]  # no source, dropped
    assert "0.62" in body["answer"]
    assert "[S" not in body["answer"]
    assert body["sources"] == [{"kind": "analysis", "id": "drivers", "campaign_id": cid,
                                "quote": body["sources"][0]["quote"]}]
    chunk = next(c for c in AssistIndex(store, "acme").chunks() if c.source_id == "drivers")
    assert body["sources"][0]["quote"] in chunk.text
    assert body["next"]["state"] == "batch_proposed"


def test_ask_drops_a_number_the_source_does_not_hold(env):
    tc, store, fake = env
    cid = _seed(store, "acme", A_SECRET)

    def reply(prompt: str) -> str:
        s = fake.label_of(prompt, "moved with lipase_titer")
        return f"Methanol has a rank association of 0.97 [{s}]."

    fake.reply = reply
    body = tc.post("/api/assist/ask", json={"question": "methanol lipase titer association", "campaign_id": cid},
                   headers=H("a-read")).json()
    assert body["abstained"] is True and body["sources"] == []


@pytest.mark.parametrize("bad", ["Raising Methanol will raise lipase_titer", "This recipe is optimal",
                                 "Methanol is validated as the driver"])
def test_ask_rejects_claim_and_banned_words_and_abstains_with_the_guide(env, bad):
    tc, store, fake = env
    cid = _seed(store, "acme", A_SECRET)

    def reply(prompt: str) -> str:
        s = fake.label_of(prompt, "moved with lipase_titer")
        return f"Methanol moved with lipase_titer [{s}]. {bad} [{s}]."

    fake.reply = reply
    body = tc.post("/api/assist/ask", json={"question": "what moved with lipase titer", "campaign_id": cid},
                   headers=H("a-read")).json()
    assert body["abstained"] is True
    assert body["sources"] == []
    assert body["answer"] == f"{body['next']['step']}. {body['next']['why']}"


def test_ask_abstains_without_calling_the_model_when_nothing_is_relevant(env):
    tc, store, fake = env
    cid = _seed(store, "acme", A_SECRET)
    body = tc.post("/api/assist/ask", json={"question": "zebra saxophone quantum", "campaign_id": cid},
                   headers=H("a-read")).json()
    assert body["abstained"] is True and body["sources"] == []
    assert body["next"]["state"] == "batch_proposed"
    assert fake.prompts == []


def test_ask_model_down_is_503(env):
    tc, store, fake = env
    _seed(store, "acme", A_SECRET)
    fake.fail = True
    r = tc.post("/api/assist/ask", json={"question": "which inputs moved with lipase titer"}, headers=H("a-read"))
    assert r.status_code == 503 and r.json()["code"] == "assist_provider_unavailable"


def test_provider_none_turns_off_ask_only(env, monkeypatch):
    tc, store, _fake = env
    _seed(store, "acme", A_SECRET)
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    app.dependency_overrides.pop(get_chat_model, None)
    r = tc.post("/api/assist/ask", json={"question": "anything"}, headers=H("a-read"))
    assert r.status_code == 404 and r.json()["code"] == "assist_disabled"
    assert tc.get("/api/assist/next", headers=H("a-read")).status_code == 200
    status = tc.get("/api/assist/index/status", headers=H("a-read")).json()
    assert status["provider"] == "none" and status["local"] is True


def test_ask_body_is_bounded(env):
    tc, _store, _fake = env
    assert tc.post("/api/assist/ask", json={"question": ""}, headers=H("a-read")).status_code == 422
    assert tc.post("/api/assist/ask", json={"question": "x" * 2001}, headers=H("a-read")).status_code == 422
    assert tc.post("/api/assist/ask", json={"question": "x", "tenant": "globex"},
                   headers=H("a-read")).status_code == 422


def test_ask_never_sees_another_tenants_data(env):
    tc, store, fake = env
    _seed(store, "acme", A_SECRET)
    b_cid = _seed(store, "globex", B_SECRET)
    AssistIndex(store, "globex").refresh()
    fake.reply = "Nothing here [G]."
    tc.post("/api/assist/ask", json={"question": f"{B_SECRET} lipase titer proposed batch"}, headers=H("a-read"))
    assert fake.prompts and all(B_SECRET not in p.split("Question:")[0] for p in fake.prompts)
    r = tc.post("/api/assist/ask", json={"question": "lipase", "campaign_id": b_cid}, headers=H("a-read"))
    assert r.status_code == 404 and r.json()["code"] == "not_found"


# --- index routes ------------------------------------------------------------------ #


def test_index_status_and_rebuild(env):
    tc, store, _fake = env
    _seed(store, "acme", A_SECRET)
    status = tc.get("/api/assist/index/status", headers=H("a-read")).json()
    assert set(status) == {"provider", "model", "local", "chunks", "built_at", "building", "retriever"}
    assert status["provider"] == "ollama" and status["local"] is True
    assert status["chunks"] == 0 and status["built_at"] is None
    assert tc.post("/api/assist/index/rebuild", headers=H("a-read")).status_code == 403
    r = tc.post("/api/assist/index/rebuild", headers=H("a-write"))
    assert r.status_code == 200
    assert r.json()["chunks"] > 0 and r.json()["built_at"]
    actions = [e["action"] for e in store.audit_entries(tenant="acme")]
    assert "assist.index.rebuild" in actions
    assert "assist.index.rebuild" not in [e["action"] for e in store.audit_entries(tenant="globex")]
    # the other tenant's index is untouched
    assert tc.get("/api/assist/index/status", headers=H("b-write")).json()["chunks"] == 0


# --- hooks ------------------------------------------------------------------------- #


def test_refused_campaign_upload_becomes_the_guide_step(env):
    tc, store, _fake = env
    cid = store.create_campaign(tenant="acme", name="empty", created_by="t")["id"]
    csv = b"Methanol,lipase_titer\n1,2\n2,3\n"
    r = tc.post(f"/api/campaigns/{cid}/upload", files={"file": ("runs.csv", csv, "text/csv")},
                headers=H("a-write"))
    assert r.status_code == 400
    code = r.json()["code"]
    nxt = tc.get(f"/api/assist/next?campaign_id={cid}", headers=H("a-read")).json()
    assert nxt["state"] == "fit_refused"
    assert nxt["blockers"][0]["code"] == code
    # the refusal note is the assistant's file, not campaign state
    assert store.get(tenant="acme", campaign_id=cid) is None


def test_recorded_result_refreshes_the_index(env):
    tc, store, _fake = env
    cid = _seed(store, "acme", A_SECRET, start=True)
    AssistIndex(store, "acme").refresh()
    run_id = store.summary(tenant="acme", campaign_id=cid)["pending"][0]["id"]
    r = tc.post(f"/api/campaigns/{cid}/runs/{run_id}/result", json={"value": 2.75}, headers=H("a-write"))
    assert r.status_code == 200, r.text
    rounds = [c for c in AssistIndex(store, "acme").chunks() if c.kind == "round"]
    assert any("Measured lipase_titer 2.75" in c.text for c in rounds)
