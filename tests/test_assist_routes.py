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
        s = fake.label_of(prompt, "went with lipase_titer")
        return (f"Methanol went with lipase_titer in this campaign [{s}]. "
                "Methanol is the setting to push hardest. "
                f"Its rank association is 0.62 [{s}].")

    fake.reply = reply
    r = tc.post("/api/assist/ask", json={"question": "Which inputs went with lipase titer?",
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
    assert body["next"]["state"] == "recipes_proposed"


def test_ask_drops_a_number_the_source_does_not_hold(env):
    tc, store, fake = env
    cid = _seed(store, "acme", A_SECRET)

    def reply(prompt: str) -> str:
        s = fake.label_of(prompt, "went with lipase_titer")
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
        s = fake.label_of(prompt, "went with lipase_titer")
        return f"Methanol went with lipase_titer [{s}]. {bad} [{s}]."

    fake.reply = reply
    body = tc.post("/api/assist/ask", json={"question": "what went with lipase titer", "campaign_id": cid},
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
    assert body["next"]["state"] == "recipes_proposed"
    assert fake.prompts == []


def test_ask_model_down_is_503(env):
    tc, store, fake = env
    _seed(store, "acme", A_SECRET)
    fake.fail = True
    r = tc.post("/api/assist/ask", json={"question": "which inputs went with lipase titer"}, headers=H("a-read"))
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
    tc.post("/api/assist/ask", json={"question": f"{B_SECRET} lipase titer proposed recipes"}, headers=H("a-read"))
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
    assert status["chunks"] > 0 and status["built_at"]  # built on first look
    assert tc.post("/api/assist/index/rebuild", headers=H("a-read")).status_code == 403
    r = tc.post("/api/assist/index/rebuild", headers=H("a-write"))
    assert r.status_code == 200
    assert r.json()["chunks"] > 0 and r.json()["built_at"]
    actions = [e["action"] for e in store.audit_entries(tenant="acme")]
    assert "assist.index.rebuild" in actions
    assert "assist.index.rebuild" not in [e["action"] for e in store.audit_entries(tenant="globex")]
    # the other tenant's index holds none of acme's records
    assert tc.get("/api/assist/index/status", headers=H("b-write")).status_code == 200
    assert A_SECRET not in AssistIndex(store, "globex").path.read_text()


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


def test_labels_after_the_full_stop_belong_to_that_sentence_only():
    from kalos.assist.chat import split_sentences

    parts = split_sentences("First claim here. Second one [S1]. Third one. [S2][G]")
    assert parts == ["First claim here.", "Second one [S1].", "Third one. [S2][G]"]


def test_a_guide_only_citation_must_restate_the_guide():
    from kalos.assist.chat import check_answer

    guide = {"step": "Record 2 results for round 1", "why": "1 of 3 started runs have a result.",
             "why_it_matters": "Every result you enter sharpens the next suggestions."}
    planted = "Methanol at 4 is the recipe to push hardest and it is safe [G]."
    checked = check_answer(planted, [], guide)
    assert checked.answer == "" and checked.dropped == 1
    ok = check_answer("Record 2 results for round 1 [G].", [], guide)
    assert ok.answer == "Record 2 results for round 1."


@pytest.mark.parametrize("planted", ["This recipe was approved [S].", "The run sheet passed validation [S].",
                                     "The result was signed [S].", "Higher Methanol increases lipase_titer [S].",
                                     "Methanol drives lipase_titer [S].", "Methanol leads to more lipase_titer [S].",
                                     "The rise is due to Methanol [S].", "It rose because of Methanol [S]."])
def test_review_banned_and_claim_words_abstain(env, planted):
    tc, store, fake = env
    cid = _seed(store, "acme", A_SECRET)

    def reply(prompt: str) -> str:
        s = fake.label_of(prompt, "went with lipase_titer")
        return planted.replace("[S]", f"[{s}]")

    fake.reply = reply
    body = tc.post("/api/assist/ask", json={"question": "what went with lipase titer", "campaign_id": cid},
                   headers=H("a-read")).json()
    assert body["abstained"] is True, planted


def test_claim_words_in_lab_written_text_support_nothing(env):
    """A campaign named with a claim word must not license that claim."""
    tc, store, fake = env
    cid = _seed(store, "acme", "Methanol causes titer")

    def reply(prompt: str) -> str:
        s = fake.label_of(prompt, "went with lipase_titer")
        return f"Methanol causes lipase_titer [{s}]."

    fake.reply = reply
    body = tc.post("/api/assist/ask", json={"question": "what went with lipase titer", "campaign_id": cid},
                   headers=H("a-read")).json()
    assert body["abstained"] is True

    def mention(prompt: str) -> str:  # naming the campaign is fine: its words are the lab's
        s = fake.label_of(prompt, "went with lipase_titer")
        return f"In campaign Methanol causes titer, Methanol went with lipase_titer [{s}]."

    fake.reply = mention
    body = tc.post("/api/assist/ask", json={"question": "what went with lipase titer", "campaign_id": cid},
                   headers=H("a-read")).json()
    assert body["abstained"] is False


def test_tenant_skill_text_supports_no_claim(tmp_path):
    from kalos.assist.index import skill_chunks

    d = tmp_path / "team" / "my-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("# Notes\n\nHigher feed will raise titer.\n")
    chunk = skill_chunks([d / "SKILL.md"])[0]
    assert chunk.trusted_text is False and chunk.kalos_text() == ""


def test_numbers_exponents_and_list_positions():
    from kalos.assist.chat import Labelled, check_answer
    from kalos.assist.index import Chunk, Hit

    chunk = Chunk("c1", "analysis", "proposals", "cmp_" + "1" * 32, "Proposed recipes",
                  "Recipe 3: Methanol = 2.5; Predicted lipase_titer 3.3")
    lab = [Labelled("S1", Hit(chunk, 1.0))]
    guide = {"step": "x", "why": "y", "why_it_matters": "z"}
    assert check_answer("The titer reached 2e9 [S1].", lab, guide).answer == ""  # exponent parsed, not in source
    assert check_answer("It was the best of 7 runs [S1].", lab, guide).answer == ""  # 7 is a value, not a position
    kept = check_answer("Recipe 3 has Predicted lipase_titer 3.3 [S1].", lab, guide).answer
    assert kept == "Recipe 3 has Predicted lipase_titer 3.3."
    big = Chunk("c2", "analysis", "noise", None, "spread", "Measured spread 2000000000")
    ok = check_answer("Measured spread was 2e9 [S1].", [Labelled("S1", Hit(big, 1.0))], guide).answer
    assert ok == "Measured spread was 2e9."


def test_ask_is_rate_limited_per_principal(env, monkeypatch):
    """sec-assist (i): a burst of questions from one user gets 429 + Retry-After;
    another user is not affected."""
    from kalos.assist import ratelimit

    tc, store, _fake = env
    ratelimit.reset()
    monkeypatch.setenv("KALOS_ASSIST_ASK_PER_MINUTE", "3")
    cid = _seed(store, "acme", A_SECRET)
    body = {"question": "zebra saxophone quantum", "campaign_id": cid}
    codes = [tc.post("/api/assist/ask", json=body, headers=H("a-read")).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = tc.post("/api/assist/ask", json=body, headers=H("a-read"))
    assert r.json()["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
    assert tc.post("/api/assist/ask", json=body, headers=H("a-write")).status_code == 200
    ratelimit.reset()


def test_rate_window_slides():
    from kalos.assist import ratelimit

    ratelimit.reset()
    import os

    os.environ["KALOS_ASSIST_ASK_PER_MINUTE"] = "2"
    try:
        assert ratelimit.check("t", "s", now=0.0) == 0
        assert ratelimit.check("t", "s", now=1.0) == 0
        assert ratelimit.check("t", "s", now=2.0) == 58
        assert ratelimit.check("t", "s", now=61.0) == 0
    finally:
        del os.environ["KALOS_ASSIST_ASK_PER_MINUTE"]
        ratelimit.reset()


def test_read_routes_do_not_rewrite_an_unchanged_index(env):
    tc, store, _fake = env
    cid = _seed(store, "acme", A_SECRET)
    tc.get("/api/assist/index/status", headers=H("a-read"))  # first look builds it
    path = AssistIndex(store, "acme").path
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    tc.get("/api/assist/index/status", headers=H("a-read"))
    tc.post("/api/assist/ask", json={"question": "zebra saxophone", "campaign_id": cid}, headers=H("a-read"))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_ask_on_a_real_analysis_shape_is_not_a_500(env):
    """sec-assist (and assist-web's report): real analyses carry `provenance` as the
    per-column LIST; /ask used to raise AttributeError at index.py:184."""
    tc, store, fake = env
    cid = store.create_campaign(tenant="acme", name="Real", created_by="t")["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid,
               analysis={**ANALYSIS, "provenance": [{"name": "Methanol", "status": "kept_feature"}],
                         "run": {"input_sha256": "ab" * 32}})
    fake.reply = "Nothing [G]."
    r = tc.post("/api/assist/ask", json={"question": "which inputs went with lipase titer", "campaign_id": cid},
                headers=H("a-read"))
    assert r.status_code == 200, r.text
    assert tc.get("/api/assist/index/status", headers=H("a-read")).json()["chunks"] > 0


@pytest.mark.parametrize("planted", ["Raising SP increases Titer [S1].", "Low pH leads to failure [S1].",
                                     "Recipe 1 is optimal [S1].", "It rose because of Methanol [S1]."])
def test_one_letter_lab_label_does_not_strip_letters_from_claims(planted):
    """sec-assist re-check (1): a text level 'A' used to strip every letter a."""
    from kalos.assist.chat import Labelled, check_answer
    from kalos.assist.index import Chunk, Hit, strip_terms

    assert strip_terms("Raising SP increases Titer", ["A"]) == "Raising SP increases Titer"
    assert strip_terms("Level A went with Titer", ["A"]).split() == ["Level", "went", "with", "Titer"]
    chunk = Chunk("c1", "analysis", "drivers", None, "t", "Higher SP went with higher Titer. Level A.",
                  user_terms=("A", "SP", "Titer"))
    checked = check_answer(planted, [Labelled("S1", Hit(chunk, 1.0))], {"step": "", "why": "", "why_it_matters": ""})
    assert checked.rejected, planted


def test_a_guide_only_sentence_adds_at_most_two_new_words():
    """sec-assist re-check nit: a long sentence could pass the 60% overlap while
    adding several words of its own."""
    from kalos.assist.chat import check_answer

    guide = {"step": "Record 2 results for round 1", "why": "1 of 3 started runs have a result.",
             "why_it_matters": "Every result you enter sharpens the next suggestions."}
    long_ = ("Record 2 results for round 1 since every started run result sharpens the next suggestions "
             "and Methanol looks safe here [G].")
    assert check_answer(long_, [], guide).answer == ""
    assert check_answer("Please record 2 results for round 1 [G].", [], guide).answer  # one new word


@pytest.mark.parametrize("term,planted", [("increases", "Methanol increases titer [S1]."),
                                          ("will", "Methanol will raise titer [S1]."),
                                          ("leads to", "Methanol leads to titer [S1]."),
                                          ("optimal", "Recipe 1 is optimal [S1].")])
def test_a_lab_term_that_is_a_claim_word_is_not_stripped(term, planted):
    """sec-assist final: a column or label named with a claim word must not hide
    that claim word in an answer."""
    from kalos.assist.chat import Labelled, check_answer
    from kalos.assist.index import Chunk, Hit, strip_terms

    assert term in strip_terms(f"x {term} y", [term], keep_claims=True)
    assert term not in strip_terms(f"x {term} y", [term])
    chunk = Chunk("c1", "analysis", "drivers", None, "t", f"Column {term}. Higher Methanol went with higher titer.",
                  user_terms=(term, "Methanol"))
    checked = check_answer(planted, [Labelled("S1", Hit(chunk, 1.0))], {"step": "", "why": "", "why_it_matters": ""})
    assert checked.rejected, planted


@pytest.mark.parametrize("word", ["safe", "skip", "approved", "ready"])
def test_a_guide_only_sentence_cannot_add_permission_words(word):
    from kalos.assist.chat import check_answer

    guide = {"step": "Record 2 results for round 1", "why": "1 of 3 started runs have a result.",
             "why_it_matters": "Every result you enter sharpens the next suggestions."}
    assert check_answer(f"Record 2 results for round 1, {word} [G].", [], guide).answer == ""
