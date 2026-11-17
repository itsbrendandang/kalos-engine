"""Round review uses fake models only; no network."""
from __future__ import annotations

import pytest
from test_assist_routes import H, env  # noqa: F401

from kalos.assist.index import Chunk, Hit
from kalos.assist.llm import AnthropicChat, ModelUnavailable
from kalos.assist.swarm import ROLES, group_model, review


class Model:
    provider = "fake"
    model = "review-test"
    local = True

    def __init__(self, critic="C1: ok: matches", planner="C1: Measured yield is 12 [S1].",
                 reporter="Measured yield is 12 [S1]."):
        self.replies = [planner, critic, reporter]
        self.prompts = []

    def complete(self, system, prompt):
        self.prompts.append((system, prompt))
        return self.replies[len(self.prompts) - 1]


def run(tmp_path, model, hits=None, tenant="a"):
    if hits is None:
        hits = [Hit(Chunk("c", "doc", "d", None, "Yield", "Measured yield is 12."), 1.0)]
    return review("private planner question", hits, {"step": "Upload a sheet", "why": "No sheet yet"}, model,
                  tmp_path, tenant)


@pytest.mark.parametrize(("severity", "verdict"), [
    ("ok", "supported"), ("minor", "supported"), ("major", "contested"),
    ("blocking", "rejected"), ("nonsense", "unreviewed"),
])
def test_verdicts(tmp_path, severity, verdict):
    result = run(tmp_path, Model(critic=f"C1: {severity}: reason"))
    assert result["claims"][0]["verdict"] == verdict
    assert result["abstained"] == (verdict != "supported")


def test_isolation_and_replay(tmp_path):
    model = Model()
    first = run(tmp_path, model)
    assert run(tmp_path, model) == first
    assert len(model.prompts) == 3
    assert "private planner question" not in model.prompts[1][1]
    assert model.prompts[0][1] not in model.prompts[1][1]
    assert "private planner question" not in str(first["ledger"])
    run(tmp_path, Model(), tenant="b")
    assert len(list(tmp_path.glob("assist/*/swarm/*.json"))) == 6


@pytest.mark.parametrize("text", ["No citation.", "Measured yield is 999 [S1].",
                                  "This guarantees yield [S1].", "This is approved [S1]."])
def test_rejected(tmp_path, text):
    model = Model(planner="C1: " + text)
    result = run(tmp_path, model)
    assert result["claims"][0]["verdict"] == "rejected"
    assert result["claims"][0]["issues"]
    assert len(model.prompts) == 1


def test_empty_and_report_fallback(tmp_path):
    model = Model()
    assert run(tmp_path, model, hits=[])["abstained"]
    assert not model.prompts
    assert run(tmp_path, Model(planner="nothing parseable"))["abstained"]
    result = run(tmp_path / "fallback", Model(reporter="Invented 999 [S1]."))
    assert result["report"] == "Measured yield is 12."
    assert not result["abstained"]


def test_group_override(monkeypatch):
    monkeypatch.setenv("KALOS_ASSIST_MODEL_CRITIC", "critic-test")
    assert group_model(ROLES[2], AnthropicChat("normal")).model == "critic-test"
    assert group_model(ROLES[1], AnthropicChat("normal")).model == "normal"


def test_routes(env, monkeypatch):  # noqa: F811 - imported pytest fixture
    from kalos.assist import ratelimit

    ratelimit.reset()
    client, _, fake = env
    monkeypatch.setattr("kalos.assist.routes.AssistIndex.search", lambda *a, **kw: [
        Hit(Chunk("d", "doc", "doc", None, "Loop",
                  "Record a result for every started run."), 1.0)])
    roles = client.get("/api/assist/review/roles", headers=H("a-read"))
    assert roles.status_code == 200
    assert [r["key"] for r in roles.json()] == [r.key for r in ROLES]
    def reply(prompt):
        if "Question:" in prompt:
            return "C1: Record a result for every started run [S1]."
        if "Supported claims:" in prompt:
            return "Record a result for every started run [S1]."
        return "C1: ok: matches source"

    fake.reply = reply
    response = client.post("/api/assist/review", headers=H("a-read"),
                           json={"question": "Record result started run"})
    assert response.status_code == 200
    assert {"claims", "report", "abstained", "ledger", "next"} == set(response.json())
    assert response.json()["claims"][0]["verdict"] == "supported"
    assert client.post("/api/assist/review", headers=H("a-read"),
                       json={"campaign_id": "bad"}).status_code == 404
    fake.fail = True
    response = client.post("/api/assist/review", headers=H("a-read"),
                           json={"question": "Record results for started runs"})
    assert response.status_code == 503
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    response = client.post("/api/assist/review", headers=H("a-read"), json={})
    assert response.status_code == 404
    assert response.json()["code"] == "assist_disabled"
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("KALOS_ASSIST_ASK_PER_MINUTE", "1")
    ratelimit.reset()
    assert client.post("/api/assist/review", headers=H("a-read"),
                       json={"campaign_id": "bad"}).status_code == 404
    limited = client.post("/api/assist/review", headers=H("a-read"), json={})
    assert limited.status_code == 429 and "Retry-After" in limited.headers
    ratelimit.reset()


def test_only_surviving_sources_reach_critic_and_reporter(tmp_path):
    model = Model(planner="C1: Measured yield is 12 [S1].\nC2: Invented 999 [S2].",
                  reporter="Other source [S2].")
    hits = [Hit(Chunk("a", "doc", "a", None, "Yield", "Measured yield is 12."), 1.0),
            Hit(Chunk("b", "doc", "b", None, "Other", "Hidden rejected source."), 0.5)]
    result = run(tmp_path, model, hits)
    assert "Hidden rejected source" not in model.prompts[1][1]
    assert "Hidden rejected source" not in model.prompts[2][1]
    assert result["report"] == "Measured yield is 12."


@pytest.mark.parametrize("critic", ["", "bad line", "C1: ok: yes\nC1: blocking: no"])
def test_missing_or_ambiguous_judgement(tmp_path, critic):
    result = run(tmp_path, Model(critic=critic))
    assert result["claims"][0]["verdict"] == "unreviewed"


def test_model_error_not_cached(tmp_path):
    class Broken(Model):
        def complete(self, system, prompt):
            raise ModelUnavailable("offline")

    with pytest.raises(ModelUnavailable):
        run(tmp_path, Broken())
    assert not list(tmp_path.rglob("*.json"))
