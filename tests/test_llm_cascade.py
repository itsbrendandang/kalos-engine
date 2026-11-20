"""The fast -> quality -> offline model cascade (kalos.providers.cascade) and
the checks every reply passes against engine data before it is shown.

No Ollama here: every model is a fake object, and the normalize path's HTTP
calls are monkeypatched. CI has no Ollama.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from kalos.assist.chat import ask, build_prompt, check_answer
from kalos.assist.index import Chunk, Hit
from kalos.assist.llm import ModelUnavailable, OllamaChat, cascade_for
from kalos.assist.tasks import classify_intent, plain_wording, short_label
from kalos.providers import cascade as cascade_module
from kalos.providers.cascade import (
    DEFAULT_FAST_MODEL,
    DEFAULT_QUALITY_MODEL,
    Cascade,
    ValidationFailed,
    fast_model,
    quality_model,
)
from kalos.providers.grounding import (
    check_columns,
    check_numbers,
    identifiers_in,
    number_keys,
    parse_json_object,
    run_ids_in,
)


class Fake:
    """A tier model returning canned replies in order (or raising)."""

    local = True

    def __init__(self, name: str, *replies: Any, provider: str = "fake") -> None:
        self.provider = provider
        self.model = name
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return str(reply)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("KALOS_LLM_FAST_MODEL", "KALOS_LLM_QUALITY_MODEL", "KALOS_OLLAMA_MODEL",
                "KALOS_LLM_CASCADE", "KALOS_LLM_PROVIDER", "KALOS_DESKTOP", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _int_validator(reply: str) -> int:
    obj = parse_json_object(reply)
    if not isinstance(obj.get("n"), int):
        raise ValidationFailed("schema", "no n")
    return int(obj["n"])


# --- the cascade ------------------------------------------------------------------ #


def test_fast_task_uses_the_fast_model_when_its_reply_passes():
    fast, quality = Fake("f", '{"n": 1}'), Fake("q", '{"n": 2}')
    result = Cascade(fast, quality).run("header_mapping", "s", "p", _int_validator, lambda: -1)
    assert (result.value, result.provider_path, result.offline) == (1, "fast", False)
    assert quality.calls == [] and result.model == "f"


def test_fast_reply_that_fails_escalates_to_quality():
    fast, quality = Fake("f", "not json"), Fake("q", '{"n": 2}')
    result = Cascade(fast, quality).run("label", "s", "p", _int_validator, lambda: -1)
    assert (result.value, result.provider_path) == (2, "fast>quality")
    assert [(a.tier, a.outcome, a.reason) for a in result.attempts] == [
        ("fast", "invalid", "invalid_json"), ("quality", "ok", None)]


def test_both_fail_uses_the_offline_fallback_and_says_so():
    fast, quality = Fake("f", "{}"), Fake("q", "[1, 2]")
    result = Cascade(fast, quality).run("intent", "s", "p", _int_validator, lambda: -1)
    assert result.value == -1 and result.offline
    assert result.provider_path == "fast>quality>offline"
    report = result.report()
    assert report["offline_fallback"] is True and report["model"] is None
    assert [a["reason"] for a in report["attempts"]] == ["schema", "invalid_json"]


def test_unavailable_fast_model_escalates_and_all_unavailable_is_detected():
    fast, quality = Fake("f", ConnectionError("down")), Fake("q", ConnectionError("down"))
    c = Cascade(fast, quality)
    result = c.run("wording", "s", "p", _int_validator, lambda: -1)
    assert result.provider_path == "fast>quality>offline" and c.all_unavailable(result)
    assert not c.all_unavailable(Cascade(Fake("f", "x"), None).run("label", "s", "p", _int_validator, lambda: 0))


def test_quality_tasks_skip_the_fast_model():
    fast, quality = Fake("f", '{"n": 1}'), Fake("q", '{"n": 2}')
    for task in ("answer", "sop", "summary"):
        quality.replies = ['{"n": 2}']
        result = Cascade(fast, quality).run(task, "s", "p", _int_validator, lambda: -1)
        assert result.provider_path == "quality" and result.value == 2
    assert fast.calls == []


def test_escalation_off_and_one_model_never_retries_the_same_model():
    fast, quality = Fake("f", "bad"), Fake("q", '{"n": 2}')
    result = Cascade(fast, quality, escalate=False).run("label", "s", "p", _int_validator, lambda: -1)
    assert result.provider_path == "fast>offline" and quality.calls == []
    same = Fake("m", "bad", '{"n": 2}')
    result = Cascade(same, same).run("label", "s", "p", _int_validator, lambda: -1)
    assert result.provider_path == "fast>offline" and len(same.calls) == 1


def test_defaults_are_mit_or_apache_models_not_llama(monkeypatch):
    """The defaults moved away from llama3.1 (Llama community licence) to
    MIT / Apache-2.0 models; the licence table in docs/ASSISTANT.md says so."""
    licences = {"phi4-mini": "MIT", "qwen2.5:7b": "Apache-2.0", "qwen2.5:14b": "Apache-2.0",
                "qwen2.5:3b": "Qwen research licence"}
    assert licences[DEFAULT_FAST_MODEL] in {"MIT", "Apache-2.0"}
    assert licences[DEFAULT_QUALITY_MODEL] in {"MIT", "Apache-2.0"}
    assert "llama" not in DEFAULT_FAST_MODEL + DEFAULT_QUALITY_MODEL
    from kalos.normalize.config import _DEFAULT_OLLAMA_MODEL, load_config
    from kalos.providers.ollama_provider import DEFAULT_MODEL

    assert DEFAULT_MODEL == _DEFAULT_OLLAMA_MODEL == DEFAULT_QUALITY_MODEL
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    assert load_config().ollama_model == DEFAULT_QUALITY_MODEL
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "ASSISTANT.md").read_text()
    for model in (DEFAULT_FAST_MODEL, DEFAULT_QUALITY_MODEL):
        assert f"`{model}`" in doc and licences[model] in doc


def test_env_selects_the_models_and_the_legacy_variable_still_works(monkeypatch):
    assert (fast_model(), quality_model()) == (DEFAULT_FAST_MODEL, DEFAULT_QUALITY_MODEL)
    monkeypatch.setenv("KALOS_OLLAMA_MODEL", "mistral:7b")  # alone: the only model, as before
    assert (fast_model(), quality_model()) == ("mistral:7b", "mistral:7b")
    monkeypatch.setenv("KALOS_LLM_FAST_MODEL", "gemma3:4b")  # with a tier var: it is the quality model
    assert (fast_model(), quality_model()) == ("gemma3:4b", "mistral:7b")
    monkeypatch.setenv("KALOS_LLM_QUALITY_MODEL", "qwen2.5:14b")
    assert quality_model() == "qwen2.5:14b"
    monkeypatch.setenv("KALOS_LLM_CASCADE", "0")
    assert cascade_module.escalation_enabled() is False


def test_cascade_for_the_ollama_chat_model_has_two_tiers(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    c = cascade_for(OllamaChat(model=quality_model(), url="http://localhost:11434"))
    assert c is not None and c.fast is not None and c.quality is not None
    assert (c.fast.model, c.quality.model) == (DEFAULT_FAST_MODEL, DEFAULT_QUALITY_MODEL)
    fake = Fake("remote")
    one = cascade_for(fake)  # a remote or fake model is the only tier
    assert one is not None and one.fast is one.quality is fake
    assert cascade_for(None) is None


# --- grounding checks ------------------------------------------------------------- #


def test_grounding_helpers():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValidationFailed) as e:
        parse_json_object("sure! here it is")
    assert e.value.reason == "invalid_json"
    assert number_keys("2e9 and 2000000000 and 3,5") == {repr(2e9), repr(3.5)}
    check_numbers("titer 4.2", number_keys("4.20 g/L"))
    with pytest.raises(ValidationFailed) as e:
        check_numbers("titer 4.3", number_keys("4.2"))
    assert e.value.reason == "mismatched_number"
    with pytest.raises(ValidationFailed) as e:
        check_columns(["titer_g_L", "ghost"], ["titer_g_L"])
    assert e.value.reason == "unknown_column"
    assert identifiers_in("raise feed_rate_mL_h and DO_pct, not ph") == {"feed_rate_mL_h", "DO_pct"}
    assert run_ids_in("run " + "ab" * 16 + " ok") == {"ab" * 16}


# --- assistant answers ------------------------------------------------------------ #

RUN_ID = "0123456789abcdef" * 2
GUIDE = {"state": "runs_awaiting", "step": "Record a result for every started run.", "why": "", "why_it_matters": ""}


def _hits() -> list[Hit]:
    chunks = [
        Chunk("c1", "round", "0", "cmp_x", "Round 1",
              f"Round 1, started runs:\nRun {RUN_ID}: feed_rate = 2.5; Measured titer_g_L 4.2",
              user_terms=("titer_g_L", "feed_rate")),
        Chunk("c2", "analysis", "drivers", "cmp_x", "Drivers",
              "Higher feed_rate went with higher titer_g_L (strength 0.62; clear signal)",
              user_terms=("titer_g_L", "feed_rate")),
    ]
    return [Hit(c, 1.0) for c in chunks]


def test_answer_checks_drop_unknown_columns_runs_numbers_and_labels():
    _, labelled = build_prompt("q", _hits(), GUIDE)
    reply = " ".join([
        f"Run {RUN_ID} measured 4.2 [S1].",            # kept
        "Higher feed_rate went with higher titer_g_L [S2].",  # kept
        "Higher glucose_g_L went with titer [S2].",      # unknown column
        f"Run {'f' * 32} measured 4.2 [S1].",            # run that does not exist
        "The titer was 9.9 [S1].",                       # number not in the source
        "The strength was 0.62 [S7].",                   # source that does not exist
    ])
    checked = check_answer(reply, labelled, GUIDE)
    assert checked.answer == f"Run {RUN_ID} measured 4.2. Higher feed_rate went with higher titer_g_L."
    assert "9.9" not in checked.answer and "f" * 32 not in checked.answer
    assert checked.drop_reasons == {"unknown_column": 1, "nonexistent_citation": 2, "mismatched_number": 1}


def test_ask_reports_the_provider_path_and_the_checks():
    quality = Fake("q", "Higher feed_rate went with higher titer_g_L [S2]. Titer was 7.7 [S1].")
    out = ask("what went with titer?", _hits(), GUIDE, Cascade(Fake("f"), quality))
    assert out["abstained"] is False and out["provider_path"] == "quality"
    assert out["model"] == "q" and "7.7" not in out["answer"]
    assert out["checks"] == {"dropped_sentences": 1, "drop_reasons": {"mismatched_number": 1}}


def test_ask_never_shows_a_failed_answer_and_says_it_fell_back():
    quality = Fake("q", "The titer will be 9.9 [S1].")
    out = ask("what went with titer?", _hits(), GUIDE, Cascade(None, quality))
    from kalos.assist.guide import guide_text

    assert out["abstained"] is True and out["answer"] == guide_text(GUIDE)
    assert out["provider_path"] == "quality>offline" and out["offline_fallback"] is True
    assert "9.9" not in json.dumps(out)


def test_ask_raises_when_no_model_answered_at_all():
    with pytest.raises(ModelUnavailable):
        ask("q", _hits(), GUIDE, Cascade(None, Fake("q", ConnectionError("down"))))


def test_ask_route_returns_provider_path(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.assist.llm import get_chat_model
    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "LOOP.md").write_text("# Loop\n\nRecord a result for every started run.\n")
    monkeypatch.setenv("KALOS_ASSIST_DOCS_DIR", str(docs))
    monkeypatch.setenv("KALOS_ASSIST_EMBED", "0")
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    store = CampaignStore(tmp_path / "state")
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    fake = Fake("fake-q", "Record a result for every started run [S1].")
    app.dependency_overrides[get_campaign_store] = lambda: store
    app.dependency_overrides[get_chat_model] = lambda: fake
    try:
        r = TestClient(app).post("/api/assist/ask", json={"question": "how do I record a result"})
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_chat_model, None)
    assert r.status_code == 200
    body = r.json()
    assert body["provider_path"] == "quality" and body["abstained"] is False


# --- header mapping (normalize) through the cascade -------------------------------- #


def _sheet() -> pd.DataFrame:
    return pd.DataFrame({
        "Operator": ["J. Rivera", "M. Chen", "J. Rivera"],
        "Titer (g/L)": [3.2, 3.6, 4.1],
        "Temp": ["34.6 C", "34.8 C", "35.0 C"],
    })


def _plan_reply(columns: list[dict[str, Any]]) -> str:
    return json.dumps({"response": json.dumps({"columns": columns})})


GOOD = [
    {"raw_name": "Titer (g/L)", "canonical_name": "titer", "role": "target", "unit_token": "g/L",
     "is_identity": False, "note": "product titer"},
    {"raw_name": "Temp", "canonical_name": "temperature", "role": "feature", "unit_token": "C",
     "is_identity": False, "note": "culture temperature"},
]


def _patch_ollama(monkeypatch, replies: list[str]) -> list[str]:
    import urllib.request

    models: list[str] = []

    class Resp:
        status = 200

        def __init__(self, body: bytes) -> None:
            self.body = body

        def read(self) -> bytes:
            return self.body

        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def urlopen(req: Any, timeout: float | None = None) -> Resp:
        if isinstance(req, str):
            return Resp(b"{}")
        models.append(json.loads(req.data)["model"])
        return Resp(replies.pop(0).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return models


@pytest.mark.parametrize("bad, reason", [
    ([{**GOOD[0], "raw_name": "Titre"}, GOOD[1]], "unknown_column"),
    ([GOOD[0]], "schema"),  # a column left out
    ([{**GOOD[0], "note": "titer around 9.5 g/L"}, GOOD[1]], "mismatched_number"),
    ([GOOD[0], {**GOOD[1], "role": "target"}], "schema"),  # two targets
])
def test_header_plan_failing_a_check_escalates_to_quality(monkeypatch, bad, reason):
    from kalos.normalize.config import load_config
    from kalos.normalize.llm import propose_plan

    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    models = _patch_ollama(monkeypatch, [_plan_reply(bad), _plan_reply(GOOD)])
    plan = propose_plan(_sheet(), config=load_config())
    assert models == [DEFAULT_FAST_MODEL, DEFAULT_QUALITY_MODEL]
    assert plan.created_by == "llm" and plan.model == DEFAULT_QUALITY_MODEL
    assert plan.provider_path == "fast>quality"
    assert plan.provider_attempts[0]["reason"] == reason
    assert {c.raw_name: c.role for c in plan.columns}["Operator"] == "identity"


def test_header_plan_both_failing_is_the_offline_plan(monkeypatch):
    from kalos.normalize.config import load_config
    from kalos.normalize.llm import propose_plan

    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    _patch_ollama(monkeypatch, [json.dumps({"response": "no"}), _plan_reply([GOOD[0]])])
    plan = propose_plan(_sheet(), config=load_config())
    assert plan.created_by == "offline" and plan.provider_path == "fast>quality>offline"
    assert [a["reason"] for a in plan.provider_attempts] == ["invalid_json", "schema"]
    assert json.loads(plan.to_json())["provider_path"] == "fast>quality>offline"


def test_header_plan_without_a_live_provider_is_offline(monkeypatch):
    from kalos.normalize.llm import propose_plan

    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    assert propose_plan(_sheet()).provider_path == "offline"


# --- fast tasks ------------------------------------------------------------------- #


def test_intent_routing_validates_the_label_and_falls_back_to_keywords():
    result = classify_intent("what is the best titer so far?", Cascade(Fake("f", '{"intent": "campaign_data"}'), None))
    assert (result.value, result.provider_path) == ("campaign_data", "fast")
    bad = Cascade(Fake("f", '{"intent": "weather"}'), Fake("q", "nope"))
    result = classify_intent("write an SOP for this round", bad)
    assert result.value == "sop" and result.provider_path == "fast>quality>offline"
    assert classify_intent("who won the match?", None).value == "out_of_scope"


def test_short_label_uses_only_engine_words_and_no_digits():
    ok = short_label("feed_rate_mL_h", "feed rate in mL per hour", Cascade(Fake("f", '{"label": "Feed rate"}'), None))
    assert ok.value == "Feed rate"
    invented = Cascade(Fake("f", '{"label": "Glucose feed 5x"}'), Fake("q", '{"label": "Optimal feed"}'))
    result = short_label("feed_rate_mL_h", "feed rate in mL per hour", invented)
    assert result.value == "feed_rate_mL_h" and result.offline


def test_plain_wording_keeps_every_number_and_adds_none():
    msg = "DO_pct has 3 values above 100 (max 140); a dissolved-oxygen percentage cannot exceed 100."
    good = '{"text": "3 readings of DO_pct are above 100, up to 140, which is not possible for a percentage of 100 or less."}'
    result = plain_wording(msg, Cascade(Fake("f", good), None))
    assert result.provider_path == "fast"
    drops = '{"text": "Some DO_pct readings are too high."}'
    adds = '{"text": "3 DO_pct values above 100 (max 140); 100 is the limit; fix 2 of them."}'
    renames = '{"text": "3 do_percent values above 100, max 140, over 100."}'
    for reply, reason in ((drops, "mismatched_number"), (adds, "mismatched_number"), (renames, "unknown_column")):
        result = plain_wording(msg, Cascade(Fake("f", reply), None))
        assert result.value == msg and result.attempts[0].reason == reason
    claim = '{"text": "3 DO_pct values above 100 (max 140) will ruin the run because 100 is the cap."}'
    assert plain_wording(msg, Cascade(Fake("f", claim), None)).attempts[0].reason == "unsupported_claim"
