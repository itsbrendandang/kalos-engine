"""Assistant provider resolution (kalos/assist/llm.py)."""
from __future__ import annotations

import pytest

from kalos.assist.llm import (
    AnthropicChat,
    OllamaChat,
    assist_provider,
    get_chat_model,
    model_info,
)
from kalos.envconfig import ConfigError


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("KALOS_LLM_PROVIDER", "KALOS_DESKTOP", "KALOS_OLLAMA_URL", "KALOS_OLLAMA_MODEL",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_unset_is_none_outside_desktop_and_ollama_in_desktop(monkeypatch):
    """sec-assist (e): nothing leaves the machine unless anthropic is set on purpose."""
    assert assist_provider() == "none"
    assert get_chat_model() is None
    assert model_info()["provider"] == "none"
    monkeypatch.setenv("KALOS_DESKTOP", "1")
    assert assist_provider() == "ollama"
    assert isinstance(get_chat_model(), OllamaChat)


def test_none_gives_no_model(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    assert get_chat_model() is None
    assert model_info() == {"provider": "none", "model": None, "local": True}


def test_ollama_info_is_local_and_names_the_model(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("KALOS_OLLAMA_MODEL", "qwen2.5:3b")
    assert model_info() == {"provider": "ollama", "model": "qwen2.5:3b", "local": True}
    monkeypatch.setenv("KALOS_OLLAMA_URL", "http://10.0.0.5:11434")
    assert model_info()["local"] is False


def test_anthropic_is_never_local_and_needs_a_key(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "anthropic")
    model = get_chat_model()
    assert isinstance(model, AnthropicChat) and model.local is False
    from kalos.assist.llm import ModelUnavailable

    with pytest.raises(ModelUnavailable):
        model.complete("s", "p")


def test_typo_fails_closed(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "nnone")
    with pytest.raises(ConfigError):
        assist_provider()


def test_docs_say_revise_sends_the_whole_draft():
    """sec-assist (f): the docs name what a revise sends, not only what a question sends."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs" / "ASSISTANT.md").read_text()
    assert "For an SOP revise, the **whole draft**" in text
    assert "For an SOP revise: the whole draft" in text


def test_anthropic_client_has_a_timeout_and_one_retry(monkeypatch):
    """sec-assist (h): the cloud call is bounded (60 s, one retry)."""
    import sys
    import types

    seen = {}

    class _Messages:
        def create(self, **kwargs):
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="ok [S1]")])

    class _Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert AnthropicChat(model="m").complete("s", "p") == "ok [S1]"
    assert seen == {"timeout": 60.0, "max_retries": 1}
