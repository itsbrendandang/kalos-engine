"""Billable model calls obey the global switch without contacting a server."""
from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from kalos.assist.llm import AnthropicChat, ModelUnavailable, model_info
from kalos.normalize.config import load_config
from kalos.normalize.llm import propose_plan
from kalos.paid import PaidCallsOff, no_paid_calls, refuse_paid
from kalos.providers.anthropic_provider import AnthropicProvider
from kalos.providers.ollama_provider import OllamaProvider


@pytest.fixture
def anthropic_client(monkeypatch):
    calls = []

    def client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(text="answer")],
        )))

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "anthropic")
    return calls


@pytest.mark.parametrize("flag", ["1", "true", "YES", " on ", "false", "0", "", None])
def test_paid_switch_assistant(monkeypatch, anthropic_client, flag):
    if flag is None:
        monkeypatch.delenv("KALOS_NO_PAID_CALLS", raising=False)
    else:
        monkeypatch.setenv("KALOS_NO_PAID_CALLS", flag)
    blocked = flag is not None and flag.strip().lower() in {"1", "true", "yes", "on"}
    assert no_paid_calls() is blocked
    assert model_info()["available"] is not blocked
    assert AnthropicProvider().status().available is not blocked
    assert AnthropicProvider().status().missing_env == ()
    assert AnthropicProvider().status().disabled_by == (("KALOS_NO_PAID_CALLS",) if blocked else ())
    if blocked:
        with pytest.raises(ModelUnavailable, match="KALOS_NO_PAID_CALLS"):
            AnthropicChat(model="test").complete("system", "prompt")
        assert anthropic_client == []
    else:
        assert AnthropicChat(model="test").complete("system", "prompt") == "answer"
        assert len(anthropic_client) == 1


@pytest.mark.parametrize("flag", ["1", "true", "YES", " on "])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_paid_switch_normalization(monkeypatch, anthropic_client, flag, explicit_config):
    monkeypatch.delenv("KALOS_NO_PAID_CALLS", raising=False)
    config = replace(load_config(), enabled_live=True)
    monkeypatch.setenv("KALOS_NO_PAID_CALLS", flag)
    assert load_config().enabled_live is False
    plan = propose_plan(pd.DataFrame({"Temp": [30, 32]}), config=config if explicit_config else None)
    assert plan.created_by == "offline"
    assert plan.provider_path == "offline"
    assert anthropic_client == []


def test_normalization_calls_client_when_switch_off(monkeypatch, anthropic_client):
    monkeypatch.delenv("KALOS_NO_PAID_CALLS", raising=False)
    assert load_config().enabled_live is True
    propose_plan(pd.DataFrame({"Temp": [30, 32]}))
    assert len(anthropic_client) == 1


def test_paid_switch_keeps_ollama_available(monkeypatch):
    monkeypatch.setenv("KALOS_NO_PAID_CALLS", "true")
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    assert load_config().enabled_live is True
    monkeypatch.setattr("kalos.providers.ollama_provider._open", lambda *a: b'{"message":{"content":"answer"}}')
    assert OllamaProvider().chat("system", "prompt") == "answer"


@pytest.mark.parametrize("url,disabled", [("http://localhost:8080/v1", ()),
                                         ("https://example.com/v1", ("KALOS_NO_PAID_CALLS",))])
def test_openai_status_reports_paid_policy(monkeypatch, url, disabled):
    from kalos.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setenv("KALOS_NO_PAID_CALLS", "true")
    monkeypatch.setenv("KALOS_OPENAI_BASE_URL", url)
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("KALOS_OPENAI_ALLOW_REMOTE", "true")
    provider = OpenAICompatibleProvider()
    status = provider.status()
    assert provider.available() is (not bool(disabled))
    assert status.available is (not bool(disabled))
    assert status.missing_env == ()
    assert status.disabled_by == disabled
    info = model_info()
    if "available" in info:
        assert info["available"] is (not bool(disabled))


def test_anthropic_status_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KALOS_NO_PAID_CALLS", raising=False)
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "anthropic")
    assert model_info()["available"] is False
    assert AnthropicProvider().status().missing_env == ("ANTHROPIC_API_KEY",)


@pytest.mark.parametrize("value", ["1", "true", "YES", " On "])
def test_refuse_true_values(monkeypatch, value):
    monkeypatch.setenv("KALOS_NO_PAID_CALLS", value)
    with pytest.raises(PaidCallsOff, match="KALOS_NO_PAID_CALLS"):
        refuse_paid("fake step")


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_unset_or_off_keeps_existing_behavior(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("KALOS_NO_PAID_CALLS", raising=False)
    else:
        monkeypatch.setenv("KALOS_NO_PAID_CALLS", value)
    refuse_paid("fake step")


def test_paid_paths_refused_before_client_creation(monkeypatch, anthropic_client):
    monkeypatch.setenv("KALOS_NO_PAID_CALLS", "1")
    with pytest.raises(ModelUnavailable, match="KALOS_NO_PAID_CALLS"):
        AnthropicChat("fake").complete("system", "prompt")
    plan = propose_plan(pd.DataFrame({"Titer": [1.0, 2.0]}), config=load_config())
    assert plan.created_by == "offline"
    assert anthropic_client == []
