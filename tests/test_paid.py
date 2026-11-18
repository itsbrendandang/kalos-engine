"""The spending guard is tested without network access or real keys."""
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from kalos.assist.llm import AnthropicChat, ModelUnavailable
from kalos.normalize.config import load_config
from kalos.normalize.llm import propose_plan
from kalos.paid import PaidCallsOff, refuse_paid


@pytest.mark.parametrize("value", ["1", "true", "YES", " On "])
def test_refuse_true_values(monkeypatch, value):
    monkeypatch.setenv("KALOS_NO_PAID_CALLS", value)
    with pytest.raises(PaidCallsOff, match="KALOS_NO_PAID_CALLS"):
        refuse_paid("fake step")


@pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
def test_unset_or_off_keeps_existing_behavior(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("KALOS_NO_PAID_CALLS")
    else:
        monkeypatch.setenv("KALOS_NO_PAID_CALLS", value)
    refuse_paid("fake step")


def test_paid_paths_refused_before_client_creation(monkeypatch):
    calls = []

    def client(**kwargs):
        calls.append(kwargs)
        raise AssertionError("client must not be constructed")

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=client))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "anthropic")
    with pytest.raises(ModelUnavailable, match="KALOS_NO_PAID_CALLS"):
        AnthropicChat("fake").complete("system", "prompt")
    plan = propose_plan(pd.DataFrame({"Titer": [1.0, 2.0]}), config=load_config())
    assert plan.created_by == "offline"
    assert calls == []
