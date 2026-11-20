"""Setup suggestions (kalos.assist.setup_suggest, POST /api/campaigns/{id}/setup/suggestions).

The model helps, the user decides: suggestions are checked against the sheet,
carry an engine-set confidence, are never applied, and never reach the fit.
No Ollama: models are fakes."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.assist import ratelimit
from kalos.assist.setup_suggest import ROLES, suggest
from kalos.providers.cascade import Cascade

ROOT = Path(__file__).resolve().parents[1]


class Fake:
    local = True
    provider = "fake"

    def __init__(self, name: str, *replies: str) -> None:
        self.model = name
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


def _sheet(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ph = rng.uniform(6.6, 7.4, n).round(2)
    temp = rng.uniform(30, 37, n).round(1)
    titer = (2 + (ph - 7) + 0.1 * (temp - 33) + rng.normal(0, 0.1, n)).round(3)
    return pd.DataFrame({
        "Operator": ["J. Doe", "A. Roe"] * (n // 2),
        "run_id": [f"R{i:03d}" for i in range(n)],
        "pH": ph,
        "temperature_C": temp,
        "clone_id": ["C1", "C2", "C3"] * (n // 3),
        "passage": rng.integers(5, 20, n),
        "titer_g_L": titer,
        "titer_x10": (titer * 10).round(2),  # derived from the target
    })


def _reply(**overrides: dict) -> str:
    base = {
        "run_id": {"role": "id", "unit": None, "reason": "Unique code per run."},
        "pH": {"role": "setting", "unit": None, "reason": "Chosen pH of the culture."},
        "temperature_C": {"role": "setting", "unit": "°C", "reason": "Culture temperature."},
        "clone_id": {"role": "clone", "unit": None, "reason": "Three clone labels."},
        "passage": {"role": "covariate", "unit": None, "reason": "Passage of the cells, recorded."},
        "titer_g_L": {"role": "outcome", "unit": "g/L", "reason": "Measured product titer."},
        "titer_x10": {"role": "derived_from_target", "unit": None, "reason": "Scaled copy of the titer."},
    }
    base.update(overrides)
    return json.dumps({"columns": [{"column": c, **v} for c, v in base.items()]})


def _by_col(out: dict) -> dict:
    return {s["column"]: s for s in out["suggestions"]}


def test_offline_suggestions_are_marked_and_use_the_engine_evidence():
    out = suggest(_sheet(), target="titer_g_L", cascade=None)
    assert out["applied"] is False and out["status"] == "suggestions"
    assert out["provider_path"] == "offline" and out["offline_fallback"] is True
    s = _by_col(out)
    assert "Operator" not in s and out["left_out_identity_columns"] == 1
    assert all(x["requires_confirmation"] for x in out["suggestions"])
    assert s["titer_x10"]["role"] == "derived_from_target" and s["titer_x10"]["roles_key"] == "outputs"
    assert any("target_leak" in e["note"] for e in s["titer_x10"]["evidence"])
    assert s["clone_id"]["roles_key"] == "strain" and s["passage"]["roles_key"] == "covariates"
    assert s["run_id"]["roles_key"] == "ids" and s["titer_g_L"]["roles_key"] == "target"
    assert s["titer_g_L"]["unit"] == "g/L" and s["titer_g_L"]["unit_source"] == "header"
    assert {r["role"] for r in out["role_vocabulary"]} == set(ROLES)
    draft = out["roles_draft"]
    assert draft["target"] == "titer_g_L" and draft["strain"] == "clone_id"
    assert "titer_x10" in draft["outputs"] and "pH" in draft["features"]


def test_the_model_sees_summaries_never_rows_or_identities():
    fast = Fake("f", _reply())
    suggest(_sheet(), target="titer_g_L", cascade=Cascade(fast, None))
    prompt = fast.prompts[0]
    assert "J. Doe" not in prompt and "Operator" not in prompt
    assert "R005" not in prompt  # an identifier's values: only counts are sent
    assert '"median"' in prompt and '"distinct"' in prompt


def test_model_suggestions_get_an_engine_confidence_not_the_models():
    reply = json.loads(_reply())
    for item in reply["columns"]:
        item["confidence"] = 0.99  # ignored: the engine sets confidence
    out = suggest(_sheet(), target="titer_g_L", cascade=Cascade(Fake("f", json.dumps(reply)), None))
    s = _by_col(out)
    assert out["provider_path"] == "fast"
    assert {x["confidence"] for x in out["suggestions"]} <= {"high", "medium", "low"}
    assert s["clone_id"]["confidence"] == "high"   # the clone name rule agrees
    assert s["pH"]["confidence"] == "medium"       # no evidence either way
    assert s["pH"]["reason_source"] == "model"
    wrong = _reply(titer_x10={"role": "setting", "unit": None, "reason": "A process input."})
    s = _by_col(suggest(_sheet(), target="titer_g_L", cascade=Cascade(Fake("f", wrong), None)))
    assert s["titer_x10"]["role"] == "setting" and s["titer_x10"]["confidence"] == "low"
    assert s["titer_x10"]["engine_role"] == "derived_from_target"


@pytest.mark.parametrize("bad, reason", [
    (_reply(pH={"role": "setting", "unit": None, "reason": "pH ranges 5.0 to 9.0."}), "mismatched_number"),
    (json.dumps({"columns": [{"column": "glucose_g_L", "role": "setting", "unit": None, "reason": "x"}]}),
     "unknown_column"),
    (_reply(pH={"role": "knob", "unit": None, "reason": "A knob."}), "schema"),
    (_reply(pH={"role": "setting", "unit": "furlongs", "reason": "Chosen pH."}), "schema"),
    (_reply(pH={"role": "setting", "unit": None, "reason": "Higher pH will raise titer_g_L."}), "unsupported_claim"),
    (_reply(pH={"role": "setting", "unit": None, "reason": "Tracks feed_rate_mL_h."}), "unknown_column"),
    ("I think pH is a setting.", "invalid_json"),
])
def test_a_failed_check_escalates_and_is_never_shown(bad, reason):
    fast, quality = Fake("f", bad), Fake("q", _reply())
    out = suggest(_sheet(), target="titer_g_L", cascade=Cascade(fast, quality))
    assert out["provider_path"] == "fast>quality" and out["attempts"][0]["reason"] == reason
    assert "5.0 to 9.0" not in json.dumps(out) and "furlongs" not in json.dumps(out)


def test_both_failing_falls_back_to_the_engine_and_says_so():
    out = suggest(_sheet(), target="titer_g_L", cascade=Cascade(Fake("f", "no"), Fake("q", "{}")))
    assert out["provider_path"] == "fast>quality>offline" and out["offline_fallback"] is True
    assert {x["reason_source"] for x in out["suggestions"]} == {"engine"}


def test_suggestions_never_reach_the_fit():
    """No module of the fit, the analysis or the campaign loop imports the
    suggestions: they cannot become priors or roles without the user."""
    offenders = []
    for path in [*(ROOT / "kalos" / "core").rglob("*.py"), *(ROOT / "kalos" / "portal").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any("setup_suggest" in n for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def test_route_reads_only_and_is_tenant_scoped(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.assist.llm import get_chat_model
    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    tokens = [
        {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
        {"token_sha256": _sha("b-read"), "subject": "gina@globex", "tenant": "globex", "scopes": ["read"]},
    ]
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(tokens))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")
    store = CampaignStore(tmp_path / "state")
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    ratelimit.reset()
    df = _sheet()
    cid = store.create_campaign(tenant="acme", name="a", created_by="t")["id"]
    store.seed(df, "titer_g_L", ["pH", "temperature_C"], tenant="acme", campaign_id=cid,
               analysis={"target": "titer_g_L"}, dataset="s.csv")
    empty = store.create_campaign(tenant="acme", name="empty", created_by="t")["id"]
    other = store.create_campaign(tenant="globex", name="b", created_by="t")["id"]
    before = (store.get(tenant="acme", campaign_id=cid), store.generation_of(tenant="acme", campaign_id=cid),
              store.load_analysis(tenant="acme", campaign_id=cid))
    fake = Fake("fake-q", _reply())
    app.dependency_overrides[get_campaign_store] = lambda: store
    app.dependency_overrides[get_chat_model] = lambda: fake
    try:
        tc = TestClient(app)
        h = {"Authorization": "Bearer a-read"}
        r = tc.post(f"/api/campaigns/{cid}/setup/suggestions", json={}, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["applied"] is False and body["campaign_id"] == cid
        assert body["provider_path"] == "fast"  # one fake model on both tiers: one try
        assert "J. Doe" not in r.text
        setup = tc.get(f"/api/campaigns/{cid}/setup", headers=h)
        assert setup.status_code == 200
        after = (store.get(tenant="acme", campaign_id=cid), store.generation_of(tenant="acme", campaign_id=cid),
                 store.load_analysis(tenant="acme", campaign_id=cid))
        assert after == before  # nothing written
        assert tc.post(f"/api/campaigns/{empty}/setup/suggestions", headers=h).status_code == 409
        assert tc.post(f"/api/campaigns/{other}/setup/suggestions", headers=h).status_code == 404
        assert tc.post(f"/api/campaigns/{cid}/setup/suggestions", json={"bogus": 1}, headers=h).status_code == 400
        monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
        off = tc.post(f"/api/campaigns/{cid}/setup/suggestions", headers=h).json()
        assert off["provider_path"] == "offline"
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_chat_model, None)
        ratelimit.reset()
