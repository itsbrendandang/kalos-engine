"""Semantic search over a campaign (kalos.assist.semantic, GET /api/campaigns/{id}/search).

No Ollama: the embedder is a deterministic fake, and the Ollama HTTP call is
monkeypatched where the transport itself is tested."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from kalos.assist import ratelimit
from kalos.assist import semantic as semantic_module
from kalos.assist.semantic import EmbeddingCache, campaign_documents, search
from kalos.portal.campaign import CampaignStore


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
    {"token_sha256": _sha("b-read"), "subject": "gina@globex", "tenant": "globex", "scopes": ["read"]},
]
SYNONYMS = {"pipetting": "error", "mistake": "error", "warm": "temperature", "hot": "temperature"}


class FakeEmbedder:
    """Bag of words hashed into 1024 dims, with a tiny synonym table so a query
    can match a record that shares no word with it (what keyword search misses)."""

    name = "fake-embed"

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        self.encoded.extend(texts)
        out = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                w = SYNONYMS.get(w, w)
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 1024] += 1.0  # noqa: S324 - not security
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def _seed(store: CampaignStore, tenant: str, marker: str) -> str:
    cid = store.create_campaign(tenant=tenant, name=f"c-{marker}", created_by="t")["id"]
    df = pd.DataFrame({"Operator": ["J. Rivera", "M. Chen", "J. Rivera"], "pH": [6.8, 7.0, 7.2],
                       "temperature_C": [30.0, 34.0, 37.0], "titer_g_L": [1.0, 2.0, 1.5]})
    store.seed(df, "titer_g_L", ["pH", "temperature_C"], tenant=tenant, campaign_id=cid,
               analysis={"target": "titer_g_L", "features": ["pH", "temperature_C"]}, dataset="s.csv")
    runs = store.start([{"recipe": {"pH": 7.1, "temperature_C": 36.5},
                         "reason": f"{marker} edge of the design"}], tenant=tenant, campaign_id=cid)
    store.set_result(runs[0]["id"], 3.3, tenant=tenant, campaign_id=cid)
    store.set_result(runs[0]["id"], 3.1, tenant=tenant, campaign_id=cid,
                     reason=f"{marker} plate re-read after an error")
    return cid


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_ASSIST_DOCS_DIR", str(tmp_path / "nodocs"))
    monkeypatch.setenv("KALOS_ASSIST_EMBED", "0")
    return CampaignStore(tmp_path / "state")


def test_documents_cover_runs_rows_columns_notes_and_index_without_identities(store):
    cid = _seed(store, "acme", "AcmeZeus")
    docs = campaign_documents(store, "acme", cid)
    kinds = {d.kind for d in docs}
    assert {"run", "row", "column", "note", "sheet"} <= kinds
    text = json.dumps([d.__dict__ for d in docs])
    assert "J. Rivera" not in text and "M. Chen" not in text and "Operator" not in text
    note = next(d for d in docs if d.kind == "note")
    assert "re-read" in note.text and "3.3" in note.text and "3.1" in note.text


def test_embedding_search_finds_meaning_and_caches_by_content_hash(store):
    cid = _seed(store, "acme", "AcmeZeus")
    fake = FakeEmbedder()
    out = search(store, "acme", cid, "pipetting mistake", k=3, embedder=fake)
    assert out["retriever"] == "embeddings" and out["model"] == "fake-embed"
    assert out["hits"][0]["kind"] == "note"  # "error" via the synonym, no shared word
    first = len(fake.encoded)
    search(store, "acme", cid, "warm run", k=3, embedder=fake)
    assert len(fake.encoded) == first + 1  # only the query: every record vector was cached
    keyword = search(store, "acme", cid, "pipetting mistake", k=3, embedder=None)
    assert keyword["retriever"] == "keyword" and keyword["hits"] == []


def test_cache_is_bounded_and_least_recently_used_goes_first(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_EMBED_CACHE_MAX", "100")
    cache = EmbeddingCache(tmp_path)
    vec = np.ones(4, dtype=np.float32)
    cache.put_many("m", [(f"t{i}", vec) for i in range(100)])
    cache.get_many("m", ["t0"])  # t0 used again: kept
    cache.put_many("m", [(f"u{i}", vec) for i in range(50)])
    assert cache.size() == 100
    assert set(cache.get_many("m", ["t0", "t1"])) == {cache.key("m", "t0")}
    assert cache.get_many("other-model", ["t0"]) == {}  # keyed by model and text


def test_keyword_fallback_when_no_local_model_or_the_model_fails(store, monkeypatch):
    cid = _seed(store, "acme", "AcmeZeus")
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")
    out = search(store, "acme", cid, "plate re-read")
    assert out["retriever"] == "keyword" and "not ollama" in out["fallback_reason"]
    assert out["hits"] and out["hits"][0]["kind"] == "note"
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "anthropic")  # never embedded remotely
    assert search(store, "acme", cid, "plate")["retriever"] == "keyword"
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "ollama")

    from kalos.providers import ollama_provider

    def down(request, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(ollama_provider, "_open", down)
    out = search(store, "acme", cid, "plate re-read")
    assert out["retriever"] == "keyword" and "nomic-embed-text" in out["fallback_reason"]
    monkeypatch.setenv("KALOS_SEMANTIC_SEARCH", "0")
    assert "turned off" in search(store, "acme", cid, "plate")["fallback_reason"]


def test_ollama_embedder_uses_the_embed_route_with_nomic_prefixes(monkeypatch):
    from kalos.providers import ollama_provider

    seen: dict = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data)
        return json.dumps({"embeddings": [[3.0, 4.0] for _ in seen["body"]["input"]]}).encode()

    monkeypatch.setattr(ollama_provider, "_open", fake_open)
    vecs = semantic_module.OllamaEmbedder().encode(["a run"], query=True)
    assert seen["url"].endswith("/api/embed") and seen["body"]["model"] == "nomic-embed-text"
    assert seen["body"]["input"] == ["search_query: a run"]
    assert np.allclose(vecs, [[0.6, 0.8]])


def test_search_route_scope_tenant_isolation_and_bounds(store, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import get_campaign_store

    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "none")  # search still works: keyword
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", store)
    ratelimit.reset()
    a = _seed(store, "acme", "AcmeZeus")
    b = _seed(store, "globex", "GlobexHera")
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        tc = TestClient(app)
        h = {"Authorization": "Bearer a-read"}
        r = tc.get(f"/api/campaigns/{a}/search", params={"q": "plate re-read"}, headers=h)
        assert r.status_code == 200
        body = r.json()
        assert body["retriever"] == "keyword" and body["campaign_id"] == a and body["hits"]
        assert "GlobexHera" not in r.text
        assert tc.get(f"/api/campaigns/{b}/search", params={"q": "plate"}, headers=h).status_code == 404
        assert tc.get("/api/campaigns/../search", params={"q": "x"}, headers=h).status_code == 404
        assert tc.get(f"/api/campaigns/{a}/search", params={"q": "x", "k": 26}, headers=h).status_code == 400
        assert tc.get(f"/api/campaigns/{a}/search", params={"q": "x" * 501}, headers=h).status_code == 400
        assert tc.get(f"/api/campaigns/{a}/search", params={"q": "x"}).status_code == 401
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        ratelimit.reset()
