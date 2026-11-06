"""POST /api/preflight needs the write scope (owner decision; docs/CAMPAIGNS_API.md,
"Preflight"): a read-only token gets 403, a write token gets the reading."""
from __future__ import annotations

import hashlib
import io
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal.app import app, get_campaign_store
from kalos.portal.campaign import CampaignStore

CSV = b"Glucose,Temp,Titer\n" + b"".join(
    f"{10 + i},{28 + i % 3},{40 + i}\n".encode() for i in range(10)
)


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("t-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("t-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _post(c, token):
    return c.post("/api/preflight", headers={"Authorization": f"Bearer {token}"},
                  files={"file": ("s.csv", io.BytesIO(CSV), "text/csv")},
                  data={"target": "Titer"})


def test_a_read_only_token_cannot_preflight(client):
    r = _post(client, "t-read")
    assert r.status_code == 403
    assert r.json() == {"error": "requires 'write' scope", "code": "forbidden"}


def test_a_write_token_can_preflight(client):
    r = _post(client, "t-write")
    assert r.status_code == 200, r.text
    assert r.json()["preflight"] is True
