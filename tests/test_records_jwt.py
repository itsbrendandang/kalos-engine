"""The records routes with real EdDSA engine tokens (review N6).

kalos-web signs a short-lived Ed25519 JWT per request (docs/ENGINE_TOKEN.md);
the engine verifies it against the JWKS and takes the tenant and scopes from the
signed claims only. These tests drive every records route through the real app
and the real verifier with tokens minted here (keys generated at test time; no
secret committed): scopes map `campaigns:read` -> read and `campaigns:write` ->
write with no implication; the tenant comes from the token, never from the
request; forged, HS256, expired and wrong-audience tokens get the generic 401.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from records_helpers import BODIES, ROUTES

from kalos.api.routers._records_common import reset_backends, reset_refusal_budget
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store

KEY = Ed25519PrivateKey.generate()
ATTACKER = Ed25519PrivateKey.generate()
KID = "k-records"
READ, WRITE = "campaigns:read", "campaigns:write"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwks(path: Path) -> None:
    pub = KEY.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    path.write_text(json.dumps({"keys": [
        {"kty": "OKP", "crv": "Ed25519", "x": _b64(pub), "kid": KID, "alg": "EdDSA", "use": "sig"}]}))


def token(tenant: str = "acme", scopes: tuple[str, ...] = (READ, WRITE), *, key: Any = None,
          **over: Any) -> str:
    now = int(time.time())
    claims = {"iss": "kalos-web", "aud": "kalos-engine", "sub": f"user-{tenant}", "sid": f"s-{tenant}",
              "tenant": tenant, "scopes": list(scopes), "jti": str(uuid.uuid4()),
              "iat": now, "nbf": now, "exp": now + 60, **over}
    return jwt.encode(claims, key or KEY, algorithm="EdDSA", headers={"kid": KID})


def hs256_with_public_key(tenant: str = "acme") -> str:
    """The classic algorithm-confusion forgery: HS256 MAC'd with the public key."""
    pub = KEY.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    now = int(time.time())
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": KID}).encode())
    body = _b64(json.dumps({"iss": "kalos-web", "aud": "kalos-engine", "sub": "x", "sid": "x",
                            "tenant": tenant, "scopes": [READ, WRITE], "jti": "j", "iat": now,
                            "nbf": now, "exp": now + 60}).encode())
    sig = _b64(hmac.new(pub, f"{header}.{body}".encode(), hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def auth(tok: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture
def client(tmp_path, monkeypatch) -> Iterator[tuple[TestClient, CampaignStore]]:
    for name in list(os.environ):
        if name.startswith("KALOS_") and name not in ("KALOS_TORCH_THREADS", "KALOS_TEST_DATABASE_URL"):
            monkeypatch.delenv(name)
    jwks = tmp_path / "jwks.json"
    _jwks(jwks)
    monkeypatch.setenv("KALOS_JWT_PUBLIC_KEYS_FILE", str(jwks))
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path / "state"))
    store = CampaignStore(tmp_path / "state")
    app.dependency_overrides[get_campaign_store] = lambda: store
    reset_backends()
    reset_refusal_budget()
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        reset_backends()


def _world(tc: TestClient, store: CampaignStore) -> dict[str, str]:
    cids = [store.create_campaign(tenant="acme", name=f"c{i}", created_by="x")["id"] for i in range(3)]
    pid = tc.post("/api/projects", headers=auth(token()), json={"name": "P", "objective": "o"}).json()["project"]["id"]
    assert tc.put(f"/api/projects/{pid}/campaigns/{cids[0]}", headers=auth(token())).status_code == 200
    lid = tc.post("/api/lineage/links", headers=auth(token()), json={
        "parent_campaign_id": cids[1], "child_campaign_id": cids[2], "relation": "scale_up"}).json()["link"]["id"]
    return {"pid": pid, "cid": cids[0], "cid2": cids[1], "lid": str(lid)}


def _routes() -> list[tuple[str, str, str, Any]]:
    return [(m, p, s, BODIES.get((m, p))) for m, p, s in ROUTES]


def _fill(template: Any, ids: dict[str, str]) -> Any:
    if isinstance(template, dict):
        return {k: _fill(v, ids) for k, v in template.items()}
    return template.format(**ids) if isinstance(template, str) else template


def test_scopes_from_signed_claims_on_every_records_route(client):
    """read-only tokens read and are refused writes; write-only tokens write and
    are refused reads (no implication); a token with both does both."""
    tc, store = client
    ids = _world(tc, store)
    for method, path, scope, body in _routes():
        url, payload = path.format(**ids), _fill(body, ids)
        r_read = tc.request(method, url, json=payload, headers=auth(token(scopes=(READ,))))
        r_write = tc.request(method, url, json=payload, headers=auth(token(scopes=(WRITE,))))
        if scope == "read":
            assert r_read.status_code == 200, (method, path, r_read.text)
            assert r_write.status_code == 403, (method, path)
        else:
            assert r_read.status_code == 403, (method, path)
            assert r_write.status_code not in (401, 403), (method, path, r_write.text)


def test_the_tenant_comes_from_the_token_only(client):
    tc, store = client
    ids = _world(tc, store)
    g = auth(token("globex"))
    assert tc.get(f"/api/projects/{ids['pid']}", headers=g).status_code == 404
    assert tc.get(f"/api/projects/{ids['pid']}?tenant=acme", headers=g).status_code == 404
    assert tc.get("/api/projects", headers={**g, "X-Kalos-Tenant": "acme"}).json()["projects"] == []
    smuggled = tc.post("/api/projects", headers=g, json={"name": "Q", "objective": "o", "tenant": "acme"})
    assert smuggled.status_code == 422  # unknown keys are refused, so no tenant field exists
    link = tc.post("/api/lineage/links", headers=g, json={
        "parent_campaign_id": ids["cid"], "child_campaign_id": ids["cid2"], "relation": "replicate"})
    assert link.status_code == 404  # acme's campaigns are invisible to a globex token


@pytest.mark.parametrize("make", [
    lambda: token(key=ATTACKER),                       # signed by a key not in the JWKS
    hs256_with_public_key,                             # algorithm confusion
    lambda: token(exp=int(time.time()) - 3600, iat=int(time.time()) - 3660, nbf=int(time.time()) - 3660),
    lambda: token(aud="someone-else"),
    lambda: token(scopes=("campaigns:admin:all",)),    # an unknown scope is not ignored
    lambda: "not.a.token",
])
def test_bad_tokens_get_the_generic_401_on_records_routes(client, make):
    tc, _ = client
    for method, path in (("GET", "/api/projects"), ("POST", "/api/projects"), ("GET", "/api/units")):
        r = tc.request(method, path, headers=auth(make()), json={"name": "n", "objective": "o"})
        assert r.status_code == 401, (path, r.text)
        assert r.json() == {"error": "invalid or expired token", "code": "unauthenticated"}
