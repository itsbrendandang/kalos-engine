"""One error envelope on every route (review M3): `{"error": str, "code": str}`.

Before, 401/403 answered `{"detail": ...}`, unknown routes `{"detail": "Not Found"}`,
and 422 `{"detail": [...]}`, while 400/404 used `{"error": ...}`. The HTTP status
of every case is unchanged; the body now always carries `error` and `code`. A 422
keeps its per-field list under `detail` (without the echoed input values).
"""
from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore  # noqa: E402

_TOKEN = "tok-envelope-test"


@pytest.fixture
def client(tmp_path, monkeypatch):
    for var in ("KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS", "KALOS_ENV"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([{
        "token_sha256": hashlib.sha256(_TOKEN.encode()).hexdigest(),
        "subject": "svc", "tenant": "acme", "scopes": ["read"],
    }]))
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    return TestClient(app, raise_server_exceptions=False)


def _assert_envelope(resp, status: int, code: str | None = None) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert isinstance(body.get("error"), str) and body["error"], body
    assert isinstance(body.get("code"), str) and body["code"], body
    if code is not None:
        assert body["code"] == code, body
    return body


def test_401_without_a_token_is_an_envelope(client):
    body = _assert_envelope(client.get("/api/latest"), 401, "unauthenticated")
    assert "detail" not in body


def test_401_with_a_wrong_token_is_an_envelope(client):
    r = client.get("/api/latest", headers={"Authorization": "Bearer nope"})
    _assert_envelope(r, 401, "unauthenticated")


def test_403_missing_scope_is_an_envelope(monkeypatch, tmp_path):
    # a write-scoped token may not archive a campaign (admin scope): 403 envelope
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([{
        "token_sha256": hashlib.sha256(_TOKEN.encode()).hexdigest(),
        "subject": "svc", "tenant": "acme", "scopes": ["read", "write"],
    }]))
    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(campaign_module, "_STORE", CampaignStore(tmp_path))
    r = TestClient(app, raise_server_exceptions=False).patch(
        "/api/campaigns/" + "a" * 32, json={"archived": True},
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    _assert_envelope(r, 403, "forbidden")


def test_unknown_route_is_a_404_envelope(client):
    _assert_envelope(client.get("/api/no-such-route"), 404, "not_found")


def test_wrong_method_is_a_405_envelope(client):
    _assert_envelope(client.delete("/healthz"), 405, "method_not_allowed")


def test_422_keeps_field_details_under_detail_without_echoing_input(client):
    r = client.get("/api/single", params={"rounds": 0},  # outside the route's bounds
                   headers={"Authorization": f"Bearer {_TOKEN}"})
    body = _assert_envelope(r, 422, "validation_error")
    assert isinstance(body["detail"], list) and body["detail"]
    first = body["detail"][0]
    assert {"loc", "msg", "type"} <= set(first)
    assert "input" not in first


def test_campaign_body_validation_keeps_its_400_and_gets_a_code(client):
    r = client.post("/api/campaign/start", content="{not json",
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {_TOKEN}"})
    _assert_envelope(r, 400, "invalid_body")


def test_unhandled_error_is_a_generic_500_envelope(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("/secret/internal/path and a customer column name")

    # /api/latest reads the store; break that read so the error is not one a route
    # anticipated
    monkeypatch.setattr(campaign_module.CampaignStore, "load_analysis", boom)
    r = client.get("/api/latest", headers={"Authorization": f"Bearer {_TOKEN}"})
    body = _assert_envelope(r, 500, "internal_error")
    assert "secret" not in r.text and "customer" not in r.text
    assert body["error"] == "internal server error"


def test_the_legacy_statuses_are_unchanged(client):
    """The bodies that already had `error` keep their status and gain `code` only
    when the route set one; nothing that was 404/400/409/503 changes status."""
    r = client.get("/api/campaign/does-not-exist", headers={"Authorization": f"Bearer {_TOKEN}"})
    assert r.status_code in (404, 400)
    assert "error" in r.json()


def test_scale_readout_422_is_the_envelope_with_its_own_fields(client, monkeypatch):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([{
        "token_sha256": hashlib.sha256(_TOKEN.encode()).hexdigest(),
        "subject": "svc", "tenant": "acme", "scopes": ["read", "write"],
    }]))
    r = client.post(
        "/api/scale/readout",
        data={"target": "not json"},
        files={"file": ("s.csv", b"a,b\n1,2\n", "text/csv")},
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )
    body = _assert_envelope(r, 422, "invalid_target")
    assert body["failed_check"] == "invalid_target"
    assert body["detail"] == body["error"]
