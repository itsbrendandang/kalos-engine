"""principal_for / require_principal fail closed in production (review fix).

The production middleware checks that tokens are loaded, but the dependency
re-reads them; a tokens file emptied between the two must be a 503, never the
anonymous read+write principal. The app module here was imported in
development mode, so its middleware lets the request through and only the
dependency stands between the caller and anonymous access - exactly the race.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app  # noqa: E402
from kalos.portal.auth import Authenticator, require_principal  # noqa: E402


@pytest.fixture(autouse=True)
def _no_tokens(monkeypatch):
    for var in ("KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS", "KALOS_ENV"):
        monkeypatch.delenv(var, raising=False)


def test_principal_for_raises_503_in_production_with_no_tokens(monkeypatch):
    monkeypatch.setenv("KALOS_ENV", "production")
    with pytest.raises(HTTPException) as exc:
        Authenticator().principal_for(None)
    assert exc.value.status_code == 503


def test_require_principal_raises_503_in_production_with_an_empty_token_list(monkeypatch):
    monkeypatch.setenv("KALOS_ENV", "production")
    monkeypatch.setenv("KALOS_AUTH_TOKENS", "[]")
    with pytest.raises(HTTPException) as exc:
        require_principal(authorization="Bearer anything", auth=Authenticator())
    assert exc.value.status_code == 503


def test_tokens_emptied_after_the_middleware_check_is_503_not_anonymous(monkeypatch, tmp_path):
    monkeypatch.setenv("KALOS_ENV", "production")
    tokens = tmp_path / "tokens.json"
    tokens.write_text("[]")
    monkeypatch.setenv("KALOS_AUTH_TOKENS_FILE", str(tokens))
    r = TestClient(app).get("/api/latest")
    assert r.status_code == 503
    assert r.json() == {"detail": "authentication is not configured"}


def test_development_open_mode_is_still_anonymous_default():
    p = Authenticator().principal_for(None)
    assert p.anonymous and p.tenant == "default"
