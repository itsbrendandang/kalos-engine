"""KALOS_ENV=production (P0 hardening).

The production posture is decided at import (startup), so every production
case runs in a SUBPROCESS with its own environment: re-importing
`kalos.portal.app` in-process would poison the module identity other tests
rely on (see tests/test_hardening.py::test_sizing_env_knobs_treat_empty_string_as_unset).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("fastapi")

from kalos.portal.config import (  # noqa: E402
    ProductionConfigError,
    assert_production_posture,
    deployment_env,
    is_production,
)

_TOKENS = json.dumps([{
    "token_sha256": hashlib.sha256(b"tok").hexdigest(),
    "subject": "s", "tenant": "acme", "scopes": ["read", "write"],
}])


def _run(script: str, tmp_path, **env: str) -> subprocess.CompletedProcess:
    full_env = {k: v for k, v in os.environ.items()
                if k not in {"KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE",
                             "KALOS_ALLOW_OPEN_ACCESS", "KALOS_ENV"}}
    full_env.update({"KALOS_STATE_DIR": str(tmp_path)}, **env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=full_env, capture_output=True, text=True, timeout=180,
    )


_PROBE = """
    import json
    from fastapi.testclient import TestClient
    from kalos.portal.app import app
    c = TestClient(app)
    auth = {"Authorization": "Bearer tok"}
    out = {}
    for p in ["/docs", "/redoc", "/openapi.json", "/healthz", "/readyz"]:
        out[p] = c.get(p).status_code
    for p in ["/api/single", "/api/multi"]:
        out[p] = c.get(p, params={"rounds": 1, "q": 1}, headers=auth).status_code
        out[p + " anon"] = c.get(p, params={"rounds": 1, "q": 1}).status_code
    out["/api/latest"] = c.get("/api/latest", headers=auth).status_code
    print(json.dumps(out))
"""


def test_production_without_tokens_refuses_to_start(tmp_path):
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production")
    assert r.returncode != 0
    assert "ProductionConfigError" in r.stderr
    assert "requires authentication" in r.stderr


def test_production_ignores_the_open_access_escape_hatch(tmp_path):
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production",
             KALOS_ALLOW_OPEN_ACCESS="1")
    assert r.returncode != 0
    assert "requires authentication" in r.stderr


def test_production_with_an_empty_token_list_refuses_to_start(tmp_path):
    """`[]` is 'configured' but enforces nothing; it must not pass."""
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production", KALOS_AUTH_TOKENS="[]")
    assert r.returncode != 0
    assert "requires authentication" in r.stderr


def test_unknown_kalos_env_refuses_to_start(tmp_path):
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="prod")
    assert r.returncode != 0
    assert "not recognised" in r.stderr
    assert "Accepted values: 'development', 'production'" in r.stderr


def test_production_disables_docs_and_demo_routes(tmp_path):
    r = _run(_PROBE, tmp_path, KALOS_ENV="production", KALOS_AUTH_TOKENS=_TOKENS)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["/docs"] == 404
    assert out["/redoc"] == 404
    assert out["/openapi.json"] == 404
    assert out["/api/single"] == 404 and out["/api/single anon"] == 404
    assert out["/api/multi"] == 404 and out["/api/multi anon"] == 404
    # the real API and the probes keep working
    assert out["/healthz"] == 200
    assert out["/readyz"] == 200
    assert out["/api/latest"] == 200


def test_development_keeps_docs_and_demo_routes(tmp_path):
    script = """
        import json
        from fastapi.testclient import TestClient
        from kalos.portal.app import app
        c = TestClient(app)
        print(json.dumps({p: c.get(p).status_code for p in ["/docs", "/redoc", "/openapi.json"]}
                         | {"single": c.get("/api/single", params={"rounds": 0}).status_code}))
    """
    r = _run(script, tmp_path, KALOS_ENV="development")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"/docs": 200, "/redoc": 200, "/openapi.json": 200, "single": 422}  # route exists


def test_production_fails_closed_if_tokens_vanish_at_runtime(tmp_path):
    """Tokens are re-read per request; emptying them after startup must not
    silently reopen the API in production - not even to local callers."""
    tokens_file = tmp_path / "tokens.json"
    tokens_file.write_text(_TOKENS)
    script = f"""
        import json, pathlib
        from fastapi.testclient import TestClient
        from kalos.portal.app import app
        c = TestClient(app)
        before = c.get("/api/latest", headers={{"Authorization": "Bearer tok"}}).status_code
        pathlib.Path({str(tokens_file)!r}).write_text("[]")
        after = c.get("/api/latest").status_code
        health = c.get("/healthz").status_code
        print(json.dumps([before, after, health]))
    """
    r = _run(script, tmp_path, KALOS_ENV="production", KALOS_AUTH_TOKENS_FILE=str(tokens_file))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [200, 503, 200]


def test_main_entrypoint_refuses_production_without_tokens_even_on_loopback(tmp_path):
    script = """
        import kalos.portal.__main__ as m
        m.main()
    """
    # The app module is not imported by __main__ before uvicorn runs, so this
    # exercises the entrypoint's own check (exit 2, message on stderr).
    r = _run(script, tmp_path, KALOS_ENV="production", KALOS_HOST="127.0.0.1")
    assert r.returncode == 2
    assert "requires authentication" in r.stderr


# --- in-process unit checks of the config helpers --------------------------- #

def test_config_helpers(monkeypatch):
    monkeypatch.delenv("KALOS_ENV", raising=False)
    assert deployment_env() == "development" and not is_production()
    assert_production_posture(auth_enforced=False)  # no-op outside production
    monkeypatch.setenv("KALOS_ENV", " Production ")
    assert is_production()
    assert_production_posture(auth_enforced=True)
    with pytest.raises(ProductionConfigError):
        assert_production_posture(auth_enforced=False)
    monkeypatch.setenv("KALOS_ENV", "staging")
    with pytest.raises(ProductionConfigError, match="Accepted values: 'development', 'production'"):
        deployment_env()
