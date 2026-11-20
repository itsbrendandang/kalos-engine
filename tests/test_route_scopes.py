"""The per-route scope manifest (docs/route-scopes.json, security review S1).

- the checked-in file matches the manifest regenerated from the live app
  (regenerate: `.venv/bin/python scripts/export_scope_manifest.py`);
- it covers every route in the OpenAPI document;
- a few known values are pinned;
- every entry is checked against behavior: a token lacking the listed scope
  gets that scope's 403, and an anonymous call to a `required` route gets 401.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kalos.portal.app import app
from kalos.portal.route_scopes import build_manifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "route-scopes.json"
CID = "/api/campaigns/{campaign_id}"


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return build_manifest(app)


def _index(manifest: list[dict]) -> dict[tuple[str, str], dict]:
    return {(e["method"], e["path"]): e for e in manifest}


def test_checked_in_manifest_has_not_drifted(manifest):
    spec = importlib.util.spec_from_file_location("export_scope_manifest", ROOT / "scripts" / "export_scope_manifest.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.render() == json.dumps(manifest, indent=2) + "\n"
    assert json.loads(MANIFEST.read_text()) == manifest, (
        "docs/route-scopes.json is stale: run .venv/bin/python scripts/export_scope_manifest.py"
    )


def test_manifest_is_sorted_and_well_formed(manifest):
    assert manifest == sorted(manifest, key=lambda e: (e["path"], e["method"]))
    for e in manifest:
        assert set(e) <= {"method", "path", "scopes", "auth", "conditional_scopes"}
        assert e["auth"] in ("required", "public")
        assert set(e["scopes"]) <= {"read", "write", "admin"}
        assert (e["auth"] == "public") == (not e["scopes"]), e  # every authed route names its scope


def test_manifest_covers_every_openapi_operation(manifest):
    listed = set(_index(manifest))
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            assert (method.upper(), path) in listed, (method, path)


@pytest.mark.parametrize("method,path,scopes", [
    ("GET", f"{CID}/setup", ["read"]),
    ("PUT", f"{CID}/setup", ["write"]),
    ("GET", f"{CID}/stages", ["read"]),
    ("PUT", f"{CID}/stages", ["write"]),
    ("GET", f"{CID}/gates", ["read"]),
    ("PUT", f"{CID}/gates/{{gate}}", ["write"]),
    ("PATCH", CID, ["write"]),
    ("POST", f"{CID}/operating-ranges", ["write"]),
    ("GET", f"{CID}/operating-ranges", ["write"]),  # refits the GP (E-S3)
    ("POST", "/api/scale/readout.json", ["write"]),
    ("GET", "/api/campaigns", ["read"]),
    ("POST", "/api/assist/ask", ["read"]),
    ("GET", "/healthz", []),
])
def test_known_route_scopes(manifest, method, path, scopes):
    entry = _index(manifest)[(method, path)]
    assert entry["scopes"] == scopes
    assert entry["auth"] == ("required" if scopes else "public")


def test_archive_needs_admin_and_is_declared_as_body_dependent(manifest):
    entry = _index(manifest)[("PATCH", CID)]
    assert entry["conditional_scopes"] == [{"scope": "admin", "when": "body sets 'archived' (archive/unarchive)"}]
    assert sum("conditional_scopes" in e for e in manifest) == 1


# --- the manifest agrees with what the app enforces -------------------------------- #

def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


_TOKENS = {
    "read": ("only-read", ["read"]),
    "write": ("only-write", ["write"]),
}
# For a route needing scope X: a token holding every OTHER non-admin scope.
_LACKING = {"read": "only-write", "write": "only-read"}


@pytest.fixture
def client(monkeypatch):
    records = [{"token_sha256": _sha(tok), "subject": f"{name}@acme", "tenant": "acme", "scopes": scopes}
               for name, (tok, scopes) in _TOKENS.items()]
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(records))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    with TestClient(app) as tc:
        yield tc


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x0", path)


def test_every_entry_matches_enforcement(client, manifest):
    for e in manifest:
        method, path = e["method"], _concrete(e["path"])
        if e["auth"] == "public":
            r = client.request(method, path)
            assert r.status_code not in (401, 403), (method, path, r.status_code)
            continue
        assert client.request(method, path).status_code == 401, (method, path)
        for scope in e["scopes"]:
            r = client.request(method, path, headers={"Authorization": f"Bearer {_LACKING[scope]}"})
            assert r.status_code == 403, (method, path, scope, r.text)
            assert r.json()["error"] == f"requires '{scope}' scope", (method, path, r.text)
