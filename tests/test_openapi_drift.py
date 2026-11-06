"""docs/openapi.json is the committed engine spec; it must match the live app.

Regenerate with `python scripts/export_openapi.py` in the same change as any
route, parameter or body edit. Also checks the `X-Kalos-Contract` header (on
every response, refusals included) and that `/api/health` reports the same
number as `contract_version`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal.app import app, get_campaign_store
from kalos.portal.campaign import CampaignStore
from kalos.portal.contract import CONTRACT_HEADER, CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "openapi.json"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_openapi_matches_the_app():
    committed = SPEC.read_text(encoding="utf-8")
    generated = _exporter().render()
    if committed != generated:
        old, new = json.loads(committed), json.loads(generated)
        added = sorted(set(new.get("paths", {})) - set(old.get("paths", {})))
        removed = sorted(set(old.get("paths", {})) - set(new.get("paths", {})))
        pytest.fail(
            "docs/openapi.json is out of date: run `python scripts/export_openapi.py` and commit it "
            f"(paths added {added}, removed {removed}; other differences are in schemas/parameters). "
            "If a field was removed or changed meaning, bump the contract version too."
        )


def test_spec_records_the_contract_version():
    assert json.loads(SPEC.read_text(encoding="utf-8"))["info"]["x-kalos-contract"] == CONTRACT_VERSION


@pytest.fixture
def client(tmp_path):
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def test_every_response_carries_the_contract_header(client):
    for path in ("/healthz", "/api/health", "/api/no-such-route"):
        response = client.get(path)
        assert response.headers.get(CONTRACT_HEADER) == str(CONTRACT_VERSION), path


def test_health_reports_the_same_contract_version(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["contract_version"] == CONTRACT_VERSION
    assert response.headers[CONTRACT_HEADER] == str(response.json()["contract_version"])


def test_header_is_added_once():
    assert sum(1 for m in app.user_middleware if m.cls.__name__ == "ContractHeaderMiddleware") == 1
