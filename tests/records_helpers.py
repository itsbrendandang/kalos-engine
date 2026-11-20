"""Shared fixtures for the records tests (projects, factors, lineage, units).

`records_env` gives a TestClient over the real app with three static tokens, a
fresh `CampaignStore` in `tmp_path`, and the records backend under test:
SQLite (always) or PostgreSQL (only when KALOS_TEST_DATABASE_URL is set; each
test gets its own schema, dropped afterwards). Nothing touches ~/.kalos.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest

TEST_DSN_ENV = "KALOS_TEST_DATABASE_URL"


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": sha("a-write"), "subject": "bob@acme", "tenant": "acme", "scopes": ["read", "write"]},
    {"token_sha256": sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
    {"token_sha256": sha("b-write"), "subject": "gina@globex", "tenant": "globex", "scopes": ["read", "write"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def pg_test_dsn() -> str | None:
    return (os.environ.get(TEST_DSN_ENV) or "").strip() or None


def dsn_with_schema(dsn: str, schema: str) -> str:
    """The DSN with `options=-csearch_path=<schema>` so every connection works
    in its own schema."""
    parts = urlsplit(dsn)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "options"]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


@pytest.fixture
def pg_schema_dsn() -> Iterator[str]:
    """A DSN bound to a fresh, empty schema (dropped afterwards). Skips when no
    test PostgreSQL is configured."""
    base = pg_test_dsn()
    if base is None:
        pytest.skip(f"{TEST_DSN_ENV} is not set (no PostgreSQL to test against)")
    import psycopg

    schema = "kt_" + uuid.uuid4().hex[:12]
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    try:
        yield dsn_with_schema(base, schema)
    finally:
        from kalos.api.routers._records_common import reset_backends

        reset_backends()
        with psycopg.connect(base, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@dataclass
class Env:
    tc: Any
    store: Any
    backend: str

    def campaign(self, tenant: str = "acme", name: str = "c") -> str:
        return str(self.store.create_campaign(tenant=tenant, name=name, created_by="x")["id"])

    def audit(self, tenant: str = "acme") -> list[tuple[Any, ...]]:
        from kalos.api.routers._records_common import _backends

        records, _ = _backends(self.store.state_dir)
        with records.txn(write=False) as conn:
            return [tuple(r) for r in conn.execute(
                "SELECT action, outcome, subject, campaign_id, detail, request_id FROM audit_log "
                "WHERE tenant = ? ORDER BY id", (tenant,)).fetchall()]


# Every records route with the scope it needs, and a body that passes validation
# (ids filled from the per-test world). tests/test_records_api.py and
# tests/test_records_jwt.py both walk this table.
ROUTES = [
    ("GET", "/api/project-statuses", "read"),
    ("GET", "/api/projects", "read"),
    ("POST", "/api/projects", "write"),
    ("GET", "/api/projects/{pid}", "read"),
    ("PATCH", "/api/projects/{pid}", "write"),
    ("PUT", "/api/projects/{pid}/campaigns/{cid}", "write"),
    ("DELETE", "/api/projects/{pid}/campaigns/{cid}", "write"),
    ("GET", "/api/projects/{pid}/factors", "read"),
    ("PUT", "/api/projects/{pid}/factors", "write"),
    ("DELETE", "/api/projects/{pid}/factors?name=x", "write"),
    ("GET", "/api/projects/{pid}/factors/setup-bounds", "read"),
    ("GET", "/api/projects/{pid}/outcomes", "read"),
    ("GET", "/api/projects/{pid}/lineage", "read"),
    ("GET", "/api/lineage/relations", "read"),
    ("POST", "/api/lineage/links", "write"),
    ("GET", "/api/lineage/links/{lid}", "read"),
    ("POST", "/api/lineage/links/{lid}/retract", "write"),
    ("GET", "/api/campaigns/{cid}/lineage", "read"),
    ("GET", "/api/units", "read"),
]
BODIES = {
    ("POST", "/api/projects"): {"name": "n2", "objective": "o"},
    ("PATCH", "/api/projects/{pid}"): {"status": "completed"},
    ("PUT", "/api/projects/{pid}/factors"): {"name": "x", "lower_bound": 0, "upper_bound": 1},
    ("POST", "/api/lineage/links"): {"parent_campaign_id": "{cid}", "child_campaign_id": "{cid2}", "relation": "replicate"},
    ("POST", "/api/lineage/links/{lid}/retract"): {"reason": "r"},
}


BACKENDS = ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]


@pytest.fixture(params=BACKENDS)
def records_env(request, tmp_path, monkeypatch) -> Iterator[Env]:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.api.routers._records_common import reset_backends, reset_refusal_budget
    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    if request.param == "postgres":
        dsn = request.getfixturevalue("pg_schema_dsn")
        monkeypatch.setenv("KALOS_DATABASE_URL", dsn)
    else:
        monkeypatch.delenv("KALOS_DATABASE_URL", raising=False)
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    reset_refusal_budget()
    try:
        yield Env(TestClient(app), store, request.param)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        reset_backends()
