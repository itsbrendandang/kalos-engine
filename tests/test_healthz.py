"""GET /healthz and GET /readyz (deploy/engine/Containerfile's HEALTHCHECK target).

Before this, the only unauthenticated route the deploy healthcheck could hit
was `/`, which renders the full portal page - heavier than a healthcheck
needs, and coupled to the UI. `/healthz` is a liveness probe (process is up,
never touches the database); `/readyz` is a readiness probe (the campaign
store is actually reachable).

Every test here uses a `tmp_path` `CampaignStore` via `app.dependency_overrides`,
per tests/test_tenant_isolation.py's convention - the real `~/.kalos/portal.db`
is never touched.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal.app import app, get_campaign_store  # noqa: E402
from kalos.portal.campaign import CampaignStore  # noqa: E402


@pytest.fixture
def client(tmp_path):
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    """Every test starts from genuinely open, matching
    tests/test_open_access_guard.py's fixture - no leftover auth config from
    another test's monkeypatch."""
    for var in ("KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS"):
        monkeypatch.delenv(var, raising=False)


# --- /healthz ----------------------------------------------------------- #


def test_engine_is_api_only_so_root_is_404(client):
    """The built-in viewer is gone; the UI is kalos-web."""
    assert client.get("/").status_code == 404



def test_healthz_returns_exactly_the_documented_body(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_needs_no_auth_token_even_when_auth_is_enforced(client, monkeypatch):
    """Unauthenticated BY DESIGN: provisioning real tokens (which makes every
    scoped route below start requiring one) must not affect /healthz."""
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        '[{"subject": "acme", "tenant": "acme", "scopes": ["read"], '
        '"token_sha256": "' + "a" * 64 + '"}]',
    )
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_auth_required_routes_still_401_without_a_token_once_enforced(client, monkeypatch):
    """The control case for the test above: /healthz being open must not mean
    auth stopped being enforced everywhere else."""
    monkeypatch.setenv(
        "KALOS_AUTH_TOKENS",
        '[{"subject": "acme", "tenant": "acme", "scopes": ["read"], '
        '"token_sha256": "' + "a" * 64 + '"}]',
    )
    resp = client.get("/api/latest")
    assert resp.status_code == 401


def test_healthz_does_not_touch_the_database(client, monkeypatch):
    """Liveness must survive a store that would raise on any real query -
    proves /healthz never calls into the store at all."""

    def _boom(*args, **kwargs):
        raise AssertionError("healthz must never touch the store")

    monkeypatch.setattr(CampaignStore, "ping", _boom)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- /readyz -------------------------------------------------------------- #


def test_readyz_is_ok_on_a_working_store(client):
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_degrades_to_503_with_a_generic_reason_when_the_store_is_broken(
    client, monkeypatch, caplog
):
    def _boom(*args, **kwargs):
        raise RuntimeError("database disk image is malformed at /srv/secret/portal.db")

    monkeypatch.setattr(CampaignStore, "ping", _boom)
    with caplog.at_level("ERROR", logger="kalos.portal"):
        resp = client.get("/readyz")
    assert resp.status_code == 503
    body = resp.json()
    # Unauthenticated endpoint: a fixed reason, never the exception text.
    assert body == {"status": "unavailable", "reason": "campaign store unavailable"}
    assert "malformed" not in resp.text and "/srv/secret" not in resp.text
    # ...but the operator still gets the detail, server-side.
    assert "malformed" in caplog.text


def test_readyz_does_not_read_tenant_data(client, monkeypatch):
    """Readiness is tenant-agnostic: it pings the store, never fetches a
    tenant's campaign row."""
    def _no_tenant_reads(*args, **kwargs):
        raise AssertionError("readyz must not read a tenant's campaign")

    monkeypatch.setattr(CampaignStore, "get", _no_tenant_reads)
    assert client.get("/readyz").json() == {"status": "ok"}


def test_healthz_is_an_async_handler():
    """A sync handler would run on the bounded threadpool and could queue
    behind upload/reanalyze work; liveness must answer from the event loop."""
    import inspect

    from kalos.portal import app as portal

    assert inspect.iscoroutinefunction(portal.healthz)


def test_readyz_performs_a_read_not_a_write(client):
    """Readiness must not mutate the store it is checking - no campaign
    should appear as a side effect of polling /readyz."""
    store = app.dependency_overrides[get_campaign_store]()
    assert store.get(tenant="default") is None  # nothing seeded yet
    client.get("/readyz")
    assert store.get(tenant="default") is None  # /readyz did not seed one


def test_readyz_is_a_generic_503_when_the_store_cannot_even_be_built(monkeypatch, caplog):
    """Building the store (opening/migrating portal.db) happens inside the
    readiness try, so a failure there is the same generic 503, not a 500."""
    def _cannot_open():
        raise PermissionError("unable to open database file /srv/secret/portal.db")

    app.dependency_overrides[get_campaign_store] = _cannot_open
    try:
        with caplog.at_level("ERROR", logger="kalos.portal"):
            resp = TestClient(app, raise_server_exceptions=False).get("/readyz")
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable", "reason": "campaign store unavailable"}
    assert "/srv/secret" not in resp.text
    assert "/srv/secret" in caplog.text


# --- /readyz checks writability and free space (D4) ----------------------- #


def test_readyz_503_when_the_store_accepts_reads_but_not_writes(client):
    """A store that passes the read `ping` but cannot write must NOT be ready.
    `query_only` makes the connection refuse writes, the same SQLITE_READONLY
    a read-only or full volume produces."""
    store = app.dependency_overrides[get_campaign_store]()
    store.ping()  # reads still work
    store._conn.execute("PRAGMA query_only = 1")
    try:
        resp = client.get("/readyz")
    finally:
        store._conn.execute("PRAGMA query_only = 0")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable", "reason": "campaign store unavailable"}
    assert "readonly" not in resp.text and "read-only" not in resp.text


def test_check_writable_does_not_change_data(tmp_path):
    """The write probe takes the lock and commits nothing: a row count and the
    schema version are the same before and after."""
    from kalos.portal.campaign import CampaignStore

    store = CampaignStore(tmp_path)
    before = store._conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()
    store.check_writable()
    store.check_writable()
    assert store._conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == before
    assert not store._conn.in_transaction
    store.close()


def test_readyz_503_when_the_state_volume_is_below_the_free_space_floor(client, monkeypatch, caplog):
    monkeypatch.setenv("KALOS_READY_MIN_FREE_MB", str(10**9))  # ~1 PB: never satisfied
    with caplog.at_level("ERROR", logger="kalos.portal"):
        resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable", "reason": "state volume is low on space"}
    # the body names no byte counts and no path ...
    assert not any(ch.isdigit() for ch in resp.text)
    assert str(client.app.dependency_overrides[get_campaign_store]().state_dir) not in resp.text
    # ... but the operator sees the number server-side
    assert "bytes free on the state volume" in caplog.text


def test_readyz_free_space_floor_of_zero_disables_the_check(client, monkeypatch):
    monkeypatch.setenv("KALOS_READY_MIN_FREE_MB", "0")
    assert client.get("/readyz").status_code == 200


def test_readyz_503_when_free_space_cannot_be_read(client, monkeypatch):
    from kalos.portal.campaign import CampaignStore

    def _no_stat(self):
        raise OSError("statvfs failed on /srv/secret")

    monkeypatch.setattr(CampaignStore, "free_bytes", _no_stat)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {"status": "unavailable", "reason": "state volume unavailable"}
    assert "/srv/secret" not in resp.text


def test_bad_free_space_setting_is_a_config_error(monkeypatch):
    from kalos.portal.config import ConfigError, ready_min_free_bytes

    monkeypatch.setenv("KALOS_READY_MIN_FREE_MB", "lots")
    with pytest.raises(ConfigError, match="KALOS_READY_MIN_FREE_MB must be a number"):
        ready_min_free_bytes()
