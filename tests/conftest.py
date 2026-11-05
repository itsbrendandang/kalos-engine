"""Suite-wide safety net: no test may touch the real `~/.kalos/portal.db`.

Opening a `CampaignStore` on a schema-v1 database migrates it (after taking a
backup), so a test that reaches the lazily-built `get_campaign_store()`
singleton without pinning it would otherwise migrate a developer's real state
directory. Every test gets a private `KALOS_STATE_DIR` and an unbuilt
singleton; tests that pin their own store (`monkeypatch.setattr(campaign,
"_STORE", ...)`) still override this, since their fixtures run later.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_kalos_state(tmp_path_factory, monkeypatch):
    state = tmp_path_factory.mktemp("kalos-state")
    monkeypatch.setenv("KALOS_STATE_DIR", str(state))
    try:
        from kalos.portal import campaign
    except ImportError:  # the portal extra is not installed
        return
    monkeypatch.setattr(campaign, "_STORE", None)


# Opt-in: run the WHOLE selected test set on PostgreSQL. With
# KALOS_TEST_PG_EVERY_TEST set to a server URL, every test gets a fresh schema
# and KALOS_DATABASE_URL pointing at it, so each CampaignStore it opens is a
# PostgreSQL store. Tests that open portal.db with sqlite3 directly (migration
# and trigger tests of the SQLite path) are expected to fail in this mode; it
# is a cross-check for the store-level tests, not a CI default. Unset: no effect.
@pytest.fixture(autouse=True)
def _postgres_every_test(monkeypatch):
    import os

    base = os.environ.get("KALOS_TEST_PG_EVERY_TEST", "").strip()
    if not base:
        yield
        return
    import uuid
    from urllib.parse import quote

    import psycopg

    from kalos.portal import storage

    schema = "ka_" + uuid.uuid4().hex[:16]
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE SCHEMA "{schema}"')
    opt = quote(f"-csearch_path={schema}", safe="")
    monkeypatch.setenv("KALOS_DATABASE_URL", f"{base}{'&' if '?' in base else '?'}options={opt}")
    pools: list = []
    real_init = storage.PgPool.__init__

    def tracking_init(self, dsn):
        real_init(self, dsn)
        pools.append(self)

    monkeypatch.setattr(storage.PgPool, "__init__", tracking_init)
    yield
    for p in pools:
        p.close()
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'DROP SCHEMA "{schema}" CASCADE')
