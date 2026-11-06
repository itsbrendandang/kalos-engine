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
    # The in-process analysis cache (kalos/portal/pipeline.py) is off by default in
    # tests, so a test that patches engine internals never reads another test's
    # analysis of the same bytes. tests/test_analysis_cache.py turns it on.
    monkeypatch.setenv("KALOS_ANALYSIS_CACHE", "0")
    try:
        from kalos.portal import campaign
    except ImportError:  # the portal extra is not installed
        return
    monkeypatch.setattr(campaign, "_STORE", None)
