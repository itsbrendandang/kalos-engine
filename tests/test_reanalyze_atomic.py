"""Replay F1: a reanalyze must never commit a fold it cannot report.

With 2 CV groups the grouped-CV interval came back as NaN, the JSON encoder
refused it after the fold was committed, and the client saw a 500 for a change
the server kept (`/api/latest` stale). Now the interval is null with a reason,
a last pass turns any non-finite float into null, and the reanalyze checks the
analysis encodes before it commits anything.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.domains import ColumnRoles
from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.analysis import _analyze
from kalos.portal.analysis_parts.sanitize import sanitize_non_finite
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store


def _two_group_sheet() -> pd.DataFrame:
    """14 rows, one categorical input with 2 levels: the recipe grouping
    gives exactly 2 CV groups (the Trichoderma condition-only shape)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "strain": ["T2"] * 14,
        "condition": ["T2 pH 4"] * 7 + ["T2 pH 3.5"] * 7,
        "mucic_acid_g_L": np.r_[rng.normal(50, 3, 7), rng.normal(35, 3, 7)].round(3),
    })


def test_two_cv_groups_give_a_null_ci_with_a_reason_and_encode():
    roles = ColumnRoles.from_dict(
        {"target": "mucic_acid_g_L", "categoricals": ["condition"], "ids": ["strain"]}
    )
    result = _analyze(_two_group_sheet(), roles=roles)
    assert result["cv_n_groups"] == 2
    assert result["cv_ci95"] is None
    assert result["reliability"]["ci95"] is None
    assert "3 CV groups" in result["cv_ci95_reason"]
    checks = [f["check"] for f in result["validation"]["findings"]]
    assert "cv_ci_unavailable" in checks
    json.dumps(result, allow_nan=False)  # must not raise


def test_sanitizer_turns_non_finite_floats_into_none():
    raw = {"a": float("nan"), "b": [1.0, float("inf"), (np.float64("nan"), 2)],
           "c": {"d": -float("inf"), "e": True, "f": 3}}
    cleaned, hits = sanitize_non_finite(raw)
    assert cleaned == {"a": None, "b": [1.0, None, [None, 2]],
                       "c": {"d": None, "e": True, "f": 3}}
    assert sorted(hits) == ["a", "b[1]", "b[2][0]", "c.d"]
    json.dumps(cleaned, allow_nan=False)


# --- HTTP: nothing is committed when the analysis fails ---------------------- #

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    return CampaignStore(tmp_path)


@pytest.fixture
def client(store, tmp_path, monkeypatch):
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _seeded_with_one_measured_run(store, client) -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Methanol": rng.uniform(0, 4, 8).round(3),
                       "pH": rng.uniform(5, 7, 8).round(2),
                       "lipase_titer": rng.uniform(1, 5, 8).round(3)})
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="default",
               analysis={"target": "lipase_titer", "n": 8, "marker": "round-0"})
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}},
        {"recipe": {"Methanol": 1.0, "pH": 5.5}},
    ]}).json()["started"]
    client.post("/api/campaign/result", json={"id": started[0]["id"], "value": 6.0})


def _snapshot(store, client):
    state = store.get(tenant="default")
    return (state["generation"], json.dumps(state["pending"], sort_keys=True),
            len(state["base_rows"]), state["round"], client.get("/api/latest").json())


@pytest.mark.parametrize("failure", ["raises", "nan"])
def test_failed_reanalyze_commits_nothing(store, client, monkeypatch, failure):
    _seeded_with_one_measured_run(store, client)
    before = _snapshot(store, client)
    assert before[4].get("marker") == "round-0"

    def broken(*_a, **_k):
        if failure == "raises":
            raise RuntimeError("boom")
        # An analysis that slipped a NaN past the sanitizer.
        return {"target": "lipase_titer", "d": 2, "cv_ci95": [float("nan"), float("nan")]}

    monkeypatch.setattr(portal_module, "_analyze", broken)
    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "analysis_failed"
    assert _snapshot(store, client) == before


def test_commit_fold_refuses_an_unencodable_analysis(store, client):
    _seeded_with_one_measured_run(store, client)
    before = _snapshot(store, client)
    generation = store.get(tenant="default")["generation"]
    with pytest.raises(campaign_module.CampaignError):
        store.commit_fold(generation, tenant="default", analysis={"x": float("nan")})
    assert _snapshot(store, client) == before
