"""The run record on a campaign re-analysis (`POST /api/campaign/reanalyze`).

Before this, the re-analysis bypassed `compute_upload_analysis`, so the stored
analysis of a campaign had no `run` block at all, and nothing recorded how many
runs were in flight. The re-analysis now attaches `kind="reanalyze"` with the
folded dataset as its input, and it keeps the anonymize setting of the analysis
it replaces (the campaign does not store the flag itself).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal import run_record
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store


def _df(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2), "lipase_titer": titer.round(3)})


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


def _loop_one_round(client):
    """Seed, start three runs, measure two; one stays in the incubator."""
    started = client.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.1, "std": 0.4,
         "mode": "explore", "reason": "a"},
        {"recipe": {"Methanol": 3.0, "pH": 6.5}, "pred": 6.2, "std": 0.2,
         "mode": "exploit", "reason": "b"},
        {"recipe": {"Methanol": 1.0, "pH": 5.5}, "pred": 3.0, "std": 0.5,
         "mode": "explore", "reason": "c"},
    ]}).json()["started"]
    client.post("/api/campaign/result", json={"id": started[0]["id"], "value": 6.0})
    client.post("/api/campaign/result", json={"id": started[1]["id"], "value": 7.5})


def test_reanalyze_attaches_a_run_block_and_stores_it(store, client):
    df = _df()
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="default")
    _loop_one_round(client)
    expected_df, _, _ = store.plan_fold(tenant="default")  # pure: what the fit will see

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    run = resp.json()["analysis"]["run"]

    assert run["kind"] == "reanalyze"
    assert run["input_sha256"] == hashlib.sha256(run_record.fold_bytes(expected_df)).hexdigest()
    assert run["n_rows"] == len(expected_df) == len(df) + 2
    assert run["n_features"] == 2
    assert run["pending_runs"] == 1
    assert run["target"] == "lipase_titer"
    assert run["gates"] == {"promotion": "promotion", "proposal_gating": "proposal_gating",
                            "validation": "validation"}
    # the record is stored with the analysis, not only returned
    assert store.load_analysis(tenant="default")["run"] == run


def test_reanalyze_of_an_anonymized_analysis_stays_anonymized(store, client):
    """Regression: the re-analysis ran with anonymize=False, so an anonymized
    upload's campaign published its identifier column names verbatim again."""
    df = _df().assign(run_number=list(range(8)))
    stored_run = {"anonymize": True, "input_sha256": "ab" * 32}
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="default",
               analysis={"run": stored_run, "target": "lipase_titer"})
    _loop_one_round(client)

    resp = client.post("/api/campaign/reanalyze")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "run_number" not in json.dumps(body["analysis"])
    assert body["analysis"]["run"]["anonymize"] is True
    assert "filename" not in body["analysis"]["run"]


def test_reanalyze_of_a_plain_analysis_is_not_anonymized(store, client):
    df = _df().assign(run_number=list(range(8)))
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="default")
    _loop_one_round(client)
    body = client.post("/api/campaign/reanalyze").json()
    assert body["analysis"]["run"]["anonymize"] is False
