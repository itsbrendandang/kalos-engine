"""The first-time-user study path end to end with the real engine (no stub):
upload with declared roles (incl. allow_beyond) -> plan -> start -> record 2
-> reanalyze. Covers Findings 1, 3, 4, 5, 7, 11, 13 and 17 together on one
campaign so the pieces from different commits agree."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

pytest.importorskip("botorch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store


def _csv(n: int = 30, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    lines = ["Carbon_conc,Gly,SP,Titer"]
    for _ in range(n):
        cc, gly, sp = rng.uniform(10, 40), rng.uniform(0, 10), rng.uniform(2, 10)
        titer = 20 + 4 * sp - 0.3 * (gly - 3) ** 2 + 0.2 * cc + rng.normal(0, 1)
        lines.append(f"{cc:.2f},{gly:.2f},{sp:.2f},{titer:.2f}")
    lines.append("25.00,3.00,6.00,999")  # an outlier with a known sheet row
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def test_one_round_end_to_end(client):
    c, _store = client
    roles = {"target": "Titer", "features": ["Carbon_conc", "Gly", "SP"],
             "allow_beyond": {"SP": 0.1}}
    r = c.post("/api/run", files={"file": ("sheet.csv", io.BytesIO(_csv()), "text/csv")},
               data={"target": "Titer", "roles": json.dumps(roles)})
    assert r.status_code == 200, r.text
    up = r.json()
    json.dumps(up, allow_nan=False)

    # Finding 1: a best bet, first, never an all-exploration batch.
    assert up["proposals"][0]["pick"] == "best_bet"
    assert up["model_optimum"]["in_batch"] == 0
    assert all("predicted_vs_best" in p for p in up["proposals"])
    # Finding 11: SP may go 10% past the tried range; the finding says so.
    assert up["allow_beyond"]["applied"] == {"SP": 0.1}
    assert any(f["code"] == "search_box" for f in up["validation"]["findings"])
    # Finding 3: one verdict object, the same on every read route.
    verdict = up["verdict"]
    assert verdict["state"] in {"reliable", "not_yet", "refused"}
    latest = c.get("/api/latest").json()
    camp = c.get("/api/campaign").json()
    assert latest["verdict"] == verdict == camp["verdict"]
    # Finding 4: one analysis id everywhere.
    assert latest["analysis_id"] == up["analysis_id"] == camp["analysis_id"]
    # Finding 13: the outlier's sheet row (header = row 1) is reported.
    titer_out = [f for f in up["validation"]["findings"]
                 if f["code"] == "outliers" and f.get("column") == "Titer"]
    assert titer_out and 32 in titer_out[0]["sheet_rows"]
    # Finding 17: every promotion check says what it checks.
    assert all(ch["checks"] and ch["why"] for ch in up["promotion"]["checks"])

    started = c.post("/api/campaign/start", json={"recipes": up["proposals"][:3]}).json()["started"]
    for run, value in zip(started[:2], [80.0, 61.0]):
        assert c.post("/api/campaign/result", json={"id": run["id"], "value": value}).status_code == 200
    body = c.post("/api/campaign/reanalyze")
    assert body.status_code == 200, body.text
    after = body.json()["analysis"]
    json.dumps(after, allow_nan=False)

    # Finding 7: declared inputs and the allow_beyond setting survive the round.
    assert after["features"] == ["Carbon_conc", "Gly", "SP"]
    assert after["roles_used"]["source"] == "declared"
    assert after["allow_beyond"]["applied"] == {"SP": 0.1}
    assert after["dataset"] == "sheet.csv"
    # Finding 5: exactly the two recorded runs, and what changed.
    used = after["runs_used"]
    assert used["n_recorded"] == 2
    assert {r["run_id"] for r in used["recorded"]} == {started[0]["id"], started[1]["id"]}
    assert after["delta"]["previous_analysis_id"] == up["analysis_id"]
    assert after["delta"]["summary"]
    # Finding 4: the reanalysis is a new analysis that says what it replaced.
    assert after["analysis_id"] != up["analysis_id"]
    assert after["supersedes"]["analysis_id"] == up["analysis_id"]
    assert after["proposals"][0]["pick"] == "best_bet" or after["model_optimum"]["already_running"]
    assert c.get("/api/latest").json()["verdict"] == after["verdict"] == c.get("/api/campaign").json()["verdict"]
