"""One verdict on the model (kalos/portal/verdict.py): the table, the 13-run
sheet that must not read "reliable", the refused sheet with no proposals, and
the same stored object on every route."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.verdict import (
    MIN_GROUPS_FOR_RELIABLE,
    MIN_RUNS_FOR_RELIABLE,
    model_verdict,
    range_width,
)

BANNED = ("spearman", "posterior", "auc", "ece", "calibration", "compliant", "validated")


def _v(**kw):
    base = dict(n_runs=30, n_groups=12, spearman=0.6, ci95=[0.3, 0.8], floor=0.2, calibration=None)
    base.update(kw)
    return model_verdict(**base)


@pytest.mark.parametrize(
    ("kw", "state", "code"),
    [
        ({}, "reliable", "reliable"),
        ({"n_runs": 13}, "not_yet", "not-yet"),
        ({"n_groups": MIN_GROUPS_FOR_RELIABLE - 1}, "not_yet", "not-yet"),
        ({"ci95": [-0.1, 0.8]}, "not_yet", "not-yet"),
        ({"spearman": 0.1, "ci95": [0.01, 0.3]}, "not_yet", "not-yet"),
        ({"spearman": -0.4, "ci95": [-0.7, 0.0]}, "refused", "no_ordering"),
        ({"spearman": 0.5, "ci95": None, "n_groups": 2}, "refused", "insufficient"),
        ({"spearman": None, "ci95": None}, "refused", "insufficient"),
        ({"spearman": float("nan"), "ci95": None}, "refused", "insufficient"),
    ],
)
def test_table(kw, state, code):
    v = _v(**kw)
    assert (v["state"], v["code"]) == (state, code)
    assert v["min_runs"] == MIN_RUNS_FOR_RELIABLE
    assert (v["reasons"] == []) == (state == "reliable")
    text = " ".join([v["headline"], v["ordering"]["plain"], *(r["plain"] for r in v["reasons"])])
    assert not any(w in text.lower() for w in BANNED)


def test_reliable_needs_every_part():
    v = _v(n_runs=MIN_RUNS_FOR_RELIABLE, n_groups=MIN_GROUPS_FOR_RELIABLE)
    assert v["state"] == "reliable" and v["ordering"]["state"] == "ok"
    small = _v(n_runs=13, n_groups=9, spearman=0.89, ci95=[0.51, 1.0])
    assert small["state"] == "not_yet"
    assert [r["code"] for r in small["reasons"]] == ["too_few_rows"]
    assert "13 runs" in small["headline"]


def test_range_width_mirrors_the_web():
    def cal(z):
        return {"available": True, "levels": [{"nominal": 0.9, "empirical": 0.8}], "z_std": z}

    assert range_width(None)["state"] == "unchecked"
    assert range_width(cal(1.3))["state"] == "too_narrow"
    assert range_width(cal(0.7))["state"] == "too_wide"
    assert range_width(cal(1.0))["state"] == "ok"
    assert _v(calibration=cal(1.3))["state"] == "reliable"  # range width is reported, not gated


# --- end to end ------------------------------------------------------------ #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.analysis import _analyze  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402


def _sheet13() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    a, b = rng.uniform(0, 10, 13), rng.uniform(0, 10, 13)
    y = 5.0 + 2.0 * a + 1.5 * b + rng.normal(0, 0.3, 13)
    return pd.DataFrame({"a": a.round(3), "b": b.round(3), "y": y.round(3)})


def _refused_sheet() -> pd.DataFrame:
    """Two declared batches: the ordering cannot be checked (no interval)."""
    rng = np.random.default_rng(0)
    m, ph = rng.uniform(0, 4, 12), rng.uniform(5, 7, 12)
    return pd.DataFrame({
        "batch": ["B1"] * 6 + ["B2"] * 6, "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m + rng.normal(0, 0.3, 12)).round(3),
    })


REFUSED_ROLES = {"target": "lipase_titer", "features": ["Methanol", "pH"], "groups": "batch"}


def test_13_row_sheet_is_not_reliable():
    out = _analyze(_sheet13())
    assert out["n"] == 13
    assert out["verdict"]["state"] != "reliable"
    assert out["verdict"]["n_runs"] == 13
    # The old fields are unchanged.
    assert out["reliability"]["clears_floor"] == (out["reliability"]["spearman"] >= 0.2)
    assert out["refusal"] is None and len(out["proposals"]) >= 1


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


def _routes_agree(c, store, posted: dict) -> dict:
    verdict = posted["verdict"]
    cid = store.default_campaign_id(tenant="default")
    assert c.get("/api/latest").json()["verdict"] == verdict
    assert c.get("/api/campaign").json()["verdict"] == verdict
    assert c.get(f"/api/campaigns/{cid}/analysis").json()["verdict"] == verdict
    assert c.get(f"/api/campaigns/{cid}").json()["summary"]["verdict"] == verdict
    return verdict


def test_same_verdict_on_every_route(client):
    c, store = client
    r = c.post("/api/run", files={"file": ("s.csv", io.BytesIO(_sheet13().to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 200, r.text
    v = _routes_agree(c, store, r.json())
    assert v["state"] == "not_yet"


def test_refused_sheet_has_no_proposals_and_nothing_500s(client):
    c, store = client
    r = c.post(
        "/api/run",
        files={"file": ("s.csv", io.BytesIO(_refused_sheet().to_csv(index=False).encode()), "text/csv")},
        data={"roles": json.dumps(REFUSED_ROLES)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"]["state"] == "refused"
    assert body["proposals"] == []
    assert set(body["refusal"]) == {"code", "plain", "what_would_help"}
    assert body["refusal"]["code"] == body["verdict"]["code"] == "insufficient"
    assert body["proposal_notice"]["code"] == "model_refused"
    assert body["proposal_notice"]["returned"] == 0
    assert body["contract_version"] == 3
    _routes_agree(c, store, body)
    summary = c.get("/api/campaign").json()
    assert summary["stage"]["current"] == "plan"
    assert summary["next_action"]["action"] == "Add runs and upload again"
    assert "Review and start" not in json.dumps(summary["next_action"])
    # Starting an empty batch is harmless.
    started = c.post("/api/campaign/start", json={"recipes": []})
    assert started.status_code == 200 and started.json() == {"started": []}
