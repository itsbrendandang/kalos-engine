"""Applicability flags per proposal (queue 3 + 18) and the F9 box clip.

`far_from_data` (distance vs a threshold) and `at_box_edge` (within 1% of the
box width of a bound) are separate flags. The threshold is a product default
unless a lab sets one (roles or env).
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.domains import (
    ColumnRoles,
    DesignSpace,
    Dimension,
    InvalidApplicabilityDistance,
)
from kalos.envconfig import ConfigError
from kalos.portal.analysis_parts.applicability import (
    APPLICABILITY_DISTANCE_VAR,
    applicability_block,
    box_edges,
    proposal_applicability,
)

DESIGN = DesignSpace((
    Dimension("a", "continuous", lower=0.0, upper=10.0),
    Dimension("c", "categorical", levels=("x", "y")),
))


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(APPLICABILITY_DISTANCE_VAR, raising=False)


def test_default_threshold_rule():
    assert applicability_block(0.05)["threshold"] == 0.25  # floor
    assert applicability_block(0.2)["threshold"] == 0.6  # 3 x median
    assert applicability_block(None)["threshold"] == 0.25
    block = applicability_block(0.2)
    assert block["source"] == "default" and block["set_by"] == "default"
    assert "product default" in block["rule"] and block["edge_fraction"] == 0.01


def test_lab_threshold_from_env_and_roles(monkeypatch):
    monkeypatch.setenv(APPLICABILITY_DISTANCE_VAR, "0.4")
    env = applicability_block(0.01)
    assert (env["threshold"], env["source"], env["set_by"]) == (0.4, "lab", "env")
    roles = applicability_block(0.01, roles_distance=0.7)  # roles wins over env
    assert (roles["threshold"], roles["source"], roles["set_by"]) == (0.7, "lab", "roles")
    monkeypatch.setenv(APPLICABILITY_DISTANCE_VAR, "-1")
    with pytest.raises(ConfigError):
        applicability_block(0.01)


def test_box_edge_flag_is_distinct_from_distance():
    assert box_edges(np.array([0.05, 1.0]), DESIGN) == [{"feature": "a", "side": "lower"}]
    assert box_edges(np.array([9.95, 0.0]), DESIGN) == [{"feature": "a", "side": "upper"}]
    assert box_edges(np.array([5.0, 0.0]), DESIGN) == []
    assert box_edges(np.array([0.2, 0.0]), DESIGN) == []  # 2% from the bound
    batch = np.array([[0.0, 0.0], [5.0, 1.0]])
    rows = proposal_applicability(batch, np.array([0.1, 0.5]), DESIGN, applicability_block(0.01))
    # Row 0: near the data but on the edge. Row 1: far from the data, mid-box.
    assert rows[0]["far_from_data"] is False and rows[0]["at_box_edge"]
    assert rows[1]["far_from_data"] is True and rows[1]["at_box_edge"] == []
    assert rows[1]["distance"] == 0.5 and rows[1]["threshold"] == 0.25
    assert rows[1]["threshold_source"] == "default"


def test_rounded_proposals_are_clipped_into_the_box():
    # Replay F9: V = 0.996 vs box max 0.995553; EAA = 0.0 vs box min 0.000008.
    design = DesignSpace((
        Dimension("V", "continuous", lower=0.1, upper=0.9955534),
        Dimension("EAA", "continuous", lower=0.000008, upper=1.0),
    ))
    v, eaa = design.decode_row(np.array([0.9955, 0.00001]))
    assert v <= 0.9955534 and v <= round(0.9955534, 6)
    assert eaa >= 0.000008
    assert design.decode_row(np.array([0.5, 0.5])) == [0.5, 0.5]


def test_roles_applicability_distance_is_parsed_and_validated():
    roles = ColumnRoles.from_dict({"target": "y", "applicability_distance": 0.3})
    assert roles.applicability_distance == 0.3
    for bad in (0, -1, "0.3", True, float("nan")):
        with pytest.raises(InvalidApplicabilityDistance):
            ColumnRoles.from_dict({"target": "y", "applicability_distance": bad})


def _sheet() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 14
    a, b = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
    return pd.DataFrame({"a": a.round(3), "b": b.round(3),
                         "y": (2 * a + b + rng.normal(0, 0.3, n)).round(3)})


def test_analysis_carries_the_flags():
    from kalos.portal.analysis import _analyze

    r = _analyze(_sheet(), roles=ColumnRoles.from_dict(
        {"target": "y", "features": ["a", "b"], "applicability_distance": 1e-6}))
    assert r["applicability"]["set_by"] == "roles"
    assert r["applicability"]["threshold"] == 1e-6  # a lab value is used as given
    for p in r["proposals"]:
        app = p["applicability"]
        assert app["distance"] == p["nearest_training_distance"]
        assert app["threshold_source"] == "lab"
        assert app["far_from_data"] is (app["distance"] > app["threshold"])
        assert app["far_from_data"] is (p["nearest_training_distance"] > 0)
        assert isinstance(app["at_box_edge"], list)
        for name, (lo, hi) in r["feature_ranges"].items():
            assert lo <= p["recipe"][name] <= hi
    caveat = r["conformal_caveat"]
    assert caveat["code"] == "coverage_not_guaranteed"
    assert "far_from_data" in caveat["message"] and "at_box_edge" in caveat["message"]
    json.dumps(r, allow_nan=False)


# --- HTTP ------------------------------------------------------------------- #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402


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


def test_http_invalid_roles_threshold_is_a_400_with_a_code(client):
    c, _store = client
    r = c.post("/api/run", files={"file": ("s.csv", io.BytesIO(_sheet().to_csv(index=False).encode()),
                                           "text/csv")},
               data={"roles": json.dumps({"target": "y", "applicability_distance": -2})})
    assert r.status_code == 400, r.text
    assert r.json()["code"] == "invalid_applicability_distance"


def test_reanalyze_keeps_a_roles_threshold(client, monkeypatch):
    c, store = client
    store.seed(_sheet(), "y", ["a", "b"], tenant="default", analysis={
        "target": "y", "applicability": {"threshold": 0.33, "set_by": "roles"}})
    started = c.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"a": 1.0, "b": 1.0}}]}).json()["started"]
    c.post("/api/campaign/result", json={"id": started[0]["id"], "value": 3.0})
    seen = {}

    def spy(*_a, **k):
        seen.update(k)
        raise RuntimeError("stop here")

    monkeypatch.setattr(portal_module, "_analyze", spy)
    assert c.post("/api/campaign/reanalyze").status_code == 400
    assert seen["applicability_distance"] == 0.33


def test_vals_and_recipe_are_the_same_clipped_numbers():
    """`vals` (the driver columns) and `recipe` come from one decode: rounded
    to 3 places and clipped into the box."""
    from kalos.domains import DesignSpace, Dimension
    from kalos.portal.analysis_parts.proposals import _annotate

    design = DesignSpace((
        Dimension("pH", "continuous", lower=6.0004, upper=7.9996),
        Dimension("temp", "continuous", lower=30.0, upper=37.0),
        Dimension("medium", "categorical", levels=("A", "B")),
    ))
    batch = np.array([[6.00041, 36.99999, 1.0], [7.99962, 30.0004, 0.0]])
    rows = _annotate(batch, np.array([1.0, 2.0]), np.array([0.1, 0.2]), 1.5,
                     cols=[0, 1], design=design)
    for row in rows:
        assert row["vals"] == [row["recipe"]["pH"], row["recipe"]["temp"]]
        assert 6.0004 <= row["vals"][0] <= 7.9996
    full = _annotate(batch, np.array([1.0, 2.0]), np.array([0.1, 0.2]), 1.5, design=design)
    assert full[0]["vals"][:2] == [full[0]["recipe"]["pH"], full[0]["recipe"]["temp"]]
    assert full[0]["vals"][2] == 1.0  # categorical code, not the label
