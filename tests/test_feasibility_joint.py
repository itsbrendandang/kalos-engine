"""P(all declared specs met): kalos/core/feasibility_joint.py and roles.specs
through _analyze (kalos/portal/analysis_parts/specs.py)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.core.feasibility_joint import (
    N_SAMPLES,
    gaussian_spec_probability,
    joint_feasibility,
    joint_spec_probability,
)
from kalos.domains import ColumnRoles, InvalidSpecs

# ---- pure function --------------------------------------------------------------------


def test_two_independent_gaussian_specs_give_the_product():
    m1, s1 = np.array([96.0, 94.0, 100.0]), np.array([2.0, 1.5, 3.0])
    m2, s2 = np.array([4.0, 6.0, 5.0]), np.array([1.0, 2.0, 0.5])
    limits = [(95.0, None), (None, 5.5)]
    p_specs, p_all = joint_feasibility([m1, m2], [s1, s2], limits, seed=7, n_samples=200_000)
    e1 = gaussian_spec_probability(m1, s1, 95.0, None)
    e2 = gaussian_spec_probability(m2, s2, None, 5.5)
    assert np.allclose(p_specs[0], e1, atol=0.005) and np.allclose(p_specs[1], e2, atol=0.005)
    assert np.allclose(p_all, e1 * e2, atol=0.005)
    assert ((p_all >= 0) & (p_all <= 1)).all()


def test_seeded_and_bounded():
    a = joint_feasibility([np.zeros(4)], [np.ones(4)], [(-1.0, 1.0)], seed=1)
    b = joint_feasibility([np.zeros(4)], [np.ones(4)], [(-1.0, 1.0)], seed=1)
    assert np.array_equal(a[1], b[1])
    assert np.allclose(a[1], 0.6827, atol=0.03)
    assert N_SAMPLES >= 1000


def test_draws_are_counted_jointly_per_draw_index():
    # Perfectly opposed draws: each spec is met half the time, never both together.
    d = np.array([[1.0], [-1.0]])
    p_specs, p_all = joint_spec_probability([d, -d], [(0.0, None), (0.0, None)])
    assert p_specs.ravel().tolist() == [0.5, 0.5] and p_all.tolist() == [0.0]
    with pytest.raises(ValueError):
        joint_spec_probability([], [])


# ---- parsing --------------------------------------------------------------------------

def test_parse_specs():
    r = ColumnRoles.from_dict({"target": "Titer", "specs": [{"column": "Purity", "min": 95}]})
    assert r.specs[0].column == "Purity" and r.specs[0].min == 95 and r.specs[0].max is None
    for bad in ([{"column": "P"}], [{"column": "P", "min": 3, "max": 1}], [{"min": 1}],
                [{"column": "P", "min": "95"}], [{"column": "P", "max": float("nan")}],
                [{"column": "P", "min": 1}, {"column": "P", "max": 2}], {"column": "P"}):
        with pytest.raises(InvalidSpecs):
            ColumnRoles.from_dict({"target": "Titer", "specs": bad})


# ---- through _analyze -----------------------------------------------------------------

def _sheet(n: int = 16) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    temp = rng.uniform(28, 34, n)
    feed = rng.uniform(1, 5, n)
    titer = 10 + 2.0 * feed - 0.5 * (temp - 31) ** 2 + rng.normal(0, 0.3, n)
    purity = 99 - 0.8 * feed + 0.2 * (temp - 28) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"Temp": temp.round(2), "Feed": feed.round(3),
                         "Titer": titer.round(3), "Purity": purity.round(3)})


def _roles(specs):
    return ColumnRoles.from_dict({"target": "Titer", "features": ["Temp", "Feed"], "specs": specs})


def test_specs_through_analyze():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    base = _analyze(_sheet(), "Titer", roles=_roles(None))
    out = _analyze(_sheet(), "Titer", roles=_roles(
        [{"column": "Purity", "min": 95}, {"column": "Titer", "min": 12}]))
    json.dumps(out, allow_nan=False)
    jf = out["joint_feasibility"]
    assert jf["available"] is True and jf["reason"] is None and jf["n_samples"] == N_SAMPLES
    assert jf["method"].startswith("independent_mc")
    assert [s["column"] for s in jf["specs"]] == ["Purity", "Titer"]
    assert out["proposals"]
    for p in [*out["proposals"], out["model_optimum"]]:
        assert 0.0 <= p["p_all_specs"] <= 1.0
        assert set(p["p_specs"]) == {"Purity", "Titer"}
        assert p["p_all_specs"] <= min(p["p_specs"].values()) + 0.03
    # Advisory: the batch and the best bet are the same as without specs.
    assert [p["recipe"] for p in out["proposals"]] == [p["recipe"] for p in base["proposals"]]
    assert out["model_optimum"]["recipe"] == base["model_optimum"]["recipe"]
    assert base["joint_feasibility"]["available"] is False
    assert all(p["p_all_specs"] is None for p in base["proposals"])


def test_missing_spec_column_is_unavailable_with_a_reason():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_sheet(), "Titer", roles=_roles([{"column": "Purity", "min": 95},
                                                     {"column": "HCP", "max": 100}]))
    jf = out["joint_feasibility"]
    assert jf["available"] is False and "HCP" in jf["reason"]
    assert all(p["p_all_specs"] is None for p in out["proposals"])
    json.dumps(out, allow_nan=False)


def test_spec_on_a_declared_input_is_refused():
    from kalos.portal.analysis import _analyze
    from kalos.portal.uploads import UploadRejected

    with pytest.raises(UploadRejected) as exc:
        _analyze(_sheet(), "Titer", roles=_roles([{"column": "Feed", "max": 3}]), preflight_only=True)
    assert exc.value.code == "invalid_specs"


def test_http_invalid_specs_is_a_400(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import io

    from fastapi.testclient import TestClient

    from kalos.portal.app import app, get_campaign_store
    from kalos.portal.campaign import CampaignStore

    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        c = TestClient(app)
        body = _sheet().to_csv(index=False).encode()
        for specs in ([{"column": "Purity"}], [{"column": "Feed", "max": 3}]):
            r = c.post("/api/preflight", files={"file": ("s.csv", io.BytesIO(body), "text/csv")},
                       data={"target": "Titer", "roles": json.dumps(
                           {"target": "Titer", "features": ["Temp", "Feed"], "specs": specs})})
            assert r.status_code == 400 and r.json()["code"] == "invalid_specs", r.text
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
