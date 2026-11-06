"""roles.linear_constraints over HTTP: mixtures (==) and budgets (<=) on
continuous inputs (kalos/portal/analysis_parts/linear.py, docs/CAMPAIGNS_API.md)."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.domains import ColumnRoles, InvalidLinearConstraint

MIX = [{"terms": {"A": 1, "B": 1, "C": 1}, "op": "==", "rhs": 100, "label": "sums to 100"}]


def _mixture_sheet(n: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    a = rng.uniform(20, 60, n)
    b = rng.uniform(10, 40, n)
    c = 100.0 - a - b
    ph = rng.uniform(6.0, 7.5, n)
    y = 0.6 * a + 0.3 * b - 0.4 * (ph - 6.8) ** 2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"A": a.round(3), "B": b.round(3), "C": c.round(3),
                         "pH": ph.round(2), "Titer": y.round(3)})


def _sugar_sheet(n: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    glc = rng.uniform(5, 25, n)
    gly = rng.uniform(5, 25, n)
    y = 1.0 * glc + 0.8 * gly + rng.normal(0, 0.3, n)  # rises with both: the cap binds
    return pd.DataFrame({"Glucose": glc.round(3), "Glycerol": gly.round(3), "Titer": y.round(3)})


# ---- parsing ------------------------------------------------------------------------

def test_parse_linear_constraints():
    r = ColumnRoles.from_dict({"target": "y", "linear_constraints": MIX})
    (con,) = r.linear_constraints
    assert con.terms == (("A", 1.0), ("B", 1.0), ("C", 1.0)) and con.op == "==" and con.rhs == 100
    assert con.text() == "A + B + C == 100"
    assert ColumnRoles.from_dict({"target": "y"}).linear_constraints == ()


@pytest.mark.parametrize("bad", [
    {"terms": {}, "op": "<=", "rhs": 1},
    {"terms": {"A": float("nan")}, "op": "<=", "rhs": 1},
    {"terms": {"A": 1}, "op": "<=", "rhs": float("inf")},
    {"terms": {"A": 1}, "op": "<", "rhs": 1},
    {"terms": {"A": "1"}, "op": "<=", "rhs": 1},
    {"terms": {"A": 0}, "op": "<=", "rhs": 1},
    {"terms": {"A": 1}, "op": "<=", "rhs": True},
    "A + B <= 3",
])
def test_bad_constraints_are_refused(bad):
    with pytest.raises(InvalidLinearConstraint):
        ColumnRoles.from_dict({"target": "y", "linear_constraints": [bad]})


def test_too_many_constraints_are_refused():
    many = [{"terms": {"A": 1}, "op": "<=", "rhs": i} for i in range(21)]
    with pytest.raises(InvalidLinearConstraint, match="most allowed"):
        ColumnRoles.from_dict({"target": "y", "linear_constraints": many})


# ---- through _analyze ---------------------------------------------------------------

def _roles(cons, feats, target="Titer"):
    return ColumnRoles.from_dict({"target": target, "features": feats, "linear_constraints": cons})


def test_mixture_equality_holds_for_every_proposal_and_the_best_bet():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_mixture_sheet(), "Titer", roles=_roles(MIX, ["A", "B", "C", "pH"]))
    json.dumps(out, allow_nan=False)
    block = out["linear_constraints"]
    tol = block["constraints"][0]["tolerance"]
    assert tol == pytest.approx(0.003)
    assert block["all_satisfied"] is True and block["constraints"][0]["text"] == "A + B + C == 100"
    assert out["proposals"]
    for p in out["proposals"]:
        r = p["recipe"]
        assert abs(r["A"] + r["B"] + r["C"] - 100) <= tol, r
        assert p["constraints_ok"] is True
    opt = out["model_optimum"]["recipe"]
    assert abs(opt["A"] + opt["B"] + opt["C"] - 100) <= tol
    assert out["model_optimum"]["constraints_ok"] is True


def test_inequality_budget_holds():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    cons = [{"terms": {"Glucose": 1, "Glycerol": 1}, "op": "<=", "rhs": 30}]
    out = _analyze(_sugar_sheet(), "Titer", roles=_roles(cons, ["Glucose", "Glycerol"]))
    for p in [*out["proposals"], out["model_optimum"]]:
        r = p["recipe"]
        assert r["Glucose"] + r["Glycerol"] <= 30 + 0.002, r
        assert p["constraints_ok"] is True
    assert out["linear_constraints"]["all_satisfied"] is True


def test_no_constraints_reports_an_empty_block():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_sugar_sheet(), "Titer", roles=_roles(None, ["Glucose", "Glycerol"]))
    assert out["linear_constraints"]["constraints"] == []
    assert out["linear_constraints"]["all_satisfied"] is True
    assert all(p["constraints_ok"] is None for p in out["proposals"])


def test_refusals_before_any_fit(monkeypatch):
    pytest.importorskip("botorch")
    from kalos.core import surrogate
    from kalos.portal.analysis import _analyze
    from kalos.portal.uploads import UploadRejected

    def no_fit(*a, **k):
        raise AssertionError("a model was fit before the refusal")

    monkeypatch.setattr(surrogate.Surrogate, "fit", no_fit)
    df = _sugar_sheet()
    df["Strain"] = ["WT", "dPEP4"] * 7
    cases = [
        ([{"terms": {"Glucose": 1, "Glycerol": 1}, "op": ">=", "rhs": 80}], ["Glucose", "Glycerol"],
         "infeasible_linear_constraints"),
        ([{"terms": {"Nope": 1}, "op": "<=", "rhs": 3}], ["Glucose", "Glycerol"],
         "invalid_linear_constraint"),
        ([{"terms": {"Strain": 1}, "op": "<=", "rhs": 3}], ["Glucose", "Glycerol", "Strain"],
         "invalid_linear_constraint"),
        ([{"terms": {"Glucose": 1}, "op": "<=", "rhs": 20}], ["Glucose", "Glycerol", "Strain"],
         "linear_constraints_with_categorical"),
    ]
    for cons, feats, code in cases:
        roles = ColumnRoles.from_dict({"target": "Titer", "features": feats,
                                       "categoricals": ["Strain"] if "Strain" in feats else [],
                                       "linear_constraints": cons})
        with pytest.raises(UploadRejected) as exc:
            _analyze(df, "Titer", roles=roles)
        assert exc.value.code == code, (cons, exc.value.code)


# ---- HTTP ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal.app import app, get_campaign_store
    from kalos.portal.campaign import CampaignStore

    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _post(c, roles, path="/api/preflight"):
    body = _sugar_sheet().to_csv(index=False).encode()
    return c.post(path, files={"file": ("s.csv", io.BytesIO(body), "text/csv")},
                  data={"target": "Titer", "roles": json.dumps(roles)})


def test_http_400_codes(client):
    base = {"target": "Titer", "features": ["Glucose", "Glycerol"]}
    r = _post(client, {**base, "linear_constraints": [{"terms": {}, "op": "<=", "rhs": 1}]})
    assert r.status_code == 400 and r.json()["code"] == "invalid_linear_constraint"
    r = _post(client, {**base, "linear_constraints": [{"terms": {"pH": 1}, "op": "<=", "rhs": 1}]})
    assert r.status_code == 400 and r.json()["code"] == "invalid_linear_constraint"
    infeasible = {**base, "linear_constraints": [
        {"terms": {"Glucose": 1, "Glycerol": 1}, "op": "==", "rhs": 500}]}
    r = _post(client, infeasible, path="/api/run")
    assert r.status_code == 400 and r.json()["code"] == "infeasible_linear_constraints"
    # The same refusal before any fit, on the preflight route (owner decision).
    pre = _post(client, infeasible, path="/api/preflight")
    assert pre.status_code == 400 and pre.json() == r.json()


def test_constraints_persist_with_the_campaign_roles_across_reanalyze(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    seen: list = []

    def stub(df, target, *, roles=None, direction=None, **_):
        seen.append(roles.linear_constraints if roles is not None else None)
        feats = ["Glucose", "Glycerol"]
        return {
            "target": "Titer", "features": feats, "proposal_features": feats, "d": 2,
            "n": len(df), "best": float(df["Titer"].max()), "seed": 1, "timestamp": 1,
            "direction": "maximize", "categorical_features": [],
            "reliability": {"clears_floor": True},
            "gp_shapes": {"available": True, "features": []},
            "proposals": [{"recipe": {"Glucose": 10.0, "Glycerol": 10.0}, "pred": 20.0, "std": 1.0}],
            "conformal_q": 1.0,
        }

    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    store = CampaignStore(tmp_path)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal_module, "_analyze", stub)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        c = TestClient(app)
        cons = [{"terms": {"Glucose": 1, "Glycerol": 1}, "op": "<=", "rhs": 30}]
        roles = {"target": "Titer", "features": ["Glucose", "Glycerol"], "linear_constraints": cons}
        assert _post(c, roles, path="/api/run").status_code == 200
        started = c.post("/api/campaign/start", json={"recipes": [
            {"recipe": {"Glucose": 10.0, "Glycerol": 10.0}, "pred": 20.0, "std": 1.0}]}).json()["started"]
        c.post("/api/campaign/result", json={"id": started[0]["id"], "value": 19.0})
        assert c.post("/api/campaign/reanalyze").status_code == 200
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
    assert len(seen) == 2 and seen[0] == seen[1] and seen[1][0].rhs == 30
    assert store.get(tenant="default")["settings"]["roles"]["linear_constraints"] == cons
