"""Per-target direction (`direction`: "maximize" | "minimize").

Minimize fits the proposal surrogate on -y, so the batch heads for LOW values,
while every reported number stays in the original units and sign.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.domains import ColumnRoles, InvalidDirection
from kalos.portal.analysis import _analyze
from kalos.portal.uploads import UploadRejected


def _sheet(n: int = 14) -> pd.DataFrame:
    """y rises with both inputs: the low region is near (0, 0)."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 10, n)
    b = rng.uniform(0, 10, n)
    y = 5.0 + 2.0 * a + 1.5 * b + rng.normal(0, 0.3, n)
    return pd.DataFrame({"a": a.round(3), "b": b.round(3), "y": y.round(3)})


def _roles(**extra) -> ColumnRoles:
    return ColumnRoles.from_dict({"target": "y", "features": ["a", "b"], **extra})


@pytest.fixture(scope="module")
def results():
    df = _sheet()
    return {
        "default": _analyze(df, roles=_roles()),
        "maximize": _analyze(df, roles=_roles(direction="maximize")),
        "minimize": _analyze(df, roles=_roles(direction="minimize")),
    }


def _mean_recipe(result) -> float:
    return float(np.mean([p["recipe"]["a"] + p["recipe"]["b"] for p in result["proposals"]]))


def test_default_is_maximize_and_unchanged(results):
    assert results["default"]["direction"] == "maximize"
    assert results["maximize"]["direction"] == "maximize"
    # An explicit "maximize" gives exactly the default batch.
    assert results["default"]["proposals"] == results["maximize"]["proposals"]
    assert results["default"]["best"] == round(float(_sheet()["y"].max()), 4)


def test_minimize_proposes_toward_the_low_region(results):
    mn, mx = results["minimize"], results["default"]
    assert mn["direction"] == "minimize"
    assert _mean_recipe(mn) < 10.0 < _mean_recipe(mx)
    assert mn["best"] == round(float(_sheet()["y"].min()), 4)
    # Predictions are in original units and sign (y is >= ~5 everywhere).
    assert all(p["pred"] > 0 for p in mn["proposals"])
    # Drivers keep the original sign: y rises with a.
    rho = {d["name"]: d["rho"] for d in mn["drivers"]}
    assert rho["a"] > 0
    assert mn["drivers"] == mx["drivers"]
    assert mn["proposal_gating"]["gated"] is False
    assert mn["proposal_gating"]["off_for_minimize"] is True
    assert "minimize" in mn["proposal_gating"]["reason"]
    assert mx["proposal_gating"]["off_for_minimize"] is False


def test_invalid_direction_is_refused():
    with pytest.raises(InvalidDirection):
        ColumnRoles.from_dict({"target": "y", "direction": "down"})
    with pytest.raises(UploadRejected) as exc:
        _analyze(_sheet(), roles=_roles(), direction="sideways", preflight_only=True)
    assert exc.value.code == "invalid_direction"
    with pytest.raises(UploadRejected) as exc:
        _analyze(_sheet(), roles=_roles(direction="maximize"), direction="minimize",
                 preflight_only=True)
    assert exc.value.code == "direction_conflict"


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


def _csv() -> bytes:
    return _sheet().to_csv(index=False).encode()


@pytest.mark.parametrize("data", [
    {"roles": json.dumps({"target": "y", "direction": "lowest"})},
    {"target": "y", "direction": "lowest"},
])
def test_http_invalid_direction_is_a_400_with_a_code(client, data):
    c, _store = client
    r = c.post("/api/run", files={"file": ("s.csv", io.BytesIO(_csv()), "text/csv")}, data=data)
    assert r.status_code == 400, r.text
    assert r.json()["code"] == "invalid_direction"


def test_http_form_direction_is_stored_and_reused_by_reanalyze(client, monkeypatch):
    c, _store = client
    r = c.post("/api/run", files={"file": ("s.csv", io.BytesIO(_csv()), "text/csv")},
               data={"target": "y", "direction": "minimize"})
    assert r.status_code == 200, r.text
    assert r.json()["direction"] == "minimize"
    summary = c.get("/api/campaign").json()
    assert summary["direction"] == "minimize"
    assert summary["best"] == pytest.approx(float(_sheet()["y"].min()))

    seen = {}
    real = portal_module._analyze

    def spy(*a, **k):
        seen["direction"] = k.get("direction")
        return real(*a, **k)

    monkeypatch.setattr(portal_module, "_analyze", spy)
    started = c.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"a": 1.0, "b": 1.0}}]}).json()["started"]
    c.post("/api/campaign/result", json={"id": started[0]["id"], "value": 7.0})
    r = c.post("/api/campaign/reanalyze")
    assert r.status_code == 200, r.text
    assert seen["direction"] == "minimize"
    assert r.json()["analysis"]["direction"] == "minimize"
    # The run record of the reanalyze carries the stored direction.
    assert r.json()["analysis"]["run"]["direction"] == "minimize"
    assert r.json()["analysis"]["proposal_gating"]["off_for_minimize"] is True


def _stub(df, target, *, direction=None, roles=None, **_):
    eff = (roles.direction if roles is not None and roles.direction else None) or direction
    return {"target": target or "y", "proposal_features": ["a", "b"], "d": 2, "n": len(df),
            "proposals": [], "seed": 1234, "timestamp": 1, "direction": eff or "maximize"}


@pytest.mark.parametrize("how", ["form", "roles"])
def test_direction_changes_the_config_hash_on_identical_bytes(how):
    from kalos.portal.pipeline import compute_upload_analysis

    def run(direction):
        if how == "form":
            return compute_upload_analysis(_csv(), "y", False, "s.csv", analyze=_stub,
                                           direction=direction)[1]
        roles = json.dumps({"target": "y", "direction": direction})
        return compute_upload_analysis(_csv(), "y", False, "s.csv", roles, analyze=_stub)[1]

    mx, mn = run("maximize"), run("minimize")
    assert (mx["run"]["direction"], mn["run"]["direction"]) == ("maximize", "minimize")
    assert mx["run"]["input_sha256"] == mn["run"]["input_sha256"]
    assert mx["proposal_provenance"]["config_sha256"] != mn["proposal_provenance"]["config_sha256"]
    assert "direction" in mx["proposal_provenance"]["config_fields"]


def test_reanalyze_record_carries_the_direction():
    from kalos.portal.pipeline import attach_reanalyze_run

    df = _sheet()
    out = attach_reanalyze_run(df, _stub(df, "y", direction="minimize"), anonymize=False,
                               pending_runs=0)
    assert out["run"]["direction"] == "minimize"


def test_cv_topk_ranks_the_lowest_rows_when_minimizing(results):
    """The decision under minimize is which LOW-valued recipes to advance, so
    cv_topk compares the true and predicted bottom k (sign * oof), not the top k."""
    from kalos.core.evaluation import top_k_overlap

    for name, sign in (("default", 1.0), ("minimize", -1.0)):
        res = results[name]
        oof = np.asarray(res["oof"], float)
        k = res["cv_topk"]["k"]
        want = top_k_overlap(sign * oof[:, 0], sign * oof[:, 1], k)
        assert res["cv_topk"]["evaluable"] is True
        assert res["cv_topk"]["overlap"] == round(float(want["overlap"]), 3), name



def test_promotion_does_not_judge_the_feasibility_gates_when_minimizing(results):
    """P(feasible) scores y > 0: under minimize its AUC/ECE/Brier gates are
    neither passed nor failed, and never block, they are not judged with a reason."""
    mn, mx = results["minimize"]["promotion"], results["default"]["promotion"]
    assert {"feasibility_auc: not applicable when minimizing", "ece: not applicable when minimizing",
            "brier: not applicable when minimizing"} <= set(mn["not_judged"])
    for key in ("feasibility_auc", "ece", "brier"):
        assert not any(f.startswith(key) for f in mn["failures"])
        assert key not in mn["unmeasured"] and key not in mn["metrics"]
    assert not any("when minimizing" in n for n in mx["not_judged"])


def test_check_gates_moves_feasibility_gates_to_not_judged():
    from kalos.core.gates import check_gates

    bad = {"surrogate_spearman": 0.9, "feasibility_auc": 0.1, "ece": 0.9, "brier": 0.9}
    assert not check_gates(bad).passed
    res = check_gates({**bad, "feasibility_not_applicable": "not applicable when minimizing"})
    assert res.passed and res.status == "passed" and res.failures == []
    assert len(res.not_judged) == 3
    # The surrogate floor still applies.
    low = check_gates({"surrogate_spearman": 0.0,
                       "feasibility_not_applicable": "not applicable when minimizing"})
    assert not low.passed and res.metrics == {"surrogate_spearman": 0.9}
