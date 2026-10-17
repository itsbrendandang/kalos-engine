"""A target that does not vary cannot support a model: refuse, never propose.

Regression for a QA finding: a run sheet whose target was 5.0 on every row was
accepted (validation `pass_with_warnings`) and came back with five proposed
recipes ranked by rounding error. These tests pin the refusal - for exact,
float-noise, integer and blank-padded constant targets - on every path that
reaches `_analyze` (upload, re-upload with another target, campaign
re-analyze), and that `propose` is never reached. Synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.validation.target import TARGET_CONSTANT_RTOL, assess_target_variation

# --- the torch-free verdict ---------------------------------------------------- #


def test_exact_constant_is_constant():
    v = assess_target_variation([5.0] * 12)
    assert v.status == "constant" and not v.varies
    assert v.n_distinct == 1 and v.value_min == v.value_max == 5.0


def test_float_noise_constant_is_constant():
    rng = np.random.default_rng(0)
    vals = 5.0 + rng.uniform(-1e-12, 1e-12, 30)
    v = assess_target_variation(vals)
    assert v.n_distinct > 1  # genuinely different floats ...
    assert v.status == "constant"  # ... but only by rounding noise


def test_near_constant_but_genuinely_varying_still_varies():
    # 100.000, 100.001, ... : a relative spread of ~1e-5, the resolution of a
    # 6-significant-figure assay. Must be fit, not refused.
    vals = 100.0 + 0.001 * np.arange(12)
    assert assess_target_variation(vals).status == "ok"


def test_tolerance_is_scale_free():
    # The same relative pattern at any magnitude gets the same verdict, so a
    # target that legitimately lives at 1e-12 (pM in molar units) is not noise.
    base = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    for scale in (1e-12, 1.0, 1e9):
        assert assess_target_variation(base * scale).varies
    just_inside = np.array([1.0, 1.0 + 0.5 * TARGET_CONSTANT_RTOL] * 4)
    just_outside = np.array([1.0, 1.0 + 2 * TARGET_CONSTANT_RTOL] * 4)
    assert assess_target_variation(just_inside).status == "constant"
    assert assess_target_variation(just_outside).status == "ok"


def test_integer_constant_is_constant():
    assert assess_target_variation(pd.Series([7] * 10, dtype="int64")).status == "constant"


def test_nan_padded_constant_is_constant():
    vals = [5.0, np.nan, 5.0, None, "", "  ", 5.0, 5.0, np.nan]
    v = assess_target_variation(vals)
    assert v.status == "constant" and v.n_numeric == 4


def test_all_missing_and_non_numeric():
    assert assess_target_variation([np.nan, None, ""]).status == "empty"
    assert assess_target_variation(["high", "low", "n/a"]).status == "not_numeric"
    assert assess_target_variation([np.inf, -np.inf]).status == "not_numeric"


def test_zero_centred_target_varies():
    # mean ~0 must not collapse the scale: max|y| is the reference, not |mean|.
    assert assess_target_variation([-1.0, 1.0, -0.5, 0.5, 0.0, 0.25]).varies


# --- the portal refuses before any fit ------------------------------------------- #

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.analysis import _analyze  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.portal.uploads import UploadRejected  # noqa: E402


def _sheet(target: np.ndarray | list, n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Methanol": rng.uniform(0, 4, n).round(3),
        "pH": rng.uniform(5, 7, n).round(2),
        "biomass_od600": rng.uniform(1, 3, n).round(3),  # a VARYING other outcome
        "lipase_titer": target,
    })


@pytest.fixture
def no_fit(monkeypatch):
    """Fail the test if anything tries to fit a surrogate or propose a batch."""
    import kalos.core.optimize as optimize
    import kalos.core.surrogate as surrogate

    def _boom(*_a, **_k):
        raise AssertionError("a model was fit / a batch proposed for a constant target")

    monkeypatch.setattr(surrogate.Surrogate, "fit", _boom)
    monkeypatch.setattr(optimize, "propose", _boom)
    monkeypatch.setattr(optimize, "propose_batch", _boom, raising=False)


rng0 = np.random.default_rng(1)
CONSTANT_TARGETS = {
    "exact": [5.0] * 20,
    "float_noise": list(5.0 + rng0.uniform(-1e-12, 1e-12, 20)),
    "integer": [5] * 20,
    "nan_padded": [5.0 if i % 3 else np.nan for i in range(20)],
}


@pytest.mark.parametrize("kind", sorted(CONSTANT_TARGETS))
def test_analyze_refuses_constant_target(kind, no_fit):
    df = _sheet(CONSTANT_TARGETS[kind])
    with pytest.raises(UploadRejected) as exc:
        _analyze(df)
    body = exc.value.to_body()
    assert body["code"] == "target_constant"
    assert body["target"] == "lipase_titer"
    assert "no proposals were generated" in body["error"]
    assert "lipase_titer" in body["error"]
    # the other measured outcome that DOES vary is offered as an alternative
    assert body["available_targets"] == ["biomass_od600"]
    assert body["target_summary"]["status"] == "constant"


def test_analyze_refuses_requested_constant_target(no_fit):
    # The target-change path: the client re-submits with an explicit target.
    df = _sheet(np.linspace(1, 9, 20))
    df["biomass_od600"] = 2.5
    with pytest.raises(UploadRejected) as exc:
        _analyze(df, "biomass_od600")
    assert exc.value.code == "target_constant"
    assert exc.value.context["available_targets"] == ["lipase_titer"]


def test_analyze_refuses_all_text_target(no_fit):
    df = _sheet(["high", "low"] * 10)
    with pytest.raises(UploadRejected) as exc:
        _analyze(df, "lipase_titer")
    assert exc.value.code == "target_not_numeric"


def test_near_constant_but_varying_target_is_still_fit():
    df = _sheet(list(100.0 + 0.001 * np.arange(20)))
    out = _analyze(df)
    assert out["target"] == "lipase_titer"
    assert len(out["proposals"]) >= 1


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal_module, "_LATEST", {})
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app), store
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


def test_upload_with_constant_target_returns_structured_400(client, no_fit):
    c, store = client
    csv = _sheet([5.0] * 20).to_csv(index=False).encode()
    resp = c.post("/api/run", files={"file": ("runs.csv", csv, "text/csv")})
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "target_constant"
    assert "does not vary" in body["error"] or "same value" in body["error"]
    assert "proposals" not in body
    assert "Traceback" not in resp.text
    # nothing was published or seeded for a refused sheet
    assert c.get("/api/latest").json() == {"has_data": False}
    assert store.get() is None


def test_campaign_reanalyze_refuses_when_every_value_is_identical(client, no_fit):
    c, store = client
    df = pd.DataFrame({
        "Methanol": np.linspace(0, 4, 8).round(3),
        "pH": np.linspace(5, 7, 8).round(2),
        "lipase_titer": [5.0] * 8,
    })
    store.seed(df, "lipase_titer", ["Methanol", "pH"])
    started = c.post("/api/campaign/start", json={"recipes": [
        {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 5.0, "std": 0.1,
         "mode": "explore", "reason": "t"},
    ]}).json()["started"]
    c.post("/api/campaign/result", json={"id": started[0]["id"], "value": 5.0})
    resp = c.post("/api/campaign/reanalyze")
    assert resp.status_code == 400
    assert resp.json()["code"] == "target_constant"
    assert store.get()["round"] == 0  # nothing committed
