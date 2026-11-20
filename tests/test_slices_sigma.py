"""R2 (1D model slices) and R3 (response-surface sigma grid).

Unit tests drive `compute_slices` with a deterministic fake surrogate; the
integration tests run the real `_analyze` and check the slices sit on the same
reference point as the 2D surface, categoricals are skipped, and anonymize
aliases every name.
"""
from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis_parts.anonymize import _anonymize_result
from kalos.portal.analysis_parts.slices import (
    SLICE_COVERAGE,
    SLICE_N_POINTS,
    compute_slices,
    reference_point,
)


class _FakeGP:
    """mean = sum(x), sd = 0.1 + |x0|: deterministic, needs no torch."""

    def posterior(self, X):
        X = np.asarray(X, float)
        return X.sum(axis=1), 0.1 + np.abs(X[:, 0])


def _toy():
    X = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.5, 1.5, 1.0], [0.2, 0.3, 0.0]])
    y = np.array([1.0, 3.0, 2.0, 0.5])
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    return X, y, bounds


def _run(**kw):
    X, y, bounds = _toy()
    return compute_slices(
        _FakeGP(), X, y, ["a", "b"], ["media"], {"media": ["M1", "M2"]}, bounds,
        [{"name": "b"}, {"name": "a"}], False, **kw,
    )


def test_slice_shapes_band_order_and_contract():
    out = _run()
    assert out["reason"] is None
    sl = out["slices"]
    # Driver order: b first, then a.
    assert [s["feature"] for s in sl] == ["b", "a"]
    z = NormalDist().inv_cdf(0.95)
    for s in sl:
        n = s["n_points"]
        assert n == SLICE_N_POINTS
        for key in ("x", "mean", "sd", "lo", "hi"):
            assert len(s[key]) == n
        assert all(sd >= 0 for sd in s["sd"])
        assert all(lo <= m <= hi for lo, m, hi in zip(s["lo"], s["mean"], s["hi"]))
        assert s["coverage"] == SLICE_COVERAGE == 0.9
        assert s["band_method"] == "gp_posterior"
        assert s["note"] == "model-based band, not held-out calibrated"
        assert "conformal" not in json.dumps(s)
        assert s["swept_at"] == "incumbent"
        assert s["x"][0] == s["range"][0] and s["x"][-1] == s["range"][1]
        # band is mean +/- z*sd at the stated coverage
        assert s["hi"][3] - s["mean"][3] == pytest.approx(z * s["sd"][3], abs=2e-4)
    # Reference = incumbent row (argmax y), categorical decoded.
    assert sl[0]["reference"] == {
        "basis": "incumbent", "incumbent_basis": "single",
        "recipe": {"a": 1.0, "b": 2.0, "media": "M2"},
    }
    # Non-swept inputs fixed at the reference: slice over b has mean = b + a_ref + media_code.
    assert sl[0]["mean"] == [round(x + 1.0 + 1.0, 4) for x in sl[0]["x"]]
    assert json.loads(json.dumps(out)) == out  # JSON-safe


def test_categorical_skipped_and_feature_cap():
    out = _run(max_features=1)
    assert [s["feature"] for s in out["slices"]] == ["b"]
    assert {"feature": "media", "reason": "categorical"} in out["slices_skipped"]
    assert {"feature": "a", "reason": "feature_cap"} in out["slices_skipped"]


def test_deterministic():
    assert json.dumps(_run()) == json.dumps(_run())


def test_unavailable_gp_and_no_continuous_give_null_with_reason():
    X, y, bounds = _toy()
    out = compute_slices(None, X, y, ["a", "b"], [], {}, bounds, [], False)
    assert out["slices"] is None and out["reason"]
    out = compute_slices(_FakeGP(), X[:, 2:], y, [], ["media"], {"media": ["M1", "M2"]},
                         bounds[:, 2:], [], False)
    assert out["slices"] is None and out["reason"]
    assert out["slices_skipped"] == [{"feature": "media", "reason": "categorical"}]


def test_reference_point_basis():
    X, y, _ = _toy()
    _vec, _rec, basis = reference_point(X, y, ["a", "b"], [], {}, True)
    assert basis == "recipe_mean"


def test_anonymize_aliases_slice_names_unit():
    result = {
        "slices": [{"feature": "Operator", "x": [0], "reference": {
            "basis": "incumbent", "recipe": {"Operator": 1.0, "Donor_Name_1": "Jane", "pH": 7.0}}}],
        "slices_skipped": [{"feature": "Donor_Name_1", "reason": "categorical"}],
    }
    out = _anonymize_result(result, categorical_levels={"Donor_Name_1": ["Jane"]})
    blob = json.dumps(out)
    assert "Operator" not in blob and "Donor_Name_1" not in blob and "Jane" not in blob
    assert out["slices"][0]["feature"].startswith("col_")
    assert "pH" in out["slices"][0]["reference"]["recipe"]


# ---- integration: real GP via _analyze -------------------------------------

_NAMES = ["Jane Smith 0", "alice_secret_op", "Bob Jones 1", "carol_secret_op"]


def _sheet() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 24
    ph = rng.uniform(6.0, 7.5, n).round(3)
    methanol = rng.uniform(0.5, 3.5, n).round(3)
    temp = rng.uniform(28, 34, n).round(2)
    people = [_NAMES[i % len(_NAMES)] for i in range(n)]
    titer = (3 + 1.2 * methanol - 1.5 * (ph - 6.8) ** 2 + 0.05 * temp
             + rng.normal(0, 0.1, n)).round(4)
    return pd.DataFrame({"Donor_Name_1": people, "Methanol": methanol, "pH": ph,
                         "Temp": temp, "lipase_titer": titer})


def _analyze_once(anonymize: bool) -> dict:
    pytest.importorskip("torch")
    from kalos.domains import ColumnRoles
    from kalos.portal.analysis import _analyze

    roles = ColumnRoles.from_dict({
        "target": "lipase_titer", "features": ["Methanol", "pH", "Temp"],
        "categoricals": ["Donor_Name_1"],
    })
    return _analyze(_sheet(), target="lipase_titer", roles=roles, anonymize=anonymize)


@pytest.fixture(scope="module")
def analysis() -> dict:
    return _analyze_once(False)


def test_surface_carries_sigma_grid(analysis):
    surf = analysis["response_surface"]
    assert surf is not None, analysis["visualization_reasons"]
    std = np.asarray(surf["std"])
    assert std.shape == np.asarray(surf["z"]).shape == (21, 21)
    assert (std >= 0).all()
    assert surf["std_kind"] == "gp_posterior_sd"
    assert surf["std_basis"] == "latent_mean"
    assert surf["z_basis"] == "predicted"


def test_real_slices_match_surface_reference(analysis):
    slices = analysis["slices"]
    assert analysis["visualization_reasons"]["slices"] is None
    assert {s["feature"] for s in slices} == {"Methanol", "pH", "Temp"}
    surf = analysis["response_surface"]
    for s in slices:
        assert s["reference"]["recipe"] == surf["fixed_recipe"]
        assert s["reference"]["incumbent_basis"] == surf["incumbent_basis"]
        assert s["swept_at"] == surf["swept_at"]
        assert s["range"] == analysis["feature_ranges"][s["feature"]]
        assert all(lo <= m <= hi for lo, m, hi in zip(s["lo"], s["mean"], s["hi"]))
        assert all(v >= 0 for v in s["sd"])
    # Driver order respected.
    drv = [d["name"] for d in analysis["drivers"]]
    assert [s["feature"] for s in slices] == [n for n in drv if n in {"Methanol", "pH", "Temp"}]
    assert {"feature": "Donor_Name_1", "reason": "categorical"} in analysis["slices_skipped"]
    # The categorical reference level is a level name, the same one the surface fixes.
    assert slices[0]["reference"]["recipe"]["Donor_Name_1"] in _NAMES


@pytest.mark.slow
@pytest.mark.serial
def test_real_slices_deterministic(analysis):
    again = _analyze_once(False)
    assert json.dumps(again["slices"]) == json.dumps(analysis["slices"])
    assert json.dumps(again["response_surface"]["std"]) == json.dumps(
        analysis["response_surface"]["std"])


def test_anonymized_slices_alias_identifier_names_and_levels():
    out = _analyze_once(True)
    blob = json.dumps({"slices": out["slices"], "skipped": out["slices_skipped"],
                       "surf": out["response_surface"]})
    assert "Donor_Name_1" not in blob
    for name in _NAMES:
        assert name not in blob
    assert out["slices"] and "Methanol" in blob
    for s in out["slices"]:
        assert any(k.startswith("col_") for k in s["reference"]["recipe"])
