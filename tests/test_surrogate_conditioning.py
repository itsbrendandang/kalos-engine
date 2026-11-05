"""Condition-number check on the fitted kernel matrix K + noise*I
(kalos/core/surrogate.py, compmodel PLAN E2). Warn above 1e8 (recorded in
`model_diagnostics`), refuse above 1e12 with a plain-language FitError.

Facts these tests pin: a well-spaced design is ~1e3; exact duplicates (replicates)
with inferred noise are fine, because the fitted noise keeps K + noise*I well
conditioned; a tight cluster with a FIXED tiny noise crosses 1e8. The refusal
threshold is not reachable through the likelihood noise floors on small designs,
so it is tested by lowering the threshold.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kalos.core import surrogate as surrogate_module
from kalos.core.surrogate import (
    COND_REFUSE,
    COND_WARN,
    FitError,
    Surrogate,
    condition_report,
)

B = np.array([[0.0, 0.0], [1.0, 1.0]])


def _spaced(n: int = 16):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (n, 2))
    return X, np.sin(3 * X[:, 0]) + X[:, 1] + rng.normal(0, 0.05, n)


def test_thresholds():
    assert COND_WARN == 1e8 and COND_REFUSE == 1e12


def test_well_spaced_design_is_not_flagged():
    d = Surrogate().fit(*_spaced(), bounds=B).fit_diagnostics
    assert d["condition_number"] < 1e5
    assert d["condition_warning"] is False and d["condition_notice"] is None


def test_duplicate_rows_with_inferred_noise_are_not_refused():
    X, y = _spaced()
    Xd = np.vstack([X, X[:4]])
    yd = np.r_[y, y[:4] + 0.02]
    d = Surrogate().fit(Xd, yd, bounds=B).fit_diagnostics
    assert np.isfinite(d["condition_number"]) and d["condition_warning"] is False


def test_near_duplicate_cluster_with_fixed_tiny_noise_warns():
    rng = np.random.default_rng(0)
    Xs = rng.uniform(0, 1, (12, 2))
    Xc = 0.5 + rng.uniform(0, 1e-6, (200, 2))
    X = np.vstack([Xs, Xc])
    y = np.r_[np.sin(3 * Xs[:, 0]), 0.5 + rng.normal(0, 1e-6, 200)]
    d = Surrogate().fit(X, y, bounds=B, noise=1e-9).fit_diagnostics
    assert d["condition_number"] > COND_WARN
    assert d["condition_warning"] is True
    assert "very close together" in d["condition_notice"]


def test_near_duplicate_is_worse_conditioned_than_well_spaced():
    X, y = _spaced()
    Xn = np.vstack([X, X[:4] + 1e-7])
    yn = np.r_[y, y[:4] + 1e-3]
    well = Surrogate().fit(X, y, bounds=B, noise=1e-6).fit_diagnostics["condition_number"]
    near = Surrogate().fit(Xn, yn, bounds=B, noise=1e-6).fit_diagnostics["condition_number"]
    assert near > 100 * well


def test_refusal_above_the_limit_is_plain_language(monkeypatch):
    monkeypatch.setattr(surrogate_module, "COND_REFUSE", 10.0)
    with pytest.raises(FitError, match="so close to each other") as exc:
        Surrogate().fit(*_spaced(), bounds=B)
    assert "near-duplicate" in str(exc.value)


def test_condition_report_refuses_inf_and_nan():
    for bad in (float("inf"), float("nan"), 2 * COND_REFUSE):
        with pytest.raises(FitError):
            condition_report(bad)
    assert condition_report(COND_WARN)["condition_warning"] is False
    assert condition_report(COND_WARN * 1.01)["condition_warning"] is True


def test_analysis_result_carries_the_conditioning_fields():
    import pandas as pd

    from kalos.portal.pipeline import compute_upload_analysis

    X, y = _spaced(24)
    raw = pd.DataFrame({"a": X[:, 0].round(4), "b": X[:, 1].round(4), "t": y.round(4)})
    _, result = compute_upload_analysis(raw.to_csv(index=False).encode(), "t", False, "x.csv")
    md = result["model_diagnostics"]
    assert md["condition_warning"] is False and md["condition_number"] > 1
    assert md["condition_warn_above"] == 1e8 and md["condition_refuse_above"] == 1e12
    json.dumps(md, allow_nan=False)  # JSON-safe
