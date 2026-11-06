"""Every finding has a stable non-null `code` (the check name when no more
specific code exists), and `promotion.status` tells a failed gate from an
unmeasurable one without changing the verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.gates import check_gates
from kalos.validation import report_dict, validate_frame


def _messy_sheet() -> pd.DataFrame:
    """Units in cells, a blank, a censored value, a null token, an outlier,
    an auto categorical, an id column."""
    rng = np.random.default_rng(5)
    n = 16
    temp = [f"{t:.1f} C" for t in rng.uniform(30, 37, n)]
    ph = rng.uniform(6.0, 7.5, n).round(2).astype(object)
    ph[3] = np.nan
    ph[4] = "n/a"
    titer = rng.uniform(1, 5, n).round(3).astype(object)
    titer[6] = "<LOD"
    titer[7] = 500.0
    return pd.DataFrame({
        "run_id": [f"R{i}" for i in range(n)], "medium": ["A", "B"] * (n // 2),
        "temperature": temp, "pH": ph, "feed_code": rng.choice([-1.0, 0.0, 1.0], n),
        "titer": titer,
    })


def _cho_like() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 14
    return pd.DataFrame({
        "": np.arange(n), "pH": rng.uniform(6.98, 7.03, n).round(3),
        "agitation_rpm": rng.choice([100.0, 150.0, 200.0], n),
        "IVCD": rng.uniform(1000, 3000, n), "OD_end": rng.uniform(1, 5, n),
    })


def _codes(findings: list[dict]) -> list:
    return [f.get("code") for f in findings]


def test_validation_report_findings_all_have_codes():
    rep = report_dict(validate_frame(_messy_sheet(), target="titer"))
    assert rep["findings"]
    assert all(isinstance(c, str) and c for c in _codes(rep["findings"]))


@pytest.mark.parametrize("sheet,target", [(_messy_sheet, "titer"), (_cho_like, "IVCD")])
def test_analysis_findings_all_have_codes(sheet, target):
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    out = _analyze(sheet(), target=target)
    findings = out["validation"]["findings"]
    assert findings
    missing = [f["check"] for f in findings if not f.get("code")]
    assert not missing, missing
    assert out["promotion"]["status"] in {"passed", "failed", "not_measurable"}
    assert (out["promotion"]["status"] == "passed") == out["promotion"]["passed"]


def test_promotion_status_tells_unmeasured_from_failed():
    nan = float("nan")
    unmeasured = check_gates({"surrogate_spearman": 0.6, "feasibility_auc": nan, "ece": nan,
                              "brier": nan})
    assert not unmeasured.passed and unmeasured.status == "not_measurable"
    assert unmeasured.unmeasured == ["feasibility_auc", "ece", "brier"]
    failed = check_gates({"surrogate_spearman": 0.05, "feasibility_auc": nan, "ece": nan,
                          "brier": nan})
    assert not failed.passed and failed.status == "failed"
    ok = check_gates({"surrogate_spearman": 0.6, "feasibility_auc": 0.9, "ece": 0.05,
                      "brier": 0.1})
    assert ok.passed and ok.status == "passed"
