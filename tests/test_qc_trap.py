"""QC trap: contaminated runs read HIGH. The optimizer must still find the
optimum once QC-failed rows are excluded, a QC-failed row must never be the
incumbent, a temperature cap must be honoured, and the portal must drop
QC-failed rows from an upload before the fit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.bench.qc_trap import (
    FACTORS,
    make_campaign,
    qc_trap_objective,
    run_qc_trap,
)
from kalos.validation.qc import exclude_qc_failed, find_qc_column


def test_ported_campaign_is_deterministic_and_trapped():
    a, b = make_campaign("C", 7), make_campaign("C", 7)
    assert a.runs == b.runs
    fails = [r for r in a.runs if r["qc"] == "fail"]
    passes = [r for r in a.runs if r["qc"] == "pass"]
    assert 1 <= len(fails) <= 2
    # The trap: every contaminated reading beats every real run.
    assert min(r["titer_g_L"] for r in fails) > max(r["titer_g_L"] for r in passes)


def test_temp_cap_binds_the_feasible_optimum():
    obj = qc_trap_objective(3, temp_cap=True)
    c = obj.campaign
    assert c.temp_cap is not None and c.temp_cap < c.opt["temp_C"]
    assert obj.bounds[1, FACTORS.index("temp_C")] == pytest.approx(c.temp_cap)
    assert obj.argmax[FACTORS.index("temp_C")] == pytest.approx(c.temp_cap)
    assert obj.optimum < c.peak  # the cap costs titer


def test_exclude_qc_failed_rules():
    df = pd.DataFrame({"pH": [7.0, 7.1, 7.2, 7.3], "titer": [1, 9, 2, 3], "QC Status": ["pass", "FAIL", " ", "weird"]})
    out, info = exclude_qc_failed(df)
    assert info.column == "QC Status" and info.n_excluded == 1
    assert list(out.index) == [0, 2, 3]  # labels kept: row numbers still point at the sheet
    assert "QC Status" not in out.columns
    df2 = pd.DataFrame({"x": [1, 2, 3], "contamination": ["Y", "N", "yes"]})
    out2, info2 = exclude_qc_failed(df2, keep_columns={"contamination"})
    assert info2.n_excluded == 2 and "contamination" in out2.columns
    assert find_qc_column(pd.DataFrame({"qc_operator": [1]})) == (None, None)


@pytest.mark.parametrize("temp_cap", [False, True])
def test_optimizer_under_the_trap_still_finds_the_optimum(temp_cap):
    regrets, trapped = [], []
    for seed in range(3):
        obj = qc_trap_objective(seed, temp_cap=temp_cap)
        r = run_qc_trap(obj, filter_qc=True, budget=10, n_init=8, seed=seed)
        assert not r["incumbent_is_qc_fail"]
        assert r["temp_cap_honoured"]
        regrets.append(r["incumbent_regret"] / r["optimum"])
        naive = run_qc_trap(obj, filter_qc=False, budget=10, n_init=8, seed=seed)
        if naive["n_qc_fail"]:
            trapped.append(naive["incumbent_is_qc_fail"])
    # Filtered BO ends within 20% of the feasible optimum on average.
    assert float(np.mean(regrets)) < 0.20
    # The unfiltered arm reports a contaminated run as its best recipe.
    assert trapped and all(trapped)


def _upload_frame(seed: int = 0) -> pd.DataFrame:
    c = make_campaign("P", seed)
    df = pd.DataFrame(c.runs)
    return df.drop(columns=["run_id"])


def test_portal_excludes_qc_failed_rows_before_the_fit():
    from kalos.portal.analysis import _analyze

    df = _upload_frame(0)
    n_fail = int((df["qc"] == "fail").sum())
    assert n_fail >= 1
    res = _analyze(df, "titer_g_L")
    assert res["qc_excluded"] == {"column": "qc", "n_excluded": n_fail, "column_removed": True}
    assert res["n"] == len(df) - n_fail
    assert "qc" not in res["features"] and "qc" not in res.get("categorical_features", [])
    # The proposed batch is not anchored on a contaminated reading: the fitted
    # incumbent is a real run, so every proposal's predicted titer is below the
    # inflated (>= 1.1 x peak) contaminated readings.
    fail_min = float(df.loc[df["qc"] == "fail", "titer_g_L"].min())
    for p in res["proposals"]:
        assert float(p["pred"]) < fail_min


def test_portal_without_qc_column_reports_nothing_excluded():
    from kalos.portal.analysis import _analyze

    df = _upload_frame(1)
    df = df[df["qc"] == "pass"].drop(columns=["qc"]).reset_index(drop=True)
    res = _analyze(df, "titer_g_L")
    assert res["qc_excluded"] == {"column": None, "n_excluded": 0, "column_removed": False}
    assert res["n"] == len(df)
