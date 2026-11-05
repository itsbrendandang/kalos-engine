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
    df = pd.DataFrame({"pH": [7.0, 7.1, 7.2, 7.3], "titer": [1, 9, 2, 3], "QC Status": ["pass", "FAIL", " ", "Rejected"]})
    out, info = exclude_qc_failed(df)
    assert info.column == "QC Status" and info.n_excluded == 2
    assert info.rows == (1, 3)
    assert info.tokens == {"fail": 1, "rejected": 1}
    assert list(out.index) == [0, 2]  # labels kept: row numbers still point at the sheet
    assert "QC Status" not in out.columns
    df2 = pd.DataFrame({"x": [1, 2, 3], "contamination": ["Y", "N", "yes"]})
    out2, info2 = exclude_qc_failed(df2, keep_columns={"contamination"})
    assert info2.n_excluded == 2 and "contamination" in out2.columns
    assert find_qc_column(pd.DataFrame({"qc_operator": [1]})) == (None, None)


def test_numeric_columns_are_never_flags():
    # A numeric contamination LEVEL (0/1/2) is a measurement, not a flag, and a
    # pure 0/1 numeric column is not read as fail on 1 either.
    for vals in ([0, 1, 2, 1], [0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]):
        lv = pd.DataFrame({"x": [1, 2, 3, 4], "contamination": vals})
        out, info = exclude_qc_failed(lv)
        assert info.n_excluded == 0 and info.skipped_reason and len(out) == 4
    # ... also when declared explicitly
    out, info = exclude_qc_failed(pd.DataFrame({"x": [1, 2], "qc": [0, 1]}), column="qc")
    assert info.n_excluded == 0
    # Free text in a qc column: not a flag, nothing dropped.
    t = pd.DataFrame({"x": [1, 2], "qc": ["fail", "operator JS"]})
    assert exclude_qc_failed(t)[1].n_excluded == 0
    # Boolean-like text is: true/false, ok/contaminated.
    tf = pd.DataFrame({"x": [1, 2, 3], "contaminated": ["true", "False", "TRUE"]})
    assert exclude_qc_failed(tf)[1].n_excluded == 2
    oc = pd.DataFrame({"x": [1, 2], "qc": ["ok", "contaminated"]})
    info = exclude_qc_failed(oc)[1]
    assert info.n_excluded == 1 and info.triggers() == [{"row": 1, "column": "qc", "value": "contaminated"}]


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


def _roles(**extra) -> dict:
    return {"target": "titer_g_L", "features": list(FACTORS), **extra}


def test_portal_keeps_flagged_rows_by_default_and_says_so():
    from kalos.portal.analysis import _analyze

    df = _upload_frame(0)
    fail_rows = [int(i) for i in df.index[df["qc"] == "fail"]]
    res = _analyze(df, "titer_g_L")
    assert res["qc_excluded"]["requested"] is False
    assert res["qc_excluded"]["n_excluded"] == 0 and res["data_drops"]["qc_rows_excluded"] == 0
    found = [f for f in res["validation"]["findings"] if f["check"] == "qc_flags"]
    assert found and found[0]["severity"] == "info" and found[0]["rows"] == fail_rows


def test_portal_opt_in_excludes_flagged_rows_attributably():
    from kalos.domains import GENERIC_PROFILE, ColumnRoles
    from kalos.portal.analysis import _analyze

    df = _upload_frame(0)
    fail_rows = [int(i) for i in df.index[df["qc"] == "fail"]]
    n_fail = len(fail_rows)
    assert n_fail >= 1
    roles = ColumnRoles.from_dict(_roles(exclude_flagged_rows=True))
    res = _analyze(df, roles=roles, profile=GENERIC_PROFILE)
    qc = res["qc_excluded"]
    assert qc["requested"] is True and qc["column"] == "qc" and qc["n_excluded"] == n_fail
    assert qc["rows"] == fail_rows
    assert qc["excluded"] == [{"row": r, "column": "qc", "value": "fail"} for r in fail_rows]
    assert res["data_drops"]["qc_rows_excluded"] == n_fail
    assert res["n"] == len(df) - n_fail
    found = [f for f in res["validation"]["findings"] if f["check"] == "qc_flags"]
    assert found and found[0]["severity"] == "warning" and found[0]["rows"] == fail_rows
    fail_min = float(df.loc[df["qc"] == "fail", "titer_g_L"].min())
    for p in res["proposals"]:
        assert float(p["pred"]) < fail_min


def test_declared_qc_flag_role_and_audit_record():
    import json

    from kalos.portal.pipeline import compute_upload_analysis

    df = _upload_frame(0).rename(columns={"qc": "assay_check"})  # not QC-named
    fail_rows = [int(i) for i in df.index[df["assay_check"] == "fail"]]
    raw = df.to_csv(index=False).encode()
    roles = json.dumps(_roles(exclude_flagged_rows=True, qc="assay_check"))
    _, res = compute_upload_analysis(raw, None, False, "qc.csv", roles)
    assert res["qc_excluded"]["rows"] == fail_rows
    audit = res["run"]["data_drops"]
    assert audit["qc_rows_excluded"] == len(fail_rows)
    assert audit["qc_excluded"] == [{"row": r, "column": "assay_check", "value": "fail"} for r in fail_rows]
    assert res["run"]["roles_declared"]["exclude_flagged_rows"] is True


def test_portal_validation_sees_the_unfiltered_sheet():
    from kalos.domains import GENERIC_PROFILE, ColumnRoles
    from kalos.portal.analysis import _analyze

    df = _upload_frame(0)
    res = _analyze(df, roles=ColumnRoles.from_dict(_roles(exclude_flagged_rows=True)), profile=GENERIC_PROFILE)
    assert res["qc_excluded"]["n_excluded"] >= 1
    assert res["validation"]["n_rows"] == len(df)


def test_portal_numeric_contamination_level_is_not_excluded():
    from kalos.domains import GENERIC_PROFILE, ColumnRoles
    from kalos.portal.analysis import _analyze

    df = _upload_frame(1).drop(columns=["qc"])
    df["contamination"] = np.resize([0, 1, 2], len(df))
    res = _analyze(df, roles=ColumnRoles.from_dict(_roles(exclude_flagged_rows=True)), profile=GENERIC_PROFILE)
    assert res["qc_excluded"]["n_excluded"] == 0 and res["n"] == len(df)
