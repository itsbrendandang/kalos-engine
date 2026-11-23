"""BRE-118: the outlier check is quiet on clean, deliberately varied sheets and
still flags x10 / x1000 entry slips, with the value a slip would have come from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.domains import BIOPROCESS_PROFILE
from kalos.validation import report_dict, validate_frame
from kalos.validation.checks import _jump_threshold, check_outliers

ROOT = Path(__file__).resolve().parents[1]
BIO = ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"
SCALE = ROOT / "examples" / "synthetic_scaleup" / "synthetic_scaleup.csv"


def _outliers(df: pd.DataFrame, target: str) -> list[dict]:
    report = report_dict(validate_frame(df, target=target, profile=BIOPROCESS_PROFILE))
    return [f for f in report["findings"] if f["check"] == "outliers"]


@pytest.mark.parametrize("n", [30, 60, 160])
def test_clean_bioprocess_sheet_raises_no_outlier(n: int) -> None:
    # Before BRE-118: 60 of 160 rows flagged (the zero-inflated titer's best runs
    # and the lowest antifoam settings).
    df = pd.read_csv(BIO).sample(n, random_state=3).reset_index(drop=True) if n < 160 else pd.read_csv(BIO)
    assert _outliers(df, "titer_g_L") == []


def test_clean_scaleup_sheet_raises_no_outlier() -> None:
    # Titer falls at the largest scales: an effect of the design, not an outlier.
    assert _outliers(pd.read_csv(SCALE), "titer_g_per_L") == []


def test_x10_and_x1000_target_slips_are_flagged_with_the_suspected_value() -> None:
    df = pd.read_csv(BIO)
    order = df["titer_g_L"].sort_values().index
    i10, i1000 = int(order[-3]), int(order[-20])
    true10, true1000 = float(df.at[i10, "titer_g_L"]), float(df.at[i1000, "titer_g_L"])
    df.loc[i10, "titer_g_L"] = true10 * 10
    df.loc[i1000, "titer_g_L"] = true1000 * 1000
    (finding,) = _outliers(df, "titer_g_L")
    assert finding["column"] == "titer_g_L"
    assert set(finding["rows"]) == {i10, i1000}
    flagged = {f["row"]: f for f in finding["detail"]["flagged"]}
    assert flagged[i1000]["suspected_value"] == pytest.approx(true1000)
    assert flagged[i1000]["slip_factor"] == 1000
    assert "nothing was changed" in finding["message"]


def test_a_second_slip_cannot_hide_the_first() -> None:
    # 60 positive titers (threshold 3x). Two slips: one lands just above the
    # clean best (1.2x), the other 3.3x above the clean best but only 2.75x
    # above the first slip. Compared only with its neighbour it would hide.
    rng = np.random.default_rng(0)
    y = np.sort(np.round(rng.lognormal(-1.0, 0.4, 60), 3))
    best = float(y[-1])
    y = np.r_[y, round(1.2 * best, 3), round(3.3 * best, 3)]
    df = pd.DataFrame({"titer": y})
    (finding,) = check_outliers(df, outcome_columns=["titer"])
    assert 61 in finding.rows
    assert finding.detail["factor"] >= 3.0


def test_uniformly_varied_setting_near_zero_is_not_an_outlier() -> None:
    # antifoam 0.2-50 mg/L, uniform: its lowest settings span two decades on a
    # log scale but are the design (the old rule flagged them).
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"antifoam_mgL": np.r_[0.17, 0.48, rng.uniform(1, 50, 60)]})
    assert check_outliers(df) == []


def test_x10_slip_on_a_varied_setting_is_flagged() -> None:
    rng = np.random.default_rng(2)
    glucose = rng.uniform(2, 12, 80).round(2)
    glucose[5] = glucose[5] * 10 if glucose[5] > 5 else 60.0
    (finding,) = check_outliers(pd.DataFrame({"glucose_gL": glucose}))
    assert 5 in finding.rows


def test_designed_ladder_levels_are_not_slips() -> None:
    ladder = [1, 3, 10, 30, 100, 300, 1000, 3000, 5000]
    rng = np.random.default_rng(3)
    scale = np.repeat(ladder, 3).astype(float)
    airflow = scale * rng.uniform(0.045, 0.06, len(scale))
    df = pd.DataFrame({"scale_L": scale, "airflow_L_per_min": airflow})
    for drop in range(0, len(df), 4):  # missing rungs are still the design
        assert check_outliers(df.drop(index=range(drop, min(drop + 6, len(df))))) == []


def test_jump_threshold_shrinks_with_more_values() -> None:
    assert _jump_threshold(5) is None
    assert _jump_threshold(10) == 10.0
    assert _jump_threshold(30) == 5.0
    assert _jump_threshold(60) == 3.0
    assert _jump_threshold(150) == 2.0


def test_zero_inflated_target_best_runs_are_not_flagged() -> None:
    rng = np.random.default_rng(4)
    y = np.where(rng.uniform(size=150) < 0.45, 0.0, rng.lognormal(-1.5, 0.9, 150)).round(3)
    df = pd.DataFrame({"pH": rng.uniform(6, 7.6, 150), "titer_g_L": y})
    assert check_outliers(df, outcome_columns=["titer_g_L"]) == []
