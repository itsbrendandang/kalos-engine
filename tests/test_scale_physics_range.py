"""F10: a scale-up readout refuses or clearly warns when the planned target's
mixing and aeration (P/V, tip speed, gas velocity, vvm, kLa) sit outside what
the measured runs cover, and always states how far beyond the measured scales
the target is."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from kalos.scale import readout as ro
from kalos.scale.physics_range import (
    PHYSICS_EDGE_TOLERANCE,
    check_physics_range,
    extrapolation_statement,
    physics_frame,
)
from kalos.scale.transfer import DEFAULT_SCALE_FEATURE_CONFIG as CFG
from kalos.scale.transfer import ScaleFeatureConfig

SHEET = Path(__file__).resolve().parents[1] / "examples" / "synthetic_scaleup" / "synthetic_scaleup.csv"
PROCESS = ["ph_setpoint", "temperature_C"]
TARGET = "titer_g_per_L"


def _le(limit: float) -> pd.DataFrame:
    df = pd.read_csv(SHEET)
    return df[df["scale_L"] <= limit].reset_index(drop=True)


def _spec(scale: float, rpm: float, air: float) -> ro.TargetSpec:
    return ro.TargetSpec(scale_L=scale, agitation_rpm=rpm, airflow_L_per_min=air,
                         process_params={"ph_setpoint": 7.0, "temperature_C": 37.0})


def test_tip_speed_and_vvm_follow_the_stated_geometry() -> None:
    row = physics_frame(pd.DataFrame([{"scale_L": 1000.0, "agitation_rpm": 60.0, "airflow_L_per_min": 50.0}]), CFG)
    tank_d = (4 * 1.0 / (math.pi * 1.0)) ** (1 / 3)  # 1 m3, H/T = 1
    assert row["tip_speed_m_per_s"].iloc[0] == pytest.approx(math.pi * tank_d / 3 * 1.0)
    assert row["vvm_per_min"].iloc[0] == pytest.approx(0.05)


def test_2000_L_at_240_rpm_from_a_300_L_ladder_is_refused_with_a_plan() -> None:
    out = ro.build_readout(_le(300), TARGET, PROCESS, _spec(2000, 240, 100))
    assert out["decision"] == ro.REFUSAL
    assert out["prediction"] is None
    phys = out["physics_range"]
    assert phys["decision"] == "refuse"
    pv = next(q for q in phys["quantities"] if q["quantity"] == "specific_power_w_per_m3")
    assert pv["status"] == "far_outside" and pv["factor_outside"] > 3
    assert 150 < pv["setting_to_stay_inside"]["agitation_rpm"] < 180
    assert any("power per volume" in r and "rpm" in r for r in out["reasons"])
    assert out["extrapolation"]["factor"] == pytest.approx(2000 / 300, rel=1e-3)
    assert "extrapolation" in out["extrapolation"]["message"]


def test_the_same_scale_with_planned_settings_inside_the_runs_is_not_refused_for_physics() -> None:
    phys = check_physics_range(_le(300), _spec(2000, 150, 100), CFG)
    assert phys["decision"] in ("none", "warn")
    assert not any(q["refuses"] for q in phys["quantities"])


def test_tip_speed_and_gas_velocity_only_ever_warn() -> None:
    # constant vvm and P/V at a larger scale push vs and tip speed above the
    # measured range by design of the usual scale-up rules
    df = _le(300)
    phys = check_physics_range(df, _spec(6000, 120, 300), CFG)
    for q in phys["quantities"]:
        if q["quantity"] in ("tip_speed_m_per_s", "superficial_gas_velocity_m_per_s"):
            assert q["refuses"] is False


def test_inside_the_measured_runs_says_nothing() -> None:
    df = _le(5000)
    phys = check_physics_range(df, _spec(300, 150, 15), CFG)
    assert phys["decision"] == "none" and phys["reasons"] == []


def test_a_5_percent_edge_is_inside() -> None:
    df = _le(5000)
    vmin = physics_frame(df, CFG)["vvm_per_min"].min()
    phys = check_physics_range(df, _spec(300, 150, 300 * vmin / (PHYSICS_EDGE_TOLERANCE - 0.01)), CFG)
    vvm = next(q for q in phys["quantities"] if q["quantity"] == "vvm_per_min")
    assert vvm["status"] == "inside"


def test_scale_only_sheets_are_not_checked_and_say_why() -> None:
    cfg = ScaleFeatureConfig(feature_set="scale_only")
    phys = check_physics_range(_le(300), _spec(2000, 240, 100), cfg)
    assert phys["checked"] is False and "agitation" in phys["reason"]


def test_extrapolation_statement() -> None:
    s = extrapolation_statement(2000.0, 300.0)
    assert s is not None and "6.7x the largest scale measured (300 L)" in s["message"]
    assert extrapolation_statement(300.0, 300.0)["message"].startswith("300 L is within")  # type: ignore[index]
    assert extrapolation_statement(10.0, None) is None
