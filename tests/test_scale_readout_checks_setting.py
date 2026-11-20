"""`KALOS_SCALE_READOUT_CHECKS` (kalos/scale/settings.py): the three readout
honesty checks are on by default and can be switched off, which restores the
earlier decisions, reasons and numbers. Both states are tested. Synthetic data
only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.envconfig import ConfigError
from kalos.portal.config import check_env_settings
from kalos.scale import readout as ro
from kalos.scale.evaluation import leave_one_scale_out_report
from kalos.scale.settings import READOUT_CHECKS_VAR, readout_checks_enabled

_DEMO_CSV = "examples/synthetic_scaleup/synthetic_scaleup.csv"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(READOUT_CHECKS_VAR, raising=False)


def _method_ladder() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scale_L": [1.5, 4.0, 20.0, 2500.0, 30000.0],
            "titer_g_L": [0.72, 0.70, 1.84, 0.69, 1.52],
            "titer_method": ["SDS-PAGE", None, "RP-HPLC", "SDS-PAGE", "RP-HPLC"],
        }
    )


def _funnel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    labels = ["384w", "96w", "24w", "SF125"]
    return pd.DataFrame(
        {
            "scale_L": [lbl for lbl in labels for _ in range(6)],
            "ph": np.round(rng.uniform(6.8, 7.2, 24), 2),
            "titer": np.round(rng.uniform(1.0, 3.0, 24), 2),
        }
    )


@pytest.mark.parametrize(("raw", "expected"), [("", True), ("on", True), (" ON ", True), ("off", False), ("Off", False)])
def test_setting_values(monkeypatch, raw, expected):
    monkeypatch.setenv(READOUT_CHECKS_VAR, raw)
    assert readout_checks_enabled() is expected


def test_unset_means_on():
    assert readout_checks_enabled() is True


def test_typo_refuses_instead_of_switching_checks_off(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "of")
    with pytest.raises(ConfigError, match=READOUT_CHECKS_VAR):
        readout_checks_enabled()
    with pytest.raises(ConfigError, match=READOUT_CHECKS_VAR):
        check_env_settings()


# --- assay-method confound ----------------------------------------------------- #


def _ladder_readout() -> dict:
    return ro.build_readout(_method_ladder(), "titer_g_L", [], ro.TargetSpec(scale_L=90000.0, process_params={}))


def test_confound_on(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "on")
    out = _ladder_readout()
    assert out["assay_confound"]["severity"] == "refuse"
    assert len(out["reasons"]) == 3
    assert out["provenance"]["readout_checks"] == "on"


def test_confound_off_restores_the_earlier_reasons(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "off")
    out = _ladder_readout()
    assert out["decision"] == ro.REFUSAL
    assert out["reasons"] == [
        "only 1 pooled ladder residual(s); need at least 10",
        "no rung with at least 3 runs, within 1 decade(s) of the largest trained scale, beat both naive "
        "baselines to license a reference step ratio",
    ]
    assert out["assay_confound"]["severity"] is None
    assert out["assay_confound"]["skipped"][0]["reason"] == f"check not run: {READOUT_CHECKS_VAR}=off"
    assert not any("assay method" in step for step in out["data_plan"])
    assert out["provenance"]["readout_checks"] == "off"


# --- plate/flask volumes ---------------------------------------------------------- #


def _funnel_readout() -> dict:
    return ro.build_readout(_funnel(), "titer", ["ph"], ro.TargetSpec(scale_L=0.25, process_params={"ph": 7.0}))


def test_volumes_on(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "on")
    out = _funnel_readout()
    assert out["gate"]["passed"] and out["working_volumes"]["applied"]


def test_volumes_off_restores_the_missing_column_refusal(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "off")
    out = _funnel_readout()
    assert out["decision"] == ro.REFUSAL
    assert out["reasons"] == ["missing required column(s): ['scale_L']"]
    assert out["working_volumes"]["applied"] is False
    assert not any(k.startswith("nominal_working_volume_L") for k in out["physics_assumptions"])


# --- LOSO ----------------------------------------------------------------------- #


def test_loso_on_raises(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "on")
    with pytest.raises(ValueError, match="agitation_rpm"):
        leave_one_scale_out_report(_method_ladder()[["scale_L", "titer_g_L"]], "titer_g_L", [])


def test_loso_off_returns_the_earlier_empty_report(monkeypatch):
    monkeypatch.setenv(READOUT_CHECKS_VAR, "off")
    report = leave_one_scale_out_report(_method_ladder()[["scale_L", "titer_g_L"]], "titer_g_L", [])
    assert (report["n_rows_used"], report["n_rows_dropped"], report["n_scales"]) == (0, 5, 0)


# --- a sheet without method columns or labels reads the same either way ---------- #


def test_plain_sheet_identical_in_both_states(monkeypatch):
    df = pd.read_csv(_DEMO_CSV)
    target = ro.TargetSpec(
        scale_L=7500.0, agitation_rpm=150.0, airflow_L_per_min=10.0,
        process_params={"ph_setpoint": 7.0, "temperature_C": 36.7},
    )
    keys = ("decision", "reasons", "prediction", "interval", "rungs", "data_plan", "physics_assumptions")
    monkeypatch.setenv(READOUT_CHECKS_VAR, "on")
    on = ro.build_readout(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], target)
    monkeypatch.setenv(READOUT_CHECKS_VAR, "off")
    off = ro.build_readout(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], target)
    assert {k: on[k] for k in keys} == {k: off[k] for k in keys}
