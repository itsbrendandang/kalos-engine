"""Tests for torch-free ShiftWeightedConformal and AdaptiveConformalState."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from kalos.core.conformal import normalized_conformal_quantile
from kalos.core.frontier.shift_conformal import (
    AdaptiveConformalState,
    ShiftWeightedConformal,
)


def test_shift_conformal_is_torch_free() -> None:
    """Importing shift_conformal in a clean subprocess must never import torch."""
    code = (
        "import sys; "
        "import kalos.core.frontier.shift_conformal as sc; "
        "assert 'torch' not in sys.modules, 'torch leaked into sys.modules'; "
        "assert 'botorch' not in sys.modules, 'botorch leaked into sys.modules'; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "ok"


def test_uniform_weights_match_standard_split_conformal_quantile() -> None:
    """With beta=0 and no X_cal/X_proposals, weights are 1.0 and match split conformal."""
    rng = np.random.default_rng(7)
    n = 39
    oof_pred = rng.normal(5.0, 1.0, size=n)
    oof_std = rng.uniform(0.3, 1.2, size=n)
    oof_actual = oof_pred + rng.normal(0.0, oof_std)

    swc = ShiftWeightedConformal(beta=0.0, w_min=0.1, w_max=10.0, min_n_eff=5.0)
    cal = swc.calibrate(oof_actual, oof_pred, oof_std, levels=(0.90, 0.95), sigma_floor_quantile=0.0)
    assert cal["ok"] is True
    assert cal["n_eff"] == pytest.approx(float(n), rel=1e-9)

    sigma_floor = float(np.min(oof_std))
    expected_scores = np.abs(oof_actual - oof_pred) / np.maximum(oof_std, sigma_floor)

    for cov in (0.90, 0.95):
        ref = normalized_conformal_quantile(expected_scores, 1.0 - cov)
        assert ref["available"] is True
        iv = swc.interval(6.0, 0.8, cal_result=cal, coverage=cov)
        assert iv["available"] is True
        assert iv["q"] == pytest.approx(float(ref["q"]), rel=1e-12)
        assert iv["lo"] == pytest.approx(6.0 - float(ref["q"]) * 0.8, rel=1e-12)
        assert iv["hi"] == pytest.approx(6.0 + float(ref["q"]) * 0.8, rel=1e-12)


def test_high_shift_widens_intervals_while_staying_tighter_than_p90_floor() -> None:
    """High-shift proposals get wider quantiles, yet well-behaved sigmas beat 90th-pct floor."""
    rng = np.random.default_rng(19)
    n = 80
    X_cal = rng.uniform(0.0, 1.0, size=(n, 3))
    oof_pred = 4.0 * X_cal[:, 0] + rng.normal(0.0, 0.1, size=n)
    # Well-behaved varying sigmas: tight near center, wide at edges (0.1 to 1.0)
    oof_std = 0.1 + 0.9 * (X_cal[:, 0] ** 2)
    # High-predicted region has slightly heavier tail residuals
    err_scale = oof_std * (1.0 + 0.6 * (X_cal[:, 0] > 0.7))
    oof_actual = oof_pred + rng.normal(0.0, err_scale)

    # Proposals concentrated in the shifted high-prediction region
    X_prop = np.column_stack([
        rng.uniform(0.80, 0.95, size=5),
        rng.uniform(0.40, 0.60, size=5),
        rng.uniform(0.40, 0.60, size=5),
    ])

    swc = ShiftWeightedConformal(beta=1.0, w_min=0.1, w_max=10.0, min_n_eff=8.0)
    cal = swc.calibrate(
        oof_actual, oof_pred, oof_std, X_cal=X_cal, X_proposals=X_prop, levels=(0.90,)
    )
    assert cal["ok"] is True

    # Low-shift point vs high-shift point at identical pred_std
    iv_low = swc.interval(
        0.5, 0.2, pred_X=np.array([0.1, 0.5, 0.5]), cal_result=cal, coverage=0.90
    )
    iv_high = swc.interval(
        3.8, 0.2, pred_X=np.array([0.9, 0.5, 0.5]), cal_result=cal, coverage=0.90
    )
    assert iv_low["available"] is True
    assert iv_high["available"] is True
    assert iv_high["w_new"] > iv_low["w_new"]
    assert iv_high["q"] >= iv_low["q"]
    assert iv_high["half_width"] >= iv_low["half_width"]

    # Well-behaved small sigma (0.15) should produce a tighter interval than blunt 90th-pct sd floor
    p90_floor = float(np.quantile(oof_std, 0.90))
    assert cal["sigma_floor"] < p90_floor
    assert iv_high["half_width"] < float(iv_high["q"]) * p90_floor


def test_fails_closed_when_p_new_exceeds_alpha_or_n_eff_below_min() -> None:
    """Intervals refuse cleanly (available=False) on extreme shift or low n_eff."""
    rng = np.random.default_rng(11)
    n = 15
    oof_pred = np.linspace(0.0, 1.0, n)
    oof_std = np.full(n, 0.2)
    oof_actual = oof_pred + rng.normal(0.0, 0.2, size=n)

    # Extreme tilt -> w_new = w_max = 10.0 so p_new > 0.10 at n=15
    swc_shift = ShiftWeightedConformal(beta=4.0, w_min=0.1, w_max=10.0, min_n_eff=2.0)
    cal_shift = swc_shift.calibrate(oof_actual, oof_pred, oof_std, levels=(0.90,))
    assert cal_shift["ok"] is True
    iv_extreme = swc_shift.interval(10.0, 0.2, cal_result=cal_shift, coverage=0.90)
    assert iv_extreme["available"] is False
    assert iv_extreme["code"] == "shift_too_large"
    assert iv_extreme["lo"] is None and iv_extreme["hi"] is None

    # n_eff < min_n_eff -> fails closed even for mild query
    swc_neff = ShiftWeightedConformal(beta=3.0, w_min=0.1, w_max=10.0, min_n_eff=50.0)
    cal_neff = swc_neff.calibrate(oof_actual, oof_pred, oof_std, levels=(0.90,))
    assert cal_neff["ok"] is True
    assert cal_neff["available"] is False
    iv_neff = swc_neff.interval(0.5, 0.2, cal_result=cal_neff, coverage=0.90)
    assert iv_neff["available"] is False
    assert iv_neff["code"] == "insufficient_n_eff"


def test_degenerate_sigma_refused() -> None:
    """Non-finite or non-positive OOF sigma refuses calibration cleanly."""
    swc = ShiftWeightedConformal()
    cal_zero = swc.calibrate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.2, 0.0, 0.3])
    assert cal_zero["ok"] is False
    assert cal_zero["code"] == "degenerate_sigma"
    iv = swc.interval(1.5, 0.2, cal_result=cal_zero, coverage=0.90)
    assert iv["available"] is False
    assert iv["code"] == "degenerate_sigma"

    cal_nan = swc.calibrate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.2, np.nan, 0.3])
    assert cal_nan["ok"] is False
    assert cal_nan["code"] == "degenerate_sigma"


def test_vectorized_interval_handles_mixed_availability() -> None:
    """Array inputs return per-element availability masks and valid bounds where available."""
    rng = np.random.default_rng(23)
    n = 25
    oof_pred = rng.normal(0.0, 1.0, size=n)
    oof_std = np.full(n, 0.5)
    oof_actual = oof_pred + rng.normal(0.0, 0.5, size=n)

    swc = ShiftWeightedConformal(beta=2.0, w_min=0.1, w_max=10.0, min_n_eff=5.0)
    cal = swc.calibrate(oof_actual, oof_pred, oof_std, levels=(0.90,))
    mus = np.array([0.0, 20.0])  # first normal, second extreme shift -> p_new > 0.10
    sds = np.array([0.5, 0.5])
    out = swc.interval(mus, sds, cal_result=cal, coverage=0.90)
    assert isinstance(out["available"], np.ndarray)
    assert out["available"].tolist() == [True, False]
    assert np.isfinite(out["lo"][0]) and np.isfinite(out["hi"][0])
    assert np.isnan(out["lo"][1]) and np.isnan(out["hi"][1])
    assert out["codes"] == [None, "shift_too_large"]


def test_adaptive_conformal_state_tracks_coverage_under_drift() -> None:
    """AdaptiveConformalState adapts alpha_t under distribution drift toward 1 - nominal_alpha."""
    rng = np.random.default_rng(101)
    aci = AdaptiveConformalState(nominal_alpha=0.10, gamma=0.10, min_alpha=0.005, max_alpha=0.50)

    # Simulate 60 rounds where noise scale suddenly doubles at round 20 (distribution drift).
    # Standard normal quantile z = ndtri(1 - current_alpha / 2) scales interval half-width.
    from scipy.special import ndtri

    for t in range(60):
        noise_sd = 1.0 if t < 20 else 2.0
        half_w = float(ndtri(1.0 - aci.current_alpha / 2.0))
        errs = np.abs(rng.normal(0.0, noise_sd, size=5))
        covered = (errs <= half_w).tolist()
        aci.update(covered)

    assert aci.n_rounds == 60
    assert len(aci.round_coverages) == 60
    # After drift, ACI must lower current_alpha below nominal 0.10 to compensate for 2x noise
    assert aci.current_alpha < 0.10
    # Cumulative coverage stays close to nominal 0.90 despite the 2x variance shift
    assert abs(aci.cumulative_coverage - 0.90) < 0.08
