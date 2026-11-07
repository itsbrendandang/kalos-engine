"""The carry-forward baseline (same recipe, nearest lower scale) in the
leave-one-scale-out report, and the readout gate refusing when the model does
not beat it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kalos.scale import readout as ro
from kalos.scale.evaluation import _naive_carry_forward, leave_one_scale_out_report

CSV = Path(__file__).resolve().parents[1] / "examples" / "synthetic_scaleup" / "synthetic_scaleup.csv"


def test_carry_forward_uses_the_nearest_lower_scale_and_recipe_identity():
    proc = np.array([[7.0], [7.2], [7.0], [7.2]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    scale = np.array([1.0, 1.0, 10.0, 10.0])
    recipe = np.array(["a", "b", "a", "b"], dtype=object)
    out = _naive_carry_forward(proc, y, scale, np.array([[7.0], [7.2], [9.9]]), 100.0,
                               recipe, np.array(["a", "b", "zz"], dtype=object))
    assert out[0] == 3.0 and out[1] == 4.0 and np.isnan(out[2])  # 10 L, not 1 L; unseen recipe -> nan
    # nothing smaller than the held-out scale -> not applicable
    assert np.isnan(_naive_carry_forward(proc, y, scale, proc[:1], 0.5)).all()
    # no recipe column: nearest recipe AT the nearest lower scale only
    nn = _naive_carry_forward(proc, y, scale, np.array([[7.19]]), 100.0)
    assert nn[0] == 4.0


def test_report_carries_the_third_baseline_on_the_synthetic_scaleup():
    df = pd.read_csv(CSV)
    rep = leave_one_scale_out_report(df, "titer_g_per_L", ["ph_setpoint", "temperature_C"], include_oof=True)
    smallest = min(r["scale_L"] for r in rep["per_scale"])
    for row in rep["per_scale"]:
        if row["scale_L"] == smallest:
            assert row["naive_cf_n"] == 0 and row["beats_naive_cf"] is True
        else:
            assert row["naive_cf_n"] == row["n"] and np.isfinite(row["naive_cf_mae"])
    assert np.isfinite(rep["overall"]["naive_cf_mae"])
    assert len(rep["oof"]["naive_cf_pred"]) == len(rep["oof"]["pred"])


def _rung(beats_cf: bool) -> ro.RungResult:
    return ro.RungResult(
        scale_L=100.0, step_ratio=3.0, n=10, mae=0.1, naive_mean_mae=1.0, naive_nn_mae=1.0,
        beats_both=True, too_few_to_judge=False, naive_cf_mae=0.05 if not beats_cf else 1.0,
        beats_carry_forward=beats_cf,
    )


def _ladder(beats_cf: bool) -> ro.LadderResult:
    return ro.LadderResult(
        rungs=[_rung(beats_cf)], skipped=[], residuals=np.full(20, 0.1), pooled_mae=0.1,
        pooled_naive_mean_mae=1.0, pooled_naive_nn_mae=1.0, pooled_beats_both=True,
        pooled_naive_cf_mae=1.0 if beats_cf else 0.05, pooled_beats_carry_forward=beats_cf,
    )


def test_gate_refuses_when_carry_forward_is_not_beaten():
    ok = ro.decide(_ladder(True), 2.0, {}, {})
    assert ok.decision != ro.REFUSAL
    bad = ro.decide(_ladder(False), 2.0, {}, {})
    assert bad.decision == ro.REFUSAL
    assert any("carry-forward" in r for r in bad.reasons)
