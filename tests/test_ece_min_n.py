"""ECE is judged only from 20 runs (compmodel PLAN E5).

Facts measured for this change (seeded simulations, 2,000 reps each, below):
- Interval ECE (`interval_calibration`, a perfectly calibrated Gaussian
  predictor): P(ECE > 0.15) ~9% at n=10, ~1% at n=20. Matches the note.
- Classifier ECE (`_classifier_ece`, 10 bins), which is the ECE the promotion
  gate actually reads (analysis.py feeds `feas["ece"]`): P(ECE > 0.15) for a
  perfectly calibrated classifier is ~87% at n=20 and ~36% at n=50. Raising the
  minimum n to 20 does not make that gate fair; see the strict xfail below.
"""
from __future__ import annotations

import numpy as np
import pytest

from kalos.core.evaluation import CALIBRATION_JUDGE_MIN_N, interval_calibration
from kalos.core.feasibility import _classifier_ece
from kalos.core.gates import NOT_ENOUGH_RUNS_FOR_CALIBRATION, GatesConfig, check_gates

GOOD = {"surrogate_spearman": 0.5, "feasibility_auc": 0.8, "brier": 0.1}


def test_default_min_n_is_20():
    assert GatesConfig().min_ece_n == 20 == CALIBRATION_JUDGE_MIN_N


@pytest.mark.parametrize("ece", [0.01, 0.9, float("nan")])
def test_below_20_runs_the_ece_is_not_judged(ece):
    r = check_gates({**GOOD, "ece": ece, "ece_n": 19})
    assert r.passed  # neither a pass nor a fail on the ECE; the others pass
    assert r.not_judged == [f"ece: {NOT_ENOUGH_RUNS_FOR_CALIBRATION} (n=19, need 20)"]
    assert "ece" not in r.metrics
    assert NOT_ENOUGH_RUNS_FOR_CALIBRATION in r.summary


def test_from_20_runs_the_ece_is_judged():
    assert not check_gates({**GOOD, "ece": 0.9, "ece_n": 20}).passed
    r = check_gates({**GOOD, "ece": 0.05, "ece_n": 20})
    assert r.passed and r.not_judged == [] and r.metrics["ece"] == 0.05


def test_without_ece_n_the_old_behaviour_holds():
    assert not check_gates({**GOOD, "ece": 0.9}).passed
    assert not check_gates({**GOOD, "ece": float("nan")}).passed  # fail closed


def test_interval_calibration_reports_judgement_between_10_and_20():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 15)
    r = interval_calibration(a, np.zeros(15), np.ones(15))
    assert r["available"] and r["ece"] is not None
    assert r["judgeable"] is False and r["judgement"] == NOT_ENOUGH_RUNS_FOR_CALIBRATION
    r20 = interval_calibration(rng.normal(0, 1, 20), np.zeros(20), np.ones(20))
    assert r20["judgeable"] is True and r20["judgement"] is None


def test_calibrated_gaussian_predictor_passes_at_n20_at_least_98pct():
    rng = np.random.default_rng(20)
    reps, n = 2000, 20
    passes = sum(
        interval_calibration(rng.normal(0, 1, n), np.zeros(n), np.ones(n))["ece"] <= 0.15
        for _ in range(reps)
    )
    assert passes / reps >= 0.98


@pytest.mark.xfail(strict=True, reason=(
    "the GATED ECE is the 10-bin classifier ECE; a perfectly calibrated classifier "
    "exceeds max_ece=0.15 in ~87% of samples at n=20 (~36% at n=50). The min-n "
    "change does not fix that; the gate's ECE statistic or threshold needs an owner decision"
))
def test_calibrated_classifier_passes_the_gated_ece_at_n20_at_least_98pct():
    rng = np.random.default_rng(21)
    reps, n = 2000, 20
    passes = 0
    for _ in range(reps):
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        passes += _classifier_ece(y, p) <= GatesConfig().max_ece
    assert passes / reps >= 0.98
