"""Tests for kalos.data.synth_metrics (synthetic-data privacy/fidelity measurements)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.data.synth_metrics import (
    c2st,
    column_ks,
    column_tvd,
    correlation_difference,
    dcr,
    exact_duplicate_rate,
    mutual_information_difference,
    nndr,
    synth_report,
)

COV = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, -0.3], [0.0, -0.3, 1.0]])


def sample(n: int, seed: int, shift: float = 0.0, cat_p=(0.5, 0.3, 0.2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(np.full(3, shift), COV, size=n)
    return pd.DataFrame({
        "temp": 30 + 2 * x[:, 0],
        "ph": 7 + 0.2 * x[:, 1],
        "feed": 10 + 3 * x[:, 2],
        "medium": rng.choice(["A", "B", "C"], size=n, p=list(cat_p)),
    })


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    return sample(300, 1)


@pytest.fixture(scope="module")
def holdout() -> pd.DataFrame:
    return sample(300, 2)


def test_identical_copy(real):
    d = dcr(real, real.copy())
    assert d["synthetic"]["median"] == 0.0
    assert d["synthetic"]["mean"] == 0.0
    assert exact_duplicate_rate(real, real.copy())["rate"] == 1.0
    # a copy is indistinguishable: every row appears in both classes
    assert c2st(real, real.copy(), seed=0)["auc"] < 0.6
    assert all(v["statistic"] == 0.0 for v in column_ks(real, real.copy()).values())


def test_fresh_sample_close_in_distribution(real):
    fresh = sample(300, 3)
    ks = column_ks(real, fresh)
    assert max(v["statistic"] for v in ks.values()) < 0.15
    assert 0.35 < c2st(real, fresh, seed=0)["auc"] < 0.65
    assert column_tvd(real, fresh)["medium"] < 0.1
    assert correlation_difference(real, fresh)["pearson"]["mean_abs"] < 0.1
    assert exact_duplicate_rate(real, fresh)["rate"] == 0.0


def test_shifted_distribution_detected(real):
    shifted = sample(300, 4, shift=1.0)
    ks = column_ks(real, shifted)
    assert min(v["statistic"] for v in ks.values()) > 0.25
    assert c2st(real, shifted, seed=0)["auc"] > 0.8
    assert c2st(real, shifted, seed=0, classifier="gbm")["auc"] > 0.75


def test_scale_only_shift_seen_by_logistic(real):
    wide = real.copy()
    rng = np.random.default_rng(9)
    wide["temp"] = 30 + 3.0 * (sample(300, 11)["temp"] - 30)  # same mean, 3x spread
    wide["medium"] = rng.permutation(wide["medium"].to_numpy())
    assert c2st(real, wide, seed=0)["auc"] > 0.7


def test_memorizer_flagged_by_dcr_fresh_sample_not(real, holdout):
    rng = np.random.default_rng(5)
    memo = real.copy()
    for c in ["temp", "ph", "feed"]:
        memo[c] = memo[c] + rng.normal(0, 1e-3 * memo[c].std(), size=len(memo))
    fresh = sample(300, 6)

    d_memo = dcr(real, memo, holdout)
    d_fresh = dcr(real, fresh, holdout)
    assert d_memo["reference_kind"] == "holdout_to_real"
    assert d_memo["share_below_reference_q"] > 0.9
    assert d_fresh["share_below_reference_q"] < 0.15
    assert exact_duplicate_rate(real, memo)["rate"] == 0.0  # jitter evades exact match
    assert nndr(real, memo, holdout)["synthetic"]["median"] < 0.05
    assert nndr(real, fresh, holdout)["synthetic"]["median"] > 0.3

    assert synth_report(real, memo, holdout)["summary"]["near_copy_signal"] is True
    assert synth_report(real, fresh, holdout)["summary"]["near_copy_signal"] is False


def test_dcr_without_holdout_uses_leave_one_out(real):
    d = dcr(real, sample(300, 7))
    assert d["reference_kind"] == "real_to_real_leave_one_out"
    assert d["reference"]["min"] > 0


def test_categorical_tvd(real):
    other = sample(300, 8, cat_p=(0.2, 0.3, 0.5))
    tvd = column_tvd(real, other)["medium"]
    assert 0.2 < tvd < 0.4  # expected 0.3
    disjoint = real.copy()
    disjoint["medium"] = "Z"
    assert column_tvd(real, disjoint)["medium"] == pytest.approx(1.0)
    hand_r = pd.DataFrame({"x": ["a", "a", "b", "b"]})
    hand_s = pd.DataFrame({"x": ["a", "b", "b", "b"]})
    assert column_tvd(hand_r, hand_s)["x"] == pytest.approx(0.25)


def test_correlation_and_mi_detect_broken_dependence(real):
    broken = real.copy()
    rng = np.random.default_rng(3)
    for c in broken.columns:
        broken[c] = rng.permutation(broken[c].to_numpy())  # same marginals, no structure
    assert correlation_difference(real, broken)["pearson"]["max_abs"] > 0.4
    assert max(v["statistic"] for v in column_ks(real, broken).values()) == 0.0
    mi = mutual_information_difference(real, broken)
    assert mi["max_abs"] > mutual_information_difference(real, real.copy())["max_abs"]


def test_column_mismatch_raises(real):
    with pytest.raises(ValueError, match="column mismatch"):
        synth_report(real, real.drop(columns=["ph"]))
    with pytest.raises(ValueError, match="column mismatch"):
        dcr(real, real.assign(extra=1.0))


def test_nan_rows_dropped_and_counted(real, holdout):
    syn = sample(300, 12)
    syn.loc[:9, "temp"] = np.nan
    syn.loc[10:14, "medium"] = None
    d = dcr(real, syn, holdout)
    assert d["rows_dropped_nan"]["synthetic"] == 15
    ks = column_ks(real, syn)
    assert ks["temp"]["statistic"] is not None


def test_report_json_serializable_and_deterministic(real, holdout):
    syn = sample(200, 13)
    rep = synth_report(real, syn, holdout, seed=3)
    text = json.dumps(rep, allow_nan=False)
    assert json.loads(text) == rep
    assert rep == synth_report(real, syn, holdout, seed=3)
    assert rep["kind"] == "synthetic_data_measurements"
    lowered = text.lower()
    for word in ("compliant", "gdpr", "validated", "certified"):
        assert word not in lowered
    assert set(rep["summary"]) >= {"c2st_auc", "max_ks", "max_tvd", "exact_duplicate_rate"}
