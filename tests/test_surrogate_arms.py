"""Alternative-surrogate harness: reports only, the GP stays the default."""
import importlib.util

import numpy as np
import pytest
from botorch.models import SingleTaskGP

from kalos.bench import surrogate_arms as sa
from kalos.bench.objectives import gaussian_bump
from kalos.core.surrogate import Surrogate


def _problem(n=40, seed=0):
    obj = gaussian_bump(2)
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))
    _, y = obj.evaluate(X, noise=0.1, rng=rng)
    return X, y, obj.bounds


@pytest.fixture(scope="module")
def two_seed_result():
    X, y, b = _problem()
    return sa.run_surrogate_bench(X, y, arms=("gp", "sk_gbr_quantile"), seeds=range(2), bounds=b)


def test_gp_and_sk_arm_metrics_finite(two_seed_result):
    for arm in ("gp", "sk_gbr_quantile"):
        a = two_seed_result["arms"][arm]
        assert a["status"] == "ok" and a["n_seeds"] == 2
        for m in ("rmse", "spearman", "picp", "width"):
            assert np.isfinite(a[m]), (arm, m)
        assert 0.0 <= a["picp"] <= 1.0
        assert a["width"] > 0
        for row in two_seed_result["per_seed"][arm]:
            assert 0.0 <= row["picp"] <= 1.0


def test_every_arm_sees_identical_folds(two_seed_result):
    log = two_seed_result["fold_log"]
    assert log["gp"] == log["sk_gbr_quantile"]
    assert len(log["gp"]) == 2 * 5
    for folds in two_seed_result["folds"].values():
        test = np.concatenate([te for _, _, te in folds])
        assert sorted(test.tolist()) == list(range(40))  # test folds partition the rows
        for fit_i, cal_i, te in folds:
            assert not (set(fit_i) & set(cal_i) or set(fit_i) & set(te) or set(cal_i) & set(te))


def test_fewer_than_five_seeds_gives_no_verdict(two_seed_result):
    a = two_seed_result["arms"]["sk_gbr_quantile"]
    assert a["beats_gp"] is None and a["verdict"] == "insufficient_seeds"
    assert "beats_gp" not in two_seed_result["arms"]["gp"]


def test_verdict_rule_needs_rmse_and_coverage():
    gp = [{"rmse": 1.0}] * 5
    agg_ok = {"rmse": 0.5, "picp": 0.9}
    rows = [{"rmse": 0.5}] * 5
    assert sa._verdict(rows, gp, 1.0, agg_ok, 0.9, 0.05)["beats_gp"] is True
    assert sa._verdict(rows, gp, 1.0, {"rmse": 0.5, "picp": 0.7}, 0.9, 0.05)["beats_gp"] is False
    worse = [{"rmse": 1.5}] * 5
    v = sa._verdict(worse, gp, 1.0, {"rmse": 1.5, "picp": 0.9}, 0.9, 0.05)
    assert v["beats_gp"] is False and v["verdict"] == "does_not_beat_gp"
    assert sa._verdict(rows[:4], gp[:4], 1.0, agg_ok, 0.9, 0.05)["verdict"] == "insufficient_seeds"


def test_verdict_is_paired_a_lower_mean_alone_is_not_a_win():
    gp = [{"rmse": 1.0}] * 5
    # lower mean RMSE (0.88) but only 2/5 seed wins -> not distinguishable
    mixed = [{"rmse": r} for r in (0.5, 0.6, 1.1, 1.1, 1.1)]
    v = sa._verdict(mixed, gp, 1.0, {"rmse": 0.88, "picp": 0.9}, 0.9, 0.05)
    assert v["beats_gp"] is False and v["verdict"] == "not_distinguishable"
    assert v["rmse_wins_vs_gp"] == "2/5"
    # tie on every seed -> not a win either
    tie = [{"rmse": 1.0}] * 5
    assert sa._verdict(tie, gp, 1.0, {"rmse": 1.0, "picp": 0.9}, 0.9, 0.05)["beats_gp"] is False


def test_grouped_folds_keep_groups_together():
    groups = np.repeat(np.arange(10), 4)
    for fit_i, cal_i, te in sa.make_folds(40, 0, groups=groups):
        assert not set(groups[te]) & (set(groups[fit_i]) | set(groups[cal_i]))
        assert not set(groups[fit_i]) & set(groups[cal_i])


def test_conformal_quantile_arm_widens_to_cover():
    rng = np.random.default_rng(0)
    cal = sa.ArmPrediction(mean=np.zeros(200), lo=np.full(200, -0.1), hi=np.full(200, 0.1))
    y_cal = rng.normal(size=200)
    lo, hi = sa.conformalize(cal, y_cal, cal, alpha=0.1)
    y_new = rng.normal(size=200)
    assert 0.8 < np.mean((y_new >= lo) & (y_new <= hi)) < 0.98  # raw band covered ~8%


@pytest.mark.parametrize("arm,pkg", [("lgbm_quantile", "lightgbm"), ("catboost_quantile", "catboost"),
                                     ("ngboost", "ngboost")])
def test_optional_arm_skipped_when_missing(arm, pkg, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == pkg else real(name, *a))
    assert sa.arm_availability([arm])[arm] == "skipped: not installed"
    X, y, b = _problem(n=20)
    res = sa.run_surrogate_bench(X, y, arms=(arm,), seeds=range(1), bounds=b)
    assert res["arms"][arm] == {"status": "skipped: not installed"}


@pytest.mark.parametrize("arm,pkg", [("lgbm_quantile", "lightgbm"), ("catboost_quantile", "catboost"),
                                     ("ngboost", "ngboost")])
def test_optional_arm_runs_when_installed(arm, pkg):
    pytest.importorskip(pkg)
    X, y, b = _problem(n=30)
    a = sa.run_surrogate_bench(X, y, arms=(arm,), seeds=range(1), bounds=b, n_splits=3)["arms"][arm]
    assert a["status"] == "ok" and np.isfinite(a["rmse"]) and 0 <= a["picp"] <= 1


def test_harness_does_not_change_default_surrogate(two_seed_result):
    X, y, b = _problem(n=8)
    assert type(Surrogate().fit(X, y, bounds=b).model) is SingleTaskGP


def test_amii_driver_skips_without_data(monkeypatch, tmp_path):
    monkeypatch.delenv("KALOS_AMII_DATA", raising=False)
    monkeypatch.delenv("KALOS_MEDIA_DATA", raising=False)
    res = sa.run_amii(arms=("gp",), seeds=range(1))
    assert "_skipped" in res
    assert sa.format_table(res).startswith("amii skipped")


def test_main_prints_table(capsys, monkeypatch):
    assert sa.main(["--arms", "gp,sk_gbr_quantile,ngboost", "--seeds", "1", "--n", "25"]) == 0
    out = capsys.readouterr().out
    assert "remains the default" in out and "gaussian_bump_4d" in out and "ackley_4d" in out


def test_table_prints_the_paired_ci_and_seed_count():
    res = {"ds": {"arms": {
        "gp": {"status": "ok", **{m: 1.0 for m in ("rmse", "spearman", "picp", "width")},
               **{f"{m}_sd": 0.1 for m in ("rmse", "spearman", "picp", "width")}},
        "sk_gbr_quantile": {"status": "ok", **{m: 0.9 for m in ("rmse", "spearman", "picp", "width")},
                            **{f"{m}_sd": 0.1 for m in ("rmse", "spearman", "picp", "width")},
                            "verdict": "beats_gp", "rmse_wins_vs_gp": "5/5",
                            "paired_rmse_gain_ci95": [0.05, 0.15]},
    }}}
    out = sa.format_table(res)
    assert "gain CI95 [+0.050, +0.150]" in out and "rmse wins 5/5" in out


def test_main_states_the_seed_count(capsys):
    assert sa.main(["--arms", "gp", "--seeds", "1", "--n", "25"]) == 0
    assert "n=1 seeds" in capsys.readouterr().out
