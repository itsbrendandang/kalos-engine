"""Per-proposal intervals (VIZ_SPEC R4): normalized split conformal on the
grouped-CV out-of-fold set, GP-posterior bands/quantiles, analytic EI and
P(improve). Synthetic data only."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kalos.core.conformal import (
    finite_sample_rank,
    min_n_for_alpha,
    normalized_conformal_quantile,
    normalized_scores,
)
from kalos.portal.analysis_parts.intervals import (
    SIGMA_ASSAY_FLOOR,
    SIGMA_FITTED_NOISE,
    calibrate_normalized,
    expected_improvement,
    interval_method_summary,
    probability_of_improvement,
    proposal_interval_fields,
    proposal_predictive_std,
)

LEVELS = (0.9, 0.95)


# --- finite-sample quantile -------------------------------------------------

def test_finite_sample_rank_and_min_n():
    assert finite_sample_rank(9, 0.1) == 9
    assert finite_sample_rank(8, 0.1) == 9  # > n -> refused
    assert finite_sample_rank(19, 0.05) == 19
    assert finite_sample_rank(24, 0.05) == 24  # ceil(23.75)
    assert finite_sample_rank(100, 0.1) == 91
    assert min_n_for_alpha(0.1) == 9
    assert min_n_for_alpha(0.05) == 19


def test_quantile_is_the_corrected_order_statistic_not_interpolated():
    scores = np.arange(1.0, 21.0)  # 1..20
    out = normalized_conformal_quantile(scores[::-1], 0.1)  # order must not matter
    # k = ceil(21 * 0.9) = 19 -> the 19th smallest score
    assert out == {"available": True, "q": 19.0, "rank": 19, "n": 20}


def test_quantile_refuses_when_rank_exceeds_n():
    out = normalized_conformal_quantile(np.ones(18), 0.05)
    assert out["available"] is False and out["q"] is None
    assert out["code"] == "too_few_calibration_points"
    assert out["rank"] == 19 and out["min_n"] == 19
    assert normalized_conformal_quantile(np.ones(19), 0.05)["available"] is True
    assert normalized_conformal_quantile([], 0.1)["available"] is False


def test_scores_refuse_degenerate_sigma_instead_of_dropping():
    a, m = np.ones(12), np.zeros(12)
    s = np.ones(12)
    assert normalized_scores(a, m, s)["ok"] is True
    s0 = s.copy()
    s0[3] = 0.0
    assert normalized_scores(a, m, s0)["code"] == "degenerate_sigma"
    snan = s.copy()
    snan[0] = np.nan
    assert normalized_scores(a, m, snan)["code"] == "degenerate_sigma"
    stiny = s.copy()
    stiny[0] = 1e-9
    assert normalized_scores(a, m, stiny)["code"] == "degenerate_sigma"
    assert normalized_scores([], [], [])["code"] == "no_calibration_points"


def test_calibrate_refuses_per_level():
    rng = np.random.default_rng(1)
    n = 12  # enough for 0.90 (needs 9), not for 0.95 (needs 19)
    sd = rng.uniform(0.5, 2.0, n)
    y = rng.normal(0, sd)
    cal = calibrate_normalized(y, np.zeros(n), sd, LEVELS)
    l90, l95 = cal["levels"]
    assert l90["available"] is True and l90["n_calibration"] == n and l90["rank"] == 12
    assert l95["available"] is False and l95["code"] == "too_few_calibration_points"
    assert "what_would_fix_it" in l95 and l95["min_n"] == 19
    # in-sample count is exact arithmetic on the scores
    scores = np.abs(y) / sd
    assert l90["n_inside"] == int((scores <= l90["_q"]).sum())
    assert l90["empirical_oof_coverage"] == round(l90["n_inside"] / n, 4)
    halves = l90["coverage_by_sigma_half"]
    assert halves["low_sigma"]["n"] + halves["high_sigma"]["n"] == n

    bad = calibrate_normalized(y, np.zeros(n), np.zeros(n), LEVELS)
    assert all(lv["available"] is False and lv["code"] == "degenerate_sigma" for lv in bad["levels"])


# --- coverage simulation -----------------------------------------------------

def _hetero(rng, n):
    x = rng.uniform(0, 1, n)
    sigma = 0.1 + 2.0 * x  # strongly heteroscedastic
    return x, sigma, np.sin(4 * x) + sigma * rng.normal(size=n)


@pytest.mark.parametrize("coverage", LEVELS)
def test_normalized_band_hits_nominal_marginal_coverage(coverage):
    """Exchangeable calibration/test draws with a deliberately mis-scaled sd
    (x1.7): marginal coverage must still sit at k/(n+1) >= nominal on average."""
    rng = np.random.default_rng(7)
    n_cal, n_test, trials = 40, 300, 300
    covs = []
    for _ in range(trials):
        x, sig, y = _hetero(rng, n_cal)
        cal = calibrate_normalized(y, np.sin(4 * x), 1.7 * sig, (coverage,))
        q = cal["levels"][0]["_q"]
        xt, sigt, yt = _hetero(rng, n_test)
        covs.append(np.mean(np.abs(yt - np.sin(4 * xt)) <= q * 1.7 * sigt))
    k = finite_sample_rank(n_cal, 1 - coverage)
    expected = k / (n_cal + 1)
    mean_cov = float(np.mean(covs))
    assert expected >= coverage
    assert abs(mean_cov - expected) < 0.01, (mean_cov, expected)


def test_normalized_band_adapts_where_a_global_band_does_not():
    """Conditional coverage in the high-noise half: the normalized band holds
    near nominal; a single global half-width under-covers there."""
    rng = np.random.default_rng(3)
    x, sig, y = _hetero(rng, 2000)
    mu = np.sin(4 * x)
    q = calibrate_normalized(y, mu, sig, (0.9,))["levels"][0]["_q"]
    res = np.abs(y - mu)
    q_glob = np.sort(res)[finite_sample_rank(len(res), 0.1) - 1]
    xt, sigt, yt = _hetero(rng, 20000)
    hi = xt > 0.5
    rt = np.abs(yt - np.sin(4 * xt))
    assert abs(np.mean(rt[hi] <= q * sigt[hi]) - 0.9) < 0.02
    assert np.mean(rt[hi] <= q_glob) < 0.85


# --- EI / P(improve) ---------------------------------------------------------

def test_ei_and_p_improve_match_closed_form():
    mu, sd, inc = np.array([1.0, 2.5, 0.2]), np.array([0.5, 1.2, 0.3]), 1.4
    z = (mu - inc) / sd
    ei_ref = (mu - inc) * norm.cdf(z) + sd * norm.pdf(z)
    np.testing.assert_allclose(expected_improvement(mu, sd, inc), ei_ref, rtol=1e-12)
    np.testing.assert_allclose(probability_of_improvement(mu, sd, inc), norm.cdf(z), rtol=1e-12)
    # minimize = maximize on the negated problem
    np.testing.assert_allclose(
        expected_improvement(mu, sd, inc, maximize=False), expected_improvement(-mu, sd, -inc), rtol=1e-12)
    np.testing.assert_allclose(
        probability_of_improvement(mu, sd, inc, maximize=False), 1 - norm.cdf(z), rtol=1e-12)


def test_ei_matches_monte_carlo_and_degenerate_sd():
    rng = np.random.default_rng(0)
    draws = 3.0 + 0.8 * rng.normal(size=400_000)
    mc = np.maximum(draws - 3.5, 0).mean()
    assert abs(expected_improvement([3.0], [0.8], 3.5)[0] - mc) < 3e-3
    np.testing.assert_allclose(expected_improvement([2.0, 1.0], [0.0, 0.0], 1.5), [0.5, 0.0])
    np.testing.assert_allclose(probability_of_improvement([2.0, 1.0], [0.0, 0.0], 1.5), [1.0, 0.0])


# --- per-proposal fields -----------------------------------------------------

def test_proposal_fields_shape_and_arithmetic():
    rng = np.random.default_rng(2)
    sd = rng.uniform(0.5, 2.0, 30)
    cal = calibrate_normalized(rng.normal(0, sd), np.zeros(30), sd, LEVELS)
    rows = proposal_interval_fields(
        [4.0, 5.0], [0.2, 0.4], [0.3, 0.5], cal, LEVELS, 4.5, "best_measured_single")
    r = rows[1]
    q90 = cal["levels"][0]["_q"]
    iv = r["intervals"][0]
    assert iv["method"] == "conformal_normalized" and iv["calibrated_on"] == "grouped_cv_oof"
    assert iv["available"] is True and iv["coverage"] == 0.9 and iv["n_calibration"] == 30
    assert iv["lo"] == round(5.0 - q90 * 0.5, 4) and iv["hi"] == round(5.0 + q90 * 0.5, 4)
    gp = r["gp_intervals"][1]
    assert gp["method"] == "gp_posterior" and gp["posterior"] == "observation"
    assert gp["hi"] == round(5.0 + norm.ppf(0.975) * 0.5, 4)
    assert r["quantiles"]["p50"] == 5.0 and r["quantiles"]["p05"] == round(5.0 + norm.ppf(0.05) * 0.5, 4)
    ei = r["expected_improvement"]
    assert ei["basis"] == "model" and ei["posterior"] == "latent" and ei["incumbent"] == 4.5
    assert math.isclose(ei["value"], float(expected_improvement([5.0], [0.4], 4.5)[0]), rel_tol=1e-3)
    assert math.isclose(r["p_improve"]["value"], norm.cdf(0.5 / 0.4), rel_tol=1e-3)
    # The "chance to beat" is on the LATENT mean (sd 0.4), not one new measurement
    # (predictive sd 0.5, which would give the lower norm.cdf(0.5 / 0.5)), and says so.
    assert not math.isclose(r["p_improve"]["value"], norm.cdf(0.5 / 0.5), rel_tol=1e-3)
    assert "latent mean" in r["p_improve"]["description"]
    assert "not the probability that one new measurement" in r["p_improve"]["description"]
    assert "latent mean" in ei["description"]
    json.dumps(rows, allow_nan=False)
    summary = interval_method_summary(cal, SIGMA_FITTED_NOISE)
    assert all("_q" not in lv for lv in summary["levels"])
    json.dumps(summary, allow_nan=False)


def test_refused_level_carries_reason_on_every_proposal():
    sd = np.ones(10)
    cal = calibrate_normalized(np.linspace(-1, 1, 10), np.zeros(10), sd, LEVELS)
    (row,) = proposal_interval_fields([1.0], [0.1], [0.2], cal, LEVELS, 0.5, "best_measured_single")
    assert row["intervals"][0]["available"] is True
    off = row["intervals"][1]
    assert off["available"] is False and off["lo"] is None and off["code"] == "too_few_calibration_points"
    assert off["reason"] and off["what_would_fix_it"]
    # model-based bands are always present
    assert [g["available"] for g in row["gp_intervals"]] == [True, True]


def test_predictive_std_sources():
    class Fake:
        def posterior(self, X, *, observation_noise=False):
            assert observation_noise is True
            return np.zeros(len(X)), np.full(len(X), 0.7)

    sd, src = proposal_predictive_std(Fake(), np.zeros((2, 1)), [0.3, 0.4])
    assert src == SIGMA_FITTED_NOISE and sd.tolist() == [0.7, 0.7]
    sd, src = proposal_predictive_std(None, np.zeros((2, 1)), [0.3, 0.4], assay_noise_var=0.16)
    assert src == SIGMA_ASSAY_FLOOR
    np.testing.assert_allclose(sd, [0.5, math.sqrt(0.16 + 0.16)])


# --- integration through the real analysis path -----------------------------

def _sheet(n: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    m, ph = rng.uniform(0, 4, n), rng.uniform(5, 7, n)
    df = pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })
    return df.to_csv(index=False).encode()


def test_analysis_proposals_carry_interval_fields():
    from kalos.portal.pipeline import compute_upload_analysis

    _, r = compute_upload_analysis(_sheet(24), "lipase_titer", False, "iv.csv")
    json.dumps(r, allow_nan=False)
    # legacy band unchanged
    assert r["conformal_coverage"] == 0.9 and isinstance(r["conformal_q"], float)
    assert r["interval_levels"] == [0.9, 0.95]
    im = r["interval_method"]
    assert im["method"] == "conformal_normalized" and im["calibrated_on"] == "grouped_cv_oof"
    assert im["sigma_source"] == SIGMA_FITTED_NOISE
    assert im["n_calibration"] == len(r["oof"])
    assert [lv["coverage"] for lv in im["levels"]] == [0.9, 0.95]
    assert all(lv["available"] for lv in im["levels"])
    assert r["proposals"]
    for p in r["proposals"]:
        assert {"intervals", "gp_intervals", "quantiles", "expected_improvement", "p_improve",
                "predictive_std"} <= set(p)
        assert p["predictive_std"] >= p["std"]  # observation >= latent
        for iv in p["intervals"]:
            assert iv["lo"] <= p["pred"] <= iv["hi"]
        assert p["intervals"][0]["hi"] - p["intervals"][0]["lo"] <= p["intervals"][1]["hi"] - p["intervals"][1]["lo"]
        assert p["expected_improvement"]["incumbent"] == r["best"]
        assert p["expected_improvement"]["incumbent_basis"] == "best_measured_single"
        assert 0.0 <= p["p_improve"]["value"] <= 1.0


def test_analysis_small_sheet_refuses_95_but_not_90():
    from kalos.portal.pipeline import compute_upload_analysis

    _, r = compute_upload_analysis(_sheet(14, seed=5), "lipase_titer", False, "iv.csv")
    n = r["interval_method"]["n_calibration"]
    assert 9 <= n < 19
    l90, l95 = r["interval_method"]["levels"]
    assert l90["available"] is True
    assert l95["available"] is False and l95["code"] == "too_few_calibration_points"
    for p in r["proposals"]:
        assert p["intervals"][0]["available"] is True
        assert p["intervals"][1]["available"] is False and p["intervals"][1]["lo"] is None
        assert len(p["gp_intervals"]) == 2


def test_replicate_aware_path_uses_single_measurement_noise():
    from kalos.portal.pipeline import compute_upload_analysis

    rng = np.random.default_rng(11)
    recipes = rng.uniform([0, 5], [4, 7], (9, 2)).round(2)
    rows = []
    for m, ph in recipes:
        for _ in range(3):
            rows.append({"Methanol": m, "pH": ph,
                         "lipase_titer": round(1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3), 3)})
    raw = pd.DataFrame(rows).to_csv(index=False).encode()
    _, r = compute_upload_analysis(raw, "lipase_titer", False, "rep.csv")
    if not r["noise"].get("replicate_aware"):
        pytest.skip("replicate-aware surrogate not engaged on this sheet")
    assert r["interval_method"]["sigma_source"] == SIGMA_ASSAY_FLOOR
    for p in r["proposals"]:
        assert p["expected_improvement"]["incumbent_basis"] == "best_reproducible_mean"
        assert p["predictive_std"] > p["std"]
