"""F02: every interval carries the basis of its coverage claim, and proposal
intervals carry the measured-at-proposals caveat as structured fields.

Fast unit tests on `kalos.portal.analysis_parts.intervals` and
`coverage_evidence`, plus one integration check through `_analyze`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kalos.portal.analysis_parts.coverage_evidence import (
    BASIS_HELD_OUT,
    BASIS_MODEL,
    CAVEAT_CODE,
    MEASURED,
    coverage_at_proposals_estimate,
    proposal_caveat,
)
from kalos.portal.analysis_parts.intervals import (
    band_from_summary,
    calibrate_normalized,
    conformal_band_caveat,
    interval_method_summary,
    proposal_interval_fields,
)

LEVELS = (0.9, 0.95)


def _cal(n: int = 40, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sd = rng.uniform(0.5, 2.0, n)
    return calibrate_normalized(rng.normal(0, sd), np.zeros(n), sd, LEVELS)


def test_estimate_has_source_and_range():
    est = coverage_at_proposals_estimate("conformal_normalized", 0.9)
    assert est is not None
    assert est["nominal"] == 0.9 and est["basis"] == "simulation"
    assert est["measured_min"] <= est["measured_max"] < 0.9  # measured BELOW nominal before F15
    assert est["source"]["benchmark"].startswith("python -m kalos.bench.coverage_at_proposals")
    assert est["source"]["doc"].startswith("VALIDATION.md")
    assert set(est["by_scenario"]) == set(est["source"]["scenarios"])
    assert coverage_at_proposals_estimate("conformal_normalized", 0.5) is None
    assert coverage_at_proposals_estimate("no_such_method", 0.9) is None


def test_caveat_states_the_measured_range_in_plain_words():
    cav = proposal_caveat("conformal_normalized", 0.9)
    assert cav["code"] == CAVEAT_CODE
    lo = int(round(cav["coverage_at_proposals_estimate"]["measured_min"] * 100))
    hi = int(round(cav["coverage_at_proposals_estimate"]["measured_max"] * 100))
    assert "held-out runs similar to past runs" in cav["message"]
    assert f"{lo}-{hi}%" in cav["message"] and "lower" in cav["message"]
    # an unmeasured method says so instead of inventing a number
    assert "not been measured" in proposal_caveat("no_such_method", 0.9)["message"]


def test_every_proposal_interval_carries_its_basis_and_caveat():
    rows = proposal_interval_fields([4.0, 5.0], [0.2, 0.4], [0.3, 0.5], _cal(), LEVELS, 4.5,
                                    "best_measured_single")
    for r in rows:
        for iv in r["intervals"]:
            assert iv["coverage_basis"] == BASIS_HELD_OUT
            assert iv["caveat"]["code"] == CAVEAT_CODE
            est = iv["caveat"]["coverage_at_proposals_estimate"]
            assert est["nominal"] == iv["coverage"]
        for g in r["gp_intervals"]:
            assert g["coverage_basis"] == BASIS_MODEL
            assert g["coverage_at_proposals_estimate"]["nominal"] == g["coverage"]
        assert r["quantiles"]["coverage_basis"] == BASIS_MODEL
    json.dumps(rows, allow_nan=False)


def test_refused_level_still_carries_basis_and_caveat():
    sd = np.ones(10)
    cal = calibrate_normalized(np.linspace(-1, 1, 10), np.zeros(10), sd, LEVELS)
    (row,) = proposal_interval_fields([1.0], [0.1], [0.2], cal, LEVELS, 0.5, "best_measured_single")
    off = row["intervals"][1]
    assert off["available"] is False
    assert off["coverage_basis"] == BASIS_HELD_OUT and off["caveat"]["code"] == CAVEAT_CODE


def test_method_summary_and_conformal_caveat():
    summary = interval_method_summary(_cal(), "surrogate_fitted_noise")
    assert summary["coverage_basis"] == BASIS_HELD_OUT
    assert set(summary["coverage_at_proposals"]) == {"0.9", "0.95"}
    cc = conformal_band_caveat(0.1)
    assert cc["code"] == "coverage_not_guaranteed" and cc["caveat_code"] == CAVEAT_CODE
    assert cc["coverage_basis"] == BASIS_HELD_OUT
    assert cc["coverage_at_proposals_estimate"]["nominal"] == 0.9
    json.dumps([summary, cc], allow_nan=False)


def test_band_from_summary_matches_the_shipped_proposal_band():
    cal = _cal()
    summary = interval_method_summary(cal, "surrogate_fitted_noise")
    (row,) = proposal_interval_fields([3.0], [0.2], [0.6], cal, LEVELS, 1.0, "best_measured_single")
    for k, lv in enumerate(LEVELS):
        lo, hi = band_from_summary(np.array([3.0]), np.array([0.6]), summary, lv)
        assert lo is not None and hi is not None
        assert abs(lo[0] - row["intervals"][k]["lo"]) < 1e-3 and abs(hi[0] - row["intervals"][k]["hi"]) < 1e-3
    assert band_from_summary(np.array([3.0]), np.array([0.6]), summary, 0.5) == (None, None)


def test_measured_table_is_in_unit_interval():
    for method, levels in MEASURED.items():
        for lv, by in levels.items():
            assert all(0.0 <= v <= 1.0 for v in by.values()), (method, lv)


def test_analysis_labels_every_band():
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(0)
    m, ph = rng.uniform(0, 4, 24), rng.uniform(5, 7, 24)
    df = pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                       "titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, 24)).round(3)})
    r = _analyze(df, target="titer")
    assert r["conformal_caveat"]["coverage_basis"] == BASIS_HELD_OUT
    assert r["interval_method"]["coverage_basis"] == BASIS_HELD_OUT
    for p in r["proposals"]:
        assert all(iv["coverage_basis"] == BASIS_HELD_OUT and iv["caveat"]["code"] == CAVEAT_CODE
                   for iv in p["intervals"])
        assert all(g["coverage_basis"] == BASIS_MODEL for g in p["gp_intervals"])


def test_evidence_table_matches_the_committed_benchmark_results():
    """The numbers a client shows are the numbers the benchmark wrote."""
    from pathlib import Path

    from kalos.portal.analysis_parts.coverage_evidence import SOURCE

    root = Path(__file__).resolve().parents[1]
    before = json.loads((root / "benchmarks/results/coverage_at_proposals_before.json").read_text())["summary"]
    for lv in ("0.9", "0.95"):
        for sc, v in MEASURED["conformal_normalized"][lv].items():
            assert before[sc]["levels"][lv]["coverage_at_proposals"] == v, (sc, lv)
            assert before[sc]["levels"][lv]["methods"] == ["conformal_normalized"]
        for sc, v in MEASURED["gp_posterior"][lv].items():
            assert before[sc]["levels"][lv]["gp_posterior_coverage_at_proposals"] == v, (sc, lv)
    for sc, v in MEASURED["conformal_abs_global"]["0.9"].items():
        assert before[sc]["levels"]["0.9"]["conformal_q_coverage_at_proposals"] == v, sc
    assert set(SOURCE["scenarios"]) <= set(before)
