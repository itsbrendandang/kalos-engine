"""Simulation regression tests for the trust layer (slow).

The full benchmarks (`python -m kalos.bench.trust_leaks`,
`python -m kalos.bench.coverage_at_proposals`) run 20-40 seeds per scenario and
their results are committed under benchmarks/results/. These tests re-run a few
seeds of the same generators and fail if the measured behaviour regresses past a
loose bound, so a change that re-opens a leak or breaks the interval calibration
is caught without a 30-minute run.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

SEEDS = range(6)


@pytest.mark.parametrize("scenario", ["batch_text", "campaign_rounds", "jitter_0.2pct", "noise_n20"])
def test_false_trust_stays_low_on_no_signal_sheets(scenario):
    """F04 + F05: before, batch/jitter sheets with no input signal were trusted
    95-100% of the time. Allow at most 1 of 6 seeds."""
    from kalos.bench.trust_leaks import run_one

    rows = [run_one(scenario, s) for s in SEEDS]
    trusted = [r for r in rows if r.get("clears_floor") or r.get("verdict") == "reliable"]
    assert len(trusted) <= 1, rows
    assert not any(r.get("batch_in_recipe") for r in rows)
    for r in rows:
        if r.get("n_proposals"):
            assert r["trust_status"] in {"exploration_untrusted", "trusted"}


@pytest.mark.parametrize("scenario", ["hartmann6_n30", "heteroscedastic_n40"])
def test_selection_aware_bands_cover_at_the_proposals(scenario):
    """F15: on fresh seeds 100-139 the 90% band at proposals covered 0.86-0.965
    on Python 3.11 (numpy 2.4.6, scipy 1.17.1) and 0.92-0.99 on Python >= 3.12
    (numpy 2.5.1, scipy 1.18.0), the two stacks uv.lock pins
    (benchmarks/results/coverage_at_proposals_main_*.json); the plain band at
    the same proposals covered 0.545-0.705 on Python 3.11. On 6 seeds x 5
    proposals require >= 0.80 at proposals (the plain band fails it on these
    simulators) and >= 0.85 on random in-box points, at both levels."""
    from kalos.bench.coverage_at_proposals import run_one

    rows = [run_one(scenario, s) for s in range(200, 206)]
    rows = [r for r in rows if "error" not in r]
    assert len(rows) >= 5
    for lv in ("0.9", "0.95"):
        inside = [b for r in rows for b in r["levels"][lv]["inside"]]
        assert len(inside) >= 20
        assert sum(inside) / len(inside) >= 0.80, (lv, sum(inside), len(inside))
        rand = [r["levels"][lv]["random_coverage"] for r in rows]
        assert min(rand) >= 0.85, (lv, rand)
