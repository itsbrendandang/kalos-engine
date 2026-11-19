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
