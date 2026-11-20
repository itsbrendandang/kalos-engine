"""Measured coverage of the engine's bands AT THE PROPOSALS (simulation evidence).

Every interval Kalos shows carries the basis of its coverage claim
(`coverage_basis`). The per-proposal bands are calibrated on held-out past runs;
proposals are selected for high predicted values, and that selection lowers the
coverage at the recipes the user actually runs. How much lower was MEASURED by
`python -m kalos.bench.coverage_at_proposals` and is recorded here, so a client
can show the measured number next to the nominal one instead of saying the
band "is calibrated".

The table is evidence, not a guarantee: synthetic simulators (Hartmann-6, the
bioprocess simulator behind examples/synthetic_bioprocess, a heteroscedastic
bump), 40 seeds x 5 proposals per scenario, one fresh measurement per proposal.
Branin is a control and is excluded from the range. Results file and protocol:
benchmarks/results/coverage_at_proposals*.json, VALIDATION.md
"Interval coverage at proposals". `tests/test_interval_labels.py` keeps this
table equal to the committed results file.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# Bases of a coverage claim.
BASIS_HELD_OUT = "held_out_past_runs"   # grouped-CV out-of-fold runs (exchangeable with past runs)
BASIS_SELECTION_AWARE = "held_out_top_ranked_runs"  # OOF runs the fold model ranked highest (F15)
BASIS_MODEL = "model_based"            # the GP's own Gaussian predictive; not held-out calibrated

CAVEAT_CODE = "coverage_lower_at_proposals"
# The selection-aware band (F15) when its measured coverage at proposals reached
# the nominal level in every scenario.
CAVEAT_CODE_MEASURED = "coverage_measured_at_proposals"
SOURCE = {
    "kind": "simulation",
    "benchmark": "python -m kalos.bench.coverage_at_proposals",
    "results": "benchmarks/results/coverage_at_proposals_before.json",
    "doc": "VALIDATION.md#interval-coverage-at-proposals",
    "scenarios": ["hartmann6_n30", "hartmann6_n80", "bioprocess_n40", "bioprocess_n100",
                  "heteroscedastic_n40"],
    "seeds_per_scenario": 40,
    "proposals_per_seed": 5,
    "measurement": "one fresh simulated measurement per proposal",
}

# Per-method differences from SOURCE (which results file, which seeds).
SOURCE_OVERRIDES: dict[str, dict[str, Any]] = {
    "conformal_selection_aware": {
        "results": "benchmarks/results/coverage_at_proposals_after.json",
        "seeds": "100-139 (fresh; the method's parameters were chosen on seeds 0-39)",
    },
}

# method -> level -> scenario -> measured coverage at the proposals.
# conformal_normalized and the GP / conformal_q rows: engine @ 0fc05c3 (before
# F15), seeds 0-39.
MEASURED: dict[str, dict[str, dict[str, float]]] = {
    "conformal_normalized": {
        "0.9": {"hartmann6_n30": 0.60, "hartmann6_n80": 0.65, "bioprocess_n40": 0.74,
                "bioprocess_n100": 0.81, "heteroscedastic_n40": 0.685},
        "0.95": {"hartmann6_n30": 0.83, "hartmann6_n80": 0.755, "bioprocess_n40": 0.825,
                 "bioprocess_n100": 0.855, "heteroscedastic_n40": 0.805},
    },
    # F15 selection-aware band: engine @ feat/trust-calibration, fresh seeds 100-139.
    "conformal_selection_aware": {
        "0.9": {"hartmann6_n30": 0.935, "hartmann6_n80": 0.94, "bioprocess_n40": 0.92,
                "bioprocess_n100": 0.99, "heteroscedastic_n40": 0.96},
        "0.95": {"hartmann6_n30": 0.97, "hartmann6_n80": 0.97, "bioprocess_n40": 0.965,
                 "bioprocess_n100": 0.995, "heteroscedastic_n40": 0.995},
    },
    "conformal_abs_global": {  # the global +/- conformal_q band (90% only)
        "0.9": {"hartmann6_n30": 0.68, "hartmann6_n80": 0.765, "bioprocess_n40": 0.79,
                "bioprocess_n100": 0.83, "heteroscedastic_n40": 0.765},
    },
    "gp_posterior": {
        "0.9": {"hartmann6_n30": 0.50, "hartmann6_n80": 0.57, "bioprocess_n40": 0.635,
                "bioprocess_n100": 0.725, "heteroscedastic_n40": 0.53},
        "0.95": {"hartmann6_n30": 0.525, "hartmann6_n80": 0.61, "bioprocess_n40": 0.655,
                 "bioprocess_n100": 0.775, "heteroscedastic_n40": 0.585},
    },
}


def _key(level: float) -> str:
    return f"{float(level):g}"


def _source(method: str) -> dict[str, Any]:
    return {**SOURCE, **SOURCE_OVERRIDES.get(method, {})}


def coverage_at_proposals_estimate(method: str, level: float) -> dict[str, Any] | None:
    """`{nominal, measured_min, measured_max, by_scenario, basis, source}` for
    `method` at `level`, or None when that combination was never measured."""
    table: Mapping[str, Mapping[str, float]] = MEASURED.get(method, {})
    by = table.get(_key(level))
    if not by:
        return None
    vals = list(by.values())
    return {
        "nominal": round(float(level), 4),
        "measured_min": round(min(vals), 3),
        "measured_max": round(max(vals), 3),
        "by_scenario": dict(by),
        "basis": "simulation",
        "source": _source(method),
    }


def compact(estimate: Mapping[str, Any] | None, where: str) -> dict[str, Any] | None:
    """The per-proposal form of an estimate: the numbers, plus a pointer to the
    one place in the response that carries the full source (`where`)."""
    if estimate is None:
        return None
    return {k: estimate[k] for k in ("nominal", "measured_min", "measured_max", "basis")} | {
        "source": where, "benchmark": SOURCE["benchmark"],
    }


def proposal_caveat(method: str, level: float) -> dict[str, Any]:
    """The structured caveat a client shows next to a proposal's interval:
    `{code, message, coverage_at_proposals_estimate}`. Plain words, no claim
    beyond the measured numbers."""
    est = coverage_at_proposals_estimate(method, level)
    pct = int(round(float(level) * 100))
    # Whole percent, rounded DOWN, so a measured 0.995 never reads as 100%.
    rng = None if est is None else (
        f"{math.floor(est['measured_min'] * 100 + 1e-9)}-{math.floor(est['measured_max'] * 100 + 1e-9)}%")
    reached = est is not None and est["measured_min"] >= float(level)
    # Wording rule (tests/test_interval_labels.py): say what the band was
    # calibrated ON and what coverage was MEASURED at proposed recipes; never
    # that a proposal's range "is calibrated".
    if method == "conformal_selection_aware":
        lead = (f"Based on held-out runs: this {pct}% range was sized on the held-out past runs the "
                "model ranked highest, to mimic how proposals are chosen. ")
        if est is None:
            tail = "Its coverage at proposed recipes has not been measured."
        elif reached:
            tail = (f"At proposed recipes this {pct}% range covered {rng} in simulation "
                    "(measured, not guaranteed).")
        else:
            tail = (f"At proposed recipes this {pct}% range covered {rng} in simulation, "
                    f"below {pct}% in some cases (measured, not guaranteed).")
        code = CAVEAT_CODE_MEASURED if reached else CAVEAT_CODE
    else:
        lead = ("Based on held-out runs similar to past runs. Proposals are chosen for high predicted "
                "values, so coverage at proposed recipes is lower: ")
        if est is None:
            tail = "it has not been measured."
        elif reached:
            tail = f"in simulation this {pct}% range covered {rng}."
        else:
            tail = f"in simulation this {pct}% range covered only {rng}."
        code = CAVEAT_CODE
    return {"code": code, "message": lead + tail, "coverage_at_proposals_estimate": est}


__all__ = [
    "BASIS_HELD_OUT", "BASIS_MODEL", "BASIS_SELECTION_AWARE", "CAVEAT_CODE", "CAVEAT_CODE_MEASURED", "MEASURED",
    "SOURCE", "SOURCE_OVERRIDES",
    "compact", "coverage_at_proposals_estimate", "proposal_caveat",
]
