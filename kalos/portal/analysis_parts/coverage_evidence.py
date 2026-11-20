"""Measured coverage of the engine's bands AT THE PROPOSALS (simulation evidence).

Every interval Kalos shows carries the basis of its coverage claim
(`coverage_basis`). The per-proposal bands are sized on held-out past runs;
proposals are selected for high predicted values, and that selection lowers the
coverage at the recipes the user actually runs. How much lower was MEASURED by
`python -m kalos.bench.coverage_at_proposals` and is recorded here, so a client
can show the measured number next to the nominal one instead of saying the
band "is calibrated".

The table is evidence, not a guarantee: synthetic simulators (Hartmann-6, the
bioprocess simulator behind examples/synthetic_bioprocess, a heteroscedastic
bump), 40 seeds x 5 proposals per scenario, one fresh measurement per proposal.
Branin is a control and is excluded from the range.

uv.lock pins TWO numerical stacks: Python < 3.12 gets numpy 2.4.6 and scipy
1.17.1 (CI runs this one), Python >= 3.12 gets numpy 2.5.1 and scipy 1.18.0 (the
engine image, python:3.12, runs this one). The proposals, and so the coverage
measured at them, differ between the two (scipy is the package that moves
them), so the benchmark was run on main on both; each results file records its
commit and environment in `meta.environment`. The served range spans both
stacks: from the lowest coverage measured on either to the highest.

`MEASURED_BY_STACK` and `CI95_BY_STACK` are
`kalos.bench.coverage_at_proposals.evidence_table` of the two results files
(`python -m kalos.bench.coverage_at_proposals --print-evidence FILE` prints
them). tests/test_interval_labels.py fails when they, or the caveat text built
from them, disagree with the files.
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
# The two stacks uv.lock pins, and the results file measured on each.
STACKS: dict[str, dict[str, Any]] = {
    "py311": {
        "label": "Python < 3.12 (numpy 2.4.6, scipy 1.17.1); CI runs this stack",
        "results": "benchmarks/results/coverage_at_proposals_main_py311_seeds100.json",
        # From the results file's meta.environment (the test checks they agree).
        "recorded_on": {
            "git_commit": "84f044ac16fa31cb77b6783763785fc234086771",
            "python": "3.11.17", "numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.13.0",
            "botorch": "0.18.1", "gpytorch": "1.15.2", "torch_threads_per_worker": 1,
            "platform": "macOS-27.2.1-arm64-arm-64bit",
        },
    },
    "py312": {
        "label": "Python >= 3.12 (numpy 2.5.1, scipy 1.18.0); the engine image (python:3.12) runs this stack",
        "results": "benchmarks/results/coverage_at_proposals_main_py312_seeds100.json",
        "recorded_on": {
            "git_commit": "84f044ac16fa31cb77b6783763785fc234086771",
            "python": "3.12.15", "numpy": "2.5.1", "scipy": "1.18.0", "torch": "2.13.0",
            "botorch": "0.18.1", "gpytorch": "1.15.2", "torch_threads_per_worker": 1,
            "platform": "macOS-27.2.1-arm64-arm-64bit",
        },
    },
}
SOURCE = {
    "kind": "simulation",
    "benchmark": "python -m kalos.bench.coverage_at_proposals",
    "command": "python -m kalos.bench.coverage_at_proposals --seeds 40 --seed-offset 100",
    "results": [st["results"] for st in STACKS.values()],
    "doc": "VALIDATION.md#interval-coverage-at-proposals",
    "scenarios": ["hartmann6_n30", "hartmann6_n80", "bioprocess_n40", "bioprocess_n100",
                  "heteroscedastic_n40"],
    "seeds": "100-139 (fresh; the selection-aware band's parameters were chosen on seeds 0-39)",
    "seeds_per_scenario": 40,
    "proposals_per_seed": 5,
    "measurement": "one fresh simulated measurement per proposal",
    "stacks": {k: {"label": v["label"], "results": v["results"], "recorded_on": v["recorded_on"]}
               for k, v in STACKS.items()},
    "range": "from the lowest to the highest coverage measured over the scenarios on both stacks",
}

# stack -> method -> level -> scenario -> measured coverage at the proposals, and
# the seed-cluster bootstrap 95% CI where the benchmark computes one. Generated
# from STACKS[stack]["results"] (`--print-evidence`); do not edit by hand.
# conformal_selection_aware: the shipped band. conformal_normalized: the plain
# band (the fallback when the selection-aware band is unavailable) rebuilt at the
# SAME proposals. conformal_abs_global: the global +/- conformal_q band (90%
# only). gp_posterior: the GP's own Gaussian band.
MEASURED_BY_STACK: dict[str, dict[str, dict[str, dict[str, float]]]] = {
    "py311": {
        "conformal_selection_aware": {
            "0.9": {"hartmann6_n30": 0.91, "hartmann6_n80": 0.94, "bioprocess_n40": 0.86, "bioprocess_n100": 0.95, "heteroscedastic_n40": 0.965},
            "0.95": {"hartmann6_n30": 0.965, "hartmann6_n80": 0.96, "bioprocess_n40": 0.935, "bioprocess_n100": 0.97, "heteroscedastic_n40": 0.99},
        },
        "conformal_normalized": {
            "0.9": {"hartmann6_n30": 0.58, "hartmann6_n80": 0.545, "bioprocess_n40": 0.58, "bioprocess_n100": 0.705, "heteroscedastic_n40": 0.65},
            "0.95": {"hartmann6_n30": 0.76, "hartmann6_n80": 0.68, "bioprocess_n40": 0.645, "bioprocess_n100": 0.83, "heteroscedastic_n40": 0.75},
        },
        "conformal_abs_global": {
            "0.9": {"hartmann6_n30": 0.685, "hartmann6_n80": 0.7, "bioprocess_n40": 0.66, "bioprocess_n100": 0.77, "heteroscedastic_n40": 0.685},
        },
        "gp_posterior": {
            "0.9": {"hartmann6_n30": 0.42, "hartmann6_n80": 0.5, "bioprocess_n40": 0.415, "bioprocess_n100": 0.655, "heteroscedastic_n40": 0.51},
            "0.95": {"hartmann6_n30": 0.47, "hartmann6_n80": 0.55, "bioprocess_n40": 0.45, "bioprocess_n100": 0.705, "heteroscedastic_n40": 0.56},
        },
    },
    "py312": {
        "conformal_selection_aware": {
            "0.9": {"hartmann6_n30": 0.935, "hartmann6_n80": 0.94, "bioprocess_n40": 0.92, "bioprocess_n100": 0.99, "heteroscedastic_n40": 0.96},
            "0.95": {"hartmann6_n30": 0.97, "hartmann6_n80": 0.97, "bioprocess_n40": 0.965, "bioprocess_n100": 0.995, "heteroscedastic_n40": 0.995},
        },
        "conformal_normalized": {
            "0.9": {"hartmann6_n30": 0.64, "hartmann6_n80": 0.595, "bioprocess_n40": 0.66, "bioprocess_n100": 0.835, "heteroscedastic_n40": 0.65},
            "0.95": {"hartmann6_n30": 0.8, "hartmann6_n80": 0.72, "bioprocess_n40": 0.765, "bioprocess_n100": 0.915, "heteroscedastic_n40": 0.765},
        },
        "conformal_abs_global": {
            "0.9": {"hartmann6_n30": 0.735, "hartmann6_n80": 0.71, "bioprocess_n40": 0.77, "bioprocess_n100": 0.87, "heteroscedastic_n40": 0.71},
        },
        "gp_posterior": {
            "0.9": {"hartmann6_n30": 0.485, "hartmann6_n80": 0.56, "bioprocess_n40": 0.53, "bioprocess_n100": 0.795, "heteroscedastic_n40": 0.51},
            "0.95": {"hartmann6_n30": 0.52, "hartmann6_n80": 0.58, "bioprocess_n40": 0.565, "bioprocess_n100": 0.815, "heteroscedastic_n40": 0.565},
        },
    },
}
CI95_BY_STACK: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
    "py311": {
        "conformal_selection_aware": {
            "0.9": {"hartmann6_n30": [0.855, 0.955], "hartmann6_n80": [0.89, 0.98], "bioprocess_n40": [0.79, 0.92], "bioprocess_n100": [0.905, 0.985], "heteroscedastic_n40": [0.94, 0.985]},
            "0.95": {"hartmann6_n30": [0.93, 0.995], "hartmann6_n80": [0.925, 0.99], "bioprocess_n40": [0.875, 0.98], "bioprocess_n100": [0.94, 0.99], "heteroscedastic_n40": [0.975, 1.0]},
        },
        "conformal_normalized": {
            "0.9": {"hartmann6_n30": [0.5, 0.66], "hartmann6_n80": [0.47, 0.625], "bioprocess_n40": [0.495, 0.66], "bioprocess_n100": [0.64, 0.765], "heteroscedastic_n40": [0.58, 0.72]},
            "0.95": {"hartmann6_n30": [0.685, 0.83], "hartmann6_n80": [0.605, 0.76], "bioprocess_n40": [0.56, 0.725], "bioprocess_n100": [0.775, 0.88], "heteroscedastic_n40": [0.675, 0.825]},
        },
    },
    "py312": {
        "conformal_selection_aware": {
            "0.9": {"hartmann6_n30": [0.895, 0.97], "hartmann6_n80": [0.89, 0.975], "bioprocess_n40": [0.87, 0.965], "bioprocess_n100": [0.975, 1.0], "heteroscedastic_n40": [0.93, 0.985]},
            "0.95": {"hartmann6_n30": [0.95, 0.99], "hartmann6_n80": [0.94, 0.995], "bioprocess_n40": [0.92, 1.0], "bioprocess_n100": [0.985, 1.0], "heteroscedastic_n40": [0.985, 1.0]},
        },
        "conformal_normalized": {
            "0.9": {"hartmann6_n30": [0.555, 0.72], "hartmann6_n80": [0.52, 0.67], "bioprocess_n40": [0.59, 0.725], "bioprocess_n100": [0.795, 0.875], "heteroscedastic_n40": [0.585, 0.715]},
            "0.95": {"hartmann6_n30": [0.73, 0.865], "hartmann6_n80": [0.65, 0.79], "bioprocess_n40": [0.705, 0.82], "bioprocess_n100": [0.885, 0.945], "heteroscedastic_n40": [0.7, 0.825]},
        },
    },
}


def _lowest_over_stacks() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for table in MEASURED_BY_STACK.values():
        for method, levels in table.items():
            for lv, by in levels.items():
                cell = out.setdefault(method, {}).setdefault(lv, {})
                for sc, v in by.items():
                    cell[sc] = min(v, cell.get(sc, v))
    return out


# method -> level -> scenario -> the LOWEST coverage measured on either stack.
MEASURED: dict[str, dict[str, dict[str, float]]] = _lowest_over_stacks()


def _key(level: float) -> str:
    return f"{float(level):g}"


def coverage_at_proposals_estimate(method: str, level: float) -> dict[str, Any] | None:
    """`{nominal, measured_min, measured_max, by_scenario, by_stack,
    ci95_by_stack, basis, source}` for `method` at `level`, or None when that
    combination was never measured. `measured_min` / `measured_max` span both
    stacks; `by_scenario` is the lowest of the two per scenario."""
    k = _key(level)
    by = MEASURED.get(method, {}).get(k)
    if not by:
        return None
    by_stack = {st: dict(t[method][k]) for st, t in MEASURED_BY_STACK.items() if k in t.get(method, {})}
    vals = [v for t in by_stack.values() for v in t.values()]
    ci = {st: dict(t[method][k]) for st, t in CI95_BY_STACK.items() if k in t.get(method, {})}
    return {
        "nominal": round(float(level), 4),
        "measured_min": round(min(vals), 3),
        "measured_max": round(max(vals), 3),
        "by_scenario": dict(by),
        "by_stack": by_stack,
        "ci95_by_stack": ci or None,
        "basis": "simulation",
        "source": dict(SOURCE),
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
    "BASIS_HELD_OUT", "BASIS_MODEL", "BASIS_SELECTION_AWARE", "CAVEAT_CODE", "CAVEAT_CODE_MEASURED", "CI95_BY_STACK",
    "MEASURED", "MEASURED_BY_STACK", "SOURCE", "STACKS",
    "compact", "coverage_at_proposals_estimate", "proposal_caveat",
]
