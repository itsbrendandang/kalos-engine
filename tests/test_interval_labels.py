"""F02: every interval carries the basis of its coverage claim, and proposal
intervals carry the measured-at-proposals caveat as structured fields.

Fast unit tests on `kalos.portal.analysis_parts.intervals` and
`coverage_evidence`, plus one integration check through `_analyze`.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from kalos.portal.analysis_parts.coverage_evidence import (
    BASIS_HELD_OUT,
    BASIS_MODEL,
    BASIS_SELECTION_AWARE,
    CAVEAT_CODE,
    CAVEAT_CODE_MEASURED,
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
    lo = math.floor(cav["coverage_at_proposals_estimate"]["measured_min"] * 100 + 1e-9)
    hi = math.floor(cav["coverage_at_proposals_estimate"]["measured_max"] * 100 + 1e-9)
    assert cav["message"].startswith("Based on held-out runs similar to past runs.")
    assert f"{lo}-{hi}%" in cav["message"] and "lower" in cav["message"]
    # an unmeasured method says so instead of inventing a number
    assert "not been measured" in proposal_caveat("no_such_method", 0.9)["message"]


_CALIBRATED_CLAIM = re.compile(r"\b(is|are|was|were|been|be)\s+(well[- ])?calibrated\b", re.I)


def _assert_honest_proposal_wording(message: str) -> None:
    """No text next to a proposal interval says it 'is calibrated'; it says
    what the band was calibrated on and what coverage was measured."""
    assert not _CALIBRATED_CLAIM.search(message), message
    assert "range is calibrated" not in message.lower(), message
    assert message.startswith("Based on held-out runs"), message


def test_no_proposal_caveat_says_the_range_is_calibrated():
    methods = (*MEASURED, "no_such_method")
    for method in methods:
        for level in LEVELS:
            _assert_honest_proposal_wording(proposal_caveat(method, level)["message"])
    msg = conformal_band_caveat(0.1)["message"]
    assert not _CALIBRATED_CLAIM.search(msg), msg
    # The held-out-run wording for the global band stays: that coverage WAS
    # measured on held-out past runs.
    assert msg.startswith("The +/- conformal_q band's coverage was measured on held-out past runs.")


def test_selection_aware_caveat_states_coverage_measured_at_proposed_recipes():
    for level, pct in ((0.9, 90), (0.95, 95)):
        cav = proposal_caveat("conformal_selection_aware", level)
        est = cav["coverage_at_proposals_estimate"]
        lo = math.floor(est["measured_min"] * 100 + 1e-9)
        hi = math.floor(est["measured_max"] * 100 + 1e-9)
        assert cav["code"] == (CAVEAT_CODE_MEASURED if est["measured_min"] >= level else CAVEAT_CODE)
        assert cav["message"].startswith("Based on held-out runs:")
        assert f"At proposed recipes this {pct}% range covered {lo}-{hi}% in simulation" in cav["message"]
        assert "not guaranteed" in cav["message"]
    # Main @ cf7105c, seeds 100-139, on the two stacks uv.lock pins: bioprocess
    # n=40 measured 0.86 at 90% and 0.935 at 95% on Python 3.11 (VERIFY_MAIN
    # claim 3), so neither level reached nominal everywhere and the caveat says so.
    msg90 = proposal_caveat("conformal_selection_aware", 0.9)["message"]
    assert "covered 86-99% in simulation, below 90% in some cases" in msg90
    msg95 = proposal_caveat("conformal_selection_aware", 0.95)["message"]
    assert "covered 93-99% in simulation, below 95% in some cases" in msg95


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
        # F15: the proposal bands are selection-aware when enough held-out runs
        # are top-ranked; the plain band (held_out_past_runs) is the fallback.
        assert all(iv["coverage_basis"] in {BASIS_HELD_OUT, BASIS_SELECTION_AWARE}
                   and iv["caveat"]["code"] in {CAVEAT_CODE, CAVEAT_CODE_MEASURED}
                   for iv in p["intervals"])
        assert all(g["coverage_basis"] == BASIS_MODEL for g in p["gp_intervals"])
        for iv in p["intervals"]:
            _assert_honest_proposal_wording(iv["caveat"]["message"])


def _results() -> dict[str, dict]:
    from kalos.portal.analysis_parts.coverage_evidence import STACKS

    root = Path(__file__).resolve().parents[1]
    return {st: json.loads((root / v["results"]).read_text()) for st, v in STACKS.items()}


def test_evidence_table_is_the_committed_results_files():
    """The numbers a client shows are the numbers the benchmark wrote: per stack,
    MEASURED_BY_STACK and CI95_BY_STACK are exactly `evidence_table` of that
    stack's results file, and MEASURED is the lower of the two per scenario."""
    from kalos.bench.coverage_at_proposals import EVIDENCE_SCENARIOS, evidence_table
    from kalos.portal.analysis_parts.coverage_evidence import (
        CI95_BY_STACK,
        MEASURED_BY_STACK,
        SOURCE,
        STACKS,
    )

    files = _results()
    assert set(files) == set(MEASURED_BY_STACK) == set(CI95_BY_STACK) == {"py311", "py312"}
    for st, res in files.items():
        measured, ci95 = evidence_table(res["summary"])
        assert measured == MEASURED_BY_STACK[st], st
        assert ci95 == CI95_BY_STACK[st], st
    for method, levels in MEASURED.items():
        for lv, by in levels.items():
            for sc, v in by.items():
                assert v == min(MEASURED_BY_STACK[st][method][lv][sc] for st in STACKS), (method, lv, sc)
    assert list(SOURCE["scenarios"]) == list(EVIDENCE_SCENARIOS)
    assert SOURCE["results"] == [v["results"] for v in STACKS.values()]


def test_results_files_record_their_commit_and_stack():
    """VERIFY_MAIN B5: a results file without its commit and environment cannot
    be reproduced. STACKS[...].recorded_on repeats each file's meta, both files
    use the documented seeds, and each stack is one of the two uv.lock pins
    (numpy/scipy fork at Python 3.12)."""
    from kalos.portal.analysis_parts.coverage_evidence import STACKS

    lock = (Path(__file__).resolve().parents[1] / "uv.lock").read_text()
    for st, res in _results().items():
        meta, env = res["meta"], res["meta"]["environment"]
        assert meta["seeds"] == 40 and meta["seed_offset"] == 100, st
        assert re.fullmatch(r"[0-9a-f]{40}", env["git_commit"]) and env["git_dirty"] is False, st
        rec = STACKS[st]["recorded_on"]
        assert rec["git_commit"] == env["git_commit"] and rec["python"] == env["python"], st
        assert rec["platform"] == env["platform"], st
        assert rec["torch_threads_per_worker"] == env["torch_threads_per_worker"] == 1, st
        for pkg in ("numpy", "scipy", "torch", "botorch", "gpytorch"):
            assert rec[pkg] == env["packages"][pkg], (st, pkg)
        marker = "python_full_version < '3.12'" if st == "py311" else "python_full_version >= '3.12'"
        assert (st == "py311") == env["python"].startswith("3.11."), st
        for pkg in ("numpy", "scipy"):
            assert f'{pkg}=={rec[pkg]} ; {marker}' in _lock_constraints(lock, pkg), (st, pkg)
        for pkg in ("torch", "botorch", "gpytorch"):
            assert f'name = "{pkg}"\nversion = "{rec[pkg]}"' in lock, (st, pkg)


def _lock_constraints(lock: str, pkg: str) -> str:
    """`pkg==version ; marker` lines for every locked version of `pkg`."""
    out = []
    pat = r'\{ name = "' + re.escape(pkg) + r'", version = "([^"]+)".*?marker = "([^"]+)" \}'
    for m in re.finditer(pat, lock):
        out.append(f"{pkg}=={m.group(1)} ; {m.group(2)}")
    return "\n".join(out)


def test_caveat_numbers_and_code_follow_the_results_files():
    """The served caveat text is built from the files: the whole-percent range
    (rounded down) from the lowest to the highest coverage measured over the five
    scenarios on both stacks, and the 'below N% in some cases' wording whenever
    the lowest is below nominal."""
    from kalos.bench.coverage_at_proposals import EVIDENCE_SCENARIOS

    files = _results()
    for level, pct in ((0.9, 90), (0.95, 95)):
        vals = [res["summary"][sc]["levels"][str(level)]["coverage_at_proposals"]
                for res in files.values() for sc in EVIDENCE_SCENARIOS]
        lo, hi = math.floor(min(vals) * 100 + 1e-9), math.floor(max(vals) * 100 + 1e-9)
        cav = proposal_caveat("conformal_selection_aware", level)
        assert f"covered {lo}-{hi}% in simulation" in cav["message"], (level, cav["message"])
        below = min(vals) < level
        assert (f"below {pct}% in some cases" in cav["message"]) is below
        assert cav["code"] == (CAVEAT_CODE if below else CAVEAT_CODE_MEASURED)
        est = cav["coverage_at_proposals_estimate"]
        assert est["measured_min"] == min(vals) and est["measured_max"] == max(vals)
        for st, res in files.items():
            for sc in EVIDENCE_SCENARIOS:
                assert est["ci95_by_stack"][st][sc] == res["summary"][sc]["levels"][str(level)]["ci95_seed_cluster"]
