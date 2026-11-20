"""kalos/bench/scoreboard.py: the diff logic and noise bands (torch-free, fast).

The measurements themselves are the existing benchmarks' (faults, coverage at
proposals, vs_baselines, trust goldens); they are exercised by their own tests
and by the CI scoreboard job. Here: what `diff` calls a regression.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from kalos.bench import scoreboard as sb

ROOT = Path(__file__).resolve().parents[1]


def _board(**over):
    b = {
        "schema": 1, "commit": "a" * 40, "branch": "x", "created_at": "t",
        "env": {"platform": "p", "python": "3.11", "cpus": 4, "torch": "2"},
        "sections": {
            "faults": {"status": "ok", "tier": "fast", "caught": 2, "noticed": 1, "missed": 1,
                       "faults_scored": 4, "clean_false_alarm": 1, "clean_scored": 2,
                       "caught_ci": sb.wilson(2, 4), "false_alarm_ci": sb.wilson(1, 2),
                       "per_fault": {"a": "caught", "b": "caught", "c": "noticed", "d": "missed"},
                       "per_clean": {"c1": True, "c2": False}},
            "coverage": {"status": "ok", "levels": {
                "0.9": {"coverage_at_proposals": 0.92, "ci95_seed_cluster": [0.88, 0.96],
                        "median_width": 1.0, "n_measured": 100},
                "0.95": {"coverage_at_proposals": 0.96, "ci95_seed_cluster": [0.93, 0.99],
                         "median_width": 1.3, "n_measured": 100}}},
            "regret": {"status": "ok", "mean_regret_norm": 0.10, "ci95": [0.08, 0.12],
                       "fallback_steps": 0, "tasks": {"t": {"mean": 0.1, "ci95": [0.08, 0.12]}}},
            "walltime": {"status": "ok", "seconds": 100.0},
            "golden": {"status": "ok", "sheets": {"s1": "d1", "s2": "d2"}},
        },
    }
    for path, val in over.items():
        cur = b
        keys = path.split("/")
        for k in keys[:-1]:
            cur = cur[k]
        cur[keys[-1]] = val
    return b


def test_identical_boards_have_no_regression():
    md, regs, notes = sb.diff(_board(), _board())
    assert regs == [] and notes == []
    assert "no regression" in md


def test_fault_outcome_getting_worse_is_a_regression_and_better_is_a_note():
    head = _board()
    head["sections"]["faults"]["per_fault"] = {"a": "noticed", "b": "caught", "c": "caught", "d": "missed"}
    _, regs, notes = sb.diff(_board(), head)
    assert regs == ["faults: `a` caught -> noticed"]
    assert any("`c` improved" in n for n in notes)


def test_new_clean_false_alarm_is_a_regression():
    head = _board()
    head["sections"]["faults"]["per_clean"] = {"c1": False, "c2": True}
    _, regs, notes = sb.diff(_board(), head)
    assert regs == ["faults: clean `c2` now false-alarms"]
    assert any("c1" in n for n in notes)


def test_coverage_drop_within_noise_band_is_not_a_regression():
    head = _board()
    head["sections"]["coverage"]["levels"]["0.9"]["coverage_at_proposals"] = 0.89  # drop 0.03 < band 0.04
    assert sb.diff(_board(), head)[1] == []


def test_coverage_drop_below_nominal_beyond_band_is_a_regression():
    head = _board()
    head["sections"]["coverage"]["levels"]["0.9"]["coverage_at_proposals"] = 0.85
    regs = sb.diff(_board(), head)[1]
    assert len(regs) == 1 and regs[0].startswith("coverage 90%")


def test_coverage_drop_that_stays_above_nominal_is_not_a_regression():
    base = _board()
    base["sections"]["coverage"]["levels"]["0.9"]["coverage_at_proposals"] = 0.99
    head = copy.deepcopy(base)
    head["sections"]["coverage"]["levels"]["0.9"]["coverage_at_proposals"] = 0.93
    assert sb.diff(base, head)[1] == []


def test_wider_bands_beyond_ten_percent_are_a_regression():
    head = _board()
    head["sections"]["coverage"]["levels"]["0.95"]["median_width"] = 1.5
    regs = sb.diff(_board(), head)[1]
    assert regs and regs[0].startswith("band width 95%")


def test_regret_band_and_fallbacks():
    head = _board()
    head["sections"]["regret"]["mean_regret_norm"] = 0.11  # +0.01 < band 0.02
    assert sb.diff(_board(), head)[1] == []
    head["sections"]["regret"]["mean_regret_norm"] = 0.15
    assert sb.diff(_board(), head)[1][0].startswith("regret: 0.1000 -> 0.1500")
    head = _board()
    head["sections"]["regret"]["fallback_steps"] = 2
    assert "fallbacks" in sb.diff(_board(), head)[1][0]


def test_walltime_band_and_environment_guard():
    head = _board()
    head["sections"]["walltime"]["seconds"] = 120.0
    assert sb.diff(_board(), head)[1] == []
    head["sections"]["walltime"]["seconds"] = 130.0
    assert sb.diff(_board(), head)[1] == ["walltime: 100.0s -> 130.0s (+30%)"]
    head["env"] = {**head["env"], "cpus": 14}
    _, regs, notes = sb.diff(_board(), head)
    assert regs == [] and any("environments differ" in n for n in notes)


def test_golden_drift_lists_changed_paths():
    head = _board()
    head["sections"]["golden"]["sheets"] = {"s1": "d1", "s2": "CHANGED"}
    bg = {"s1": {"x": 1}, "s2": {"p": {"q": 1.0}, "r": [1, 2]}}
    hg = {"s1": {"x": 1}, "s2": {"p": {"q": 1.5}, "r": [1, 2]}}
    md, regs, _ = sb.diff(_board(), head, bg, hg)
    assert regs == ["golden drift: `s2` (1 leaf paths, e.g. p.q)"]
    assert "`s2` `p.q`: 1.0 -> 1.5" in md


def test_a_failed_head_section_is_a_regression():
    head = _board()
    head["sections"]["coverage"] = {"status": "error", "error": "boom"}
    assert sb.diff(_board(), head)[1] == ["coverage: head section failed (boom)"]


def test_wilson_matches_the_fault_suite():
    from kalos.bench.faults.score import wilson

    for k, n in [(0, 5), (3, 7), (25, 36), (6, 8)]:
        assert sb.wilson(k, n) == [round(x, 3) for x in wilson(k, n)]


def test_cli_diff_exit_code(tmp_path):
    for name, board in [("base", _board()), ("head", _board(**{"sections/walltime/seconds": 500.0}))]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "scoreboard.json").write_text(json.dumps(board))
    script = ROOT / "kalos" / "bench" / "scoreboard.py"
    ok = subprocess.run([sys.executable, str(script), "diff", str(tmp_path / "base"), str(tmp_path / "base")],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run([sys.executable, str(script), "diff", str(tmp_path / "base"), str(tmp_path / "head"),
                          "--md", str(tmp_path / "d.md")], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "walltime: 100.0s -> 500.0s" in (tmp_path / "d.md").read_text()


def test_render_marks_sections_not_run():
    board = _board()
    del board["sections"]["regret"]
    text = sb.render(board)
    assert "| Regret | not run |" in text
    assert "Planted faults caught (fast tier) | 2/4" in text


def test_merge_unions_sections_of_one_commit_and_refuses_mixed_commits(tmp_path):
    import pytest

    a, b = _board(), _board()
    a["sections"] = {k: v for k, v in a["sections"].items() if k in ("faults", "golden")}
    b["sections"] = {k: v for k, v in b["sections"].items() if k in ("coverage", "regret", "walltime")}
    for name, board in [("a", a), ("b", b)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "scoreboard.json").write_text(json.dumps(board))
    (tmp_path / "a" / "golden-outputs.json").write_text('{"s1": {}}')
    merged = sb.merge(tmp_path / "m", [tmp_path / "a", tmp_path / "b"])
    assert set(merged["sections"]) == {"faults", "golden", "coverage", "regret", "walltime"}
    assert (tmp_path / "m" / "golden-outputs.json").exists()
    assert sb.diff(merged, merged)[1] == []
    b["commit"] = "b" * 40
    (tmp_path / "b" / "scoreboard.json").write_text(json.dumps(b))
    with pytest.raises(SystemExit):
        sb.merge(tmp_path / "m2", [tmp_path / "a", tmp_path / "b"])


def test_full_tier_faults_are_diffed_separately():
    base = _board()
    base["sections"]["faults_full"] = copy.deepcopy(base["sections"]["faults"])
    head = copy.deepcopy(base)
    head["sections"]["faults_full"]["per_fault"]["b"] = "missed"
    regs = sb.diff(base, head)[1]
    assert regs == ["faults_full: `b` caught -> missed"]
    assert "Planted faults caught (full tier)" in sb.render(base)
