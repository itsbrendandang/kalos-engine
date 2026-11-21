"""Planted-fault honesty suite (kalos/bench/faults, CRUX_PORT_PLAN Rank 4 / F03).

Fast and torch-free, except the one `slow` test at the end. Marker
`planted_faults` selects them all: `pytest -m planted_faults`.

- the suite builds byte-identically and the case folders leak no truth;
- every planter changes exactly the cells its fault names;
- the scorer matches a hand-labelled mini-suite, and Wilson intervals are right;
- the runner is blind (never imports the catalogue or reads the truth key);
- the fast tier never imports torch;
- the CI regression gate: the fast tier on the repo-only bases does no worse
  than the stored baseline (no fault outcome drops, no new false alarm).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.bench.faults import catalogue as cat
from kalos.bench.faults import score
from kalos.bench.faults.bases import load_bases
from kalos.bench.faults.cases import CASES_DIR, TRUTH_DIR, build_suite, case_numbers
from kalos.bench.faults.catalogue import FAULT_COLUMN, Matcher
from kalos.bench.faults.plant import PLANTERS, plant_clean, plant_fault
from kalos.bench.faults.signals import signal

pytestmark = pytest.mark.planted_faults

PKG = Path(cat.__file__).parent
BASELINE = PKG / "results" / "baseline_fast_gate.json"


@pytest.fixture(scope="module")
def no_datasets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty datasets folder: the optional Kaggle base is skipped, as in CI."""
    return tmp_path_factory.mktemp("no-datasets")


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory, no_datasets: Path) -> Path:
    out = tmp_path_factory.mktemp("faults-suite")
    build_suite(out, datasets_dir=no_datasets)
    return out


# --- catalogue -------------------------------------------------------------- #


def test_catalogue_is_consistent() -> None:
    ids = [f.id for f in cat.FAULTS] + [c.id for c in cat.CLEAN]
    assert len(ids) == len(set(ids))
    clean_ids = {c.id for c in cat.CLEAN}
    for f in cat.FAULTS:
        assert f.parent in clean_ids, f.id
        assert f.id in PLANTERS, f.id
        assert f.caught_if, f.id
        assert f.owner_module and f.owner_plan and f.expected, f.id
        assert set(f.tiers) <= {"fast", "full"} and f.tiers, f.id
        if f.surface == "scale":
            assert f.tiers == ("full",), f.id
    assert set(PLANTERS) == {f.id for f in cat.FAULTS}


# --- build: determinism and blinding ----------------------------------------- #


def test_build_is_byte_identical(tmp_path: Path, no_datasets: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    build_suite(a, datasets_dir=no_datasets)
    build_suite(b, datasets_dir=no_datasets)
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_case_folders_carry_no_truth(suite: Path) -> None:
    cases = suite / CASES_DIR
    assert not list(cases.rglob("truth*"))
    names = [f.id for f in cat.FAULTS] + [c.id for c in cat.CLEAN]
    words = {"fault", "planted", "truth", "clean_", "expected"}
    for path in cases.rglob("*"):
        if path.is_dir():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in [*names, *words]:
            assert token.lower() not in text, (path, token)
    for path in cases.rglob("case.json"):
        assert set(json.loads(path.read_text())) <= {"surface", "filename", "target", "entries", "request"}
    truth = json.loads((suite / TRUTH_DIR / "truth.json").read_text())["cases"]
    assert set(truth) == set(case_numbers().values())


def test_case_numbers_do_not_follow_catalogue_order() -> None:
    numbers = case_numbers()
    ordered = [numbers[f.id] for f in cat.FAULTS]
    assert ordered != sorted(ordered)


def test_kaggle_cases_are_skipped_not_scored_without_the_file(suite: Path) -> None:
    truth = json.loads((suite / TRUTH_DIR / "truth.json").read_text())["cases"]
    for rec in truth.values():
        needs = rec["base"] == "kaggle150"
        assert bool(rec["skipped"]) == needs, rec["id"]
        assert (suite / CASES_DIR / rec["case"]).exists() == (not needs)


# --- planters change exactly the intended cells ------------------------------- #

_CELL_FAULTS = [
    "fahrenheit_rows_bare", "fahrenheit_rows_labeled", "typo_x10_target", "typo_x1000_target",
    "typo_x10_feature", "typo_x1000_feature", "swapped_columns", "swapped_levels",
    "duplicate_run_id", "impossible_value", "mixed_units_unlabeled",
]


@pytest.mark.parametrize("fault_id", _CELL_FAULTS)
def test_planter_changes_only_planted_cells(fault_id: str) -> None:
    fault = cat.fault_by_id(fault_id)
    base = load_bases(None)[fault.base]
    assert base is not None
    planted = plant_fault(fault, base)
    clean, dirty = base.df, planted.df
    assert dirty.shape == clean.shape and list(dirty.columns) == list(clean.columns)
    assert planted.rows and planted.columns
    for col in clean.columns:
        for i in range(len(clean)):
            changed = str(clean.at[i, col]) != str(dirty.at[i, col])
            inside = i in planted.rows and col in planted.columns
            if not inside:
                assert not changed, (fault_id, i, col)
    # every planted row has a changed planted cell (a duplicated id changes only
    # the second of its two rows)
    changed_rows = [i for i in planted.rows
                    if any(str(clean.at[i, c]) != str(dirty.at[i, c]) for c in planted.columns)]
    expected = planted.rows[1:] if fault_id == "duplicate_run_id" else planted.rows
    assert changed_rows == expected, fault_id
    assert base.df.equals(load_bases(None)[fault.base].df)  # the base itself is untouched


def test_mixed_units_labeled_keeps_values_outside_planted_rows() -> None:
    fault = cat.fault_by_id("mixed_units_labeled")
    base = load_bases(None)[fault.base]
    assert base is not None
    planted = plant_fault(fault, base)
    col = fault.params["column"]
    for i in range(len(base.df)):
        text = str(planted.df.at[i, col])
        value = float(text.split()[0])
        if i in planted.rows:
            assert text.endswith("mg/L") and value == pytest.approx(base.df.at[i, col] * 1000, abs=0.06)
        else:
            assert text.endswith("g/L") and value == pytest.approx(base.df.at[i, col], abs=5e-4)
    other = [c for c in base.df.columns if c != col]
    pd.testing.assert_frame_equal(planted.df[other], base.df[other])


def test_duplicate_planters_append_copies() -> None:
    for fid, same_id in (("duplicated_rows_exact", True), ("duplicated_rows_new_id", False)):
        fault = cat.fault_by_id(fid)
        base = load_bases(None)[fault.base]
        assert base is not None
        planted = plant_fault(fault, base)
        n = fault.params["n_rows"]
        assert len(planted.df) == len(base.df) + n
        src, copies = planted.rows[:n], planted.rows[n:]
        body = [c for c in base.df.columns if c != "run_id"]
        for s, c in zip(src, copies):
            assert planted.df.loc[s, body].tolist() == planted.df.loc[c, body].tolist()
            assert (planted.df.at[s, "run_id"] == planted.df.at[c, "run_id"]) == same_id
        pd.testing.assert_frame_equal(planted.df.iloc[: len(base.df)], base.df)


def test_result_planters_change_one_entry() -> None:
    bases = load_bases(None)
    clean = plant_clean(cat.clean_by_id("clean_results_plain"), bases["results155"])  # type: ignore[arg-type]
    for fid in ("result_x1000", "result_x10", "result_negative", "result_copy_paste"):
        fault = cat.fault_by_id(fid)
        planted = plant_fault(fault, bases["results155"])  # type: ignore[arg-type]
        diff = [k for k, (a, b) in enumerate(zip(clean.entries, planted.entries)) if a != b]
        assert diff == planted.entry_faults == [fault.params["entry"]], fid
        assert all(a["recipe"] == b["recipe"] for a, b in zip(clean.entries, planted.entries))
    assert all(e["value"] > 0.3 for e in clean.entries)


def test_pure_noise_keeps_the_target_marginal() -> None:
    base = load_bases(None)["bioprocess30"]
    assert base is not None
    for k in (1, 2, 3):
        planted = plant_fault(cat.fault_by_id(f"pure_noise_target_s{k}"), base)
        a, b = base.df[base.target].to_numpy(), planted.df[base.target].to_numpy()
        assert sorted(a) == sorted(b) and not np.array_equal(a, b)
        pd.testing.assert_frame_equal(planted.df.drop(columns=base.target), base.df.drop(columns=base.target))


# --- scorer ------------------------------------------------------------------- #


def test_wilson_interval_known_values() -> None:
    assert score.wilson(0, 0) is None
    assert score.wilson(5, 10) == pytest.approx([0.237, 0.763], abs=1e-3)
    assert score.wilson(0, 10) == pytest.approx([0.0, 0.278], abs=1e-3)
    assert score.wilson(10, 10) == pytest.approx([0.722, 1.0], abs=1e-3)


def test_matcher_semantics() -> None:
    s = signal("validation", "outliers", "warning", column="titer", rows=[3, 7])
    assert score.matches(Matcher("validation", ("outliers",)), s)
    assert score.matches(Matcher("validation", ("*",)), s)
    assert not score.matches(Matcher("validation", ("outliers",), min_severity="error"), s)
    assert score.matches(Matcher("validation", ("outliers",), column=FAULT_COLUMN), s, columns=["titer"])
    assert not score.matches(Matcher("validation", ("outliers",), column=FAULT_COLUMN), s, columns=["pH"])
    assert score.matches(Matcher("validation", ("outliers",), rows=True), s, rows=[7])
    assert not score.matches(Matcher("validation", ("outliers",), rows=True), s, rows=[8])
    assert not score.matches(Matcher("refusal", ("outliers",)), s)
    w = signal("scale_readout", "warning", "warning", message="target ratio 9x exceeds 2x")
    assert score.matches(Matcher("scale_readout", ("warning",), message="ratio"), w)
    assert not score.matches(Matcher("scale_readout", ("warning",), message="tip speed"), w)


def _mini_suite(tmp_path: Path, observations: dict[str, dict]) -> dict:
    """A hand-labelled suite: truth for four faults and two clean cases."""
    truth = {
        "case-01": {"case": "case-01", "id": "typo_x10_target", "kind": "fault", "base": "bioprocess160",
                    "surface": "sheet", "tiers": ["fast", "full"], "parent": "clean_bioprocess160",
                    "parent_case": "case-05", "rows": [4, 9], "columns": ["titer_g_L"], "skipped": None},
        "case-02": {"case": "case-02", "id": "duplicate_run_id", "kind": "fault", "base": "bioprocess160",
                    "surface": "sheet", "tiers": ["fast", "full"], "parent": "clean_bioprocess160",
                    "parent_case": "case-05", "rows": [1, 2], "columns": ["run_id"], "skipped": None},
        "case-03": {"case": "case-03", "id": "swapped_levels", "kind": "fault", "base": "bioprocess160",
                    "surface": "sheet", "tiers": ["fast", "full"], "parent": "clean_bioprocess160",
                    "parent_case": "case-05", "rows": [0], "columns": ["medium_base"], "skipped": None},
        "case-04": {"case": "case-04", "id": "constant_target", "kind": "fault", "base": "bioprocess160",
                    "surface": "sheet", "tiers": ["fast", "full"], "parent": "clean_bioprocess160",
                    "parent_case": "case-05", "rows": [], "columns": ["titer_g_L"], "skipped": None},
        "case-05": {"case": "case-05", "id": "clean_bioprocess160", "kind": "clean", "base": "bioprocess160",
                    "surface": "sheet", "tiers": ["fast", "full"], "rows": [], "columns": [], "skipped": None},
        "case-06": {"case": "case-06", "id": "clean_results_plain", "kind": "clean", "base": "results155",
                    "surface": "result", "tiers": ["fast", "full"], "rows": [], "watch_rows": [155, 156],
                    "columns": [], "skipped": None},
    }
    (tmp_path / TRUTH_DIR).mkdir()
    (tmp_path / TRUTH_DIR / "truth.json").write_text(json.dumps({"cases": truth}))
    obs_dir = tmp_path / "observations" / "fast"
    obs_dir.mkdir(parents=True)
    for case, obs in observations.items():
        (obs_dir / f"{case}.json").write_text(json.dumps({"case": case, **obs}))
    return score.score_suite(tmp_path, "fast")


def test_scorer_matches_hand_labels(tmp_path: Path) -> None:
    standing = signal("validation", "controls_present", "warning")
    card = _mini_suite(tmp_path, {
        # caught: outliers on the target naming planted row 9
        "case-01": {"status": "ok", "signals": [
            signal("validation", "outliers", "warning", column="titer_g_L", rows=[9, 30]), standing]},
        # noticed: a new warning naming a planted row, but not the right reason
        "case-02": {"status": "ok", "signals": [
            signal("validation", "outliers", "warning", column="glucose_gL", rows=[2]), standing]},
        # missed: only the standing warning the clean parent also raises
        "case-03": {"status": "ok", "signals": [standing]},
        # a crash is missed even if it looked like a refusal elsewhere
        "case-04": {"status": "error", "signals": [], "error": "RuntimeError at x.py:1"},
        # clean sheet: an outlier warning is a catch detector firing = false alarm
        "case-05": {"status": "ok", "signals": [
            standing, signal("validation", "outliers", "warning", column="titer_g_L", rows=[1])]},
        # clean results: a warning on a base row (not a watched fold row) is not judged
        "case-06": {"status": "ok", "signals": [
            signal("validation", "outliers", "warning", column="titer_g_L", rows=[3])]},
    })
    outcome = {f["id"]: f["outcome"] for f in card["faults"]}
    assert outcome == {"typo_x10_target": "caught", "duplicate_run_id": "noticed",
                       "swapped_levels": "missed", "constant_target": "missed"}
    clean = {c["id"]: c for c in card["clean"]}
    assert clean["clean_bioprocess160"]["false_alarm"] is True
    assert clean["clean_bioprocess160"]["standing_warnings"] == ["validation:controls_present"]
    assert clean["clean_results_plain"]["false_alarm"] is False
    assert card["totals"]["caught"] == 1 and card["totals"]["crashed"] == 1
    assert card["rates"]["caught"]["n"] == 4
    assert card["rates"]["caught_detectable"]["n"] == 3  # swapped_levels is a declared blind spot
    md = score.to_markdown(card)
    assert "| `typo_x10_target` | sheet | **caught** |" in md
    assert "missed (blind spot)" in md


def test_signals_carry_the_engines_covariate_role() -> None:
    """The adapter re-labels the engine's `covariates` block as
    `columns:covariate [col]`, the planned code the catalogue matches on."""
    from kalos.bench.faults import signals

    out = signals.from_analysis({"covariates": {"columns": ["passage"]}})
    assert ("columns", "covariate", "passage") in {(s["source"], s["code"], s["column"]) for s in out}
    assert all(s["code"] != "covariate" for s in signals.from_analysis({"covariates": {"columns": []}}))


def test_regression_gate_logic() -> None:
    base = {"faults": {"a": "caught", "b": "noticed", "c": "missed"}, "clean": {"x": [], "y": ["v:outliers"]}}
    same = {"faults": {"a": "caught", "b": "caught", "c": "missed"}, "clean": {"x": [], "y": []}}
    assert score.regressions(base, same) == []
    worse = {"faults": {"a": "noticed", "b": "noticed"}, "clean": {"x": ["refusal:too_few_rows"], "y": ["v:outliers"]}}
    problems = score.regressions(base, worse)
    assert problems == ["a: caught -> noticed", "x: new false alarm(s) ['refusal:too_few_rows']"]


# --- blindness and torch-freedom ----------------------------------------------- #


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(("." * node.level) + (node.module or ""))
            names.update(f"{'.' * node.level}{node.module or ''}:{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def test_runner_is_blind() -> None:
    for mod in ("runner.py", "signals.py"):
        imported = _imports(PKG / mod)
        assert not any("catalogue" in n or "score" in n or "plant" in n or "bases" in n for n in imported), mod
        assert not any(n.endswith(":TRUTH_DIR") or n.endswith(":load_truth") for n in imported), mod
        text = (PKG / mod).read_text(encoding="utf-8")
        assert "evaluator" not in text.replace("never opens `evaluator/`", "").replace(
            "(never `evaluator/`", ""), mod


def test_fast_tier_is_torch_free(suite: Path) -> None:
    code = (
        "import sys, pathlib\n"
        "from kalos.bench.faults.runner import run_suite\n"
        f"run_suite(pathlib.Path({str(suite)!r}), 'fast')\n"
        "assert 'torch' not in sys.modules, 'fast tier imported torch'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


# --- the CI regression gate ------------------------------------------------------ #


def test_fast_tier_regression_gate(suite: Path) -> None:
    """The fast tier on the repo-only bases must do no worse than the stored
    baseline: no fault outcome drops, no clean case gains a false alarm. After an
    engine fix improves a case, regenerate the baseline in the same PR:
    `python -m kalos.bench.faults all --tier fast --datasets /nonexistent --out /tmp/f`
    then `python -m kalos.bench.faults baseline --tier fast --out /tmp/f`
    (docs/PLANTED_FAULTS.md)."""
    from kalos.bench.faults.runner import run_suite

    run_suite(suite, "fast")
    card = score.score_suite(suite, "fast")
    assert card["totals"]["crashed"] == 0, [f["id"] for f in card["faults"] if f["status"] == "error"]
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["protocol_version"] == cat.PROTOCOL_VERSION
    current = score.gate_view(card)
    assert set(current["faults"]) == set(baseline["faults"])
    assert score.regressions(baseline, current) == []


# --- the full tier (slow): a few cases end to end through the real pipeline ------ #


@pytest.mark.slow
def test_full_tier_smoke_matches_recorded_baseline(suite: Path) -> None:
    """Runs the full (torch) tier on four small cases and checks each outcome
    against the recorded full baseline scorecard. The complete full run is the
    CLI (`python -m kalos.bench.faults all --tier full`), about 30 minutes."""
    from kalos.bench.faults.runner import run_suite

    # The latest recorded full scorecard (feat/data-checks); the earlier ones are
    # kept unchanged as scorecard_full.json and scorecard_full_baseline.json.
    recorded = json.loads((PKG / "results" / "scorecard_full_data_checks.json").read_text(encoding="utf-8"))
    want = {f["id"]: f["outcome"] for f in recorded["faults"]}
    pick = {"pure_noise_target_s1", "too_few_rows_8", "constant_target", "result_x1000"}
    numbers = case_numbers()
    cases = [numbers[i] for i in sorted(pick)] + [numbers["clean_bioprocess30"], numbers["clean_results_plain"]]
    run_suite(suite, "full", only=cases)
    card = score.score_suite(suite, "full")
    got = {f["id"]: f["outcome"] for f in card["faults"] if f["id"] in pick}
    assert got == {i: want[i] for i in pick}
