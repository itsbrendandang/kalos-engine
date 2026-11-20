"""Pure per-clone statistics (`kalos/core/strain.py`, STRAIN_VIEWS.md 3.2-3.6).

Fast and torch-free: every test here runs on numpy/scipy only.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats

from kalos.core import strain as st
from kalos.core.drivers import benjamini_hochberg


def _strict(obj: object) -> str:
    return json.dumps(obj, allow_nan=False)


# --------------------------------------------------------------------------- #
# measured_summary
# --------------------------------------------------------------------------- #


def test_t_ci_matches_scipy_reference_unrounded():
    rng = np.random.default_rng(0)
    for n in (3, 5, 8, 20):
        y = rng.normal(2.9, 0.3, n)
        out = st.measured_summary(y, round_digits=None)
        ref = stats.t.interval(0.95, n - 1, loc=y.mean(), scale=stats.sem(y))
        assert abs(out["ci95"][0] - ref[0]) < 1e-9
        assert abs(out["ci95"][1] - ref[1]) < 1e-9
        assert abs(out["mean"] - y.mean()) < 1e-12
        assert abs(out["sd"] - y.std(ddof=1)) < 1e-12
        assert out["ci_reason"] is None
        assert out["coverage"] == 0.95
        # Rounded output agrees to its 6 significant digits.
        rounded = st.measured_summary(y)
        assert rounded["ci95"][0] == pytest.approx(ref[0], rel=1e-5)


def test_ci_none_below_three_runs_and_nan_ignored():
    out = st.measured_summary([1.0, float("nan"), 2.0], run_ids=["a", "b", "c"])
    assert out["n_runs"] == 2 and out["n_missing"] == 1
    assert out["ci95"] is None and out["ci_reason"] == "n<3"
    assert out["sd"] is not None
    assert out["best"] == 2.0 and out["best_run_id"] == "c"
    one = st.measured_summary([4.0])
    assert one["sd"] is None and one["ci95"] is None and one["ci_reason"] == "n<3"
    empty = st.measured_summary([float("nan")])
    assert empty["n_runs"] == 0 and empty["mean"] is None
    _strict([out, one, empty])


def test_best_run_respects_direction_and_zero_sd_interval():
    y = [3.0, 1.0, 2.0]
    assert st.measured_summary(y, run_ids=["r1", "r2", "r3"], maximize=False)["best_run_id"] == "r2"
    flat = st.measured_summary([2.0, 2.0, 2.0])
    assert flat["ci95"] == [2.0, 2.0]


def test_replicate_sd_is_df_weighted_pooled():
    y = [1.0, 1.2, 1.4, 5.0, 5.5, 9.0]
    keys = ["A", "A", "A", "B", "B", "C"]
    out = st.measured_summary(y, recipe_key=keys, round_digits=None)
    var_a = np.var([1.0, 1.2, 1.4], ddof=1)
    var_b = np.var([5.0, 5.5], ddof=1)
    expected = math.sqrt((2 * var_a + 1 * var_b) / 3)
    assert out["replicate_sd"] == pytest.approx(expected, abs=1e-12)
    assert out["n_recipes"] == 3 and out["n_replicated_recipes"] == 2
    single = st.measured_summary([1.0, 1.1, 3.0], recipe_key=["A", "A", "B"])
    assert single["replicate_sd"] is None and "2 recipes" in single["replicate_sd_reason"]
    assert st.measured_summary([1.0, 2.0])["replicate_sd_reason"] == "no recipe key supplied"


# --------------------------------------------------------------------------- #
# eb_clone_effects
# --------------------------------------------------------------------------- #


def _groups(spec: dict[str, list[float]]) -> tuple[list[float], list[str]]:
    r: list[float] = []
    lab: list[str] = []
    for k, vals in spec.items():
        r += vals
        lab += [k] * len(vals)
    return r, lab


def test_eb_tau_zero_gives_full_pooling():
    # Identical residual sets per clone: Q = 0, tau2 = 0, every theta = mu.
    base = [-1.0, 0.0, 1.0, 0.5, -0.5]
    r, lab = _groups({"A": base, "B": base, "C": base})
    out = st.eb_clone_effects(r, lab, reference="A")
    assert out["tau2"] == 0.0 and out["tau"] == 0.0
    assert out["i2"] is None  # Q == 0
    for k in "ABC":
        ent = out["levels"][k]
        assert ent["shrinkage_w"] == 0.0
        assert ent["theta"] == out["mu"]
        assert ent["estimate"] == 0.0
        assert ent["mostly_pooled"] is True
    # Full pooling moves the point estimate, never shrinks the interval to [0, 0].
    lo, hi = out["levels"]["B"]["ci95"]
    assert lo < 0 < hi


def test_eb_reference_estimate_zero_and_relative_effects():
    rng = np.random.default_rng(3)
    r, lab = _groups({k: list(off + rng.normal(0, 0.2, 8)) for k, off in [("A", 0.0), ("B", 2.0), ("C", -2.0)]})
    out = st.eb_clone_effects(r, lab, reference="A")
    a, b, c = (out["levels"][k] for k in "ABC")
    assert a["estimate"] == 0.0 and a["is_reference"] is True and a["ci95"] is None
    assert b["estimate"] == pytest.approx(b["theta"] - a["theta"], abs=1e-5)
    assert b["estimate"] > 1.5 and c["estimate"] < -1.5
    assert b["approximate"] is True and b["coverage"] == 0.95
    assert out["method"] == "oof_residual_empirical_bayes"
    assert out["tau"] > 1 and out["i2"] > 0.9
    _strict(out)


def test_eb_shrinkage_weight_monotone_in_n():
    # Same between-level spread and the same within-level SD; only n differs.
    unit = np.array([-1.0, 1.0])  # sd = sqrt(2) for each pair
    spec = {}
    for k, (off, reps) in {"A": (0.0, 3), "B": (1.0, 2), "C": (-1.0, 4), "D": (0.5, 8)}.items():
        spec[k] = list(off + np.tile(unit, reps))
    r, lab = _groups(spec)
    out = st.eb_clone_effects(r, lab, reference="A")
    assert out["tau2"] > 0
    w = {k: out["levels"][k]["shrinkage_w"] for k in "ABCD"}
    n = {k: out["levels"][k]["n"] for k in "ABCD"}
    by_n = sorted("ABCD", key=lambda k: n[k])
    assert [w[k] for k in by_n] == sorted(w[k] for k in by_n)
    assert len(set(w.values())) == 4
    # theta sits between rbar and mu, closer to mu for smaller w.
    mu = out["mu"]
    for k in "ABCD":
        ent = out["levels"][k]
        assert min(ent["rbar"], mu) - 1e-6 <= ent["theta"] <= max(ent["rbar"], mu) + 1e-6


def test_eb_zero_within_sd_does_not_give_infinite_weight():
    r, lab = _groups({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.5, -0.5, 0.2], "C": [2.0, 2.5, 1.5]})
    out = st.eb_clone_effects(r, lab, reference="B")
    assert out["levels"]["A"]["v"] > 0
    _strict(out)


def test_eb_refusals():
    r, lab = _groups({"A": [0.1, 0.2, 0.3, 0.0], "B": [1.0, 1.1], "C": [0.5, 0.4, 0.6]})
    out = st.eb_clone_effects(r, lab, reference="A")
    b = out["levels"]["B"]
    assert b["estimate"] is None and b["refusal"]["code"] == "clone_too_few_runs"
    assert set(b["refusal"]) >= {"code", "level", "reason", "what_would_fix_it"}
    assert out["levels"]["C"]["estimate"] is not None
    assert out["refusal"] is None

    two = st.eb_clone_effects(*_groups({"A": [0.1, 0.2, 0.3], "B": [1.0, 1.1]}), reference="A")
    assert two["refusal"]["code"] == "clone_effect_needs_two_clones"
    assert all(e["estimate"] is None for e in two["levels"].values())
    assert {x["code"] for x in two["refusals"]} == {"clone_too_few_runs", "clone_effect_needs_two_clones"}

    ref = st.eb_clone_effects(r, lab, reference="B")
    assert ref["refusal"]["code"] == "reference_clone_too_few_runs"
    assert all(e["estimate"] is None for e in ref["levels"].values())
    assert ref["refusal"]["what_would_fix_it"]
    _strict([out, two, ref])


def _eb_coverage(offsets: list[float], sd: float, n_sets: int = 200) -> float:
    hit = tot = 0
    for seed in range(n_sets):
        rng = np.random.default_rng(seed)
        k = int(rng.integers(3, 5))
        r: list[float] = []
        lab: list[str] = []
        for i in range(k):
            n = int(rng.integers(6, 11))
            r += list(offsets[i] + rng.normal(0, sd, n))
            lab += [f"C{i}"] * n
        out = st.eb_clone_effects(r, lab, reference="C0")
        for i in range(1, k):
            lo, hi = out["levels"][f"C{i}"]["ci95"]
            tot += 1
            hit += lo <= offsets[i] - offsets[0] <= hi
    return hit / tot


@pytest.mark.parametrize(
    ("offsets", "sd"),
    [
        ([0.0, 0.6, 1.1, 0.3], 0.5),  # STRAIN_VIEWS probe offsets
        ([0.0, 0.2, 0.4, 0.1], 1.0),  # small offsets: tau2 often 0; the pure EB interval covers ~46% here
        ([0.0, 2.0, 4.0, 1.0], 0.5),  # well-separated clones: little shrinkage
    ],
)
def test_eb_interval_coverage_simulation(offsets, sd):
    """200 seeded datasets, 3-4 clones x 6-10 runs, known offsets, Gaussian noise.

    The interval is documented as approximate; the requirement is >= 90%
    coverage of theta_k - theta_ref's true value (measured 0.94, 0.95, 0.94 for the three cases; the pure EB interval measured 0.86, 0.46, 0.91).
    """
    assert _eb_coverage(offsets, sd) >= 0.90


# --------------------------------------------------------------------------- #
# confounding
# --------------------------------------------------------------------------- #


def test_cramers_v_one_when_each_clone_has_its_own_round():
    clones = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    rounds = [0] * 4 + [1] * 4 + [2] * 4
    assert st.cramers_v(clones, rounds) == 1.0
    assert st.cramers_v(clones, ["x"] * 12) is None
    # Independent design: V = 0.
    assert st.cramers_v(["A", "B", "A", "B"], [0, 0, 1, 1]) == 0.0
    out = st.confounding(clones, {"round": rounds})
    col = out["columns"][0]
    assert col["fully"] is True and col["aliased"] == ["A", "B", "C"]
    assert out["not_separable"] == {"A": ["round"], "B": ["round"], "C": ["round"]}
    _strict(out)


def test_aliased_detection_and_confined_but_shared_is_not_aliased():
    #   thaw T1: A, B     thaw T2: B     thaw T3: C, C (C alone in T3 and T4)
    clones = ["A", "A", "B", "B", "C", "C", "C"]
    thaw = ["T1", "T1", "T1", "T2", "T3", "T3", "T4"]
    out = st.confounding(clones, {"thaw": thaw, "lot": [None, "L1", "L1", "L2", "", "L2", "L1"]})
    th = next(c for c in out["columns"] if c["column"] == "thaw")
    assert th["confined"] == {"A": "T1"}  # A confined to T1 but T1 also holds B
    assert th["aliased"] == ["C"]  # C occupies T3/T4, which hold no other clone
    assert th["fully"] is False
    assert out["not_separable"] == {"C": ["thaw"]}
    lot = next(c for c in out["columns"] if c["column"] == "lot")
    assert lot["n"] == 5  # blanks dropped pairwise
    single = st.confounding(clones, {"scale": ["ambr15"] * 7})
    assert single["columns"] == [] and single["skipped"][0]["column"] == "scale"


# --------------------------------------------------------------------------- #
# process_balance
# --------------------------------------------------------------------------- #


def test_process_balance_flags_disjoint_temperatures():
    rng = np.random.default_rng(1)
    temp = np.r_[rng.uniform(30, 33, 6), rng.uniform(36, 37, 6)]
    ph = rng.uniform(6.8, 7.2, 12)
    labels = ["CHO-A12"] * 6 + ["CHO-B07"] * 6
    out = st.process_balance(np.c_[temp, ph], labels, ["temp_C", "pH"])
    assert out["flag"] is True
    t = out["by_input"]["temp_C"]
    assert t["flag"] is True and t["smd_max"] > 0.8 and t["coverage_min"] < 0.3
    assert any("CHO-B07 ran only at temp_C" in m and "partly reflects temp_C" in m for m in out["messages"])
    _strict(out)


def test_process_balance_small_clone_reported_not_flagged():
    temp = [30, 32, 34, 36, 38, 31, 33, 35, 37, 36.5, 36.6]
    labels = ["A"] * 5 + ["B"] * 4 + ["C"] * 2
    out = st.process_balance(np.array(temp, dtype=float), labels, ["temp_C"])
    c = out["by_input"]["temp_C"]["by_level"]["C"]
    assert c["n"] == 2 and c["eligible"] is False and c["flagged"] is False
    assert out["flag"] is False and out["messages"] == []


# --------------------------------------------------------------------------- #
# interaction_screen
# --------------------------------------------------------------------------- #


def _interaction_data(seed: int, n: int, slopes: list[float], sd: float = 0.5):
    rng = np.random.default_rng(seed)
    X: list[tuple[float, float]] = []
    y: list[float] = []
    lab: list[str] = []
    for i, b in enumerate(slopes):
        t = rng.uniform(30, 38, n)
        ph = rng.uniform(6.8, 7.2, n)
        X += list(zip(t, ph, strict=True))
        y += list(1 + b * (t - 34) + 0.5 * (ph - 7) + rng.normal(0, sd, n))
        lab += [f"C{i}"] * n
    return np.array(X), np.array(y), lab


NAMES = ["temp_C", "pH"]


def test_interaction_opposite_slopes_flag():
    X, y, lab = _interaction_data(1, 10, [0.3, -0.3, 0.0])
    out = st.interaction_screen(X, y, lab, NAMES, seed=7)
    row = out["rows"]["temp_C"]
    assert row["status"] == "tested" and row["flag"] is True
    assert row["p"] < 0.05 and row["i2"] >= 0.5 and row["n_qualifying"] == 3
    assert out["flags"] == ["temp_C"]
    cell = out["cells"]["C0"]["temp_C"]
    assert cell["status"] == "estimated" and cell["adjusted_for_other_inputs"] is True
    assert cell["ci95"][0] < cell["slope"] < cell["ci95"][1]
    assert out["thin_rule"] == {"min_runs": 6, "min_levels": 3, "min_range_frac": 0.5}
    # Deterministic for a seed.
    assert _strict(out) == _strict(st.interaction_screen(X, y, lab, NAMES, seed=7))


def test_interaction_four_runs_insufficient_never_no_interaction():
    X, y, lab = _interaction_data(1, 4, [0.3, -0.3, 0.0])
    out = st.interaction_screen(X, y, lab, NAMES, seed=7)
    for name, row in out["rows"].items():
        assert row["status"] == "insufficient"
        assert row["flag"] is False and row["p"] is None
        assert "n=4 < 6 runs" in row["reason"] and name in row["reason"]
    texts = _strict({"rows": out["rows"], "cells": out["cells"]}).lower()
    assert not re.search(r"no (significant )?interaction|not interact|no difference|no evidence", texts)
    assert all(c["status"] == "insufficient" for per in out["cells"].values() for c in per.values())
    assert "not that there is no interaction" in out["note"]


def test_interaction_thin_rule_reasons():
    rng = np.random.default_rng(2)
    # C0: full range, 8 runs. C1: 8 runs but only 2 distinct temps. C2: 8 runs in a narrow band.
    t = np.r_[rng.uniform(30, 38, 8), np.repeat([33.0, 35.0], 4), rng.uniform(34, 36, 8)]
    y = rng.normal(0, 1, 24)
    lab = ["C0"] * 8 + ["C1"] * 8 + ["C2"] * 8
    out = st.interaction_screen(t, y, lab, ["temp_C"], seed=0, n_boot=50)
    cells = out["cells"]
    assert cells["C0"]["temp_C"]["status"] == "estimated"
    assert cells["C0"]["temp_C"]["adjusted_for_other_inputs"] is False
    assert "2 distinct levels < 3" in cells["C1"]["temp_C"]["reason"]
    assert re.search(r"covers \d+% < 50% of the range", cells["C2"]["temp_C"]["reason"])
    row = out["rows"]["temp_C"]
    assert row["status"] == "insufficient" and "1 of 3 clones qualify" in row["reason"]


def test_interaction_null_p_value_calibrated():
    """No interaction: Q's p-value >= 0.05 in >= 90% of 100 seeds (measured 99/100; chi2 reference 95/100)."""
    keep = 0
    for seed in range(100):
        X, y, lab = _interaction_data(seed, 10, [0.2, 0.2, 0.2])
        out = st.interaction_screen(X, y, lab, NAMES, seed=seed, n_boot=0)
        keep += out["rows"]["temp_C"]["p"] >= 0.05
    assert keep >= 90


def test_bh_adjusted_matches_drivers_bh():
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = rng.uniform(0, 0.2, int(rng.integers(1, 12)))
        adj = st.bh_adjusted(p)
        assert np.all(adj >= p - 1e-15) and np.all(adj <= 1)
        np.testing.assert_array_equal(adj <= 0.05, benjamini_hochberg(p, q=0.05))


# --------------------------------------------------------------------------- #
# by_round_best and JSON
# --------------------------------------------------------------------------- #


def test_by_round_best_sorted_measured():
    y = [1.0, 3.0, 2.0, float("nan"), 5.0, 4.0]
    lab = ["A", "A", "B", "B", "A", None]
    rounds = [10, 2, 2, 10, 10, 2]
    out = st.by_round_best(y, lab, rounds)
    assert [r["round"] for r in out] == [2, 10]
    assert out[0]["levels"] == {"A": {"best": 3.0, "n": 1}, "B": {"best": 2.0, "n": 1}}
    assert out[1]["levels"] == {"A": {"best": 5.0, "n": 2}}
    low = st.by_round_best(y, lab, rounds, maximize=False)
    assert low[1]["levels"]["A"]["best"] == 1.0
    _strict(out)


def test_labels_are_str_and_outputs_strict_json():
    labels = [329.0, 329.0, 329.0, 410.5, 410.5, 410.5, 7, 7, 7]
    y = [1.0, 1.5, 2.0, 3.0, float("inf"), 3.5, 0.0, 0.1, 0.2]
    eb = st.eb_clone_effects(y, labels, reference=329)
    assert set(eb["levels"]) == {"329", "410.5", "7"}
    pb = st.process_balance(np.zeros((9, 1)), labels, ["temp_C"])
    assert pb["by_input"]["temp_C"]["smd_max"] is None
    for obj in (eb, pb, st.measured_summary(y), st.interaction_screen(np.zeros(9), y, labels, ["t"], seed=0)):
        _strict(obj)


def test_module_is_torch_free():
    code = "import sys, kalos.core.strain; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
