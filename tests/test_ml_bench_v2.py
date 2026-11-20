"""ML benchmark v2 harness (benchmarks/ml_v2/PROTOCOL.md): fast checks only.

The reference arm itself is not run here (one closed-loop job is minutes of GP
fits); these tests pin the protocol constants, the problem generators, the
adapter contract, the held-out guard and the report/criteria arithmetic using the
cheap random and tree arms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.bench.ml_v2 import problems, protocol
from kalos.bench.ml_v2.adapter import Arm, config_hash, load_arm
from kalos.bench.ml_v2.report import compare_records, holm, summarize, wilson
from kalos.bench.ml_v2.space import Space
from kalos.bench.ml_v2.suites import run_job, tasks_for


def test_tuning_heldout_and_ci_seeds_are_disjoint():
    for suite, tune in protocol.TUNING_SEEDS.items():
        held = set(protocol.HELDOUT_SEEDS[suite])
        assert not held & set(tune), suite
        assert not held & set(protocol.CI_SEEDS.get(suite, ())), suite
        assert not set(tune) & set(protocol.CI_SEEDS.get(suite, ())), suite
    assert set(protocol.HELDOUT_DATASETS) & set(protocol.TUNING_DATASETS) == {
        "fixture/synthetic_bioprocess", "fixture/synthetic_scaleup", "kaggle/fermentation"}


def test_tuning_split_never_lists_heldout_only_datasets():
    held_only = set(protocol.HELDOUT_DATASETS) - set(protocol.TUNING_DATASETS)
    for suite in ("pool", "emulator", "snapshot_real", "backtest"):
        names = {t.split(":")[-1] for t in tasks_for(suite, "tuning")}
        assert not names & held_only, suite
        assert not any(n.startswith("internal") for n in names), suite
    assert not [t for t in tasks_for("scale", "tuning") if t.startswith("internal")]


@pytest.mark.parametrize("family", problems.SIGNAL_FAMILIES)
def test_instances_are_deterministic_and_optimum_bounds_the_truth(family):
    a, b = problems.make(family, 3), problems.make(family, 3)
    assert a.meta == b.meta and a.f_star == b.f_star
    x = a.space.sample(512, np.random.default_rng(0), fixed=True)
    x = pd.concat([x, a.space.draw_covariates(len(x), np.random.default_rng(1))], axis=1)
    f = a.truth(x)
    assert np.all(a.space.feasible(x))
    assert f.max() <= a.f_star + 1e-6 * (1 + abs(a.f_star))
    assert a.f_star > a.f_med


def test_instances_differ_across_seeds():
    metas = {json.dumps(problems.make("bumps", s).meta, sort_keys=True, default=str) for s in range(6)}
    assert len(metas) > 1


def test_constrained_family_cuts_off_the_unconstrained_optimum():
    p = problems.make("constrained", 0)
    c = p.space.linear_constraints[0]
    assert c["op"] in ("<=", ">=") and len(c["coefficients"]) >= 2


@pytest.mark.parametrize("family", problems.NULL_FAMILIES)
def test_null_designs_have_no_signal(family):
    df, rounds, prob = problems.null_design(family, 1, 20)
    assert len(df) == 20 and len(rounds) == 20 and not prob.signal
    if family == "null_batch_column":
        assert "batch" in df


def test_space_projection_and_feasibility():
    sp = Space(continuous={"a": (0.0, 1.0), "b": (0.0, 2.0)}, categorical={"s": ("x", "y")}, fixed={"s": "y"},
               linear_constraints=[{"coefficients": {"a": 1.0, "b": 1.0}, "op": "<=", "rhs": 1.0}])
    rec = sp.project({"a": 5, "b": -1, "s": "x"})
    assert rec == {"a": 1.0, "b": 0.0, "s": "y"}
    pts = sp.sample(50, np.random.default_rng(0))
    assert sp.feasible(pts).all()


def test_load_arm_and_config_hash():
    arm = load_arm("kalos.bench.ml_v2.arms.baselines:make_random")
    assert isinstance(arm, Arm) and arm.shippable is False
    ref = load_arm("kalos.bench.ml_v2.arms.kalos_ref:make_arm")
    assert ref.shippable is True and ref.name == "kalos_ref"
    assert config_hash("m:f", {"a": 1}) != config_hash("m:f", {"a": 2})
    with pytest.raises(ValueError):
        load_arm("no_colon")


def test_heldout_guard_refuses_without_a_committed_freeze(monkeypatch):
    from kalos.bench.ml_v2 import harness

    monkeypatch.setattr(harness, "git_state", lambda: {"commit": "x", "dirty": False, "dirty_files": []})
    with pytest.raises(SystemExit, match="freeze"):
        harness.guard_heldout("no_such_arm_xyz", "kalos.bench.ml_v2.arms.baselines:make_random", {})
    monkeypatch.setattr(harness, "git_state", lambda: {"commit": "x", "dirty": True, "dirty_files": ["a.py"]})
    with pytest.raises(SystemExit, match="uncommitted"):
        harness.guard_heldout("kalos_ref", "kalos.bench.ml_v2.arms.kalos_ref:make_arm", {})


def test_random_arm_closed_loop_snapshot_and_report():
    arm = load_arm("kalos.bench.ml_v2.arms.baselines:make_random")
    recs = [run_job(arm, "closed_loop", "heteroscedastic", s, "smoke") for s in (0, 1, 2)]
    for r in recs:
        assert len(r["rounds"]) == protocol.ROUNDS + 1
        assert r["rounds"][-1]["n_total"] == r["n0"] + protocol.ROUNDS * protocol.Q
        assert 0 <= r["rounds"][-1]["regret_obs"] <= r["rounds"][0]["regret_obs"] + 1e-12
    recs.append(run_job(arm, "snapshot", "null_noise", 0, "smoke"))
    summary = summarize(recs)
    t = summary["tasks"]["closed_loop|heteroscedastic"]
    assert t["n_instances"] == 3 and t["regret_obs_final"]["mean"] is not None
    assert summary["pooled"]["false_trust_all_nulls"]["n"] == 1
    # identical records compare as no change: C1 fails, C4 passes, no hidden loss
    cmp = compare_records(recs, recs, [])
    assert cmp["criteria"]["C1a_regret"]["pass"] is False
    assert cmp["criteria"]["C4_false_trust"]["pass"] is True
    assert cmp["criteria"]["C6_no_hidden_loss"]["pass"] is True
    assert cmp["ships"] is False


def test_same_instance_same_initial_design_for_every_arm():
    a = run_job(load_arm("kalos.bench.ml_v2.arms.baselines:make_random"), "closed_loop", "bumps", 4, "smoke")
    b = run_job(load_arm("kalos.bench.ml_v2.arms.baselines:make_lhs"), "closed_loop", "bumps", 4, "smoke")
    assert a["rounds"][0] == b["rounds"][0]


def test_rf_arm_gives_bands_trust_and_a_batch():
    arm = load_arm("kalos.bench.ml_v2.arms.trees:make_rf", {"members": 2})
    rec = run_job(arm, "snapshot", "heteroscedastic", 2, "smoke")  # n = 40 (seed % 4 == 2)
    assert rec["error"] is None and rec["n_arm_proposals"] == protocol.Q
    assert rec["heldout"]["0.9"] is not None and rec["heldout"]["0.9"]["n"] == protocol.N_HELDOUT_POINTS


def test_stats_helpers():
    assert wilson(0, 10)[0] == 0.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == 0.5


def test_user_folder_loader(tmp_path: Path):
    from kalos.bench.ml_v2.datasets import load_user_folder

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.random(40), "b": rng.random(40)})
    df["titer"] = df["a"] + rng.normal(0, 0.1, 40)
    df.to_csv(tmp_path / "mine.csv", index=False)
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "bad.csv", index=False)
    found, skipped = load_user_folder(tmp_path)
    assert [d.name for d in found] == ["user/mine"] and found[0].target == "titer"
    assert "user/bad" in skipped
