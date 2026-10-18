"""Head-to-head benchmark harness (docs/BENCHMARK.md): the protocol's invariants.

These tests check the wiring, not the result. They use tiny budgets so the suite
stays fast. The optional arms (Optuna, Ax) are skipped when the bench extra is absent.
"""
from __future__ import annotations

import numpy as np
import pytest

from kalos.bench import vs_baselines as vb
from kalos.bench.objectives import bioprocess_like

torch = pytest.importorskip("torch")  # the Kalos and GP-EI arms need the ml extra


def test_seed_lists_are_fixed_and_disjoint():
    assert vb.CONFIRMATORY_SEEDS == tuple(range(1000, 1030))
    assert len(vb.CONFIRMATORY_SEEDS) >= 30
    assert vb.PILOT_SEED not in vb.CONFIRMATORY_SEEDS
    assert not set(vb.QUICK_SEEDS) & set(vb.CONFIRMATORY_SEEDS)


def test_bioprocess_optimum_is_exact_and_reachable():
    obj = bioprocess_like()
    best = np.array([[0.2, 0.2, 0.8, 3, 2]], dtype=float)  # strain 3, inducer 2, its peak
    assert abs(float(obj.evaluate(best, 0.0)[0][0]) - obj.optimum) < 1e-9
    assert obj.optimum == pytest.approx(10.5)


def test_task_budgets_match_the_protocol():
    assert (vb.TASKS["gaussian_4d"].n_init, vb.TASKS["gaussian_4d"].q,
            vb.TASKS["gaussian_4d"].rounds, vb.TASKS["gaussian_4d"].n_total) == (5, 1, 18, 23)
    bio = vb.TASKS["bioprocess_like_mixed"]
    assert (bio.n_init, bio.q, bio.rounds, bio.depth, bio.n_total) == (8, 4, 5, 2, 28)
    assert bio.noises == (0.15,)


def test_shared_init_and_noise_do_not_depend_on_method():
    task = vb.TASKS["ackley_4d"]
    a = vb.shared_init(task, 1000)
    b = vb.shared_init(task, 1000)
    assert np.array_equal(a, b)
    # noise is a function of (task, eta, seed) only: common random numbers across methods
    assert np.array_equal(vb.noise_vector(task, 0.15, 1000), vb.noise_vector(task, 0.15, 1000))
    assert not np.array_equal(vb.noise_vector(task, 0.15, 1000), vb.noise_vector(task, 0.15, 1001))


def test_f_med_and_f_star_are_consistent():
    for name in vb.TASKS:
        task = vb.get_task(name)
        assert task.f_star > vb.f_med(task)  # the optimum sits above the random median


class _SpyArm(vb._Arm):
    """Random proposals that record the pending set they were given."""

    seen_pending: list = []

    def tell(self, X, y):
        return None

    def ask(self, q, pending):
        _SpyArm.seen_pending.append(np.asarray(pending).shape[0])
        return vb._uniform(self.task, q, np.random.default_rng(self.round))


def test_pipeline_passes_inflight_points_and_measures_everything(monkeypatch):
    _SpyArm.seen_pending = []
    monkeypatch.setattr(vb, "make_arm", lambda m, t, s: _SpyArm(t, s))
    run = vb.run_unit("bioprocess_like_mixed", "spy", 0.15, 1000)
    # depth 2 pipeline: nothing pending on the first ask, then 2 in flight each round
    assert _SpyArm.seen_pending == [0, 2, 2, 2, 2]
    assert run["N"] == 28
    assert len(run["curve_p"]) == 28
    assert run["fallback_steps"] == 0


def test_a_failing_arm_is_counted_as_fallback_not_hidden(monkeypatch):
    class _Boom(vb._Arm):
        def tell(self, X, y):
            return None

        def ask(self, q, pending):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(vb, "make_arm", lambda m, t, s: _Boom(t, s))
    run = vb.run_unit("gaussian_4d", "boom", 0.0, 1000)
    assert run["fallback_steps"] == 18
    assert len(run["errors"]) == 18
    assert run["N"] == 23 and len(run["curve_p"]) == 23


def test_curves_are_monotone_and_reach_levels_are_ordered():
    run = vb.run_unit("gaussian_4d", "random", 0.0, 1000)
    curve = np.asarray(run["curve_p"])
    assert np.all(np.diff(curve) >= -1e-12)
    assert run["reach75"] <= run["reach90"] <= run["N"] + 1


@pytest.mark.parametrize("method", ["kalos", "random", "lhs", "botorch_gp_ei"])
def test_arm_runs_end_to_end_on_a_tiny_budget(monkeypatch, method):
    tiny = vb.Task("tiny_gaussian", 9, vb.TASKS["gaussian_4d"].objective, 5, 2, 2, 0, (0.0,))
    monkeypatch.setitem(vb.TASKS, "tiny_gaussian", tiny)
    run = vb.run_unit("tiny_gaussian", method, 0.0, 1000)
    assert run["N"] == 9
    assert run["fallback_steps"] == 0, run["errors"]


def test_optuna_and_ax_arms_run_end_to_end_on_a_tiny_budget(monkeypatch):
    pytest.importorskip("optuna")
    tiny = vb.Task("tiny_bio", 8, bioprocess_like(), 8, 4, 2, 2, (0.15,))
    monkeypatch.setitem(vb.TASKS, "tiny_bio", tiny)
    run = vb.run_unit("tiny_bio", "optuna_tpe", 0.15, 1000)
    assert run["N"] == 16 and run["fallback_steps"] == 0, run["errors"]
    pytest.importorskip("ax")
    run = vb.run_unit("tiny_bio", "ax", 0.15, 1000)
    assert run["N"] == 16 and run["fallback_steps"] == 0, run["errors"]


def test_holm_adjustment_matches_the_textbook_example():
    adj = vb._holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_analysis_reports_paired_calls_and_wtl():
    def fake(method, seed, regret):
        return {
            "task": "t", "eta": 0.0, "method": method, "seed": seed, "N": 23,
            "regret_raw": regret, "regret_norm": regret, "obs_regret_norm": regret,
            "obs_regret_raw": regret, "reach90": 10, "reach75": 5, "fallback_steps": 0,
            "ask_seconds_total": 1.0, "ask_seconds_median": 0.1, "ask_seconds_p95": 0.2,
            "errors": [], "curve_p": [],
        }

    seeds = range(1000, 1012)
    runs = [fake("kalos", s, 0.1) for s in seeds] + [fake("random", s, 0.5) for s in seeds]
    summary = vb.analyse(runs)
    comp = summary["cells"][0]["paired_vs_kalos"]["random"]
    assert comp["call"] == "Kalos better"
    assert comp["wtl"] == {"kalos_wins": 12, "ties": 0, "kalos_losses": 0}
    assert "kalos" in summary["cells"][0]["methods"]
    md = vb.render_markdown(summary)
    assert "| random |" in md
