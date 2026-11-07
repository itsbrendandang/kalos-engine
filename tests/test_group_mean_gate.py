"""Group-mean floor as a promotion gate (compmodel PLAN E13).

The surrogate must beat the leave-one-row-out recipe mean by more than a paired
bootstrap's uncertainty: lower bound of the 95% CI of
Spearman(surrogate) - Spearman(group mean) above 0. The raw numbers stay in the
analysis (`cv_group_mean_baseline` and its `margin`).
"""
from __future__ import annotations

import numpy as np

from kalos.core.evaluation import _loo_group_mean, group_mean_margin
from kalos.core.gates import check_gates

GOOD = {"surrogate_spearman": 0.5, "feasibility_auc": 0.8, "ece": 0.05, "brier": 0.1}


def _replicated(n_groups: int = 12, reps: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n_groups)
    groups = np.repeat([f"r{i}" for i in range(n_groups)], reps)
    y = np.repeat(mu, reps) + rng.normal(0, 0.3, n_groups * reps)
    return y, groups


def test_a_model_equal_to_the_floor_fails():
    y, g = _replicated()
    idx = np.arange(len(y))
    pred = _loo_group_mean(y, g)  # the model IS the group-mean floor
    r = group_mean_margin(y, g, idx, pred)
    assert r["evaluable"] and r["difference"] == 0.0
    assert r["passed"] is False and r["ci95"][0] <= 0
    gate = check_gates({**GOOD, "group_mean_margin": r})
    assert not gate.passed
    assert any("group-mean floor" in f for f in gate.failures)


def test_a_model_clearly_better_than_the_floor_passes():
    y, g = _replicated()
    idx = np.arange(len(y))
    pred = y + np.random.default_rng(1).normal(0, 0.01, len(y))  # near-perfect OOF
    r = group_mean_margin(y, g, idx, pred)
    assert r["passed"] is True and r["ci95"][0] > 0
    gate = check_gates({**GOOD, "group_mean_margin": r})
    assert gate.passed and gate.metrics["group_mean_margin_ci_lo"] > 0


def test_oof_order_is_respected():
    """Pairing is by input row index, not by position."""
    y, g = _replicated()
    perm = np.random.default_rng(2).permutation(len(y))
    r1 = group_mean_margin(y, g, np.arange(len(y)), y)
    r2 = group_mean_margin(y, g, perm, y[perm])
    assert r1["difference"] == r2["difference"]


def test_too_few_replicated_recipes_is_not_judged():
    y = np.arange(10.0)
    g = np.array([f"r{i}" for i in range(10)])  # all singletons
    r = group_mean_margin(y, g, np.arange(10), y)
    assert r["evaluable"] is False and r["passed"] is None
    gate = check_gates({**GOOD, "group_mean_margin": r})
    assert gate.passed and gate.not_judged and "group-mean floor" in gate.not_judged[0]


def test_absent_key_keeps_library_behaviour():
    assert check_gates(GOOD).passed


def test_analysis_carries_the_margin_and_the_raw_floor():
    import pandas as pd

    from kalos.portal.pipeline import compute_upload_analysis

    rng = np.random.default_rng(3)
    rows = []
    for _k in range(10):
        a, b = rng.uniform(0, 4), rng.uniform(5, 7)
        for _ in range(3):
            rows.append({"Methanol": round(a, 3), "pH": round(b, 2),
                         "lipase_titer": 1.5 * a - (b - 6) ** 2 + rng.normal(0, 0.2)})
    raw = pd.DataFrame(rows).to_csv(index=False).encode()
    _, result = compute_upload_analysis(raw, "lipase_titer", False, "x.csv")
    base = result["cv_group_mean_baseline"]
    assert "spearman" in base and "margin" in base
    m = base["margin"]
    assert set(m) >= {"evaluable", "passed", "reason"}
    if m["evaluable"]:
        assert {"difference", "ci95", "surrogate_spearman", "group_mean_spearman"} <= set(m)
        judged = any("group-mean floor" in f for f in result["promotion"]["failures"])
        assert judged == (m["passed"] is False)
