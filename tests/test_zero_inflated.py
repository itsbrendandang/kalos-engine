"""Zero-inflated surface with an x-dependent crash region: the optimum is
real and feasible, the crash region is deterministic, the portal's GATE POLICY
switches the feasibility gate on for it, and over 10 seeds the gated loop
proposes fewer crashing experiments than the ungated one."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from kalos.bench import get_objective
from kalos.bench.benchmark import run_feasibility_race
from kalos.bench.objectives import ZERO_INFLATED_FACTORS, zero_inflated_titer


def test_optimum_is_feasible_and_crashes_are_deterministic():
    obj = get_objective("zero_inflated_titer_5d")
    best = np.array([[19.0, 36.5, 3.2, 7.0, 9.0]])
    assert obj.evaluate(best)[0][0] == obj.optimum
    assert not obj.crashed(best)[0]
    over = best.copy()
    over[0, 4] = 10.5
    assert obj.crashed(over)[0] and obj.evaluate(over)[0][0] == 0.0
    # random crashes only touch y_obs, and only at the stated rate
    X = np.repeat(best, 2000, axis=0)
    _, y_obs = obj.evaluate(X, noise=0.0, rng=np.random.default_rng(0))
    assert 0.03 < float((y_obs == 0).mean()) < 0.07


def _sheet(n: int = 30, seed: int = 0) -> pd.DataFrame:
    obj = zero_inflated_titer()
    b = obj.bounds
    X = b[0] + qmc.LatinHypercube(d=5, seed=seed).random(n) * (b[1] - b[0])
    _, y = obj.evaluate(X, noise=0.05, rng=np.random.default_rng(seed))
    df = pd.DataFrame(X, columns=list(ZERO_INFLATED_FACTORS))
    df["titer_g_L"] = y
    return df


def test_portal_gate_policy_turns_the_gate_on():
    from kalos.portal.analysis import _analyze

    obj = zero_inflated_titer()
    res = _analyze(_sheet(), "titer_g_L")
    g = res["proposal_gating"]
    assert g["gated"], g["reason"]
    assert g["feasibility_auc"] >= 0.65
    batch = np.array([[p["recipe"][f] for f in ZERO_INFLATED_FACTORS] for p in res["proposals"]])
    # most of the gated batch stays out of the crash region
    assert float(obj.crashed(batch).mean()) <= 0.5


def test_gated_loop_crashes_less_than_ungated_over_10_seeds():
    obj = zero_inflated_titer()
    gated = [run_feasibility_race(obj, gated=True, seed=s)["crash_rate"] for s in range(10)]
    ungated = [run_feasibility_race(obj, gated=False, seed=s)["crash_rate"] for s in range(10)]
    assert np.mean(gated) < np.mean(ungated)
