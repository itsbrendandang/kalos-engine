"""F04: trust gates that gate. The verdict from the leak-free CV withholds or
labels the proposals, the floor needs the interval, small sheets are not
judged, and the promotion gate can pass on an ordinary target.

The verdict table itself is tested in tests/test_verdict.py (cherry-picked from
arch-safe-now with the verdict module)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.core.gates import NOT_APPLICABLE_FEW_NON_PRODUCERS, check_gates
from kalos.portal.analysis_parts.trust_gate import (
    MIN_ROWS_TO_JUDGE,
    clears_floor,
    label_proposals,
    proposal_trust,
)


@pytest.mark.parametrize(
    ("kw", "ok", "judged"),
    [
        ({}, True, True),
        ({"ci95": [-0.05, 0.8]}, False, True),       # point estimate alone no longer clears
        ({"rho": 0.15, "ci95": [0.01, 0.3]}, False, True),
        ({"n_runs": MIN_ROWS_TO_JUDGE - 1}, False, False),
        ({"n_groups": 2, "ci95": None}, False, False),
        ({"rho": float("nan"), "ci95": None}, False, False),
    ],
)
def test_clears_floor_needs_the_interval_and_enough_runs(kw, ok, judged):
    base = {"rho": 0.5, "ci95": [0.1, 0.8], "floor": 0.2, "n_runs": 30, "n_groups": 10}
    base.update(kw)
    out = clears_floor(**base)
    assert out["clears_floor"] is ok and out["judged"] is judged
    assert (out["not_judged_reason"] is None) == judged


@pytest.mark.parametrize(("state", "status"), [("reliable", "trusted"), ("not_yet", "exploration_untrusted"),
                                               ("refused", "withheld")])
def test_proposal_trust_follows_the_verdict(state, status):
    t = proposal_trust({"state": state, "reasons": [{"code": "too_few_rows", "plain": "x"}]},
                       n_returned=0 if state == "refused" else 5, n_requested=5)
    assert t["status"] == status and t["gated"] is (status != "trusted")
    assert t["withheld"] is (status == "withheld")
    assert t["reasons"] == ["too_few_rows"]
    rows = label_proposals([{"pred": 1.0}, {"pred": 2.0}], t)
    assert all(r["trust"]["status"] == status for r in rows)


def test_feasibility_gates_not_applicable_lets_promotion_pass():
    stats = {"surrogate_spearman": 0.9, "feasibility_auc": float("nan"), "ece": float("nan"),
             "brier": float("nan"), "feasibility_not_applicable": NOT_APPLICABLE_FEW_NON_PRODUCERS}
    res = check_gates(stats)
    assert res.passed and res.status == "passed"
    assert any(NOT_APPLICABLE_FEW_NON_PRODUCERS in s for s in res.not_judged)
    # without the flag the same stats fail closed, and say they were unmeasurable
    res2 = check_gates({k: v for k, v in stats.items() if k != "feasibility_not_applicable"})
    assert not res2.passed and res2.status == "not_measurable"


def _smooth(n: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m, ph = rng.uniform(0, 4, n), rng.uniform(5, 7, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + 4 + rng.normal(0, 0.1, n)).round(3)})


def test_an_ordinary_learnable_target_can_be_promoted_and_is_trusted():
    from kalos.portal.analysis import _analyze

    r = _analyze(_smooth(24), target="titer")
    assert r["reliability"]["clears_floor"] is True and r["reliability"]["judged"] is True
    assert r["promotion"]["passed"] is True, r["promotion"]["summary"]
    assert r["verdict"]["state"] == "reliable"
    assert r["proposal_trust"]["status"] == "trusted" and len(r["proposals"]) == 5
    assert all(p["trust"]["status"] == "trusted" for p in r["proposals"])
    json.dumps(r, allow_nan=False)


def test_pure_noise_is_labelled_exploration_or_withheld():
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(13)
    df = pd.DataFrame(rng.uniform(0, 1, (20, 3)).round(4), columns=["a", "b", "c"])
    df["titer"] = rng.normal(5, 1, 20).round(4)
    r = _analyze(df, target="titer")
    assert r["reliability"]["clears_floor"] is False
    assert r["proposal_trust"]["status"] in {"exploration_untrusted", "withheld"}
    if r["proposal_trust"]["withheld"]:
        assert r["proposals"] == [] and r["proposal_notice"]["code"] == "model_refused"
    else:
        assert all(p["trust"]["status"] == "exploration_untrusted" for p in r["proposals"])
        assert r["reliability"]["space_filling_advice"]


def test_small_sheet_is_not_judged():
    from kalos.portal.analysis import _analyze

    r = _analyze(_smooth(10, seed=3), target="titer")
    rel = r["reliability"]
    assert rel["clears_floor"] is False and rel["judged"] is False and "10 runs" in rel["not_judged_reason"]
    assert r["verdict"]["state"] != "reliable"


def test_cv_topk_scores_the_lowest_runs_when_minimizing():
    """cherry-picked from arch-safe-now 3935dfa (test re-written for this branch)."""
    from kalos.core.evaluation import top_k_overlap
    from kalos.portal.analysis import _analyze
    from kalos.portal.setup_settings import analysis_setup, normalize_setup

    df = _smooth(24, seed=5)
    r = _analyze(df, target="titer", setup=analysis_setup(normalize_setup({"goal": "minimize"})))
    oof = np.array(r["oof"], float)
    expect = top_k_overlap(-oof[:, 0], -oof[:, 1], r["cv_topk"]["k"])
    assert r["cv_topk"]["overlap"] == round(float(expect["overlap"]), 3)
    assert any("not applicable when minimizing" in s for s in r["promotion"]["not_judged"])
