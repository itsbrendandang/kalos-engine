"""`promotion.checks`: each promotion check in plain words (kalos/portal/gate_text.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.gates import check_gates
from kalos.portal.gate_text import GATE_TEXT, ORDER, checks_block

# The assistant glossary's why-lines (kalos/assist/why.py on the assistant
# branch), which `why` must repeat word for word. `ece` is deliberately its own
# line (see gate_text's docstring).
GLOSSARY_WHY = {
    "surrogate_spearman": "If the model cannot rank the runs it was tested on, it cannot tell a good recipe from a poor one, and its suggestions are guesses.",
    "group_mean_margin": "If the model does no better than the average of each group of runs, it has learned the groups, not the recipes, and suggests more of the same.",
    "feasibility_auc": "If the model cannot tell which recipes fail outright, it can suggest runs that produce nothing.",
    "brier": "If the model's confidence that a recipe works is unreliable, runs get spent on recipes that fail.",
}
BANNED = ("spearman", "posterior", "auc", "ece", "calibration", "brier", "compliant", "validated")


def _checks(stats):
    r = check_gates(stats)
    return {
        c["code"]: c
        for c in checks_block(
            failures=r.failures, unmeasured=list(r.unmeasured), not_judged=r.not_judged,
            stats=stats, min_groups=5, min_ece_n=20,
        )
    }


def _plain(c) -> str:
    return " ".join(x for x in (c["checks"], c["why"], c["advice"] or "") if x)


def test_why_lines_match_the_glossary():
    for code, line in GLOSSARY_WHY.items():
        assert GATE_TEXT[code]["why"] == line
    assert set(ORDER) == set(GATE_TEXT)


def test_states_and_texts():
    all_producer = _checks({
        "surrogate_spearman": 0.6, "feasibility_auc": float("nan"), "ece": float("nan"),
        "brier": float("nan"), "ece_n": None,
        "group_mean_margin": {"evaluable": False, "passed": None, "reason": "x"},
    })
    assert list(all_producer) == list(ORDER)
    assert all_producer["surrogate_spearman"]["state"] == "passed"
    assert all_producer["surrogate_spearman"]["advice"] is None
    assert all_producer["surrogate_spearman"]["optional"] is False
    feas = all_producer["feasibility_auc"]
    assert feas["state"] == "not_measurable" and feas["optional"] is True
    assert feas["checks"] == (
        "whether the model can tell recipes that fail outright (target 0) from ones that work"
    )
    assert "you do not need to run failures on purpose" in feas["advice"]
    assert feas["advice"].startswith("Only needed if some recipes can fail outright")
    assert all_producer["brier"]["state"] == "not_measurable"
    assert all_producer["ece"]["state"] == "not_measurable"
    assert all_producer["group_mean_margin"]["state"] == "not_judged"
    assert "5 recipes" in all_producer["group_mean_margin"]["advice"]

    mixed = _checks({
        "surrogate_spearman": 0.1, "feasibility_auc": 0.5, "ece": 0.05, "brier": 0.3,
        "ece_n": 12, "group_mean_margin": {"evaluable": True, "passed": False, "ci95": [-0.1, 0.2]},
    })
    assert mixed["surrogate_spearman"]["state"] == "failed"
    assert mixed["feasibility_auc"]["state"] == "failed"
    assert mixed["brier"]["state"] == "failed"
    assert mixed["ece"]["state"] == "not_judged" and "20 runs" in mixed["ece"]["advice"]
    assert mixed["group_mean_margin"]["state"] == "failed"

    for checks in (all_producer, mixed):
        for c in checks.values():
            text = _plain(c).lower()
            assert not any(w in text for w in BANNED), (c["code"], text)
            assert "failed or produced nothing" not in text
            assert "include some that failed" not in text


def test_analysis_carries_checks_and_keeps_the_old_fields():
    pytest.importorskip("fastapi")
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(3)
    a, b = rng.uniform(0, 10, 13), rng.uniform(0, 10, 13)
    df = pd.DataFrame({"a": a.round(3), "b": b.round(3),
                       "y": (5.0 + 2.0 * a + 1.5 * b + rng.normal(0, 0.3, 13)).round(3)})
    promo = _analyze(df)["promotion"]
    assert {"passed", "failures", "metrics", "summary", "not_judged", "status", "unmeasured",
            "meaning"} <= set(promo)
    checks = {c["code"]: c for c in promo["checks"]}
    assert set(checks) == set(ORDER)
    assert checks["feasibility_auc"]["state"] == "not_measurable"  # every run produced
    assert "feasibility_auc" in promo["unmeasured"]
    for c in checks.values():
        assert set(c) == {"code", "state", "checks", "why", "advice", "optional"}
        assert c["state"] in {"passed", "failed", "not_measurable", "not_judged"}
