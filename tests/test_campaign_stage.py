"""Pure tests for `derive_stage` (kalos/portal/campaign_stage.py): no store, no app."""
from __future__ import annotations

from typing import Any

from kalos.portal.campaign_stage import STAGES, derive_stage


def _summary(round_: int = 0, *, awaiting: int = 0, measured: int = 0) -> dict[str, Any]:
    pending = [{"awaiting": True}] * awaiting + [{"awaiting": False}] * measured
    return {
        "has_campaign": True,
        "round": round_,
        "pending": pending,
        "n_awaiting": awaiting,
        "n_measured": measured,
    }


def _states(out: dict[str, Any]) -> list[str]:
    return [s["state"] for s in out["stage"]["steps"]]


def test_no_analysis_is_setup_current():
    for summary in ({"has_campaign": False}, _summary()):
        out = derive_stage(summary, analysis_exists=False, n_proposed=None)
        assert out["stage"]["current"] == "setup"
        assert out["stage"]["round"] is None
        assert _states(out) == ["current", "upcoming", "upcoming", "upcoming", "upcoming"]
        assert out["next_action"]["stage"] == "setup"
        assert out["next_action"]["action"] == "Upload a run sheet"


def test_analysis_no_runs_is_plan_current():
    out = derive_stage(_summary(), analysis_exists=True, n_proposed=4)
    assert out["stage"]["current"] == "plan"
    assert out["stage"]["round"] == 1
    assert out["stage"]["started"] == 0
    assert _states(out) == ["done", "current", "upcoming", "upcoming", "upcoming"]
    assert out["stage"]["steps"][0]["detail"] == "Run sheet analyzed"
    assert out["stage"]["steps"][1]["detail"] == "4 recipes proposed"
    assert out["next_action"]["stage"] == "plan"
    assert out["next_action"]["action"] == "Review and start round 1"

    unknown = derive_stage(_summary(), analysis_exists=True, n_proposed=0)
    assert unknown["stage"]["steps"][1]["detail"] == "Batch ready to review"


def test_partial_results_keep_run_current():
    out = derive_stage(_summary(awaiting=2, measured=1), analysis_exists=True, n_proposed=3)
    assert out["stage"]["current"] == "run"
    assert out["stage"]["started"] == 3
    assert _states(out) == ["done", "done", "current", "upcoming", "upcoming"]
    assert out["stage"]["steps"][3]["detail"] == "1 of 3 recorded"
    assert out["next_action"] == {
        "stage": "record",
        "action": "Record 2 results",
        "reason": "1 of 3 results are in. Enter the rest as they come back from the lab.",
    }

    nothing = derive_stage(_summary(awaiting=1), analysis_exists=True, n_proposed=1)
    assert nothing["stage"]["steps"][3]["detail"] == "Nothing recorded yet"
    assert nothing["next_action"]["action"] == "Record 1 result"


def test_all_measured_is_learn_current():
    out = derive_stage(_summary(measured=3), analysis_exists=True, n_proposed=3)
    assert out["stage"]["current"] == "learn"
    assert _states(out) == ["done", "done", "done", "done", "current"]
    assert out["next_action"]["stage"] == "learn"
    assert out["next_action"]["action"] == "Learn from round 1"


def test_after_reanalyze_new_round_restarts_at_plan():
    out = derive_stage(_summary(round_=1), analysis_exists=True, n_proposed=2)
    assert out["stage"]["current"] == "plan"
    assert out["stage"]["round"] == 2
    assert out["next_action"]["action"] == "Review and start round 2"


def test_carried_over_awaiting_after_reanalyze_is_run_current():
    out = derive_stage(_summary(round_=1, awaiting=1), analysis_exists=True, n_proposed=2)
    assert out["stage"]["current"] == "run"
    assert out["stage"]["round"] == 2
    assert out["next_action"]["stage"] == "record"


def test_steps_always_five_ordered_with_one_current():
    cases = [
        ({"has_campaign": False}, False, None),
        (_summary(), True, 2),
        (_summary(awaiting=1), True, 2),
        (_summary(awaiting=1, measured=2), True, None),
        (_summary(measured=2), True, 2),
        (_summary(awaiting=1), False, None),
    ]
    for summary, exists, proposed in cases:
        out = derive_stage(summary, analysis_exists=exists, n_proposed=proposed)
        steps = out["stage"]["steps"]
        assert [s["key"] for s in steps] == list(STAGES)
        assert [s["state"] for s in steps].count("current") == 1
        assert all(isinstance(s["detail"], str) for s in steps)
        assert out["stage"]["current"] in STAGES
        assert out["next_action"]["action"] and out["next_action"]["reason"]
