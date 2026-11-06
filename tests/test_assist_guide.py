"""The guide (kalos/assist/guide.py): every reachable state yields exactly one step.

The property test enumerates every combination of the facts the guide reads
(an exhaustive grid, not a random sample), builds a matching record, and checks
that exactly one state predicate matches, that `next_step` returns that state,
and that the output is well formed and in plain words.
"""
from __future__ import annotations

import itertools

import pytest

from kalos.assist.guide import (
    ROUTES,
    STATE_NAMES,
    STATES,
    Facts,
    classify,
    facts_from,
    guide_text,
    next_step,
)
from kalos.assist.words import banned_in, claims_in

GRID = {
    "has_sheet": (False, True),
    "refused": (False, True),
    "n_started": (0, 1, 3),
    "n_awaiting": (0, 1, 3),
    "n_errors": (0, 2),
    "n_warnings": (0, 1),
    "trusted": (None, False, True),
    "n_proposed": (0, 4),
    "round": (0, 2),
}


def _all_facts():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        facts = Facts(**dict(zip(keys, values)))
        if facts.n_awaiting > facts.n_started:
            continue  # not reachable: awaiting runs are a subset of started runs
        if not facts.has_sheet and (facts.n_errors or facts.n_warnings or facts.n_proposed
                                    or facts.trusted is not None):
            continue  # not reachable: findings, checks and a batch come from an analysis
        yield facts


def _record(f: Facts) -> tuple[dict, dict | None, dict | None]:
    """A (summary, analysis, refusal) triple whose facts are exactly `f`."""
    pending = [
        {"id": f"{i:032x}", "recipe": {"x": 1.0}, "result": None if i < f.n_awaiting else 2.0,
         "awaiting": i < f.n_awaiting}
        for i in range(f.n_started)
    ]
    summary = {"has_campaign": True, "round": f.round, "pending": pending, "target": "titer"}
    analysis = None
    if f.has_sheet:
        findings = (
            [{"check": "physical_bounds", "severity": "error", "message": "m", "column": "pH"}] * f.n_errors
            + [{"check": "outliers", "severity": "warning", "message": "m", "column": None}] * f.n_warnings
            + [{"check": "controls_present", "severity": "info", "message": "m"}]
        )
        analysis = {
            "target": "titer",
            "timestamp": 1000,
            "validation": {"status": "fail" if f.n_errors else "pass", "findings": findings},
            "proposals": [{"recipe": {"x": 0.5}}] * f.n_proposed,
        }
        if f.trusted is not None:
            analysis["promotion"] = {
                "passed": f.trusted,
                "failures": [] if f.trusted else ["surrogate_spearman=0.100 < min 0.3",
                                                  "feasibility_auc missing/NaN (unmeasured)"],
            }
    refusal = None
    if f.refused:
        refusal = {"code": "too_many_fit_rows", "context": {"n_rows": 2400, "max_fit_rows": 2000},
                   "at": 2000.0}
    return summary, analysis, refusal


def test_every_reachable_state_yields_exactly_one_step():
    seen: set[str] = set()
    n = 0
    for facts in _all_facts():
        matched = [name for name, pred in STATES if pred(facts)]
        assert len(matched) == 1, (facts, matched)
        summary, analysis, refusal = _record(facts)
        assert facts_from(summary, analysis, refusal) == facts
        out = next_step(summary, analysis, refusal, campaign_id="cmp_" + "0" * 32)
        assert out["state"] == matched[0] == classify(facts)
        assert set(out) == {"state", "step", "why", "why_it_matters", "action_route", "blockers",
                            "campaign_id"}
        assert out["action_route"] in ROUTES
        for key in ("step", "why", "why_it_matters"):
            assert isinstance(out[key], str) and out[key].strip()
        for b in out["blockers"]:
            assert set(b) == {"code", "text", "why"} and all(isinstance(v, str) and v for v in b.values())
        # What the guide composes itself carries no claim word; the why-lines are
        # the glossary's, checked in test_assist_why.py.
        composed = " ".join([out["step"], out["why"]] + [b["text"] for b in out["blockers"]])
        assert claims_in(composed) == [], (out["state"], composed)
        everything = composed + " " + out["why_it_matters"] + " ".join(b["why"] for b in out["blockers"])
        assert banned_in(everything) == [], (out["state"], everything)
        seen.add(out["state"])
        n += 1
    assert seen == set(STATE_NAMES), set(STATE_NAMES) - seen
    assert n > 500


def test_routes_per_state():
    expected = {
        "runs_awaiting": "/campaign/record", "runs_all_recorded": "/campaign/learn",
        "fit_refused": "/campaign/setup", "no_sheet": "/campaign/setup",
        "sheet_errors": "/campaign/setup", "not_trusted": "/campaign/plan",
        "sheet_warnings": "/campaign/plan", "no_batch": "/campaign/plan",
        "batch_proposed": "/campaign/plan", "round_complete": "/campaign/plan",
    }
    for facts in _all_facts():
        out = next_step(*_record(facts))
        assert out["action_route"] == expected[out["state"]]


def test_no_campaign_at_all_is_no_sheet():
    out = next_step({"has_campaign": False}, None, None)
    assert out["state"] == "no_sheet"
    assert out["action_route"] == "/campaign/setup"
    assert guide_text(out).startswith("Upload a run sheet.")


def test_fit_row_cap_refusal_names_the_engine_numbers():
    out = next_step({"has_campaign": True, "pending": []}, None,
                    {"code": "too_many_fit_rows", "context": {"n_rows": 2400, "max_fit_rows": 2000}, "at": 5})
    assert out["state"] == "fit_refused"
    assert "2400" in out["why"] and "2000" in out["why"]
    assert out["blockers"][0]["code"] == "too_many_fit_rows"


@pytest.mark.parametrize("code", ["too_few_complete_rows", "fit_failed", "target_not_found", "something_new"])
def test_refusal_codes_give_a_plain_blocker(code):
    out = next_step(None, None, {"code": code, "context": {}, "at": 1})
    assert out["state"] == "fit_refused"
    assert out["blockers"][0]["code"] == code
    assert out["blockers"][0]["why"]


def test_old_refusal_is_ignored_once_a_newer_analysis_exists():
    analysis = {"timestamp": 500, "proposals": [{}], "validation": {"findings": []}}
    out = next_step({"pending": []}, analysis, {"code": "fit_failed", "at": 100})
    assert out["state"] == "batch_proposed"


def test_numbers_come_from_the_record():
    summary = {"round": 1, "pending": [{"result": None}, {"result": 3.0}, {"result": None}]}
    out = next_step(summary, {"timestamp": 1, "proposals": []})
    assert out["state"] == "runs_awaiting"
    assert out["step"] == "Record 2 results for round 2"
    assert out["why"].startswith("1 of 3 started runs have a result.")


def test_desktop_promotion_status_is_read():
    analysis = {"timestamp": 1, "proposals": [{}], "promotion": {"status": "not_measurable", "failures": []}}
    out = next_step({"pending": []}, analysis)
    assert out["state"] == "not_trusted"
    assert out["blockers"][0]["code"] == "model_check"
