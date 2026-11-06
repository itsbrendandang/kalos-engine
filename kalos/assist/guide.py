"""The one next step for a campaign, from its record alone. No model.

Input: the campaign's loop summary (`CampaignStore.summary`), its own latest
published analysis (or None), and the latest refusal recorded for it (or
None). Output: `{state, step, why, why_it_matters, action_route, blockers,
campaign_id}` in plain words.

The states are mutually exclusive by construction: `STATES` holds one
predicate per state over `Facts`, and `classify` asserts exactly one matches.
`tests/test_assist_guide.py` enumerates every combination of the facts and
checks that, so a new rule cannot overlap an old one or leave a gap.

Precedence, in words:
1. Runs in the lab come first. While any started run has no result, record
   it. When every started run has one, learn from the round.
2. With no runs started: a refusal newer than the last analysis, else no
   sheet at all, else errors on the sheet.
3. A sheet with an analysis: a model not yet shown fit, else warnings, else
   the proposed recipes (the first round's, a later round's, or none).

Nothing here writes, approves, or states a number the record does not hold.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kalos.assist.why import WHY, why_for

ROUTE_SETUP = "/campaign/setup"
ROUTE_PLAN = "/campaign/plan"
ROUTE_RECORD = "/campaign/record"
ROUTE_LEARN = "/campaign/learn"
ROUTES = (ROUTE_SETUP, ROUTE_PLAN, ROUTE_RECORD, ROUTE_LEARN)


@dataclass(frozen=True)
class Facts:
    """What the guide reads off the record. Every field is a plain count or
    flag, so the property test can enumerate them."""

    has_sheet: bool  # the campaign holds a published analysis
    refused: bool  # a refusal is newer than that analysis (or there is none)
    n_started: int  # runs started in the current round
    n_awaiting: int  # of those, runs with no recorded result
    n_errors: int  # sheet findings at severity "error"
    n_warnings: int  # sheet findings at severity "warning"
    trusted: bool | None  # the model checks passed; None when not reported
    n_proposed: int  # recipes in the published analysis
    round: int  # re-analyses so far (0 right after an upload)


def _no_runs(f: Facts) -> bool:
    return f.n_started == 0


def _sheet_ok(f: Facts) -> bool:
    return _no_runs(f) and not f.refused and f.has_sheet and f.n_errors == 0


def _clean(f: Facts) -> bool:
    return _sheet_ok(f) and f.trusted is not False and f.n_warnings == 0


STATES: tuple[tuple[str, Callable[[Facts], bool]], ...] = (
    ("runs_awaiting", lambda f: f.n_awaiting > 0),
    ("runs_all_recorded", lambda f: f.n_started > 0 and f.n_awaiting == 0),
    ("fit_refused", lambda f: _no_runs(f) and f.refused),
    ("no_sheet", lambda f: _no_runs(f) and not f.refused and not f.has_sheet),
    ("sheet_errors", lambda f: _no_runs(f) and not f.refused and f.has_sheet and f.n_errors > 0),
    ("not_trusted", lambda f: _sheet_ok(f) and f.trusted is False),
    ("sheet_warnings", lambda f: _sheet_ok(f) and f.trusted is not False and f.n_warnings > 0),
    ("no_recipes", lambda f: _clean(f) and f.n_proposed == 0),
    ("recipes_proposed", lambda f: _clean(f) and f.n_proposed > 0 and f.round == 0),
    ("round_complete", lambda f: _clean(f) and f.n_proposed > 0 and f.round > 0),
)
STATE_NAMES = tuple(name for name, _ in STATES)
# Earlier names of two states, for clients built against them (kalos-web maps
# both). The engine emits only the new names.
LEGACY_STATE_NAMES = {"batch_proposed": "recipes_proposed", "no_batch": "no_recipes"}


def classify(facts: Facts) -> str:
    """The single state `facts` are in. Raises if zero or two match (a bug)."""
    matched = [name for name, pred in STATES if pred(facts)]
    if len(matched) != 1:
        raise AssertionError(f"guide states overlap or leave a gap: {matched} for {facts}")
    return matched[0]


# --- reading the record --------------------------------------------------------- #


def _findings(analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    validation = (analysis or {}).get("validation") or {}
    found = validation.get("findings") if isinstance(validation, dict) else None
    return [f for f in (found or []) if isinstance(f, dict)]


def finding_code(finding: dict[str, Any]) -> str:
    """A finding's stable code: `code` when the engine sends one, else `check`."""
    return str(finding.get("code") or finding.get("check") or "finding")


def _by_severity(analysis: dict[str, Any] | None, severity: str) -> list[dict[str, Any]]:
    return [f for f in _findings(analysis) if f.get("severity") == severity]


def _trusted(analysis: dict[str, Any] | None) -> bool | None:
    promotion = (analysis or {}).get("promotion")
    if not isinstance(promotion, dict):
        return None
    status = promotion.get("status")
    if status in ("failed", "not_measurable"):
        return False
    if status == "passed":
        return True
    passed = promotion.get("passed")
    return passed if isinstance(passed, bool) else None


def _analysis_time(analysis: dict[str, Any] | None) -> float | None:
    if not analysis:
        return None
    for key in ("timestamp", "saved_at", "created_at"):
        value = analysis.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _refusal_is_current(refusal: dict[str, Any] | None, analysis: dict[str, Any] | None) -> bool:
    if not refusal or not refusal.get("code"):
        return False
    if not analysis:
        return True
    at, published = refusal.get("at"), _analysis_time(analysis)
    if not isinstance(at, (int, float)) or published is None:
        return False
    return float(at) > published


def facts_from(
    summary: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    refusal: dict[str, Any] | None = None,
) -> Facts:
    """`Facts` from a campaign summary, its own analysis, and its latest refusal."""
    summary = summary or {}
    pending = [r for r in (summary.get("pending") or []) if isinstance(r, dict)]
    awaiting = sum(1 for r in pending if r.get("awaiting", r.get("result") is None))
    proposals = (analysis or {}).get("proposals") or []
    return Facts(
        has_sheet=bool(analysis),
        refused=_refusal_is_current(refusal, analysis),
        n_started=len(pending),
        n_awaiting=awaiting,
        n_errors=len(_by_severity(analysis, "error")),
        n_warnings=len(_by_severity(analysis, "warning")),
        trusted=_trusted(analysis),
        n_proposed=len(proposals) if isinstance(proposals, list) else 0,
        round=int(summary.get("round") or 0),
    )


# --- words ---------------------------------------------------------------------- #


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _refusal_text(code: str, context: dict[str, Any]) -> str:
    """One plain sentence naming what the refusal was about, with the numbers the
    engine sent (never a number it did not send)."""
    n, cap, need = context.get("n_rows"), context.get("max_fit_rows"), context.get("min_rows")
    if code == "too_many_fit_rows" and isinstance(n, int) and isinstance(cap, int):
        return f"The sheet has {n} complete runs and Kalos fits at most {cap}."
    if code in ("too_few_rows", "too_few_complete_rows") and isinstance(n, int) and isinstance(need, int):
        return f"The sheet has {_plural(n, 'usable run')} and Kalos needs at least {need}."
    if code == "too_many_rows" and isinstance(context.get("max_rows"), int):
        return f"The file has more than {context['max_rows']} rows."
    return {
        "too_many_fit_rows": "The sheet has more complete runs than Kalos fits at once.",
        "too_few_rows": "The sheet has too few runs to learn from.",
        "too_few_complete_rows": "Too few runs have every cell filled in.",
        "fit_failed": "Kalos could not build a model from these numbers.",
    }.get(code, f"Kalos did not analyse the last sheet (reason code {code}).")


_REFUSAL_FIX = {
    "too_many_fit_rows": "Split the sheet, or keep the runs you trust most, and upload again.",
    "too_few_rows": "Add more runs to the sheet and upload again.",
    "too_few_complete_rows": "Fill in the blank or unreadable cells, then upload again.",
    "target_not_found": "Choose the result column again and upload.",
    "fit_failed": "Look for repeated or extreme rows, then upload again.",
}


def _refusal_blocker(refusal: dict[str, Any]) -> dict[str, str]:
    code = str(refusal.get("code"))
    context = refusal.get("context") if isinstance(refusal.get("context"), dict) else {}
    return {"code": code, "text": _refusal_text(code, context or {}), "why": why_for(code)}


def _finding_blockers(analysis: dict[str, Any] | None, severity: str, limit: int = 8) -> list[dict[str, str]]:
    out = []
    for f in _by_severity(analysis, severity)[:limit]:
        code = finding_code(f)
        column = f.get("column")
        text = f"{'Error' if severity == 'error' else 'Warning'}: {code.replace('_', ' ')}"
        if isinstance(column, str) and column:
            text += f" in column {column[:80]!r}"
        out.append({"code": code, "text": text + ".", "why": why_for(code)})
    return out


_GATE_TEXT = {
    "surrogate_spearman": "Checked against runs it was not shown, the model did not order them well enough.",
    "feasibility_auc": "The model could not tell failing recipes from working ones well enough.",
    "ece": "The model's chance of failure did not match how often recipes failed.",
    "brier": "The model's guesses about failing recipes were wrong too often.",
    "group_mean_margin": "The model did no better than averaging repeat runs of each recipe.",
}
_GATE_UNMEASURED = {
    "surrogate_spearman": "There were too few runs to check how well the model orders them.",
    "feasibility_auc": "Every run produced something, so Kalos could not check whether the model can spot failing recipes.",
    "ece": "There were too few runs to check the model's chance of failure.",
    "brier": "There were too few runs to check the model's guesses about failing recipes.",
}


def _gate_code(failure: str) -> str:
    if failure.startswith("surrogate does not beat the group-mean floor"):
        return "group_mean_margin"
    head = failure.split("=", 1)[0].split(" ", 1)[0]
    return head if head in _GATE_TEXT else "model_check"


def _gate_blockers(analysis: dict[str, Any] | None) -> list[dict[str, str]]:
    promotion = (analysis or {}).get("promotion") or {}
    out = []
    for failure in promotion.get("failures") or []:
        if not isinstance(failure, str):
            continue
        code = _gate_code(failure)
        unmeasured = "missing" in failure or "unmeasured" in failure
        text = (_GATE_UNMEASURED if unmeasured else _GATE_TEXT).get(
            code, "One of the model checks did not pass."
        )
        out.append({"code": code, "text": text, "why": why_for("unmeasured" if unmeasured else code)})
    if not out:
        out.append({
            "code": "model_check",
            "text": "The model has not passed every check yet.",
            "why": why_for("model_check"),
        })
    return out


def _notice_blocker(analysis: dict[str, Any] | None) -> list[dict[str, str]]:
    notice = (analysis or {}).get("proposal_notice")
    if not isinstance(notice, dict):
        return []
    returned, requested = notice.get("returned"), notice.get("requested")
    text = "Fewer recipes were proposed than asked for."
    if isinstance(returned, int) and isinstance(requested, int):
        text = f"Kalos proposed {returned} of the {requested} recipes asked for."
    code = str(notice.get("code") or "fewer_distinct_candidates")
    return [{"code": code, "text": text, "why": why_for(code)}]


def next_step(
    summary: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    refusal: dict[str, Any] | None = None,
    *,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """The guide output for one campaign. Pure: same input, same output."""
    f = facts_from(summary, analysis, refusal)
    state = classify(f)
    people_round = f.round + 1
    blockers: list[dict[str, str]] = []

    if state == "runs_awaiting":
        done = f.n_started - f.n_awaiting
        step = f"Record {_plural(f.n_awaiting, 'result')} for round {people_round}"
        why = (
            f"{done} of {f.n_started} started runs have a result. "
            f"{_plural(f.n_awaiting, 'run')} still {'has' if f.n_awaiting == 1 else 'have'} none."
        )
        matters = WHY["record_partial" if done else "run_waiting"]
        route = ROUTE_RECORD
    elif state == "runs_all_recorded":
        step = f"Learn from round {people_round}"
        why = f"Every one of the {_plural(f.n_started, 'started run')} has a recorded result."
        matters = WHY["learn_ready"]
        route = ROUTE_LEARN
        if refusal and refusal.get("code") and _refusal_is_current(refusal, analysis):
            blockers.append(_refusal_blocker(refusal))
    elif state == "fit_refused":
        assert refusal is not None
        code = str(refusal.get("code"))
        blocker = _refusal_blocker(refusal)
        step = "Fix the sheet and upload it again"
        why = f"{blocker['text']} {_REFUSAL_FIX.get(code, 'The reason is below.')}"
        matters = (
            "Kalos did not build a model from the last sheet, so there are no recipes to run yet. "
            "Uploading the same sheet again gives the same result."
        )
        route = ROUTE_SETUP
        blockers.append(blocker)
    elif state == "no_sheet":
        step = "Upload a run sheet"
        why = "This campaign has no runs yet. Kalos needs the runs you already have."
        matters = WHY["setup_no_sheet"]
        route = ROUTE_SETUP
    elif state == "sheet_errors":
        step = f"Fix {_plural(f.n_errors, 'error')} in the sheet"
        why = f"Kalos found {_plural(f.n_errors, 'error')} in the sheet. They are listed below."
        matters = WHY["sheet_has_errors"]
        route = ROUTE_SETUP
        blockers.extend(_finding_blockers(analysis, "error"))
    elif state == "not_trusted":
        step = "Read why the model is not yet reliable, then decide whether to run these recipes"
        why = (
            f"Kalos proposed {_plural(f.n_proposed, 'recipe')}, but its own checks did not show "
            "the model reliable yet."
        )
        blockers.extend(_gate_blockers(analysis))
        all_unmeasured = all(b["why"] == WHY["unmeasured"] for b in blockers)
        matters = WHY["insufficient" if all_unmeasured else "not-yet"]
        route = ROUTE_PLAN
    elif state == "sheet_warnings":
        step = f"Review {_plural(f.n_warnings, 'warning')}, then start the round"
        why = (
            f"Kalos proposed {_plural(f.n_proposed, 'recipe')}. "
            f"The sheet has {_plural(f.n_warnings, 'warning')} worth a look first."
        )
        matters = WHY["sheet_usable_with_warnings"]
        route = ROUTE_PLAN
        blockers.extend(_finding_blockers(analysis, "warning"))
    elif state == "no_recipes":
        step = "Look at why no recipes were proposed"
        why = "The analysis finished, but it holds no proposed recipes."
        matters = WHY["empty_batch"]
        route = ROUTE_PLAN
        blockers.extend(_notice_blocker(analysis))
    elif state == "recipes_proposed":
        step = f"Review and start round {people_round}"
        why = f"Kalos proposed {_plural(f.n_proposed, 'recipe')}. Untick any you do not plan to run, then start the rest."
        matters = WHY["plan_ready"]
        route = ROUTE_PLAN
        blockers.extend(_notice_blocker(analysis))
    else:  # round_complete
        step = f"Round {f.round} is done. Review and start round {people_round}"
        why = (
            f"The model now includes round {f.round}. "
            f"Kalos proposed {_plural(f.n_proposed, 'recipe')} for round {people_round}."
        )
        matters = WHY["plan_ready"]
        route = ROUTE_PLAN
        blockers.extend(_notice_blocker(analysis))

    return {
        "state": state,
        "step": step,
        "why": why,
        "why_it_matters": matters,
        "action_route": route,
        "blockers": blockers,
        "campaign_id": campaign_id,
    }


def guide_text(output: dict[str, Any]) -> str:
    """The guide output as one plain paragraph (used when the assistant abstains)."""
    return f"{output['step']}. {output['why']}"


__all__ = [
    "LEGACY_STATE_NAMES",
    "ROUTES",
    "STATES",
    "STATE_NAMES",
    "Facts",
    "classify",
    "facts_from",
    "finding_code",
    "guide_text",
    "next_step",
]
