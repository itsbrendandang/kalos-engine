"""One verdict on the model, so no page says "Reliable" next to a failure.

`model_verdict` is pure: it reads numbers the analysis already computed (the
held-out rank agreement, its 95% interval, how many runs and CV groups, the
interval calibration) and returns one dict. The analysis stores it as
`verdict`, and every route that describes the analysis or the campaign returns
that stored object; nothing recomputes it.

THE TABLE (docs/CAMPAIGNS_API.md, "Verdict"), checked top to bottom:

| state     | code          | rule                                                        |
|-----------|---------------|-------------------------------------------------------------|
| refused   | insufficient  | no held-out score, or fewer than MIN_GROUPS_TO_PROPOSE CV    |
|           |               | groups (the ordering cannot be checked at all)              |
| refused   | no_ordering   | upper end of the 95% interval of the held-out score <= 0    |
| reliable  | reliable      | score >= floor, lower end of the interval > 0, and at least |
|           |               | MIN_RUNS_FOR_RELIABLE runs and MIN_GROUPS_FOR_RELIABLE      |
|           |               | CV groups                                                   |
| not_yet   | not-yet       | everything else                                             |

Codes reuse the plain-words glossary (`reliable`, `not-yet`, `insufficient`,
`surrogate_spearman`, `too_few_rows`, `bands_*`); `no_ordering` and
`too_few_groups` are new and were sent to the glossary owner.

The range-width part mirrors kalos-web's `calibrationVerdict` (z_std above
1.15 too narrow, below 0.85 too wide) so both say the same thing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Why 20 runs and 8 groups before "reliable". The held-out rank agreement is a
# Spearman over the pooled out-of-fold runs. At 13 runs a score of 0.5 has a
# 95% interval of roughly -0.06 to 0.82, so one lucky fold can carry it; at 20
# the same score's interval starts near 0.07. 20 is also the run count at which
# the engine starts judging calibration (GatesConfig.min_ece_n,
# CALIBRATION_JUDGE_MIN_N), so "reliable" never rests on a range check that was
# not judged. 8 CV groups gives the 5-fold grouped CV at least one group per
# fold with some left over, and gives the group bootstrap behind the interval
# enough distinct groups that its resamples are not a handful of repeats.
MIN_RUNS_FOR_RELIABLE = 20
MIN_GROUPS_FOR_RELIABLE = 8
# Below 3 CV groups the group bootstrap cannot form an interval, so there is no
# way to say whether the ordering is better than chance: no proposals.
MIN_GROUPS_TO_PROPOSE = 3
# Same thresholds as kalos-web components/campaign/data/model-math.ts.
Z_STD_TOO_NARROW = 1.15
Z_STD_TOO_WIDE = 0.85

HEADLINES = {
    "reliable": (
        "Kalos put your past runs in the right order, with enough runs to check the "
        "ranking, so its suggestions are worth running."
    ),
    "not-yet": (
        "Kalos cannot yet show that it puts your past runs in the right order, "
        "so its suggestions are closer to guesses."
    ),
    "insufficient": (
        "Kalos cannot suggest recipes from this sheet: there are too few groups of "
        "runs to check whether the model is any good."
    ),
    "no_ordering": (
        "Kalos cannot suggest recipes from this sheet: the model does no better than "
        "chance at putting your past runs in order."
    ),
}

HELP = {
    "insufficient": (
        "Add measured runs of at least {need} different recipes, then upload the sheet again."
    ),
    "no_ordering": (
        "Check that the target is the result you care about and that the sheet holds "
        "the settings that changed between runs. Then add runs of different recipes "
        "and upload again."
    ),
}

RULE = (
    f"Refused: no held-out score, fewer than {MIN_GROUPS_TO_PROPOSE} groups of runs, or "
    "the upper end of the score's 95% interval at or below 0. Reliable: score at or above "
    "the floor, the lower end of its 95% interval above 0, at least "
    f"{MIN_RUNS_FOR_RELIABLE} runs and {MIN_GROUPS_FOR_RELIABLE} groups of runs. "
    "Not yet: everything else."
)


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if x == x and abs(float(x)) != float("inf") else None


def range_width(calibration: Mapping[str, Any] | None) -> dict[str, str]:
    """`{state, code, plain}` for the likely range: ok, too_narrow, too_wide or unchecked."""
    cal = calibration or {}
    z = _num(cal.get("z_std"))
    if not cal.get("available") or not cal.get("levels") or z is None:
        return {"state": "unchecked", "code": "bands_unchecked",
                "plain": "The likely range has not been checked on measured runs yet."}
    if z > Z_STD_TOO_NARROW:
        return {"state": "too_narrow", "code": "bands_narrow",
                "plain": "The likely range is too narrow: fewer results land inside it than it promises."}
    if z < Z_STD_TOO_WIDE:
        return {"state": "too_wide", "code": "bands_wide",
                "plain": "The likely range is wider than it needs to be."}
    return {"state": "ok", "code": "bands_ok", "plain": "The likely range is about the right width."}


def model_verdict(
    *,
    n_runs: int,
    n_groups: int,
    spearman: float | None,
    ci95: Sequence[float] | None,
    floor: float,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The one verdict on the model (see the module docstring for the table).

    `spearman`/`ci95` are the held-out rank agreement and its 95% interval
    (`reliability.spearman` / `reliability.ci95`), `floor` the reliability floor,
    `n_runs` the runs the model learned from, `n_groups` the CV groups.
    """
    rho = _num(spearman)
    lo = _num(ci95[0]) if ci95 is not None and len(ci95) == 2 else None
    hi = _num(ci95[1]) if ci95 is not None and len(ci95) == 2 else None
    n_runs, n_groups = int(n_runs), int(n_groups)
    reasons: list[dict[str, str]] = []

    # Ordering, in plain words.
    if rho is None:
        ordering = {"state": "unchecked",
                    "plain": "Kalos could not check whether the model puts past runs in the right order."}
    elif lo is None or hi is None:
        ordering = {"state": "unchecked",
                    "plain": "There are too few groups of runs to check how sure the ordering is."}
    elif hi <= 0:
        ordering = {"state": "absent",
                    "plain": "The model does no better than chance at putting past runs in order."}
    elif rho >= floor and lo > 0:
        ordering = {"state": "ok",
                    "plain": "The model put past runs it had not seen in the right order most of the time."}
    elif rho >= floor:
        ordering = {"state": "unsure",
                    "plain": "The model put past runs in roughly the right order, but Kalos is not sure "
                             "that is better than chance."}
    else:
        ordering = {"state": "weak",
                    "plain": "The model put past runs it had not seen in the right order too rarely."}

    if rho is None or n_groups < MIN_GROUPS_TO_PROPOSE or lo is None or hi is None:
        state, code = "refused", "insufficient"
        if rho is None and n_groups >= MIN_GROUPS_TO_PROPOSE:
            reasons.append({"code": "cv_ci_unavailable", "plain": ordering["plain"]})
        else:
            reasons.append({"code": "too_few_groups",
                            "plain": f"The sheet has {n_groups} group"
                                     f"{'' if n_groups == 1 else 's'} of runs; Kalos needs at least "
                                     f"{MIN_GROUPS_TO_PROPOSE} to check the model."})
    elif hi <= 0:
        state, code = "refused", "no_ordering"
        reasons.append({"code": "surrogate_spearman", "plain": ordering["plain"]})
    else:
        if ordering["state"] != "ok":
            reasons.append({"code": "surrogate_spearman", "plain": ordering["plain"]})
        if n_runs < MIN_RUNS_FOR_RELIABLE:
            reasons.append({"code": "too_few_rows",
                            "plain": f"The model learned from {n_runs} runs; Kalos needs at least "
                                     f"{MIN_RUNS_FOR_RELIABLE} before it calls the model reliable."})
        if n_groups < MIN_GROUPS_FOR_RELIABLE:
            reasons.append({"code": "too_few_groups",
                            "plain": f"The sheet has {n_groups} groups of runs; Kalos needs at least "
                                     f"{MIN_GROUPS_FOR_RELIABLE} before it calls the model reliable."})
        state, code = ("reliable", "reliable") if not reasons else ("not_yet", "not-yet")

    headline = HEADLINES[code]
    if state == "not_yet" and ordering["state"] == "ok":
        # Only the size of the sheet holds it back: say that, not "cannot rank".
        headline = (
            f"The model put your past runs in the right order, but {n_runs} runs in "
            f"{n_groups} groups are too few to be sure, so treat its "
            "suggestions as ideas to test."
        )
    out: dict[str, Any] = {
        "state": state,
        "code": code,
        "headline": headline,
        "reasons": reasons,
        "ordering": ordering,
        "range_width": range_width(calibration),
        "n_runs": n_runs,
        "min_runs": MIN_RUNS_FOR_RELIABLE,
        "n_groups": n_groups,
        "min_groups": MIN_GROUPS_FOR_RELIABLE,
        "rule": RULE,
    }
    return out


def hold_for_flagged_drivers(
    verdict: Mapping[str, Any], flagged: Sequence[str], top_inputs: Sequence[str],
) -> dict[str, Any]:
    """Hold a `reliable` verdict at `not_yet` when one of the inputs the model
    leans on most (`top_inputs`: the largest share or the top two) is an input
    flagged `possibly_measured_after_run`: a result used as a setting makes the
    model look better than it is (security review S1; a `VCD_day14` column
    tracking the target gave "reliable" at 0.995). Adds a plain reason to a
    `not_yet` verdict; a `refused` verdict is unchanged."""
    hits = [c for c in top_inputs if c in set(flagged)]
    if not hits or verdict.get("state") == "refused":
        return dict(verdict)
    names = " and ".join(hits)
    plain = (
        f"The model leans most on {names}, which may be measured during the run rather "
        "than set. Until you confirm it is a setting, Kalos does not call the model reliable."
    )
    out = dict(verdict)
    out["reasons"] = [*verdict.get("reasons", []), {"code": "possibly_measured_driver", "plain": plain}]
    if verdict.get("state") == "reliable":
        out["state"], out["code"], out["headline"] = "not_yet", "not-yet", plain
    return out


def refusal_for(verdict: Mapping[str, Any]) -> dict[str, str] | None:
    """`{code, plain, what_would_help}` when the verdict is `refused`, else None."""
    if verdict.get("state") != "refused":
        return None
    code = str(verdict["code"])
    return {
        "code": code,
        "plain": str(verdict["headline"]),
        "what_would_help": HELP[code].format(need=MIN_GROUPS_FOR_RELIABLE),
    }


def refusal_notice(refusal: Mapping[str, str], requested: int) -> dict[str, Any]:
    """The `proposal_notice` for a refused analysis: no recipes, and why."""
    return {
        "code": "model_refused",
        "reason": refusal["code"],
        "requested": int(requested),
        "returned": 0,
        "message": f"No recipes this round. {refusal['plain']} {refusal['what_would_help']}",
    }


__all__ = [
    "MIN_GROUPS_FOR_RELIABLE", "MIN_GROUPS_TO_PROPOSE", "MIN_RUNS_FOR_RELIABLE",
    "hold_for_flagged_drivers", "model_verdict", "range_width", "refusal_for", "refusal_notice",
]
