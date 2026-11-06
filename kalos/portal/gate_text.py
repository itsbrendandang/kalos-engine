"""Plain words for each promotion check (`promotion.checks`).

One table: what each check checks, why it matters, what to do when it falls
short or cannot run, and whether it is optional. Keys are the codes the
assistant's glossary uses (kalos/assist/why.py `MODEL_CHECKS`). `why` is the
glossary's why-line verbatim, except `ece`: the glossary line speaks of the
likely range, while this check is about the model's chance that a recipe
works, so it has its own line (sent to the glossary owner).

Feasibility checks (`feasibility_auc`, `brier`, `ece`) only matter when some
recipes can fail outright. They are optional, and the advice never asks for
failed runs on purpose.

`checks_block` is pure: it reads the `check_gates` result and the stats it was
given, and returns the list.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_FAIL_OUTRIGHT_ADVICE = (
    "Only needed if some recipes can fail outright. If they can, record those runs "
    "with a result of 0 as they happen; you do not need to run failures on purpose."
)

# code -> checks, why, advice per state (None = nothing to do), optional.
GATE_TEXT: dict[str, dict[str, Any]] = {
    "surrogate_spearman": {
        "checks": "whether the model puts runs it has not seen in the right order",
        "why": (
            "If the model cannot rank the runs it was tested on, it cannot tell a good "
            "recipe from a poor one, and its suggestions are guesses."
        ),
        "advice": {
            "failed": "Add runs of more different recipes, so the model has more to learn the order from.",
            "not_measurable": "Add runs of more different recipes, so the model can be tested on runs it has not seen.",
        },
        "optional": False,
    },
    "group_mean_margin": {
        "checks": "whether the model orders runs better than the average of the repeats of each recipe",
        "why": (
            "If the model does no better than the average of each group of runs, it has "
            "learned the groups, not the recipes, and suggests more of the same."
        ),
        "advice": {
            "failed": "Add runs of new recipes, not only repeats of recipes you already ran.",
            "not_judged": (
                "Only checked when at least {min_groups} recipes were run two or more times; "
                "nothing to do now."
            ),
        },
        "optional": False,
    },
    "feasibility_auc": {
        "checks": "whether the model can tell recipes that fail outright (target 0) from ones that work",
        "why": "If the model cannot tell which recipes fail outright, it can suggest runs that produce nothing.",
        "advice": {"failed": _FAIL_OUTRIGHT_ADVICE, "not_measurable": _FAIL_OUTRIGHT_ADVICE},
        "optional": True,
    },
    "brier": {
        "checks": "how close the model's chance that a recipe works is to what happened",
        "why": (
            "If the model's confidence that a recipe works is unreliable, runs get spent "
            "on recipes that fail."
        ),
        "advice": {"failed": _FAIL_OUTRIGHT_ADVICE, "not_measurable": _FAIL_OUTRIGHT_ADVICE},
        "optional": True,
    },
    "ece": {
        "checks": "whether the model's chance that a recipe works matches how often such recipes worked",
        "why": (
            "If the model's chance that a recipe works does not match how often it does, "
            "you cannot tell a safe bet from a long shot."
        ),
        "advice": {
            "failed": _FAIL_OUTRIGHT_ADVICE,
            "not_measurable": _FAIL_OUTRIGHT_ADVICE,
            "not_judged": (
                "Judged once the model has learned from at least {min_ece_n} runs; "
                "nothing to do now."
            ),
        },
        "optional": True,
    },
}

ORDER = ("surrogate_spearman", "group_mean_margin", "feasibility_auc", "ece", "brier")


def _state(
    code: str,
    failures: list[str],
    unmeasured: list[str],
    not_judged: list[str],
    stats: Mapping[str, Any],
) -> str:
    if code == "group_mean_margin":
        gm = stats.get("group_mean_margin")
        if not isinstance(gm, Mapping) or not gm.get("evaluable"):
            return "not_judged"
        return "passed" if gm.get("passed") else "failed"
    if code in unmeasured:
        return "not_measurable"
    if code == "ece" and any(n.startswith("ece:") for n in not_judged):
        return "not_judged"
    return "failed" if any(f.startswith(f"{code}=") for f in failures) else "passed"


def checks_block(
    *,
    failures: list[str],
    unmeasured: list[str],
    not_judged: list[str],
    stats: Mapping[str, Any],
    min_groups: int,
    min_ece_n: int,
) -> list[dict[str, Any]]:
    """`[{code, state, checks, why, advice, optional}]`, one per promotion check.

    `state` is `passed`, `failed`, `not_measurable` (the check could not be run
    on this sheet) or `not_judged` (too little data to judge; it does not block).
    `advice` is None when there is nothing to do.
    """
    out: list[dict[str, Any]] = []
    for code in ORDER:
        text = GATE_TEXT[code]
        state = _state(code, failures, unmeasured, not_judged, stats)
        advice = text["advice"].get(state)
        out.append({
            "code": code,
            "state": state,
            "checks": text["checks"],
            "why": text["why"],
            "advice": None if advice is None else advice.format(
                min_groups=min_groups, min_ece_n=min_ece_n
            ),
            "optional": bool(text["optional"]),
        })
    return out


__all__ = ["GATE_TEXT", "ORDER", "checks_block"]
