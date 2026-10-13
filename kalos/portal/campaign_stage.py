"""Kalos portal - where a campaign is in its round, and the one next action.

Pure and torch-free: derived from a `CampaignStore.summary()` dict plus two
facts about the analysis on `/api/latest`, so the rules are unit testable with no
store. The wording mirrors kalos-web's `lib/campaign-stage.ts` exactly.

Rounds: the store counts re-analyses (`round`, 0 right after an upload). People
count batches, so `stage.round` is `round + 1`.

The engine cannot tell a run still in the lab from one that is back but not yet
entered: both are "awaiting". So while any run awaits, the round is in `run` and
the next action is `record`; `record` and `learn` only complete once every
started run has a result.
"""
from __future__ import annotations

from typing import Any

STAGES = ("setup", "plan", "run", "record", "learn")


def _plural(n: int, one: str) -> str:
    return f"{n} {one if n == 1 else one + 's'}"


def derive_stage(
    summary: dict[str, Any], *, analysis_exists: bool, n_proposed: int | None
) -> dict[str, Any]:
    """`{"stage": {...}, "next_action": {...}}` for a campaign summary.

    `analysis_exists` says whether `/api/latest` holds an analysis; `n_proposed`
    is how many recipes it proposed (None when there is no analysis).
    """
    started = len(summary.get("pending", []))
    awaiting = summary.get("n_awaiting", 0)
    measured = summary.get("n_measured", 0)

    def result(
        current: str,
        details: dict[str, str],
        action: str,
        reason: str,
        round_: int | None,
        next_stage: str | None = None,
    ) -> dict[str, Any]:
        at = STAGES.index(current)
        steps = [
            {
                "key": key,
                "state": "done" if i < at else "current" if i == at else "upcoming",
                "detail": details.get(key, ""),
            }
            for i, key in enumerate(STAGES)
        ]
        return {
            "stage": {"round": round_, "current": current, "started": started, "steps": steps},
            "next_action": {"stage": next_stage or current, "action": action, "reason": reason},
        }

    if not analysis_exists and started == 0:
        return result(
            "setup",
            {"setup": "No run sheet yet"},
            "Upload a run sheet",
            "Kalos fits a model to the runs you already have, then proposes the first batch.",
            None,
        )

    round_ = summary.get("round", 0) + 1
    setup = "Run sheet analyzed" if analysis_exists else "Campaign seeded"

    if started == 0:
        proposed = f"{_plural(n_proposed, 'recipe')} proposed" if n_proposed else ""
        return result(
            "plan",
            {"setup": setup, "plan": proposed or "Batch ready to review"},
            f"Review and start round {round_}",
            f"The model proposed {_plural(n_proposed, 'recipe')}. Untick any you will not run, then start the rest."
            if n_proposed
            else "Review the proposed batch, untick any recipe you will not run, then start the rest.",
            round_,
        )

    details = {"setup": setup, "plan": f"{_plural(started, 'run')} started"}
    if awaiting > 0:
        details["run"] = f"{awaiting} awaiting results"
        details["record"] = f"{measured} of {started} recorded" if measured else "Nothing recorded yet"
        return result(
            "run",
            details,
            f"Record {_plural(awaiting, 'result')}",
            f"{measured} of {started} results are in. Enter the rest as they come back from the lab."
            if measured
            else f"{_plural(awaiting, 'run')} from round {round_} {'is' if awaiting == 1 else 'are'} waiting for a measured result.",
            round_,
            next_stage="record",
        )

    details["run"] = f"All {started} back"
    details["record"] = f"{_plural(measured, 'result')} recorded"
    details["learn"] = "Ready to update the model"
    return result(
        "learn",
        details,
        f"Learn from round {round_}",
        f"Every run in round {round_} has a result. See what changed, then plan round {round_ + 1}.",
        round_,
    )
