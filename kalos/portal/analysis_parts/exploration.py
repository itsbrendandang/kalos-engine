"""When the batch is an exploration design instead of the model's picks
(QUEUE round 3 item 6).

The batch acquisition trusts the model: it spends runs where the model expects
gains or is unsure. When the model cannot rank recipes yet, or the sheet has
fewer distinct recipes than about twice the number of inputs, those picks are
little better than guesses. The batch is then the model's best bet (kept, and
labelled) plus recipes spread over the untested parts of the search box
(`kalos.core.space_filling.maximin_batch`), and the analysis says so.

The rule (`roles.exploration_design`, "auto" by default, "off" to always use
the model's picks):
  - fewer than 2 x (number of inputs) distinct recipes -> `few_recipes`;
  - the verdict is `not_yet` because the model's ordering is weak or not yet
    better than chance (`verdict.ordering.state` weak or unsure) ->
    `cannot_rank_yet`;
  - otherwise, or when the verdict is `refused` (no recipes at all), the
    batch is the model's.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RULE = (
    "Exploration design when the sheet has fewer than 2 x inputs distinct recipes, or "
    "when the model cannot yet rank recipes (verdict not yet, ordering weak or unsure). "
    "Off with roles.exploration_design = \"off\"."
)

PLAIN = {
    "few_recipes": (
        "The sheet has {n} different recipes for {d} inputs, too few for the model to "
        "map the range. Apart from the best bet, this round spreads recipes over the "
        "parts of the range nobody has tried."
    ),
    "cannot_rank_yet": (
        "The model cannot yet rank recipes reliably. Apart from the best bet, this "
        "round spreads recipes over the parts of the range nobody has tried, which "
        "teaches the model more than its own guesses would."
    ),
}


def exploration_reason(
    verdict: Mapping[str, Any] | None, *, n_distinct: int, n_inputs: int, setting: str | None,
) -> str | None:
    """`few_recipes`, `cannot_rank_yet`, or None (the model's batch)."""
    if (setting or "auto") == "off":
        return None
    if verdict is not None and verdict.get("state") == "refused":
        return None
    if n_distinct < 2 * max(int(n_inputs), 1):
        return "few_recipes"
    ordering = (verdict or {}).get("ordering") or {}
    if (verdict or {}).get("state") == "not_yet" and ordering.get("state") in {"weak", "unsure"}:
        return "cannot_rank_yet"
    return None


def batch_strategy(reason: str | None, *, n_distinct: int, n_inputs: int) -> dict[str, Any]:
    return {
        "kind": "exploration_design" if reason else "model",
        "reason": reason,
        "plain": PLAIN[reason].format(n=n_distinct, d=n_inputs) if reason else (
            "The recipes are the model's picks: where it expects gains or is least sure."
        ),
        "n_distinct_recipes": int(n_distinct),
        "n_inputs": int(n_inputs),
        "rule": RULE,
    }


EXPLORE_REASON = "spreads runs over a part of the range nobody has tried (exploration design)"

__all__ = ["EXPLORE_REASON", "PLAIN", "RULE", "batch_strategy", "exploration_reason"]
