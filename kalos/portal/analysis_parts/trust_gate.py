"""Trust gates that gate (F04, ML audit B3).

Before this, `reliability.clears_floor` and the space-filling advice were only
REPORTED: five proposals came back on every sheet, including pure noise, and
the floor passed on a point estimate alone (15.8% of pure-noise sheets at
n = 6-40). Now the verdict from the leak-free CV (F05) acts on the proposals:

| verdict (kalos/portal/verdict.py) | proposals | `proposal_trust.status` |
|---|---|---|
| `refused` (< 3 CV groups, no score, or the score's 95% interval at or below 0) | withheld: `proposals: []`, `refusal`, `proposal_notice.code: model_refused` | `withheld` |
| `not_yet` (anything short of reliable) | returned, each labelled exploration | `exploration_untrusted` |
| `reliable` (score >= 0.20 with interval above 0, >= 20 runs, >= 8 groups) | returned | `trusted` |

`clears_floor` itself now needs the lower end of the held-out score's 95%
interval above 0 (not only the point estimate), at least `MIN_ROWS_TO_JUDGE`
runs and at least 3 CV groups; below 12 runs it is `false` with
`judged: false` and a reason (audit N3).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Below this many fitted runs the CV verdict is not judged (audit N3: at n=6 a
# 5-fold CV on 6 points still issued a verdict and cleared on 10% of noise sheets).
MIN_ROWS_TO_JUDGE = 12
MIN_GROUPS_TO_JUDGE = 3

STATUS = {"reliable": "trusted", "not_yet": "exploration_untrusted", "refused": "withheld"}
MESSAGES = {
    "trusted": "The model ranked held-out past runs well enough to trust: these are model-guided recipes.",
    "exploration_untrusted": (
        "Exploration, not a recommendation: the model has not yet shown it can rank held-out past runs, "
        "so a space-filling design would be expected to do about as well."
    ),
    "withheld": "No recipes: the model's ranking of held-out past runs is no better than chance or cannot be checked.",
}


def clears_floor(
    *, rho: float, ci95: Sequence[float] | None, floor: float, n_runs: int, n_groups: int,
) -> dict[str, Any]:
    """`{clears_floor, judged, not_judged_reason, rule}` for `reliability`."""
    rule = (
        f"held-out Spearman >= {floor} AND the lower end of its 95% interval > 0, "
        f"judged only with at least {MIN_ROWS_TO_JUDGE} runs and {MIN_GROUPS_TO_JUDGE} CV groups"
    )
    if n_runs < MIN_ROWS_TO_JUDGE:
        return {"clears_floor": False, "judged": False, "rule": rule,
                "not_judged_reason": f"{n_runs} runs; the trust floor is judged from {MIN_ROWS_TO_JUDGE}"}
    if n_groups < MIN_GROUPS_TO_JUDGE or ci95 is None or rho != rho:
        return {"clears_floor": False, "judged": False, "rule": rule,
                "not_judged_reason": (f"{n_groups} CV group(s): no held-out score with an interval "
                                      f"(needs {MIN_GROUPS_TO_JUDGE})")}
    ok = bool(rho >= floor and float(ci95[0]) > 0)
    return {"clears_floor": ok, "judged": True, "rule": rule, "not_judged_reason": None}


def proposal_trust(verdict: Mapping[str, Any], *, n_returned: int, n_requested: int) -> dict[str, Any]:
    """The top-level `proposal_trust` block: what the verdict did to the batch."""
    state = str(verdict.get("state"))
    status = STATUS.get(state, "exploration_untrusted")
    return {
        "status": status,
        "verdict": state,
        "gated": status != "trusted",
        "withheld": status == "withheld",
        "n_requested": int(n_requested),
        "n_returned": int(n_returned),
        "reasons": [r.get("code") for r in verdict.get("reasons", []) if isinstance(r, Mapping)],
        "message": MESSAGES[status],
        "rule": "refused -> proposals withheld; not_yet -> exploration_untrusted; reliable -> trusted",
    }


def label_proposals(proposals: list[dict[str, Any]], trust: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each returned proposal carries `trust: {status, verdict, message}`."""
    tag = {"status": trust["status"], "verdict": trust["verdict"], "message": trust["message"]}
    return [{**p, "trust": dict(tag)} for p in proposals]


__all__ = ["MESSAGES", "MIN_GROUPS_TO_JUDGE", "MIN_ROWS_TO_JUDGE", "STATUS", "clears_floor",
           "label_proposals", "proposal_trust"]
