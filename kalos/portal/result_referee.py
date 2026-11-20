"""Result-entry checks: a recorded result is checked before it is believed.

CRUX_PORT_PLAN Rank 1 ("result referee"), first release: **warn-only**. A
result that looks wrong is recorded exactly as entered, folded as usual, and
returned with a `verdict` that says what looks wrong, why, and what value it
probably should have been. Nothing is held (`held` is always False); holding a
result for review is a later product decision.

The checks are blind on purpose: they never look at the engine's prediction
for the run. A surprising but real result must never be flagged because it
disagrees with the model (that would build confirmation bias into the loop).
They look only at the value, the target column's physical bounds, the
campaign's measured history (`base_rows`), and the other results of the same
round.

Checks (stable codes; each has a why-line in `kalos/assist/why.py`):

- `result_physically_impossible`: outside the physically possible range for
  the target's kind (a negative titer, a viability above 100 %), from the
  unit-aware bounds table in `kalos/validation/bounds.py`.
- `result_magnitude_jump`: the value stands apart from every result measured
  in this campaign by the same detached-value rule the sheet check uses
  (`kalos.validation.checks.detached_from_reference`: compared with the best
  measured value, by a factor that shrinks as the history grows; 3x for
  50-99 measured producers). When a x10, x100 or x1000 slip would put it back
  inside the measured range, `suspected_value` is that value.
- `result_duplicate_entry`: exactly the same value as another result of this
  round, when the value is precise enough (3 or more significant digits, not
  0) and not a value the campaign often records (a reporting limit), so a
  coincidence is uncommon (the message says it can still be one). `suspected_value` is None: the run needs its own
  measurement, which only the lab record has.

Pure and torch-free.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kalos.validation.bounds import bounds_for_column
from kalos.validation.checks import detached_from_reference, significant_digits

MODE = "report"  # warn-only: the first release never holds a result
VALID = "valid"
IMPOSSIBLE = "result_physically_impossible"
MAGNITUDE = "result_magnitude_jump"
DUPLICATE = "result_duplicate_entry"
# A value recorded in at least this share of the measured runs is a common
# reading (0, a reporting limit), so repeating it is not evidence of a paste.
COMMON_VALUE_SHARE = 0.05
MIN_DUPLICATE_SIG_DIGITS = 3


@dataclass(frozen=True)
class SameRoundResult:
    run_id: str
    value: float
    recipe: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefereeInput:
    """What the checks may see. Deliberately no prediction, band or analysis."""

    value: float
    target: str
    measured: tuple[float, ...]  # the campaign's measured target values (base rows)
    same_round: tuple[SameRoundResult, ...] = ()
    recipe: Mapping[str, Any] = field(default_factory=dict)
    unit: str | None = None


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def _impossible(inp: RefereeInput) -> dict[str, Any] | None:
    found = bounds_for_column(inp.target)
    if found is None:
        return None
    bounds, source = found
    if bounds.hard_lo <= inp.value <= bounds.hard_hi:
        return None
    lo, hi = bounds.hard_lo, bounds.hard_hi
    rng = f"{_fmt(lo)} or more" if math.isinf(hi) else f"{_fmt(lo)} to {_fmt(hi)}"
    suspect = abs(inp.value) if inp.value < 0 and lo == 0 else None
    return {
        "code": IMPOSSIBLE,
        "message": (
            f"{_fmt(inp.value)} is not a possible {inp.target}: a {bounds.dimension.replace('_', ' ')} "
            f"value must be {rng} ({source}). Kept as entered."
        ),
        "suspected_value": suspect,
        "what_would_fix_it": (
            "Check the assay record and enter a correction with a reason"
            + (f"; if the minus sign is a typo the value is {_fmt(suspect)}." if suspect is not None else ".")
        ),
        "detail": {"hard_range": [lo, hi], "bounds_source": source},
    }


def _magnitude(inp: RefereeInput) -> dict[str, Any] | None:
    found = detached_from_reference(inp.value, inp.measured)
    if found is None:
        return None
    side = "above the highest" if found["direction"] == "high" else "below the lowest"
    suspect = found["suspected_value"]
    slip = (
        f" A x{found['slip_factor']} slip (for example mg/L typed as g/L, or a misplaced decimal) "
        f"would explain it: the value would then be {_fmt(suspect)}."
        if suspect is not None else ""
    )
    return {
        "code": MAGNITUDE,
        "message": (
            f"{_fmt(inp.value)} is {found['factor']:.1f}x {side} {inp.target} measured in this campaign "
            f"({_fmt(found['neighbour'])}, from {found['n_reference']} measured runs).{slip} "
            "A real result this far out is possible but rare; kept as entered."
        ),
        "suspected_value": suspect,
        "what_would_fix_it": (
            "Check the unit and the decimal point in the assay record. If the value is wrong, enter a "
            "correction with a reason; if it is real, nothing needs doing."
        ),
        "detail": {k: found[k] for k in ("factor", "threshold_factor", "direction", "slip_factor", "n_reference")},
    }


def _duplicate(inp: RefereeInput) -> dict[str, Any] | None:
    v = inp.value
    if v == 0 or significant_digits(v) < MIN_DUPLICATE_SIG_DIGITS:
        return None
    same = [r for r in inp.same_round if r.value == v]
    if not same:
        return None
    measured = [m for m in inp.measured if math.isfinite(m)]
    if measured and sum(1 for m in measured if m == v) >= max(2, COMMON_VALUE_SHARE * len(measured)):
        return None  # a common reading (e.g. a reporting limit), not evidence of a paste
    other = same[0]
    same_recipe = dict(other.recipe) == dict(inp.recipe) if inp.recipe else False
    what = "a replicate of the same recipe" if same_recipe else "a different recipe"
    return {
        "code": DUPLICATE,
        "message": (
            f"{_fmt(v)} is exactly the value already recorded for run {other.run_id[:8]} in this round "
            f"({what}). Either it was pasted from that run or the two results happen to match; at "
            f"{significant_digits(v)} significant digits a match is uncommon. Kept as entered."
        ),
        "suspected_value": None,
        "what_would_fix_it": (
            "Look up both runs in the assay record. If this run's value was pasted from the other, enter "
            "its own measurement as a correction with a reason."
        ),
        "detail": {"same_as_run": other.run_id, "same_recipe": same_recipe,
                   "significant_digits": significant_digits(v)},
    }


def judge(inp: RefereeInput) -> dict[str, Any]:
    """Run every check; the verdict is the first failing check's code, in the
    order impossible > magnitude > duplicate, or `valid`."""
    checks: list[dict[str, Any]] = []
    for fn in (_impossible, _magnitude, _duplicate):
        try:
            found = fn(inp)
        except Exception:  # noqa: BLE001 - a check must never block recording a result
            found = None
        if found is not None:
            checks.append(found)
    first = checks[0] if checks else None
    return {
        "verdict": first["code"] if first else VALID,
        "severity": "warning" if first else "ok",
        "reason": first["message"] if first else None,
        "suspected_value": first["suspected_value"] if first else None,
        "what_would_fix_it": first["what_would_fix_it"] if first else None,
        "checks": checks,
        "mode": MODE,
        "held": False,
    }


def measured_values(base_rows: Iterable[Mapping[str, Any]], target: str) -> tuple[float, ...]:
    """The campaign's measured target values: numeric, finite cells of `target`."""
    out: list[float] = []
    for row in base_rows:
        raw = row.get(target)
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return tuple(out)


def judge_run(
    value: float, run: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    """Judge one result for `run` against a campaign `state` (CampaignStore's
    JSON state): its base rows and the other measured runs of this round."""
    target = str(state.get("target"))
    same_round: Sequence[SameRoundResult] = tuple(
        SameRoundResult(str(r.get("id")), float(r["result"]), dict(r.get("recipe") or {}))
        for r in state.get("pending") or []
        if r.get("id") != run.get("id") and r.get("result") is not None
        and r.get("round") == run.get("round")
    )
    return judge(RefereeInput(
        value=float(value), target=target,
        measured=measured_values(state.get("base_rows") or [], target),
        same_round=tuple(same_round), recipe=dict(run.get("recipe") or {}),
    ))


__all__ = [
    "DUPLICATE", "IMPOSSIBLE", "MAGNITUDE", "MODE", "VALID", "RefereeInput", "SameRoundResult",
    "judge", "judge_run", "measured_values",
]
