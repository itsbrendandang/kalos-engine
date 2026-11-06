"""The search box versus the tried range (first-time-user study, Finding 11).

Kalos searches each continuous input inside the range the sheet has tried. A
scientist whose best runs sit at the top of a range ("higher molasses went with
higher biomass, best runs at 120") cannot get a proposal at 130 and was never
told why. Two changes:

- `roles.allow_beyond` (or the `allow_beyond` analysis argument) lets the search
  go a fraction of the tried range past each end, per input or for every input
  ("*"). Default 0: the box is the tried range, as before. Hard physical bounds
  still apply (`ranges.widen_box`).
- Every analysis carries one plain finding, code `search_box`, saying what the
  box is and how to widen it; each proposal lists the inputs it sets outside the
  tried range (`beyond_tried_range`).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kalos.validation.report import Finding

MAX_LISTED = 6


def _allow_beyond_map(roles: Any, allow_beyond: float | Mapping[str, float] | None) -> dict[str, float]:
    """`{input or "*": fraction}`. The argument wins over roles; a plain number
    is the "*" default. Values are checked by `parse_allow_beyond`."""
    from kalos.domains.profile import parse_allow_beyond

    if allow_beyond is not None:
        return dict(parse_allow_beyond(dict(allow_beyond) if isinstance(allow_beyond, Mapping) else allow_beyond))
    pairs = tuple(getattr(roles, "allow_beyond", None) or ()) if roles is not None else ()
    return dict(pairs)


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def beyond_tried(recipe: Mapping[str, Any], tried: Mapping[str, list[float]]) -> list[dict[str, Any]]:
    """The inputs a proposed recipe sets outside their tried range."""
    out: list[dict[str, Any]] = []
    for name, (lo, hi) in tried.items():
        v = recipe.get(name)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        width = hi - lo
        if v > hi:
            side, limit, by = "upper", hi, v - hi
        elif v < lo:
            side, limit, by = "lower", lo, lo - v
        else:
            continue
        out.append({
            "feature": name, "side": side, "tried_limit": round(float(limit), 6),
            "proposed": float(v), "by": round(float(by), 6),
            "fraction": round(float(by / width), 4) if width > 0 else None,
        })
    return out


def search_box_finding(
    tried: Mapping[str, list[float]], applied: Mapping[str, float], design: Any,
) -> Finding:
    """One info finding that says, in plain words, where Kalos searches."""
    box = {d.name: (float(d.lower), float(d.upper)) for d in design.dims if not d.is_categorical}
    names = list(tried)
    shown = names[:MAX_LISTED]
    more = len(names) - len(shown)
    tail = f", and {more} more" if more > 0 else ""
    if not applied:
        listed = "; ".join(f"{n} {_fmt(tried[n][0])} to {_fmt(tried[n][1])}" for n in shown)
        message = (
            "Kalos only searches inside the range you have tried for each input "
            f"({listed}{tail}), so it never proposes a value outside it. To let it look "
            "further, set allow_beyond in the column roles: 0.10 lets it go 10% of the "
            "tried range past each end."
        )
    else:
        parts = []
        for n in shown:
            if n in applied:
                lo, hi = box.get(n, tried[n])
                parts.append(
                    f"{n} {_fmt(lo)} to {_fmt(hi)} (tried {_fmt(tried[n][0])} to "
                    f"{_fmt(tried[n][1])}, {round(applied[n] * 100):g}% past each end)"
                )
            else:
                parts.append(f"{n} {_fmt(tried[n][0])} to {_fmt(tried[n][1])} (tried range only)")
        message = (
            f"Kalos searches past what you have tried where you allowed it: {'; '.join(parts)}"
            f"{tail}. Outside the tried range the model has no runs to go on, so a recipe "
            "there is flagged and its prediction is a guess."
        )
    return Finding(
        check="search_box", code="search_box", severity="info", message=message,
        detail={
            "tried_ranges": {n: list(v) for n, v in tried.items()},
            "search_box": {n: list(v) for n, v in box.items()},
            "allow_beyond": dict(applied),
        },
    )


__all__ = ["beyond_tried", "search_box_finding"]
