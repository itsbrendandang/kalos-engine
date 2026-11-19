"""The engine's prediction for a run, recorded when the run is started (VIZ_SPEC R5).

`CampaignStore.start` calls `prediction_at_start(...)` inside its write
transaction with the campaign's CURRENTLY PUBLISHED analysis. If the started
recipe is one of that analysis's `proposals[].recipe`, the run stores a small
server-side block with the engine's predicted value, its posterior std, the
conformal band in force at that moment, and a reference to the analysis. If the
recipe is not one of the published proposals (the user edited it, or there is
no analysis), the block says so (`available: false` + reason): the engine never
predicted that recipe, and a caller-supplied `pred` is never promoted to an
engine prediction.

Matching rule (docs/CAMPAIGN_LOOP.md, "Prediction recorded at start"): the
proposal's recipe is passed through the SAME value normalization `start()`
applies to the caller's recipe (`check_recipe_values`), then the two must have
the same keys and, per key, equal values: categorical/None values exactly; numeric
values equal after rounding both to `PROPOSAL_DECIMALS` (the precision the engine
rounds proposal coordinates to, `DesignSpace.decode_row`). The first matching
proposal wins (`proposal_index`).

`residual_fields(run)` derives `residual` (measured - predicted) and
`inside_band` at READ time from the stored block and the run's current result,
so a corrected result never leaves a stale residual behind. Nothing here is
backfilled: a run started before this block existed has no
`prediction_at_start` and reads `null` for every derived field.

Torch-free, pure functions; no I/O.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from kalos.portal.recipe_values import ColumnKind, check_recipe_values

# `DesignSpace.decode_row` rounds continuous proposal coordinates to 3 places.
PROPOSAL_DECIMALS = 3
# Per-proposal interval blocks (`intervals`, `gp_intervals`) are copied verbatim,
# but only if they are small: the campaign state is one JSON document.
MAX_COPIED_BLOCK_CHARS = 2048
COPIED_PROPOSAL_KEYS = ("intervals", "gp_intervals")

NOT_A_PROPOSAL_REASON = (
    "recipe is not one of the published proposals; the engine has no prediction recorded for it"
)
NO_ANALYSIS_REASON = (
    "the campaign has no published analysis with proposals; the engine has no prediction "
    "recorded for this recipe"
)
NO_BAND_REASON = (
    "the published analysis carries no conformal band (conformal_q/conformal_coverage absent "
    "or not finite), so no calibrated interval was recorded"
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _same_value(a: Any, b: Any) -> bool:
    na, nb = _number(a), _number(b)
    if na is not None or nb is not None:
        if na is None or nb is None:
            return False
        return round(na, PROPOSAL_DECIMALS) == round(nb, PROPOSAL_DECIMALS)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b and type(a) is type(b)


def recipes_match(started: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    """True when two NORMALIZED recipes describe the same experiment."""
    if set(started) != set(proposal):
        return False
    return all(_same_value(started[k], proposal[k]) for k in started)


def match_proposal(
    recipe: Mapping[str, Any], analysis: Mapping[str, Any] | None,
    kinds: Mapping[str, ColumnKind],
) -> tuple[int, dict[str, Any]] | None:
    """`(index, proposal)` of the first published proposal whose recipe equals
    `recipe` (already normalized by `check_recipe_values`), else None."""
    proposals = (analysis or {}).get("proposals")
    if not isinstance(proposals, list):
        return None
    for i, prop in enumerate(proposals):
        if not isinstance(prop, dict):
            continue
        prop_recipe = prop.get("recipe")
        if not isinstance(prop_recipe, dict) or not prop_recipe:
            continue
        normalized, problem = check_recipe_values(prop_recipe, kinds)
        if problem:
            continue
        if recipes_match(recipe, normalized):
            return i, prop
    return None


def _small_copy(value: Any) -> Any | None:
    """`value` if it serializes to at most MAX_COPIED_BLOCK_CHARS, else None."""
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return json.loads(text) if len(text) <= MAX_COPIED_BLOCK_CHARS else None


def _conformal_interval(pred: float, analysis: Mapping[str, Any]) -> dict[str, Any] | None:
    q = _number(analysis.get("conformal_q"))
    coverage = _number(analysis.get("conformal_coverage"))
    if q is None or coverage is None or q < 0:
        return None
    return {
        "lo": round(pred - q, 6),
        "hi": round(pred + q, 6),
        "half_width": q,
        "coverage": coverage,
        "method": "conformal_global",
    }


def prediction_at_start(
    recipe: Mapping[str, Any], analysis: Mapping[str, Any] | None,
    kinds: Mapping[str, ColumnKind], *, analysis_ref: Mapping[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    """The `prediction_at_start` block for a run started with `recipe`
    (normalized) while `analysis` is the published analysis (None = none)."""
    if not isinstance(analysis, Mapping) or not isinstance(analysis.get("proposals"), list):
        return {"available": False, "reason": NO_ANALYSIS_REASON, "recorded_at": now}
    found = match_proposal(recipe, analysis, kinds)
    if found is None:
        return {"available": False, "reason": NOT_A_PROPOSAL_REASON, "recorded_at": now}
    index, prop = found
    pred = _number(prop.get("pred"))
    if pred is None:
        return {
            "available": False,
            "reason": "the matching published proposal carries no finite predicted value",
            "recorded_at": now,
        }
    interval = _conformal_interval(pred, analysis)
    gen = analysis.get("campaign_generation")
    block: dict[str, Any] = {
        "available": True,
        "basis": "predicted",
        "pred": pred,
        # The surrogate's posterior std for this proposal: model-based, not a
        # held-out calibrated band (that is `interval`).
        "std": _number(prop.get("std")),
        "std_method": "gp_posterior",
        "interval": interval,
        "interval_unavailable_reason": None if interval is not None else NO_BAND_REASON,
        "proposal_index": index,
        "analysis_provenance": dict(analysis_ref) if analysis_ref else None,
        "analysis_generation": gen if isinstance(gen, str) else None,
        "recorded_at": now,
    }
    for key in COPIED_PROPOSAL_KEYS:
        if key in prop:
            copied = _small_copy(prop[key])
            if copied is not None:
                block[key] = copied
            else:
                block[f"{key}_omitted"] = "too large or not JSON-safe to store per run"
    return block


def residual_fields(run: Mapping[str, Any]) -> dict[str, Any]:
    """`residual`, `residual_basis` and `inside_band` for a run, derived from its
    current measured `result` and its stored engine `prediction_at_start`.
    All null when either is missing (legacy runs, edited recipes, awaiting runs)."""
    block = run.get("prediction_at_start")
    measured = _number(run.get("result"))
    if not isinstance(block, dict) or not block.get("available") or measured is None:
        return {"residual": None, "residual_basis": None, "inside_band": None}
    pred = _number(block.get("pred"))
    if pred is None:
        return {"residual": None, "residual_basis": None, "inside_band": None}
    interval = block.get("interval")
    inside: bool | None = None
    if isinstance(interval, dict):
        lo, hi = _number(interval.get("lo")), _number(interval.get("hi"))
        if lo is not None and hi is not None:
            inside = lo <= measured <= hi
    return {
        "residual": round(measured - pred, 6),
        "residual_basis": "derived",
        "inside_band": inside,
    }


def run_view(run: Mapping[str, Any]) -> dict[str, Any]:
    """A stored run as returned by the API: the stored fields, the
    prediction-at-start keys (null on legacy runs, never backfilled), and the
    derived residual fields."""
    return {
        **run,
        "prediction_at_start": run.get("prediction_at_start"),
        "pred_source": run.get("pred_source"),
        **residual_fields(run),
    }
