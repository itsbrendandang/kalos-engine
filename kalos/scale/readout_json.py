"""JSON shape of the Scale-Up Readout (engine request R1).

`readout_to_json` turns `kalos.scale.readout.build_readout`'s dict into the
versioned, chartable payload `POST /api/scale/readout?format=json` (and
`/api/scale/readout.json`) returns. It only RESHAPES what `build_readout`
computed - every number comes from the engine; nothing here estimates,
fills, or guesses a value. Non-finite floats (e.g. a pooled MAE with no
residuals, a nearest-recipe baseline with no finite match) become `null`
rather than a fake number.

`compute_readout_id` is a deterministic SHA-256 over the raw uploaded
sheet, the canonicalized target, and the engine version, so a gate decision
can cite the exact evidence it was made on without the engine storing
anything (D6: the readout stays stateless).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from kalos import __version__ as ENGINE_VERSION

from .readout import (
    ALPHA,
    BACKTEST_METHOD,
    HARD_CAP_RATIO,
    REFUSE_RATIO_MULT,
    WARN_RATIO_MULT,
    TargetSpec,
)

KIND = "scale_readout"
SCHEMA_VERSION = 1
INTERVAL_METHOD = "split_conformal_ladder_residuals"
_READOUT_ID_DOMAIN = "kalos.scale_readout.v1"

# Honest wording for the band. The coverage figure is the NOMINAL level
# implied by ALPHA; it is not a measured hit rate, and exchangeability does
# not hold across a scale the ladder never tested.
INTERVAL_LABEL = (
    "approximate {pct:g}% coverage (nominal, from alpha={alpha:g}): split-conformal band on the pooled "
    "ladder backtest residuals; not calibrated at the target scale, because exchangeability does not "
    "hold across an untested scale"
)
INTERVAL_LABEL_REFUSED = "no interval: the readout refused to issue a number"

ZONES_RULE = (
    "scale axis only: with reference step ratio r and largest trained scale M, a target up to "
    "M*WARN_RATIO_MULT*r passes the ratio check, up to M*REFUSE_RATIO_MULT*r gets a ratio warning, beyond "
    "that is refused; never beyond M*HARD_CAP_RATIO (gate). Process-parameter range and pooled-backtest "
    "checks still apply to any target"
)


def compute_readout_id(
    raw_bytes: bytes,
    target_column: str,
    target: TargetSpec,
    physics_overrides: dict[str, Any] | None,
    *,
    engine_version: str = ENGINE_VERSION,
) -> str:
    """Deterministic 64-hex SHA-256 of (raw sheet bytes, canonical target,
    engine version). Whitespace/key order in the caller's target JSON does
    not change the id; any change to the sheet, the target values, or the
    engine version does."""
    canonical_target = json.dumps(
        {
            "target_column": target_column,
            "scale_L": target.scale_L,
            "agitation_rpm": target.agitation_rpm,
            "airflow_L_per_min": target.airflow_L_per_min,
            "process_params": dict(sorted(target.process_params.items())),
            "physics_overrides": dict(sorted((physics_overrides or {}).items())),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    h = hashlib.sha256()
    for part in (_READOUT_ID_DOMAIN, engine_version, hashlib.sha256(raw_bytes).hexdigest(), canonical_target):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _finite(value: Any) -> Any:
    """Recursively replace NaN/inf with None (strict JSON, and no fake numbers)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _zones(max_trained: float | None, reference_ratio: float | None) -> dict[str, Any] | None:
    if max_trained is None or reference_ratio is None:
        return None
    hard_cap_L = max_trained * HARD_CAP_RATIO
    return {
        "licensed_max_L": min(max_trained * WARN_RATIO_MULT * reference_ratio, hard_cap_L),
        "warn_max_L": min(max_trained * REFUSE_RATIO_MULT * reference_ratio, hard_cap_L),
        "hard_cap_L": hard_cap_L,
        "basis": "derived",
        "rule": ZONES_RULE,
    }


def readout_to_json(readout: dict[str, Any], *, readout_id: str) -> dict[str, Any]:
    """The JSON readout payload. See docs/SCALE_READOUT.md "JSON readout"."""
    has_interval = readout.get("interval") is not None
    provenance = readout.get("provenance") or {}
    max_trained = readout.get("max_trained_scale_L")
    payload: dict[str, Any] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "readout_id": readout_id,
        "generated_at_utc": readout.get("generated_at_utc"),
        "target_column": readout.get("target_column"),
        "target_column_unit": provenance.get("target_column_unit"),
        "decision": readout["decision"],
        "reasons": list(readout.get("reasons") or []),
        "prediction": readout.get("prediction") if has_interval else None,
        "prediction_basis": "predicted" if has_interval else None,
        "interval": list(readout["interval"]) if has_interval else None,
        "interval_coverage": (1.0 - ALPHA) if has_interval else None,
        "interval_method": INTERVAL_METHOD if has_interval else None,
        "interval_half_width": readout.get("interval_half_width") if has_interval else None,
        "interval_label": (
            INTERVAL_LABEL.format(pct=round((1.0 - ALPHA) * 100, 6), alpha=ALPHA)
            if has_interval
            else INTERVAL_LABEL_REFUSED
        ),
        "rungs": list(readout.get("rungs") or []),
        "skipped_rungs": list(readout.get("skipped_rungs") or []),
        "reference_ratio": readout.get("reference_ratio"),
        "requested_ratio": readout.get("requested_ratio"),
        "out_of_range_params": list(readout.get("out_of_range_params") or []),
        "baseline_comparison": readout.get("baseline_comparison"),
        "data_plan": list(readout.get("data_plan") or []),
        "target_inputs": readout.get("target_inputs"),
        "feature_set": readout.get("feature_set"),
        "missing_physics_inputs": list(readout.get("missing_physics_inputs") or []),
        "physics_assumptions": readout.get("physics_assumptions"),
        "scale_points": list(readout.get("scale_points") or []),
        "backtest": readout.get("backtest") or {"method": BACKTEST_METHOD, "per_scale": [], "skipped": []},
        "divergence": readout.get("divergence")
        or {"computed": False, "reason": "no backtest was run", "rule": None},
        "max_trained_scale_L": max_trained,
        "warn_ratio_mult": WARN_RATIO_MULT,
        "refuse_ratio_mult": REFUSE_RATIO_MULT,
        "hard_cap_ratio": HARD_CAP_RATIO,
        "zones": _zones(max_trained, readout.get("reference_ratio")),
        "provenance": provenance,
    }
    if readout["decision"] == "REFUSAL":
        # never a guessed number on a refusal
        payload["prediction"] = None
        payload["interval"] = None
    return _finite(payload)


__all__ = [
    "INTERVAL_METHOD",
    "KIND",
    "SCHEMA_VERSION",
    "compute_readout_id",
    "readout_to_json",
]
