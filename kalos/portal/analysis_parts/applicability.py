"""Applicability flags per proposal: far from the data, and at the box edge.

Two DISTINCT flags. `far_from_data` compares `nearest_training_distance` with a
threshold. `at_box_edge` lists the continuous inputs whose proposed value sits
within 1% of the box width of a search-box bound. A proposal can be near the
data and on the edge, or far from the data in the middle of the box.

The threshold is a product default unless a lab sets one. The default rule is
`max(0.25, 3 x training_nn_distance_median)`: three times the typical spacing
between training rows, and never below a quarter of one box width. It is a
starting point, not a validated number. A lab sets its own value with
`roles.applicability_distance` (per upload) or `KALOS_APPLICABILITY_DISTANCE`
(per deployment); roles wins.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from kalos.domains import DesignSpace
from kalos.envconfig import ConfigError

APPLICABILITY_DISTANCE_VAR = "KALOS_APPLICABILITY_DISTANCE"
# Product default (see the module docstring): max(floor, multiple x NN median).
DEFAULT_DISTANCE_FLOOR = 0.25
DEFAULT_NN_MULTIPLE = 3.0
# A value within this fraction of the box width from a bound is "at the edge".
EDGE_FRACTION = 0.01

DEFAULT_RULE = (
    f"product default: max({DEFAULT_DISTANCE_FLOOR}, {DEFAULT_NN_MULTIPLE:g} x "
    "training_nn_distance_median); not a validated number"
)


def _positive_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return v if math.isfinite(v) and v > 0 else None


def env_applicability_distance() -> float | None:
    """`KALOS_APPLICABILITY_DISTANCE`: a positive number, or None when unset or
    empty. Any other value raises `ConfigError` (fail closed, kalos.envconfig)."""
    raw = os.environ.get(APPLICABILITY_DISTANCE_VAR, "").strip()
    if not raw:
        return None
    value = _positive_finite(raw)
    if value is None:
        raise ConfigError(
            f"{APPLICABILITY_DISTANCE_VAR}={raw[:40]!r} must be a positive number "
            "(or unset for the product default)."
        )
    return value


def applicability_block(
    nn_median: float | None, roles_distance: float | None = None
) -> dict[str, Any]:
    """The top-level `applicability` block: the threshold, where it came from
    (`source` "default" or "lab"; `set_by` "default", "env" or "roles"), the
    rule in words, and the edge fraction."""
    env_value = env_applicability_distance()
    if roles_distance is not None:
        threshold, source, set_by = float(roles_distance), "lab", "roles"
        rule = "lab-set: roles.applicability_distance"
    elif env_value is not None:
        threshold, source, set_by = env_value, "lab", "env"
        rule = f"lab-set: {APPLICABILITY_DISTANCE_VAR}"
    else:
        base = 0.0 if nn_median is None else DEFAULT_NN_MULTIPLE * float(nn_median)
        threshold = round(max(DEFAULT_DISTANCE_FLOOR, base), 4)
        source, set_by = "default", "default"
        rule = DEFAULT_RULE
    return {
        "threshold": float(threshold),
        "source": source,
        "set_by": set_by,
        "rule": rule,
        "edge_fraction": EDGE_FRACTION,
    }


def box_edges(row: np.ndarray, design: DesignSpace) -> list[dict[str, str]]:
    """Continuous inputs of `row` within `EDGE_FRACTION` of the box width of a
    bound, each as `{"feature", "side"}` with side "lower" or "upper"."""
    out: list[dict[str, str]] = []
    vals = np.asarray(row, float).reshape(-1)
    for i, dim in enumerate(design.dims):
        if dim.is_categorical:
            continue
        lo, hi = float(dim.lower), float(dim.upper)
        width = hi - lo
        if not width > 0:
            continue
        tol = EDGE_FRACTION * width
        v = float(vals[i])
        if v - lo <= tol:
            out.append({"feature": dim.name, "side": "lower"})
        elif hi - v <= tol:
            out.append({"feature": dim.name, "side": "upper"})
    return out


def proposal_applicability(
    batch: np.ndarray, distances: np.ndarray, design: DesignSpace, block: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-proposal `applicability`: `distance` (the rounded
    `nearest_training_distance`), the `threshold` and `threshold_source`,
    `far_from_data` (distance > threshold; None if the distance is unknown) and
    `at_box_edge`."""
    rows = []
    threshold = float(block["threshold"])
    for i in range(len(batch)):
        d = float(distances[i])
        dist = round(d, 4) if math.isfinite(d) else None
        rows.append({
            "distance": dist,
            "threshold": threshold,
            "threshold_source": block["source"],
            "far_from_data": None if dist is None else bool(dist > threshold),
            "at_box_edge": box_edges(batch[i], design),
        })
    return rows
