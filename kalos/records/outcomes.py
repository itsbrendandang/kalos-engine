"""Outcomes: a read view of what was measured, per run, across a project.

Voyager stored outcomes as ``Outcome`` rows with a JSONB ``DataFile`` and one
``OutcomeDataPoint`` (a JSONB value) per factor, beside ``ExperimentDataPoint``
rows for the inputs. Kalos already has all of it, so nothing is duplicated:

- a Voyager experiment data point (the input values of one iteration) is a Kalos
  RUN's recipe;
- a Voyager outcome data point (a measured value) is a Kalos RESULT, kept
  append-only with every correction (`run_results`, schema v5);
- the prediction made before the result came back is recorded at start
  (`run_predictions`, schema v8) and is never overwritten.

This module only shapes those rows. Measured and predicted values stay in
separate objects (`measured`, `prediction_at_start`); a run with no result says
so; nothing is averaged, filled in or extrapolated. Results are written only
through the existing result route (`POST /api/campaigns/{id}/runs/{run}/result`),
which keeps the campaign state and the history in one transaction.
"""
from __future__ import annotations

from typing import Any

MAX_RUNS_PER_PAGE = 500
DEFAULT_RUNS_PER_PAGE = 100


def run_outcome(
    campaign_id: str, run_id: str, versions: list[dict[str, Any]], prediction: dict[str, Any] | None,
) -> dict[str, Any]:
    """One run: its newest measured value (with how many versions exist and
    whether the newest is a correction) and the prediction recorded at start."""
    versions = sorted(versions, key=lambda v: int(v["version"]))
    latest = versions[-1] if versions else None
    measured: dict[str, Any] | None = None
    if latest is not None:
        measured = {
            "value": latest["value"],
            "version": int(latest["version"]),
            "n_versions": len(versions),
            "is_correction": int(latest["version"]) > 1,
            "previous_value": latest.get("previous_value"),
            "reason": latest.get("reason"),
            "recorded_at": latest.get("recorded_at"),
            "basis_kind": "measured",
        }
    pred: dict[str, Any] | None = None
    if prediction is not None:
        if prediction.get("available"):
            pred = {
                "available": True,
                "mean": prediction.get("pred"),
                "interval_90": _interval(prediction, "90"),
                "interval_95": _interval(prediction, "95"),
                "round": prediction.get("round"),
                "model_version": prediction.get("model_version"),
                "recorded_at": prediction.get("recorded_at"),
                "basis_kind": "predicted",
            }
        else:
            pred = {"available": False, "reason": prediction.get("unavailable_reason"),
                    "round": prediction.get("round"), "basis_kind": "predicted"}
    return {
        "campaign_id": campaign_id,
        "run_id": run_id,
        "measured": measured,
        "measured_reason": None if measured is not None else "no result recorded yet",
        "prediction_at_start": pred,
        "prediction_reason": None if pred is not None else (
            "no prediction was recorded when this run started (runs started before "
            "schema v8, or uploaded as data rather than started)"
        ),
    }


def _interval(prediction: dict[str, Any], level: str) -> dict[str, Any] | None:
    lo, hi = prediction.get(f"lo_{level}"), prediction.get(f"hi_{level}")
    if lo is None or hi is None:
        return None
    return {
        "lo": lo, "hi": hi, "coverage": int(level) / 100,
        "method": prediction.get(f"method_{level}"), "basis": prediction.get(f"basis_{level}"),
    }


def outcomes_page(
    rows: list[tuple[str, str]],
    versions: dict[tuple[str, str], list[dict[str, Any]]],
    predictions: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """The run outcomes for `rows` ((campaign_id, run_id) pairs, already paged)."""
    return [run_outcome(c, r, versions.get((c, r), []), predictions.get((c, r))) for c, r in rows]


__all__ = ["DEFAULT_RUNS_PER_PAGE", "MAX_RUNS_PER_PAGE", "outcomes_page", "run_outcome"]
