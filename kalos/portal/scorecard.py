"""Prediction scorecard: each proposed run's saved prediction and band against
the result the lab measured for it (docs/CAMPAIGN_LOOP.md, "Prediction scorecard").

A run started from a proposal keeps the prediction it was started with (`pred`,
`std`), the published analysis's prediction band (`band_half_width`, the
conformal half-width, and `band_coverage`, the coverage that band states) and the
best measured value at the time (`best_at_start`). Once a result is logged the
scorecard says how far off the prediction was, whether the result fell inside
the band, and whether the run beat the best so far. It is the check the
cross-validated band cannot give: the band is measured on held-out PAST runs,
while proposals sit where the model knows least.

Measured runs are copied here at fold time (`scored_record`), because the fold
drops them from `pending`; without the copy the check would vanish the moment
the loop closes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

# Below this many measured proposals with a band, observed coverage is reported
# but not judged: with 10 runs a 90% band still misses 0-3 by chance.
MIN_SCORED_RUNS = 10

_KEPT = ("id", "round", "mode", "pred", "std", "band_half_width", "band_coverage",
         "best_at_start", "result", "measured_at")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def scored_record(run: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a measured run the scorecard needs, kept after the fold."""
    return {k: run.get(k) for k in _KEPT}


def _row(run: Mapping[str, Any]) -> dict[str, Any] | None:
    pred, result = _num(run.get("pred")), _num(run.get("result"))
    if pred is None or result is None:
        return None
    half = _num(run.get("band_half_width"))
    best = _num(run.get("best_at_start"))
    error = result - pred
    return {
        "id": run.get("id"),
        "round": run.get("round"),
        "mode": run.get("mode"),
        "pred": pred,
        "std": _num(run.get("std")),
        "band_half_width": half,
        "band_coverage": _num(run.get("band_coverage")),
        "result": result,
        "error": round(error, 6),
        "in_band": None if half is None else bool(abs(error) <= half),
        "best_at_start": best,
        "beat_best": None if best is None else bool(result > best),
    }


def prediction_scorecard(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score every run that has both a saved prediction and a measured result.

    `band_coverage_observed` is the share of banded runs whose result fell inside
    the band; `band_coverage_stated` is the coverage those bands claimed (their
    mean, should analyses with different alphas be mixed). `judgeable` is False
    below `MIN_SCORED_RUNS` banded runs, and `judgement` then says so instead of
    reading a verdict into a handful of runs. `spearman` is the rank agreement of
    prediction and result (None below 3 runs or when either side is constant).
    """
    rows = [r for r in (_row(run) for run in runs) if r is not None]
    banded = [r for r in rows if r["in_band"] is not None]
    with_best = [r for r in rows if r["beat_best"] is not None]
    n_banded = len(banded)

    observed = sum(r["in_band"] for r in banded) / n_banded if n_banded else None
    stated_vals = [r["band_coverage"] for r in banded if r["band_coverage"] is not None]
    stated = sum(stated_vals) / len(stated_vals) if stated_vals else None

    rho = None
    preds, results = [r["pred"] for r in rows], [r["result"] for r in rows]
    if len(rows) >= 3 and len(set(preds)) > 1 and len(set(results)) > 1:
        rho = float(pd.Series(preds).corr(pd.Series(results), method="spearman"))
        rho = round(rho, 3) if math.isfinite(rho) else None

    judgeable = n_banded >= MIN_SCORED_RUNS
    if not judgeable:
        judgement = (f"not enough measured proposals to judge the band "
                     f"(n={n_banded}, need {MIN_SCORED_RUNS})")
    elif stated is None:
        judgement = "band coverage was not stated by the analyses these runs came from"
    elif observed + 1e-9 >= stated:
        judgement = "band held: measured proposals fell inside it at least as often as stated"
    else:
        judgement = "band too narrow: measured proposals fell outside it more often than stated"

    return {
        "n_scored": len(rows),
        "n_banded": n_banded,
        "band_hits": sum(r["in_band"] for r in banded),
        "band_coverage_observed": None if observed is None else round(observed, 4),
        "band_coverage_stated": None if stated is None else round(stated, 4),
        "mean_abs_error": (round(sum(abs(r["error"]) for r in rows) / len(rows), 6)
                           if rows else None),
        "spearman": rho,
        "n_beat_best": sum(r["beat_best"] for r in with_best),
        "n_with_best": len(with_best),
        "judgeable": judgeable,
        "judgement": judgement,
        "runs": rows,
    }
