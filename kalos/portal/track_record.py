"""The prospective track record (BRE-32): each run's prediction RECORDED AT START
scored against the result the lab measured, per round, with coverage so far.

Pure functions over the rows `kalos.portal.prediction_history.track_rows` reads
(no I/O, torch-free). The rules, each enforced here and stated in every response:

- **Only predictions recorded before the result.** A run is scored only when the
  engine recorded a prediction for it (`available`) and that prediction's
  `recorded_at` is EARLIER than the run's first recorded result
  (`run_results` version 1). A run whose result predates its prediction, or
  whose result time is unknown (measured before result history existed), is
  excluded with that reason, never scored.
- **The band recorded at start, never the current analysis.** Inside/outside
  uses the interval stored in `run_predictions` at the requested coverage:
  `band=proposal` (default) the per-proposal interval (`lo_90`/`hi_90` or
  `lo_95`/`hi_95`, with its method and coverage basis), `band=global` the
  global conformal band (`band_lo`/`band_hi`), used only when its recorded
  coverage equals the requested one. A run without that band is excluded with
  `no_band_at_coverage`; its error is still reported.
- **Measured = the run's latest recorded result version.** Corrections are
  counted (`n_corrected`) and every version is listed by the per-run route.
- **Counts before rates.** Coverage carries `n`, `k` and a Wilson score 95%
  interval. Below `MIN_JUDGED_N` scored runs the cumulative coverage is
  `judged: false` with a reason; it is never called calibrated or not.
- **Error = measured - predicted** (`basis: "measured_minus_recorded_prediction"`),
  over every run with a recorded engine prediction and a result recorded after it.

`basis: "prospective"` on every block: these are predictions made BEFORE the
outcome, not cross-validation (`retrospective_cv`), and the two are never pooled.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kalos.portal.prediction_history import level_key

MIN_JUDGED_N = 10
WILSON_Z = 1.959963984540054  # two-sided 95%
BANDS = ("proposal", "global")
BASIS = "prospective"
ERROR_BASIS = "measured_minus_recorded_prediction"

EXCLUDED_REASONS: dict[str, str] = {
    "awaiting": "no result recorded yet",
    "no_engine_prediction": "the engine recorded no prediction for this run (not a published "
                            "proposal, or started before predictions were recorded)",
    "result_before_prediction": "the run's first result was recorded before its prediction; "
                                "it is never scored against a later prediction",
    "result_time_unknown": "the run's result was recorded before result history existed, so it "
                           "cannot be shown to come after the prediction",
    "no_band_at_coverage": "no band at the requested coverage was recorded for this run",
}

RULES = {
    "scored_only_if": "the engine recorded a prediction for the run before the run's first "
                      "recorded result",
    "band": "the band recorded when the run was started, at the requested coverage; never the "
            "current analysis",
    "measured": "the run's latest recorded result version; corrections are counted",
    "error_formula": "measured minus the prediction recorded at start",
    "coverage_interval": "Wilson score interval, 95%",
    "pooling": "a coverage figure pools every scored band at the requested coverage; by_method and "
               "by_coverage_basis show how many came from each kind of band",
    "caveat": "every band is returned with the caveat the engine served with it when the run "
              "started (code caveat_not_recorded when none was stored)",
    "judged_from_n": MIN_JUDGED_N,
}


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials (None when n == 0)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# Served when no caveat was stored with a band (a run copied from a v7 state
# whose analysis was not linked, or an analysis published before caveats
# existed). No coverage figure: only what is true of every engine band.
CAVEAT_NOT_RECORDED: dict[str, Any] = {
    "code": "caveat_not_recorded",
    "message": ("No caveat was recorded with this band when the run started. The band was sized on "
                "held-out past runs; coverage at newly proposed recipes can be lower."),
    "recorded": False,
}


def caveat_of(code: Any, message: Any) -> dict[str, Any]:
    """The caveat stored with a band (verbatim, `recorded: true`), else
    `CAVEAT_NOT_RECORDED`."""
    if isinstance(message, str) and message:
        return {"code": code, "message": message, "recorded": True}
    return dict(CAVEAT_NOT_RECORDED)


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


def band_for(row: Mapping[str, Any], coverage: float, band: str) -> dict[str, Any] | None:
    """The band recorded at start for `row` at `coverage`, or None."""
    if band == "global":
        bc = row.get("band_coverage")
        if (isinstance(bc, (int, float)) and abs(float(bc) - coverage) < 1e-9
                and row.get("band_lo") is not None and row.get("band_hi") is not None):
            return {"lo": row["band_lo"], "hi": row["band_hi"], "coverage": float(bc),
                    "method": row.get("band_method"), "coverage_basis": row.get("band_basis"),
                    "caveat": caveat_of(row.get("band_caveat_code"), row.get("band_caveat")),
                    "recorded_at": row.get("recorded_at")}
        return None
    key = level_key(coverage)
    if key is None or row.get(f"lo_{key}") is None or row.get(f"hi_{key}") is None:
        return None
    return {"lo": row[f"lo_{key}"], "hi": row[f"hi_{key}"], "coverage": coverage,
            "method": row.get(f"method_{key}"), "coverage_basis": row.get(f"basis_{key}"),
            "caveat": caveat_of(row.get(f"caveat_code_{key}"), row.get(f"caveat_{key}")),
            "recorded_at": row.get("recorded_at")}


def status_of(row: Mapping[str, Any], in_flight: set[str]) -> str:
    if row.get("folded_in_round") is not None:
        return "folded"
    if row.get("run_id") in in_flight:
        return "measured" if row.get("n_versions") else "awaiting"
    return "discarded"


def score_run(
    row: Mapping[str, Any], *, coverage: float, band: str, in_flight: set[str],
) -> dict[str, Any]:
    """One run's prospective score. `excluded` is None when it is scored."""
    measured = row.get("value") if row.get("n_versions") else None
    status = status_of(row, in_flight)
    out: dict[str, Any] = {
        "status": status, "measured": measured, "band": None, "inside": None, "error": None,
        "has_error": False, "scored": False, "excluded": None,
        "corrected": bool((row.get("n_versions") or 0) > 1),
    }
    if not row.get("available") or row.get("pred") is None:
        out["excluded"] = "no_engine_prediction"
        return out
    if measured is None:
        # Folded/discarded runs measured before result history existed carry no
        # run_results row; their value cannot be timed against the prediction.
        out["excluded"] = "awaiting" if status in ("awaiting", "measured") else "result_time_unknown"
        return out
    first = row.get("first_result_at")
    if first is None:
        out["excluded"] = "result_time_unknown"
        return out
    if not float(row["recorded_at"]) < float(first):
        out["excluded"] = "result_before_prediction"
        return out
    out["error"] = _r(float(measured) - float(row["pred"]))
    out["has_error"] = True
    b = band_for(row, coverage, band)
    out["band"] = b
    if b is None:
        out["excluded"] = "no_band_at_coverage"
        return out
    out["inside"] = bool(float(b["lo"]) <= float(measured) <= float(b["hi"]))
    out["scored"] = True
    return out


def _split(scored: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, int]]:
    """Scored runs counted by the band's `method` or `coverage_basis`: one
    coverage figure can pool bands built different ways, so the split is shown."""
    out: dict[str, dict[str, int]] = {}
    for s in scored:
        name = str((s.get("band") or {}).get(field) or "unknown")
        cell = out.setdefault(name, {"n": 0, "inside": 0})
        cell["n"] += 1
        cell["inside"] += 1 if s.get("inside") else 0
    return dict(sorted(out.items()))


def _coverage(scored: Sequence[Mapping[str, Any]], nominal: float) -> dict[str, Any]:
    n = len(scored)
    k = sum(1 for s in scored if s.get("inside"))
    ci = wilson_interval(k, n)
    return {
        "n": n, "inside": k, "outside": n - k,
        "observed": _r(k / n, 4) if n else None,
        "nominal": nominal,
        "wilson_95": [_r(ci[0], 4), _r(ci[1], 4)] if ci else None,
        "by_method": _split(scored, "method"),
        "by_coverage_basis": _split(scored, "coverage_basis"),
        "basis": BASIS,
    }


def _judged(cov: dict[str, Any]) -> dict[str, Any]:
    n = cov["n"]
    if n < MIN_JUDGED_N:
        return {**cov, "judged": False,
                "reason": f"needs {MIN_JUDGED_N} scored runs to judge; has {n}",
                "nominal_inside_interval": None}
    lo, hi = cov["wilson_95"]
    return {**cov, "judged": True, "reason": None,
            "nominal_inside_interval": bool(lo <= cov["nominal"] <= hi)}


def _errors(errs: Sequence[float]) -> dict[str, Any]:
    n = len(errs)
    return {
        "n": n,
        "mean": _r(sum(errs) / n) if n else None,
        "mean_abs": _r(sum(abs(e) for e in errs) / n) if n else None,
        "rmse": _r(math.sqrt(sum(e * e for e in errs) / n)) if n else None,
        "basis": ERROR_BASIS,
    }


def build_track_record(
    rows: Iterable[Mapping[str, Any]], *, coverage: float, band: str,
    state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Per-round prospective scores (chronological), coverage so far, and the
    pooled score. `state` is the campaign state (current pending runs, round,
    seed id); None when the campaign has never been seeded."""
    pending = (state or {}).get("pending")
    in_flight = {str(r.get("id")) for r in pending if isinstance(r, dict)} if isinstance(pending, list) else set()
    current_seed = (state or {}).get("seed_id")
    groups: dict[tuple[Any, Any], list[tuple[Mapping[str, Any], dict[str, Any]]]] = {}
    first_seen: dict[tuple[Any, Any], float] = {}
    epochs: dict[Any, int] = {}
    for row in rows:
        key = (row.get("seed_id"), row.get("round"))
        if row.get("seed_id") not in epochs:
            epochs[row.get("seed_id")] = len(epochs) + 1
        groups.setdefault(key, []).append((row, score_run(row, coverage=coverage, band=band,
                                                          in_flight=in_flight)))
        first_seen[key] = min(first_seen.get(key, math.inf), float(row.get("recorded_at") or 0))
    order = sorted(groups, key=lambda k: first_seen[k])
    rounds: list[dict[str, Any]] = []
    cum_scored: list[dict[str, Any]] = []
    cum_err: list[float] = []
    for key in order:
        items = groups[key]
        scored = [s for _, s in items if s["scored"]]
        errs = [float(s["error"]) for _, s in items if s["has_error"]]
        cum_scored += scored
        cum_err += errs
        excluded: dict[str, int] = {}
        for _, s in items:
            if s["excluded"]:
                excluded[s["excluded"]] = excluded.get(s["excluded"], 0) + 1
        seed, rnd = key
        rounds.append({
            "round": rnd,
            "dataset_epoch": epochs[seed],
            "current_dataset": seed == current_seed,
            "n_started": len(items),
            "n_engine_predictions": sum(1 for r, _ in items if r.get("available")),
            "n_measured": sum(1 for _, s in items if s["measured"] is not None),
            "n_corrected": sum(1 for _, s in items if s["corrected"]),
            "n_folded": sum(1 for _, s in items if s["status"] == "folded"),
            "coverage": _coverage(scored, coverage),
            "coverage_so_far": _judged(_coverage(cum_scored, coverage)),
            "error": _errors(errs),
            "error_so_far": _errors(cum_err),
            "excluded": excluded,
            "first_recorded_at": first_seen[key],
        })
    return {
        "basis": BASIS,
        "coverage": coverage,
        "band": band,
        "rounds": rounds,
        "pooled": {"coverage": _judged(_coverage(cum_scored, coverage)),
                   "error": _errors(cum_err)},
        "rules": RULES,
        "excluded_reasons": EXCLUDED_REASONS,
    }


def rounds_without_record(
    state: Mapping[str, Any] | None, rounds: Sequence[Mapping[str, Any]],
) -> list[int]:
    """Rounds of the CURRENT dataset that were folded but have no recorded
    prediction (folded before schema v8, or no run was started from a proposal)."""
    if not isinstance(state, Mapping):
        return []
    cur = state.get("round")
    if not isinstance(cur, int) or isinstance(cur, bool):
        return []
    have = {r["round"] for r in rounds if r.get("current_dataset")}
    return [r for r in range(cur) if r not in have]
