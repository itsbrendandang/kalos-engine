"""Turn engine output into flat signals the scorer can match.

A signal is `{source, code, severity, column, rows, message}`. This module only
re-labels what the engine returned; it never judges a case and it knows nothing
about faults (it is part of the blind runner).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

Signal = dict[str, Any]
MESSAGE_CHARS = 240
NON_LEVEL_MAX_LEVELS = 12


def signal(
    source: str, code: str, severity: str = "info", *, column: str | None = None,
    rows: Iterable[int] = (), message: str = "", n_rows: int | None = None,
) -> Signal:
    """`rows` are the rows the engine names (validation caps them at 20);
    `n_rows` is how many it says are affected, when it says so."""
    named = sorted(int(r) for r in rows)
    return {
        "source": source, "code": str(code), "severity": severity,
        "column": None if column is None else str(column),
        "rows": named, "n_rows": len(named) if n_rows is None else int(n_rows),
        "message": str(message)[:MESSAGE_CHARS],
    }


def from_refusal(code: str | None, message: str) -> Signal:
    return signal("refusal", code or "unknown", "error", message=message)


def from_validation(report: Mapping[str, Any]) -> list[Signal]:
    out = []
    for f in report.get("findings") or []:
        out.append(signal(
            "validation", f.get("check", "unknown"), f.get("severity", "info"),
            column=f.get("column"), rows=f.get("rows") or (), message=f.get("message", ""),
            n_rows=(f.get("detail") or {}).get("n_rows_affected"),
        ))
    return out


def from_orientation(info: Mapping[str, Any] | None) -> list[Signal]:
    if not info:
        return []
    orientation = str(info.get("orientation", ""))
    if orientation in ("transposed", "ambiguous"):
        return [signal("orientation", orientation, "info",
                       message=f"confidence {info.get('confidence')}")]
    return []


def _proposal_signals(result: Mapping[str, Any], df: pd.DataFrame | None) -> list[Signal]:
    proposals = result.get("proposals") or []
    proposed: dict[str, list[Any]] = {}
    for p in proposals:
        for col, val in (p.get("recipe") or {}).items():
            proposed.setdefault(str(col), []).append(val)
    out = [signal("proposals", "proposes", column=col,
                  message=f"{len(vals)} proposal(s), e.g. {vals[0]!r}")
           for col, vals in sorted(proposed.items())]
    if proposals:
        out.append(signal("proposals", "batch_returned", message=f"{len(proposals)} proposal(s)"))
    notice = result.get("proposal_notice")
    if notice:
        out.append(signal("proposals", "notice", "warning", message=str(notice)))
    if df is None:
        return out
    # A numeric column that only ever holds a few integer values (an id-like code)
    # proposed at a value it never took.
    for col, vals in sorted(proposed.items()):
        if col not in df.columns:
            continue
        observed = pd.to_numeric(df[col], errors="coerce").dropna()
        if observed.empty or observed.nunique() > NON_LEVEL_MAX_LEVELS:
            continue
        if not (observed == observed.round()).all():
            continue
        levels = set(observed.astype(float))
        bad = [v for v in vals if isinstance(v, (int, float)) and float(v) not in levels]
        if bad:
            out.append(signal("proposals", "proposes_non_level", column=col,
                              message=f"proposed {bad[0]!r}; observed levels {sorted(levels)}"))
    return out


def from_analysis(result: Mapping[str, Any], df: pd.DataFrame | None = None) -> list[Signal]:
    """Signals from a full `_analyze` result (the upload response body)."""
    out = from_validation(result.get("validation") or {})
    out += from_orientation(result.get("orientation"))
    rel = result.get("reliability") or {}
    clears = rel.get("clears_floor")
    if clears is False:
        out.append(signal("reliability", "does_not_clear_floor", "warning",
                          message=f"cv spearman {rel.get('spearman')} ci {rel.get('ci95')}"))
    elif clears is True:
        out.append(signal("reliability", "clears_floor",
                          message=f"cv spearman {rel.get('spearman')} ci {rel.get('ci95')}"))
    elif "clears_floor" in rel or rel.get("verdict") == "not_judged":
        out.append(signal("reliability", "not_judged", "warning", message=str(rel.get("reason", ""))))
    if rel.get("space_filling_advice"):
        out.append(signal("reliability", "space_filling_advice", "warning",
                          message=str(rel.get("space_filling_advice"))))
    promo = result.get("promotion") or {}
    if promo and promo.get("passed") is False:
        out.append(signal("promotion", "not_passed", message=str(promo.get("summary", ""))))
    for col in result.get("provenance") or []:
        out.append(signal("columns", col.get("status", "unknown"), column=col.get("name")))
    # Columns the engine itself read as covariates (recorded run properties such
    # as passage: fitted or not, never proposed; docs/CLONES.md).
    for col in (result.get("covariates") or {}).get("columns") or []:
        out.append(signal("columns", "covariate", column=col,
                          message="engine role: covariate (never proposed)"))
    cats = {str(c) for c in result.get("categorical_features") or []}
    for col in result.get("features") or []:
        out.append(signal("columns", "categorical" if str(col) in cats else "continuous", column=col))
    if result.get("group_col"):
        out.append(signal("cv", "group_col", column=result["group_col"]))
    noise = result.get("noise") or {}
    if (noise.get("n_replicated") or 0) > 0:
        out.append(signal("noise", "replicates_detected",
                          message=f"{noise.get('n_replicated')} replicated recipe(s)"))
    for ex in result.get("design_box_exclusions") or []:
        out.append(signal("design_box", "excluded_from_box", "warning", column=ex.get("column"),
                          message=f"{ex.get('n_excluded')} cell(s) outside physical bounds"))
    out += _proposal_signals(result, df)
    return out


def from_readout(readout: Mapping[str, Any]) -> list[Signal]:
    """Signals from `build_readout` (scale surface)."""
    gate = readout.get("gate") or {}
    if gate and not gate.get("passed", True):
        return [signal("scale_readout", "refusal", "error",
                       message=f"gate {gate.get('failed_check')}: {gate.get('detail')}")]
    decision = readout.get("decision")
    reasons = "; ".join(str(r) for r in readout.get("reasons") or [])
    out: list[Signal] = []
    if decision == "REFUSAL":
        out.append(signal("scale_readout", "refusal", "error", message=reasons))
    elif decision == "NUMBER_WITH_WARNING":
        out.append(signal("scale_readout", "warning", "warning", message=reasons))
    elif decision == "NUMBER":
        out.append(signal("scale_readout", "number", message=f"prediction {readout.get('prediction')}"))
    for name in readout.get("out_of_range_params") or []:
        out.append(signal("scale_readout", "out_of_range_param", "warning", column=name))
    return out


def keys(signals: Sequence[Signal], min_rank: int = 1) -> set[tuple[str, str, str | None]]:
    rank = {"info": 0, "warning": 1, "error": 2}
    return {(s["source"], s["code"], s["column"]) for s in signals if rank.get(s["severity"], 0) >= min_rank}


__all__ = ["Signal", "from_analysis", "from_orientation", "from_readout", "from_refusal",
           "from_validation", "keys", "signal"]
