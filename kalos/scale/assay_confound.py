"""Assay-method confound check for the Scale-Up Readout.

A scale ladder whose outcome was quantified by different assays at different
scales (e.g. a gel-densitometry titer at bench scale and an HPLC titer at pilot
scale) cannot separate "the scale changed the titer" from "the assay changed
the number". Nothing in the ladder backtest can see that, because the method
column is text and never enters the model. This module looks for it directly.

For every candidate method column (a text column whose name reads like
method/assay/quantification, or a column the caller declares):

* `cramers_v_scale`: Cramer's V between the method and the scale level
  (`kalos.core.strain.cramers_v`, no bias correction). A blank method cell is
  its own level, "not recorded": an unrecorded method cannot be assumed equal
  to a recorded one.
* `target_share`: the share of the outcome's variance that lies between method
  levels (eta squared, measured values only), with an exact (n <= 8) or seeded
  Monte Carlo permutation p-value. At small n eta squared is large by chance
  (about (k - 1)/(n - 1) for k levels), so the p-value and n are reported
  beside it and the share alone is never called an effect.
* `target_separates_completely`: the outcome ranges of the method levels do not
  overlap at all.
* `every_step_changes_method`: walking the distinct scales in order, every
  step to the next scale also changes the set of methods used.

Severity (rule in `RULE`):

* `refuse`: the method is aliased with scale (V = 1), the outcome tracks the
  method (share >= 0.5 or complete separation) and every scale step changes the
  method. No step in the ladder compares like with like.
* `warn`: the method is associated with scale (V >= 0.8) and the outcome tracks
  the method; or the method is not tied to scale but the outcome differs by
  method beyond chance (share >= 0.5 or complete separation, permutation
  p <= 0.05), so values from different methods may not be comparable.

The readout applies the severity on top of its own decision: `refuse` turns any
decision into REFUSAL, `warn` turns NUMBER into NUMBER_WITH_WARNING, and every
finding's sentence is added to the readout's reasons. Pure statistics, no I/O,
deterministic (the Monte Carlo draw uses a fixed seed).
"""
from __future__ import annotations

import itertools
import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos.core.strain import cramers_v

FINDING_CODE = "assay_method_confound"
NOT_RECORDED = "not recorded"
V_ALIASED = 0.8
V_PERFECT = 1.0 - 1e-9
TARGET_SHARE_STRONG = 0.5
PERM_ALPHA = 0.05
_EXACT_PERM_MAX_N = 8
_MC_DRAWS = 5000
_MC_SEED = 20261120
# Name-matched columns qualify only when they hold labels, not measurements
# (a numeric `assay_cv_pct` is not a method): mostly non-numeric, or a few
# integer codes.
_MAX_NUMERIC_LEVELS = 6
METHOD_NAME_RE = re.compile(
    r"(method|assay|quantif|quant_by|analytic|instrument|technique|measured_by|detection)", re.IGNORECASE
)

RULE = (
    "refuse: method aliased with scale (Cramer's V = 1), outcome tracks the method (between-method share "
    f">= {TARGET_SHARE_STRONG:g} or complete separation) and every scale step changes the method. "
    f"warn: V >= {V_ALIASED:g} and outcome tracks the method; or outcome differs by method beyond chance "
    f"(share >= {TARGET_SHARE_STRONG:g} or complete separation, permutation p <= {PERM_ALPHA:g}). "
    "A blank method cell counts as its own level, 'not recorded'."
)
WHAT_WOULD_FIX_IT = (
    "measure retained samples from every scale with one assay method, or run both methods on the same "
    "samples so the readout can bridge them; or declare which columns are assay methods if the name "
    "did not say so"
)


def _labels(values: pd.Series) -> list[str]:
    out: list[str] = []
    for v in values.tolist():
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            out.append(NOT_RECORDED)
        elif isinstance(v, float) and v.is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v).strip())
    return out


def _eta_squared(y: np.ndarray, labels: Sequence[str]) -> float | None:
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 0.0:
        return None
    arr = np.asarray(labels, dtype=object)
    between = 0.0
    for lvl in set(labels):
        grp = y[arr == lvl]
        between += len(grp) * (float(grp.mean()) - float(y.mean())) ** 2
    return between / total


def _permutation_p(y: np.ndarray, labels: list[str], observed: float) -> tuple[float, str]:
    """P(eta^2 >= observed) under random relabelling. Exact over the distinct
    rearrangements of the label multiset when n <= 8, else a seeded Monte Carlo."""
    tol = 1e-12
    levels = sorted(set(labels))
    codes = np.array([levels.index(lbl) for lbl in labels])
    if len(labels) <= _EXACT_PERM_MAX_N:
        perms = np.array(sorted(set(itertools.permutations(codes.tolist()))))
        etas = _eta_squared_batch(y, perms, len(levels))
        return float((etas >= observed - tol).mean()), "exact"
    rng = np.random.default_rng(_MC_SEED)
    perms = np.array([rng.permutation(codes) for _ in range(_MC_DRAWS)])
    hits = int((_eta_squared_batch(y, perms, len(levels)) >= observed - tol).sum())
    return (hits + 1) / (_MC_DRAWS + 1), f"monte_carlo_{_MC_DRAWS}_seed_{_MC_SEED}"


def _eta_squared_batch(y: np.ndarray, perms: np.ndarray, k: int) -> np.ndarray:
    """eta^2 for each row of integer label codes in `perms` (same y)."""
    total = float(((y - y.mean()) ** 2).sum())
    between = np.zeros(perms.shape[0])
    for lvl in range(k):
        mask = perms == lvl
        counts = mask.sum(axis=1)
        sums = (mask * y).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            dev = np.where(counts > 0, sums / np.maximum(counts, 1) - y.mean(), 0.0)
        between += counts * dev**2
    return between / total


def _separates_completely(y: np.ndarray, labels: list[str]) -> bool:
    arr = np.asarray(labels, dtype=object)
    ranges = sorted((float(y[arr == lvl].min()), float(y[arr == lvl].max())) for lvl in set(labels))
    if len(ranges) < 2 or float(np.ptp(y)) == 0.0:
        return False
    return all(hi < nxt_lo for (_, hi), (nxt_lo, _) in itertools.pairwise(ranges))


def _is_label_column(values: pd.Series) -> bool:
    filled = values.dropna()
    filled = filled[filled.astype(str).str.strip() != ""]
    if filled.empty:
        return False
    as_num = pd.to_numeric(filled, errors="coerce")
    if as_num.notna().mean() < 0.5:
        return True
    # numeric: only small sets of integer codes ("method 1 / method 2") are labels
    nums = as_num.dropna()
    return bool((nums == nums.round()).all() and nums.nunique() <= _MAX_NUMERIC_LEVELS)


def candidate_columns(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str],
    declared: Sequence[str] = (),
) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    """`[(column, source)]` to check (`source` is `declared` or `name`) and the
    declared columns that could not be checked, with a reason."""
    skipped: list[dict[str, str]] = []
    out: list[tuple[str, str]] = []
    excluded = set(exclude)
    for col in declared:
        if col not in df.columns:
            skipped.append({"column": col, "reason": "declared as an assay method column but not in the sheet"})
        elif col in excluded:
            skipped.append({"column": col, "reason": "is the target, scale or a process column"})
        else:
            out.append((col, "declared"))
    for col in df.columns:
        name = str(col)
        if name in excluded or name in {c for c, _ in out}:
            continue
        if METHOD_NAME_RE.search(name) and _is_label_column(df[col]):
            out.append((name, "name"))
    return out, skipped


def _column_finding(
    col: str, source: str, labels: list[str], y: np.ndarray, scale: np.ndarray, target_column: str
) -> dict[str, Any]:
    levels = sorted(set(labels))
    entry: dict[str, Any] = {
        "column": col,
        "source": source,
        "n": len(labels),
        "levels": {lvl: labels.count(lvl) for lvl in levels},
        "code": FINDING_CODE,
        "severity": None,
        "message": None,
    }
    if len(levels) < 2:
        entry["skipped"] = "one method level only: nothing to confound"
        return entry
    scale_labels = [f"{s:.6g}" for s in scale]
    v = cramers_v(labels, scale_labels) if len(set(scale_labels)) >= 2 else None
    share = _eta_squared(y, labels)
    separates = _separates_completely(y, labels)
    p, p_method = _permutation_p(y, labels, share) if share is not None else (None, None)
    arr = np.asarray(labels, dtype=object)
    by_method = {
        lvl: {"n": int((arr == lvl).sum()), "min": float(y[arr == lvl].min()), "max": float(y[arr == lvl].max()),
              "mean": float(y[arr == lvl].mean()), "basis": "measured"}
        for lvl in levels
    }
    distinct = sorted(set(np.round(scale, 6).tolist()))
    per_scale = [frozenset(arr[np.round(scale, 6) == s].tolist()) for s in distinct]
    steps = list(itertools.pairwise(per_scale))
    every_step = bool(steps) and all(a != b for a, b in steps)
    tracks = (share is not None and share >= TARGET_SHARE_STRONG) or separates
    aliased = v is not None and v >= V_ALIASED
    entry.update({
        "cramers_v_scale": v,
        "target_share": None if share is None else round(share, 4),
        "target_share_p": None if p is None else round(p, 4),
        "target_share_p_method": p_method,
        "target_separates_completely": separates,
        "every_step_changes_method": every_step,
        "n_scales": len(distinct),
        "target_by_method": by_method,
    })
    lvl_text = "; ".join(f"{k}: n={d['n']}, {d['min']:.4g} to {d['max']:.4g}" for k, d in by_method.items())
    share_text = "complete separation" if share is None else f"{share:.0%} of its variance between methods"
    if aliased and tracks:
        refuse = v is not None and v >= V_PERFECT and every_step
        entry["severity"] = "refuse" if refuse else "warn"
        entry["message"] = (
            f"outcome tracks assay method; effects confounded: '{col}' changes with scale "
            f"(Cramer's V {v:.2f} over {len(labels)} runs at {len(distinct)} scales) and '{target_column}' "
            f"differs by method ({share_text}; {lvl_text}). A change across scales cannot be told apart "
            "from a change of assay."
            + (" Every scale step also changes the method, so no step compares like with like." if refuse else "")
        )
    elif tracks and p is not None and p <= PERM_ALPHA:
        entry["severity"] = "warn"
        entry["message"] = (
            f"outcome tracks assay method: '{target_column}' differs by '{col}' ({share_text}, permutation "
            f"p={p:.3g}, n={len(labels)}; {lvl_text}). Values measured by different methods may not be "
            "comparable."
        )
    return entry


def assay_method_confound(
    df: pd.DataFrame,
    target_column: str,
    scale_column: str,
    *,
    exclude: Sequence[str] = (),
    declared: Sequence[str] = (),
) -> dict[str, Any]:
    """Check every candidate assay-method column against the scale and the
    outcome. Only rows with a finite target and a finite scale are used (the
    same rows the readout can model). Returns a JSON-safe dict:
    `columns` (one finding per checked column), `skipped`, `severity`
    (`refuse` | `warn` | None), `reasons` (plain sentences),
    `what_would_fix_it`, `rule`, `n_rows`."""
    result: dict[str, Any] = {
        "code": FINDING_CODE,
        "columns": [],
        "skipped": [],
        "severity": None,
        "reasons": [],
        "what_would_fix_it": None,
        "rule": RULE,
        "n_rows": 0,
    }
    if target_column not in df.columns or scale_column not in df.columns:
        result["skipped"].append({"column": target_column, "reason": "target or scale column absent"})
        return result
    y_all = pd.to_numeric(df[target_column], errors="coerce").to_numpy(dtype=float)
    s_all = pd.to_numeric(df[scale_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(y_all) & np.isfinite(s_all)
    result["n_rows"] = int(keep.sum())
    cands, skipped = candidate_columns(
        df, exclude=[target_column, scale_column, *exclude], declared=declared
    )
    result["skipped"].extend(skipped)
    if keep.sum() < 2:
        for col, _ in cands:
            result["skipped"].append({"column": col, "reason": "fewer than 2 rows with a finite target and scale"})
        return result
    y, s = y_all[keep], s_all[keep]
    for col, source in cands:
        labels = _labels(df[col].reset_index(drop=True)[keep])
        result["columns"].append(_column_finding(col, source, labels, y, s, target_column))
    sev = [c["severity"] for c in result["columns"] if c["severity"]]
    result["severity"] = "refuse" if "refuse" in sev else ("warn" if sev else None)
    result["reasons"] = [c["message"] for c in result["columns"] if c["message"]]
    if result["severity"]:
        result["what_would_fix_it"] = WHAT_WOULD_FIX_IT
    return result


__all__ = [
    "FINDING_CODE",
    "METHOD_NAME_RE",
    "RULE",
    "TARGET_SHARE_STRONG",
    "V_ALIASED",
    "assay_method_confound",
    "candidate_columns",
]
