"""Proposal shaping: annotating each proposed experiment with its prediction and
explore/exploit rationale, and encoding in-flight (pending) recipes into the
fitted design's coordinates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos.domains import DesignSpace


def _annotate(
    batch: np.ndarray,
    mean,
    std,
    best: float,
    cols=None,
    design: DesignSpace | None = None,
    *,
    p_feasible: np.ndarray | None = None,
    constraint: tuple[str, np.ndarray, np.ndarray] | None = None,
) -> list:
    """Attach predicted value, uncertainty, and an explore/exploit rationale to
    each proposed experiment. Explore = high model uncertainty (chosen to learn);
    exploit = high predicted value (chosen to win). Current human-in-the-loop BO
    research says a recommendation must carry exactly this.

    When a `design` is given, each row also carries `recipe`: the full proposed
    experiment decoded to `{feature: value}`, with categorical dimensions decoded
    back to their labels instead of raw integer codes.

    `p_feasible`, when given, is a length-`len(batch)` array and every row gets a
    `p_feasible` key - `None` for a NaN entry, a rounded float otherwise. `None`
    (the default, unchanged for the `kalos.portal.app` call site) omits the key
    entirely rather than adding it with a placeholder value, so a caller that
    never computed a feasibility probability is not told one exists.

    `constraint`, when given, is `(column_name, pred_mean, pred_std)` - three
    length-`len(batch)` arrays/name - and every row gets a
    `pred_<column_name>` key of `{"mean": ..., "std": ...}`. `None` (the
    default) omits the key entirely: it is only ever passed when PART 2's
    constraint was actually applied."""
    mean = np.asarray(mean, float).reshape(-1)
    std = np.asarray(std, float).reshape(-1)
    thr = float(np.quantile(std, 2 / 3)) if len(std) > 2 else float(std.max() if len(std) else 0.0)
    eps = 0.02 * max(abs(best), 1e-9)
    rows = []
    for i in range(len(batch)):
        m, sd = float(mean[i]), float(std[i])
        gain = m - best
        if gain > eps:
            mode, reason = "exploit", f"predicted high (+{gain:.3g} vs best)"
        elif sd >= thr:
            mode, reason = "explore", f"reduce model uncertainty here (±{sd:.3g})"
        else:
            mode, reason = "explore", f"diversifies the batch (predicted {m:.3g}, ±{sd:.3g})"
        row = {"pred": round(m, 3), "std": round(sd, 3), "mode": mode, "reason": reason}
        row["vals"] = (np.round(batch[i], 3).tolist() if cols is None
                       else [round(float(batch[i][j]), 3) for j in cols])
        if design is not None:
            row["recipe"] = dict(zip(design.names, design.decode_row(batch[i])))
        if p_feasible is not None:
            pf = float(p_feasible[i])
            row["p_feasible"] = None if pf != pf else round(pf, 3)  # NaN -> null (ungated)
        if constraint is not None:
            cname, cmean, cstd = constraint
            row[f"pred_{cname}"] = {
                "mean": round(float(cmean[i]), 3),
                "std": round(float(cstd[i]), 3),
            }
        rows.append(row)
    return rows


def _encode_pending(
    pending: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    cont_feats: list,
    kept_cats: list,
    code_maps: dict,
    width: int,
    *,
    absent_zero: frozenset[str] | set[str] = frozenset(),
) -> tuple[np.ndarray | None, int]:
    """Encode in-flight recipes into the fitted design's X layout, or None.

    Mirrors how the FITTED rows are built, so a pending recipe lands at the same
    coordinates its measured counterpart would: continuous components are coerced
    to numeric and zero-filled (the same `Xc_zf` convention), categoricals are
    mapped through the level codes derived from the fitted rows, and the columns
    are assembled continuous-first, categorical-last.

    A row is DROPPED when a kept categorical is missing, blank, or names a level
    that does not exist in this design, or when a continuous component is missing
    (blank, non-numeric, or the column is absent from the record). Such a recipe
    has no honest coordinate: inventing one (zero-filling a pH, or code 0 for a
    level) would mark the wrong region of the space as taken (review H2). The only
    exception is an `absent_zero` column (a sparse / concentration component),
    where a blank means 0 (see `_absent_is_zero_columns`).

    Returns `(encoded, n_dropped)`. `encoded` is None when nothing is left, which
    leaves the acquisition exactly as it is today; `n_dropped` counts every
    in-flight record that was dropped, so the caller can report it. Never raises -
    a malformed in-flight record must degrade the proposal's information, not fail
    the analysis.
    """
    if pending is None:
        return None, 0
    frame = pending if isinstance(pending, pd.DataFrame) else pd.DataFrame(list(pending))
    if frame.empty:
        return None, 0
    cols: list[np.ndarray] = []
    usable = np.ones(len(frame), dtype=bool)
    for c in cont_feats:
        if c in frame.columns:
            vals = pd.to_numeric(frame[c], errors="coerce")
        else:
            vals = pd.Series(np.nan, index=frame.index)
        if c in absent_zero:
            vals = vals.fillna(0.0)
        arr_c = vals.to_numpy(float)
        usable &= np.isfinite(arr_c)
        cols.append(arr_c)
    for c in kept_cats:
        codes = np.full(len(frame), np.nan)
        if c in frame.columns:
            labels = frame[c].fillna("").astype(str).str.strip()
            codes = labels.map(code_maps[c]).to_numpy(dtype=float)
        usable &= np.isfinite(codes)
        cols.append(codes)
    if not cols:
        return None, 0
    arr = np.column_stack(cols)
    usable &= np.isfinite(arr).all(axis=1)
    n_dropped = int(len(frame) - usable.sum())
    arr = arr[usable]
    if arr.shape[1] != width:
        return None, len(frame)
    if arr.shape[0] == 0:
        return None, n_dropped
    return arr, n_dropped
