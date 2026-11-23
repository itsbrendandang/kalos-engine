"""Leak-free cross-validation grouping (F05, ML audit B2).

The trust verdict is a held-out score, and a held-out score is only as honest as
the grouping that decides what "held out" means. Three leaks were measured on
the previous grouping (exact-recipe hash, or a declared group column):

- **As-run replicates.** Replicates recorded as run (0.2% jitter on the
  settings) are distinct rows to an exact hash, so the replicates of one recipe
  landed on both sides of a split: CV 0.95 and 100% false trust on a sheet whose
  inputs carry no signal (true-group CV -0.21).
- **Batches, plates, lots, dates.** A day-to-day offset shared by the runs of a
  batch is learnable from the batch's neighbours in the same fold: CV 0.61 and
  95% false trust, against a prospective Spearman of 0.04 on a new batch.
- **Campaign rounds.** Each round's proposals cluster in space and share the
  round's conditions; folded back into the sheet they are the same leak.

What this module does:

1. `replicate_clusters`: rows whose range-normalised settings are all within
   `NEAR_DUPLICATE_TOL` (0.05 of each input's box width, the separation the
   proposal batch already uses) and share every categorical label are one
   group. Complete linkage under the Chebyshev distance, so every pair inside a
   group is within the tolerance (no chaining across a dense design).
2. `structure_candidates`: columns named like a batch, lot, plate, block, date,
   week or round, scale columns (scale, vessel, working volume, reactor size),
   and the campaign round index of folded rows (`df.attrs["kalos_round_index"]`)
   are CV-group CANDIDATES, whether or not each level is one recipe. Each is
   merged with the replicate clusters (union-find), so neither replicates nor a
   batch can straddle a split.
3. The analysis cross-validates every candidate with at least
   `MIN_GROUPS_FOR_CV` groups and the LOWEST held-out Spearman drives
   `cv_spearman`, the reliability verdict and the conformal calibration. The
   grouping used, every candidate's score and every warning are reported in
   `cv_grouping`.

Torch-free; scipy's hierarchical clustering only.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos.portal.cells import is_null_label

# Same number as kalos.core.optimize.NEAR_DUPLICATE_TOL (kept torch-free here;
# tests/test_cv_grouping.py asserts they agree).
NEAR_DUPLICATE_TOL = 0.05
# Below 3 groups the group bootstrap cannot form an interval and a 2-fold
# split says almost nothing; a candidate with fewer groups is reported, not used.
MIN_GROUPS_FOR_CV = 3
# Below this many groups the held-out score is coarse: reported with a warning.
FEW_GROUPS_WARN = 5
ROUND_ATTR = "kalos_round_index"

# Batch-like structure: never a setpoint, always a grouping candidate.
_SEP = r"(?:^|[\s_\-./])"
# The structure word must END the name (optionally followed by an id-like
# suffix or a number): `media_lot`, `Batch_ID`, `Plate 3` match; a setting that
# merely starts with the word (`batch_volume_mL`) does not.
BATCH_HINT = re.compile(
    _SEP + r"(batch|lot|plate|block|round|date|week|run[\s_\-]?date|experiment[\s_\-]?date)"
    + r"(?:[\s_\-]?(?:id|no|nr|number|num|index|label|name|code))?(?:[\s_\-#]?\d+)?$",
    re.I,
)
# Scale: may be a real input (a scale-up model), and is ALSO a candidate, so the
# held-out score includes a leave-one-scale-out view when there are 3+ scales.
SCALE_HINT = re.compile(
    _SEP + r"(scale|vessel|working[\s_\-]?volume|reactor[\s_\-]?(?:size|volume)|bioreactor[\s_\-]?(?:size|volume))"
    + r"(?:[\s_\-]?(?:l|ml|litre|liter|liters|litres))?$",
    re.I,
)


def is_batch_like(name: str) -> bool:
    return bool(BATCH_HINT.search(str(name)))


def is_scale_like(name: str) -> bool:
    return bool(SCALE_HINT.search(str(name)))


def replicate_clusters(
    cont: pd.DataFrame | np.ndarray,
    widths: Sequence[float],
    labels: pd.DataFrame | None = None,
    *,
    tol: float = NEAR_DUPLICATE_TOL,
) -> np.ndarray:
    """Group ids: rows within `tol` of each other on EVERY continuous input
    (difference / box width; complete linkage, Chebyshev) and with the same
    categorical labels and the same missing-value pattern.

    `cont` keeps NaN for a missing component (two rows missing different
    components are never merged). `widths` are the design-box widths of the
    continuous inputs (0 or non-finite widths count as 1)."""
    from scipy.cluster.hierarchy import fcluster, linkage

    arr = np.asarray(cont, float)
    n = arr.shape[0] if arr.ndim == 2 else 0
    if n == 0:
        return np.zeros(0, dtype=int)
    w = np.asarray(widths, float).reshape(-1)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    norm = arr / w if arr.shape[1] else arr
    # Blocks that may merge: same NaN pattern and same categorical labels.
    nan_key = ["".join("1" if v else "0" for v in row) for row in np.isnan(norm)] if arr.shape[1] else [""] * n
    if labels is not None and len(labels.columns):
        lab = labels.fillna("").astype(str).agg("\x1f".join, axis=1).tolist()
    else:
        lab = [""] * n
    block_keys = [f"{a}|{b}" for a, b in zip(nan_key, lab)]
    out = np.full(n, -1, dtype=int)
    next_id = 0
    for key in dict.fromkeys(block_keys):
        idx = np.array([i for i, k in enumerate(block_keys) if k == key])
        sub = np.nan_to_num(norm[idx], nan=0.0) if arr.shape[1] else np.zeros((len(idx), 1))
        if len(idx) == 1 or sub.shape[1] == 0:
            ids = np.ones(len(idx), dtype=int)
        else:
            ids = fcluster(linkage(sub, method="complete", metric="chebyshev"), t=float(tol), criterion="distance")
        for c in np.unique(ids):
            out[idx[ids == c]] = next_id
            next_id += 1
    # Canonical ids by first appearance, so the same partition always gets the
    # same labels (and the same GroupKFold folds) as an exact-recipe hash.
    return np.asarray(pd.factorize(out)[0])


def _union(a: Sequence[Any], b: Sequence[Any]) -> np.ndarray:
    """Connected components of 'same a' OR 'same b' (union-find)."""
    n = len(a)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in (a, b):
        first: dict[Any, int] = {}
        for i, k in enumerate(keys):
            if k in first:
                ri, rj = find(i), find(first[k])
                if ri != rj:
                    parent[ri] = rj
            else:
                first[k] = i
    return np.asarray(pd.factorize(np.array([find(i) for i in range(n)]))[0])


def _n_groups(g: Any) -> int:
    return len(pd.unique(np.asarray(pd.Series(list(g)).astype(str))))


def structure_candidates(
    frame: pd.DataFrame,
    *,
    exclude: Sequence[str],
    round_index: Sequence[Any] | None,
    clusters: np.ndarray,
) -> list[dict[str, Any]]:
    """Grouping candidates from the fitted rows `frame` (one row per fitted run).

    Each candidate: `{basis: "column"|"scale_column"|"campaign_round", column,
    labels (raw, as str), groups (merged with the replicate clusters),
    n_levels, merged_with_replicates}`. A column needs 2+ non-null levels and at
    most n/2 levels (a column with a new value on almost every row is an id,
    not a batch). Rows with a null label form their own group."""
    out: list[dict[str, Any]] = []
    n = len(frame)
    excl = {str(c) for c in exclude}
    sources: list[tuple[str, str, list[str]]] = []
    for c in frame.columns:
        name = str(c)
        if name in excl:
            continue
        col = frame[c]
        if isinstance(col, pd.DataFrame):
            continue
        if is_batch_like(name):
            basis = "column"
        elif is_scale_like(name):
            basis = "scale_column"
        else:
            continue
        labels = col.astype(object).where(col.notna(), "").astype(str).str.strip().tolist()
        sources.append((basis, name, labels))
    if round_index is not None and len(round_index) == n:
        sources.append(("campaign_round", "round", [str(r) for r in round_index]))
    for basis, name, labels in sources:
        levels = [v for v in dict.fromkeys(labels) if not is_null_label(v)]
        if len(levels) < 2 or len(levels) > max(2, n // 2):
            continue
        lab = [v if not is_null_label(v) else "\x00null" for v in labels]
        merged = _union(lab, clusters.tolist())
        use_merged = _n_groups(merged) >= MIN_GROUPS_FOR_CV or _n_groups(lab) < MIN_GROUPS_FOR_CV
        out.append({
            "basis": basis,
            "column": name,
            "labels": lab,
            "groups": merged if use_merged else np.asarray(pd.factorize(np.array(lab))[0]),
            "n_levels": len(levels),
            "merged_with_replicates": bool(use_merged),
        })
    return out


def round_index_from_attrs(raw: Any, frame_index: pd.Index, keep_index: pd.Index) -> list[Any] | None:
    """The campaign round of each fitted row (`keep_index`, labels of the
    analysed frame whose original index was `frame_index`), or None when the
    attribute is absent or does not line up with the frame."""
    if raw is None:
        return None
    try:
        seq = list(raw)
    except TypeError:
        return None
    if len(seq) != len(frame_index):
        return None
    pos = frame_index.get_indexer(keep_index)
    if (pos < 0).any():
        return None
    return [seq[int(p)] for p in pos]


def same_partition(a: Any, b: Any) -> bool:
    """True when two labelings put exactly the same rows together."""
    fa = pd.factorize(np.asarray(pd.Series(list(a)).astype(str)))[0]
    fb = pd.factorize(np.asarray(pd.Series(list(b)).astype(str)))[0]
    return bool(len(fa) == len(fb) and np.array_equal(fa, fb))


def union_groups(a: Any, b: Any) -> np.ndarray:
    """Groups joined when they share a label in `a` OR in `b`."""
    return _union([str(v) for v in a], [str(v) for v in b])


def candidate_report(basis: str, column: str | None, groups: Any, rep: Mapping[str, Any] | None,
                     *, n_levels: int | None = None, merged: bool | None = None) -> dict[str, Any]:
    """One entry of `cv_grouping.candidates`."""
    ng = _n_groups(groups)
    entry: dict[str, Any] = {"basis": basis, "column": column, "n_groups": ng}
    if n_levels is not None:
        entry["n_levels"] = int(n_levels)
    if merged is not None:
        entry["merged_with_replicates"] = bool(merged)
    if rep is None:
        entry.update(evaluable=False, spearman=None, ci95=None,
                     reason=f"{ng} group(s); cross-validation needs at least {MIN_GROUPS_FOR_CV}")
        return entry
    rho = rep["spearman"]
    lo, hi = rep["ci95"]
    entry.update(
        evaluable=bool(rho == rho),
        spearman=None if rho != rho else round(float(rho), 3),
        ci95=None if lo != lo else [round(float(lo), 3), round(float(hi), 3)],
        reason=None if rho == rho else "the held-out score could not be computed",
    )
    return entry


def _coverage_gap(rep: Mapping[str, Any], n_runs: int) -> str | None:
    """Why a candidate's held-out score is not a full held-out score, or None."""
    n_oof = int(rep.get("n_oof", 0))
    lo, hi = rep["ci95"]
    if n_oof < n_runs:
        return f"only {n_oof} of {n_runs} runs were held out; a fold too large to train without was skipped"
    if lo != lo or hi != hi:
        return f"the held-out runs span only {int(rep['n_groups'])} group(s), too few for an interval"
    return None


def leak_free_cv(
    primary_rep: dict[str, Any],
    primary_groups: Any,
    primary_basis: str,
    primary_column: str | None,
    candidates: Sequence[Mapping[str, Any]],
    run_cv: Any,
    *,
    n_exact_recipes: int,
    n_clusters: int,
    require_full_coverage: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], Any, str]:
    """Cross-validate every candidate grouping and pick the binding one.

    `run_cv(groups)` returns a `grouped_cv_report` dict. Returns `(cv_grouping
    block, binding report, binding groups, binding basis)`. The binding grouping
    is the evaluable one with the LOWEST held-out Spearman (the primary when
    none is evaluable). Candidates identical to the primary partition are not
    re-run.

    `require_full_coverage` (default on): a candidate counts as evaluable only
    when every one of its runs was held out once and its 95% interval exists.
    Off restores the old rule, where a candidate whose CV skipped a fold could
    bind on a rank over a few rows."""
    entries: list[tuple[str, str | None, Any, dict[str, Any] | None, dict[str, Any]]] = []
    prim_entry = candidate_report(primary_basis, primary_column, primary_groups, primary_rep)
    entries.append((primary_basis, primary_column, primary_groups, primary_rep, prim_entry))
    warnings: list[dict[str, Any]] = []
    for c in candidates:
        g = c["groups"]
        if same_partition(g, primary_groups):
            e = candidate_report(c["basis"], c["column"], g, primary_rep, n_levels=c["n_levels"],
                                 merged=c["merged_with_replicates"])
            e["same_partition_as"] = primary_basis
            entries.append((c["basis"], c["column"], g, primary_rep, e))
            continue
        if _n_groups(g) < MIN_GROUPS_FOR_CV:
            e = candidate_report(c["basis"], c["column"], g, None, n_levels=c["n_levels"],
                                 merged=c["merged_with_replicates"])
            entries.append((c["basis"], c["column"], g, None, e))
            warnings.append({
                "code": "structure_not_checkable",
                "column": c["column"],
                "message": (
                    f"Only {c['n_levels']} level(s) of {c['column']!r}: the held-out score cannot "
                    f"check whether the model carries over to a new {c['column']}. Runs from at "
                    f"least {MIN_GROUPS_FOR_CV} levels are needed."
                ),
            })
            continue
        r = run_cv(g)
        e = candidate_report(c["basis"], c["column"], g, r, n_levels=c["n_levels"],
                             merged=c["merged_with_replicates"])
        if require_full_coverage and e["evaluable"]:
            gap = _coverage_gap(r, len(g))
            if gap is not None:
                # A fold too big to train without (round 0 of a young campaign) is
                # skipped, so the score is a rank over a handful of rows - at 3
                # rows only +/-0.5 and +/-1 exist - and would bind as the lowest.
                e.update(evaluable=False, reason=gap)
                warnings.append({
                    "code": "structure_not_checkable",
                    "column": c["column"],
                    "message": (
                        f"Holding out each {c['column']!r} group does not give an honest held-out score yet "
                        f"({gap}), so it does not drive the reliability verdict. More runs in the "
                        f"smaller {c['column']!r} groups are needed."
                    ),
                })
        entries.append((c["basis"], c["column"], g, r, e))
    evaluable = [t for t in entries if t[4]["evaluable"]]
    binding = min(evaluable, key=lambda t: float(t[3]["spearman"])) if evaluable else entries[0]  # type: ignore[index]
    b_basis, b_col, b_groups, b_rep, b_entry = binding
    if b_rep is None:
        b_rep = primary_rep
    n_used = _n_groups(b_groups)
    if n_used < MIN_GROUPS_FOR_CV:
        warnings.append({
            "code": "too_few_cv_groups",
            "message": (
                f"Only {n_used} independent group(s) of runs: there is no honest held-out score, "
                "so the model's ranking cannot be checked. Add runs of different recipes."
            ),
        })
    elif n_used < FEW_GROUPS_WARN:
        warnings.append({
            "code": "few_cv_groups",
            "message": (
                f"The held-out score rests on only {n_used} groups of runs, so it is coarse and "
                "its interval is wide."
            ),
        })
    if n_clusters < n_exact_recipes:
        warnings.append({
            "code": "near_duplicate_recipes_grouped",
            "message": (
                f"{n_exact_recipes} distinct recipes form {n_clusters} groups of near-identical settings "
                f"(every input within {NEAR_DUPLICATE_TOL:g} of its range). Each group is held out together, "
                "so a held-out run never has a near-twin in training. Replicates recorded as run fall in "
                "the same group; the replicate noise model still counts exact matches only."
            ),
        })
    for t in entries:
        t[4]["used"] = t is binding
    block = {
        "used": {"basis": b_basis, "column": b_col, "n_groups": n_used,
                 **({"merged_with_replicates": b_entry["merged_with_replicates"]}
                    if "merged_with_replicates" in b_entry else {})},
        "candidates": [t[4] for t in entries],
        "rule": (
            "every evaluable grouping (replicate clusters or the declared group, plus batch, lot, plate, "
            "date, round and scale columns and campaign rounds) is cross-validated; the lowest held-out "
            "Spearman drives cv_spearman, the reliability verdict and the conformal calibration"
        ),
        "replicate_tolerance": NEAR_DUPLICATE_TOL,
        "n_exact_recipes": int(n_exact_recipes),
        "n_replicate_clusters": int(n_clusters),
        "min_groups": MIN_GROUPS_FOR_CV,
        "warnings": warnings,
    }
    return block, b_rep, b_groups, b_basis


__all__ = [
    "BATCH_HINT", "FEW_GROUPS_WARN", "MIN_GROUPS_FOR_CV", "NEAR_DUPLICATE_TOL", "ROUND_ATTR", "SCALE_HINT",
    "candidate_report", "is_batch_like", "is_scale_like", "leak_free_cv", "replicate_clusters",
    "round_index_from_attrs", "same_partition", "structure_candidates", "union_groups",
]
