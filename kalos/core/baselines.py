"""Baselines a claim must beat before it gets credit ("beat the obvious").

Two pure functions, no I/O and no torch:

`carry_forward_rank_agreement`
    The obvious forecast for a scale-up or clone-transfer question is "the
    ranking you already measured at the previous scale". Before a model that
    predicts the larger scale gets credit, it has to rank the units (clones or
    recipes) better than their own earlier-scale measurement does. This
    function measures that bar: the Spearman and top-k agreement between each
    unit's earlier-scale value and its measured value at the target scale,
    with a cluster bootstrap 95% CI. Given the model's out-of-fold predictions
    for the same rows it also reports the paired difference (model minus
    carry-forward) with its own paired CI, and says plainly when the model
    does not clear the bar.

    The earlier-scale value is a BASELINE here, never a model input: a model
    that uses it as a feature mostly re-ranks by it, and its skill reads as the
    model's own. The caller must keep that column out of the feature set.

`random_split_inflation`
    A diagnostic row: what a row-level random K-fold split would have claimed,
    next to the group-aware split's score on the same data. Replicates of one
    group on both sides of a random split grade the model against a near-twin
    it trained on, which inflates every metric. The gap shows how much. It is
    a diagnostic, never the headline number (`role="diagnostic"`,
    `headline=False`).

Both functions refuse with a `code`, a `reason` and an `action` when the data
cannot support an answer; they never return a number that only looks measured.

Idea credit: the carry-forward baseline follows the earlier-scale baseline in
Bioqore/voyager-brain-rebuild@45ccce8:voyager_rebuild/core/evaluation.py
(`_carry_forward_col`). Re-implemented; no code copied (see NOTICE).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .splits import make_splits

MIN_SHARED_UNITS = 5
"""Fewer units measured at both scales than this: refuse (`too_few_shared_units`)."""

MIN_BOOTSTRAP_CLUSTERS = 5
"""Fewer bootstrap clusters than this: report the point value, no CI, no verdict."""

MIN_VERDICT_UNITS = 10
"""Fewer paired units than this: the model comparison is reported, but
`beats_carry_forward` is None (not judged). In a synthetic coverage check the
paired-difference CI covered 0.85 at 5 units and 0.95 at 10 (see the PR)."""

DEFAULT_N_BOOT = 1000
CI_LEVEL = 0.95
_MIN_VALID_BOOT_FRACTION = 0.5
_ROUND = 4

FitPredict = Callable[[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], ArrayLike]
"""`fit_predict(X_train, y_train, X_test) -> predictions for X_test`."""


# --------------------------------------------------------------------------- helpers


def _spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Spearman rho, NaN when undefined (fewer than 3 points or a constant side).
    `np.ptp` decides constancy (exactly 0 on identical values)."""
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    r = float(spearmanr(a, b).statistic)
    return r if math.isfinite(r) else float("nan")


def _top_k_overlap(actual: NDArray[np.float64], pred: NDArray[np.float64], k: int, maximize: bool) -> float:
    """|true top-k & predicted top-k| / k. Stable argsort: ties keep input order."""
    sign = -1.0 if maximize else 1.0
    true_top = set(np.argsort(sign * actual, kind="stable")[:k].tolist())
    pred_top = set(np.argsort(sign * pred, kind="stable")[:k].tolist())
    return len(true_top & pred_top) / k


def _r(x: float) -> float:
    return round(float(x), _ROUND)


def _ci(samples: list[float], n_boot: int) -> list[float] | None:
    if len(samples) < max(1, int(_MIN_VALID_BOOT_FRACTION * n_boot)):
        return None
    tail = 100.0 * (1.0 - CI_LEVEL) / 2.0
    lo, hi = np.percentile(samples, [tail, 100.0 - tail])
    return [_r(lo), _r(hi)]


def _as_1d(x: ArrayLike, name: str, n: int | None = None) -> NDArray[Any]:
    arr = np.asarray(x).reshape(-1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows; expected {n}")
    return arr


def _keys(x: ArrayLike) -> NDArray[Any]:
    return np.asarray(pd.Series(np.asarray(x).reshape(-1)).astype(str).to_numpy())


def _cluster_index(cluster_of_unit: NDArray[Any]) -> tuple[NDArray[Any], list[NDArray[np.intp]]]:
    uniq = np.unique(cluster_of_unit)
    return uniq, [np.flatnonzero(cluster_of_unit == c) for c in uniq]


# --------------------------------------------------------------------------- carry-forward


def carry_forward_rank_agreement(
    earlier: ArrayLike,
    later: ArrayLike,
    units: ArrayLike,
    *,
    clusters: ArrayLike | None = None,
    model_pred: ArrayLike | None = None,
    k: int | None = None,
    maximize: bool = True,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    min_units: int = MIN_SHARED_UNITS,
) -> dict[str, Any]:
    """How well does each unit's earlier-scale value rank the units at the target scale?

    Arguments are per row and aligned:
      - `earlier`: the unit's MEASURED value at the nearest earlier scale (the
        carry-forward forecast). NaN when the unit was not measured there.
      - `later`: the MEASURED outcome at the target scale. NaN when missing.
      - `units`: the linking key (clone ID or recipe ID) that says two rows are
        the same unit. Without a linking key there is no carry-forward; the
        caller reports "not computable", never a fallback.
      - `clusters` (optional): the resampling cluster per row (e.g. campaign or
        batch). Default: each unit is its own cluster. A unit must sit in one
        cluster.
      - `model_pred` (optional): the model's OUT-OF-FOLD prediction at the
        target scale for each row, from a group-aware CV that keeps every unit
        in one fold. In-sample predictions make the comparison meaningless.

    Replicate rows are averaged per unit before ranking (earlier, later and
    model prediction separately, ignoring NaN), so the ranking is over units,
    not rows. A unit is "shared" when both averages are finite.

    `k` defaults to ceil(n_units / 4), clipped to [1, n_units - 1]. `maximize`
    says which end is the top (False for an outcome to minimise). Spearman does
    not depend on direction; top-k does. `top_k_chance` (= k / n_units) is the
    overlap a random ranking scores on average.

    CI: percentile bootstrap over clusters (whole clusters drawn with
    replacement, duplicates kept), `n_boot` draws seeded by `seed`. It is
    reported only with at least `MIN_BOOTSTRAP_CLUSTERS` clusters and when at
    least half the draws are defined; otherwise `*_ci95` is None and
    `ci_reason` says why. At 5 to 10 units the CI is very wide; that is the
    honest answer at that n, not a defect.

    Model comparison (`model`, present when `model_pred` is given): the paired
    difference model minus carry-forward on the units where all three values
    exist, for Spearman and top-k, each with a paired cluster-bootstrap CI.
    `beats_carry_forward` is True only when the lower bound of the Spearman
    difference CI is above 0, False otherwise, and None (not judged) when
    there is no CI or fewer than `MIN_VERDICT_UNITS` (10) paired units (the
    paired CI under-covers below that). `verdict` states the result in one sentence.

    Refusals (`evaluable=False`, with `code`, `reason`, `action`):
    `too_few_shared_units` (fewer than `min_units`, default 5),
    `unit_in_several_clusters`, `constant_values` (a side has no variation).
    Mismatched lengths raise ValueError (a caller bug, not a data condition).
    """
    e = _as_1d(earlier, "earlier").astype(float)
    n_rows = e.shape[0]
    lt = _as_1d(later, "later", n_rows).astype(float)
    u = _keys(_as_1d(units, "units", n_rows))
    c = _keys(_as_1d(clusters, "clusters", n_rows)) if clusters is not None else u
    mp = _as_1d(model_pred, "model_pred", n_rows).astype(float) if model_pred is not None else None
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")

    ci_method = (
        f"percentile bootstrap over clusters ({'units' if clusters is None else 'given clusters'}), "
        f"{int(CI_LEVEL * 100)}% CI, B={int(n_boot)}, seed={int(seed)}"
    )
    out: dict[str, Any] = {
        "evaluable": False, "code": None, "reason": None, "action": None,
        "n_rows": int(n_rows), "n_units": 0, "n_units_earlier_only": 0, "n_units_later_only": 0,
        "n_clusters": 0, "k": None, "maximize": bool(maximize),
        "spearman": None, "spearman_ci95": None,
        "top_k_overlap": None, "top_k_overlap_ci95": None, "top_k_chance": None,
        "ci_method": ci_method, "ci_reason": None, "n_boot_valid": 0,
        "model": None,
    }

    df = pd.DataFrame({"u": u, "c": c, "e": e, "l": lt})
    if mp is not None:
        df["m"] = mp
    per_cluster = df.groupby("u")["c"].nunique()
    multi = per_cluster[per_cluster > 1]
    if len(multi):
        return {**out, "code": "unit_in_several_clusters",
                "reason": f"{len(multi)} unit(s) appear in more than one cluster (e.g. {sorted(multi.index)[:3]})",
                "action": "give each unit one cluster (e.g. the campaign it was screened in), or omit clusters"}

    agg = df.groupby("u", sort=True).agg(
        e=("e", "mean"), l=("l", "mean"), c=("c", "first"),
        **({"m": ("m", "mean")} if mp is not None else {}),
    )
    has_e = np.isfinite(agg["e"].to_numpy())
    has_l = np.isfinite(agg["l"].to_numpy())
    shared = agg[has_e & has_l]
    n_units = len(shared)
    out.update(
        n_units=int(n_units),
        n_units_earlier_only=int((has_e & ~has_l).sum()),
        n_units_later_only=int((~has_e & has_l).sum()),
        n_clusters=int(shared["c"].nunique()),
    )
    if n_units < max(3, int(min_units)):
        return {**out, "code": "too_few_shared_units",
                "reason": f"only {n_units} unit(s) were measured at both scales; need at least {max(3, int(min_units))}",
                "action": "measure more of the same clones or recipes at both scales, linked by one ID"}

    ev = shared["e"].to_numpy(float)
    lv = shared["l"].to_numpy(float)
    rho = _spearman(ev, lv)
    if not math.isfinite(rho):
        return {**out, "code": "constant_values",
                "reason": "the earlier-scale or target-scale values have no variation across the shared units",
                "action": "rank agreement needs units that differ at both scales"}

    kk = int(k) if k is not None else math.ceil(n_units / 4)
    kk = min(max(kk, 1), n_units - 1)
    tk = _top_k_overlap(lv, ev, kk, maximize)
    out.update(evaluable=True, spearman=_r(rho), top_k_overlap=_r(tk), k=kk, top_k_chance=_r(kk / n_units))

    # Model comparison rows: the units where the model prediction exists too.
    mv: NDArray[np.float64] | None = None
    paired_mask = np.zeros(n_units, dtype=bool)
    if mp is not None:
        m_all = shared["m"].to_numpy(float)
        ok = np.isfinite(m_all)
        out["model"] = _model_block(ev[ok], lv[ok], m_all[ok], shared["c"].to_numpy()[ok], kk, maximize, min_units)
        if out["model"]["evaluable"]:
            mv = m_all
            paired_mask = ok
    clusters_u, members = _cluster_index(shared["c"].to_numpy())
    if len(clusters_u) < MIN_BOOTSTRAP_CLUSTERS:
        out["ci_reason"] = (f"only {len(clusters_u)} cluster(s); a bootstrap CI needs at least "
                            f"{MIN_BOOTSTRAP_CLUSTERS}")
        if out["model"] is not None and out["model"]["evaluable"]:
            out["model"].update(beats_carry_forward=None, verdict=(
                "not judged: too few clusters for a confidence interval on the paired difference"))
        return out

    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    tks: list[float] = []
    d_rho: list[float] = []
    d_tk: list[float] = []
    for _ in range(int(n_boot)):
        take = rng.integers(0, len(clusters_u), len(clusters_u))
        idx = np.concatenate([members[t] for t in take])
        r_b = _spearman(ev[idx], lv[idx])
        if math.isfinite(r_b):
            rhos.append(r_b)
        if len(idx) > kk:
            tks.append(_top_k_overlap(lv[idx], ev[idx], kk, maximize))
        if mv is not None:
            pidx = idx[paired_mask[idx]]
            if len(pidx) > kk:
                a = _spearman(mv[pidx], lv[pidx])
                b = _spearman(ev[pidx], lv[pidx])
                if math.isfinite(a) and math.isfinite(b):
                    d_rho.append(a - b)
                    d_tk.append(_top_k_overlap(lv[pidx], mv[pidx], kk, maximize)
                                - _top_k_overlap(lv[pidx], ev[pidx], kk, maximize))
    out["n_boot_valid"] = len(rhos)
    out["spearman_ci95"] = _ci(rhos, n_boot)
    out["top_k_overlap_ci95"] = _ci(tks, n_boot)
    if out["spearman_ci95"] is None:
        out["ci_reason"] = f"only {len(rhos)} of {n_boot} bootstrap draws were defined"
    if mv is not None:
        _judge_model(out["model"], d_rho, d_tk, n_boot)
    return out


def _model_block(
    ev: NDArray[np.float64], lv: NDArray[np.float64], mv: NDArray[np.float64],
    cl: NDArray[Any], k: int, maximize: bool, min_units: int,
) -> dict[str, Any]:
    n = len(lv)
    block: dict[str, Any] = {
        "evaluable": False, "code": None, "reason": None, "action": None, "n_units": int(n),
        "spearman": None, "carry_forward_spearman": None, "spearman_difference": None,
        "spearman_difference_ci95": None, "top_k_overlap": None, "carry_forward_top_k_overlap": None,
        "top_k_difference": None, "top_k_difference_ci95": None, "n_boot_valid": 0,
        "beats_carry_forward": None, "verdict": None,
    }
    need = max(3, int(min_units))
    if n < need or n <= k:
        return {**block, "code": "too_few_shared_units",
                "reason": f"the model has a prediction for only {n} of the shared units; need at least {max(need, k + 1)}",
                "action": "pass out-of-fold predictions for every shared unit",
                "verdict": "not judged: too few units with a model prediction"}
    a, b = _spearman(mv, lv), _spearman(ev, lv)
    if not (math.isfinite(a) and math.isfinite(b)):
        return {**block, "code": "constant_values",
                "reason": "the model predictions or the paired values have no variation",
                "action": "rank agreement needs predictions that differ across units",
                "verdict": "not judged: a Spearman on the paired units is undefined"}
    tm, tc = _top_k_overlap(lv, mv, k, maximize), _top_k_overlap(lv, ev, k, maximize)
    return {**block, "evaluable": True, "spearman": _r(a), "carry_forward_spearman": _r(b),
            "spearman_difference": _r(a - b), "top_k_overlap": _r(tm),
            "carry_forward_top_k_overlap": _r(tc), "top_k_difference": _r(tm - tc)}


def _judge_model(block: dict[str, Any], d_rho: list[float], d_tk: list[float], n_boot: int) -> None:
    if not block["evaluable"]:
        return
    block["n_boot_valid"] = len(d_rho)
    block["spearman_difference_ci95"] = _ci(d_rho, n_boot)
    block["top_k_difference_ci95"] = _ci(d_tk, n_boot)
    ci = block["spearman_difference_ci95"]
    diff = block["spearman_difference"]
    if block["n_units"] < MIN_VERDICT_UNITS:
        block["beats_carry_forward"] = None
        block["verdict"] = (f"not judged: {block['n_units']} paired units; the paired confidence interval "
                            f"is too unreliable below {MIN_VERDICT_UNITS}")
        return
    if ci is None:
        block["beats_carry_forward"] = None
        block["verdict"] = (f"not judged: only {len(d_rho)} of {n_boot} bootstrap draws of the paired "
                            "difference were defined")
        return
    beats = ci[0] > 0.0
    block["beats_carry_forward"] = bool(beats)
    span = f"Spearman difference {diff:+.2f}, 95% CI {ci[0]:+.2f} to {ci[1]:+.2f}"
    block["verdict"] = (
        f"the model ranks the units better than their earlier-scale measurement ({span})"
        if beats else
        f"the model does not beat carrying the earlier-scale measurement forward ({span})"
    )


# --------------------------------------------------------------------------- random-split inflation


def _oof(
    X: NDArray[np.float64], y: NDArray[np.float64],
    splits: list[tuple[NDArray[np.intp], NDArray[np.intp]]], fit_predict: FitPredict,
) -> NDArray[np.float64]:
    pred = np.full(len(y), np.nan)
    for tr, te in splits:
        p = np.asarray(fit_predict(X[tr], y[tr], X[te]), dtype=float).reshape(-1)
        if p.shape[0] != len(te):
            raise ValueError(f"fit_predict returned {p.shape[0]} predictions for {len(te)} test rows")
        pred[te] = p
    return pred


def random_split_inflation(
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    fit_predict: FitPredict,
    *,
    n_splits: int = 5,
    seed: int = 0,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """What a random K-fold split would have claimed, next to the grouped split.

    Runs the same `fit_predict(X_train, y_train, X_test)` under two K-fold
    partitions of the same rows: group-aware (`splits.make_splits`, shuffled
    with `seed`, leakage-checked) and row-level random (`KFold`, shuffled with
    `seed`), with the same number of folds. Reports the pooled out-of-fold
    Spearman of each, `inflation` = random minus grouped, and a paired
    percentile-bootstrap 95% CI of the inflation over groups (whole groups
    resampled, both predictions of a row kept together; at least
    `MIN_BOOTSTRAP_CLUSTERS` groups). The CI is conditional on these two
    partitions: it covers which groups were sampled, not partition-to-partition
    variance.

    `leaked_test_fraction` is the share of random-split test rows whose group
    also had a row in that fold's training set: the mechanism behind the gap.

    Always `role="diagnostic"`, `headline=False`: the grouped score is the
    number to quote; this row only shows what a careless split would have said.

    Refusals (`evaluable=False`, `code`, `reason`, `action`):
    `too_few_groups` (< 3 groups after dropping non-finite `y`),
    `no_replicated_groups` (every group has one row, so a random split cannot
    leak and there is nothing to compare), `undefined_spearman` (constant
    predictions or target). Non-finite `y` rows are dropped and counted.
    Exceptions from `fit_predict` propagate; mismatched lengths raise
    ValueError.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n = X_arr.shape[0]
    y_arr = _as_1d(y, "y", n).astype(float)
    g_arr = _keys(_as_1d(groups, "groups", n))

    keep = np.isfinite(y_arr)
    X_arr, y_arr, g_arr = X_arr[keep], y_arr[keep], g_arr[keep]
    n_used = int(keep.sum())
    uniq, counts = np.unique(g_arr, return_counts=True)
    out: dict[str, Any] = {
        "role": "diagnostic", "headline": False,
        "evaluable": False, "code": None, "reason": None, "action": None,
        "n_rows": n_used, "n_rows_dropped": int(n - n_used),
        "n_groups": len(uniq), "n_replicated_groups": int((counts >= 2).sum()),
        "n_splits": None, "seed": int(seed),
        "grouped_spearman": None, "random_spearman": None, "inflation": None,
        "inflation_ci95": None, "ci_method": None, "ci_reason": None, "n_boot_valid": 0,
        "leaked_test_fraction": None,
    }
    if len(uniq) < 3:
        return {**out, "code": "too_few_groups",
                "reason": f"only {len(uniq)} group(s); a group-aware split needs at least 3",
                "action": "add runs from more recipes or batches"}
    if out["n_replicated_groups"] == 0:
        return {**out, "code": "no_replicated_groups",
                "reason": "every group has one row, so a random split cannot put a group on both sides",
                "action": "nothing to fix: with no replicated groups the random and grouped splits do not differ"}

    k = min(int(n_splits), len(uniq))
    if k < 2:
        return {**out, "code": "too_few_groups", "reason": "fewer than 2 folds are possible",
                "action": "add runs from more recipes or batches"}
    grouped = make_splits(X_arr, y_arr, g_arr, n_splits=k, shuffle=True, random_state=seed)
    if not grouped:
        return {**out, "code": "too_few_groups", "reason": "no group-aware split was possible",
                "action": "add runs from more recipes or batches"}
    k = len(grouped)
    rand = [(tr, te) for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(X_arr)]
    out["n_splits"] = k

    leaked = sum(int(np.isin(g_arr[te], g_arr[tr]).sum()) for tr, te in rand)
    out["leaked_test_fraction"] = _r(leaked / n_used)

    p_g = _oof(X_arr, y_arr, grouped, fit_predict)
    p_r = _oof(X_arr, y_arr, rand, fit_predict)
    ok = np.isfinite(p_g) & np.isfinite(p_r)
    rg, rr = _spearman(p_g[ok], y_arr[ok]), _spearman(p_r[ok], y_arr[ok])
    if not (math.isfinite(rg) and math.isfinite(rr)):
        return {**out, "code": "undefined_spearman",
                "reason": "a pooled out-of-fold Spearman is undefined (constant predictions or target)",
                "action": "check that the model predictions vary across rows"}
    out.update(evaluable=True, grouped_spearman=_r(rg), random_spearman=_r(rr), inflation=_r(rr - rg))

    pg, pr, yy, gg = p_g[ok], p_r[ok], y_arr[ok], g_arr[ok]
    g_u, members = _cluster_index(gg)
    out["ci_method"] = (f"paired percentile bootstrap over groups, {int(CI_LEVEL * 100)}% CI, "
                        f"B={int(n_boot)}, seed={int(seed)}")
    if len(g_u) < MIN_BOOTSTRAP_CLUSTERS:
        out["ci_reason"] = f"only {len(g_u)} group(s); a bootstrap CI needs at least {MIN_BOOTSTRAP_CLUSTERS}"
        return out
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(int(n_boot)):
        take = rng.integers(0, len(g_u), len(g_u))
        idx = np.concatenate([members[t] for t in take])
        a, b = _spearman(pr[idx], yy[idx]), _spearman(pg[idx], yy[idx])
        if math.isfinite(a) and math.isfinite(b):
            diffs.append(a - b)
    out["n_boot_valid"] = len(diffs)
    out["inflation_ci95"] = _ci(diffs, n_boot)
    if out["inflation_ci95"] is None:
        out["ci_reason"] = f"only {len(diffs)} of {n_boot} bootstrap draws were defined"
    return out


__all__ = [
    "CI_LEVEL",
    "DEFAULT_N_BOOT",
    "MIN_BOOTSTRAP_CLUSTERS",
    "MIN_SHARED_UNITS",
    "MIN_VERDICT_UNITS",
    "FitPredict",
    "carry_forward_rank_agreement",
    "random_split_inflation",
]
