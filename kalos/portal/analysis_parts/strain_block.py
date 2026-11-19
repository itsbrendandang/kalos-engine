"""The analysis-side clone / strain blocks (docs/CLONES.md, STRAIN_VIEWS P1-P3).

Builds, from the rows the analysis already holds and the fitted proposal GP:

- `strain_summary`: per-clone MEASURED statistics for every outcome column
  (n, recipes, mean, 95% t-interval only when n >= 3), the process-adjusted
  clone effect (out-of-fold residuals of a process-only GP, empirical-Bayes
  shrinkage, data weight), the GP's PREDICTED value per clone at one reference
  process, confounding / balance flags, the clone x input interaction screen,
  the predicted optimum per clone, leave-one-clone-out, and every refusal with a
  reason. Measured, derived and predicted values carry `basis` and are never
  mixed in one field.
- `slices_by_level`: the R2 slices of the same GP with the clone set to each
  level (predicted only; measured runs are never in the payload).
- `oof_levels`: the clone label of each `oof` row, aligned.

The statistics themselves live in `kalos.core.strain` (numpy/scipy). The
surrogate is only touched through `.posterior`; the two extra GP evaluations
(the process-only CV and leave-one-clone-out) are run by the caller AFTER every
proposal-path fit so they cannot move the proposals (seeded RNG order).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos.core import strain as st
from kalos.portal.analysis_parts.slices import compute_slices
from kalos.portal.analysis_parts.strain_roles import (
    ROUND_NAME_HINT,
    SCALE_NAME_HINT,
    StrainPlan,
    block_columns,
    normalized_labels,
)

logger = logging.getLogger(__name__)

RUN_ID_NAME = re.compile(
    r"^(?:run|run[ _.-]?(?:id|no|number)|id|sample[ _.-]?id|experiment[ _.-]?id)$", re.I
)
GP_REFERENCE_COVERAGE = 0.9
OPTIMUM_SHIFT_FRAC = 0.25  # predicted argmaxes this far apart (fraction of the range) are "different"
UNAVAILABLE_NO_COLUMN = (
    "no clone column: declare one (strain_column in the campaign setup, or roles.strain) "
    "to see per-clone results"
)


def _r(x: Any, nd: int = 4) -> float | None:
    if x is None:
        return None
    f = float(x)
    return round(f, nd) if np.isfinite(f) else None


def _refusal(code: str, reason: str, fix: str | None, level: str | None = None, **extra: Any) -> dict:
    out: dict[str, Any] = {"code": code, "level": level, "reason": reason, "what_would_fix_it": fix}
    out.update(extra)
    return out


def unavailable(plan: StrainPlan | None) -> dict[str, Any]:
    """`strain_summary` when there is no clone column (or it cannot be used)."""
    reason = UNAVAILABLE_NO_COLUMN
    refusals: list[dict] = []
    if plan is not None:
        refusals = list(plan.refusals)
        if plan.unavailable_reason:
            reason = plan.unavailable_reason
    return {
        "available": False, "reason": reason, "column": None, "source": None,
        "policy": None, "fixed_level": None, "reference_level": None, "levels": [],
        "refusals": refusals,
    }


# --- policy: which level proposals hold ----------------------------------------- #
def choose_fixed_level(
    plan: StrainPlan, levels: Sequence[str], n_runs: Mapping[str, int],
) -> tuple[str | None, str | None, dict | None]:
    """`(level, source, refusal)` for policy `fixed`. A requested level that has
    no runs is refused (`clone_not_in_training`) when the user named it; a
    registry lead with no runs falls back to the clone with the most runs."""
    if plan.policy != "fixed" or not levels:
        return None, None, None
    req = plan.requested_level
    if req is not None:
        if req in levels:
            return req, plan.requested_level_source, None
        if plan.requested_level_source != "registry_lead":
            return None, None, _refusal(
                "clone_not_in_training",
                "the fixed clone level has no runs in this data; Kalos never predicts an unseen "
                "clone (the categorical kernel has nothing to say about a new label), so "
                "proposals are withheld",
                "choose a clone that has runs (strain_level), or run the new clone first",
            )
    best = sorted(levels, key=lambda lv: (-int(n_runs.get(lv, 0)), lv))[0]
    return best, "most_runs", None


def held_covariate_values(
    plan: StrainPlan, X: np.ndarray, cont_names: Sequence[str], bounds: np.ndarray,
    labels_fit: np.ndarray | None, fixed_level: str | None,
) -> dict[str, dict[str, Any]]:
    """Reference value each fitted numeric covariate is held at in proposals:
    the median over the fixed clone's runs (policy fixed), else over all fitted
    runs, clipped into the design box."""
    out: dict[str, dict[str, Any]] = {}
    for c in plan.covariates_numeric:
        if c not in cont_names:
            continue
        j = list(cont_names).index(c)
        col = X[:, j]
        basis = "median of the fitted runs"
        if fixed_level is not None and labels_fit is not None:
            mask = labels_fit == fixed_level
            if mask.any():
                col = col[mask]
                basis = "median of the fitted runs on the fixed clone"
        v = float(np.clip(np.median(col), bounds[0, j], bounds[1, j]))
        out[c] = {"dim": j, "value": round(v, 6), "basis": basis}
    return out


# --- predicted pieces (need the live surrogate) -------------------------------- #
def predicted_blocks(
    s: Any,
    *,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    cont_feats: Sequence[Any],
    kept_cats: Sequence[Any],
    cat_dims_map: dict,
    bounds: np.ndarray,
    drv: Sequence[dict],
    replicate_aware: bool,
    incumbent: float,
    strain_dim: int,
    levels: Sequence[str],
    n_runs: Mapping[str, int],
    process_names: Sequence[str],
    interval_fields: Callable[[np.ndarray, np.ndarray, np.ndarray], list[dict]],
    column: str,
) -> dict[str, Any]:
    """Per-clone slices, the GP at one reference process per clone, and the
    predicted optimum per clone and input. All `basis: "predicted"`."""
    slices: dict[str, list[dict]] = {}
    skipped: list[dict] = []
    gp_ref: dict[str, dict] = {}
    for code, lv in enumerate(levels):
        n = int(n_runs.get(lv, 0))
        if n < st.MIN_RUNS_CI:
            slices[lv] = []
            skipped.append({"level": lv, "reason": "clone_too_few_runs", "n_runs": n,
                            "message": f"slice not shown: n={n} < {st.MIN_RUNS_CI} runs on this clone"})
            continue
        out = compute_slices(
            s, X_fit, y_fit, cont_feats, kept_cats, cat_dims_map, bounds, drv, replicate_aware,
            incumbent=incumbent, reference_codes={strain_dim: code},
        )
        entries = []
        for sl in out.get("slices") or []:
            if str(sl.get("feature")) not in set(process_names):
                continue  # covariates are held, not swept per clone
            entries.append({
                **sl, "level": lv, "band_kind": "gp_posterior", "calibrated": False,
            })
        slices[lv] = entries
        if out.get("reason"):
            skipped.append({"level": lv, "reason": "slice_failed", "n_runs": n,
                            "message": str(out["reason"])})

    # The GP at one shared reference process (the incumbent's settings) per clone.
    from kalos.portal.analysis_parts.slices import reference_point

    ref_vec, ref_recipe, _basis = reference_point(
        X_fit, y_fit, cont_feats, kept_cats, cat_dims_map, replicate_aware, incumbent,
    )
    at = {str(c): ref_recipe.get(str(c)) for c in cont_feats}
    eligible = [lv for lv in levels if int(n_runs.get(lv, 0)) >= st.MIN_RUNS_CI]
    if eligible:
        pts = np.tile(ref_vec, (len(eligible), 1))
        for i, lv in enumerate(eligible):
            pts[i, strain_dim] = float(list(levels).index(lv))
        mean, sd = s.posterior(pts)
        mean = np.asarray(mean, float).reshape(-1)
        sd = np.asarray(sd, float).reshape(-1)
        fields = interval_fields(pts, mean, sd)
        for i, lv in enumerate(eligible):
            iv = next(
                (v for v in fields[i].get("intervals", [])
                 if abs(float(v.get("coverage", 0)) - GP_REFERENCE_COVERAGE) < 1e-9),
                None,
            )
            entry: dict[str, Any] = {
                "basis": "predicted", "pred": _r(mean[i]), "at": at,
                "coverage": GP_REFERENCE_COVERAGE, "refusal": None,
            }
            if iv is not None and iv.get("available"):
                entry.update({"lo": iv.get("lo"), "hi": iv.get("hi"),
                              "method": "conformal_normalized", "calibrated_on": "grouped_cv_oof"})
            else:
                z = 1.6448536269514722
                entry.update({"lo": _r(mean[i] - z * sd[i]), "hi": _r(mean[i] + z * sd[i]),
                              "method": "gp_posterior", "calibrated": False,
                              "note": "conformal band unavailable; model-based band, not held-out calibrated"})
            gp_ref[lv] = entry
    for lv in levels:
        if lv not in gp_ref:
            gp_ref[lv] = {"basis": "predicted", "pred": None, "refusal": _refusal(
                "clone_too_few_runs",
                f"not predicted: n={int(n_runs.get(lv, 0))} < {st.MIN_RUNS_CI} runs on this clone",
                f"run at least {st.MIN_RUNS_CI - int(n_runs.get(lv, 0))} more recipe(s) on this clone",
                lv)}

    # Predicted optimum per input and clone, read off the per-clone slices.
    optimum: dict[str, dict[str, dict]] = {}
    flags: list[dict] = []
    for name in process_names:
        per: dict[str, dict] = {}
        curves: dict[str, dict] = {}
        for lv in eligible:
            sl = next((e for e in slices.get(lv, []) if e["feature"] == name), None)
            if sl is None:
                continue
            k = int(np.argmax(sl["mean"]))
            per[lv] = {"basis": "predicted", "argmax": sl["x"][k], "mean": sl["mean"][k],
                       "lo": sl["lo"][k], "hi": sl["hi"][k], "coverage": sl["coverage"]}
            curves[lv] = sl
        if not per:
            continue
        optimum[name] = per
        labs = sorted(per)
        for a_i, a in enumerate(labs):
            for b in labs[a_i + 1:]:
                ca, cb = curves[a], curves[b]
                span = float(ca["range"][1] - ca["range"][0]) or 1.0
                xa, xb = per[a]["argmax"], per[b]["argmax"]
                if abs(xa - xb) <= OPTIMUM_SHIFT_FRAC * span:
                    continue
                ka, kb = int(np.argmax(ca["mean"])), int(np.argmax(cb["mean"]))
                # each clone's band at its own argmax sits above its band at the other's
                a_prefers = ca["lo"][ka] > ca["hi"][kb]
                b_prefers = cb["lo"][kb] > cb["hi"][ka]
                if a_prefers and b_prefers:
                    flags.append({"code": "predicted_optimum_differs_by_clone", "basis": "predicted",
                                  "input": name, "levels": [a, b]})
    return {
        "slices_by_level": {column: slices},
        "slices_by_level_skipped": {column: skipped},
        "gp_at_reference": gp_ref,
        "predicted_optimum_by_clone": {"basis": "predicted", "by_input": optimum, "flags": flags,
                                       "shift_threshold_frac": OPTIMUM_SHIFT_FRAC},
    }


# --- the summary -------------------------------------------------------------- #
def _run_ids(df: pd.DataFrame, anonymize: bool) -> list[str | None]:
    if anonymize:
        return [None] * len(df)
    col = next((c for c in df.columns if RUN_ID_NAME.match(str(c))), None)
    if col is None:
        return [f"row {i + 1}" for i in range(len(df))]
    return [None if pd.isna(v) else str(v) for v in df[col].tolist()]


def _label_column(df: pd.DataFrame, pattern: re.Pattern[str], exclude: set[str]) -> str | None:
    for c in df.columns:
        if str(c) in exclude or not pattern.search(str(c)):
            continue
        series = df[c]
        if isinstance(series, pd.DataFrame):
            continue
        labels = normalized_labels(series)
        if len({v for v in labels.tolist() if v is not None}) >= 2:
            return str(c)
    return None


def build_summary(
    *,
    plan: StrainPlan,
    df: pd.DataFrame,
    target: str,
    outcomes: Sequence[str],
    outcome_units: Mapping[str, str | None],
    outcome_direction: Mapping[str, str],
    keep_index: pd.Index,
    y: np.ndarray,
    X: np.ndarray,
    cont_names: Sequence[str],
    process_names: Sequence[str],
    kept_cats: Sequence[Any],
    features: Sequence[str],
    fixed_level: str | None,
    fixed_level_source: str | None,
    held: Mapping[str, Mapping[str, Any]],
    predicted: Mapping[str, Any] | None,
    maximize: bool,
    seed: int,
    anonymize: bool,
    extra_refusals: Sequence[dict] = (),
) -> dict[str, Any]:
    """The `strain_summary` block, without the GP-heavy parts that run last
    (`adjusted_effect` and `leave_one_clone_out`, see `finish_summary`)."""
    col = plan.column
    assert col is not None
    labels_all = df[col]
    labels_fit = np.asarray(df.loc[keep_index, col].tolist(), dtype=object)
    present = [v for v in labels_all.tolist() if v is not None]
    levels = sorted(set(present))
    run_ids = _run_ids(df, anonymize)
    feat_cols = [c for c in df.columns if str(c) in set(map(str, features))]
    from kalos.core.splits import row_hash_groups

    recipe_all = np.asarray(row_hash_groups(df[feat_cols]) if feat_cols else [str(i) for i in range(len(df))])
    scale_col = _label_column(df, SCALE_NAME_HINT, {col, target, *map(str, cont_names)})
    round_col = next((str(c) for c in df.columns if ROUND_NAME_HINT.search(str(c))), None)
    scales_all = normalized_labels(df[scale_col]) if scale_col else None
    scale_levels = sorted({v for v in scales_all.tolist() if v is not None}) if scales_all is not None else []
    scales_mixed = len(scale_levels) >= 2

    refusals: list[dict] = [*plan.refusals, *extra_refusals]
    out_levels: list[dict] = []
    n_runs_target: dict[str, int] = {}
    for lv in levels:
        mask_lv = (labels_all == lv).to_numpy()
        entry: dict[str, Any] = {"label": lv, "outcomes": {}, "flags": []}
        for oc in outcomes:
            yv = pd.to_numeric(df[oc], errors="coerce").to_numpy(float)
            m = mask_lv & np.isfinite(yv)
            ids = [run_ids[i] for i in np.flatnonzero(m)]
            summ = st.measured_summary(
                yv[m], recipe_key=recipe_all[m], run_ids=ids,
                maximize=maximize if oc == target else True,
            )
            if oc != target:
                summ["best"] = None  # "best" needs a direction; only the target has one
                summ["best_run_id"] = None
            if scales_mixed and scales_all is not None:
                by_scale = {}
                sc = scales_all.to_numpy(object)
                for sv in scale_levels:
                    ms = m & (sc == sv)
                    if ms.any():
                        by_scale[sv] = st.measured_summary(
                            yv[ms], recipe_key=recipe_all[ms],
                            run_ids=[run_ids[i] for i in np.flatnonzero(ms)],
                            maximize=maximize if oc == target else True,
                        )
                summ = {**summ, "mean": None, "sd": None, "median": None, "ci95": None,
                        "ci_reason": "scales_mixed: per-scale summaries in by_scale, never pooled",
                        "by_scale": by_scale}
            entry["outcomes"][oc] = summ
        tgt = entry["outcomes"].get(target) or {}
        n_t = int(tgt.get("n_runs") or 0)
        n_runs_target[lv] = n_t
        entry["n_runs"] = n_t
        entry["n_recipes"] = tgt.get("n_recipes")
        entry["n_replicated_recipes"] = tgt.get("n_replicated_recipes")
        entry["n_fitted"] = int((labels_fit == lv).sum())
        if round_col is not None:
            rv = normalized_labels(df[round_col])[mask_lv]
            entry["rounds"] = sorted({v for v in rv.tolist() if v is not None},
                                     key=lambda v: (len(v), v))
        else:
            entry["rounds"] = None
        if scales_all is not None:
            entry["scales"] = sorted({v for v in scales_all[mask_lv].tolist() if v is not None})
        else:
            entry["scales"] = None
        covs: dict[str, Any] = {}
        for c in [*plan.covariates_numeric, *plan.covariates_unfitted]:
            if c in df.columns:
                vals = pd.to_numeric(df.loc[mask_lv, c], errors="coerce").dropna()
                covs[c] = {"min": _r(vals.min()), "max": _r(vals.max()), "n": len(vals)} if len(vals) else \
                    {"min": None, "max": None, "n": 0}
        for c in plan.covariates_label:
            if c in df.columns:
                vals = [v for v in normalized_labels(df.loc[mask_lv, c]).tolist() if v is not None]
                covs[c] = {"n_distinct": len(set(vals)), "n": len(vals)}
        entry["covariates"] = covs
        if n_t < st.MIN_RUNS_CI:
            entry["flags"].append("clone_too_few_runs")
            refusals.append(_refusal(
                "clone_too_few_runs",
                f"n={n_t} < {st.MIN_RUNS_CI} runs: no interval, no adjusted effect"
                + ("; in factor mode a proposal on this clone is marked as exploration" if plan.policy == "factor" else ""),
                f"run at least {st.MIN_RUNS_CI - n_t} more recipe(s) on this clone", lv, n_runs=n_t,
            ))
        if predicted is not None:
            entry["gp_at_reference"] = predicted["gp_at_reference"].get(lv)
        else:
            entry["gp_at_reference"] = {"basis": "predicted", "pred": None, "refusal": _refusal(
                "clone_not_modeled", "the clone is not a model input in this analysis", None, lv)}
        out_levels.append(entry)

    reference = fixed_level if fixed_level in levels else (
        sorted(levels, key=lambda lv: (-n_runs_target.get(lv, 0), lv))[0] if levels else None
    )
    if fixed_level is not None and n_runs_target.get(fixed_level, 0) < st.MIN_RUNS_CI:
        refusals.append(_refusal(
            "fixed_clone_too_few_runs",
            "few runs on the chosen clone; proposals are still generated and lean on the other "
            "clones for the shape of the process response",
            "run more recipes on the fixed clone", fixed_level,
            n_runs=n_runs_target.get(fixed_level, 0),
        ))
    n_with3 = sum(1 for v in n_runs_target.values() if v >= st.MIN_RUNS_CI)

    # Balance, confounding, interaction: over the FITTED rows.
    Xp = X[:, [list(cont_names).index(c) for c in process_names]] if process_names else np.empty((len(y), 0))
    Xbal = X[:, : len(cont_names)]
    balance: dict[str, Any] = st.process_balance(Xbal, labels_fit, list(cont_names)) if len(cont_names) else \
        {"flag": False, "by_input": {}, "messages": []}
    blocks = block_columns(df.loc[keep_index], target=target, plan=plan,
                           kept_cats=list(kept_cats), features=list(features))
    conf = st.confounding(labels_fit, {k: v.to_numpy(object) for k, v in blocks.items()})
    not_sep = dict(conf.get("not_separable") or {})
    if process_names:
        inter = st.interaction_screen(Xp, y, labels_fit, list(process_names), seed=seed)
    else:
        inter = {"rows": {}, "cells": {}, "thin_rule": {}, "flags": [],
                 "note": "no continuous process inputs to screen"}
    for name, row in (inter.get("rows") or {}).items():
        if row.get("status") == "insufficient":
            refusals.append(_refusal("interaction_insufficient", str(row.get("reason")),
                                     "more runs per clone spread over this input's range",
                                     None, input=name))
    for lv, cols in not_sep.items():
        refusals.append(_refusal(
            "clone_effect_not_separable",
            "this clone's runs share no level of " + ", ".join(cols) + " with any other clone, "
            "so its difference cannot be separated from that column; the adjusted effect is withheld",
            "run this clone alongside another clone in the same " + ", ".join(cols), lv, columns=list(cols),
        ))
    by_lv = {e["label"]: e for e in out_levels}
    for lv, cols in not_sep.items():
        if lv in by_lv:
            by_lv[lv]["flags"].extend(f"clone_confounded_with:{c}" for c in cols)
    for name, info in (balance.get("by_input") or {}).items():
        for lv, cell in (info.get("by_level") or {}).items():
            smd, cov = cell.get("smd"), cell.get("coverage")
            if lv in by_lv and int(cell.get("n") or 0) >= st.MIN_RUNS_CI and (
                (smd is not None and abs(smd) > st.SMD_FLAG)
                or (cov is not None and cov < st.COVERAGE_FLAG)
            ):
                flag = f"clone_process_imbalance:{name}"
                if flag not in by_lv[lv]["flags"]:
                    by_lv[lv]["flags"].append(flag)

    if round_col is not None:
        keep_mask = np.isfinite(pd.to_numeric(df[target], errors="coerce").to_numpy(float))
        lab = labels_all.to_numpy(object)
        rounds_lab = normalized_labels(df[round_col]).to_numpy(object)
        ok = keep_mask & np.array([v is not None for v in lab]) & np.array([v is not None for v in rounds_lab])
        by_round: dict[str, Any] = {
            "available": True, "column": round_col, "basis": "measured",
            "rounds": st.by_round_best(
                pd.to_numeric(df[target], errors="coerce").to_numpy(float)[ok],
                lab[ok], rounds_lab[ok], maximize=maximize,
            ),
        }
    else:
        by_round = {"available": False, "column": None,
                    "reason": "no round column in the sheet (campaign runs record no round label)"}

    scale_block = {
        "column": scale_col, "levels": scale_levels, "mixed": scales_mixed,
        "note": ("per-clone outcomes are split by scale and never pooled across scales"
                 if scales_mixed else None),
    }
    return {
        "available": True,
        "reason": None,
        "column": col,
        "source": plan.source,
        "policy": plan.policy,
        "modeled": bool(plan.modeled),
        "numeric_looking_labels": bool(plan.numeric_labels),
        "fixed_level": fixed_level,
        "fixed_level_source": fixed_level_source,
        "fixed_level_note": _fixed_level_note(fixed_level, fixed_level_source, n_runs_target),
        "reference_level": reference,
        "n_levels": len(levels),
        "n_levels_with_3_runs": n_with3,
        "target": target,
        "outcome_units": dict(outcome_units),
        "outcome_direction": dict(outcome_direction),
        "levels": out_levels,
        "held_in_proposals": {k: {"value": v["value"], "basis": v["basis"]} for k, v in held.items()},
        "process_balance": balance,
        "confounded_with": conf.get("columns") or [],
        "scale": scale_block,
        "interaction": inter,
        "predicted_optimum_by_clone": (predicted or {}).get("predicted_optimum_by_clone"),
        "by_round": by_round,
        # filled by finish_summary (GP-heavy, runs last)
        "adjusted_method": None,
        "tau": None, "tau2": None, "i2": None,
        "leave_one_clone_out": None,
        "refusals": refusals,
        "_not_separable": not_sep,
        "_n_runs": n_runs_target,
    }


def _fixed_level_note(
    level: str | None, source: str | None, n_runs: Mapping[str, int],
) -> str | None:
    """Say plainly how a default fixed clone was chosen: by run count, never by
    a performance score, and by label order on a tie."""
    if level is None or source != "most_runs":
        return None
    top = max(n_runs.values(), default=0)
    tied = sorted(lv for lv, n in n_runs.items() if n == top)
    if len(tied) > 1:
        return (f"no strain_level and no registry lead: {len(tied)} clones tie on {top} runs, so the "
                "first label in sort order is held (not a ranking); set strain_level to choose")
    return ("no strain_level and no registry lead: the clone with the most runs is held "
            "(not a ranking); set strain_level to choose")


def finish_summary(
    summary: dict[str, Any],
    *,
    labels_fit: np.ndarray,
    oof_index: Sequence[int] | None,
    oof_resid: np.ndarray | None,
    aux_reason: str | None,
    loco: dict[str, Any],
    oof_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Adds the process-adjusted clone effects (empirical Bayes on out-of-fold
    residuals of a process-only GP), leave-one-clone-out and the per-clone OOF
    coverage, then drops the internal keys."""
    not_sep: dict[str, list[str]] = summary.pop("_not_separable", {})
    n_runs: dict[str, int] = summary.pop("_n_runs", {})
    ref = summary.get("reference_level")
    levels = summary["levels"]
    eb: dict[str, Any] | None = None
    top_refusal: dict | None = None
    if oof_resid is None or oof_index is None:
        top_refusal = _refusal("clone_effect_not_estimated", aux_reason or "no residuals", None)
    elif ref in not_sep:
        top_refusal = _refusal(
            "clone_effect_not_separable",
            "the reference clone is confounded with " + ", ".join(not_sep[ref])
            + "; no clone effect relative to it can be separated",
            "run the reference clone alongside the others", ref)
    else:
        lab = np.asarray([labels_fit[int(i)] for i in oof_index], dtype=object)
        ok = np.array([v is not None and v not in not_sep for v in lab])
        eb = st.eb_clone_effects(np.asarray(oof_resid, float)[ok], lab[ok], reference=ref)
        if eb.get("refusals"):
            summary["refusals"].extend(eb["refusals"])
        top = eb.get("refusal")
        if top:
            top_refusal = top
    if top_refusal is not None:
        summary["refusals"].append(top_refusal)
    eb_levels = (eb or {}).get("levels") or {}
    for e in levels:
        lv = e["label"]
        if lv in not_sep:
            e["adjusted_effect"] = {
                "basis": "derived", "estimate": None, "ci95": None,
                "refusal": _refusal("clone_effect_not_separable",
                                    "confounded with " + ", ".join(not_sep[lv]), None, lv),
            }
        elif lv in eb_levels and eb_levels[lv].get("estimate") is not None:
            e["adjusted_effect"] = {**eb_levels[lv], "relative_to": ref}
        else:
            refusal = (eb_levels.get(lv) or {}).get("refusal") or top_refusal or _refusal(
                "clone_too_few_runs", f"n={n_runs.get(lv, 0)} < {st.MIN_RUNS_EFFECT}", None, lv)
            e["adjusted_effect"] = {"basis": "derived", "estimate": None, "ci95": None,
                                    "refusal": refusal}
        ae = e["adjusted_effect"]
        if ae.get("mostly_pooled") and "mostly_pooled" not in e["flags"]:
            e["flags"].append("mostly_pooled")
        # The two methods disagree in sign: derived vs predicted, flagged, never merged.
        gp = e.get("gp_at_reference") or {}
        gp_ref = next((x.get("gp_at_reference") or {} for x in levels if x["label"] == ref), {})
        est = ae.get("estimate")
        if (
            lv != ref and est is not None and gp.get("pred") is not None
            and gp_ref.get("pred") is not None
        ):
            diff = float(gp["pred"]) - float(gp_ref["pred"])
            if est * diff < 0:
                e["flags"].append("clone_effect_methods_disagree")
        e["oof_coverage"] = oof_coverage.get(lv)
    if eb is not None:
        summary.update({
            "adjusted_method": eb.get("method"),
            "tau": eb.get("tau"), "tau2": eb.get("tau2"), "i2": eb.get("i2"),
        })
    summary["adjusted_note"] = (
        "clone effect at these process conditions (out-of-fold residuals of a process-only GP, "
        "empirical-Bayes shrinkage toward the mean clone); approximate 95% interval; an "
        "association, not a ranking of clones"
    )
    summary["leave_one_clone_out"] = loco
    return summary


def leave_one_clone_out_plan(n_runs: Mapping[str, int]) -> str | None:
    """None when leave-one-clone-out may run, else the reason it is not reported."""
    big = [lv for lv, n in n_runs.items() if n >= st.LOCO_MIN_RUNS]
    if len(big) < st.LOCO_MIN_CLONES:
        return (f"need at least {st.LOCO_MIN_CLONES} clones with at least {st.LOCO_MIN_RUNS} runs "
                f"each; {len(big)} qualify")
    return None


def oof_coverage_by_level(
    labels_oof: Sequence[Any], oof_a: Sequence[float], oof_p: Sequence[float],
    oof_s: Sequence[float], q_norm: float | None, coverage: float, min_n: int = 5,
) -> dict[str, Any]:
    """Empirical OOF coverage of the normalized conformal band, per clone (n >= 5)."""
    out: dict[str, Any] = {}
    if q_norm is None:
        return out
    a, p, sd = (np.asarray(v, float) for v in (oof_a, oof_p, oof_s))
    lab = np.asarray(list(labels_oof), dtype=object)
    for lv in sorted({v for v in lab.tolist() if v is not None}):
        m = lab == lv
        n = int(m.sum())
        if n < min_n:
            out[lv] = {"n": n, "nominal": coverage, "empirical": None,
                       "reason": f"n={n} < {min_n} held-out runs"}
            continue
        inside = np.abs(a[m] - p[m]) <= q_norm * np.maximum(sd[m], 1e-12)
        out[lv] = {"n": n, "nominal": coverage, "empirical": _r(inside.mean()), "reason": None}
    return out


__all__ = [
    "build_summary",
    "choose_fixed_level",
    "finish_summary",
    "held_covariate_values",
    "leave_one_clone_out_plan",
    "oof_coverage_by_level",
    "predicted_blocks",
    "unavailable",
]
