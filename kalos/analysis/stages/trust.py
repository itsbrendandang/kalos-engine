"""Stage 2 (trust): constraint surrogate, leak-free grouped CV, conformal calibration, gates, verdict.

Torch-free at import time: all `kalos.core.*` ML imports are deferred inside
`evaluate_trust`. Preserves the exact RNG consumption order of `_analyze_body`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kalos.analysis.stages.prepare import PreparedAnalysis
from kalos.core.conformal import q_from_residuals
from kalos.core.splits import row_hash_groups
from kalos.domains import ColumnRoles
from kalos.portal.analysis_parts.cv_grouping import (
    MIN_GROUPS_FOR_CV,
    leak_free_cv,
    replicate_clusters,
    round_index_from_attrs,
    same_partition,
    structure_candidates,
    union_groups,
)
from kalos.portal.analysis_parts.intervals import (
    calibrate_normalized,
    calibrate_selection_aware,
)
from kalos.portal.analysis_parts.policy import (
    CONFORMAL_ALPHA,
    CONSTRAINT_MIN_ROWS,
    CV_N_REPEATS,
    CV_TOPK_K,
    GATE_MIN_N_INFEASIBLE,
    GROUP_MEAN_BASELINE_MIN_GROUPS,
    INTERVAL_LEVELS,
    LOGO_MAX_GROUPS,
    RELIABILITY_SPEARMAN_FLOOR,
)
from kalos.portal.analysis_parts.trust_gate import clears_floor as trust_clears_floor
from kalos.portal.cells import cell_labels, to_numbers
from kalos.portal.config import cv_workers_setting, feas_onehot_cat_setting
from kalos.portal.verdict import model_verdict, refusal_for


@dataclass
class TrustEvaluation:
    """Outputs of stage 2 (`evaluate_trust`), consumed by propose, explain, and assemble."""

    constraint_surrogate: Any
    constraint_col: str | None
    constraint_floor_val: float | None
    constraint_applied: bool
    constraint_report: dict[str, Any]
    recipe_key: Any
    rep_clusters: np.ndarray
    groups: Any
    binding_basis: str
    cv_grouping: dict[str, Any]
    rep: dict[str, Any]
    rho: float
    oof_a: Any
    oof_p: Any
    oof_s: Any
    conformal_q: float | None
    interval_cal: dict[str, Any]
    selection_cal: dict[str, Any]
    ci95: list[float] | None
    calibration: dict[str, Any]
    reliability: dict[str, Any]
    feas: dict[str, Any]
    feas_clf: Any
    n_infeasible: int
    feas_auc: float
    gated: bool
    gate_reason: str
    gm_margin: dict[str, Any]
    promotion: dict[str, Any]
    verdict: dict[str, Any]
    refusal: dict[str, Any] | None
    cv_logo: dict[str, Any]
    cv_topk: dict[str, Any]
    cv_group_mean_baseline: dict[str, Any]


def evaluate_trust(
    prep: PreparedAnalysis,
    *,
    roles: ColumnRoles | None,
    constraint: Mapping[str, Any] | None,
    progress: Callable[[str, float], None],
    gate_min_n_infeasible: int = GATE_MIN_N_INFEASIBLE,
) -> TrustEvaluation:
    """Fit optional constraint surrogate, run leak-free CV, calibration, gates, and verdict."""
    from kalos.core.evaluation import (
        cv_partitions,
        group_mean_baseline_spearman,
        group_mean_margin,
        grouped_cv_report,
        interval_calibration,
        logo_report,
        producer_only_spearman,
        top_k_overlap,
    )
    from kalos.core.feasibility import (
        FeasibilityClassifier,
        feasibility_cv_report,
        feasible_labels,
    )
    from kalos.core.gates import (
        NOT_APPLICABLE_FEW_NON_PRODUCERS,
        NOT_APPLICABLE_WHEN_MINIMIZING,
        GatesConfig,
        check_gates,
    )
    from kalos.core.surrogate import FitError, Surrogate

    df = prep.df
    keep_index = prep.keep_index
    X = prep.X
    y = prep.y
    bounds = prep.bounds
    cat_dims = prep.cat_dims
    cont_feats = prep.cont_feats
    kept_cats = prep.kept_cats
    gcol = prep.gcol
    _minimize = prep.minimize

    constraint_surrogate: Surrogate | None = None
    constraint_col: str | None = None
    constraint_floor_val: float | None = None
    constraint_applied = False
    if constraint is None:
        constraint_report: dict[str, Any] = {
            "column": None, "floor": None, "n_rows_with_value": 0,
            "applied": False, "reason": "no constraint requested",
        }
    else:
        constraint_col = str(constraint["column"])
        constraint_floor_val = float(constraint["floor"])
        if constraint_col not in df.columns:
            constraint_report = {
                "column": constraint_col, "floor": constraint_floor_val,
                "n_rows_with_value": 0, "applied": False,
                "reason": f"constraint column {constraint_col!r} not found in the uploaded sheet",
            }
        else:
            cy_all = to_numbers(df.loc[keep_index, constraint_col])
            c_mask = cy_all.notna().to_numpy()
            n_rows_with_value = int(c_mask.sum())
            if n_rows_with_value < CONSTRAINT_MIN_ROWS:
                constraint_report = {
                    "column": constraint_col, "floor": constraint_floor_val,
                    "n_rows_with_value": n_rows_with_value, "applied": False,
                    "reason": (
                        f"only {n_rows_with_value} rows have a numeric "
                        f"{constraint_col!r} value; need at least {CONSTRAINT_MIN_ROWS} "
                        "to fit a constraint model"
                    ),
                }
            else:
                Xc = X[c_mask]
                yc = cy_all.to_numpy(float)[c_mask]
                try:
                    constraint_surrogate = Surrogate().fit(Xc, yc, bounds=bounds, cat_dims=cat_dims)
                    constraint_applied = True
                    constraint_report = {
                        "column": constraint_col, "floor": constraint_floor_val,
                        "n_rows_with_value": n_rows_with_value, "applied": True, "reason": None,
                    }
                except FitError as exc:
                    constraint_report = {
                        "column": constraint_col, "floor": constraint_floor_val,
                        "n_rows_with_value": n_rows_with_value, "applied": False,
                        "reason": f"constraint model could not be fit: {exc}",
                    }

    recipe_gf = prep.Xc_raw.copy()
    for c in kept_cats:
        recipe_gf[c] = cell_labels(df.loc[keep_index, c])
    recipe_key = row_hash_groups(recipe_gf)

    cont_widths = (bounds[1] - bounds[0])[: len(cont_feats)]
    rep_clusters = replicate_clusters(
        prep.Xc_raw.to_numpy(float), cont_widths,
        recipe_gf[[c for c in kept_cats]] if kept_cats else None,
    )
    if gcol:
        groups: Any = df.loc[keep_index, gcol].fillna("").astype(str).tolist()
    else:
        groups = recipe_key
    if not gcol and not same_partition(rep_clusters, recipe_key):
        groups = rep_clusters
    elif gcol and not same_partition(union_groups(groups, rep_clusters), groups):
        merged_gcol = union_groups(groups, rep_clusters)
        if len(set(merged_gcol.tolist())) >= MIN_GROUPS_FOR_CV:
            groups = merged_gcol.tolist()

    cv_workers = cv_workers_setting()
    progress("fitting", 0.15)
    rep = grouped_cv_report(
        X, y, groups=groups, n_splits=5, bounds=bounds, cat_dims=cat_dims,
        n_repeats=CV_N_REPEATS, n_workers=cv_workers,
    )
    primary_basis = (
        ("declared_column" if roles is not None and roles.groups else "group_column") if gcol
        else "replicate_clusters"
    )
    cv_grouping, rep, groups, binding_basis = leak_free_cv(
        rep, groups, primary_basis, gcol,
        structure_candidates(
            df.loc[keep_index],
            exclude=[str(prep.target), *([str(gcol)] if gcol else []),
                     *([str(constraint["column"])] if constraint is not None else [])],
            round_index=round_index_from_attrs(prep.round_attr, prep.frame_index, keep_index),
            clusters=rep_clusters,
        ),
        lambda g: grouped_cv_report(
            X, y, groups=g, n_splits=5, bounds=bounds, cat_dims=cat_dims,
            n_repeats=CV_N_REPEATS, n_workers=cv_workers,
        ),
        n_exact_recipes=len(set(np.asarray(recipe_key).tolist())),
        n_clusters=len(set(rep_clusters.tolist())),
        folds=lambda g: cv_partitions(X, y, g, n_splits=5, n_repeats=CV_N_REPEATS),
    )
    rho = rep["spearman"]
    oof_a, oof_p, oof_s = rep["oof_actual"], rep["oof_pred"], rep["oof_std"]

    resid = np.asarray(oof_a, float) - np.asarray(oof_p, float)
    conformal_q = round(q_from_residuals(resid, alpha=CONFORMAL_ALPHA), 4) if len(resid) else None
    interval_cal = calibrate_normalized(oof_a, oof_p, oof_s, INTERVAL_LEVELS)
    selection_cal = calibrate_selection_aware(
        rep.get("oof_repeats", []), INTERVAL_LEVELS, maximize=not _minimize
    )

    _ci_lo, _ci_hi = rep["ci95"]
    ci95 = (
        None if (rho != rho or _ci_lo != _ci_lo or _ci_hi != _ci_hi)
        else [round(_ci_lo, 3), round(_ci_hi, 3)]
    )

    progress("calibrating", 0.55)
    calibration = interval_calibration(oof_a, oof_p, oof_s)
    prod = producer_only_spearman(oof_a, oof_p)
    floor_check = trust_clears_floor(
        rho=rho, ci95=ci95, floor=RELIABILITY_SPEARMAN_FLOOR,
        n_runs=len(y), n_groups=int(rep["n_groups"]),
    )
    reliability: dict[str, Any] = {
        "spearman": None if rho != rho else round(rho, 3),
        "ci95": ci95,
        "spearman_floor": RELIABILITY_SPEARMAN_FLOOR,
        "clears_floor": floor_check["clears_floor"],
        "clears_floor_rule": floor_check["rule"],
        "judged": floor_check["judged"],
        "verdict": floor_check["verdict"],
        "not_judged_reason": floor_check["not_judged_reason"],
        "producer_spearman": (
            None if prod["spearman"] != prod["spearman"] else round(float(prod["spearman"]), 3)
        ),
        "producer_clears_floor": bool(
            prod["evaluable"] and float(prod["spearman"]) >= RELIABILITY_SPEARMAN_FLOOR
        ),
        "n_producers": int(prod["n_producers"]),
        "producer_threshold": float(prod["threshold"]),
        "ci_excludes_zero": bool(ci95 is not None and ci95[0] > 0),
        "calibration": calibration,
        "unmodeled": [
            "feasibility probability",
            *(
                []
                if calibration["available"]
                else ["calibration (ECE): " + str(calibration["reason"])]
            ),
            "scale-up transfer",
            *(
                []
                if prod["evaluable"]
                else ["producer ranking (too few producing runs to score it separately)"]
            ),
        ],
    }

    reliability["space_filling_advice"] = (
        None
        if reliability["clears_floor"]
        else (
            "The model has not yet shown it can rank unseen runs (CV Spearman "
            f"below the {RELIABILITY_SPEARMAN_FLOOR} floor, its 95% interval reaching 0, "
            "or too few runs to judge), so this batch explores "
            "rather than exploits. On data like this a space-filling design (e.g. "
            "Latin hypercube) is expected to do about as well; the model's "
            "advantage appears once it clears the floor."
        )
    )

    _feas_cat_kw: dict[str, Any] = (
        {"cat_dims": cat_dims, "cat_cardinalities": prep.cat_cardinalities}
        if feas_onehot_cat_setting() and cat_dims and prep.cat_cardinalities
        else {}
    )
    feas = feasibility_cv_report(X, y, groups=groups, **_feas_cat_kw)
    feas_clf = FeasibilityClassifier(**_feas_cat_kw).fit(X, feasible_labels(y))
    n_infeasible = int((feasible_labels(y) == 0).sum())
    feas_auc = feas["auc"]
    _gate_auc_floor = GatesConfig().min_feasibility_auc
    _gate_fail_reasons: list[str] = []
    if not feas_clf.fitted:
        _gate_fail_reasons.append(
            "feasibility classifier did not fit (cold-start fallback: too few "
            "non-producer examples to train on)"
        )
    if n_infeasible < gate_min_n_infeasible:
        _gate_fail_reasons.append(
            f"not zero-inflated enough: n_infeasible={n_infeasible} < {gate_min_n_infeasible}"
        )
    _auc_ok = feas_auc == feas_auc and feas_auc >= _gate_auc_floor
    if not _auc_ok:
        _gate_fail_reasons.append(
            "feasibility CV AUC not measurable on this sheet"
            if feas_auc != feas_auc
            else f"feasibility CV AUC too low to trust: auc={feas_auc:.3f} < {_gate_auc_floor}"
        )
    gated = feas_clf.fitted and n_infeasible >= gate_min_n_infeasible and _auc_ok
    if _minimize:
        _gate_fail_reasons.append("not applied: goal is minimize (the gate favors non-zero targets)")
        gated = False
    gate_reason = (
        "; ".join(_gate_fail_reasons)
        if _gate_fail_reasons
        else (
            f"zero-inflated (n_infeasible={n_infeasible}) with a trustworthy "
            f"feasibility classifier (auc={feas_auc:.3f} >= {_gate_auc_floor}); "
            "gating acquisition by P(feasible)"
        )
    )

    _n_rep_groups = int((pd.Series(recipe_key).value_counts() >= 2).sum())
    if _n_rep_groups < GROUP_MEAN_BASELINE_MIN_GROUPS:
        gm_margin: dict[str, Any] = {
            "evaluable": False, "passed": None,
            "reason": (
                f"only {_n_rep_groups} recipes have 2+ rows; need at least "
                f"{GROUP_MEAN_BASELINE_MIN_GROUPS} to form a group-mean floor"
            ),
        }
    else:
        gm_margin = group_mean_margin(y, recipe_key, rep["oof_index"], oof_p)
    gate_stats: dict[str, Any] = {
        "surrogate_spearman": rho,
        "group_mean_margin": gm_margin,
        "feasibility_auc": feas["auc"],
        "ece": feas["ece"],
        "ece_n": feas["n"] if feas.get("evaluable") else None,
        "brier": feas["brier"],
    }
    if _minimize:
        gate_stats["feasibility_not_applicable"] = NOT_APPLICABLE_WHEN_MINIMIZING
    elif n_infeasible < gate_min_n_infeasible:
        gate_stats["feasibility_not_applicable"] = (
            f"{NOT_APPLICABLE_FEW_NON_PRODUCERS} ({n_infeasible} non-producing runs, "
            f"need {gate_min_n_infeasible})"
        )
    promotion_result = check_gates(gate_stats)
    promotion: dict[str, Any] = {
        "passed": promotion_result.passed,
        "failures": promotion_result.failures,
        "metrics": promotion_result.metrics,
        "summary": promotion_result.summary,
        "not_judged": promotion_result.not_judged,
        "status": promotion_result.status,
        "unmeasured": list(promotion_result.unmeasured),
        "meaning": (
            'passed=false means "not yet shown fit to promote", which '
            "includes the case where a required metric is unmeasurable on "
            "this data (e.g. an all-producer sheet with no non-producers to "
            "score feasibility against). It is not a rejection of the analysis."
        ),
    }

    verdict = model_verdict(
        n_runs=len(y), n_groups=int(rep["n_groups"]), spearman=None if rho != rho else float(rho),
        ci95=ci95, floor=RELIABILITY_SPEARMAN_FLOOR, calibration=calibration,
    )
    refusal = refusal_for(verdict)

    cv_logo: dict[str, Any]
    if gcol is None and binding_basis == "replicate_clusters":
        cv_logo = {
            "spearman": None,
            "n_groups": None,
            "n_oof": None,
            "reason": (
                "no declared/detected group column; leave-one-group-out over "
                "the recipe-hash fallback would just be a costlier copy of "
                "the grouped CV already reported as cv_spearman"
            ),
        }
    else:
        n_logo_groups = len(np.unique(np.asarray(pd.Series(groups).astype(str).values)))
        if n_logo_groups > LOGO_MAX_GROUPS:
            cv_logo = {
                "spearman": None,
                "n_groups": n_logo_groups,
                "n_oof": None,
                "reason": (
                    f"{n_logo_groups} groups exceeds the leave-one-group-out cap "
                    f"of {LOGO_MAX_GROUPS} groups (one GP fit per held-out group)"
                ),
            }
        else:
            logo = logo_report(X, y, groups, bounds=bounds, cat_dims=cat_dims)
            logo_rho = logo["spearman"]
            cv_logo = {
                "spearman": None if logo_rho != logo_rho else round(float(logo_rho), 3),
                "n_groups": logo["n_groups"],
                "n_oof": logo["n_oof"],
                "reason": None,
            }

    _topk_sign = -1.0 if _minimize else 1.0
    topk = top_k_overlap(
        _topk_sign * np.asarray(oof_a, float), _topk_sign * np.asarray(oof_p, float), CV_TOPK_K
    )
    cv_topk: dict[str, Any] = {
        "overlap": None if not topk["evaluable"] else round(float(topk["overlap"]), 3),
        "k": topk["k"],
        "n": topk["n"],
        "direction": "lowest" if _minimize else "highest",
        "evaluable": bool(topk["evaluable"]),
        "reason": topk["reason"],
    }

    n_groups_with_reps = int((pd.Series(recipe_key).value_counts() >= 2).sum())
    cv_group_mean_baseline: dict[str, Any]
    if n_groups_with_reps < GROUP_MEAN_BASELINE_MIN_GROUPS:
        cv_group_mean_baseline = {
            "spearman": None,
            "n_evaluable": 0,
            "n_groups_used": 0,
            "interpretation": None,
            "reason": (
                f"only {n_groups_with_reps} recipes have 2+ rows; need at "
                f"least {GROUP_MEAN_BASELINE_MIN_GROUPS} to compute a "
                "group-mean floor"
            ),
        }
    else:
        gmb = group_mean_baseline_spearman(y, recipe_key)
        if gmb["evaluable"]:
            cv_group_mean_baseline = {
                "spearman": round(float(gmb["spearman"]), 3),
                "n_evaluable": gmb["n_evaluable"],
                "n_groups_used": gmb["n_groups_used"],
                "interpretation": (
                    "A model whose CV Spearman does not clearly beat this "
                    "floor may be recognizing recipes, not modeling the "
                    "process."
                ),
                "reason": None,
            }
        else:
            cv_group_mean_baseline = {
                "spearman": None,
                "n_evaluable": gmb["n_evaluable"],
                "n_groups_used": gmb["n_groups_used"],
                "interpretation": None,
                "reason": gmb["reason"],
            }

    return TrustEvaluation(
        constraint_surrogate=constraint_surrogate,
        constraint_col=constraint_col,
        constraint_floor_val=constraint_floor_val,
        constraint_applied=constraint_applied,
        constraint_report=constraint_report,
        recipe_key=recipe_key,
        rep_clusters=rep_clusters,
        groups=groups,
        binding_basis=binding_basis,
        cv_grouping=cv_grouping,
        rep=rep,
        rho=rho,
        oof_a=oof_a,
        oof_p=oof_p,
        oof_s=oof_s,
        conformal_q=conformal_q,
        interval_cal=interval_cal,
        selection_cal=selection_cal,
        ci95=ci95,
        calibration=calibration,
        reliability=reliability,
        feas=feas,
        feas_clf=feas_clf,
        n_infeasible=n_infeasible,
        feas_auc=feas_auc,
        gated=gated,
        gate_reason=gate_reason,
        gm_margin=gm_margin,
        promotion=promotion,
        verdict=verdict,
        refusal=refusal,
        cv_logo=cv_logo,
        cv_topk=cv_topk,
        cv_group_mean_baseline=cv_group_mean_baseline,
    )
