"""Stage 5 (assemble): build provenance, combine stage outputs, apply refusals and anonymization.

Torch-free at import time.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from kalos import __version__ as ENGINE_VERSION
from kalos.analysis.stages.explain import (
    DriversResult,
    PostFitDiagnostics,
    SurrogateExplanation,
)
from kalos.analysis.stages.prepare import PreparedAnalysis
from kalos.analysis.stages.propose import ProposalResult
from kalos.analysis.stages.trust import TrustEvaluation
from kalos.domains import ColumnRoles, DomainProfile
from kalos.portal.analysis_parts.anonymize import _anonymize_result
from kalos.portal.analysis_parts.categorical_effects import (
    skipped as categorical_skipped,
)
from kalos.portal.analysis_parts.columns import is_outcome_name
from kalos.portal.analysis_parts.diagnostics import (
    _data_drops_block,
    _data_findings,
    _with_data_findings,
    derived_output_findings,
)
from kalos.portal.analysis_parts.identifiers import is_identifier_name
from kalos.portal.analysis_parts.intervals import (
    conformal_band_caveat,
    interval_method_summary,
    proposal_interval_fields,
)
from kalos.portal.analysis_parts.noise_blocks import _noise_block
from kalos.portal.analysis_parts.policy import (
    ANALYZE_SEED,
    CONFORMAL_ALPHA,
    DRIVER_FDR_Q,
    DRIVER_TOP_K,
    INTERVAL_LEVELS,
)
from kalos.portal.analysis_parts.proposals import (
    _annotate,
    nearest_training_distance,
    training_nn_distance_median,
)
from kalos.portal.analysis_parts.strain_state import covariates_block
from kalos.portal.analysis_parts.trust_gate import label_proposals, proposal_trust
from kalos.portal.analysis_parts.units import target_unit_block
from kalos.portal.validate import column_provenance, provenance_dicts
from kalos.portal.verdict import refusal_notice
from kalos.validation import report_dict


def assemble_result(
    prep: PreparedAnalysis,
    trust: TrustEvaluation,
    drivers: DriversResult,
    prop: ProposalResult,
    explanation: SurrogateExplanation,
    post_fit: PostFitDiagnostics,
    *,
    roles: ColumnRoles | None,
    profile: DomainProfile,
    anonymize: bool,
    user_target_unit: str | None,
    setup: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Combine all stage outputs into the analysis response dictionary."""
    from kalos.core.optimize import MAX_MIXED_COMBOS

    kept_features = [str(c) for c in prep.cont_feats] + [str(c) for c in prep.kept_cats]
    provenance = provenance_dicts(
        column_provenance(
            prep.df,
            target=str(prep.target),
            features=kept_features,
            numeric_cols=[str(c) for c in prep.num],
            is_identifier=lambda name: is_identifier_name(name, profile.id_hint),
            outcome_hint=profile.outcome_hint,
            is_outcome=lambda name: is_outcome_name(name, profile),
            constant_on_fitted_rows=[str(c) for c in prep.constant_on_fitted]
            + [str(c) for c in prep.constant_cats],
            declared={str(c) for c in prep.declared},
            declared_ids={str(c) for c in roles.ids} if roles is not None else set(),
            declared_features={str(c) for c in roles.features} if roles is not None else set(),
            cv_group_columns={str(c) for c in prep.batch_cols},
            derived_outputs={k: str(v.get("reason")) for k, v in prep.derived_outputs.items()},
        )
    )

    if not prep.cat_dims:
        proposal_optimizer = "continuous"
    else:
        free_cards = [
            int(k)
            for j, k in zip(prep.cat_dims, prep.cat_cardinalities or [])
            if j not in prep.strain.held
        ]
        if not free_cards:
            proposal_optimizer = "continuous"
        else:
            n_combos = 1
            for k in free_cards:
                n_combos *= k
            proposal_optimizer = "mixed_exact" if n_combos <= MAX_MIXED_COMBOS else "mixed_alternating"

    result: dict[str, Any] = {
        "n": len(prep.y), "d": len(kept_features), "target": str(prep.target),
        "group_col": prep.gcol,
        "targets": [str(c) for c in prep.candidate_targets], "features": kept_features,
        "categorical_features": [str(c) for c in prep.kept_cats],
        "n_dropped_incomplete": prep.n_rows_dropped,
        "n_censored_excluded": prep.n_censored_rows,
        "data_drops": _data_drops_block(
            prep.by_column, prep.n_rows_dropped, prep.n_censored_rows, prep.zero_filled
        ),
        "auto_categorical": [str(c) for c in prep.auto_cats if c in prep.kept_cats],
        "header_row": prep.header_info.get("header_row"),
        "proposal_optimizer": proposal_optimizer,
        "n_pending_considered": prop.n_pending,
        "n_pending_dropped": int(prop.n_pending_dropped),
        "cv_spearman": None if trust.rho != trust.rho else round(trust.rho, 3),
        "cv_ci95": trust.ci95,
        "cv_n_groups": trust.rep["n_groups"],
        "cv_grouping": trust.cv_grouping,
        "cv_n_repeats": trust.rep["n_repeats"],
        "cv_spearman_per_repeat": [
            None if s != s else round(float(s), 3) for s in trust.rep["spearman_per_repeat"]
        ],
        "conformal_q": trust.conformal_q,
        "conformal_coverage": round(1.0 - CONFORMAL_ALPHA, 4),
        "interval_levels": [float(c) for c in INTERVAL_LEVELS],
        "interval_method": interval_method_summary(
            trust.interval_cal, prop.sigma_source, trust.selection_cal
        ),
        "conformal_caveat": conformal_band_caveat(CONFORMAL_ALPHA),
        "driver_selection": {
            "n_tested": drivers.n_tested,
            "n_reported": len(drivers.drivers),
            "top_k": DRIVER_TOP_K,
            "fdr_method": "benjamini_hochberg",
            "fdr_q": DRIVER_FDR_Q,
            "n_bootstrap": 200,
            "ranked_by": "abs_rho",
            "unit": "recipe",
            "n_units": drivers.n_driver_units,
            "n_rows": len(prep.y),
        },
        "reliability": trust.reliability,
        "promotion": trust.promotion,
        "verdict": trust.verdict,
        "refusal": trust.refusal,
        "proposal_gating": {
            "gated": bool(trust.gated),
            "reason": trust.gate_reason,
            "n_infeasible": trust.n_infeasible,
            "feasibility_auc": (
                None if trust.feas_auc != trust.feas_auc else round(float(trust.feas_auc), 3)
            ),
            "threshold": 0.0,
        },
        "constraint": trust.constraint_report,
        "cv_logo": trust.cv_logo,
        "cv_topk": trust.cv_topk,
        "cv_group_mean_baseline": {**trust.cv_group_mean_baseline, "margin": trust.gm_margin},
        "best": round(prop.best_single, 4),
        "noise": _noise_block(
            prop.nr, prop.best_single, prop.best_reproducible, prop.replicate_aware, prop.het
        ),
        "alternative_scale": post_fit.alt_scale,
        "drivers": drivers.drivers,
        "gp_shapes": explanation.gp_shapes,
        "proposal_features": prop.show,
        "feature_ranges": {
            dim.name: [round(float(dim.lower), 6), round(float(dim.upper), 6)]
            for dim in prep.design.dims
            if not dim.is_categorical
        },
        "proposal_notice": prop.proposal.notice,
        "batch_diversity": prop.proposal.diversity_report(),
        "proposals": _annotate(
            prop.batch, prop.p_mean, prop.p_std, prop.incumbent,
            cols=prop.show_idx, design=prep.design,
            p_feasible=prop.p_feasible_arr, constraint=prop.constraint_pred,
            minimize=prep.minimize,
            distance=nearest_training_distance(prop.batch, prep.X, prep.bounds, prep.cat_dims),
            interval_fields=proposal_interval_fields(
                prop.p_mean, prop.p_std, prop.p_std_obs, trust.interval_cal, INTERVAL_LEVELS,
                prop.incumbent,
                "best_reproducible_mean" if prop.replicate_aware else "best_measured_single",
                maximize=not prep.minimize, selection=trust.selection_cal),
        ),
        "training_nn_distance_median": training_nn_distance_median(
            prep.X, prep.bounds, prep.cat_dims
        ),
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(trust.oof_a, trust.oof_p)],
        "provenance": provenance,
        "orientation": prep.df.attrs.get("kalos_orientation"),
        "validation": report_dict(
            _with_data_findings(
                prep.validation,
                _data_findings(
                    prep.by_column, prep.n_rows_dropped, prep.n_censored_rows, prep.zero_filled,
                    auto_cats=[str(c) for c in prep.auto_cats if c in prep.kept_cats],
                    header=prep.header_info,
                ) + list(prep.strain_plan.findings) + derived_output_findings(prep.derived_outputs),
            )
        ),
        "design_box_exclusions": prep.box_exclusions,
        "embedding": explanation.embedding,
        "correlation": explanation.correlation,
        "response_surface": explanation.response_surface,
        "slices": explanation.slices_result["slices"],
        "slices_skipped": explanation.slices_result["slices_skipped"],
        "visualization_reasons": explanation.viz_reasons,
        "model_diagnostics": explanation.model_diagnostics,
        "seed": ANALYZE_SEED,
        "timestamp": int(time.time()),
        "engine_version": ENGINE_VERSION,
        "target_unit": target_unit_block(
            str(prep.target),
            user_target_unit if user_target_unit is not None
            else (setup or {}).get("user_target_unit"),
        ),
    }
    if prep.setup_plan is not None:
        result.update(prep.setup_plan.result_fields())
    if prep.strain.active:
        result["drivers"] = [
            {**d, "role": "covariate"} if d["name"] in prep.strain.held_covariates else d
            for d in result["drivers"]
        ]
        result["proposals"] = prep.strain.annotate_proposals(
            result["proposals"], prop.batch, prep.X, prep.bounds, prep.cat_dims,
            nearest_training_distance,
        )
    result["strain_summary"] = post_fit.strain_summary
    _pred = (
        explanation.strain_pred
        if isinstance(explanation.strain_pred, dict) and "error" not in explanation.strain_pred
        else None
    )
    result["slices_by_level"] = (_pred or {}).get("slices_by_level") or {}
    result["slices_by_level_skipped"] = (_pred or {}).get("slices_by_level_skipped") or {}
    result["oof_levels"] = post_fit.oof_levels
    result["covariates"] = covariates_block(prep.strain.plan, prep.strain.held_info)
    result["drivers_skipped"] = categorical_skipped([str(c) for c in prep.kept_cats])
    result["correlation_skipped"] = categorical_skipped([str(c) for c in prep.kept_cats])
    result["categorical_effects"] = post_fit.cat_effects
    if prep.strain.active:
        _roles = {prep.strain.plan.column: "strain"} if prep.strain.plan.column else {}
        _roles.update({c: "covariate" for c in covariates_block(prep.strain.plan, {})["columns"]})
        result["provenance"] = [
            {**row, "role": _roles[row["name"]]} if row.get("name") in _roles else row
            for row in result["provenance"]
        ]
    if trust.refusal is not None:
        result["proposals"] = []
        result["proposal_notice"] = refusal_notice(trust.refusal, 5)
    result["proposal_trust"] = proposal_trust(
        trust.verdict, n_returned=len(result["proposals"]), n_requested=5
    )
    result["proposals"] = label_proposals(result["proposals"], result["proposal_trust"])
    if anonymize:
        result = _anonymize_result(
            result,
            categorical_levels={str(c): prep.cat_dims_map[c] for c in prep.kept_cats},
            label_columns=[prep.strain.plan.column] if prep.strain.plan.column else [],
        )
    return result
