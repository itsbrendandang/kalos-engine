"""Stage 4 (explain): drivers, GP shapes, viz/slices, and post-fit diagnostics.

Split into three functions to preserve the exact RNG consumption order of
`_analyze_body`:
  1. `compute_drivers`: runs before `propose` (uses numpy RNG for bootstrap).
  2. `explain_surrogate`: runs while the fitted proposal surrogate is alive, then
     releases the surrogate tensors and runs `gc.collect()`.
  3. `compute_post_fit_diagnostics`: runs after `propose` + surrogate release so
     alternative-scale and clone CV fits cannot shift proposal RNG draws.
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from kalos.analysis.stages.prepare import PreparedAnalysis
from kalos.analysis.stages.trust import TrustEvaluation
from kalos.core.drivers import (
    benjamini_hochberg,
    bootstrap_spearman,
    spearman_driver_matrix,
)
from kalos.core.gp_shape import gp_shape_report
from kalos.core.replicates import aggregate_replicates
from kalos.portal.analysis_parts.categorical_effects import categorical_effects
from kalos.portal.analysis_parts.intervals import (
    proposal_interval_fields,
    proposal_predictive_std,
)
from kalos.portal.analysis_parts.noise_blocks import _alternative_scale_block
from kalos.portal.analysis_parts.policy import (
    ANALYZE_SEED,
    DRIVER_FDR_Q,
    DRIVER_TOP_K,
    INTERVAL_LEVELS,
    RELIABILITY_SPEARMAN_FLOOR,
)
from kalos.portal.analysis_parts.strain_state import strain_outputs
from kalos.portal.analysis_parts.units import target_unit_block
from kalos.portal.cells import cell_labels

if TYPE_CHECKING:
    from kalos.analysis.stages.propose import ProposalResult


@dataclass
class DriversResult:
    drivers: list[dict[str, Any]]
    n_tested: int
    n_driver_units: int


@dataclass
class SurrogateExplanation:
    gp_shapes: dict[str, Any]
    embedding: dict[str, Any] | None
    correlation: dict[str, Any] | None
    response_surface: dict[str, Any] | None
    slices_result: dict[str, Any]
    viz_reasons: dict[str, str | None]
    strain_pred: Any
    model_diagnostics: dict[str, Any]


@dataclass
class PostFitDiagnostics:
    alt_scale: dict[str, Any]
    strain_summary: dict[str, Any]
    oof_levels: dict[str, Any]
    cat_effects: dict[str, Any] | None


def compute_drivers(prep: PreparedAnalysis, trust: TrustEvaluation) -> DriversResult:
    """Recipe-averaged Spearman driver panel with BH-FDR and bootstrap 95% CIs."""
    drv: list[dict[str, Any]] = []
    n_tested = len(prep.cont_feats)
    n_driver_units = len(prep.y)
    if prep.cont_feats:
        names = [str(c) for c in prep.cont_feats]
        Xr, yr, _rv, _rn = aggregate_replicates(prep.X, prep.y, groups=trust.recipe_key)
        n_driver_units = int(Xr.shape[0])
        Xcont = Xr[:, : len(prep.cont_feats)]
        matrix = spearman_driver_matrix(Xcont, yr, feature_names=names)
        point = np.asarray(matrix["rho"]).astype(float)
        pvals = np.asarray(matrix["pvals"]).astype(float)
        bh_survives = benjamini_hochberg(pvals, q=DRIVER_FDR_Q)
        boot = bootstrap_spearman(Xcont, yr, feature_names=names)
        boot_lo = np.asarray(boot["lo"], dtype=float)
        boot_hi = np.asarray(boot["hi"], dtype=float)
        for j, c in enumerate(prep.cont_feats):
            lo, hi = round(float(boot_lo[j]), 3), round(float(boot_hi[j]), 3)
            ci_excludes_zero = bool(lo > 0 or hi < 0)
            drv.append(
                {
                    "_idx": j,
                    "name": str(c),
                    "rho": round(float(point[j]), 3),
                    "ci95": [lo, hi],
                    "p": round(float(pvals[j]), 6),
                    "ci_excludes_zero": ci_excludes_zero,
                    "survives_fdr": bool(bh_survives[j]),
                    "significant": bool(ci_excludes_zero and bh_survives[j]),
                }
            )
    drv.sort(key=lambda d: -abs(float(d["rho"])))
    drv = drv[:DRIVER_TOP_K]
    return DriversResult(drivers=drv, n_tested=n_tested, n_driver_units=n_driver_units)


def explain_surrogate(
    prep: PreparedAnalysis,
    trust: TrustEvaluation,
    prop: ProposalResult,
    drivers: DriversResult,
    *,
    gp_shape_fn: Callable[..., Any] = gp_shape_report,
) -> SurrogateExplanation:
    """Compute GP shapes, PCA/Spearman/surface/slice viz, and per-clone predictions, then free GP."""
    from kalos.portal.analysis_parts.slices import compute_slices
    from kalos.portal.viz_utils import (
        _correlation_spearman,
        _embedding_pca,
        _response_surface_gp,
    )

    s = prop.surrogate
    gp_shapes = gp_shape_fn(
        s,
        prop.X_fit if prep.cont_feats else np.empty((len(prop.y_fit), 0)),
        prop.y_fit,
        feature_names=[str(c) for c in prep.cont_feats],
        cv_spearman=None if trust.rho != trust.rho else float(trust.rho),
        rho_floor=RELIABILITY_SPEARMAN_FLOOR,
        sweep_columns=list(range(len(prep.cont_feats))) if prep.cont_feats else None,
    ).to_dict()

    viz_reasons: dict[str, str | None] = {
        "embedding": None, "correlation": None, "response_surface": None,
    }

    _emb = _embedding_pca(
        prep.X, prep.y, prep.cont_feats, prep.kept_cats,
        prep.cat_dims_map if prep.kept_cats else None,
    )
    embedding: dict[str, Any] | None = None
    if _emb is not None:
        if "reason" in _emb:
            viz_reasons["embedding"] = _emb["reason"]
        else:
            embedding = _emb

    _corr = _correlation_spearman(prep.X, prep.y, prep.cont_feats, groups=trust.recipe_key)
    correlation: dict[str, Any] | None = None
    if _corr is not None:
        if "reason" in _corr:
            viz_reasons["correlation"] = _corr["reason"]
        else:
            correlation = _corr

    _surf = _response_surface_gp(
        s, prop.X_fit, prop.y_fit, prep.cont_feats, prep.kept_cats,
        prep.cat_dims_map or {}, prep.code_maps, prep.bounds,
        drivers.drivers, prep.cat_dims, prop.incumbent, prop.replicate_aware,
    )
    response_surface: dict[str, Any] | None = None
    if _surf is not None:
        if "reason" in _surf:
            viz_reasons["response_surface"] = _surf["reason"]
        else:
            response_surface = _surf

    _sl = compute_slices(
        s, prop.X_fit, prop.y_fit, prep.cont_feats, prep.kept_cats,
        prep.cat_dims_map or {}, prep.bounds, drivers.drivers, prop.replicate_aware,
        incumbent=prop.incumbent,
    )
    viz_reasons["slices"] = _sl["reason"]

    strain_pred = prep.strain.predicted(
        s, X_fit=prop.X_fit, y_fit=prop.y_fit, cont_feats=prep.cont_feats,
        kept_cats=prep.kept_cats, cat_dims_map=prep.cat_dims_map or {},
        bounds=prep.bounds, drv=drivers.drivers,
        replicate_aware=prop.replicate_aware, incumbent=prop.incumbent,
        interval_fields=lambda pts, m, sd, _s=s: proposal_interval_fields(
            m, sd,
            proposal_predictive_std(
                _s, pts, sd, float(prop.nr["noise_var"]) if prop.replicate_aware else None)[0],
            trust.interval_cal, INTERVAL_LEVELS, prop.incumbent,
            "best_reproducible_mean" if prop.replicate_aware else "best_measured_single",
            maximize=not prep.minimize),
    )

    model_diagnostics = {**s.fit_diagnostics, "cv_fit_restarts": 1}
    prop.surrogate = None
    trust.constraint_surrogate = None
    del s
    gc.collect()

    return SurrogateExplanation(
        gp_shapes=gp_shapes,
        embedding=embedding,
        correlation=correlation,
        response_surface=response_surface,
        slices_result=_sl,
        viz_reasons=viz_reasons,
        strain_pred=strain_pred,
        model_diagnostics=model_diagnostics,
    )


def compute_post_fit_diagnostics(
    prep: PreparedAnalysis,
    trust: TrustEvaluation,
    prop: ProposalResult,
    explanation: SurrogateExplanation,
    *,
    anonymize: bool,
    user_target_unit: str | None,
    setup: Mapping[str, Any] | None,
) -> PostFitDiagnostics:
    """Alternative-scale CV, per-clone summary/OOF, and categorical effects (after proposals)."""
    from kalos.core.evaluation import (
        grouped_cv_report,
        interval_calibration,
        logo_report,
    )

    alt_scale = _alternative_scale_block(
        prop.het, prep.X, prep.y, groups=trust.groups, bounds=prep.bounds, cat_dims=prep.cat_dims,
        grouped_cv_report=grouped_cv_report, interval_calibration=interval_calibration,
    )

    if prep.strain.has_clone:
        _units = {
            str(c): target_unit_block(
                str(c), (user_target_unit if user_target_unit is not None
                         else (setup or {}).get("user_target_unit"))
                if str(c) == str(prep.target) else None,
            ).get("unit")
            for c in prep.candidate_targets
        }
        _direction = {str(prep.target): "lower_is_better" if prep.minimize else "higher_is_better"}
        if trust.constraint_applied and trust.constraint_col is not None:
            _direction[trust.constraint_col] = "floor"
        strain_summary, oof_levels = strain_outputs(
            prep.strain, df=prep.df, target=str(prep.target),
            outcomes=[
                str(c) for c in prep.candidate_targets
                if str(c) in {str(x) for x in prep.df.columns}
            ],
            outcome_units=_units, outcome_direction=_direction, keep_index=prep.keep_index,
            y=prep.y, X=prep.X, kept_cats=prep.kept_cats,
            features=[str(c) for c in prep.cont_feats] + [str(c) for c in prep.kept_cats],
            maximize=not prep.minimize, seed=ANALYZE_SEED, anonymize=anonymize,
            predicted=explanation.strain_pred, groups=trust.groups,
            bounds=prep.bounds, cat_dims=prep.cat_dims,
            oof_index=trust.rep["oof_index"], oof_a=trust.oof_a,
            oof_p=trust.oof_p, oof_s=trust.oof_s,
            interval_cal=trust.interval_cal, grouped_cv_report=grouped_cv_report,
            logo_report=logo_report,
        )
    else:
        strain_summary, oof_levels = strain_outputs(
            prep.strain, df=prep.df, target=str(prep.target), outcomes=[], outcome_units={},
            outcome_direction={}, keep_index=prep.keep_index, y=prep.y, X=prep.X,
            kept_cats=prep.kept_cats, features=[], maximize=not prep.minimize,
            seed=ANALYZE_SEED, anonymize=anonymize, predicted=None,
            groups=trust.groups, bounds=prep.bounds, cat_dims=prep.cat_dims,
            oof_index=[], oof_a=[], oof_p=[], oof_s=[], interval_cal={},
            grouped_cv_report=grouped_cv_report, logo_report=logo_report,
        )

    cat_effects = categorical_effects(
        prep.y,
        {str(c): cell_labels(prep.df.loc[prep.keep_index, c]).tolist() for c in prep.kept_cats},
        list(trust.recipe_key), fdr_q=DRIVER_FDR_Q,
    ) if prep.kept_cats else None

    return PostFitDiagnostics(
        alt_scale=alt_scale,
        strain_summary=strain_summary,
        oof_levels=oof_levels,
        cat_effects=cat_effects,
    )
