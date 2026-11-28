"""Stage 3 (propose): fit proposal surrogate (replicate-aware when applicable) and acquire batch.

Torch-free at import time: `Surrogate` and `propose_batch` are imported inside
`run_proposals`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kalos.analysis.stages.explain import DriversResult
from kalos.analysis.stages.prepare import PreparedAnalysis
from kalos.analysis.stages.trust import TrustEvaluation
from kalos.core.replicates import (
    aggregate_replicates,
    heteroscedasticity_report,
    noise_report,
)
from kalos.portal.analysis_parts.intervals import proposal_predictive_std
from kalos.portal.analysis_parts.policy import ANALYZE_SEED
from kalos.portal.analysis_parts.proposals import _encode_pending
from kalos.portal.analysis_parts.setup_apply import _propose_with_setup, withheld_batch
from kalos.portal.config import space_filling_small_n_setting


@dataclass
class ProposalResult:
    """Outputs of stage 3 (`run_proposals`), including the live proposal surrogate."""

    nr: dict[str, Any]
    het: dict[str, Any]
    best_single: float
    best_reproducible: float | None
    replicate_aware: bool
    incumbent: float
    X_fit: np.ndarray
    y_fit: np.ndarray
    surrogate: Any
    X_pending: np.ndarray | None
    n_pending: int
    n_pending_dropped: int
    proposal: Any
    batch: np.ndarray
    show: list[str]
    show_idx: list[int]
    p_mean: np.ndarray
    p_std: np.ndarray
    p_std_obs: np.ndarray
    sigma_source: str
    p_feasible_arr: np.ndarray
    constraint_pred: tuple[str, np.ndarray, np.ndarray] | None


def run_proposals(
    prep: PreparedAnalysis,
    trust: TrustEvaluation,
    drivers: DriversResult,
    *,
    pending: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    progress: Callable[[str, float], None],
    proposal_fit_restarts: int,
) -> ProposalResult:
    """Fit the proposal surrogate, acquire the next batch, and score batch posteriors."""
    from kalos.core.optimize import propose_batch
    from kalos.core.surrogate import Surrogate, proposal_restarts

    X = prep.X
    y = prep.y
    bounds = prep.bounds
    cat_dims = prep.cat_dims
    cat_cardinalities = prep.cat_cardinalities
    _minimize = prep.minimize

    nr = noise_report(X, y, groups=trust.recipe_key)
    het = heteroscedasticity_report(X, y, groups=trust.recipe_key)
    best_single = float(y.min() if _minimize else y.max())
    best_reproducible: float | None = None
    replicate_aware = False
    incumbent = best_single
    X_fit, y_fit = X, y
    s: Any = None
    if nr["n_replicated"] >= 1:
        Xf, yf, _yvar_g, n_reps_g = aggregate_replicates(X, y, groups=trust.recipe_key)
        best_reproducible = float(yf.min() if _minimize else yf.max())
        sigma2 = nr["noise_var"]
        if nr["n_recipes"] >= 6 and np.isfinite(sigma2) and sigma2 > 0:
            replicate_aware = True
            per_point_var = float(sigma2) / np.maximum(n_reps_g, 1).astype(float)
            s = Surrogate().fit(
                Xf, yf, bounds=bounds, noise=per_point_var, cat_dims=cat_dims,
                check_conditioning=True,
                restarts=proposal_restarts(len(yf), Xf.shape[1], proposal_fit_restarts),
                restart_seed=ANALYZE_SEED,
            )
            incumbent = best_reproducible
            X_fit, y_fit = Xf, yf

    progress("proposing", 0.8)
    if not replicate_aware:
        s = Surrogate().fit(
            X, y, bounds=bounds, cat_dims=cat_dims, check_conditioning=True,
            restarts=proposal_restarts(len(y), X.shape[1], proposal_fit_restarts),
            restart_seed=ANALYZE_SEED,
        )

    X_pending, n_pending_dropped = _encode_pending(
        prep.strain.pending(pending) if prep.strain.active else pending,
        prep.cont_feats, prep.kept_cats, prep.code_maps, X.shape[1],
        absent_zero=prep.absent_zero & set(map(str, prep.cont_feats)),
    )
    n_pending = 0 if X_pending is None else int(X_pending.shape[0])
    n_distinct_recipes = len(set(np.asarray(trust.recipe_key).tolist()))
    use_space_filling = (
        space_filling_small_n_setting()
        and trust.verdict.get("state") != "reliable"
        and n_distinct_recipes < 2 * X.shape[1]
    )
    if prep.setup_plan is not None and prep.setup_plan.refused:
        proposal = withheld_batch(prep.setup_plan, q=5, d=X.shape[1])
    elif prep.strain.refusal is not None:
        proposal = prep.strain.withheld_batch(q=5, d=X.shape[1])
    elif use_space_filling:
        from kalos.core.space_filling import propose_space_filling

        sf_bounds = prep.setup_plan.bounds if prep.setup_plan is not None else bounds
        sf_constraints = (
            prep.setup_plan.linear_constraints
            if prep.setup_plan is not None and prep.setup_plan.linear_constraints
            else None
        )
        try:
            proposal = propose_space_filling(
                sf_bounds,
                q=5,
                X_train=X,
                cat_dims=cat_dims,
                cat_cardinalities=cat_cardinalities,
                fixed_features=prep.strain.propose_kwargs().get("fixed_features"),
                inequality_constraints=sf_constraints,
                pending=X_pending,
                seed=ANALYZE_SEED,
            )
        except (RuntimeError, ValueError):
            if prep.setup_plan is not None and sf_constraints:
                prep.setup_plan._refuse(
                    "constraints_unsatisfiable",
                    "the optimizer could not find recipes that satisfy every linear constraint "
                    "inside the search range; proposals are withheld rather than dropping a constraint",
                    "loosen the linear constraints or widen the bounds",
                )
                prep.setup_plan.linear_constraints = []
                proposal = withheld_batch(prep.setup_plan, q=5, d=X.shape[1])
            else:
                raise
    else:
        proposal = _propose_with_setup(propose_batch, prep.setup_plan, dict(
            **prep.strain.propose_kwargs(),
            surrogate=s,
            bounds=bounds,
            q=5,
            cat_dims=cat_dims,
            cat_cardinalities=cat_cardinalities,
            pending=X_pending,
            seed=ANALYZE_SEED,
            feasibility_classifier=trust.feas_clf if trust.gated else None,
            constraint_surrogate=trust.constraint_surrogate if trust.constraint_applied else None,
            constraint_floor=trust.constraint_floor_val if trust.constraint_applied else None,
        ), d=X.shape[1])
    batch = proposal.X if prep.setup_plan is None else prep.setup_plan.post_check(proposal.X)
    _shown = [d for d in drivers.drivers if d["name"] not in prep.strain.held_covariates][:4]
    show = [d["name"] for d in _shown]
    show_idx = [d["_idx"] for d in _shown]
    for d in drivers.drivers:
        d.pop("_idx", None)
    p_mean, p_std = s.posterior(batch) if len(batch) else (np.empty(0), np.empty(0))
    if len(batch):
        p_std_obs, sigma_source = proposal_predictive_std(
            s, batch, p_std, float(nr["noise_var"]) if replicate_aware else None
        )
    else:
        p_std_obs = np.empty(0)
        sigma_source = (
            "latent_plus_assay_noise_floor" if replicate_aware else "surrogate_fitted_noise"
        )

    p_feasible_arr = (
        trust.feas_clf.predict_proba(batch)
        if trust.gated and len(batch)
        else np.full(len(batch), np.nan)
    )
    constraint_pred: tuple[str, np.ndarray, np.ndarray] | None = None
    if trust.constraint_applied and trust.constraint_surrogate is not None:
        c_mean, c_std = (
            trust.constraint_surrogate.posterior(batch)
            if len(batch)
            else (np.empty(0), np.empty(0))
        )
        assert trust.constraint_col is not None
        constraint_pred = (trust.constraint_col, c_mean, c_std)

    return ProposalResult(
        nr=nr,
        het=het,
        best_single=best_single,
        best_reproducible=best_reproducible,
        replicate_aware=replicate_aware,
        incumbent=incumbent,
        X_fit=X_fit,
        y_fit=y_fit,
        surrogate=s,
        X_pending=X_pending,
        n_pending=n_pending,
        n_pending_dropped=n_pending_dropped,
        proposal=proposal,
        batch=batch,
        show=show,
        show_idx=show_idx,
        p_mean=p_mean,
        p_std=p_std,
        p_std_obs=p_std_obs,
        sigma_source=sigma_source,
        p_feasible_arr=p_feasible_arr,
        constraint_pred=constraint_pred,
    )
