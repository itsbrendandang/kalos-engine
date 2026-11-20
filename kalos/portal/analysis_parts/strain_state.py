"""Per-analysis clone / covariate state (docs/CLONES.md).

One object built after the design space exists, consulted at each step of
`_analyze_body` that the clone or covariate roles touch: the proposal box
(`fixed_features`), the setup plan (held dims, linear constraints with a clone),
in-flight recipes, the per-proposal clone annotations, the predicted per-clone
blocks and the summary. With no clone and no covariate column, `active` is
False and the analysis takes exactly its historical path.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kalos.portal.analysis_parts import strain_block as sb
from kalos.portal.analysis_parts.strain_roles import StrainPlan, normalize_label


@dataclass
class StrainState:
    plan: StrainPlan
    labels_fit: np.ndarray | None = None
    levels: tuple[str, ...] = ()
    n_runs: dict[str, int] = field(default_factory=dict)
    strain_dim: int | None = None
    fixed_level: str | None = None
    fixed_level_source: str | None = None
    refusal: dict | None = None
    held: dict[int, float] = field(default_factory=dict)
    held_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    cont_names: tuple[str, ...] = ()

    # --- construction ---------------------------------------------------------- #
    @classmethod
    def build(
        cls, plan: StrainPlan, *, df: pd.DataFrame, keep_index: pd.Index, X: np.ndarray,
        bounds: np.ndarray, cont_feats: Sequence[Any], kept_cats: Sequence[Any],
        cat_dims_map: Mapping[Any, Sequence[str]],
    ) -> StrainState:
        state = cls(plan=plan, cont_names=tuple(str(c) for c in cont_feats))
        if not state.active:
            return state
        col = plan.column
        if col is not None:
            state.labels_fit = np.asarray(df.loc[keep_index, col].tolist(), dtype=object)
            names = [str(c) for c in kept_cats]
            if col in names:
                k = names.index(col)
                state.strain_dim = len(cont_feats) + k
                state.levels = tuple(cat_dims_map[kept_cats[k]])
            else:
                state.levels = tuple(sorted({v for v in state.labels_fit.tolist() if v is not None}))
            state.n_runs = {lv: int((state.labels_fit == lv).sum()) for lv in state.levels}
            if state.strain_dim is not None:
                lvl, src, refusal = sb.choose_fixed_level(plan, state.levels, state.n_runs)
                state.fixed_level, state.fixed_level_source, state.refusal = lvl, src, refusal
                if lvl is not None:
                    state.held[state.strain_dim] = float(state.levels.index(lvl))
        state.held_info = sb.held_covariate_values(
            plan, X, state.cont_names, bounds, state.labels_fit, state.fixed_level,
        )
        for info in state.held_info.values():
            state.held[int(info["dim"])] = float(info["value"])
        return state

    # --- properties ------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        p = self.plan
        return p.column is not None or bool(
            p.covariates_numeric or p.covariates_label or p.covariates_unfitted
        )

    @property
    def has_clone(self) -> bool:
        return self.plan.column is not None

    @property
    def held_covariates(self) -> set[str]:
        return set(self.held_info)

    @property
    def held_names(self) -> list[str]:
        names = list(self.held_info)
        if self.fixed_level is not None and self.plan.column is not None:
            names.append(self.plan.column)
        return names

    @property
    def process_names(self) -> list[str]:
        covs = set(self.plan.covariates_numeric)
        return [c for c in self.cont_names if c not in covs]

    def mixed_constraints_ok(
        self, cat_dims: Sequence[int], cards: Sequence[int], max_combos: int,
    ) -> bool:
        """Linear constraints with categoricals are applied in a clone campaign
        when the free (not held) categorical combinations can be enumerated."""
        if not self.has_clone:
            return False
        free = [int(k) for j, k in zip(cat_dims, cards) if j not in self.held]
        return int(np.prod(free)) <= max_combos if free else True

    def propose_kwargs(self) -> dict[str, Any]:
        return {"fixed_features": dict(self.held)} if self.held else {}

    def withheld_batch(self, *, q: int, d: int) -> Any:
        from kalos.core.optimize import NEAR_DUPLICATE_TOL, ProposalBatch

        reason = (self.refusal or {}).get("reason", "")
        return ProposalBatch(np.empty((0, d)), q, 0, float(NEAR_DUPLICATE_TOL), {
            "code": (self.refusal or {}).get("code", "clone_not_in_training"),
            "requested": int(q), "returned": 0, "min_separation": None,
            "message": f"No proposals: {reason}. See strain_summary.refusals.",
        })

    def pending(self, pending: Any) -> Any:
        """In-flight recipes with clone labels normalized and held covariates
        filled with the value proposals hold them at (a proposed recipe never
        carries a covariate, and dropping it would forget the run is in flight)."""
        if pending is None:
            return None
        frame = pending if isinstance(pending, pd.DataFrame) else pd.DataFrame(list(pending))
        if frame.empty:
            return pending
        frame = frame.copy()
        col = self.plan.column
        if col is not None and col in frame.columns:
            frame[col] = [normalize_label(v) for v in frame[col].tolist()]
        for name, info in self.held_info.items():
            if name not in frame.columns:
                frame[name] = info["value"]
            else:
                frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(info["value"])
        return frame

    # --- predicted blocks --------------------------------------------------------- #
    def predicted(self, s: Any, **kw: Any) -> dict[str, Any] | None:
        if self.strain_dim is None or self.plan.column is None:
            return None
        try:
            return sb.predicted_blocks(
                s, strain_dim=self.strain_dim, levels=self.levels, n_runs=self.n_runs,
                process_names=self.process_names, column=self.plan.column, **kw,
            )
        except Exception:  # a diagnostic must not take down the analysis
            sb.logger.exception("per-clone predicted blocks failed")
            return {"error": "per-clone predicted blocks failed"}

    # --- proposals ------------------------------------------------------------------ #
    def annotate_proposals(
        self, proposals: list[dict], batch: np.ndarray, X: np.ndarray, bounds: np.ndarray,
        cat_dims: Sequence[int] | None, distance: Callable[..., Any],
    ) -> list[dict]:
        """Covariates out of the recipe (into `held`); per-clone extrapolation."""
        col = self.plan.column
        out: list[dict] = []
        for i, p in enumerate(proposals):
            p = dict(p)
            recipe = dict(p.get("recipe") or {})
            if self.held_info:
                p["held"] = {
                    name: {"value": info["value"], "basis": info["basis"], "proposed": False}
                    for name, info in self.held_info.items()
                }
                for name in self.held_info:
                    recipe.pop(name, None)
                p["recipe"] = recipe
            if col is not None and self.labels_fit is not None and col in recipe:
                lv = recipe[col]
                mask = self.labels_fit == lv
                p["clone"] = {"column": col, "level": lv, "policy": self.plan.policy,
                              "n_runs": int(mask.sum())}
                if mask.any() and i < len(batch):
                    Xl = X[mask]
                    d_same = distance(batch[i : i + 1], Xl, bounds, cat_dims)
                    d0 = float(d_same[0]) if len(d_same) else float("nan")
                    p["nearest_training_distance_same_clone"] = (
                        round(d0, 4) if np.isfinite(d0) else None
                    )
                    outside = []
                    for j, name in enumerate(self.cont_names):
                        if name in self.held_info:
                            continue
                        lo, hi = float(Xl[:, j].min()), float(Xl[:, j].max())
                        v = float(batch[i, j])
                        if v < lo - 1e-9 or v > hi + 1e-9:
                            outside.append({"feature": name, "value": round(v, 6),
                                            "clone_range": [round(lo, 6), round(hi, 6)]})
                    p["extrapolated_for_clone"] = bool(outside)
                    p["extrapolated_for_clone_features"] = outside
                n = int(mask.sum())
                if self.plan.policy == "factor" and n < 3:
                    p["mode"] = "explore"
                    p["clone_refusal"] = {
                        "code": "clone_too_few_runs",
                        "reason": (f"this proposal uses a clone with n={n} < 3 runs; treat it as "
                                   "exploration of that clone, not as a predicted improvement"),
                    }
            out.append(p)
        return out




def covariates_block(plan: StrainPlan, held_info: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Top-level `covariates`: which columns are covariates, which are model
    inputs (held in proposals), which are not and why. Never proposed."""
    cols = [*plan.covariates_numeric, *plan.covariates_unfitted, *plan.covariates_label]
    return {
        "columns": cols,
        "source": plan.covariate_source,
        "fitted": list(plan.covariates_numeric),
        "held": {k: {"value": v["value"], "basis": v["basis"]} for k, v in held_info.items()},
        "not_fitted": {
            **dict(plan.covariates_unfitted),
            **{c: ("label covariate (thaw / vial / bank): used for the confounding checks, not as "
                   "a model input; the next run uses a new label the model has never seen")
               for c in plan.covariates_label},
        },
        "note": "covariates are recorded run properties: never proposed, never in a recipe",
    }


def strain_outputs(
    state: StrainState,
    *,
    df: pd.DataFrame,
    target: str,
    outcomes: Sequence[str],
    outcome_units: Mapping[str, str | None],
    outcome_direction: Mapping[str, str],
    keep_index: pd.Index,
    y: np.ndarray,
    X: np.ndarray,
    kept_cats: Sequence[Any],
    features: Sequence[str],
    maximize: bool,
    seed: int,
    anonymize: bool,
    predicted: Mapping[str, Any] | None,
    groups: Sequence[Any],
    bounds: np.ndarray,
    cat_dims: Sequence[int] | None,
    oof_index: Sequence[int],
    oof_a: Sequence[float],
    oof_p: Sequence[float],
    oof_s: Sequence[float],
    interval_cal: Mapping[str, Any],
    grouped_cv_report: Callable[..., dict],
    logo_report: Callable[..., dict],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(strain_summary, oof_levels)`. Runs the two GP-heavy parts (the
    process-only CV for the adjusted effects, and leave-one-clone-out); the caller
    runs this after every proposal-path fit."""
    plan = state.plan
    if not state.has_clone or state.labels_fit is None:
        return sb.unavailable(plan), {}
    col = plan.column
    assert col is not None
    pred = None if predicted is None or "error" in predicted else predicted
    summary = sb.build_summary(
        plan=plan, df=df, target=target, outcomes=outcomes, outcome_units=outcome_units,
        outcome_direction=outcome_direction, keep_index=keep_index, y=y, X=X,
        cont_names=state.cont_names, process_names=state.process_names, kept_cats=kept_cats,
        features=features, fixed_level=state.fixed_level,
        fixed_level_source=state.fixed_level_source, held=state.held_info, predicted=pred,
        maximize=maximize, seed=seed, anonymize=anonymize,
        extra_refusals=[state.refusal] if state.refusal else [],
    )
    labels_fit = state.labels_fit
    oof_levels = {col: [labels_fit[int(i)] for i in oof_index]}

    # Process-adjusted clone effect: out-of-fold residuals of a process-only GP.
    oof_resid: np.ndarray | None = None
    resid_index: list[int] | None = None
    aux_reason: str | None = None
    proc = [state.cont_names.index(c) for c in state.process_names]
    try:
        if proc:
            rep = grouped_cv_report(
                X[:, proc], y, groups=list(groups), n_splits=5, bounds=bounds[:, proc],
                n_boot=0, n_repeats=1,
            )
            resid_index = [int(i) for i in rep["oof_index"]]
            oof_resid = np.asarray(rep["oof_actual"], float) - np.asarray(rep["oof_pred"], float)
        else:
            resid_index = list(range(len(y)))
            oof_resid = np.asarray(y, float) - float(np.mean(y))
            aux_reason = "no continuous process inputs: residuals are deviations from the grand mean"
    except Exception as exc:  # reported, never fatal
        sb.logger.warning("process-only CV for clone effects failed: %s", type(exc).__name__)
        aux_reason = f"the process-only model could not be cross-validated ({type(exc).__name__})"
        oof_resid, resid_index = None, None

    # Leave-one-clone-out: report only, never a gate.
    why_not = sb.leave_one_clone_out_plan(state.n_runs)
    if why_not is None and state.strain_dim is not None:
        try:
            lo = logo_report(X, y, [str(v) for v in labels_fit], bounds=bounds, cat_dims=cat_dims)
            rho = lo.get("spearman")
            loco: dict[str, Any] = {
                "spearman": None if rho is None or rho != rho else round(float(rho), 3),
                "n_clones": lo.get("n_groups"), "n_oof": lo.get("n_oof"), "reason": None,
                "note": ("how well the process model ranks runs of a clone it has never seen; "
                         "reported only, never used as a gate"),
            }
        except Exception as exc:
            loco = {"spearman": None, "reason": f"failed ({type(exc).__name__})"}
    else:
        loco = {"spearman": None, "reason": why_not or "the clone is not a model input"}

    q90 = next(
        (lv.get("q_norm") for lv in interval_cal.get("levels", [])
         if lv.get("available") and abs(float(lv.get("coverage", 0)) - 0.9) < 1e-9),
        None,
    )
    cov = sb.oof_coverage_by_level(
        oof_levels[col], oof_a, oof_p, oof_s, None if q90 is None else float(q90), 0.9,
    )
    summary = sb.finish_summary(
        summary, labels_fit=labels_fit, oof_index=resid_index, oof_resid=oof_resid,
        aux_reason=aux_reason, loco=loco, oof_coverage=cov,
    )
    summary["oof_coverage_method"] = (
        "share of this clone's held-out runs inside the 90% normalized conformal band "
        "(calibrated over all clones)"
    )
    return summary, oof_levels


__all__ = ["StrainState", "covariates_block", "strain_outputs"]
