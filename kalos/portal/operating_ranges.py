"""Operating ranges: a model-based region where the campaign's target is likely
to meet a threshold, computed ONLY when it is defensible.

Used by `POST|GET /api/campaigns/{id}/operating-ranges` (kalos/portal/exports_api.py).

What is computed, stated so nobody over-reads it:

1. The campaign's published analysis must be current (`campaign_generation`
   equals the campaign's live generation) and its reliability must clear the
   engine's own floor (`reliability.clears_floor` AND
   `reliability.ci_excludes_zero`). Anything else is `not_computed` with a
   reason; nothing is estimated from a model the engine itself does not trust.
2. The GP is refit on the campaign's folded rows (`base_rows`), with the
   analysis's own feature list, categorical features and design box
   (`feature_ranges`). The rows are selected with the analysis's cell rules
   (`kalos.portal.cells`); if the refit does not see exactly the analysis's `n`
   rows and `best`, the refit is refused (`refit_data_mismatch`) rather than
   describing a different dataset.
3. A scrambled Sobol sample of the continuous design box (categorical inputs
   held at the best measured recipe's levels) is scored with the LATENT GP
   posterior: P(f(x) >= threshold) for a maximize goal, P(f(x) <= threshold)
   for minimize. A point "meets the rule" when that probability is at least
   `probability`.
4. `method = "gp_posterior_box"`: starting from the best measured recipe, the
   largest normalized cube (equal relative half-width per feature) that holds
   no failing sample is grown face by face (features in analysis order, lower
   face then upper) until each face touches a failing sample or the design
   bound. The result is MAXIMAL (no face can move out without taking in a
   failing sample), not necessarily the maximum-volume box. It is then
   tightened to the extremes of the passing samples inside it, so every
   reported endpoint is a sampled coordinate. Every sample inside meets the
   rule. It is a model-based statement that must be verified experimentally.

Torch-free at import time: torch/botorch load inside `fit_and_score`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# CPU bound: the posterior is evaluated on exactly this many quasi-random points
# (a power of two, so the scrambled Sobol sequence stays balanced).
N_SAMPLES = 4096
SOBOL_SEED = 20240611
# A box supported by fewer sampled points than this is not reported: the claim
# "every sampled point inside meets the rule" would rest on almost nothing.
MIN_BOX_SAMPLES = 32
PROBABILITY_MIN = 0.5
PROBABILITY_MAX = 0.99
METHOD = "gp_posterior_box"


@dataclass
class NotComputed(Exception):
    """A refusal: the route answers 200 `{status: "not_computed", ...}`."""

    reason_code: str
    reason: str
    what_would_fix_it: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "not_computed",
            "reason_code": self.reason_code,
            "reason": self.reason,
            "what_would_fix_it": self.what_would_fix_it,
        }
        if self.diagnostics:
            out["diagnostics"] = self.diagnostics
        return out


@dataclass
class FitPlan:
    """Everything the slot-bound fit needs, prepared without torch."""

    X: np.ndarray
    y: np.ndarray
    lower: np.ndarray            # continuous dims only, analysis order
    upper: np.ndarray
    cont_names: list[str]        # names as the analysis shows them
    cat_names: list[str]
    cat_levels: list[list[str]]  # per categorical, sorted labels (code = index)
    cat_display: list[list[str]]  # the labels as the analysis shows them
    direction: str               # "at_least" | "at_most"
    goal: str
    goal_source: str
    threshold: float
    probability: float
    target: str
    best_index: int
    context: dict[str, Any]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def resolve_goal(setup: dict[str, Any] | None) -> tuple[str, str]:
    """`(goal, source)` from the campaign's `setup` settings document. Default
    maximize (the engine's own objective) when no goal was set."""
    value = (setup or {}).get("value") if isinstance(setup, dict) else None
    goal = value.get("goal") if isinstance(value, dict) else None
    if goal is None:
        return "maximize", "default"
    if goal in ("maximize", "minimize"):
        return str(goal), "setup"
    raise NotComputed(
        "goal_unrecognized",
        "the campaign setup names a goal this engine does not recognize",
        "set the campaign setup goal to 'maximize' or 'minimize'",
    )


def check_preconditions(
    state: dict[str, Any] | None, analysis: dict[str, Any] | None,
    threshold: float | None,
) -> dict[str, Any]:
    """The cheap refusals, in order. Returns the analysis's reliability block."""
    if threshold is None:
        raise NotComputed(
            "threshold_missing",
            "no threshold was given; an operating range is the region meeting a threshold on the target",
            "pass 'threshold' (target units) and optionally 'probability' (0.5 to 0.99)",
        )
    if state is None:
        raise NotComputed(
            "no_data", "the campaign has no data yet", "upload a run sheet to this campaign",
        )
    if not analysis:
        raise NotComputed(
            "no_analysis", "the campaign has no published analysis",
            "re-upload the run sheet or re-analyze the campaign",
        )
    if analysis.get("campaign_generation") is None or (
        analysis.get("campaign_generation") != state.get("generation")
    ):
        raise NotComputed(
            "analysis_stale",
            "the published analysis does not describe the campaign's current data "
            "(results were recorded or the campaign changed since it was computed)",
            "re-analyze the campaign (POST /api/campaigns/{id}/reanalyze)",
        )
    rel = analysis.get("reliability")
    if not isinstance(rel, dict) or "clears_floor" not in rel or "ci_excludes_zero" not in rel:
        raise NotComputed(
            "reliability_unknown",
            "the analysis carries no reliability verdict (it predates it)",
            "re-analyze the campaign so its reliability is measured",
        )
    if rel.get("clears_floor") is not True or rel.get("ci_excludes_zero") is not True:
        failed = [
            name for name in ("clears_floor", "ci_excludes_zero") if rel.get(name) is not True
        ]
        raise NotComputed(
            "model_not_reliable",
            "the model has not shown it can rank held-out runs (grouped cross-validation): "
            + ", ".join(f"{n}=false" for n in failed),
            "add more measured runs (ideally spread over the design space) and re-analyze "
            "until reliability clears the floor",
            diagnostics={
                "spearman": rel.get("spearman"), "ci95": rel.get("ci95"),
                "spearman_floor": rel.get("spearman_floor"),
                "clears_floor": rel.get("clears_floor"),
                "ci_excludes_zero": rel.get("ci_excludes_zero"),
            },
        )
    return rel


def build_plan(
    state: dict[str, Any], analysis: dict[str, Any], *, threshold: float, probability: float,
    goal: str, goal_source: str, column_for: dict[str, str], display_level: Any,
) -> FitPlan:
    """Rebuild the analysis's design matrix from the campaign's rows.

    `column_for` maps each name the analysis shows to the raw `base_rows` key
    (identity unless anonymized). `display_level(raw_column, label)` returns the
    label as the analysis would show it."""
    import pandas as pd

    from kalos.portal.cells import BLANK, NUMERIC, classify_cells, is_null_label

    features = analysis.get("features")
    cats = analysis.get("categorical_features") or []
    ranges = analysis.get("feature_ranges")
    target = analysis.get("target")
    n_expected = analysis.get("n")
    if (
        not isinstance(features, list) or not isinstance(cats, list)
        or not isinstance(ranges, dict) or not isinstance(target, str)
        or not isinstance(n_expected, int)
    ):
        raise NotComputed(
            "analysis_incomplete",
            "the analysis lacks the feature list, design box or row count needed to refit",
            "re-analyze the campaign",
        )
    cont = [str(f) for f in features if f not in cats]
    if not cont:
        raise NotComputed(
            "categorical_only_design",
            "every model input is categorical; a per-feature numeric range is not defined",
            "an operating range needs at least one continuous process input",
        )
    missing = [f for f in [*cont, *cats, target] if f not in column_for]
    if missing or any(f not in ranges for f in cont):
        raise NotComputed(
            "refit_data_mismatch",
            "the analysis names columns that cannot be matched to the campaign's rows",
            "re-analyze the campaign",
        )
    df = pd.DataFrame(state.get("base_rows") or [])
    if df.empty:
        raise NotComputed("no_data", "the campaign has no rows", "upload a run sheet")
    for col in [*cont, *cats, target]:
        if column_for[col] not in df.columns:
            df[column_for[col]] = None
    drops = analysis.get("data_drops") if isinstance(analysis.get("data_drops"), dict) else {}
    absent_zero = set((drops or {}).get("blank_cells_as_absent") or {})

    tkind = classify_cells(df[column_for[target]]).to_numpy(object)
    y_all = pd.to_numeric(df[column_for[target]], errors="coerce").to_numpy(float)
    keep = tkind == NUMERIC
    ok = keep.copy()
    cont_cols: list[np.ndarray] = []
    for name in cont:
        raw = df[column_for[name]]
        kinds = classify_cells(raw).to_numpy(object)
        vals = pd.to_numeric(raw, errors="coerce").to_numpy(float)
        if name in absent_zero:
            vals = np.where(kinds == BLANK, 0.0, vals)
            ok &= (kinds == NUMERIC) | (kinds == BLANK)
        else:
            ok &= kinds == NUMERIC
        cont_cols.append(vals)
    cat_levels: list[list[str]] = []
    cat_codes: list[np.ndarray] = []
    for name in cats:
        labels = df[column_for[name]].map(lambda v: "" if v is None else str(v).strip())
        levels = sorted({v for v in labels[keep].tolist() if not is_null_label(v)})
        code = {lvl: i for i, lvl in enumerate(levels)}
        codes = labels.map(lambda v, code=code: code.get(v, -1)).to_numpy(float)
        ok &= codes >= 0
        cat_levels.append(levels)
        cat_codes.append(codes)
    idx = np.flatnonzero(ok)
    y = y_all[idx]
    expected_best = _finite(analysis.get("best"))
    if len(idx) != n_expected or (
        expected_best is not None and len(y) and round(float(y.max()), 4) != round(expected_best, 4)
    ):
        raise NotComputed(
            "refit_data_mismatch",
            "the rows rebuilt from the campaign do not match the rows the analysis was fit on",
            "re-analyze the campaign, then retry",
            diagnostics={"rows_rebuilt": int(len(idx)), "rows_in_analysis": n_expected},
        )
    X = np.column_stack([*(c[idx] for c in cont_cols), *(c[idx] for c in cat_codes)])
    lower = np.array([float(ranges[n][0]) for n in cont], float)
    upper = np.array([float(ranges[n][1]) for n in cont], float)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()) or (upper <= lower).any():
        raise NotComputed(
            "design_box_degenerate",
            "a continuous input has a zero-width or invalid design range",
            "measure runs that vary every process input",
        )
    direction = "at_least" if goal == "maximize" else "at_most"
    best_index = int(np.argmax(y) if goal == "maximize" else np.argmin(y))
    return FitPlan(
        X=X, y=y, lower=lower, upper=upper, cont_names=cont, cat_names=[str(c) for c in cats],
        cat_levels=cat_levels,
        cat_display=[
            [display_level(column_for[c], lvl) for lvl in levels]
            for c, levels in zip(cats, cat_levels)
        ],
        direction=direction, goal=goal, goal_source=goal_source,
        threshold=float(threshold), probability=float(probability), target=target,
        best_index=best_index, context={},
    )


def maximal_box(u: np.ndarray, meets: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The `gp_posterior_box` construction in the unit cube (see the module
    docstring). `u` (m x d) sampled points, `meets` (m,) bool, `anchor` (d,)
    the best measured recipe. Returns `(lo, hi)` tight to the passing samples
    inside, or None when no passing sample lies in the open box."""
    d = u.shape[1]
    fail = u[~meets]
    lo = np.zeros(d)
    hi = np.ones(d)
    if len(fail):
        r = float(np.min(np.max(np.abs(fail - anchor), axis=1)))
        lo = np.maximum(anchor - r, 0.0)
        hi = np.minimum(anchor + r, 1.0)
        for i in range(d):
            for side in (0, 1):
                others = np.ones(len(fail), bool)
                for j in range(d):
                    if j != i:
                        others &= (fail[:, j] > lo[j]) & (fail[:, j] < hi[j])
                col = fail[others, i]
                if side == 0:
                    below = col[col <= lo[i]]
                    lo[i] = float(below.max()) if len(below) else 0.0
                else:
                    above = col[col >= hi[i]]
                    hi[i] = float(above.min()) if len(above) else 1.0
    # Open on faces that touch a failing sample, closed on the design bound.
    inside = np.ones(len(u), bool)
    for i in range(d):
        left = u[:, i] >= lo[i] if lo[i] <= 0.0 else u[:, i] > lo[i]
        right = u[:, i] <= hi[i] if hi[i] >= 1.0 else u[:, i] < hi[i]
        inside &= left & right
    if (inside & ~meets).any():  # cannot happen by construction; never report it
        return None
    pts = u[inside & meets]
    if not len(pts):
        return None
    return pts.min(axis=0), pts.max(axis=0)


def _probability(mean: np.ndarray, sd: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    from scipy.stats import norm

    z = (np.asarray(mean, float) - threshold) / np.maximum(np.asarray(sd, float), 1e-12)
    return norm.cdf(z) if direction == "at_least" else norm.cdf(-z)


def fit_and_score(plan: FitPlan) -> dict[str, Any]:
    """The slot-bound part: refit the GP, score the Sobol sample, build the box.
    Raises `NotComputed` for a failed fit or no defensible region."""
    from scipy.stats import qmc

    from kalos.core.surrogate import FitError, Surrogate, proposal_restarts
    from kalos.portal.analysis import (
        ANALYZE_SEED,
        PROPOSAL_FIT_RESTARTS,
        _seeded_analysis,
    )
    from kalos.portal.pipeline import ensure_torch_threads

    ensure_torch_threads()
    n_cont = len(plan.cont_names)
    d = plan.X.shape[1]
    bounds = np.vstack([
        np.concatenate([plan.lower, np.zeros(d - n_cont)]),
        np.concatenate([plan.upper, np.array([max(len(lv) - 1, 1) for lv in plan.cat_levels], float)]),
    ])
    cat_dims = list(range(n_cont, d)) or None
    restarts = proposal_restarts(len(plan.y), d, PROPOSAL_FIT_RESTARTS)
    try:
        with _seeded_analysis(ANALYZE_SEED):
            model = Surrogate().fit(
                plan.X, plan.y, bounds=bounds, cat_dims=cat_dims,
                restarts=restarts, restart_seed=ANALYZE_SEED,
            )
    except (FitError, ValueError) as exc:
        raise NotComputed(
            "model_fit_failed", "the model could not be refit on the campaign's rows",
            "re-analyze the campaign; if that also fails, the data are ill-conditioned",
        ) from exc

    anchor_x = plan.X[plan.best_index]
    anchor_u = (anchor_x[:n_cont] - plan.lower) / (plan.upper - plan.lower)
    anchor_u = np.clip(anchor_u, 0.0, 1.0)
    u = qmc.Sobol(d=n_cont, scramble=True, seed=SOBOL_SEED).random(N_SAMPLES)
    x_cont = plan.lower + u * (plan.upper - plan.lower)
    x = np.column_stack([x_cont, np.tile(anchor_x[n_cont:], (len(u), 1))]) if d > n_cont else x_cont
    mean, sd = model.posterior(np.vstack([x, anchor_x[None, :]]))
    prob = _probability(mean, sd, plan.threshold, plan.direction)
    meets = prob[:-1] >= plan.probability
    anchor_prob = float(prob[-1])
    fraction = float(np.mean(meets))

    anchor = {
        "recipe": {
            **{n: float(anchor_x[i]) for i, n in enumerate(plan.cont_names)},
            **{
                n: plan.cat_display[k][int(anchor_x[n_cont + k])]
                for k, n in enumerate(plan.cat_names)
            },
        },
        "measured_value": float(plan.y[plan.best_index]),
        "measured_basis": "measured",
        "predicted_mean": float(mean[-1]),
        "predicted_sd": float(sd[-1]),
        "probability_meets_rule": round(anchor_prob, 4),
        "predicted_basis": "predicted",
    }
    held = {
        n: plan.cat_display[k][int(anchor_x[n_cont + k])] for k, n in enumerate(plan.cat_names)
    }
    diagnostics = {
        "fraction_of_design_space_meeting_rule": round(fraction, 4),
        "n_samples": int(len(u)),
        "anchor": anchor,
    }
    if anchor_prob < plan.probability:
        raise NotComputed(
            "no_region_found",
            "the model does not give the best measured recipe the required probability of meeting "
            "the threshold, so no region around it can be claimed",
            "lower the threshold or the probability, or measure more runs near the best recipe",
            diagnostics=diagnostics,
        )
    box = maximal_box(u, meets, anchor_u)
    n_inside = 0
    if box is not None:
        lo_u, hi_u = box
        inside = np.all((u >= lo_u) & (u <= hi_u), axis=1)
        n_inside = int(inside.sum())
    if box is None or n_inside < MIN_BOX_SAMPLES:
        raise NotComputed(
            "region_too_small",
            f"the box around the best recipe holds {n_inside} sampled points "
            f"(at least {MIN_BOX_SAMPLES} are needed to report it)",
            "lower the threshold or the probability, or measure more runs near the best recipe",
            diagnostics={**diagnostics, "n_samples_in_box": n_inside},
        )
    lo_u, hi_u = box
    lo = plan.lower + lo_u * (plan.upper - plan.lower)
    hi = plan.lower + hi_u * (plan.upper - plan.lower)
    ranges = [
        {
            "feature": name,
            "low": round(float(lo[i]), 6),
            "high": round(float(hi[i]), 6),
            "design_low": float(plan.lower[i]),
            "design_high": float(plan.upper[i]),
            "basis": "predicted",
        }
        for i, name in enumerate(plan.cont_names)
    ]
    return {
        "status": "computed",
        "method": METHOD,
        "basis": "predicted",
        "model_based": True,
        "must_verify_experimentally": True,
        "statement": (
            "Model-based estimate, not a measurement: the largest axis-aligned box around the "
            "best measured recipe (grown face by face, then tightened to the sampled points) in "
            "which EVERY sampled point has GP posterior probability >= "
            f"{plan.probability} of the target being {'at least' if plan.direction == 'at_least' else 'at most'} "
            f"{plan.threshold}. Verify it experimentally before relying on it."
        ),
        "rule": {
            "target": plan.target,
            "threshold": plan.threshold,
            "probability": plan.probability,
            "direction": plan.direction,
            "goal": plan.goal,
            "goal_source": plan.goal_source,
            "basis": "target",
        },
        "model": {
            "kind": "gp_posterior",
            "posterior": "latent (excludes assay noise)",
            "noise": "inferred",
            "refit_rows": int(len(plan.y)),
            "fit_restarts": int(restarts),
            "seed": ANALYZE_SEED,
            "note": (
                "refit on the campaign's rows with the analysis's features and design box; "
                "probabilities come from the GP's Gaussian posterior (not conformal / not held-out calibrated)"
            ),
        },
        "sampling": {
            "method": "sobol_scrambled",
            "seed": SOBOL_SEED,
            "n_samples": int(len(u)),
            "space": "the continuous design box (analysis feature_ranges)",
            "categoricals_held_at": held,
        },
        "anchor": anchor,
        "ranges": ranges,
        "box": {
            "construction": (
                "largest normalized cube around the anchor free of failing samples, each face then "
                "pushed out (features in analysis order, lower then upper) to the nearest failing "
                "sample or the design bound, then tightened to the passing samples inside; maximal, "
                "not necessarily maximum-volume"
            ),
            "n_samples_inside": n_inside,
            "all_samples_inside_meet_rule": True,
            "min_samples_required": MIN_BOX_SAMPLES,
        },
        "fraction_of_design_space_meeting_rule": round(fraction, 4),
        "fraction_scope": (
            "share of the Sobol sample of the continuous design box that meets the rule"
            + (", with categorical inputs held at the anchor's levels" if plan.cat_names else "")
        ),
    }
