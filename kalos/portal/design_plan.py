"""Cold-start exploration designs for a campaign (no model in them).

The numeric generator is `kalos.core.design` (torch-free). This module turns a
campaign into a design request and a design back into an honest response:

- `campaign_design`: `POST /api/campaigns/{id}/design` (kalos.portal.design_api).
  Factors come from the campaign setup (bounds, linear constraints, clone column
  and policy), the latest analysis (features, categoricals, observed ranges) and
  the request (setpoint steps or level lists, units, categorical levels, and
  bounds for factors the setup does not bound). Runs already measured or in
  flight are augmented (gap filling), never ignored.
- `analysis_design`: the same design for the analysis itself, attached as
  `exploration_design` whenever the model is not trusted (F04: withheld or
  exploration), so a refusal comes with a concrete plan, not only advice.
- `design_csv`: the plan in the run-sheet (upload template) layout.

Honesty rules, enforced here and tested (tests/test_cold_start_design.py):
every row reads LABEL ("exploration design, no model yet"), no row carries a
predicted value, standard deviation, interval or expected improvement, and the
response says why the model is not used (`model_not_used`).
"""
from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from kalos.core.design import (
    CategoricalFactor,
    ContinuousFactor,
    DesignError,
    DesignResult,
    LinearConstraint,
    generate_design,
)
from kalos.portal.analysis_parts.trust_gate import MIN_ROWS_TO_JUDGE
from kalos.portal.verdict import MIN_GROUPS_FOR_RELIABLE, MIN_RUNS_FOR_RELIABLE

LABEL = "exploration design, no model yet"
LABEL_MODEL_TRUSTED = "exploration design, model not used"
KIND = "exploration_design"
# Default method and starting-round size: chosen from the benchmark in
# benchmarks/cold_start/ (see its RESULTS.md and docs/COLD_START_DESIGN.md).
DEFAULT_METHOD = "sobol"
DEFAULT_N = 12
API_METHODS = ("sobol", "lhs_maximin")
MAX_DESIGN_ROWS = 96
# Rows in the analysis's `exploration_design`: the size of the batch it stands in for.
ANALYSIS_DESIGN_ROWS = 5
MAX_CENTRE_POINTS = 8
MAX_REPLICATE_SLOTS = 16
MIN_UPLOAD_ROWS = 6  # kalos.portal.uploads: a run sheet needs at least 6 runs
DEFAULT_TARGET_COLUMN = "result"
RUN_ID_COLUMN = "run_id"
# Fields a design row must never carry (tested): nothing predicted.
FORBIDDEN_ROW_KEYS = ("pred", "std", "mean", "interval", "intervals", "ei", "p_improve", "quantiles")


# --------------------------------------------------------------------------- #
# why the model is not used
# --------------------------------------------------------------------------- #


def model_not_used(
    *, n_measured: int, analysis: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """`{code, message, n_measured, thresholds, verdict, proposal_trust}`: the
    reason this plan comes from a design and not from the model."""
    thresholds = {
        "min_runs_to_judge": MIN_ROWS_TO_JUDGE,
        "min_runs_for_reliable": MIN_RUNS_FOR_RELIABLE,
        "min_groups_for_reliable": MIN_GROUPS_FOR_RELIABLE,
    }
    trust = (analysis or {}).get("proposal_trust")
    status = trust.get("status") if isinstance(trust, Mapping) else None
    verdict = (analysis or {}).get("verdict")
    vstate = verdict.get("state") if isinstance(verdict, Mapping) else None
    vreasons = [
        r.get("code") for r in (verdict.get("reasons") or []) if isinstance(r, Mapping)
    ] if isinstance(verdict, Mapping) else []
    base = {"n_measured": int(n_measured), "thresholds": thresholds,
            "verdict": vstate, "verdict_reasons": vreasons, "proposal_trust": status}
    if n_measured == 0:
        return {**base, "code": "no_measured_runs",
                "message": "No run has been measured yet, so there is nothing to fit a model to. "
                           "This plan only spreads the first runs over the declared region."}
    if status == "trusted":
        return {**base, "code": "model_trusted_not_used",
                "message": "The model is trusted on this campaign and its model-guided proposals are "
                           "in the analysis. This plan deliberately ignores the model: it only fills "
                           "gaps between the runs already made."}
    if n_measured < MIN_ROWS_TO_JUDGE:
        return {**base, "code": "n_below_trust_threshold",
                "message": f"{n_measured} measured run(s): the model is not judged below "
                           f"{MIN_ROWS_TO_JUDGE} runs and is not called reliable below "
                           f"{MIN_RUNS_FOR_RELIABLE}, so the next runs should cover the region "
                           "rather than follow an unchecked model."}
    if n_measured < MIN_RUNS_FOR_RELIABLE:
        return {**base, "code": "n_below_trust_threshold",
                "message": f"{n_measured} measured runs: Kalos does not call the model reliable below "
                           f"{MIN_RUNS_FOR_RELIABLE} runs (and {MIN_GROUPS_FOR_RELIABLE} groups), so "
                           "this plan fills the gaps between them instead."}
    return {**base, "code": "model_not_trusted",
            "message": "The model has not shown it can rank held-out past runs (verdict: "
                       f"{vstate or 'not computed'}), so this plan fills the gaps between the runs "
                       "already made instead of following it."}


# --------------------------------------------------------------------------- #
# physical-range check (unit-aware when kalos.validation.bounds offers it)
# --------------------------------------------------------------------------- #


def physical_check(name: str, unit: str | None, lo: float, hi: float) -> dict[str, Any]:
    """Refuse a bound outside the physically possible range of the column's
    dimension; warn outside the typical range. Uses
    `kalos.validation.bounds.column_bounds` (unit-aware) when available; on an
    engine without it, only unit-free dimensions (pH) are checked. Unknown
    dimension: not checked, and said so."""
    from kalos.validation import bounds as vb

    header = f"{name} ({unit})" if unit else name
    column_bounds = getattr(vb, "column_bounds", None)
    hard: tuple[float, float] | None = None
    typical: tuple[float, float] | None = None
    dim: str | None = None
    source = None
    if callable(column_bounds):
        cb = column_bounds(header)
        if cb is not None:
            dim, hard, typical = cb.dimension, (cb.hard_lo, cb.hard_hi), (cb.typical_lo, cb.typical_hi)
            source = "kalos.validation.bounds.column_bounds"
    else:
        dim = vb.infer_dimension(header)
        spec = vb.DIMENSION_BOUNDS.get(dim) if dim else None
        if spec is not None and spec.unit == "":
            hard, typical = (spec.hard_lo, spec.hard_hi), (spec.typical_lo, spec.typical_hi)
            source = "kalos.validation.bounds.DIMENSION_BOUNDS (unit-free dimensions only)"
    if hard is None:
        return {"checked": False, "dimension": dim, "reason": "no physical range is known for this column"}
    if lo < hard[0] or hi > hard[1]:
        raise DesignError(
            "physically_impossible_bounds",
            f"a factor's bounds reach outside what is physically possible for its dimension "
            f"({dim}: {hard[0]:g} to {hard[1]:g})",
            "set bounds inside the physical range (check the unit)",
        )
    out: dict[str, Any] = {"checked": True, "dimension": dim, "source": source,
                           "hard": list(hard), "typical": list(typical) if typical else None}
    if typical is not None and (lo < typical[0] or hi > typical[1]):
        out["warning"] = (f"the bounds reach outside the typical range for {dim} "
                          f"({typical[0]:g} to {typical[1]:g}); possible, but check them")
    return out


# --------------------------------------------------------------------------- #
# building the request
# --------------------------------------------------------------------------- #


def _finite(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
    v = float(v)
    return v if math.isfinite(v) else None


def _label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    text = str(v).strip()
    return text or None


def _existing_points(
    state: Mapping[str, Any] | None, cont: Sequence[str], cats: Sequence[str], include_pending: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Measured base rows plus started runs (awaiting or measured, not yet folded)."""
    rows: list[Mapping[str, Any]] = []
    counts = {"n_measured": 0, "n_pending": 0, "n_awaiting": 0}
    if state:
        target = state.get("target")
        for r in state.get("base_rows") or []:
            if isinstance(r, Mapping):
                rows.append(r)
                if target is not None and _finite(r.get(target)) is not None:
                    counts["n_measured"] += 1
        for run in state.get("pending") or []:
            if not isinstance(run, Mapping) or not isinstance(run.get("recipe"), Mapping):
                continue
            measured = _finite(run.get("result")) is not None
            counts["n_pending"] += 1
            if measured:
                counts["n_measured"] += 1
            else:
                counts["n_awaiting"] += 1
            if include_pending or measured:
                rows.append(run["recipe"])
    pts = np.array(
        [[_finite(r.get(c)) if _finite(r.get(c)) is not None else np.nan for c in cont] for r in rows],
        dtype=float,
    ).reshape(len(rows), len(cont))
    labels = [{c: _label(r.get(c)) for c in cats} for r in rows]
    return pts, labels, counts


def _observed_levels(state: Mapping[str, Any] | None, column: str) -> list[str]:
    seen: dict[str, None] = {}
    for r in (state or {}).get("base_rows") or []:
        if isinstance(r, Mapping):
            lv = _label(r.get(column))
            if lv is not None:
                seen.setdefault(lv, None)
    return sorted(seen)


def build_spec(
    *,
    setup: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    factors: Mapping[str, Mapping[str, Any]] | None = None,
    categoricals: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """The factor list for a design: `{continuous, categorical, constraints,
    factor_report, not_designed, warnings, columns}`. Raises `DesignError`."""
    from kalos.portal.setup_settings import dataset_context

    setup = setup or {}
    factors = dict(factors or {})
    categoricals = {k: list(v) for k, v in (categoricals or {}).items()}
    ctx = dataset_context(dict(analysis)) if analysis else None
    s_bounds: Mapping[str, Mapping[str, float]] = setup.get("bounds") or {}
    warnings: list[str] = []
    not_designed: list[dict[str, Any]] = []

    strain_col = setup.get("strain_column") or (ctx.strain_column if ctx else None)
    policy = setup.get("strain_policy") or ("fixed" if strain_col else None)
    covariates = set(setup.get("covariates") or (ctx.covariates if ctx else ()))
    target = ctx.target if ctx else None

    # Continuous names: the analysis's continuous features (seeded campaign), then
    # setup-bounded and request-declared names that are not categorical.
    cont_names: list[str] = []
    cat_names: list[str] = []
    if ctx is not None:
        cont_names = [c for c in ctx.continuous if c not in covariates]
        cat_names = [c for c in ctx.categorical if c not in covariates]
        unknown = [k for k in [*factors, *categoricals] if k not in ctx.continuous and k not in ctx.categorical]
        if unknown:
            raise DesignError(
                "unknown_factor",
                f"{len(unknown)} declared factor(s) are not settings of this campaign's current analysis",
                "declare only the campaign's own feature columns (analysis `features`)",
            )
        if strain_col and strain_col not in cat_names and strain_col not in cont_names:
            cat_names.append(strain_col)
    else:
        for name in [*s_bounds, *factors]:
            if name not in cont_names and name not in categoricals:
                cont_names.append(name)
        cat_names = list(categoricals)
        if strain_col and strain_col not in cat_names:
            raise DesignError(
                "clone_levels_required",
                "the setup names a clone column, but the campaign has no data and the request does not "
                "list its clone labels",
                "list the clone labels under `categoricals` for the clone column",
            )
    for c in sorted(covariates):
        not_designed.append({"column": c, "reason": "covariate: a recorded run property, never set by a plan"})
    if target is not None and (target in factors or target in categoricals):
        raise DesignError("target_not_a_factor", "the target cannot be a design factor")

    continuous: list[ContinuousFactor] = []
    report: list[dict[str, Any]] = []
    for name in cont_names:
        spec = factors.get(name) or {}
        lo_r, hi_r = spec.get("lo"), spec.get("hi")
        unit = spec.get("unit")
        if name in s_bounds:
            lo, hi = float(s_bounds[name]["lo"]), float(s_bounds[name]["hi"])
            source = "setup_bounds"
            if (lo_r is not None and float(lo_r) != lo) or (hi_r is not None and float(hi_r) != hi):
                raise DesignError(
                    "bounds_conflict",
                    "a declared factor's bounds differ from the campaign setup's bounds for the same column",
                    "change the bounds in the campaign setup (PUT .../setup), or omit lo/hi here",
                )
        elif lo_r is not None and hi_r is not None:
            lo, hi, source = float(lo_r), float(hi_r), "request"
        elif ctx is not None and name in ctx.ranges:
            lo, hi = ctx.ranges[name]
            source = "observed_range"
        else:
            raise DesignError(
                "factor_without_bounds",
                "a design factor has no bounds (not in the setup, not declared, and never observed)",
                "give it bounds in the campaign setup or as lo/hi in `factors`",
            )
        if not lo < hi:
            raise DesignError(
                "invalid_bounds",
                "a design factor's range is empty (lo must be below hi; an observed column that never "
                "varied cannot be designed)",
                "declare bounds for it in the campaign setup",
            )
        levels = spec.get("levels")
        try:
            cf = ContinuousFactor(name, lo, hi, step=spec.get("step"),
                                  levels=tuple(float(v) for v in levels) if levels is not None else None)
        except DesignError:
            raise
        phys = physical_check(name, unit, lo, hi)
        if phys.get("warning"):
            warnings.append(phys["warning"])
        observed = ctx.ranges.get(name) if ctx is not None else None
        beyond = bool(observed is not None and (lo < observed[0] or hi > observed[1]))
        report.append({
            "name": name, "kind": "discrete" if cf.discrete else "continuous", "lo": lo, "hi": hi,
            "bounds_source": source, "unit": unit,
            "step": cf.step, "levels": [float(v) for v in cf.grid()] if cf.levels is not None else None,
            "snapped": cf.discrete,
            "observed_range": list(observed) if observed is not None else None,
            "beyond_observed_range": beyond, "physical_check": phys,
        })
        if source == "observed_range":
            warnings.append("factors without setup bounds stay inside the range already measured; "
                            "set bounds in the campaign setup to explore beyond it")
        continuous.append(cf)

    categorical: list[CategoricalFactor] = []
    for name in cat_names:
        levels = categoricals.get(name) or _observed_levels(state, name)
        levels = [str(v).strip() for v in levels if str(v).strip()]
        if not levels:
            raise DesignError("categorical_without_levels", "a categorical factor has no levels",
                              "list its levels under `categoricals`")
        fixed = None
        role = "categorical"
        if name == strain_col:
            role = "clone"
            if policy == "fixed":
                fixed = setup.get("strain_level")
                if fixed is None and analysis:
                    ss = analysis.get("strain_summary")
                    fixed = ss.get("fixed_level") if isinstance(ss, Mapping) else None
                if fixed is None:
                    raise DesignError(
                        "clone_level_required",
                        "the clone policy is 'fixed' but no clone is named to hold every run at",
                        "set strain_level in the campaign setup, or use strain_policy 'factor' to "
                        "spread the runs over the clones",
                    )
                if fixed not in levels:
                    levels = [*levels, fixed]
        categorical.append(CategoricalFactor(name, tuple(dict.fromkeys(levels)), fixed=fixed))
        report.append({"name": name, "kind": role, "levels": list(dict.fromkeys(levels)),
                       "policy": ("held" if fixed is not None else "balanced"),
                       "held_level": fixed, "clone_policy": policy if role == "clone" else None})

    constraints = [
        LinearConstraint(dict(c["coefficients"]), str(c["op"]), float(c["rhs"]))
        for c in setup.get("linear_constraints") or []
    ]
    if constraints and not continuous:
        raise DesignError("constraint_unknown_factor", "linear constraints need continuous factors")
    columns: list[str] = []
    if state and state.get("base_rows"):
        for r in state["base_rows"]:
            if isinstance(r, Mapping):
                for k in r:
                    if k not in columns:
                        columns.append(str(k))
    return {"continuous": continuous, "categorical": categorical, "constraints": constraints,
            "factor_report": report, "not_designed": not_designed, "warnings": warnings,
            "columns": columns, "target": target, "strain_column": strain_col}


# --------------------------------------------------------------------------- #
# the response
# --------------------------------------------------------------------------- #


def _assemble(
    res: DesignResult, spec: Mapping[str, Any], reason: Mapping[str, Any], counts: Mapping[str, int],
    *, include_pending: bool,
) -> dict[str, Any]:
    label = LABEL_MODEL_TRUSTED if reason.get("code") == "model_trusted_not_used" else LABEL
    width = max(2, len(str(len(res.rows))))
    rows = []
    for i, (rec, role, src) in enumerate(zip(res.rows, res.roles, res.replicate_of), 1):
        rows.append({
            "row": i, "run_label": f"design-{i:0{width}d}", "role": role,
            "replicate_of": None if src is None else src + 1,
            "recipe": rec, "label": label, "model_used": False,
        })
    warnings = list(spec.get("warnings") or [])
    warnings += list(res.notes)
    warnings = list(dict.fromkeys(warnings))
    return {
        "kind": KIND,
        "label": label,
        "model_used": False,
        "predictions": None,
        "predictions_note": "No predicted value, standard deviation or interval is given for any row: "
                            "no model was used to choose them.",
        "model_not_used": dict(reason),
        "method": res.method,
        "requested_method": res.requested_method,
        "seed": res.seed,
        "deterministic": "the same request on the same campaign data returns the same rows",
        "n_rows": len(rows),
        "rows": rows,
        "factors": list(spec.get("factor_report") or []),
        "not_designed": list(spec.get("not_designed") or []),
        "constraints": res.constraints,
        "snapping": res.snapping,
        "levels": res.levels,
        "quality": res.quality,
        "existing": {
            "n_measured": int(counts.get("n_measured", 0)),
            "n_awaiting": int(counts.get("n_awaiting", 0)),
            "pending_included": bool(include_pending),
            "n_used_for_gap_filling": res.n_existing,
            "mode": "augment (fill gaps between existing runs)" if res.n_existing else "fresh design",
        },
        "shortfall": res.shortfall,
        "warnings": warnings,
        "next_step": (
            "Run these recipes and record the results (start them as runs with mode "
            "'exploration_design', or download the CSV, fill in the target and upload it). "
            f"The model is judged from {MIN_ROWS_TO_JUDGE} measured runs and can be called reliable "
            f"from {MIN_RUNS_FOR_RELIABLE}."
        ),
    }


def campaign_design(
    *,
    setup: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    n: int,
    method: str = DEFAULT_METHOD,
    seed: int = 0,
    centre_points: int = 0,
    replicate_slots: int = 0,
    factors: Mapping[str, Mapping[str, Any]] | None = None,
    categoricals: Mapping[str, Sequence[str]] | None = None,
    include_pending: bool = True,
) -> dict[str, Any]:
    """The design block for a campaign. Raises `DesignError`."""
    total = n + centre_points + replicate_slots
    if total > MAX_DESIGN_ROWS or total < 1:
        raise DesignError("invalid_size", f"a design has 1 to {MAX_DESIGN_ROWS} rows in total "
                                          "(n + centre_points + replicate_slots)")
    spec = build_spec(setup=setup, analysis=analysis, state=state, factors=factors,
                      categoricals=categoricals)
    cont = [f.name for f in spec["continuous"]]
    cats = [c.name for c in spec["categorical"]]
    pts, labels, counts = _existing_points(state, cont, cats, include_pending)
    res = generate_design(
        spec["continuous"], spec["categorical"], spec["constraints"], n=n, method=method, seed=seed,
        existing=pts if len(pts) else None, existing_labels=labels,
        centre_points=centre_points, replicate_slots=replicate_slots,
    )
    reason = model_not_used(n_measured=counts["n_measured"], analysis=analysis)
    out = _assemble(res, spec, reason, counts, include_pending=include_pending)
    if counts["n_measured"] == 0 and len(out["rows"]) < MIN_UPLOAD_ROWS:
        out["warnings"].append(
            f"a first run sheet needs at least {MIN_UPLOAD_ROWS} measured runs before Kalos can analyse it; "
            f"this plan has {len(out['rows'])}")
    out["template"] = {
        "columns": template_columns(spec, state),
        "target": spec.get("target") or DEFAULT_TARGET_COLUMN,
    }
    return out


def analysis_design(
    *,
    names: Sequence[str],
    categorical: Mapping[str, Sequence[str]],
    bounds: np.ndarray,
    X: np.ndarray,
    setup: Mapping[str, Any] | None,
    held: Mapping[str, Any],
    n: int,
    analysis_view: Mapping[str, Any],
    n_measured: int,
) -> dict[str, Any]:
    """`exploration_design` for an analysis whose model is not trusted.

    `names`/`bounds`/`X` are the analysis's design space (continuous first, then
    categorical codes), `categorical` the level list per categorical, `held` the
    columns every proposal holds (a fixed clone level, covariates) with their
    values. Region: the setup's typed bounds where declared, else the observed
    range. Never raises: a design that cannot be built is reported as
    `{available: false, reason}`."""
    try:
        s_bounds = (setup or {}).get("bounds") or {}
        cont: list[ContinuousFactor] = []
        report: list[dict[str, Any]] = []
        cont_idx: list[int] = []
        for j, name in enumerate(names):
            if name in categorical or name in held:
                continue
            lo, hi = float(bounds[0, j]), float(bounds[1, j])
            src = "observed_range"
            if name in s_bounds:
                lo, hi, src = float(s_bounds[name]["lo"]), float(s_bounds[name]["hi"]), "setup_bounds"
            if not lo < hi:
                continue
            cont.append(ContinuousFactor(name, lo, hi))
            cont_idx.append(j)
            report.append({"name": name, "kind": "continuous", "lo": lo, "hi": hi, "bounds_source": src,
                           "step": None, "levels": None, "snapped": False})
        cats: list[CategoricalFactor] = []
        cat_idx: list[int] = []
        for j, name in enumerate(names):
            if name not in categorical:
                continue
            levels = tuple(str(v) for v in categorical[name])
            fixed = str(held[name]) if name in held else None
            cats.append(CategoricalFactor(name, levels, fixed=fixed))
            cat_idx.append(j)
            report.append({"name": name, "kind": "categorical", "levels": list(levels),
                           "policy": "held" if fixed is not None else "balanced", "held_level": fixed})
        constraints = [
            LinearConstraint(dict(c["coefficients"]), str(c["op"]), float(c["rhs"]))
            for c in (setup or {}).get("linear_constraints") or []
            if all(k in {f.name for f in cont} for k in c["coefficients"])
        ]
        existing = X[:, cont_idx] if cont_idx else None
        labels = [
            {names[j]: categorical[names[j]][int(round(X[i, j]))] for j in cat_idx
             if 0 <= int(round(X[i, j])) < len(categorical[names[j]])}
            for i in range(X.shape[0])
        ]
        res = generate_design(cont, cats, constraints, n=n, method=DEFAULT_METHOD, seed=0,
                              existing=existing, existing_labels=labels)
        held_cont = {k: v for k, v in held.items() if k not in categorical}
        if held_cont:
            for r in res.rows:
                r.update(held_cont)
        spec = {"factor_report": report, "warnings": [], "not_designed": [
            {"column": k, "reason": "held at the value every proposal holds (covariate)"} for k in held_cont]}
        reason = model_not_used(n_measured=n_measured, analysis=analysis_view)
        out = _assemble(res, spec, reason, {"n_measured": n_measured, "n_awaiting": 0}, include_pending=False)
        out["available"] = True
        out["answers"] = (
            "reliability.space_filling_advice"
            if (analysis_view.get("proposal_trust") or {}).get("status") != "withheld"
            else "proposal_notice (model_refused: proposals withheld)"
        )
        out["route"] = "POST /api/campaigns/{id}/design (more rows, setpoint steps, centre points, CSV)"
        return out
    except DesignError as exc:
        return {"available": False, "kind": KIND, "code": exc.code, "reason": exc.reason}
    except Exception as exc:  # never fail the analysis for a design  # noqa: BLE001
        return {"available": False, "kind": KIND, "code": "design_failed",
                "reason": f"the exploration design could not be built ({type(exc).__name__})"}


def attach_exploration_design(
    result: dict[str, Any],
    *,
    strain: Any,
    names: Sequence[str],
    categorical: Mapping[str, Sequence[str]],
    bounds: np.ndarray,
    X: np.ndarray,
    setup: Mapping[str, Any] | None,
    verdict: Mapping[str, Any],
    n_measured: int,
) -> None:
    """The analysis hook (F04 small-sample path). When `proposal_trust` is not
    `trusted` (the batch is withheld, or labelled exploration and
    `reliability.space_filling_advice` applies), add `exploration_design`: 5
    gap-filling rows. Additive only, with its own seeded RNG, so every other
    field stays byte-identical. `strain` is the analysis's `StrainState`: a
    fixed clone level and held covariates are held, as proposals hold them."""
    trust = result.get("proposal_trust") or {}
    if trust.get("status") == "trusted":
        return
    held: dict[str, Any] = {k: v["value"] for k, v in strain.held_info.items()}
    if strain.fixed_level is not None and strain.plan.column is not None:
        held[strain.plan.column] = strain.fixed_level
    result["exploration_design"] = analysis_design(
        names=names, categorical=categorical, bounds=bounds, X=X, setup=setup, held=held,
        n=ANALYSIS_DESIGN_ROWS, analysis_view={"proposal_trust": trust, "verdict": verdict},
        n_measured=n_measured,
    )


# --------------------------------------------------------------------------- #
# CSV (run-sheet template)
# --------------------------------------------------------------------------- #

CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _neutralize(text: str) -> str:
    return "'" + text if text.startswith(CSV_FORMULA_PREFIXES) else text


def template_columns(spec: Mapping[str, Any], state: Mapping[str, Any] | None) -> list[str]:
    """The upload layout: the campaign's own columns in sheet order when it has
    data; else `run_id`, the factors, the target. A `run_id` column is added in
    front when the sheet has no id column the engine recognises."""
    from kalos.portal.analysis_parts.identifiers import _is_identifier_name

    cols = list(spec.get("columns") or [])
    factor_names = [f.name for f in spec["continuous"]] + [c.name for c in spec["categorical"]]
    target = spec.get("target") or DEFAULT_TARGET_COLUMN
    if not cols:
        cols = [*factor_names, target]
    else:
        for name in factor_names:
            if name not in cols:
                cols.append(name)
        if target not in cols:
            cols.append(target)
    if not any(_is_identifier_name(c) for c in cols):
        name = RUN_ID_COLUMN
        while name in cols:
            name = "_" + name
        cols = [name, *cols]
    return cols


def design_csv(design: Mapping[str, Any]) -> str:
    """The plan as a run sheet: the template's header row, then one row per
    design row. The id column carries the row's `run_label`; the target and every
    column the plan does not set are blank. Text starting with = + - @ TAB or CR
    gets a leading apostrophe (formula neutralization)."""
    from kalos.portal.analysis_parts.identifiers import _is_identifier_name

    cols: list[str] = list(design["template"]["columns"])
    id_col = next((c for c in cols if _is_identifier_name(c)), None)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow([_neutralize(c) for c in cols])
    for row in design["rows"]:
        rec = row["recipe"]
        cells = []
        for c in cols:
            if c == id_col:
                cells.append(row["run_label"])
            elif c in rec:
                v = rec[c]
                cells.append(repr(float(v)) if isinstance(v, float) else _neutralize(str(v)))
            else:
                cells.append("")
        w.writerow(cells)
    return buf.getvalue()


__all__ = [
    "API_METHODS", "DEFAULT_METHOD", "DEFAULT_N", "FORBIDDEN_ROW_KEYS", "KIND", "LABEL",
    "MAX_DESIGN_ROWS", "analysis_design", "attach_exploration_design", "build_spec", "campaign_design", "design_csv",
    "model_not_used", "physical_check", "template_columns",
]
