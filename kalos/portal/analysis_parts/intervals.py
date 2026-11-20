"""Per-proposal intervals, predictive quantiles and improvement probabilities
(VIZ_SPEC engine request R4).

Two kinds of band, never mixed and always labeled:

- `intervals` (`method: "conformal_normalized"`, `calibrated_on:
  "grouped_cv_oof"`): normalized split conformal. Each grouped-CV out-of-fold
  point is scored `|y - mu_oof| / sigma_oof`, where `sigma_oof` is the held-out
  fold GP's PREDICTIVE sd (latent + fitted observation noise, see
  `kalos.core.evaluation._oof`). The finite-sample quantile `q_norm` of those
  scores (rank `ceil((n+1)(1-alpha))`) is scaled by each proposal's own
  predictive sd: `pred_i +/- q_norm * sigma_pred_i`. Held-out calibrated, but
  only approximately (grouped OOF scores come from fold models, the proposal
  from the full-data model). A level is refused, per level, when n is too small
  for its finite-sample rank or the OOF sds are degenerate.
- `gp_intervals` / `quantiles` (`method`/`basis: "gp_posterior"`): the GP's own
  Gaussian predictive distribution for ONE NEW MEASUREMENT (observation, i.e.
  latent + observation noise). Model-based, not held-out calibrated.

`expected_improvement` and `p_improve` are computed on the LATENT posterior
(the process mean, no observation noise) against the incumbent the proposal
path uses, the convention of analytic EI. Model-based.

Every band carries the BASIS of its coverage claim (F02, ML audit B1):
`coverage_basis` is `"held_out_past_runs"` for the conformal bands (calibrated
on grouped-CV out-of-fold runs, i.e. runs like past runs) and `"model_based"`
for the GP's own bands. The conformal bands at proposals also carry a
structured `caveat` (`code: "coverage_lower_at_proposals"`, a plain message and
`coverage_at_proposals_estimate`, the coverage MEASURED at proposals in
simulation with its source; `kalos/portal/analysis_parts/coverage_evidence.py`).
A client must not say a proposal's range "is calibrated": it shows the caveat,
which names what the band was calibrated on and the coverage measured at
proposed recipes.

Torch-free: the surrogate is only ever called through its `posterior` method.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from kalos.core.conformal import normalized_conformal_quantile, normalized_scores
from kalos.portal.analysis_parts.coverage_evidence import (
    BASIS_HELD_OUT,
    BASIS_MODEL,
    BASIS_SELECTION_AWARE,
    CAVEAT_CODE,
    compact,
    coverage_at_proposals_estimate,
    proposal_caveat,
)

METHOD_CONFORMAL = "conformal_normalized"
CALIBRATED_ON = "grouped_cv_oof"
METHOD_GP = "gp_posterior"
GP_NOTE = "model-based, not held-out calibrated"
QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
CALIBRATED_ON_SELECTED = "grouped_cv_oof_top_ranked"

# How `sigma_pred` at a proposal was obtained (reported in `interval_method`).
SIGMA_FITTED_NOISE = "surrogate_fitted_noise"
SIGMA_ASSAY_FLOOR = "latent_plus_assay_noise_floor"


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _sig(x: float, digits: int = 4) -> float:
    """Significant-figure rounding, so a tiny EI or probability is not shown as 0."""
    return float(f"{float(x):.{digits}g}")


def _fix_text(code: str, n: int, min_n: int | None) -> str:
    if code == "too_few_calibration_points" and min_n is not None:
        return (f"record at least {max(min_n - n, 1)} more run(s) so grouped cross-validation yields "
                f"{min_n} or more held-out predictions; the global conformal_q band still applies")
    if code == "degenerate_sigma":
        return ("the held-out GP sd collapsed (often exact duplicate rows or a fit at its noise floor); "
                "use the global conformal_q band, which does not depend on the sd")
    return "more held-out data"


def calibrate_normalized(
    oof_actual, oof_pred, oof_std, levels: Sequence[float],
) -> dict:
    """Per-level normalized-conformal calibration from grouped-CV OOF predictions.

    Returns `{"n_calibration", "levels": [...]}`; each level entry is either
    `{"coverage", "available": True, "q_norm", "rank", "n_calibration",
    "empirical_oof_coverage", "n_inside", "coverage_by_sigma_half"}` or
    `{"coverage", "available": False, "code", "reason", "what_would_fix_it",
    "n_calibration", ...}`. Never raises on data.

    `empirical_oof_coverage` is IN-SAMPLE: it counts the same scores that set
    `q_norm`, so it is >= rank / n by construction. `coverage_by_sigma_half`
    splits those points at the median OOF sd and shows whether the band adapts
    (equal coverage in both halves) or not.
    """
    sc = normalized_scores(oof_actual, oof_pred, oof_std)
    n = int(sc["n"])
    out_levels: list[dict] = []
    for lv in levels:
        cov = float(lv)
        alpha = 1.0 - cov
        if not sc["ok"]:
            out_levels.append({
                "coverage": _r(cov), "available": False, "code": sc["code"], "reason": sc["reason"],
                "what_would_fix_it": _fix_text(sc["code"], n, None), "n_calibration": n,
            })
            continue
        qd = normalized_conformal_quantile(sc["scores"], alpha)
        if not qd["available"]:
            out_levels.append({
                "coverage": _r(cov), "available": False, "code": qd["code"], "reason": qd["reason"],
                "what_would_fix_it": _fix_text(qd["code"], n, qd["min_n"]),
                "n_calibration": n, "rank": qd["rank"], "min_n": qd["min_n"],
            })
            continue
        scores = np.asarray(sc["scores"], float)
        q = float(qd["q"])
        inside = scores <= q
        sd = np.asarray(oof_std, float).reshape(-1)
        med = float(np.median(sd))
        low = sd <= med
        halves = {}
        for name, mask in (("low_sigma", low), ("high_sigma", ~low)):
            k = int(mask.sum())
            halves[name] = {"n": k, "coverage": _r(inside[mask].mean()) if k else None}
        out_levels.append({
            "coverage": _r(cov),
            "available": True,
            "q_norm": _r(q),
            "rank": int(qd["rank"]),
            "n_calibration": n,
            "empirical_oof_coverage": _r(inside.mean()),
            "n_inside": int(inside.sum()),
            "coverage_by_sigma_half": halves,
            "_q": q,  # unrounded, internal; stripped by interval_method_summary
        })
    return {"n_calibration": n, "levels": out_levels}


METHOD_SELECTION_AWARE = "conformal_selection_aware"
# F15 parameters, chosen on simulation seeds 0-39 (benchmarks/results/
# coverage_at_proposals_*.json, VALIDATION.md): the top 30% of each fold's
# held-out runs by the fold model's prediction (the runs a proposal step would
# have picked), pooled over the CV repeats, normalized scores, and the proposal
# sd floored at the 90th percentile of the held-out sds.
SELECTION_TOP_FRACTION = 0.3
SIGMA_FLOOR_QUANTILE = 0.9


def calibrate_selection_aware(oof_repeats, levels: Sequence[float], *, maximize: bool = True,
                              top_fraction: float = SELECTION_TOP_FRACTION,
                              sigma_floor_quantile: float = SIGMA_FLOOR_QUANTILE) -> dict:
    """Selection-aware normalized conformal (F15).

    Proposals are the recipes the model ranks highest, and their errors are not
    exchangeable with past runs: the model is optimistic there and its sd is
    smaller than the held-out sds the plain band was normalized by (measured
    sd ratio 0.6-0.8). This calibration mimics the selection inside the CV:

    1. in every fold of every repeat, keep the ceil(top_fraction * n_fold)
       held-out runs the FOLD model ranked best (highest prediction; lowest when
       minimizing) - the runs a proposal step would have picked;
    2. score them |y - mu| / sigma (the fold model's own predictive sd), and
       take `q_sel` = the ceil((n_sel + 1) * coverage)-th smallest score;
    3. at a proposal, use sd = max(predictive_sd, floor), where `floor` is the
       `sigma_floor_quantile` quantile of every held-out sd: the full-data
       model's sd at its own proposals was measured at 0.6-0.8 of the held-out
       sds while its errors there were larger, so the band never claims less
       uncertainty at a proposal than the model had on its less-certain
       held-out runs. Band: pred +/- q_sel * max(predictive_sd, floor).

    Not a guarantee: the pooled repeats re-use each run once per repeat with
    correlated errors, and held-out runs are only a proxy for proposals. The
    coverage at proposals is MEASURED by the benchmark and shipped with the
    band. A level is refused (with the plain normalized band as the fallback)
    when there are too few top-ranked runs for its rank.
    """
    sel_scores: list[float] = []
    all_sd: list[float] = []
    n_pool = 0
    ok = True
    for rep in oof_repeats or []:
        p = np.asarray(rep.get("pred", []), float)
        s = np.asarray(rep.get("std", []), float)
        a = np.asarray(rep.get("actual", []), float)
        f = np.asarray(rep.get("fold", []), int)
        if not (len(p) == len(s) == len(a) == len(f)) or len(p) == 0:
            continue
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s)) and np.all(np.isfinite(a)) and np.all(s > 0)):
            ok = False
            continue
        all_sd.extend(s.tolist())
        n_pool += len(p)
        rank_by = p if maximize else -p
        for k in np.unique(f):
            idx = np.where(f == k)[0]
            m = int(np.ceil(top_fraction * len(idx) - 1e-9))
            top = idx[np.argsort(-rank_by[idx], kind="stable")[:m]]
            sel_scores.extend((np.abs(a[top] - p[top]) / s[top]).tolist())
    floor = float(np.quantile(all_sd, sigma_floor_quantile)) if all_sd else float("nan")
    scores = np.asarray(sel_scores, float)
    n_sel = int(len(scores))
    out_levels: list[dict] = []
    for lv in levels:
        cov = float(lv)
        base = {"coverage": _r(cov), "n_selected": n_sel, "n_pooled": n_pool}
        if not ok or not np.isfinite(floor) or floor <= 0:
            out_levels.append({**base, "available": False, "code": "degenerate_sigma",
                               "reason": "a held-out sd is non-finite or zero"})
            continue
        k = int(np.ceil((n_sel + 1) * cov - 1e-9))
        if n_sel == 0 or k > n_sel:
            need = int(np.ceil(cov / (1 - cov) - 1e-9))  # smallest n with ceil((n+1)cov) <= n
            out_levels.append({**base, "available": False, "code": "too_few_top_ranked_runs", "rank": k,
                               "min_selected": need,
                               "reason": (f"{n_sel} top-ranked held-out runs; the {int(round(cov * 100))}% "
                                          f"selection-aware band needs {need}")})
            continue
        q = float(np.sort(scores)[k - 1])
        out_levels.append({**base, "available": True, "q_sel": _r(q), "rank": k, "_q": q})
    return {"method": METHOD_SELECTION_AWARE, "top_fraction": top_fraction,
            "sigma_floor_quantile": sigma_floor_quantile,
            "sigma_floor": None if not np.isfinite(floor) else _r(floor), "_floor": floor,
            "n_selected": n_sel, "n_pooled": n_pool, "levels": out_levels}


def proposal_predictive_std(
    surrogate: Any, batch, latent_std, assay_noise_var: float | None = None,
) -> tuple[np.ndarray, str]:
    """Predictive sd for ONE new measurement at each proposal, matching the
    sd the OOF scores were normalized by (latent + observation noise).

    With `assay_noise_var` (the replicate-aware path, whose surrogate has FIXED
    per-recipe-mean noise sigma^2 / n_reps), the single-measurement noise floor
    is added to the latent variance explicitly; BoTorch would otherwise use the
    mean fixed noise, i.e. the noise of a replicate MEAN. Otherwise the
    surrogate's own `observation_noise=True` posterior (fitted noise)."""
    lat = np.asarray(latent_std, float).reshape(-1)
    if assay_noise_var is not None:
        return np.sqrt(lat ** 2 + float(assay_noise_var)), SIGMA_ASSAY_FLOOR
    if lat.size == 0:
        return lat, SIGMA_FITTED_NOISE
    _, sd = surrogate.posterior(batch, observation_noise=True)
    return np.asarray(sd, float).reshape(-1), SIGMA_FITTED_NOISE


def expected_improvement(mean, std, incumbent: float, *, maximize: bool = True) -> np.ndarray:
    """Analytic EI of a Gaussian posterior vs `incumbent`.
    maximize: E[max(f - inc, 0)]; minimize: E[max(inc - f, 0)]. sd <= 0 gives
    the deterministic improvement."""
    mu = np.asarray(mean, float).reshape(-1)
    sd = np.asarray(std, float).reshape(-1)
    imp = (mu - incumbent) if maximize else (incumbent - mu)
    out = np.maximum(imp, 0.0)
    pos = sd > 0
    z = imp[pos] / sd[pos]
    out[pos] = imp[pos] * ndtr(z) + sd[pos] * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return out


def probability_of_improvement(mean, std, incumbent: float, *, maximize: bool = True) -> np.ndarray:
    """P(f > incumbent) (maximize) or P(f < incumbent) (minimize) under a
    Gaussian posterior. sd <= 0 gives 1.0 / 0.0."""
    mu = np.asarray(mean, float).reshape(-1)
    sd = np.asarray(std, float).reshape(-1)
    imp = (mu - incumbent) if maximize else (incumbent - mu)
    out = (imp > 0).astype(float)
    pos = sd > 0
    out[pos] = ndtr(imp[pos] / sd[pos])
    return out


def proposal_interval_fields(
    mean, latent_std, predictive_std, calibration: Mapping[str, Any], levels: Sequence[float],
    incumbent: float, incumbent_basis: str, *, maximize: bool = True,
    selection: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Additive per-proposal fields (one dict per proposal, merged by `_annotate`):
    `predictive_std`, `intervals`, `gp_intervals`, `quantiles`,
    `expected_improvement`, `p_improve`. See the module docstring."""
    mu = np.asarray(mean, float).reshape(-1)
    lat = np.asarray(latent_std, float).reshape(-1)
    pre = np.asarray(predictive_std, float).reshape(-1)
    ei = expected_improvement(mu, lat, incumbent, maximize=maximize)
    pi = probability_of_improvement(mu, lat, incumbent, maximize=maximize)
    cal_levels = list(calibration.get("levels", []))
    # Per proposal: the caveat with the measured numbers; the full source is
    # once per analysis, in `interval_method.coverage_at_proposals`.
    caveats = {}
    sel_levels = {float(lv["coverage"]): lv for lv in (selection or {}).get("levels", [])}
    sel_floor = float((selection or {}).get("_floor", float("nan")))
    for c in cal_levels:
        for meth in (METHOD_CONFORMAL, METHOD_SELECTION_AWARE):
            cav = proposal_caveat(meth, float(c["coverage"]))
            cav["coverage_at_proposals_estimate"] = compact(
                cav["coverage_at_proposals_estimate"],
                f"interval_method.coverage_at_proposals.{float(c['coverage']):g}")
            caveats[meth, float(c["coverage"])] = cav
    gp_estimates = {float(lv): compact(coverage_at_proposals_estimate(METHOD_GP, float(lv)),
                                       "interval_method.gp_coverage_at_proposals")
                    for lv in levels}
    rows: list[dict] = []
    for i in range(len(mu)):
        m, sp = float(mu[i]), float(pre[i])
        sp_ok = bool(np.isfinite(sp) and sp > 0)
        ivs: list[dict] = []
        for c in cal_levels:
            sl = sel_levels.get(float(c["coverage"]))
            if sl is not None and sl.get("available") and sp_ok and np.isfinite(sel_floor):
                # F15: the selection-aware band (see `calibrate_selection_aware`).
                half = float(sl["_q"]) * max(sp, sel_floor)
                ivs.append({
                    "coverage": c["coverage"], "method": METHOD_SELECTION_AWARE,
                    "calibrated_on": CALIBRATED_ON_SELECTED, "n_calibration": sl["n_selected"],
                    "coverage_basis": BASIS_SELECTION_AWARE,
                    "caveat": caveats[METHOD_SELECTION_AWARE, float(c["coverage"])],
                    "available": True, "lo": _r(m - half), "hi": _r(m + half),
                    "q_sel": sl["q_sel"], "sigma_floor": _r(sel_floor),
                    "sigma_used": _r(max(sp, sel_floor)),
                })
                continue
            base = {"coverage": c["coverage"], "method": METHOD_CONFORMAL, "calibrated_on": CALIBRATED_ON,
                    "n_calibration": c["n_calibration"], "coverage_basis": BASIS_HELD_OUT,
                    "caveat": caveats[METHOD_CONFORMAL, float(c["coverage"])]}
            if not c["available"]:
                ivs.append({**base, "available": False, "lo": None, "hi": None, "code": c["code"],
                            "reason": c["reason"], "what_would_fix_it": c["what_would_fix_it"]})
            elif not sp_ok:
                ivs.append({**base, "available": False, "lo": None, "hi": None,
                            "code": "degenerate_proposal_sigma",
                            "reason": "the predictive sd at this proposal is non-finite or zero",
                            "what_would_fix_it": "use the global conformal_q band"})
            else:
                half = float(c["_q"]) * sp
                ivs.append({**base, "available": True, "lo": _r(m - half), "hi": _r(m + half),
                            "q_norm": c["q_norm"], "empirical_oof_coverage": c["empirical_oof_coverage"]})
            if sl is not None:
                ivs[-1]["selection_aware_unavailable"] = {
                    "code": sl.get("code", "degenerate_proposal_sigma"),
                    "reason": sl.get("reason", "the predictive sd at this proposal is non-finite or zero"),
                }
        gp: list[dict] = []
        for lv in levels:
            cov = float(lv)
            entry: dict = {"coverage": _r(cov), "method": METHOD_GP, "posterior": "observation", "note": GP_NOTE,
                           "coverage_basis": BASIS_MODEL,
                           "coverage_at_proposals_estimate": gp_estimates[cov]}
            if np.isfinite(sp) and sp >= 0:
                half = float(ndtri(0.5 + cov / 2.0)) * sp
                entry.update(available=True, lo=_r(m - half), hi=_r(m + half))
            else:
                entry.update(available=False, lo=None, hi=None)
            gp.append(entry)
        quant: dict[str, Any] = {"basis": METHOD_GP, "posterior": "observation", "note": GP_NOTE,
                                 "coverage_basis": BASIS_MODEL}
        for p in QUANTILE_LEVELS:
            key = f"p{int(round(p * 100)):02d}"
            quant[key] = _r(m + float(ndtri(p)) * sp) if np.isfinite(sp) and sp >= 0 else None
        common = {"incumbent": _r(incumbent), "incumbent_basis": incumbent_basis, "maximize": bool(maximize),
                  "basis": "model", "posterior": "latent", "note": GP_NOTE}
        rows.append({
            "predictive_std": _r(sp) if np.isfinite(sp) else None,
            "intervals": ivs,
            "gp_intervals": gp,
            "quantiles": quant,
            "expected_improvement": {"value": _sig(ei[i]) if np.isfinite(ei[i]) else None,
                                     "method": "analytic_ei_gp_posterior", **common},
            "p_improve": {"value": _sig(pi[i]) if np.isfinite(pi[i]) else None,
                          "method": "gp_posterior_cdf", **common},
        })
    return rows


def interval_method_summary(
    calibration: Mapping[str, Any], sigma_source: str, selection: Mapping[str, Any] | None = None,
) -> dict:
    """Top-level `interval_method`: how `proposals[].intervals` were built, per
    level, plus what the GP-based fields are. Internal keys are stripped."""
    levels = [{k: v for k, v in lv.items() if not k.startswith("_")} for lv in calibration.get("levels", [])]
    sel_levels = {float(lv["coverage"]): lv for lv in (selection or {}).get("levels", [])}

    def used(cov: float) -> str:
        lv = sel_levels.get(cov)
        return METHOD_SELECTION_AWARE if lv is not None and lv.get("available") else METHOD_CONFORMAL

    methods = {used(float(lv["coverage"])) for lv in calibration.get("levels", [])}
    selection_block = None if selection is None else {
        **{k: v for k, v in selection.items() if not k.startswith("_") and k != "levels"},
        "levels": [{k: v for k, v in lv.items() if not k.startswith("_")} for lv in selection.get("levels", [])],
        "score": "abs(y - mu_oof) / sigma_oof over the selected held-out runs",
        "band": "pred +/- q_sel * max(predictive_sd, sigma_floor)",
        "selection": ("in every fold of every CV repeat, the ceil(top_fraction * n_fold) held-out runs the "
                      "fold model ranked best (highest prediction; lowest when minimizing)"),
        "coverage_guarantee": (
            "none: held-out top-ranked runs are a proxy for proposals and the pooled repeats are "
            "correlated; the coverage at proposals is measured in simulation and shipped with the band"
        ),
    }
    return {
        "method": METHOD_SELECTION_AWARE if methods == {METHOD_SELECTION_AWARE} else (
            METHOD_CONFORMAL if methods == {METHOD_CONFORMAL} else "mixed"),
        "method_by_level": {f"{float(lv['coverage']):g}": used(float(lv["coverage"]))
                            for lv in calibration.get("levels", [])},
        "selection_aware": selection_block,
        "calibrated_on": CALIBRATED_ON,
        # F02: what the coverage claim rests on, and what was measured where the
        # user acts (the proposals). Per level, same object each proposal carries.
        "coverage_basis": BASIS_HELD_OUT,
        "coverage_at_proposals": {
            f"{float(lv['coverage']):g}": proposal_caveat(used(float(lv["coverage"])), float(lv["coverage"]))
            for lv in calibration.get("levels", [])
        },
        "gp_coverage_at_proposals": {
            f"{float(lv['coverage']):g}": coverage_at_proposals_estimate(METHOD_GP, float(lv["coverage"]))
            for lv in calibration.get("levels", [])
        },
        "score": "abs(y - mu_oof) / sigma_oof",
        "quantile": "k-th smallest score, k = ceil((n + 1) * coverage)",
        "sigma": "gp_predictive_sd",
        "sigma_source": sigma_source,
        "n_calibration": calibration.get("n_calibration", 0),
        "coverage_guarantee": (
            "approximate: out-of-fold scores come from grouped-CV fold models and are not exactly "
            "exchangeable with a proposal from the full-data model"
        ),
        "empirical_oof_coverage_note": (
            "in-sample: counted on the same scores that set q_norm, so it is >= rank / n by construction; "
            "coverage_by_sigma_half shows whether the band adapts to the model's sd"
        ),
        "levels": levels,
        "gp_posterior": {
            "gp_intervals": "observation (latent + observation noise), one new measurement",
            "quantiles": "observation (latent + observation noise), one new measurement",
            "expected_improvement": "latent mean (no observation noise)",
            "p_improve": "latent mean (no observation noise)",
            "incumbent": ("the value the proposal annotation compares against (best single measurement, or "
                          "best replicate-averaged recipe mean when the replicate-aware surrogate proposed); "
                          "analytic EI is reported for ranking context, the batch itself was chosen by qLogNEI"),
            "note": GP_NOTE,
        },
    }


def conformal_band_caveat(alpha: float) -> dict:
    """`conformal_caveat` for the global +/- `conformal_q` band (F02; the
    arch-safe-now `coverage_not_guaranteed` field, now with the measured
    coverage at proposals and its source)."""
    level = 1.0 - float(alpha)
    est = coverage_at_proposals_estimate("conformal_abs_global", level)
    cav = proposal_caveat("conformal_abs_global", level)
    return {
        "code": "coverage_not_guaranteed",
        "coverage_basis": BASIS_HELD_OUT,
        "message": (
            "The +/- conformal_q band's coverage was measured on held-out past runs. It is not "
            "guaranteed for proposals, especially those far from the data or at the box edge. "
            + cav["message"].split(". ", 1)[1]
        ),
        "caveat_code": CAVEAT_CODE,
        "coverage_at_proposals_estimate": est,
    }


def band_from_summary(mean, sd, method_summary: Mapping[str, Any], level: float):
    """Rebuild the shipped per-proposal band at arbitrary points from the
    response's `interval_method` (benchmarks and tests: the band a proposal at
    those points would carry). Returns `(lo, hi)` arrays, or `(None, None)` when
    the level is unavailable."""
    mu = np.asarray(mean, float).reshape(-1)
    sp = np.asarray(sd, float).reshape(-1)
    sel = method_summary.get("selection_aware") or {}
    for lv in sel.get("levels", []):
        if abs(float(lv["coverage"]) - float(level)) < 1e-9 and lv.get("available"):
            half = float(lv["q_sel"]) * np.maximum(sp, float(sel["sigma_floor"]))
            return mu - half, mu + half
    for lv in method_summary.get("levels", []):
        if abs(float(lv["coverage"]) - float(level)) < 1e-9 and lv.get("available"):
            half = float(lv["q_norm"]) * sp
            return mu - half, mu + half
    return None, None


__all__ = [
    "METHOD_SELECTION_AWARE",
    "band_from_summary",
    "calibrate_normalized",
    "calibrate_selection_aware",
    "conformal_band_caveat", "expected_improvement", "interval_method_summary",
    "probability_of_improvement", "proposal_interval_fields", "proposal_predictive_std",
]
