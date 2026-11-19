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

Torch-free: the surrogate is only ever called through its `posterior` method.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr, ndtri

from kalos.core.conformal import normalized_conformal_quantile, normalized_scores

METHOD_CONFORMAL = "conformal_normalized"
CALIBRATED_ON = "grouped_cv_oof"
METHOD_GP = "gp_posterior"
GP_NOTE = "model-based, not held-out calibrated"
QUANTILE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

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
    rows: list[dict] = []
    for i in range(len(mu)):
        m, sp = float(mu[i]), float(pre[i])
        sp_ok = bool(np.isfinite(sp) and sp > 0)
        ivs: list[dict] = []
        for c in cal_levels:
            base = {"coverage": c["coverage"], "method": METHOD_CONFORMAL, "calibrated_on": CALIBRATED_ON,
                    "n_calibration": c["n_calibration"]}
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
        gp: list[dict] = []
        for lv in levels:
            cov = float(lv)
            entry: dict = {"coverage": _r(cov), "method": METHOD_GP, "posterior": "observation", "note": GP_NOTE}
            if np.isfinite(sp) and sp >= 0:
                half = float(ndtri(0.5 + cov / 2.0)) * sp
                entry.update(available=True, lo=_r(m - half), hi=_r(m + half))
            else:
                entry.update(available=False, lo=None, hi=None)
            gp.append(entry)
        quant: dict[str, Any] = {"basis": METHOD_GP, "posterior": "observation", "note": GP_NOTE}
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


def interval_method_summary(calibration: Mapping[str, Any], sigma_source: str) -> dict:
    """Top-level `interval_method`: how `proposals[].intervals` were built, per
    level, plus what the GP-based fields are. Internal keys are stripped."""
    levels = [{k: v for k, v in lv.items() if not k.startswith("_")} for lv in calibration.get("levels", [])]
    return {
        "method": METHOD_CONFORMAL,
        "calibrated_on": CALIBRATED_ON,
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


__all__ = [
    "calibrate_normalized", "expected_improvement", "interval_method_summary",
    "probability_of_improvement", "proposal_interval_fields", "proposal_predictive_std",
]
