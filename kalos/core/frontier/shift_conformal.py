"""Shift-weighted split conformal prediction and adaptive conformal control.

Pure NumPy / scikit-learn implementation (100% torch-free) for calibrated
prediction intervals under BO proposal covariate and selection shift:

1. ``ShiftWeightedConformal``: weighted split conformal prediction (Tibshirani
   et al. 2019) combining a regularized logistic density-ratio estimator
   ``w(x)`` (when ``X_cal`` and ``X_proposals`` are supplied) with an
   exponential predicted-rank selection tilt ``exp(beta * z_i)``. Evaluates
   the exact finite-sample weighted quantile with mass ``p_new`` at infinity
   and fails closed (``available=False``) when ``n_eff < min_n_eff`` or
   ``p_new > alpha``.
2. ``AdaptiveConformalState``: online Adaptive Conformal Inference (Gibbs &
   Candes 2021) state tracker adjusting nominal miscoverage ``alpha_t`` across
   sequential campaign rounds to maintain target ``1 - nominal_alpha`` coverage
   under distribution drift.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _validate_levels(levels: Sequence[float]) -> tuple[float, ...]:
    out: list[float] = []
    for lv in levels:
        cov = float(lv)
        if not math.isfinite(cov) or not (0.0 < cov < 1.0):
            raise ValueError(f"coverage level must lie strictly in (0, 1), got {lv!r}")
        out.append(cov)
    return tuple(out)


def _failure_calibration(
    *,
    code: str,
    reason: str,
    n_calibration: int,
    levels: Sequence[float],
    beta: float,
    w_min: float,
    w_max: float,
    min_n_eff: float,
    sigma_floor_quantile: float,
) -> dict[str, Any]:
    level_entries = [
        {
            "coverage": float(lv),
            "alpha": float(1.0 - float(lv)),
            "available": False,
            "code": code,
            "reason": reason,
            "n_calibration": int(n_calibration),
        }
        for lv in levels
    ]
    return {
        "ok": False,
        "available": False,
        "code": code,
        "reason": reason,
        "n_calibration": int(n_calibration),
        "n_eff": 0.0,
        "sigma_floor": None,
        "sigma_floor_quantile": float(sigma_floor_quantile),
        "beta": float(beta),
        "w_min": float(w_min),
        "w_max": float(w_max),
        "min_n_eff": float(min_n_eff),
        "pred_mean_ref": None,
        "pred_std_ref": None,
        "scores_sorted": np.empty(0, dtype=float),
        "weights_sorted": np.empty(0, dtype=float),
        "weight_sum": 0.0,
        "density_model": None,
        "levels": level_entries,
    }


def _weighted_quantile(
    scores_sorted: np.ndarray,
    weights_sorted: np.ndarray,
    weight_sum: float,
    w_new: float,
    alpha: float,
) -> tuple[float | None, float, bool, str | None, str | None]:
    """Exact Tibshirani et al. (2019) finite-sample weighted quantile."""
    tot = weight_sum + w_new
    if tot <= 0.0 or not math.isfinite(tot):
        return None, 1.0, False, "degenerate_weights", "total conformal weight is non-positive"
    p_new = float(w_new / tot)
    if p_new > alpha:
        return (
            None,
            p_new,
            False,
            "shift_too_large",
            (
                f"proposal weight mass p_new={p_new:.4f} exceeds alpha={alpha:.4f}; "
                "proposal is too shifted for finite weighted conformal coverage"
            ),
        )
    p_cal = weights_sorted / tot
    cum = np.cumsum(p_cal)
    target = (1.0 - alpha) - 1e-12
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(idx, len(scores_sorted) - 1)
    return float(scores_sorted[idx]), p_new, True, None, None


class ShiftWeightedConformal:
    """Weighted split conformal prediction under selection and covariate shift."""

    def __init__(
        self,
        beta: float = 1.0,
        w_min: float = 0.1,
        w_max: float = 10.0,
        min_n_eff: float = 8.0,
    ) -> None:
        if not math.isfinite(beta):
            raise ValueError(f"beta must be finite, got {beta!r}")
        if not math.isfinite(w_min) or w_min <= 0.0:
            raise ValueError(f"w_min must be finite and > 0, got {w_min!r}")
        if not math.isfinite(w_max) or w_max < w_min:
            raise ValueError(f"w_max must be finite and >= w_min ({w_min}), got {w_max!r}")
        if not math.isfinite(min_n_eff) or min_n_eff <= 0.0:
            raise ValueError(f"min_n_eff must be finite and > 0, got {min_n_eff!r}")
        self.beta = float(beta)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.min_n_eff = float(min_n_eff)

    def _tilt_weights(
        self, pred: np.ndarray, mu_ref: float, sd_ref: float
    ) -> np.ndarray:
        z = (pred - mu_ref) / sd_ref
        return np.clip(
            np.exp(np.clip(self.beta * z, -50.0, 50.0)),
            self.w_min,
            self.w_max,
        )

    def _fit_density_estimator(
        self, X_cal: Any, X_proposals: Any, n_expected: int
    ) -> dict[str, Any] | None:
        if X_cal is None or X_proposals is None:
            return None
        Xc = np.asarray(X_cal, dtype=float)
        Xp = np.asarray(X_proposals, dtype=float)
        if Xc.ndim == 1:
            Xc = Xc.reshape(-1, 1)
        if Xp.ndim == 1:
            Xp = Xp.reshape(1, -1)
        if Xc.ndim != 2 or Xp.ndim != 2:
            raise ValueError("X_cal and X_proposals must be 2-D arrays")
        if Xc.shape[0] != n_expected:
            raise ValueError(
                f"X_cal length {Xc.shape[0]} does not match calibration length {n_expected}"
            )
        if Xp.shape[0] < 1 or Xc.shape[0] < 8:
            return None
        if Xc.shape[1] != Xp.shape[1]:
            raise ValueError(
                f"X_cal feature width {Xc.shape[1]} != X_proposals width {Xp.shape[1]}"
            )
        if not (np.isfinite(Xc).all() and np.isfinite(Xp).all()):
            raise ValueError("X_cal and X_proposals must be finite")

        n_cal = int(Xc.shape[0])
        n_prop = int(Xp.shape[0])
        X_all = np.vstack([Xc, Xp])
        y_dom = np.concatenate([np.zeros(n_cal, dtype=int), np.ones(n_prop, dtype=int)])
        scaler = StandardScaler().fit(X_all)
        clf = LogisticRegression(C=0.5, max_iter=1000, random_state=0)
        clf.fit(scaler.transform(X_all), y_dom)
        return {
            "scaler": scaler,
            "clf": clf,
            "n_cal": n_cal,
            "n_prop": n_prop,
            "n_features": int(Xc.shape[1]),
        }

    def _density_weights(
        self, X: np.ndarray, density_model: dict[str, Any]
    ) -> np.ndarray:
        scaler: StandardScaler = density_model["scaler"]
        clf: LogisticRegression = density_model["clf"]
        ratio = float(density_model["n_cal"]) / float(density_model["n_prop"])
        p_prop = np.clip(
            clf.predict_proba(scaler.transform(X))[:, 1],
            1e-6,
            1.0 - 1e-6,
        )
        raw = ratio * (p_prop / (1.0 - p_prop))
        return np.clip(raw, self.w_min, self.w_max)

    def _combine_weights(
        self, w_tilt: np.ndarray, w_dens: np.ndarray | None
    ) -> np.ndarray:
        if w_dens is None:
            return w_tilt
        if self.beta == 0.0:
            return w_dens
        return np.clip(w_tilt * np.sqrt(w_dens), self.w_min, self.w_max)

    def calibrate(
        self,
        oof_actual: Any,
        oof_pred: Any,
        oof_std: Any,
        *,
        X_cal: Any = None,
        X_proposals: Any = None,
        levels: Sequence[float] = (0.90, 0.95),
        sigma_floor_quantile: float = 0.60,
    ) -> dict[str, Any]:
        """Calibrate weighted normalized nonconformity scores on OOF predictions."""
        checked_levels = _validate_levels(levels)
        q_floor = float(sigma_floor_quantile)
        if not math.isfinite(q_floor) or not (0.0 <= q_floor <= 1.0):
            raise ValueError(
                f"sigma_floor_quantile must be in [0, 1], got {sigma_floor_quantile!r}"
            )

        actual = np.asarray(oof_actual, dtype=float).reshape(-1)
        pred = np.asarray(oof_pred, dtype=float).reshape(-1)
        std = np.asarray(oof_std, dtype=float).reshape(-1)
        if not (actual.shape == pred.shape == std.shape):
            raise ValueError("oof_actual, oof_pred, and oof_std must have the same length")

        n = int(actual.size)

        def _fail(code: str, reason: str) -> dict[str, Any]:
            return _failure_calibration(
                code=code,
                reason=reason,
                n_calibration=n,
                levels=checked_levels,
                beta=self.beta,
                w_min=self.w_min,
                w_max=self.w_max,
                min_n_eff=self.min_n_eff,
                sigma_floor_quantile=q_floor,
            )

        if n == 0:
            return _fail("no_calibration_points", "no out-of-fold predictions to calibrate on")
        if not (np.isfinite(actual).all() and np.isfinite(pred).all()):
            return _fail("nonfinite_oof", "an out-of-fold actual or prediction is not finite")
        if not np.isfinite(std).all() or (std <= 0.0).any():
            return _fail(
                "degenerate_sigma",
                "an out-of-fold predictive sd is non-finite or non-positive",
            )

        sigma_floor = float(np.quantile(std, q_floor))
        if not math.isfinite(sigma_floor) or sigma_floor <= 0.0:
            return _fail("degenerate_sigma", "calibrated sigma floor is non-finite or non-positive")

        scores = np.abs(actual - pred) / std
        mu_ref = float(np.mean(pred))
        sd_ref = float(max(float(np.std(pred)), 1e-12))
        w_tilt = self._tilt_weights(pred, mu_ref, sd_ref)

        density_model = self._fit_density_estimator(X_cal, X_proposals, n)
        w_dens: np.ndarray | None = None
        if density_model is not None:
            Xc = np.asarray(X_cal, dtype=float)
            if Xc.ndim == 1:
                Xc = Xc.reshape(-1, 1)
            w_dens = self._density_weights(Xc, density_model)

        weights = self._combine_weights(w_tilt, w_dens)
        w_sum = float(np.sum(weights))
        w_sq_sum = float(np.sum(weights * weights))
        n_eff = float((w_sum * w_sum) / w_sq_sum) if w_sq_sum > 0.0 else 0.0

        order = np.argsort(scores, kind="mergesort")
        scores_sorted = scores[order]
        weights_sorted = weights[order]
        n_eff_ok = n_eff >= self.min_n_eff

        w_med = float(np.median(weights))
        level_entries: list[dict[str, Any]] = []
        for cov in checked_levels:
            alpha = 1.0 - cov
            if not n_eff_ok:
                level_entries.append({
                    "coverage": cov,
                    "alpha": alpha,
                    "available": False,
                    "code": "insufficient_n_eff",
                    "reason": f"effective sample size n_eff={n_eff:.2f} < min_n_eff={self.min_n_eff:.2f}",
                    "n_calibration": n,
                    "n_eff": n_eff,
                })
                continue
            q_ref, p_new_ref, avail, code, reason = _weighted_quantile(
                scores_sorted, weights_sorted, w_sum, w_med, alpha
            )
            entry: dict[str, Any] = {
                "coverage": cov,
                "alpha": alpha,
                "available": avail,
                "n_calibration": n,
                "n_eff": n_eff,
                "q_ref": q_ref,
                "p_new_ref": p_new_ref,
            }
            if not avail:
                entry["code"] = code
                entry["reason"] = reason
            level_entries.append(entry)

        return {
            "ok": True,
            "available": n_eff_ok,
            "code": None if n_eff_ok else "insufficient_n_eff",
            "reason": (
                None
                if n_eff_ok
                else f"effective sample size n_eff={n_eff:.2f} < min_n_eff={self.min_n_eff:.2f}"
            ),
            "n_calibration": n,
            "n_eff": n_eff,
            "sigma_floor": sigma_floor,
            "sigma_floor_quantile": q_floor,
            "beta": self.beta,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "min_n_eff": self.min_n_eff,
            "pred_mean_ref": mu_ref,
            "pred_std_ref": sd_ref,
            "scores_sorted": scores_sorted,
            "weights_sorted": weights_sorted,
            "weight_sum": w_sum,
            "density_model": density_model,
            "levels": level_entries,
        }

    def interval(
        self,
        pred_mean: Any,
        pred_std: Any,
        *,
        pred_X: Any = None,
        cal_result: dict[str, Any],
        coverage: float = 0.90,
    ) -> dict[str, Any]:
        """Compute shift-weighted conformal interval(s) for scalar or 1-D inputs."""
        cov = float(coverage)
        if not math.isfinite(cov) or not (0.0 < cov < 1.0):
            raise ValueError(f"coverage must lie strictly in (0, 1), got {coverage!r}")
        alpha = 1.0 - cov

        is_scalar = np.ndim(pred_mean) == 0 and np.ndim(pred_std) == 0
        mu_arr = np.asarray(pred_mean, dtype=float).reshape(-1)
        sd_arr = np.asarray(pred_std, dtype=float).reshape(-1)
        if mu_arr.size == 1 and sd_arr.size > 1:
            mu_arr = np.broadcast_to(mu_arr, sd_arr.shape).copy()
        elif sd_arr.size == 1 and mu_arr.size > 1:
            sd_arr = np.broadcast_to(sd_arr, mu_arr.shape).copy()
        elif mu_arr.shape != sd_arr.shape:
            raise ValueError("pred_mean and pred_std must have compatible shapes")

        m = int(mu_arr.size)
        if not cal_result.get("ok", False):
            code = str(cal_result.get("code") or "calibration_unavailable")
            reason = str(cal_result.get("reason") or "calibration failed")
            if is_scalar:
                return {
                    "available": False,
                    "coverage": cov,
                    "lo": None,
                    "hi": None,
                    "half_width": None,
                    "q": None,
                    "p_new": None,
                    "w_new": None,
                    "sigma_used": None,
                    "n_eff": float(cal_result.get("n_eff", 0.0)),
                    "code": code,
                    "reason": reason,
                }
            return {
                "available": np.zeros(m, dtype=bool),
                "all_available": False,
                "any_available": False,
                "coverage": cov,
                "lo": np.full(m, np.nan, dtype=float),
                "hi": np.full(m, np.nan, dtype=float),
                "half_width": np.full(m, np.nan, dtype=float),
                "q": np.full(m, np.nan, dtype=float),
                "p_new": np.full(m, np.nan, dtype=float),
                "w_new": np.full(m, np.nan, dtype=float),
                "sigma_used": np.full(m, np.nan, dtype=float),
                "n_eff": float(cal_result.get("n_eff", 0.0)),
                "codes": [code] * m,
            }

        sigma_floor = float(cal_result["sigma_floor"])
        mu_ref = float(cal_result["pred_mean_ref"])
        sd_ref = float(cal_result["pred_std_ref"])
        scores_sorted = np.asarray(cal_result["scores_sorted"], dtype=float)
        weights_sorted = np.asarray(cal_result["weights_sorted"], dtype=float)
        w_sum = float(cal_result["weight_sum"])
        n_eff = float(cal_result["n_eff"])
        density_model = cal_result.get("density_model")

        w_tilt = self._tilt_weights(mu_arr, mu_ref, sd_ref)
        w_dens: np.ndarray | None = None
        if pred_X is not None and density_model is not None:
            Xp = np.asarray(pred_X, dtype=float)
            n_feat = int(density_model["n_features"])
            if Xp.ndim == 1:
                Xp = Xp.reshape(1, -1) if Xp.size == n_feat else Xp.reshape(-1, n_feat)
            if Xp.shape != (m, n_feat):
                raise ValueError(f"pred_X must have shape ({m}, {n_feat}), got {Xp.shape}")
            if not np.isfinite(Xp).all():
                raise ValueError("pred_X must be finite")
            w_dens = self._density_weights(Xp, density_model)
        w_new_arr = self._combine_weights(w_tilt, w_dens)

        lo_arr = np.full(m, np.nan, dtype=float)
        hi_arr = np.full(m, np.nan, dtype=float)
        half_arr = np.full(m, np.nan, dtype=float)
        q_arr = np.full(m, np.nan, dtype=float)
        p_new_arr = np.full(m, np.nan, dtype=float)
        sig_arr = np.full(m, np.nan, dtype=float)
        avail_arr = np.zeros(m, dtype=bool)
        codes: list[str | None] = [None] * m
        reasons: list[str | None] = [None] * m

        for i in range(m):
            mu_i = float(mu_arr[i])
            sd_i = float(sd_arr[i])
            w_i = float(w_new_arr[i])
            if not math.isfinite(mu_i):
                codes[i] = "nonfinite_prediction"
                reasons[i] = "prediction mean is not finite"
                continue
            if not math.isfinite(sd_i) or sd_i < 0.0:
                codes[i] = "degenerate_proposal_sigma"
                reasons[i] = "predictive sd at proposal is non-finite or negative"
                continue
            sig_i = max(sd_i, sigma_floor)
            sig_arr[i] = sig_i
            p_new_i = float(w_i / (w_sum + w_i))
            p_new_arr[i] = p_new_i

            if n_eff < self.min_n_eff:
                codes[i] = "insufficient_n_eff"
                reasons[i] = (
                    f"effective sample size n_eff={n_eff:.2f} < min_n_eff={self.min_n_eff:.2f}"
                )
                continue

            q_i, p_new_i, avail_i, code_i, reason_i = _weighted_quantile(
                scores_sorted, weights_sorted, w_sum, w_i, alpha
            )
            p_new_arr[i] = p_new_i
            if not avail_i or q_i is None:
                codes[i] = code_i
                reasons[i] = reason_i
                continue

            half_i = q_i * sig_i
            q_arr[i] = q_i
            half_arr[i] = half_i
            lo_arr[i] = mu_i - half_i
            hi_arr[i] = mu_i + half_i
            avail_arr[i] = True

        if is_scalar:
            ok0 = bool(avail_arr[0])
            return {
                "available": ok0,
                "coverage": cov,
                "lo": float(lo_arr[0]) if ok0 else None,
                "hi": float(hi_arr[0]) if ok0 else None,
                "half_width": float(half_arr[0]) if ok0 else None,
                "q": float(q_arr[0]) if ok0 else None,
                "p_new": float(p_new_arr[0]) if math.isfinite(p_new_arr[0]) else None,
                "w_new": float(w_new_arr[0]),
                "sigma_used": float(sig_arr[0]) if math.isfinite(sig_arr[0]) else None,
                "n_eff": n_eff,
                "code": codes[0],
                "reason": reasons[0],
            }

        return {
            "available": avail_arr,
            "all_available": bool(avail_arr.all()),
            "any_available": bool(avail_arr.any()),
            "coverage": cov,
            "lo": lo_arr,
            "hi": hi_arr,
            "half_width": half_arr,
            "q": q_arr,
            "p_new": p_new_arr,
            "w_new": w_new_arr,
            "sigma_used": sig_arr,
            "n_eff": n_eff,
            "codes": codes,
        }


class AdaptiveConformalState:
    """Online Adaptive Conformal Inference (ACI) state across sequential rounds."""

    def __init__(
        self,
        nominal_alpha: float = 0.10,
        gamma: float = 0.05,
        min_alpha: float = 0.005,
        max_alpha: float = 0.50,
    ) -> None:
        if not math.isfinite(nominal_alpha) or not (0.0 < nominal_alpha < 1.0):
            raise ValueError(f"nominal_alpha must lie strictly in (0, 1), got {nominal_alpha!r}")
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError(f"gamma must be finite and >= 0, got {gamma!r}")
        if not math.isfinite(min_alpha) or not (0.0 < min_alpha < 1.0):
            raise ValueError(f"min_alpha must lie strictly in (0, 1), got {min_alpha!r}")
        if not math.isfinite(max_alpha) or not (min_alpha <= max_alpha < 1.0):
            raise ValueError(
                f"max_alpha must lie in [min_alpha, 1), got {max_alpha!r}"
            )
        self.nominal_alpha = float(nominal_alpha)
        self.gamma = float(gamma)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.current_alpha: float = float(np.clip(self.nominal_alpha, self.min_alpha, self.max_alpha))
        self.round_coverages: list[float] = []
        self.alpha_history: list[float] = [self.current_alpha]
        self._total_covered: int = 0
        self._total_count: int = 0

    @property
    def n_rounds(self) -> int:
        return len(self.round_coverages)

    @property
    def cumulative_coverage(self) -> float:
        if self._total_count == 0:
            return float("nan")
        return float(self._total_covered / self._total_count)

    def update(self, covered: Sequence[bool]) -> float:
        """Update ``current_alpha`` after observing a round's coverage indicators."""
        arr = np.asarray(list(covered), dtype=bool).reshape(-1)
        if arr.size == 0:
            return self.current_alpha
        cov_rate = float(np.mean(arr))
        mean_err = 1.0 - cov_rate
        self.round_coverages.append(cov_rate)
        self._total_covered += int(np.sum(arr))
        self._total_count += int(arr.size)
        self.current_alpha = float(
            np.clip(
                self.current_alpha + self.gamma * (self.nominal_alpha - mean_err),
                self.min_alpha,
                self.max_alpha,
            )
        )
        self.alpha_history.append(self.current_alpha)
        return self.current_alpha


__all__ = ["AdaptiveConformalState", "ShiftWeightedConformal"]
