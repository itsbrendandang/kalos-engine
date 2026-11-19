"""Split-conformal prediction intervals — distribution-free coverage.

Ported from the lean engine. Calibrate on a held-out set's absolute residuals to
get a half-width q such that [mu - q, mu + q] covers at least (1 - alpha) of
outcomes, with NO assumption about the error distribution (valid even when titer
residuals are skewed or heteroscedastic). Pairs naturally with the BoTorch
surrogate: pass `surrogate.posterior(X)[0]` (the mean) as the prediction.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def q_from_residuals(residuals, alpha: float = 0.1) -> float:
    """Conformal half-width q from precomputed absolute residuals.

    Same finite-sample quantile as `split_conformal`, but for when you already
    hold the held-out residuals (e.g. pooled out-of-fold residuals from grouped
    CV) instead of a calibration predict-fn. Coverage is exact for iid
    split-conformal; with grouped OOF residuals treat it as approximate.
    """
    res = np.abs(np.asarray(residuals, float).reshape(-1))
    n = len(res)
    if n == 0:
        raise ValueError("empty residual set")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(res, level))


def split_conformal(predict_fn: Callable[[np.ndarray], np.ndarray], X_cal, y_cal, alpha: float = 0.1) -> float:
    """Half-width q for >= 1-alpha coverage, from absolute residuals on a
    calibration set (held out from training to guarantee valid coverage)."""
    mu = np.asarray(predict_fn(X_cal), float).reshape(-1)
    return q_from_residuals(np.asarray(y_cal, float).reshape(-1) - mu, alpha)


def conformal_interval(mean: np.ndarray, q: float) -> np.ndarray:
    """Return an (n, 2) array of [lower, upper] = mean +/- q."""
    m = np.asarray(mean, float).reshape(-1)
    return np.column_stack([m - q, m + q])


# --- Normalized (studentized / locally-weighted) split conformal ----------------
#
# The plain band above is one half-width for every point. When the model's own
# uncertainty varies across the design (it always does for a GP: wide far from
# data, narrow near it), a single half-width over-covers where the model is sure
# and under-covers where it is not. Normalized conformal (Lei et al. 2018,
# "Distribution-free predictive inference for regression", section 5.2; Papadopoulos
# et al. 2008) scores each calibration point by its residual IN UNITS OF the
# model's own sd, takes the conformal quantile of those scores, and scales it back
# by the sd at the new point:
#
#     s_i = |y_i - mu_i| / sigma_i,   q = s_(k),   k = ceil((n + 1)(1 - alpha))
#     interval(x) = mu(x) +/- q * sigma(x)
#
# Marginal coverage >= 1 - alpha holds under exchangeability of calibration and
# test scores, for ANY positive sigma - sigma only shapes the band; a bad sigma
# makes it wide, never invalid. With grouped-CV out-of-fold scores the points are
# not exactly exchangeable with a new proposal (different fold models, and the
# proposal comes from the full-data model), so the guarantee is approximate.

# Relative floor below which a calibration sd is treated as degenerate. A score
# |y - mu| / sigma with sigma ~ 0 is either 0/0 or explodes, and either way it
# stops measuring anything about the model's calibration.
SIGMA_DEGENERATE_REL = 1e-6


def finite_sample_rank(n: int, alpha: float) -> int:
    """1-based order-statistic rank k = ceil((n + 1)(1 - alpha)) of the split
    conformal quantile. When k > n the quantile is +infinity: n calibration
    scores cannot certify 1 - alpha coverage."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    # The small epsilon guards ceil() against float noise such as
    # (9 + 1) * 0.9 = 9.000000000000002 -> 10.
    return int(np.ceil((n + 1) * (1.0 - alpha) - 1e-9))


def min_n_for_alpha(alpha: float) -> int:
    """Smallest calibration size n with finite_sample_rank(n, alpha) <= n."""
    n = 1
    while finite_sample_rank(n, alpha) > n:
        n += 1
    return n


def normalized_scores(actual, pred, std) -> dict:
    """Normalized nonconformity scores |y - mu| / sigma, or a structured refusal.

    Returns `{"ok": True, "scores": ndarray, "n": n}` or `{"ok": False, "code",
    "reason", "n"}`. Refuses (never silently drops points, which would bias the
    quantile toward the rows that happened to have a usable sd) when any sd is
    non-finite or non-positive, or when the smallest sd is below
    `SIGMA_DEGENERATE_REL` x the median sd.
    """
    a = np.asarray(actual, float).reshape(-1)
    m = np.asarray(pred, float).reshape(-1)
    s = np.asarray(std, float).reshape(-1)
    if not (a.shape == m.shape == s.shape):
        raise ValueError("actual, pred and std must be the same length")
    n = int(a.size)
    if n == 0:
        return {"ok": False, "n": 0, "code": "no_calibration_points",
                "reason": "no out-of-fold predictions to calibrate on"}
    if not (np.isfinite(a).all() and np.isfinite(m).all()):
        return {"ok": False, "n": n, "code": "nonfinite_oof",
                "reason": "an out-of-fold actual or prediction is not finite"}
    if not np.isfinite(s).all() or (s <= 0).any():
        return {"ok": False, "n": n, "code": "degenerate_sigma",
                "reason": "an out-of-fold posterior sd is non-finite or non-positive"}
    med = float(np.median(s))
    if float(s.min()) < SIGMA_DEGENERATE_REL * med:
        return {"ok": False, "n": n, "code": "degenerate_sigma",
                "reason": ("an out-of-fold posterior sd is ~0 relative to the others "
                           f"(min/median below {SIGMA_DEGENERATE_REL:g}), so its score is not meaningful")}
    return {"ok": True, "n": n, "scores": np.abs(a - m) / s}


def normalized_conformal_quantile(scores, alpha: float) -> dict:
    """Finite-sample split-conformal quantile of normalized scores.

    q = the k-th smallest score, k = ceil((n + 1)(1 - alpha)) (an order statistic,
    no interpolation - interpolating would shave the guarantee). Returns
    `{"available": True, "q", "rank", "n"}` or, when k > n (too few scores to
    certify the level), `{"available": False, "code": "too_few_calibration_points",
    "reason", "rank", "n", "min_n"}` - never an infinite or guessed q.
    """
    sc = np.sort(np.asarray(scores, float).reshape(-1))
    n = int(sc.size)
    k = finite_sample_rank(n, alpha)
    if n == 0 or k > n:
        need = min_n_for_alpha(alpha)
        return {
            "available": False, "q": None, "rank": k, "n": n, "min_n": need,
            "code": "too_few_calibration_points",
            "reason": (f"{n} held-out scores cannot certify {1 - alpha:.0%} coverage: the "
                       f"finite-sample rank ceil((n+1)(1-alpha)) = {k} exceeds n; "
                       f"at least {need} are needed"),
        }
    return {"available": True, "q": float(sc[k - 1]), "rank": k, "n": n}


__all__ = [
    "SIGMA_DEGENERATE_REL", "conformal_interval", "finite_sample_rank", "min_n_for_alpha",
    "normalized_conformal_quantile", "normalized_scores", "q_from_residuals", "split_conformal",
]
