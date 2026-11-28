"""Warped, Sparse, Robust, and Tobit-Censored Exact-Cholesky GP Surrogate.

Pure NumPy/SciPy float64 implementation with analytical gradients and L-BFGS-B
MAP optimization (zero torch/botorch/gpytorch imports). Designed for wet-lab
bioprocess physics:

1. Monotonic Kumaraswamy input warping per dimension (saturation/Hill curves).
2. Exact-Jacobian shifted Box-Cox output warping with 11-point Gauss-Hermite
   quadrature back-transformation.
3. Horseshoe / Cauchy sparsity prior on ARD inverse lengthscales.
4. Tobit / Laplace-Newton censored ``< LOD`` likelihood.
5. Exact closed-form Cholesky LOO-IRLS robust outlier down-weighting.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri

_SQRT5 = float(np.sqrt(5.0))
_LOG_2PI = float(np.log(2.0 * np.pi))
_INPUT_EPS = 1e-4
_JITTER_LADDER = (1e-8, 1e-6, 1e-4)

# Precompute 11-point physicist's Gauss-Hermite quadrature nodes/weights for Z ~ N(0, 1).
_GH_X, _GH_W = hermgauss(11)
_GH_Z = (np.sqrt(2.0) * _GH_X).astype(np.float64)
_GH_WEIGHTS = (_GH_W / np.sqrt(np.pi)).astype(np.float64)

# Parameter box bounds
_THETA_MIN = float(np.log(0.05))
_THETA_MAX = float(np.log(25.0))
_WARP_MIN = float(np.log(0.15))
_WARP_MAX = float(np.log(8.0))
_LOG_SF_BOUNDS = (-4.0, 3.0)
_LOG_SN_BOUNDS = (-5.5, 1.5)
_MEAN_BOUNDS = (-5.0, 5.0)
_LAMBDA_BOUNDS = (-1.5, 1.5)

# Prior scales
_SPARSITY_SCALE_S0 = 0.25
_SPARSITY_MULT = 5.0
_WARP_PRIOR_STD = 0.8
_LAMBDA_PRIOR_STD = 1.25


def _sanitize_bounds(bounds: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, ordered ``(lower, upper)`` rows with zero-width dims widened."""
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 2 or arr.shape[1] == 0:
        raise ValueError("bounds must have shape (2, d) with d >= 1")
    lower = arr[0].copy()
    upper = arr[1].copy()
    for row, partner in ((lower, upper), (upper, lower)):
        bad = ~np.isfinite(row)
        if bad.any():
            fill = np.where(np.isfinite(partner), partner, 0.0)
            row[bad] = fill[bad]
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    degenerate = (upper - lower) <= 0.0
    if degenerate.any():
        pad = 1e-6 * np.maximum(np.abs(lower), 1.0)
        upper = np.where(degenerate, lower + pad, upper)
    return lower, upper


def _normalize_inputs(
    X: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Normalize ``X`` linearly into ``[_INPUT_EPS, 1 - _INPUT_EPS]^d``."""
    u_raw = (X - lower[None, :]) / (upper[None, :] - lower[None, :])
    u_clip = np.clip(u_raw, 0.0, 1.0)
    return _INPUT_EPS + (1.0 - 2.0 * _INPUT_EPS) * u_clip


def _kumaraswamy_warp(
    u: np.ndarray, log_alpha: np.ndarray, log_beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Kumaraswamy CDF warp ``w = 1 - (1 - u^alpha)^beta`` and parameter derivs."""
    alpha = np.exp(log_alpha)[None, :]
    beta = np.exp(log_beta)[None, :]
    log_u = np.log(u)
    p = np.exp(np.clip(alpha * log_u, -50.0, 0.0))
    q = np.clip(1.0 - p, 1e-12, 1.0 - 1e-12)
    log_q = np.log(q)
    r = np.exp(np.clip(beta * log_q, -50.0, 0.0))
    w = 1.0 - r
    dw_dlog_a = beta * alpha * (r * p / q) * log_u
    dw_dlog_b = -beta * r * log_q
    return w, dw_dlog_a, dw_dlog_b


def _boxcox_forward(
    y_pos: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shifted Box-Cox transform, derivative wrt ``lam``, and ``log |g'(y)|``."""
    log_t = np.log(y_pos)
    if abs(lam) > 1e-5:
        lt = lam * log_t
        exp_lt = np.exp(np.clip(lt, -50.0, 50.0))
        g = (exp_lt - 1.0) / lam
        dg_dlam = (lt * exp_lt - (exp_lt - 1.0)) / (lam * lam)
    else:
        lt = lam * log_t
        g = log_t * (1.0 + 0.5 * lt + (lt * lt) / 6.0 + (lt * lt * lt) / 24.0)
        dg_dlam = (log_t * log_t) * (0.5 + lt / 3.0 + (lt * lt) / 8.0)
    log_deriv = (lam - 1.0) * log_t
    return g, dg_dlam, log_deriv


def _boxcox_inverse(g: np.ndarray, lam: float, max_y_pos: float) -> np.ndarray:
    """Numerically safe inverse Box-Cox mapping warped values back to positive scale."""
    if abs(lam) <= 1e-5:
        # First-order corrected exponential around lam=0
        val = np.clip(g, -40.0, 40.0)
        if abs(lam) > 0.0:
            # Solve (exp(lam*t)-1)/lam = g -> t = log(max(1+lam*g, eps))/lam
            base = np.maximum(1.0 + lam * val, 1e-12)
            return np.exp(np.log(base) / lam)
        return np.exp(val)
    if lam > 0.0:
        base = np.maximum(1.0 + lam * g, 1e-12)
        return np.exp(np.log(base) / lam)
    # lam < 0: upper bound at 1 + lam * g -> 0+; clamp safely above extreme positive cap
    min_base = float(np.exp(lam * np.log(max(20.0 * max_y_pos, 10.0))))
    base = np.maximum(1.0 + lam * g, min_base)
    return np.exp(np.log(base) / lam)


def _boxcox_inv_deriv(g: np.ndarray, lam: float, max_y_pos: float) -> np.ndarray:
    """Derivative ``d(g_inv)/dg`` for delta-method covariance fallback."""
    y_pos = _boxcox_inverse(g, lam, max_y_pos)
    return np.power(np.maximum(y_pos, 1e-12), 1.0 - lam)


def _matern52_kernel(
    w1: np.ndarray, w2: np.ndarray, log_sf: float, theta: np.ndarray
) -> np.ndarray:
    """Matérn-5/2 ARD kernel matrix between ``w1 (n1, d)`` and ``w2 (n2, d)``."""
    inv_l = np.exp(-theta)[None, :]
    z1 = w1 * inv_l
    z2 = w2 * inv_l
    n1_sq = np.sum(z1 * z1, axis=1)[:, None]
    n2_sq = np.sum(z2 * z2, axis=1)[None, :]
    r2 = np.maximum(n1_sq + n2_sq - 2.0 * (z1 @ z2.T), 0.0)
    d_mat = np.sqrt(r2)
    s = _SQRT5 * d_mat
    sf2 = float(np.exp(2.0 * log_sf))
    return sf2 * (1.0 + s + (s * s) / 3.0) * np.exp(-s)


def _matern52_gram_and_aux(
    w: np.ndarray, log_sf: float, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Matérn-5/2 Gram matrix ``K`` and derivative factor ``dK_dneg_r2``."""
    w_c = w - w.mean(axis=0, keepdims=True)
    rho = np.exp(-2.0 * theta)
    z = w_c * np.exp(-theta)[None, :]
    z_sq = np.sum(z * z, axis=1)
    r2 = np.maximum(z_sq[:, None] + z_sq[None, :] - 2.0 * (z @ z.T), 0.0)
    np.fill_diagonal(r2, 0.0)
    d_mat = np.sqrt(r2)
    s = _SQRT5 * d_mat
    exp_neg_s = np.exp(-s)
    sf2 = float(np.exp(2.0 * log_sf))
    K = sf2 * (1.0 + s + (s * s) / 3.0) * exp_neg_s
    # dK / d(r^2) = -(5/6) * sf2 * (1 + s) * exp(-s); return M_fac = (5/6)*sf2*(1+s)*exp(-s)
    m_fac = (5.0 / 6.0) * sf2 * (1.0 + s) * exp_neg_s
    return K, m_fac, rho


def _robust_cholesky(mat: np.ndarray) -> tuple[np.ndarray, float]:
    """Cholesky factor ``L`` with escalating diagonal jitter fallback."""
    n = mat.shape[0]
    try:
        L, _ = cho_factor(mat, lower=True, check_finite=False)
        return L, 0.0
    except np.linalg.LinAlgError:
        pass
    diag_idx = np.diag_indices(n)
    for j in _JITTER_LADDER:
        cand = mat.copy()
        cand[diag_idx] += j
        try:
            L, _ = cho_factor(cand, lower=True, check_finite=False)
            return L, j
        except np.linalg.LinAlgError:
            continue
    # Last resort: eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(0.5 * (mat + mat.T))
    floor = max(1e-4, float(np.max(eigvals)) * 1e-6)
    repaired = (eigvecs * np.maximum(eigvals, floor)[None, :]) @ eigvecs.T
    L, _ = cho_factor(0.5 * (repaired + repaired.T), lower=True, check_finite=False)
    return L, floor


def _tobit_log_lik_and_derivs(
    f_c: np.ndarray, lod_c: np.ndarray, sig_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(log_cdf, d1_f, neg_d2_f, d3_f)`` for ``log Phi((lod - f) / sig)``."""
    z = (lod_c - f_c) / sig_c
    log_cdf = log_ndtr(z)
    log_phi = -0.5 * _LOG_2PI - 0.5 * z * z
    # Inverse Mills ratio r(z) = phi(z) / Phi(z)
    r = np.empty_like(z)
    big_neg = z < -25.0
    if np.any(~big_neg):
        r[~big_neg] = np.exp(log_phi[~big_neg] - log_cdf[~big_neg])
    if np.any(big_neg):
        zb = z[big_neg]
        inv_z = 1.0 / zb
        inv_z2 = inv_z * inv_z
        # Asymptotic expansion: Phi(z) ~ phi(z) * (-1/z + 1/z^3 - 3/z^5)
        r[big_neg] = -zb / (1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2)

    # r(z)*(z + r(z)) is strictly in (0, 1)
    rz_sum = np.clip(r * (z + r), 1e-15, 1.0 - 1e-15)
    inv_sig = 1.0 / sig_c
    d1_f = -r * inv_sig
    w_c = rz_sum * (inv_sig * inv_sig)
    # d^3/df^3 log Phi((lod-f)/sig) = r * [(z+r)*(z+2r) - 1] / sig^3
    d3_f = r * ((z + r) * (z + 2.0 * r) - 1.0) * (inv_sig * inv_sig * inv_sig)
    return log_cdf, d1_f, w_c, d3_f


def _solve_tobit_laplace(
    K: np.ndarray,
    m: float,
    y_std: np.ndarray,
    lod_std: np.ndarray,
    censored_mask: np.ndarray,
    lik_var: np.ndarray,
    max_iter: int = 25,
) -> dict[str, Any]:
    """Damped Newton-Laplace solver for mixed Gaussian + Tobit ``< LOD`` observations."""
    n = K.shape[0]
    unc = ~censored_mask
    cens = censored_mask
    sig_c = np.sqrt(lik_var[cens])
    lod_c = lod_std[cens]
    y_u = y_std[unc]
    inv_var_u = 1.0 / lik_var[unc]

    # Warm-start f at prior mean (or exact GP fit on uncensored points if helpful)
    f = np.full(n, m, dtype=np.float64)
    a = np.zeros(n, dtype=np.float64)

    def _eval_site(f_vec: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        log_p = np.empty(n, dtype=np.float64)
        d1 = np.empty(n, dtype=np.float64)
        w = np.empty(n, dtype=np.float64)
        d3 = np.zeros(n, dtype=np.float64)
        if np.any(unc):
            res_u = y_u - f_vec[unc]
            log_p[unc] = -0.5 * (_LOG_2PI + np.log(lik_var[unc]) + res_u * res_u * inv_var_u)
            d1[unc] = res_u * inv_var_u
            w[unc] = inv_var_u
        if np.any(cens):
            lp_c, d1_c, w_c, d3_c = _tobit_log_lik_and_derivs(f_vec[cens], lod_c, sig_c)
            log_p[cens] = lp_c
            d1[cens] = d1_c
            w[cens] = w_c
            d3[cens] = d3_c
        psi = float(np.sum(log_p)) - 0.5 * float(np.dot(f_vec - m, a))
        return psi, d1, w, d3

    psi_cur, d1, w, d3 = _eval_site(f)
    L_B = np.eye(n, dtype=np.float64)
    S = np.sqrt(w)

    for _ in range(max_iter):
        S = np.sqrt(w)
        b = w * (f - m) + d1
        B = (S[:, None] * K) * S[None, :]
        B[np.diag_indices(n)] += 1.0
        L_B, _ = _robust_cholesky(B)
        Kb = K @ b
        solve_B = cho_solve((L_B, True), S * Kb, check_finite=False)
        a_new = b - S * solve_B
        f_new = m + K @ a_new

        delta_f = f_new - f
        if float(np.max(np.abs(delta_f))) < 1e-7:
            f = f_new
            a = a_new
            psi_cur, d1, w, d3 = _eval_site(f)
            break

        delta_a = a_new - a
        step = 1.0
        accepted = False
        for _ls in range(10):
            f_try = f + step * delta_f
            a_try = a + step * delta_a
            # Evaluate log-posterior at f_try using exact a_try = K^{-1}(f_try - m)
            log_p_try = 0.0
            if np.any(unc):
                res_u = y_u - f_try[unc]
                log_p_try += float(
                    np.sum(-0.5 * (_LOG_2PI + np.log(lik_var[unc]) + res_u * res_u * inv_var_u))
                )
            if np.any(cens):
                lp_c, _, _, _ = _tobit_log_lik_and_derivs(f_try[cens], lod_c, sig_c)
                log_p_try += float(np.sum(lp_c))
            psi_try = log_p_try - 0.5 * float(np.dot(f_try - m, a_try))
            if psi_try >= psi_cur - 1e-10:
                f = f_try
                a = a_try
                psi_cur, d1, w, d3 = _eval_site(f)
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    # Finalize B at converged f
    S = np.sqrt(w)
    B = (S[:, None] * K) * S[None, :]
    B[np.diag_indices(n)] += 1.0
    L_B, _ = _robust_cholesky(B)
    log_q = psi_cur - float(np.sum(np.log(np.diag(L_B))))
    tau = w
    nu = w * f + d1
    return {
        "f_hat": f,
        "a": a,
        "L_B": L_B,
        "S": S,
        "tau": tau,
        "nu": nu,
        "d3": d3,
        "log_q": log_q,
    }


class WarpedSparseGP:
    """First-principles Warped, Sparse, Robust, and Tobit-Censored Exact GP."""

    def __init__(
        self,
        use_input_warp: bool = True,
        use_output_warp: bool = True,
        sparsity_strength: float = 0.1,
        robust: bool = True,
    ) -> None:
        self.use_input_warp = bool(use_input_warp)
        self.use_output_warp = bool(use_output_warp)
        self.sparsity_strength = float(sparsity_strength)
        self.robust = bool(robust)

        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._bounds: np.ndarray | None = None
        self._lower: np.ndarray | None = None
        self._upper: np.ndarray | None = None
        self._censored_mask: np.ndarray | None = None
        self._lod: np.ndarray | None = None

        # Fitted model state
        self._m: float = 0.0
        self._log_sf: float = 0.0
        self._log_sn: float = -2.0
        self._lam: float = 1.0
        self._theta: np.ndarray | None = None
        self._warp_a: np.ndarray | None = None
        self._warp_b: np.ndarray | None = None
        self._y_shift: float = 0.0
        self._max_y_pos: float = 1.0
        self._g_mean: float = 0.0
        self._g_std: float = 1.0
        self._w_train: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._L_chol: np.ndarray | None = None
        self._L_B: np.ndarray | None = None
        self._S_laplace: np.ndarray | None = None
        self._is_laplace: bool = False
        self._fixed_noise: bool = False
        self._active_input_warp: bool = False

        self.fit_diagnostics: dict[str, Any] = {}

    def _transform_y(
        self, y: np.ndarray, lod: np.ndarray, uncensored: np.ndarray, lam: float
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        float,
    ]:
        """Forward shifted Box-Cox + standardization on uncensored data."""
        y_pos = y + self._y_shift
        lod_pos = lod + self._y_shift
        g_y, dg_y, log_deriv_y = _boxcox_forward(y_pos, lam)
        g_lod, dg_lod, _ = _boxcox_forward(lod_pos, lam)

        ref = g_y[uncensored] if int(np.sum(uncensored)) >= 2 else g_lod
        dref = dg_y[uncensored] if int(np.sum(uncensored)) >= 2 else dg_lod
        g_mean = float(np.mean(ref))
        dg_mean = float(np.mean(dref))
        diff = ref - g_mean
        var_g = float(np.mean(diff * diff))
        g_std = float(np.sqrt(var_g + 1e-12))
        dg_std = float(np.mean(diff * (dref - dg_mean)) / g_std)

        y_std = (g_y - g_mean) / g_std
        lod_std = (g_lod - g_mean) / g_std
        dy_std = ((dg_y - dg_mean) - y_std * dg_std) / g_std
        dlod_std = ((dg_lod - dg_mean) - lod_std * dg_std) / g_std
        return y_std, lod_std, dy_std, dlod_std, log_deriv_y, g_mean, g_std, dg_mean, dg_std

    def _prior_and_grad(
        self,
        m: float,
        log_sf: float,
        log_sn: float,
        lam: float,
        theta: np.ndarray,
        warp_a: np.ndarray,
        warp_b: np.ndarray,
        *,
        include_sn: bool,
        include_lam: bool,
        n_unc: int = 35,
        active_input_warp: bool | None = None,
    ) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate MAP log-priors and exact gradients."""
        log_prior = 0.0
        # Mean m ~ N(0, 4)
        log_prior += -0.5 * (m * m) / 4.0
        dm = -m / 4.0

        # log_sf ~ N(0, 2.25)
        log_prior += -0.5 * (log_sf * log_sf) / 2.25
        dlog_sf = -log_sf / 2.25

        dlog_sn = 0.0
        if include_sn:
            # log_sn ~ N(-2.0, 4.0)
            log_prior += -0.5 * ((log_sn + 2.0) ** 2) / 4.0
            dlog_sn = -(log_sn + 2.0) / 4.0

        dlam = 0.0
        if include_lam:
            s2 = _LAMBDA_PRIOR_STD * _LAMBDA_PRIOR_STD
            log_prior += -0.5 * ((lam - 1.0) ** 2) / s2
            dlam = -(lam - 1.0) / s2

        d = int(theta.shape[0])
        dtheta = np.zeros_like(theta)
        # Underdetermined high-D (d > 6, n_unc < 2*d): Hvarfner-style LogNormal prior
        # prevents MAP collapse onto a single spurious noise dimension.
        if d > 6 and n_unc < 2 * d:
            mu_d = float(np.log(np.clip(0.35 * np.sqrt(float(d)), 0.5, 4.0)))
            v_th = 1.0
            log_prior += -0.5 * float(np.sum((theta - mu_d) ** 2)) / v_th
            dtheta = dtheta - (theta - mu_d) / v_th

        # Horseshoe / Cauchy sparsity prior activates once n_unc > d (identifiable ARD).
        if self.sparsity_strength > 0.0 and n_unc > d:
            dim_factor = min(1.0, max(0.0, (d - 4.0) / 12.0))
            n_factor = min(1.0, float(n_unc - d) / max(float(d), 10.0))
            gamma = _SPARSITY_MULT * self.sparsity_strength * dim_factor * n_factor
            if gamma > 0.0:
                inv_s02 = 1.0 / (_SPARSITY_SCALE_S0 * _SPARSITY_SCALE_S0)
                ratio = np.exp(-2.0 * theta) * inv_s02
                log_prior += -gamma * float(np.sum(np.log1p(ratio)))
                dtheta = dtheta + 2.0 * gamma * (ratio / (1.0 + ratio))

        use_warp = self.use_input_warp if active_input_warp is None else bool(active_input_warp)
        dw_a = np.zeros_like(warp_a)
        dw_b = np.zeros_like(warp_b)
        if use_warp:
            ws2 = _WARP_PRIOR_STD * _WARP_PRIOR_STD
            log_prior += -0.5 * float(np.sum(warp_a * warp_a + warp_b * warp_b)) / ws2
            dw_a = -warp_a / ws2
            dw_b = -warp_b / ws2

        return log_prior, dm, dlog_sf, dlog_sn, dlam, dtheta, dw_a, dw_b

    def _fit_single_stage(
        self,
        u: np.ndarray,
        y: np.ndarray,
        lod: np.ndarray,
        censored: np.ndarray,
        noise_arr: np.ndarray | None,
        outlier_var: np.ndarray,
        *,
        seed: int,
        init_lam: float | None = None,
        warm_params: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run one multi-start L-BFGS-B optimization for given site outlier variances."""
        n, d = u.shape
        unc = ~censored
        has_censored = bool(np.any(censored))
        n_unc = int(np.sum(unc))
        fixed_noise = noise_arr is not None
        active_input_warp = self.use_input_warp and (n_unc >= max(16, 2 * d))

        # Parameter layout:
        # [m, log_sf, (log_sn if not fixed), (lam if use_output_warp), theta(d), (a(d), b(d) if warp)]
        idx_m = 0
        idx_sf = 1
        cur = 2
        idx_sn = cur if not fixed_noise else -1
        if not fixed_noise:
            cur += 1
        idx_lam = cur if self.use_output_warp else -1
        if self.use_output_warp:
            cur += 1
        idx_theta0 = cur
        cur += d
        idx_a0 = cur if active_input_warp else -1
        idx_b0 = (cur + d) if active_input_warp else -1
        if active_input_warp:
            cur += 2 * d
        n_params = cur

        bounds_list: list[tuple[float, float]] = [_MEAN_BOUNDS, _LOG_SF_BOUNDS]
        if not fixed_noise:
            bounds_list.append(_LOG_SN_BOUNDS)
        if self.use_output_warp:
            bounds_list.append(_LAMBDA_BOUNDS)
        bounds_list.extend([(_THETA_MIN, _THETA_MAX)] * d)
        if active_input_warp:
            bounds_list.extend([(_WARP_MIN, _WARP_MAX)] * (2 * d))

        def _unpack(p: np.ndarray) -> tuple[
            float, float, float, float, np.ndarray, np.ndarray, np.ndarray
        ]:
            m = float(p[idx_m])
            log_sf = float(p[idx_sf])
            log_sn = float(p[idx_sn]) if not fixed_noise else -10.0
            lam = float(p[idx_lam]) if self.use_output_warp else 1.0
            theta = p[idx_theta0 : idx_theta0 + d]
            if active_input_warp:
                warp_a = p[idx_a0 : idx_a0 + d]
                warp_b = p[idx_b0 : idx_b0 + d]
            else:
                warp_a = np.zeros(d, dtype=np.float64)
                warp_b = np.zeros(d, dtype=np.float64)
            return m, log_sf, log_sn, lam, theta, warp_a, warp_b

        def _neg_log_posterior_and_grad(p: np.ndarray) -> tuple[float, np.ndarray]:
            m, log_sf, log_sn, lam, theta, warp_a, warp_b = _unpack(p)

            if active_input_warp:
                w, dw_da, dw_db = _kumaraswamy_warp(u, warp_a, warp_b)
            else:
                w = u
                dw_da = np.empty(0, dtype=np.float64)
                dw_db = np.empty(0, dtype=np.float64)

            if self.use_output_warp:
                (
                    y_std,
                    lod_std,
                    dy_std,
                    _dlod_std,
                    log_deriv_y,
                    _,
                    g_std,
                    _,
                    dg_std,
                ) = self._transform_y(y, lod, unc, lam)
            else:
                y_std = (y - self._g_mean) / self._g_std
                lod_std = (lod - self._g_mean) / self._g_std
                dy_std = np.zeros(n, dtype=np.float64)
                log_deriv_y = np.zeros(n, dtype=np.float64)
                g_std = self._g_std
                dg_std = 0.0

            # Noise / site variance in standardized space
            base_var = np.zeros(n, dtype=np.float64)
            if fixed_noise and noise_arr is not None:
                # Scale original-unit noise variance through Box-Cox derivative and g_std
                deriv_y = np.power(np.maximum(y + self._y_shift, 1e-12), lam - 1.0)
                deriv_lod = np.power(np.maximum(lod + self._y_shift, 1e-12), lam - 1.0)
                deriv_eff = np.where(unc, deriv_y, deriv_lod)
                base_var = noise_arr * (deriv_eff / g_std) ** 2
            else:
                base_var[:] = float(np.exp(2.0 * log_sn))
            lik_var = np.maximum(base_var + outlier_var, 1e-6)

            K, m_fac, rho = _matern52_gram_and_aux(w, log_sf, theta)

            (
                log_prior,
                p_dm,
                p_dlog_sf,
                p_dlog_sn,
                p_dlam,
                p_dtheta,
                p_dwa,
                p_dwb,
            ) = self._prior_and_grad(
                m,
                log_sf,
                log_sn,
                lam,
                theta,
                warp_a,
                warp_b,
                include_sn=not fixed_noise,
                include_lam=self.use_output_warp,
                n_unc=n_unc,
                active_input_warp=active_input_warp,
            )

            # Jacobian of y -> standardized g(y) on uncensored observations
            jac_val = 0.0
            jac_dlam = 0.0
            if self.use_output_warp and n_unc > 0:
                log_pos_u = np.log(np.maximum(y[unc] + self._y_shift, 1e-12))
                jac_val = float(np.sum(log_deriv_y[unc])) - n_unc * float(np.log(g_std))
                jac_dlam = float(np.sum(log_pos_u)) - n_unc * (dg_std / g_std)

            if not has_censored:
                # Fast Exact-Cholesky Gaussian MLL + analytical gradients
                Ky = K.copy()
                Ky[np.diag_indices(n)] += lik_var
                L, _ = _robust_cholesky(Ky)
                r = y_std - m
                alpha = cho_solve((L, True), r, check_finite=False)
                mll = (
                    -0.5 * float(np.dot(r, alpha))
                    - float(np.sum(np.log(np.diag(L))))
                    - 0.5 * n * _LOG_2PI
                )

                L_inv = solve_triangular(L, np.eye(n, dtype=np.float64), lower=True, check_finite=False)
                Ky_inv = L_inv.T @ L_inv
                # W_mat = 0.5 * (alpha alpha^T - Ky_inv)
                W_mat = 0.5 * (np.outer(alpha, alpha) - Ky_inv)

                dm = float(np.sum(alpha))
                dlog_sf = 2.0 * float(np.sum(W_mat * K))

                dlog_sn = 0.0
                if not fixed_noise:
                    sn2 = float(np.exp(2.0 * log_sn))
                    dlog_sn = 2.0 * sn2 * float(np.sum(np.diag(W_mat)))

                dlam = 0.0
                if self.use_output_warp:
                    dlam = -float(np.dot(alpha, dy_std)) + jac_dlam
                    if fixed_noise and noise_arr is not None:
                        deriv_y = np.power(np.maximum(y + self._y_shift, 1e-12), lam - 1.0)
                        log_pos_y = np.log(np.maximum(y + self._y_shift, 1e-12))
                        dvar_dlam = (
                            2.0
                            * base_var
                            * (log_pos_y - dg_std / g_std)
                        )
                        dlam += float(np.sum(np.diag(W_mat) * dvar_dlam))

                # M = 2 * m_fac * W_mat (since dK/d(-r^2) = m_fac, and dK/d(theta_j) = 2*m_fac*rho_j*diff^2)
                M = m_fac * W_mat
                row_sum = np.sum(M, axis=1)
                Mw_diff = row_sum[:, None] * w - (M @ w)
                dtheta = 2.0 * rho * np.sum(w * Mw_diff, axis=0)
                if active_input_warp:
                    dw = -2.0 * rho[None, :] * Mw_diff
                    dwa = np.sum(dw * dw_da, axis=0)
                    dwb = np.sum(dw * dw_db, axis=0)
            else:
                # Tobit Laplace-Newton MLL + analytical explicit & implicit gradients
                lap = _solve_tobit_laplace(K, m, y_std, lod_std, censored, lik_var)
                mll = lap["log_q"]
                a = lap["a"]
                L_B = lap["L_B"]
                S = lap["S"]
                d3 = lap["d3"]

                # B^{-1} via Cholesky
                L_B_inv = solve_triangular(
                    L_B, np.eye(n, dtype=np.float64), lower=True, check_finite=False
                )
                B_inv = L_B_inv.T @ L_B_inv
                # R = S * B_inv * S = (K + W^{-1})^{-1}
                R = (S[:, None] * B_inv) * S[None, :]
                # Posterior covariance diag: diag(K) - diag(K R K)
                C_mat = L_B_inv @ (S[:, None] * K)
                post_var_diag = np.maximum(np.diag(K) - np.sum(C_mat * C_mat, axis=0), 0.0)
                s2 = 0.5 * post_var_diag * d3

                # Implicit adjoint b_imp = (I + W K)^{-1} s2 = s2 - R (K s2)
                b_imp = s2 - R @ (K @ s2)
                # Combined symmetric weight matrix for K derivatives:
                # 0.5 * (a a^T - R) + 0.5 * (b_imp a^T + a b_imp^T)
                W_mat = 0.5 * (
                    np.outer(a, a) - R + np.outer(b_imp, a) + np.outer(a, b_imp)
                )

                # Gradient wrt prior mean m:
                # dm = a^T 1 + s2^T (I + K W)^{-1} 1 = sum(a) + sum(b_imp)
                dm = float(np.sum(a)) + float(np.sum(b_imp))
                dlog_sf = 2.0 * float(np.sum(W_mat * K))

                M = m_fac * W_mat
                row_sum = np.sum(M, axis=1)
                Mw_diff = row_sum[:, None] * w - (M @ w)
                dtheta = 2.0 * rho * np.sum(w * Mw_diff, axis=0)
                if active_input_warp:
                    dw = -2.0 * rho[None, :] * Mw_diff
                    dwa = np.sum(dw * dw_da, axis=0)
                    dwb = np.sum(dw * dw_db, axis=0)

                # For likelihood-only scalars (log_sn, lam), use fast central FD on
                # the Laplace objective holding K fixed (only 2 scalars -> negligible cost)
                f_hat = lap["f_hat"]
                dlog_sn = 0.0
                if not fixed_noise:
                    eps_fd = 1e-5
                    lp_vals = []
                    for sign in (1.0, -1.0):
                        sn2_p = float(np.exp(2.0 * (log_sn + sign * eps_fd)))
                        lv_p = np.maximum(sn2_p + outlier_var, 1e-6)
                        lap_p = _solve_tobit_laplace(K, m, y_std, lod_std, censored, lv_p)
                        lp_vals.append(lap_p["log_q"])
                    dlog_sn = (lp_vals[0] - lp_vals[1]) / (2.0 * eps_fd)

                dlam = 0.0
                if self.use_output_warp:
                    eps_fd = 1e-5
                    lp_vals = []
                    for sign in (1.0, -1.0):
                        lam_p = float(np.clip(lam + sign * eps_fd, _LAMBDA_BOUNDS[0], _LAMBDA_BOUNDS[1]))
                        y_p, lod_p, _, _, ldy_p, _, gs_p, _, _ = self._transform_y(
                            y, lod, unc, lam_p
                        )
                        if fixed_noise and noise_arr is not None:
                            dy_e = np.where(
                                unc,
                                np.power(np.maximum(y + self._y_shift, 1e-12), lam_p - 1.0),
                                np.power(np.maximum(lod + self._y_shift, 1e-12), lam_p - 1.0),
                            )
                            bv_p = noise_arr * (dy_e / gs_p) ** 2
                        else:
                            bv_p = base_var
                        lv_p = np.maximum(bv_p + outlier_var, 1e-6)
                        lap_p = _solve_tobit_laplace(K, m, y_p, lod_p, censored, lv_p)
                        jv_p = (
                            float(np.sum(ldy_p[unc])) - n_unc * float(np.log(gs_p))
                            if n_unc > 0
                            else 0.0
                        )
                        lp_vals.append(lap_p["log_q"] + jv_p)
                    denom_fd = (
                        min(lam + eps_fd, _LAMBDA_BOUNDS[1])
                        - max(lam - eps_fd, _LAMBDA_BOUNDS[0])
                    )
                    dlam = (lp_vals[0] - lp_vals[1]) / max(denom_fd, 1e-12)
                _ = f_hat

            total_log_post = mll + jac_val + log_prior
            grad = np.zeros(n_params, dtype=np.float64)
            grad[idx_m] = -(dm + p_dm)
            grad[idx_sf] = -(dlog_sf + p_dlog_sf)
            if not fixed_noise:
                grad[idx_sn] = -(dlog_sn + p_dlog_sn)
            if self.use_output_warp:
                grad[idx_lam] = -(dlam + p_dlam)
            grad[idx_theta0 : idx_theta0 + d] = -(dtheta + p_dtheta)
            if active_input_warp:
                grad[idx_a0 : idx_a0 + d] = -(dwa + p_dwa)
                grad[idx_b0 : idx_b0 + d] = -(dwb + p_dwb)
            return -total_log_post, grad

        # Build deterministic start(s)
        init_theta = float(np.log(np.clip(0.35 * np.sqrt(d), 0.5, 5.0)))
        starts: list[np.ndarray] = []
        p0 = np.zeros(n_params, dtype=np.float64)
        p0[idx_m] = 0.0
        p0[idx_sf] = 0.0
        if not fixed_noise:
            p0[idx_sn] = -1.5
        if self.use_output_warp:
            p0[idx_lam] = 1.0 if init_lam is None else float(init_lam)
        p0[idx_theta0 : idx_theta0 + d] = init_theta
        starts.append(p0)

        # Second start when output warp is enabled: try log start (lam=0.0) as well
        if self.use_output_warp and init_lam is None and warm_params is None:
            p1 = p0.copy()
            p1[idx_lam] = 0.0
            starts.append(p1)

        # Short-lengthscale, low-noise start so sharp cliffs/peaks escape flat optima
        if warm_params is None:
            p_short = p0.copy()
            p_short[idx_theta0 : idx_theta0 + d] = float(np.log(0.6))
            if not fixed_noise:
                p_short[idx_sn] = -2.5
            starts.append(p_short)

        if warm_params is not None:
            starts.append(warm_params.copy())

        rng = np.random.default_rng(seed)
        best_res = None
        best_val = np.inf
        for s_idx, p_init in enumerate(starts):
            p_start = p_init.copy()
            if s_idx > 0 and warm_params is None:
                p_start[idx_theta0 : idx_theta0 + d] += rng.normal(0.0, 0.15, size=d)
                for idx_b, (lo, hi) in enumerate(bounds_list):
                    p_start[idx_b] = float(np.clip(p_start[idx_b], lo, hi))
            res = minimize(
                _neg_log_posterior_and_grad,
                p_start,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds_list,
                options={"maxiter": 120, "ftol": 1e-9, "gtol": 1e-6},
            )
            if float(res.fun) < best_val:
                best_val = float(res.fun)
                best_res = res

        assert best_res is not None
        p_opt = best_res.x
        m, log_sf, log_sn, lam, theta, warp_a, warp_b = _unpack(p_opt)

        # Recompute final exact matrices at p_opt
        w_opt = _kumaraswamy_warp(u, warp_a, warp_b)[0] if active_input_warp else u
        if self.use_output_warp:
            y_std, lod_std, _, _, log_deriv_y, g_mean, g_std, _, _ = self._transform_y(
                y, lod, unc, lam
            )
            jac_val = (
                float(np.sum(log_deriv_y[unc])) - n_unc * float(np.log(g_std))
                if n_unc > 0
                else 0.0
            )
        else:
            g_mean = self._g_mean
            g_std = self._g_std
            y_std = (y - g_mean) / g_std
            lod_std = (lod - g_mean) / g_std
            jac_val = 0.0

        base_var = np.zeros(n, dtype=np.float64)
        if fixed_noise and noise_arr is not None:
            deriv_eff = np.where(
                unc,
                np.power(np.maximum(y + self._y_shift, 1e-12), lam - 1.0),
                np.power(np.maximum(lod + self._y_shift, 1e-12), lam - 1.0),
            )
            base_var = noise_arr * (deriv_eff / g_std) ** 2
        else:
            base_var[:] = float(np.exp(2.0 * log_sn))
        lik_var = np.maximum(base_var + outlier_var, 1e-6)
        K, _, _ = _matern52_gram_and_aux(w_opt, log_sf, theta)

        out: dict[str, Any] = {
            "p_opt": p_opt,
            "m": m,
            "log_sf": log_sf,
            "log_sn": log_sn,
            "lam": lam,
            "theta": theta,
            "warp_a": warp_a,
            "warp_b": warp_b,
            "w_train": w_opt,
            "y_std": y_std,
            "lod_std": lod_std,
            "g_mean": g_mean,
            "g_std": g_std,
            "base_var": base_var,
            "lik_var": lik_var,
            "K": K,
            "active_input_warp": active_input_warp,
        }

        if not has_censored:
            Ky = K.copy()
            Ky[np.diag_indices(n)] += lik_var
            L, _ = _robust_cholesky(Ky)
            r = y_std - m
            alpha = cho_solve((L, True), r, check_finite=False)
            mll = (
                -0.5 * float(np.dot(r, alpha))
                - float(np.sum(np.log(np.diag(L))))
                - 0.5 * n * _LOG_2PI
                + jac_val
            )
            out.update(
                {
                    "is_laplace": False,
                    "alpha": alpha,
                    "L_chol": L,
                    "L_B": None,
                    "S_laplace": None,
                    "mll": mll,
                }
            )
        else:
            lap = _solve_tobit_laplace(K, m, y_std, lod_std, censored, lik_var)
            out.update(
                {
                    "is_laplace": True,
                    "alpha": lap["a"],
                    "L_chol": None,
                    "L_B": lap["L_B"],
                    "S_laplace": lap["S"],
                    "f_hat": lap["f_hat"],
                    "tau": lap["tau"],
                    "nu": lap["nu"],
                    "mll": lap["log_q"] + jac_val,
                }
            )
        return out

    def _compute_loo_residuals(
        self, fit_state: dict[str, Any], censored: np.ndarray, outlier_var: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact closed-form Cholesky LOO residuals and base LOO stds."""
        n = fit_state["K"].shape[0]
        unc = ~censored
        e = np.zeros(n, dtype=np.float64)
        sig_loo_base = np.ones(n, dtype=np.float64)

        if not fit_state["is_laplace"]:
            L = fit_state["L_chol"]
            alpha = fit_state["alpha"]
            L_inv = solve_triangular(
                L, np.eye(n, dtype=np.float64), lower=True, check_finite=False
            )
            k_inv_diag = np.maximum(np.sum(L_inv * L_inv, axis=0), 1e-12)
            e = alpha / k_inv_diag
            v_loo = 1.0 / k_inv_diag
            v_base = np.maximum(v_loo - outlier_var, 1e-8)
            sig_loo_base = np.sqrt(v_base)
        else:
            if np.any(unc):
                K = fit_state["K"]
                L_B = fit_state["L_B"]
                S = fit_state["S_laplace"]
                f_hat = fit_state["f_hat"]
                y_std = fit_state["y_std"]
                tau = fit_state["tau"]
                L_B_inv = solve_triangular(
                    L_B, np.eye(n, dtype=np.float64), lower=True, check_finite=False
                )
                C_mat = L_B_inv @ (S[:, None] * K)
                post_var = np.maximum(np.diag(K) - np.sum(C_mat * C_mat, axis=0), 1e-12)
                denom = np.maximum(1.0 - tau[unc] * post_var[unc], 1e-6)
                e[unc] = (y_std[unc] - f_hat[unc]) / denom
                v_loo_u = 1.0 / np.maximum(tau[unc] * denom, 1e-12)
                v_base_u = np.maximum(v_loo_u - outlier_var[unc], 1e-8)
                sig_loo_base[unc] = np.sqrt(v_base_u)
        return e, sig_loo_base

    def fit(
        self,
        X: Any,
        y: Any,
        bounds: Any,
        *,
        noise: Any = None,
        censored_mask: Any = None,
        lod: Any = None,
        seed: int = 0,
    ) -> WarpedSparseGP:
        """Fit the WarpedSparseGP surrogate on ``(X, y)`` within ``bounds``."""
        t0 = time.perf_counter()
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1)
        if Xa.ndim != 2 or Xa.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array (n x d)")
        n, d = Xa.shape
        if ya.shape[0] != n:
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(Xa).all():
            raise ValueError("X must be finite (no NaN/inf)")

        lower, upper = _sanitize_bounds(bounds)
        if lower.shape[0] != d:
            raise ValueError(f"bounds width {lower.shape[0]} does not match X width {d}")

        if censored_mask is None:
            censored = np.zeros(n, dtype=bool)
        else:
            censored = np.asarray(censored_mask, dtype=bool).reshape(-1)
            if censored.shape[0] != n:
                raise ValueError("censored_mask must have length n")

        unc = ~censored
        if not np.isfinite(ya[unc]).all():
            raise ValueError("Uncensored y values must be finite (no NaN/inf)")

        if lod is None:
            lod_arr = np.where(np.isfinite(ya), ya, 0.0)
        else:
            lod_arr = np.broadcast_to(np.asarray(lod, dtype=np.float64), (n,)).copy()
        if np.any(censored) and not np.isfinite(lod_arr[censored]).all():
            raise ValueError("lod must be finite at all censored rows")
        # Ensure y_clean has finite values even at censored rows (for storage/shift safety)
        y_clean = np.where(unc, ya, lod_arr)

        noise_arr: np.ndarray | None = None
        if noise is not None:
            noise_arr = np.broadcast_to(np.asarray(noise, dtype=np.float64), (n,)).copy()
            if not np.isfinite(noise_arr).all() or np.any(noise_arr < 0.0):
                raise ValueError("noise must be finite and non-negative")

        # Determine positive shift for Box-Cox
        vals_pos_check = y_clean[unc] if np.any(unc) else lod_arr
        if np.any(censored):
            vals_pos_check = np.concatenate([vals_pos_check, lod_arr[censored]])
        min_val = float(np.min(vals_pos_check))
        max_val = float(np.max(vals_pos_check))
        if not self.use_output_warp:
            self._y_shift = 0.0
            self._max_y_pos = max(max_val, 1.0)
            ref_vals = y_clean[unc] if int(np.sum(unc)) >= 2 else lod_arr
            self._g_mean = float(np.mean(ref_vals))
            self._g_std = float(max(np.std(ref_vals), 1e-6))
        elif min_val > 0.0:
            self._y_shift = 0.0
            self._max_y_pos = max_val
        else:
            span = max(max_val - min_val, float(np.std(vals_pos_check)), 1.0)
            offset = 0.05 * span
            self._y_shift = -min_val + offset
            self._max_y_pos = max_val + self._y_shift

        u = _normalize_inputs(Xa, lower, upper)
        outlier_var = np.zeros(n, dtype=np.float64)

        fit_state = self._fit_single_stage(
            u, y_clean, lod_arr, censored, noise_arr, outlier_var, seed=int(seed)
        )

        # Robust LOO-IRLS outlier down-weighting
        if self.robust and int(np.sum(unc)) >= 5:
            prev_flagged = np.zeros(n, dtype=bool)
            for _irls in range(3):
                e_loo, sig_loo_base = self._compute_loo_residuals(
                    fit_state, censored, outlier_var
                )
                z_loo = np.abs(e_loo) / sig_loo_base
                flagged = unc & (z_loo > 3.0)
                if not np.any(flagged) and not np.any(prev_flagged):
                    break
                new_outlier_var = np.zeros(n, dtype=np.float64)
                if np.any(flagged):
                    # Student-t / Huber inflation: scale variance by (z_loo / 1.5)^2
                    new_outlier_var[flagged] = (
                        (e_loo[flagged] ** 2) * np.maximum((z_loo[flagged] / 3.0) ** 2, 1.0)
                    )
                if np.array_equal(flagged, prev_flagged) and np.allclose(
                    new_outlier_var, outlier_var, rtol=0.1, atol=1e-4
                ):
                    break
                outlier_var = new_outlier_var
                prev_flagged = flagged
                fit_state = self._fit_single_stage(
                    u,
                    y_clean,
                    lod_arr,
                    censored,
                    noise_arr,
                    outlier_var,
                    seed=int(seed),
                    warm_params=fit_state["p_opt"],
                )

        # Persist model state
        self._X = Xa
        self._y = y_clean
        self._bounds = np.vstack([lower, upper])
        self._lower = lower
        self._upper = upper
        self._censored_mask = censored
        self._lod = lod_arr
        self._fixed_noise = noise_arr is not None

        self._m = float(fit_state["m"])
        self._log_sf = float(fit_state["log_sf"])
        self._log_sn = float(fit_state["log_sn"])
        self._lam = float(fit_state["lam"])
        self._theta = np.asarray(fit_state["theta"], dtype=np.float64)
        self._warp_a = np.asarray(fit_state["warp_a"], dtype=np.float64)
        self._warp_b = np.asarray(fit_state["warp_b"], dtype=np.float64)
        self._g_mean = float(fit_state["g_mean"])
        self._g_std = float(fit_state["g_std"])
        self._w_train = np.asarray(fit_state["w_train"], dtype=np.float64)
        self._alpha = np.asarray(fit_state["alpha"], dtype=np.float64)
        self._L_chol = fit_state["L_chol"]
        self._L_B = fit_state["L_B"]
        self._S_laplace = fit_state["S_laplace"]
        self._is_laplace = bool(fit_state["is_laplace"])
        self._active_input_warp = bool(fit_state["active_input_warp"])

        lengthscales = np.exp(self._theta)
        rho = np.exp(-2.0 * self._theta)
        rho_sum = float(np.sum(rho))
        relevance_shares = rho / rho_sum if rho_sum > 0.0 else np.full(d, 1.0 / d)
        self._relevance_shares = relevance_shares
        active_dims = self.active_dimensions(0.05)

        self.fit_diagnostics = {
            "surrogate_kind": "warped_sparse_gp",
            "output_lambda": round(self._lam, 6),
            "warp_alpha": [round(float(v), 6) for v in np.exp(self._warp_a)],
            "warp_beta": [round(float(v), 6) for v in np.exp(self._warp_b)],
            "lengthscales": [round(float(v), 6) for v in lengthscales],
            "relevance_shares": [round(float(v), 6) for v in relevance_shares],
            "active_dims": active_dims,
            "n_censored": int(np.sum(censored)),
            "n_outliers_downweighted": int(np.sum(outlier_var > 0.0)),
            "marginal_log_likelihood": round(float(fit_state["mll"]), 6),
            "signal_variance": round(float(np.exp(2.0 * self._log_sf)), 6),
            "noise_variance": (
                None if self._fixed_noise else round(float(np.exp(2.0 * self._log_sn)), 6)
            ),
            "fit_seconds": round(time.perf_counter() - t0, 4),
        }
        return self

    def _predict_warped_space(
        self, X: Any, *, observation_noise: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict ``(mean_g, var_g, cov_g)`` in unstandardized Box-Cox space."""
        assert self._w_train is not None and self._theta is not None
        assert self._lower is not None and self._upper is not None
        assert self._warp_a is not None and self._warp_b is not None
        assert self._alpha is not None

        Xq = np.asarray(X, dtype=np.float64)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        if Xq.ndim != 2 or Xq.shape[1] != self._w_train.shape[1]:
            raise ValueError(
                f"X must have shape (m, {self._w_train.shape[1]}), got {Xq.shape}"
            )
        uq = _normalize_inputs(Xq, self._lower, self._upper)
        wq = _kumaraswamy_warp(uq, self._warp_a, self._warp_b)[0] if self._active_input_warp else uq

        K_star = _matern52_kernel(self._w_train, wq, self._log_sf, self._theta)
        K_ss = _matern52_kernel(wq, wq, self._log_sf, self._theta)

        mean_std = self._m + K_star.T @ self._alpha
        if not self._is_laplace:
            assert self._L_chol is not None
            v = solve_triangular(
                self._L_chol, K_star, lower=True, check_finite=False
            )
            cov_std = K_ss - v.T @ v
        else:
            assert self._L_B is not None and self._S_laplace is not None
            v = solve_triangular(
                self._L_B, self._S_laplace[:, None] * K_star, lower=True, check_finite=False
            )
            cov_std = K_ss - v.T @ v

        cov_std = 0.5 * (cov_std + cov_std.T)
        if observation_noise and not self._fixed_noise:
            sn2 = float(np.exp(2.0 * self._log_sn))
            cov_std[np.diag_indices(Xq.shape[0])] += sn2
        elif observation_noise and self._fixed_noise:
            # Small floor when fixed training noise was supplied without test noise
            cov_std[np.diag_indices(Xq.shape[0])] += 1e-6

        mean_g = self._g_mean + self._g_std * mean_std
        scale2 = self._g_std * self._g_std
        cov_g = scale2 * cov_std
        # Ensure PSD against tiny float64 roundoff
        var_g = np.maximum(np.diag(cov_g), 1e-12)
        return mean_g, var_g, cov_g

    def posterior(
        self, X: Any, *, observation_noise: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` in original ``y`` units via 11-point Gauss-Hermite."""
        mean_g, var_g, _ = self._predict_warped_space(
            X, observation_noise=observation_noise
        )
        if not self.use_output_warp:
            return mean_g, np.sqrt(var_g)

        std_g = np.sqrt(var_g)
        # Shape (m, 11)
        g_nodes = mean_g[:, None] + std_g[:, None] * _GH_Z[None, :]
        y_nodes = _boxcox_inverse(g_nodes, self._lam, self._max_y_pos) - self._y_shift
        mean_y = y_nodes @ _GH_WEIGHTS
        diff = y_nodes - mean_y[:, None]
        var_y = np.maximum((diff * diff) @ _GH_WEIGHTS, 1e-12)
        return mean_y, np.sqrt(var_y)

    def posterior_cov(
        self, X: Any, *, observation_noise: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean (m,), cov (m, m))`` in original ``y`` units (symmetric PSD)."""
        mean_g, var_g, cov_g = self._predict_warped_space(
            X, observation_noise=observation_noise
        )
        m_pts = mean_g.shape[0]
        if not self.use_output_warp:
            eigvals, eigvecs = np.linalg.eigh(cov_g)
            cov_psd = (eigvecs * np.maximum(eigvals, 1e-12)[None, :]) @ eigvecs.T
            cov_psd = 0.5 * (cov_psd + cov_psd.T)
            return mean_g, cov_psd

        std_g = np.sqrt(var_g)
        g_nodes = mean_g[:, None] + std_g[:, None] * _GH_Z[None, :]
        y_nodes = _boxcox_inverse(g_nodes, self._lam, self._max_y_pos) - self._y_shift
        mean_y = y_nodes @ _GH_WEIGHTS
        diff = y_nodes - mean_y[:, None]
        var_y = np.maximum((diff * diff) @ _GH_WEIGHTS, 1e-12)
        std_y = np.sqrt(var_y)

        # Calibrate marginal scale by GH std_y / std_g (fallback to analytical Jacobian)
        jac = np.where(
            std_g > 1e-6,
            std_y / np.maximum(std_g, 1e-12),
            _boxcox_inv_deriv(mean_g, self._lam, self._max_y_pos),
        )
        cov_y = (jac[:, None] * cov_g) * jac[None, :]
        cov_y = 0.5 * (cov_y + cov_y.T)
        eigvals, eigvecs = np.linalg.eigh(cov_y)
        if np.any(eigvals < 1e-12):
            cov_y = (eigvecs * np.maximum(eigvals, 1e-12)[None, :]) @ eigvecs.T
            cov_y = 0.5 * (cov_y + cov_y.T)
        # Ensure diagonal matches exact 11-pt GH variance
        np.fill_diagonal(cov_y, np.maximum(np.diag(cov_y), 1e-12))
        assert cov_y.shape == (m_pts, m_pts)
        return mean_y, cov_y

    def posterior_quantiles(
        self,
        X: Any,
        quantiles: tuple[float, ...] | list[float] = (0.05, 0.5, 0.95),
        *,
        observation_noise: bool = True,
    ) -> np.ndarray:
        """Return monotonic posterior quantiles of shape ``(m, len(quantiles))``."""
        q_arr = np.asarray(quantiles, dtype=np.float64).reshape(-1)
        if q_arr.size == 0 or np.any(q_arr <= 0.0) or np.any(q_arr >= 1.0):
            raise ValueError("quantiles must be strictly inside (0, 1)")
        mean_g, var_g, _ = self._predict_warped_space(
            X, observation_noise=observation_noise
        )
        std_g = np.sqrt(var_g)
        z_q = ndtri(q_arr)
        g_q = mean_g[:, None] + std_g[:, None] * z_q[None, :]
        if not self.use_output_warp:
            return g_q
        return _boxcox_inverse(g_q, self._lam, self._max_y_pos) - self._y_shift

    def warped_posterior_mean_std(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean_std, std_std)`` in standardized warped GP space."""
        mean_g, var_g, _ = self._predict_warped_space(X, observation_noise=False)
        mean_std = (mean_g - self._g_mean) / self._g_std
        std_std = np.sqrt(np.maximum(var_g / (self._g_std * self._g_std), 1e-12))
        return mean_std, std_std

    def posterior_warped_and_grad(
        self, x: Any
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Return exact standardized warped ``(mu_std, var_std, dmu_dx, dvar_dx)`` at ``x``."""
        assert self._w_train is not None and self._theta is not None
        assert self._lower is not None and self._upper is not None
        assert self._warp_a is not None and self._warp_b is not None
        assert self._alpha is not None

        d = int(self._w_train.shape[1])
        x_arr = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x_arr.shape != (1, d):
            raise ValueError(f"x must have {d} coordinates, got shape {x_arr.shape}")

        width = np.maximum(self._upper - self._lower, 1e-12)
        u_raw = ((x_arr - self._lower[None, :]) / width[None, :])[0]
        u = _normalize_inputs(x_arr, self._lower, self._upper)[0]
        du_dx = np.where(
            (u_raw >= 0.0) & (u_raw <= 1.0),
            (1.0 - 2.0 * _INPUT_EPS) / width,
            0.0,
        )

        if self._active_input_warp:
            a = np.exp(self._warp_a)
            b = np.exp(self._warp_b)
            p = np.power(u, a)
            q = np.clip(1.0 - p, 1e-12, 1.0)
            r_k = np.power(q, b)
            w = 1.0 - r_k
            dw_du = b * a * (r_k / q) * (p / np.maximum(u, 1e-12))
        else:
            w = u
            dw_du = np.ones_like(u)

        sf2 = float(np.exp(2.0 * self._log_sf))
        rho = np.exp(-2.0 * self._theta)
        diff = w[None, :] - self._w_train
        r2 = np.sum((diff * diff) * rho[None, :], axis=1)
        r = np.sqrt(np.maximum(r2, 1e-18))
        sr = _SQRT5 * r
        e_sr = np.exp(-sr)
        k = sf2 * (1.0 + sr + (5.0 / 3.0) * r2) * e_sr
        dk_fac = -(5.0 / 3.0) * sf2 * (1.0 + sr) * e_sr
        dk_dw = (dk_fac[:, None] * diff) * rho[None, :]

        mu_std = float(self._m + np.dot(k, self._alpha))
        dmu_dw = dk_dw.T @ self._alpha

        if not self._is_laplace:
            assert self._L_chol is not None
            v = solve_triangular(self._L_chol, k, lower=True, check_finite=False)
            var_std = max(sf2 - float(np.dot(v, v)), 1e-10)
            kinv_k = solve_triangular(self._L_chol.T, v, lower=False, check_finite=False)
            dvar_dw = -2.0 * (dk_dw.T @ kinv_k)
        else:
            assert self._L_B is not None and self._S_laplace is not None
            sk = self._S_laplace * k
            v = solve_triangular(self._L_B, sk, lower=True, check_finite=False)
            var_std = max(sf2 - float(np.dot(v, v)), 1e-10)
            binv_sk = solve_triangular(self._L_B.T, v, lower=False, check_finite=False)
            dvar_dw = -2.0 * (dk_dw.T @ (self._S_laplace * binv_sk))

        chain = dw_du * du_dx
        return mu_std, var_std, dmu_dw * chain, dvar_dw * chain

    def qlognei_context(self, X_train: Any = None) -> dict[str, Any]:
        """Precompute standardized-warped posterior operators for exact ``JointQLogNEI``."""
        assert self._w_train is not None and self._theta is not None
        assert self._lower is not None and self._upper is not None
        assert self._warp_a is not None and self._warp_b is not None
        assert self._alpha is not None and self._X is not None

        if X_train is not None:
            Xa = np.asarray(X_train, dtype=np.float64)
            if Xa.ndim != 2 or Xa.shape != self._X.shape:
                raise ValueError(
                    f"X_train must match fitted training shape {self._X.shape}, got {Xa.shape}"
                )

        n = int(self._w_train.shape[0])
        mu_tr_g, _, cov_tr_g = self._predict_warped_space(self._X, observation_noise=False)
        mu_tr = (mu_tr_g - self._g_mean) / self._g_std
        Sigma_tr = cov_tr_g / (self._g_std * self._g_std)

        sf2 = float(np.exp(2.0 * self._log_sf))
        rho = np.exp(-2.0 * self._theta)
        K_tt = _matern52_kernel(self._w_train, self._w_train, self._log_sf, self._theta)

        if not self._is_laplace:
            assert self._L_chol is not None
            L_gp = self._L_chol
            S_laplace: np.ndarray | None = None
            V_tt = solve_triangular(L_gp, K_tt, lower=True, check_finite=False)
        else:
            assert self._L_B is not None and self._S_laplace is not None
            L_gp = self._L_B
            S_laplace = self._S_laplace
            V_tt = solve_triangular(
                L_gp,
                S_laplace[:, None] * K_tt,
                lower=True,
                check_finite=False,
            )

        _ = n
        width = np.maximum(self._upper - self._lower, 1e-12)
        return {
            "lower": self._lower.copy(),
            "upper": self._upper.copy(),
            "width": width,
            "input_eps": float(_INPUT_EPS),
            "sf2": sf2,
            "rho": rho.copy(),
            "m": float(self._m),
            "alpha": self._alpha.copy(),
            "w_train": self._w_train.copy(),
            "active_input_warp": bool(self._active_input_warp),
            "warp_a": self._warp_a.copy(),
            "warp_b": self._warp_b.copy(),
            "mu_tr": mu_tr,
            "Sigma_tr": Sigma_tr,
            "L_gp": L_gp.copy(),
            "S_laplace": None if S_laplace is None else S_laplace.copy(),
            "V_tt": V_tt,
            "is_laplace": bool(self._is_laplace),
        }

    def active_dimensions(self, threshold_share: float = 0.05) -> list[int]:
        """Return dimension indices whose ARD relevance share >= ``threshold_share``."""
        if not hasattr(self, "_relevance_shares"):
            raise RuntimeError("fit() must be called before active_dimensions()")
        return [
            int(j)
            for j, share in enumerate(self._relevance_shares)
            if float(share) >= float(threshold_share)
        ]

    @property
    def best_f(self) -> float:
        """Best observed uncensored target value."""
        if self._y is None:
            raise RuntimeError("fit() must be called before best_f")
        if self._censored_mask is not None and np.any(~self._censored_mask):
            return float(np.max(self._y[~self._censored_mask]))
        return float(np.max(self._y))
