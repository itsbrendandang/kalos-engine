"""Pure NumPy/SciPy hierarchical mixed-effects and multi-scale transfer GPs.

This module is 100% torch-free (uses float64 exact Cholesky via
``scipy.linalg.cho_factor`` / ``cho_solve`` and ``scipy.optimize.minimize``
L-BFGS-B with analytical MLL gradients and escalating Cholesky jitter).

Two first-principles models are provided:

1. ``HierarchicalBatchGP``:
   Models multi-round / multi-batch bioprocess data
   :math:`y_{i,b} = f_{\\text{process}}(x_i) + u_b + \\varepsilon_{i,b}` with exact
   additive kernel:

   .. math::
       K((x_i, b_i), (x_j, b_j))
       = \\sigma_f^2 k_{\\text{Matern52}}^{\\text{ARD}}(x_i, x_j)
       + \\tau_{\\text{batch}}^2 \\mathbb{I}[b_i = b_j]
       + \\sigma_{\\text{assay}}^2 \\delta_{ij}

   Predicting with ``group=None`` sets the batch cross-covariance to zero,
   yielding pure de-biased process predictions :math:`f_{\\text{process}}(x_*)`.
   Predicting with ``group=<seen_batch>`` conditions on the learned BLUP
   batch offset :math:`u_b`.

2. ``MultiScaleTransferGP``:
   Exact Intrinsic Coregionalization Model (ICM) for multi-scale /
   multi-fidelity transfer (e.g., bench -> pilot -> production):

   .. math::
       K((x_i, s_i), (x_j, s_j))
       = k_{\\text{Matern52}}^{\\text{ARD}}(x_i, x_j) \\cdot B_{s_i, s_j}
       + \\sigma_{s_i}^2 \\delta_{ij},
       \\quad B = L L^\\top + \\operatorname{diag}(\\kappa)
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_FIT_JITTERS: tuple[float, ...] = (1e-8, 1e-6, 1e-4)
_SQRT5 = math.sqrt(5.0)
_LOG_2PI = math.log(2.0 * math.pi)
_MIN_STD = 1e-12
_ChoFactor = tuple[NDArray[np.float64], NDArray[np.bool_]]


def _sanitize_bounds(
    bounds: ArrayLike,
    expected_d: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and sanitize ``(2, d)`` bounds into finite ``(lower, upper)`` rows."""
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim != 2 or arr.shape != (2, expected_d):
        raise ValueError(f"bounds must have shape (2, {expected_d}), got {arr.shape}")
    lower = arr[0].copy()
    upper = arr[1].copy()
    for row, partner in ((lower, upper), (upper, lower)):
        bad = ~np.isfinite(row)
        if bad.any():
            fill = np.where(np.isfinite(partner), partner, 0.0)
            row[bad] = fill[bad]
    lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
    degenerate = upper - lower <= 0.0
    if degenerate.any():
        pad = 1e-6 * np.maximum(np.abs(lower), 1.0)
        upper = np.where(degenerate, lower + pad, upper)
    return lower, upper


def _coerce_labels(labels: Sequence[Any] | NDArray[Any], n: int, name: str) -> list[str]:
    """Validate and convert per-row group/scale labels to strings."""
    seq = list(labels)
    if len(seq) != n:
        raise ValueError(f"{name} must have length {n} matching X, got {len(seq)}")
    out: list[str] = []
    for i, item in enumerate(seq):
        if item is None:
            raise ValueError(f"{name}[{i}] must not be None")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name}[{i}] must be finite")
        if isinstance(item, np.floating) and not np.isfinite(item):
            raise ValueError(f"{name}[{i}] must be finite")
        s = str(item)
        if not s:
            raise ValueError(f"{name}[{i}] must not be empty")
        out.append(s)
    return out


def _validate_fit_inputs(
    X: ArrayLike,
    y: ArrayLike,
    labels: Sequence[Any] | NDArray[Any],
    bounds: ArrayLike,
    label_name: str,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    list[str],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate training arrays, labels, and bounds."""
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    if Xa.ndim != 2 or Xa.shape[0] == 0 or Xa.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array (n x d)")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"X ({Xa.shape[0]}) and y ({ya.shape[0]}) must have the same number of rows")
    if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
        raise ValueError("X and y must be finite (no NaN or inf)")
    str_labels = _coerce_labels(labels, Xa.shape[0], label_name)
    lower, upper = _sanitize_bounds(bounds, Xa.shape[1])
    return Xa, ya, str_labels, lower, upper


def _normalize_query(
    X: ArrayLike,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    d: int,
) -> NDArray[np.float64]:
    """Validate query X and normalize to [0, 1]^d using fixed bounds."""
    Xa = np.asarray(X, dtype=np.float64)
    if Xa.ndim == 1:
        Xa = Xa.reshape(1, -1)
    if Xa.ndim != 2 or Xa.shape[1] != d:
        raise ValueError(f"X must have shape (m, {d}), got {Xa.shape}")
    if Xa.shape[0] == 0:
        return np.empty((0, d), dtype=np.float64)
    if not np.isfinite(Xa).all():
        raise ValueError("X must be finite (no NaN or inf)")
    return (Xa - lower) / (upper - lower)


def _matern52_ard(
    X1: NDArray[np.float64],
    X2: NDArray[np.float64],
    lengthscales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute ARD Matern-5/2 correlation matrix between X1 and X2."""
    r = cdist(X1 / lengthscales, X2 / lengthscales, metric="euclidean")
    s5r = _SQRT5 * r
    return (1.0 + s5r + (5.0 / 3.0) * (r * r)) * np.exp(-s5r)


def _matern52_ard_train(
    diff: NDArray[np.float64],
    lengthscales: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Matern-5/2 kernel matrix and intermediates needed for ARD gradients."""
    scaled = diff / lengthscales[None, None, :]
    r2 = np.sum(scaled * scaled, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    s5r = _SQRT5 * r
    exp_s5r = np.exp(-s5r)
    K = (1.0 + s5r + (5.0 / 3.0) * r2) * exp_s5r
    return K, s5r, exp_s5r


def _solve_gls_mean(
    cho: _ChoFactor,
    design: NDArray[np.float64],
    y_std: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve exact GLS mean c_hat = (D^T K^{-1} D)^{-1} D^T K^{-1} y and alpha = K^{-1}(y - D c_hat)."""
    K_inv_D = cho_solve(cho, design)
    K_inv_y = cho_solve(cho, y_std)
    gram = design.T @ K_inv_D
    rhs = design.T @ K_inv_y
    try:
        gram_cho = cho_factor(0.5 * (gram + gram.T), lower=True)
        c_hat = cho_solve(gram_cho, rhs)
    except LinAlgError:
        p = gram.shape[0]
        gram_reg = 0.5 * (gram + gram.T) + 1e-10 * np.eye(p, dtype=np.float64)
        c_hat = np.linalg.solve(gram_reg, rhs)
    residual = y_std - design @ c_hat
    alpha = K_inv_y - K_inv_D @ c_hat
    return c_hat, residual, alpha


def _mll_and_intermediates(
    K: NDArray[np.float64],
    design: NDArray[np.float64],
    y_std: NDArray[np.float64],
) -> tuple[
    float,
    NDArray[np.float64],
    NDArray[np.float64],
    _ChoFactor,
    NDArray[np.float64],
]:
    """Factor K, evaluate exact profile MLL, and return W = alpha alpha^T - K^{-1}."""
    n = K.shape[0]
    cho = cho_factor(0.5 * (K + K.T), lower=True)
    c_hat, residual, alpha = _solve_gls_mean(cho, design, y_std)
    log_det = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
    mll = -0.5 * (float(residual @ alpha) + log_det + n * _LOG_2PI)
    K_inv = cho_solve(cho, np.eye(n, dtype=np.float64))
    W = np.outer(alpha, alpha) - K_inv
    return mll, W, alpha, cho, c_hat


def _ensure_psd(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize covariance matrix and project tiny negative numerical eigenvalues."""
    sym = 0.5 * (cov + cov.T)
    if sym.shape[0] == 0:
        return sym
    eigvals = np.linalg.eigvalsh(sym)
    if eigvals.min() < 0.0:
        vals, vecs = np.linalg.eigh(sym)
        vals = np.maximum(vals, 0.0)
        sym = (vecs * vals) @ vecs.T
        sym = 0.5 * (sym + sym.T)
    return sym


def _optimize_multistart(
    eval_fn: Callable[[NDArray[np.float64], float], tuple[float, NDArray[np.float64]]],
    starts: list[NDArray[np.float64]],
    opt_bounds: list[tuple[float, float]],
) -> tuple[NDArray[np.float64], float, float]:
    """Run multi-start L-BFGS-B with escalating Cholesky jitter (1e-8 -> 1e-6 -> 1e-4)."""
    best_theta: NDArray[np.float64] | None = None
    best_mll = -math.inf
    best_jitter = _FIT_JITTERS[-1]

    for theta0 in starts:
        theta0_c = np.clip(
            theta0,
            [b[0] for b in opt_bounds],
            [b[1] for b in opt_bounds],
        )
        for jitter in _FIT_JITTERS:
            try:
                eval_fn(theta0_c, jitter)
            except LinAlgError:
                continue

            def _obj(
                theta: NDArray[np.float64],
                _j: float = jitter,
            ) -> tuple[float, NDArray[np.float64]]:
                try:
                    mll, grad = eval_fn(theta, _j)
                except LinAlgError:
                    return 1e12, np.zeros_like(theta)
                if not (math.isfinite(mll) and np.isfinite(grad).all()):
                    return 1e12, np.zeros_like(theta)
                return -mll, -grad

            res = minimize(
                _obj,
                theta0_c,
                method="L-BFGS-B",
                jac=True,
                bounds=opt_bounds,
                options={"maxiter": 250, "ftol": 1e-11, "gtol": 1e-7},
            )
            try:
                final_mll, _ = eval_fn(res.x, jitter)
            except LinAlgError:
                continue
            if not math.isfinite(final_mll):
                continue
            if best_theta is None or final_mll > best_mll:
                best_theta = res.x.copy()
                best_mll = final_mll
                best_jitter = jitter
            break

    if best_theta is None:
        raise RuntimeError(
            "GP hyperparameter optimization failed across all starts and Cholesky jitter rungs "
            f"{_FIT_JITTERS}"
        )
    return best_theta, best_mll, best_jitter


class HierarchicalBatchGP:
    """Hierarchical mixed-effects Gaussian Process for multi-batch / multi-round data.

    Models :math:`y_{i,b} = f_{\\text{process}}(x_i) + u_b + \\varepsilon_{i,b}` with
    exact additive kernel over normalized inputs :math:`x \\in [0, 1]^d`:

    .. math::
        K((x_i, b_i), (x_j, b_j))
        = \\sigma_f^2 k_{\\text{Matern52}}^{\\text{ARD}}(x_i, x_j)
        + \\tau_{\\text{batch}}^2 \\mathbb{I}[b_i = b_j]
        + \\sigma_{\\text{assay}}^2 \\delta_{ij}
    """

    def __init__(self) -> None:
        self._X: NDArray[np.float64] | None = None
        self._X_norm: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None
        self._bounds: NDArray[np.float64] | None = None
        self._lower: NDArray[np.float64] | None = None
        self._upper: NDArray[np.float64] | None = None
        self._groups: list[str] | None = None
        self._unique_groups: list[str] = []
        self._group_masks: dict[str, NDArray[np.float64]] = {}
        self._y_mean: float = 0.0
        self._y_scale: float = 1.0
        self._c_hat: float = 0.0
        self._sigma_f_std: float = 1.0
        self._tau_batch_std: float = 0.0
        self._sigma_assay_std: float = 0.0
        self._lengthscales: NDArray[np.float64] | None = None
        self._cho: _ChoFactor | None = None
        self._alpha: NDArray[np.float64] | None = None
        self._jitter_used: float = _FIT_JITTERS[0]
        self.fit_diagnostics: dict[str, Any] = {}

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Sequence[Any] | NDArray[Any],
        bounds: ArrayLike,
        *,
        seed: int = 0,
        n_restarts: int = 5,
    ) -> HierarchicalBatchGP:
        """Fit the hierarchical mixed-effects GP by exact marginal log-likelihood."""
        t0 = time.perf_counter()
        Xa, ya, group_list, lower, upper = _validate_fit_inputs(X, y, groups, bounds, "groups")
        n, d = Xa.shape
        X_norm = (Xa - lower) / (upper - lower)

        y_mean = float(np.mean(ya))
        y_std_raw = float(np.std(ya, ddof=0))
        y_scale = y_std_raw if y_std_raw > _MIN_STD else 1.0
        y_std = (ya - y_mean) / y_scale

        unique_groups: list[str] = []
        seen: dict[str, int] = {}
        g_idx = np.empty(n, dtype=np.int64)
        for i, g in enumerate(group_list):
            idx = seen.get(g)
            if idx is None:
                idx = len(unique_groups)
                seen[g] = idx
                unique_groups.append(g)
            g_idx[i] = idx

        M_batch = (g_idx[:, None] == g_idx[None, :]).astype(np.float64)
        diff = X_norm[:, None, :] - X_norm[None, :, :]
        diff_sq = diff * diff
        design = np.ones((n, 1), dtype=np.float64)
        eye_n = np.eye(n, dtype=np.float64)

        def _unpack(
            theta: NDArray[np.float64],
        ) -> tuple[float, NDArray[np.float64], float, float]:
            sigma_f = float(np.exp(theta[0]))
            ls = np.exp(theta[1 : 1 + d])
            tau = float(np.exp(theta[1 + d]))
            sigma_a = float(np.exp(theta[2 + d]))
            return sigma_f, ls, tau, sigma_a

        def _eval_mll_and_grad(
            theta: NDArray[np.float64],
            jitter: float,
        ) -> tuple[float, NDArray[np.float64]]:
            sigma_f, ls, tau, sigma_a = _unpack(theta)
            sf2 = sigma_f * sigma_f
            tau2 = tau * tau
            sa2 = sigma_a * sigma_a
            K_f, s5r, exp_s5r = _matern52_ard_train(diff, ls)
            K = sf2 * K_f + tau2 * M_batch + (sa2 + jitter) * eye_n
            mll, W, _, _, _ = _mll_and_intermediates(K, design, y_std)

            grad = np.empty_like(theta)
            grad[0] = sf2 * float(np.sum(W * K_f))
            Q = W * (sf2 * (5.0 / 3.0) * (1.0 + s5r) * exp_s5r)
            grad[1 : 1 + d] = np.einsum("ij,ijk->k", Q, diff_sq) / (2.0 * ls * ls)
            grad[1 + d] = tau2 * float(np.sum(W * M_batch))
            grad[2 + d] = sa2 * float(np.trace(W))
            return mll, grad

        # Empirical initial guess for start 0.
        group_means = [float(np.mean(y_std[g_idx == b])) for b in range(len(unique_groups))]
        tau_init = float(np.clip(np.std(group_means), 0.1, 1.5)) if len(unique_groups) >= 2 else 0.1
        sf_init = float(np.clip(math.sqrt(max(1.0 - tau_init * tau_init, 0.1)), 0.2, 1.5))
        starts: list[NDArray[np.float64]] = [
            np.concatenate([
                [math.log(sf_init)],
                np.full(d, math.log(0.35), dtype=np.float64),
                [math.log(tau_init), math.log(0.15)],
            ])
        ]
        rng = np.random.default_rng(int(seed))
        for _ in range(max(0, int(n_restarts) - 1)):
            starts.append(
                np.concatenate([
                    [rng.uniform(math.log(0.1), math.log(2.0))],
                    rng.uniform(math.log(0.05), math.log(2.5), size=d),
                    [
                        rng.uniform(math.log(0.02), math.log(2.0)),
                        rng.uniform(math.log(0.01), math.log(0.8)),
                    ],
                ])
            )

        opt_bounds: list[tuple[float, float]] = (
            [(math.log(1e-3), math.log(10.0))]
            + [(math.log(0.02), math.log(20.0))] * d
            + [(math.log(1e-4), math.log(10.0)), (math.log(1e-3), math.log(5.0))]
        )

        best_theta, best_mll_std, best_jitter = _optimize_multistart(
            _eval_mll_and_grad, starts, opt_bounds
        )
        sigma_f, ls, tau, sigma_a = _unpack(best_theta)
        sf2 = sigma_f * sigma_f
        tau2 = tau * tau
        sa2 = sigma_a * sigma_a
        K_f, _, _ = _matern52_ard_train(diff, ls)
        K_final = sf2 * K_f + tau2 * M_batch + (sa2 + best_jitter) * eye_n
        cho = cho_factor(0.5 * (K_final + K_final.T), lower=True)
        c_hat_arr, _, alpha = _solve_gls_mean(cho, design, y_std)

        self._X = Xa
        self._X_norm = X_norm
        self._y = ya
        self._bounds = np.vstack([lower, upper])
        self._lower = lower
        self._upper = upper
        self._groups = group_list
        self._unique_groups = unique_groups
        self._group_masks = {
            g: (g_idx == idx).astype(np.float64) for idx, g in enumerate(unique_groups)
        }
        self._y_mean = y_mean
        self._y_scale = y_scale
        self._c_hat = float(c_hat_arr[0])
        self._sigma_f_std = sigma_f
        self._tau_batch_std = tau
        self._sigma_assay_std = sigma_a
        self._lengthscales = ls.copy()
        self._cho = cho
        self._alpha = alpha
        self._jitter_used = best_jitter

        mll_orig = best_mll_std - n * math.log(y_scale)
        vd = self.variance_decomposition()
        self.fit_diagnostics = {
            "surrogate_kind": "hierarchical_batch_gp",
            "lengthscales": [float(v) for v in ls],
            "process_var": vd["process_var"],
            "batch_var": vd["batch_var"],
            "assay_var": vd["assay_var"],
            "n_batches": len(unique_groups),
            "marginal_log_likelihood": float(mll_orig),
            "jitter_used": float(best_jitter),
            "n_restarts": len(starts),
            "seed": int(seed),
            "fit_seconds": round(time.perf_counter() - t0, 4),
        }
        return self

    def _check_fitted(self) -> None:
        if (
            self._X_norm is None
            or self._lower is None
            or self._upper is None
            or self._lengthscales is None
            or self._cho is None
            or self._alpha is None
        ):
            raise RuntimeError("HierarchicalBatchGP must be fitted before querying")

    def posterior_cov(
        self,
        X: ArrayLike,
        *,
        group: Any = None,
        observation_noise: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, cov)`` in original ``y`` units.

        When ``group=None`` (unseen batch), batch cross-covariance is zero,
        yielding the pure de-biased process posterior :math:`f_{\\text{process}}(x_*)`
        (and adding ``tau_batch^2 + sigma_assay^2`` on the diagonal when
        ``observation_noise=True``). When ``group`` is a seen batch label, the
        learned BLUP batch offset :math:`u_b` and its conditioning variance
        reduction are included (adding ``sigma_assay^2`` when
        ``observation_noise=True``).
        """
        self._check_fitted()
        assert self._X_norm is not None
        assert self._lower is not None and self._upper is not None
        assert self._lengthscales is not None
        assert self._cho is not None and self._alpha is not None

        d = self._X_norm.shape[1]
        X_norm = _normalize_query(X, self._lower, self._upper, d)
        m = X_norm.shape[0]
        if m == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        sf2 = self._sigma_f_std * self._sigma_f_std
        tau2 = self._tau_batch_std * self._tau_batch_std
        sa2 = self._sigma_assay_std * self._sigma_assay_std

        K_f_cross = _matern52_ard(X_norm, self._X_norm, self._lengthscales)
        K_f_test = _matern52_ard(X_norm, X_norm, self._lengthscales)

        K_cross = sf2 * K_f_cross
        K_test = sf2 * K_f_test

        if group is None:
            noise_diag = (tau2 + sa2) if observation_noise else 0.0
        else:
            g_str = str(group)
            mask = self._group_masks.get(g_str)
            if mask is not None:
                K_cross = K_cross + tau2 * mask[None, :]
            K_test = K_test + tau2 * np.ones((m, m), dtype=np.float64)
            noise_diag = sa2 if observation_noise else 0.0

        mean_std = self._c_hat + K_cross @ self._alpha
        v = cho_solve(self._cho, K_cross.T)
        cov_std = K_test - K_cross @ v
        if noise_diag > 0.0:
            cov_std = cov_std + noise_diag * np.eye(m, dtype=np.float64)

        mean = self._y_mean + self._y_scale * mean_std
        cov = _ensure_psd(cov_std * (self._y_scale * self._y_scale))
        return mean, cov

    def posterior(
        self,
        X: ArrayLike,
        *,
        group: Any = None,
        observation_noise: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, std)`` in original ``y`` units."""
        self._check_fitted()
        mean, cov = self.posterior_cov(X, group=group, observation_noise=observation_noise)
        if mean.shape[0] == 0:
            return mean, np.empty(0, dtype=np.float64)
        min_var = _MIN_STD * (self._y_scale * self._y_scale)
        std = np.sqrt(np.maximum(np.diag(cov), min_var))
        return mean, std

    def batch_effects(self) -> dict[str, dict[str, float]]:
        """Return per-batch BLUP random-effect ``mean`` and ``std`` in original ``y`` units."""
        self._check_fitted()
        assert self._cho is not None and self._alpha is not None
        tau2 = self._tau_batch_std * self._tau_batch_std
        scale2 = self._y_scale * self._y_scale
        out: dict[str, dict[str, float]] = {}
        for g in self._unique_groups:
            k_ug = tau2 * self._group_masks[g]
            mean_std = float(k_ug @ self._alpha)
            var_std = max(tau2 - float(k_ug @ cho_solve(self._cho, k_ug)), 0.0)
            out[g] = {
                "mean": self._y_scale * mean_std,
                "std": math.sqrt(var_std * scale2),
            }
        return out

    def variance_decomposition(self) -> dict[str, float]:
        """Return variance components and shares (summing to 1.0) in original ``y`` units."""
        self._check_fitted()
        scale2 = self._y_scale * self._y_scale
        process_var = (self._sigma_f_std * self._sigma_f_std) * scale2
        batch_var = (self._tau_batch_std * self._tau_batch_std) * scale2
        assay_var = (self._sigma_assay_std * self._sigma_assay_std) * scale2
        total = process_var + batch_var + assay_var
        if total <= 0.0:
            return {
                "process_var": 0.0,
                "batch_var": 0.0,
                "assay_var": 0.0,
                "process_share": 0.0,
                "batch_share": 0.0,
                "assay_share": 1.0,
            }
        return {
            "process_var": float(process_var),
            "batch_var": float(batch_var),
            "assay_var": float(assay_var),
            "process_share": float(process_var / total),
            "batch_share": float(batch_var / total),
            "assay_share": float(assay_var / total),
        }

    @property
    def best_f(self) -> float:
        """Best observed target value in original units."""
        if self._y is None:
            raise RuntimeError("HierarchicalBatchGP must be fitted before accessing best_f")
        return float(self._y.max())


class MultiScaleTransferGP:
    """Intrinsic Coregionalization Model (ICM) GP for multi-scale / multi-fidelity transfer.

    Models :math:`S` experimental scales (e.g., ``bench``, ``pilot``, ``production``)
    with shared spatial correlation :math:`k_{\\text{Matern52}}^{\\text{ARD}}` and
    learned positive-definite task coregionalization matrix
    :math:`B = L L^\\top + \\operatorname{diag}(\\kappa)`:

    .. math::
        K((x_i, s_i), (x_j, s_j))
        = k_{\\text{Matern52}}^{\\text{ARD}}(x_i, x_j) \\cdot B_{s_i, s_j}
        + \\sigma_{s_i}^2 \\delta_{ij}

    Per-scale centering/scaling plus exact GLS per-scale intercepts allow seamless
    transfer across arbitrary linear shifts and gain differences (e.g., 10x titer
    unit differences between shake flask and bioreactor).
    """

    def __init__(self) -> None:
        self._X: NDArray[np.float64] | None = None
        self._X_norm: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None
        self._bounds: NDArray[np.float64] | None = None
        self._lower: NDArray[np.float64] | None = None
        self._upper: NDArray[np.float64] | None = None
        self._scales: list[str] | None = None
        self._scale_names: list[str] = []
        self._scale_to_idx: dict[str, int] = {}
        self._s_idx: NDArray[np.int64] | None = None
        self._scale_shift: NDArray[np.float64] | None = None
        self._scale_factor: NDArray[np.float64] | None = None
        self._c_hat: NDArray[np.float64] | None = None
        self._lengthscales: NDArray[np.float64] | None = None
        self._B_std: NDArray[np.float64] | None = None
        self._B_orig: NDArray[np.float64] | None = None
        self._sigma_sq_std: NDArray[np.float64] | None = None
        self._cho: _ChoFactor | None = None
        self._alpha: NDArray[np.float64] | None = None
        self._rank: int = 2
        self._jitter_used: float = _FIT_JITTERS[0]
        self.fit_diagnostics: dict[str, Any] = {}

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        scales: Sequence[Any] | NDArray[Any],
        bounds: ArrayLike,
        *,
        rank: int = 2,
        seed: int = 0,
        n_restarts: int = 5,
    ) -> MultiScaleTransferGP:
        """Fit the ICM multi-scale GP by exact profile marginal log-likelihood."""
        t0 = time.perf_counter()
        if int(rank) < 1:
            raise ValueError("rank must be >= 1")
        Xa, ya, scale_list, lower, upper = _validate_fit_inputs(X, y, scales, bounds, "scales")
        n, d = Xa.shape
        X_norm = (Xa - lower) / (upper - lower)

        scale_names: list[str] = []
        scale_to_idx: dict[str, int] = {}
        s_idx = np.empty(n, dtype=np.int64)
        for i, s in enumerate(scale_list):
            idx = scale_to_idx.get(s)
            if idx is None:
                idx = len(scale_names)
                scale_to_idx[s] = idx
                scale_names.append(s)
            s_idx[i] = idx

        S = len(scale_names)
        eff_rank = min(int(rank), S)

        # Per-scale centering & scaling so arbitrary 10x/1000x unit or offset differences
        # live on O(1) variance scales while GLS c_hat corrects for biased subset sampling.
        global_std = float(np.std(ya, ddof=0))
        fallback_scale = global_std if global_std > _MIN_STD else 1.0
        scale_shift = np.empty(S, dtype=np.float64)
        scale_factor = np.empty(S, dtype=np.float64)
        for s_i in range(S):
            ys = ya[s_idx == s_i]
            scale_shift[s_i] = float(np.mean(ys))
            s_std = float(np.std(ys, ddof=0)) if ys.shape[0] >= 2 else 0.0
            scale_factor[s_i] = s_std if s_std > 1e-6 * fallback_scale else fallback_scale

        y_std = (ya - scale_shift[s_idx]) / scale_factor[s_idx]
        D = np.zeros((n, S), dtype=np.float64)
        D[np.arange(n), s_idx] = 1.0

        diff = X_norm[:, None, :] - X_norm[None, :, :]
        diff_sq = diff * diff
        n_L = S * eff_rank

        def _unpack(
            theta: NDArray[np.float64],
        ) -> tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]:
            ls = np.exp(theta[:d])
            L = theta[d : d + n_L].reshape(S, eff_rank)
            kappa = np.exp(theta[d + n_L : d + n_L + S])
            sigma = np.exp(theta[d + n_L + S : d + n_L + 2 * S])
            return ls, L, kappa, sigma

        def _eval_mll_and_grad(
            theta: NDArray[np.float64],
            jitter: float,
        ) -> tuple[float, NDArray[np.float64]]:
            ls, L, kappa, sigma = _unpack(theta)
            B = L @ L.T + np.diag(kappa)
            sigma2 = sigma * sigma
            K_x, s5r, exp_s5r = _matern52_ard_train(diff, ls)
            B_mat = B[s_idx[:, None], s_idx[None, :]]
            K = K_x * B_mat
            diag_idx = np.arange(n)
            K[diag_idx, diag_idx] += sigma2[s_idx] + jitter

            mll, W, _, _, _ = _mll_and_intermediates(K, D, y_std)

            grad = np.empty_like(theta)
            Q = (W * B_mat) * ((5.0 / 3.0) * (1.0 + s5r) * exp_s5r)
            grad[:d] = np.einsum("ij,ijk->k", Q, diff_sq) / (2.0 * ls * ls)

            G_B = 0.5 * (D.T @ (W * K_x) @ D)
            G_B = 0.5 * (G_B + G_B.T)
            grad_L = 2.0 * (G_B @ L)
            grad[d : d + n_L] = grad_L.reshape(-1)
            grad[d + n_L : d + n_L + S] = kappa * np.diag(G_B)
            grad[d + n_L + S : d + n_L + 2 * S] = sigma2 * (D.T @ np.diag(W))
            return mll, grad

        # Non-zero L starts avoid the L=0 saddle point (dMLL/dL = 2 G_B L = 0).
        starts: list[NDArray[np.float64]] = []
        L_shared = np.zeros((S, eff_rank), dtype=np.float64)
        L_shared[:, 0] = 0.7
        if eff_rank > 1:
            for sc in range(min(S, eff_rank)):
                L_shared[sc, sc] += 0.25
        starts.append(
            np.concatenate([
                np.full(d, math.log(0.35), dtype=np.float64),
                L_shared.reshape(-1),
                np.full(S, math.log(0.2), dtype=np.float64),
                np.full(S, math.log(0.15), dtype=np.float64),
            ])
        )
        # Include one near-independent start so uncorrelated scales find low-rho easily.
        rng = np.random.default_rng(int(seed))
        L_weak = rng.normal(0.0, 0.15, size=(S, eff_rank))
        starts.append(
            np.concatenate([
                np.full(d, math.log(0.35), dtype=np.float64),
                L_weak.reshape(-1),
                np.full(S, math.log(0.8), dtype=np.float64),
                np.full(S, math.log(0.15), dtype=np.float64),
            ])
        )
        for _ in range(max(0, int(n_restarts) - 2)):
            starts.append(
                np.concatenate([
                    rng.uniform(math.log(0.05), math.log(2.5), size=d),
                    rng.normal(0.0, 0.7, size=n_L),
                    rng.uniform(math.log(0.02), math.log(1.5), size=S),
                    rng.uniform(math.log(0.01), math.log(0.6), size=S),
                ])
            )

        opt_bounds: list[tuple[float, float]] = (
            [(math.log(0.02), math.log(20.0))] * d
            + [(-5.0, 5.0)] * n_L
            + [(math.log(1e-4), math.log(10.0))] * S
            + [(math.log(1e-3), math.log(5.0))] * S
        )

        best_theta, best_mll_std, best_jitter = _optimize_multistart(
            _eval_mll_and_grad, starts, opt_bounds
        )
        ls, L, kappa, sigma = _unpack(best_theta)
        B_std = L @ L.T + np.diag(kappa)
        B_std = 0.5 * (B_std + B_std.T)
        sigma2 = sigma * sigma

        K_x, _, _ = _matern52_ard_train(diff, ls)
        K_final = K_x * B_std[s_idx[:, None], s_idx[None, :]]
        diag_idx = np.arange(n)
        K_final[diag_idx, diag_idx] += sigma2[s_idx] + best_jitter
        cho = cho_factor(0.5 * (K_final + K_final.T), lower=True)
        c_hat, _, alpha = _solve_gls_mean(cho, D, y_std)

        B_orig = B_std * np.outer(scale_factor, scale_factor)
        B_orig = 0.5 * (B_orig + B_orig.T)

        self._X = Xa
        self._X_norm = X_norm
        self._y = ya
        self._bounds = np.vstack([lower, upper])
        self._lower = lower
        self._upper = upper
        self._scales = scale_list
        self._scale_names = scale_names
        self._scale_to_idx = scale_to_idx
        self._s_idx = s_idx
        self._scale_shift = scale_shift
        self._scale_factor = scale_factor
        self._c_hat = c_hat
        self._lengthscales = ls.copy()
        self._B_std = B_std
        self._B_orig = B_orig
        self._sigma_sq_std = sigma2
        self._cho = cho
        self._alpha = alpha
        self._rank = eff_rank
        self._jitter_used = best_jitter

        mll_orig = best_mll_std - float(np.sum(np.log(scale_factor[s_idx])))
        _, corr = self.transfer_correlation_matrix()
        self.fit_diagnostics = {
            "surrogate_kind": "multi_scale_transfer_gp",
            "lengthscales": [float(v) for v in ls],
            "scales": list(scale_names),
            "rank": eff_rank,
            "scale_signal_vars": {
                s: float(B_orig[i, i]) for i, s in enumerate(scale_names)
            },
            "scale_noise_vars": {
                s: float(sigma2[i] * (scale_factor[i] ** 2))
                for i, s in enumerate(scale_names)
            },
            "correlation_matrix": [[float(v) for v in row] for row in corr],
            "marginal_log_likelihood": float(mll_orig),
            "jitter_used": float(best_jitter),
            "n_restarts": len(starts),
            "seed": int(seed),
            "fit_seconds": round(time.perf_counter() - t0, 4),
        }
        return self

    def _check_fitted(self) -> None:
        if (
            self._X_norm is None
            or self._lower is None
            or self._upper is None
            or self._s_idx is None
            or self._scale_shift is None
            or self._scale_factor is None
            or self._c_hat is None
            or self._lengthscales is None
            or self._B_std is None
            or self._B_orig is None
            or self._sigma_sq_std is None
            or self._cho is None
            or self._alpha is None
        ):
            raise RuntimeError("MultiScaleTransferGP must be fitted before querying")

    def posterior_cov(
        self,
        X: ArrayLike,
        *,
        target_scale: Any,
        observation_noise: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, cov)`` for ``target_scale`` in that scale's original units."""
        self._check_fitted()
        assert self._X_norm is not None
        assert self._lower is not None and self._upper is not None
        assert self._s_idx is not None
        assert self._scale_shift is not None and self._scale_factor is not None
        assert self._c_hat is not None and self._lengthscales is not None
        assert self._B_std is not None and self._sigma_sq_std is not None
        assert self._cho is not None and self._alpha is not None

        t_key = str(target_scale)
        if t_key not in self._scale_to_idx:
            raise ValueError(
                f"Unknown target_scale {target_scale!r}; known scales: {self._scale_names}"
            )
        t_idx = self._scale_to_idx[t_key]

        d = self._X_norm.shape[1]
        X_norm = _normalize_query(X, self._lower, self._upper, d)
        m = X_norm.shape[0]
        if m == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64)

        K_x_cross = _matern52_ard(X_norm, self._X_norm, self._lengthscales)
        K_x_test = _matern52_ard(X_norm, X_norm, self._lengthscales)

        B_cross = self._B_std[t_idx, self._s_idx]
        K_cross = K_x_cross * B_cross[None, :]
        K_test = K_x_test * float(self._B_std[t_idx, t_idx])

        mean_std = float(self._c_hat[t_idx]) + K_cross @ self._alpha
        v = cho_solve(self._cho, K_cross.T)
        cov_std = K_test - K_cross @ v
        if observation_noise:
            cov_std = cov_std + float(self._sigma_sq_std[t_idx]) * np.eye(m, dtype=np.float64)

        sf = float(self._scale_factor[t_idx])
        mean = float(self._scale_shift[t_idx]) + sf * mean_std
        cov = _ensure_psd(cov_std * (sf * sf))
        return mean, cov

    def posterior(
        self,
        X: ArrayLike,
        *,
        target_scale: Any,
        observation_noise: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(mean, std)`` for ``target_scale`` in that scale's original units."""
        self._check_fitted()
        assert self._scale_factor is not None
        mean, cov = self.posterior_cov(
            X, target_scale=target_scale, observation_noise=observation_noise
        )
        if mean.shape[0] == 0:
            return mean, np.empty(0, dtype=np.float64)
        t_idx = self._scale_to_idx[str(target_scale)]
        sf = float(self._scale_factor[t_idx])
        std = np.sqrt(np.maximum(np.diag(cov), _MIN_STD * (sf * sf)))
        return mean, std

    def transfer_correlation_matrix(self) -> tuple[list[str], NDArray[np.float64]]:
        """Return ``(scale_names, R)`` with ``R_{s,s'} = B_{s,s'} / sqrt(B_{s,s} B_{s',s'})``."""
        self._check_fitted()
        assert self._B_orig is not None
        diag = np.sqrt(np.maximum(np.diag(self._B_orig), _MIN_STD))
        R = self._B_orig / np.outer(diag, diag)
        R = 0.5 * (R + R.T)
        R = np.clip(R, -1.0, 1.0)
        np.fill_diagonal(R, 1.0)
        return list(self._scale_names), R

    @property
    def best_f(self) -> float:
        """Best observed target value across all scales in original units."""
        if self._y is None:
            raise RuntimeError("MultiScaleTransferGP must be fitted before accessing best_f")
        return float(self._y.max())


__all__ = ["HierarchicalBatchGP", "MultiScaleTransferGP"]
