"""Torch-free frontier acquisition primitives: Adaptive TuRBO and VOI Allocation.

Provides pure NumPy/SciPy implementations of:
  1. `TrustRegionState` and `AdaptiveTuRBO`: adaptive ARD-scaled trust-region
     Bayesian optimization using Joint Cholesky Thompson Sampling blended with
     Analytic Log-Expected Improvement and Kriging Believer pending repulsion.
  2. `VOIAllocator`: exact One-Step Knowledge Gradient batch allocator solving
     the wet-lab replicate-vs-explore trade-off via vectorized 15-point
     Gauss-Hermite quadrature and sequential Schur-complement fantasy updates.

Kept 100% torch-free at import and execution time. Accepts any surrogate
exposing `.posterior(X)` and optionally `.posterior_cov(X)` and
`.fit_diagnostics.get("lengthscales")`.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import erfcx, log_ndtr, ndtr
from scipy.stats.qmc import Sobol

from kalos.core.replicates import aggregate_replicates, estimate_noise_floor
from kalos.core.space_filling import (
    _linear_ok,
    _pairwise_separation,
    _sanitize_bounds_np,
    _sanitize_pending_np,
    _validate_linear_constraints,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ActionBatch",
    "ActionSlot",
    "AdaptiveTuRBO",
    "TrustRegionState",
    "VOIAllocator",
]

_TR_MIN_SIDE = 0.05
_TR_MAX_SIDE = 1.0
_VAR_FLOOR = 1e-12
_STD_FLOOR = 1e-12
_KB_NOISE_REL = 1e-6


def _log_ei_gaussian(
    mu: np.ndarray,
    std: np.ndarray,
    best_f: float,
    *,
    maximize: bool,
) -> np.ndarray:
    """Numerically stable analytic Log-Expected Improvement for Gaussian posteriors.

    Computes ``log(std * (phi(u) + u * Phi(u)))`` across all ``u`` without
    underflow to ``-inf`` in the deep negative tail (``u << -10``).
    """
    std_safe = np.maximum(np.asarray(std, dtype=float), _STD_FLOOR)
    mu_arr = np.asarray(mu, dtype=float)
    u = (mu_arr - float(best_f)) / std_safe if maximize else (float(best_f) - mu_arr) / std_safe

    log_h = np.empty_like(u, dtype=float)
    hi = u > -1.0
    mid = (~hi) & (u >= -25.0)
    lo = u < -25.0

    if np.any(hi):
        uh = u[hi]
        phi = np.exp(-0.5 * uh * uh) / math.sqrt(2.0 * math.pi)
        log_h[hi] = np.log(np.maximum(phi + uh * ndtr(uh), 1e-300))

    if np.any(mid):
        um = u[mid]
        log_phi = -0.5 * um * um - 0.5 * math.log(2.0 * math.pi)
        mills = math.sqrt(math.pi / 2.0) * erfcx(-um / math.sqrt(2.0))
        log_h[mid] = log_phi + np.log(np.maximum(1.0 + um * mills, 1e-300))

    if np.any(lo):
        ul = u[lo]
        log_phi = -0.5 * ul * ul - 0.5 * math.log(2.0 * math.pi)
        inv_u2 = 1.0 / (ul * ul)
        series = np.maximum(1.0 - 3.0 * inv_u2 + 15.0 * inv_u2 * inv_u2, 1e-300)
        log_h[lo] = log_phi - 2.0 * np.log(-ul) + np.log(series)

    return np.log(std_safe) + log_h


def _safe_cholesky(cov: np.ndarray) -> np.ndarray | None:
    """Compute a lower-triangular Cholesky factor with escalating jitter."""
    sym = 0.5 * (cov + cov.T)
    n = sym.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    diag_scale = max(1.0, float(np.mean(np.maximum(np.diag(sym), 0.0))))
    eye = np.eye(n, dtype=float)
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3):
        try:
            return np.linalg.cholesky(sym + (jitter * diag_scale) * eye)
        except np.linalg.LinAlgError:
            continue
    return None


def _extract_lengthscales(surrogate: Any, d: int) -> np.ndarray | None:
    """Extract positive finite ARD lengthscales of length ``d`` if available."""
    diag = getattr(surrogate, "fit_diagnostics", None)
    raw = diag.get("lengthscales") if isinstance(diag, dict) else None
    if raw is None:
        raw = getattr(surrogate, "lengthscales", None)
    if raw is None:
        return None
    ls = np.asarray(raw, dtype=float).reshape(-1)
    if ls.shape != (d,) or not np.all(np.isfinite(ls)) or not np.all(ls > 0.0):
        return None
    return ls


def _ard_matern52_corr(
    X1_norm: np.ndarray,
    X2_norm: np.ndarray,
    lengthscales: np.ndarray,
) -> np.ndarray:
    """ARD Matern-5/2 correlation matrix in normalized ``[0, 1]^d`` coordinates."""
    ls = np.maximum(lengthscales, 1e-6)
    diff = (X1_norm[:, None, :] - X2_norm[None, :, :]) / ls[None, None, :]
    r = np.sqrt(np.maximum(np.sum(diff * diff, axis=2), 0.0))
    s5r = math.sqrt(5.0) * r
    return (1.0 + s5r + (5.0 / 3.0) * r * r) * np.exp(-s5r)


def _query_surrogate_joint(
    surrogate: Any,
    X: np.ndarray,
    lower: np.ndarray,
    orig_width: np.ndarray,
    lengthscales: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Query surrogate posterior mean and covariance matrix.

    Uses ``surrogate.posterior_cov(X)`` when exposed and valid; otherwise builds
    a PSD correlation proxy from marginal ``surrogate.posterior(X)`` stds and
    an ARD Matern-5/2 correlation kernel so sequential Kriging Believer and KG
    updates remain coherent across all surrogates.
    """
    n, d = X.shape
    mean_raw, std_raw = surrogate.posterior(X)
    mean = np.asarray(mean_raw, dtype=float).reshape(-1)
    std = np.maximum(np.asarray(std_raw, dtype=float).reshape(-1), _STD_FLOOR)
    if mean.shape != (n,) or std.shape != (n,):
        raise ValueError(f"surrogate.posterior returned mismatched shapes for n={n}")

    cov_fn = getattr(surrogate, "posterior_cov", None)
    if callable(cov_fn):
        res = cov_fn(X)
        if isinstance(res, tuple) and len(res) == 2:
            mean_c, cov_raw = res
            mean_c_arr = np.asarray(mean_c, dtype=float).reshape(-1)
            if mean_c_arr.shape == (n,) and np.all(np.isfinite(mean_c_arr)):
                mean = mean_c_arr
        else:
            cov_raw = res
        cov = np.asarray(cov_raw, dtype=float)
        if cov.shape == (n, n) and np.all(np.isfinite(cov)):
            cov = 0.5 * (cov + cov.T)
            diag = np.maximum(np.diag(cov), _VAR_FLOOR)
            np.fill_diagonal(cov, diag)
            return mean, cov, True

    X_norm = (X - lower) / orig_width
    ls = lengthscales if lengthscales is not None else np.full(d, 0.2, dtype=float)
    corr = _ard_matern52_corr(X_norm, X_norm, ls)
    cov = (std[:, None] * corr) * std[None, :]
    cov = 0.5 * (cov + cov.T)
    np.fill_diagonal(cov, np.maximum(np.diag(cov), _VAR_FLOOR))
    return mean, cov, False


def _schur_condition(cov: np.ndarray, idx: int, obs_noise_var: float) -> np.ndarray:
    """Exact rank-1 posterior covariance update via Schur complement."""
    denom = max(float(cov[idx, idx]), _VAR_FLOOR) + max(float(obs_noise_var), 0.0)
    col = cov[:, idx : idx + 1]
    updated = cov - (col @ col.T) / denom
    updated = 0.5 * (updated + updated.T)
    np.fill_diagonal(updated, np.maximum(np.diag(updated), 0.0))
    return updated


def _prepare_constraints(
    bounds: Any,
    fixed_features: Mapping[int, float] | None,
    inequality_constraints: Sequence[tuple[Sequence[int], Sequence[float], float]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, float], list]:
    """Sanitize bounds, pin fixed features, and validate linear constraints."""
    lower, upper = _sanitize_bounds_np(bounds)
    d = int(lower.shape[0])
    orig_width = np.where((upper - lower) > 0.0, upper - lower, 1.0)

    fixed: dict[int, float] = {}
    lower_eff = lower.copy()
    upper_eff = upper.copy()
    for k, val in (fixed_features or {}).items():
        j = int(k)
        if not 0 <= j < d:
            raise ValueError("fixed_features names a dimension outside the design")
        fv = float(val)
        if not math.isfinite(fv):
            raise ValueError("fixed_features values must be finite")
        fixed[j] = fv
        lower_eff[j] = fv
        upper_eff[j] = fv

    lin = _validate_linear_constraints(inequality_constraints, lower_eff, upper_eff)
    return lower, upper, orig_width, lower_eff, upper_eff, fixed, lin


def _hit_and_run_polytope(
    n_target: int,
    d: int,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    fixed: dict[int, float],
    lin: list,
    seed: int,
) -> np.ndarray:
    """Sample interior feasible points in a constrained box via LP anchors + hit-and-run."""
    if not lin or n_target <= 0:
        return np.empty((0, d), dtype=float)

    a_mat = np.zeros((len(lin), d), dtype=float)
    b_vec = np.empty(len(lin), dtype=float)
    for r, (idx, coef, rhs) in enumerate(lin):
        a_mat[r, idx] = coef
        b_vec[r] = rhs

    rng = np.random.default_rng(int(seed))
    free_dims = [j for j in range(d) if j not in fixed and (box_upper[j] - box_lower[j]) > 0.0]
    bounds_list = list(zip(box_lower.tolist(), box_upper.tolist()))
    anchors: list[np.ndarray] = []

    n_probes = min(32, max(8, 2 * len(free_dims) + 2))
    for k in range(n_probes):
        c = np.zeros(d, dtype=float)
        if free_dims:
            if k < 2 * len(free_dims):
                dim_j = free_dims[k // 2]
                c[dim_j] = 1.0 if (k % 2 == 0) else -1.0
            else:
                c[free_dims] = rng.standard_normal(len(free_dims))
        res = linprog(c, A_ub=-a_mat, b_ub=-b_vec, bounds=bounds_list, method="highs")
        if res.success and res.x is not None:
            pt = np.clip(res.x, box_lower, box_upper)
            for j, v in fixed.items():
                pt[j] = v
            if _linear_ok(pt.reshape(1, -1), lin)[0]:
                anchors.append(pt)

    if not anchors:
        return np.empty((0, d), dtype=float)

    anchor_arr = np.vstack(anchors)
    if not free_dims:
        return anchor_arr[:1]

    x = np.mean(anchor_arr, axis=0)
    x = np.clip(x, box_lower, box_upper)
    for j, v in fixed.items():
        x[j] = v
    if not _linear_ok(x.reshape(1, -1), lin)[0]:
        x = anchor_arr[0].copy()

    samples: list[np.ndarray] = []
    # Mix across anchor convex combinations and hit-and-run chord steps.
    n_burn = 16
    total_steps = n_target + n_burn
    for step in range(total_steps):
        if len(anchors) >= 2 and step % 4 == 0:
            w = rng.dirichlet(np.ones(len(anchors), dtype=float))
            x_try = np.clip(w @ anchor_arr, box_lower, box_upper)
            for j, v in fixed.items():
                x_try[j] = v
            if _linear_ok(x_try.reshape(1, -1), lin)[0]:
                x = x_try

        direction = np.zeros(d, dtype=float)
        direction[free_dims] = rng.standard_normal(len(free_dims))
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        direction /= norm

        t_min = -np.inf
        t_max = np.inf
        for j in free_dims:
            vj = float(direction[j])
            if abs(vj) <= 1e-14:
                continue
            t0 = (box_lower[j] - x[j]) / vj
            t1 = (box_upper[j] - x[j]) / vj
            lo_t, hi_t = (t0, t1) if t0 <= t1 else (t1, t0)
            t_min = max(t_min, lo_t)
            t_max = min(t_max, hi_t)

        av = a_mat @ direction
        slack = b_vec - a_mat @ x
        for r in range(len(lin)):
            avr = float(av[r])
            if avr > 1e-12:
                t_min = max(t_min, float(slack[r]) / avr)
            elif avr < -1e-12:
                t_max = min(t_max, float(slack[r]) / avr)

        if math.isfinite(t_min) and math.isfinite(t_max) and t_max > t_min + 1e-12:
            t_step = float(rng.uniform(t_min, t_max))
            x_next = np.clip(x + t_step * direction, box_lower, box_upper)
            for j, v in fixed.items():
                x_next[j] = v
            if _linear_ok(x_next.reshape(1, -1), lin)[0]:
                x = x_next

        if step >= n_burn:
            samples.append(x.copy())

    return np.vstack(samples) if samples else anchor_arr[:n_target]


def _sample_feasible_box(
    n_target: int,
    d: int,
    box_lower: np.ndarray,
    box_upper: np.ndarray,
    fixed: dict[int, float],
    lin: list,
    seed: int,
) -> np.ndarray:
    """Generate deterministic scrambled Sobol points inside ``[box_lower, box_upper]``
    respecting ``fixed`` features and linear ``inequality_constraints``."""
    if n_target <= 0:
        return np.empty((0, d), dtype=float)

    free_dims = [j for j in range(d) if j not in fixed and (box_upper[j] - box_lower[j]) > 0.0]
    if not free_dims:
        pt = box_lower.copy()
        for j, v in fixed.items():
            pt[j] = v
        pt = np.clip(pt, box_lower, box_upper).reshape(1, d)
        return pt if _linear_ok(pt, lin)[0] else np.empty((0, d), dtype=float)

    if not lin:
        m = max(1, int(math.ceil(math.log2(max(n_target, 2)))))
        u = Sobol(d=len(free_dims), scramble=True, seed=int(seed)).random_base2(m=m)[:n_target]
        cand = np.broadcast_to(box_lower, (u.shape[0], d)).copy()
        for j, v in fixed.items():
            cand[:, j] = v
        for k, dim_j in enumerate(free_dims):
            cand[:, dim_j] = box_lower[dim_j] + u[:, k] * (box_upper[dim_j] - box_lower[dim_j])
        return np.clip(cand, box_lower, box_upper)

    min_m = max(10, int(math.ceil(math.log2(max(n_target * 4, 1024)))))
    collected: list[np.ndarray] = []
    n_collected = 0
    for m in range(min_m, 15):
        u = Sobol(d=len(free_dims), scramble=True, seed=int(seed) + m).random_base2(m=m)
        cand = np.broadcast_to(box_lower, (u.shape[0], d)).copy()
        for j, v in fixed.items():
            cand[:, j] = v
        for k, dim_j in enumerate(free_dims):
            cand[:, dim_j] = box_lower[dim_j] + u[:, k] * (box_upper[dim_j] - box_lower[dim_j])
        cand = np.clip(cand, box_lower, box_upper)
        feas = cand[_linear_ok(cand, lin)]
        if len(feas) > 0:
            collected.append(feas)
            n_collected += len(feas)
        if n_collected >= n_target:
            break

    if n_collected >= max(n_target // 2, 1):
        return np.vstack(collected)[:n_target]

    poly = _hit_and_run_polytope(n_target, d, box_lower, box_upper, fixed, lin, seed=seed)
    if collected and len(poly) > 0:
        return np.vstack([*collected, poly])[:n_target]
    if collected:
        return np.vstack(collected)[:n_target]
    return poly[:n_target]


def _sanitize_xy(
    X_train: Any,
    y_train: Any,
    d: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return finite aligned ``(X_arr, y_arr)`` or ``None`` when absent."""
    if X_train is None:
        return None, None
    Xa = np.asarray(X_train, dtype=float)
    if Xa.ndim == 1 and Xa.size == d:
        Xa = Xa.reshape(1, d)
    if Xa.ndim != 2 or Xa.shape[0] == 0 or Xa.shape[1] != d:
        return None, None

    valid_x = np.isfinite(Xa).all(axis=1)
    if y_train is None:
        Xa_clean = Xa[valid_x]
        return (Xa_clean if len(Xa_clean) > 0 else None), None

    ya = np.asarray(y_train, dtype=float).reshape(-1)
    if ya.shape[0] != Xa.shape[0]:
        raise ValueError("X_train and y_train must have the same number of rows")
    valid = valid_x & np.isfinite(ya)
    Xa_clean = Xa[valid]
    ya_clean = ya[valid]
    if len(Xa_clean) == 0:
        return None, None
    return Xa_clean, ya_clean


def _resolve_tr_center_and_sides(
    d: int,
    lower: np.ndarray,
    upper: np.ndarray,
    orig_width: np.ndarray,
    lower_eff: np.ndarray,
    upper_eff: np.ndarray,
    fixed: dict[int, float],
    lin: list,
    base_length: float,
    lengthscales: np.ndarray | None,
    X_tr: np.ndarray | None,
    y_tr: np.ndarray | None,
    surrogate: Any,
    incumbent: Any,
    maximize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ARD-scaled trust-region bounds ``(tr_lower, tr_upper, side_lengths)``."""
    if lengthscales is not None:
        ls_tr = np.minimum(lengthscales, 2.0)
        geomean = float(np.exp(np.mean(np.log(ls_tr))))
        side_lengths = np.clip(base_length * (ls_tr / geomean), _TR_MIN_SIDE, _TR_MAX_SIDE)
    else:
        side_lengths = np.full(d, float(np.clip(base_length, _TR_MIN_SIDE, _TR_MAX_SIDE)), dtype=float)

    center: np.ndarray | None = None
    if incumbent is not None:
        inc = np.asarray(incumbent, dtype=float).reshape(-1)
        if inc.shape != (d,) or not np.all(np.isfinite(inc)):
            raise ValueError(f"incumbent must have shape ({d},) with finite entries")
        center = np.clip(inc, lower_eff, upper_eff)
    elif X_tr is not None and len(X_tr) > 0:
        in_box = np.all((X_tr >= lower - 1e-9) & (X_tr <= upper + 1e-9), axis=1)
        feas_mask = in_box & _linear_ok(X_tr, lin)
        for j, v in fixed.items():
            feas_mask &= np.abs(X_tr[:, j] - v) <= 1e-6 * max(1.0, abs(v))
        pool_idx = np.flatnonzero(feas_mask) if np.any(feas_mask) else np.arange(len(X_tr))
        if y_tr is not None:
            sub_y = y_tr[pool_idx]
            rel = int(np.argmax(sub_y)) if maximize else int(np.argmin(sub_y))
            center = X_tr[pool_idx[rel]].copy()
        else:
            sub_x = X_tr[pool_idx]
            mu_tr, _ = surrogate.posterior(sub_x)
            mu_arr = np.asarray(mu_tr, dtype=float).reshape(-1)
            rel = int(np.argmax(mu_arr)) if maximize else int(np.argmin(mu_arr))
            center = sub_x[rel].copy()

    if center is None:
        center_norm = np.full(d, 0.5, dtype=float)
    else:
        center_norm = np.clip((center - lower) / orig_width, 0.0, 1.0)

    for j, v in fixed.items():
        center_norm[j] = float(np.clip((v - lower[j]) / orig_width[j], 0.0, 1.0))

    tr_lo_norm = np.clip(center_norm - 0.5 * side_lengths, 0.0, 1.0)
    tr_hi_norm = np.clip(center_norm + 0.5 * side_lengths, 0.0, 1.0)
    tr_lower = np.maximum(lower_eff, lower + tr_lo_norm * orig_width)
    tr_upper = np.minimum(upper_eff, lower + tr_hi_norm * orig_width)
    for j, v in fixed.items():
        tr_lower[j] = v
        tr_upper[j] = v
    return tr_lower, tr_upper, side_lengths


def _zscore_1d(vals: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalization with zero guard for constant arrays."""
    sd = float(np.std(vals))
    if not math.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(vals, dtype=float)
    return (vals - float(np.mean(vals))) / sd


def _polish_slot_analytic_lbfgs(
    surrogate: Any,
    x_start: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
    orig_width: np.ndarray,
    fixed: dict[int, float],
    best_std: float,
    maximize: bool,
    repel_points: list[np.ndarray],
    lengthscales: np.ndarray | None,
    refs: np.ndarray | None,
    min_separation: float,
) -> np.ndarray:
    """Polish a single slot candidate via exact-gradient L-BFGS-B in warped GP space."""
    d = int(x_start.shape[0])
    free_idx = [j for j in range(d) if j not in fixed]
    if not free_idx:
        return x_start

    x_base = np.asarray(x_start, dtype=float).copy()
    for j, v in fixed.items():
        x_base[j] = float(v)
    x_base = np.clip(x_base, box_lo, box_hi)
    for j, v in fixed.items():
        x_base[j] = float(v)

    inv_w = 1.0 / np.maximum(orig_width, 1e-12)
    if lengthscales is not None:
        rho_repel = 1.0 / (np.maximum(np.asarray(lengthscales, dtype=float), 1e-3) ** 2)
    else:
        rho_repel = np.ones(d, dtype=float)
    repel_arr = [np.asarray(r, dtype=float).reshape(-1) for r in repel_points]
    sign = 1.0 if maximize else -1.0
    sqrt5 = float(np.sqrt(5.0))
    half_log2pi = 0.5 * float(np.log(2.0 * np.pi))

    def _eval(x_free: np.ndarray) -> tuple[float, np.ndarray]:
        x_full = x_base.copy()
        x_full[free_idx] = x_free
        mu, var, dmu_dx, dvar_dx = surrogate.posterior_warped_and_grad(x_full)
        var_cur = float(var)
        dvar_cur = np.asarray(dvar_dx, dtype=float).copy()

        for xr in repel_arr:
            diff_u = (x_full - xr) * inv_w
            r2 = float(np.sum((diff_u * diff_u) * rho_repel))
            r = math.sqrt(max(r2, 1e-18))
            sr = sqrt5 * r
            e_sr = math.exp(-sr)
            k_corr = (1.0 + sr + (5.0 / 3.0) * r2) * e_sr
            mult = max(1.0 - 0.98 * k_corr * k_corr, 1e-4)
            if mult > 1e-4:
                dk_dx = -(5.0 / 3.0) * (1.0 + sr) * e_sr * rho_repel * diff_u * inv_w
                dmult_dx = -1.96 * k_corr * dk_dx
                dvar_cur = dvar_cur * mult + var_cur * dmult_dx
            else:
                dvar_cur = dvar_cur * mult
            var_cur = max(var_cur * mult, 1e-10)

        sd = math.sqrt(max(var_cur, 1e-10))
        dsd_dx = 0.5 * dvar_cur / sd
        z = sign * (float(mu) - best_std) / sd
        log_phi = -0.5 * z * z - half_log2pi
        if z > -6.0:
            Phi = float(np.exp(float(log_ndtr(z))))
            phi = float(np.exp(log_phi))
            ei = max(sd * (z * Phi + phi), 1e-30)
            val = float(np.log(ei))
            dval_dmu = sign * Phi / ei
            dval_dsd = phi / ei
        else:
            val = float(math.log(sd) + log_phi - 2.0 * math.log(max(-z, 1.0)))
            dval_dz = -z - 2.0 / z
            dval_dmu = sign * dval_dz / sd
            dval_dsd = (1.0 - z * dval_dz) / sd

        grad = dval_dmu * np.asarray(dmu_dx, dtype=float) + dval_dsd * dsd_dx
        return -val, -grad[free_idx]

    lbfgs_bounds = [(float(box_lo[j]), float(box_hi[j])) for j in free_idx]
    f0, _ = _eval(x_base[free_idx])
    res = minimize(
        _eval,
        x_base[free_idx],
        method="L-BFGS-B",
        jac=True,
        bounds=lbfgs_bounds,
        options={"maxiter": 20},
    )
    if not np.all(np.isfinite(res.x)) or float(res.fun) > f0:
        return x_start

    x_cand = x_base.copy()
    x_cand[free_idx] = res.x
    x_cand = np.clip(x_cand, box_lo, box_hi)
    for j, v in fixed.items():
        x_cand[j] = float(v)

    if min_separation > 0.0 and refs is not None and len(refs) > 0:
        sep = float(
            _pairwise_separation(
                x_cand.reshape(1, -1), refs, list(range(d)), [], orig_width
            ).min()
        )
        if sep < min_separation - 1e-12:
            return x_start
    return x_cand


class TrustRegionState:
    """State machine tracking trust-region side length, streaks, and restarts."""

    def __init__(
        self,
        bounds: Any,
        *,
        base_length: float = 0.8,
        min_length: float = 0.05,
        max_length: float = 1.0,
        success_tol: int = 3,
        failure_tol: int = 3,
        maximize: bool = True,
        best_value: float | None = None,
    ) -> None:
        lower, upper = _sanitize_bounds_np(bounds)
        self.bounds = np.vstack([lower, upper])
        self.dim = int(lower.shape[0])
        if not (0.0 < float(min_length) <= float(base_length) <= float(max_length)):
            raise ValueError("require 0 < min_length <= base_length <= max_length")
        if int(success_tol) < 1 or int(failure_tol) < 1:
            raise ValueError("success_tol and failure_tol must be >= 1")
        self.base_length = float(base_length)
        self.min_length = float(min_length)
        self.max_length = float(max_length)
        self.success_tol = int(success_tol)
        self.failure_tol = int(failure_tol)
        self.maximize = bool(maximize)
        self.length: float = float(base_length)
        self.success_count: int = 0
        self.failure_count: int = 0
        if best_value is None:
            self.best_value: float = float("-inf") if self.maximize else float("inf")
        else:
            bv = float(best_value)
            if not math.isfinite(bv):
                raise ValueError("best_value must be finite when provided")
            self.best_value = bv
        self.n_restarts: int = 0

    def update(self, best_observed: float) -> TrustRegionState:
        """Update trust-region length from the latest observed incumbent value."""
        val = float(best_observed)
        if not math.isfinite(val):
            raise ValueError("best_observed must be finite")

        improved = (val > self.best_value) if self.maximize else (val < self.best_value)
        if improved:
            self.best_value = val
            self.success_count += 1
            self.failure_count = 0
        else:
            self.success_count = 0
            self.failure_count += 1

        if self.success_count >= self.success_tol:
            self.length = min(self.max_length, self.length * 1.6)
            self.success_count = 0
        elif self.failure_count >= self.failure_tol:
            self.length *= 0.5
            self.failure_count = 0

        if self.length < self.min_length:
            self.length = self.base_length
            self.success_count = 0
            self.failure_count = 0
            self.n_restarts += 1
        return self


class AdaptiveTuRBO:
    """ARD-scaled Trust Region Bayesian Optimization over scrambled Sobol pools.

    Combines Joint Cholesky Thompson Sampling (when ``posterior_cov`` is
    available) with Analytic Log-Expected Improvement and Kriging Believer
    Schur-complement repulsion + Chebyshev ``min_separation`` enforcement.
    """

    def __init__(
        self,
        n_candidates: int = 512,
        ts_weight: float = 0.25,
        global_frac: float = 0.15,
        min_separation: float = 0.05,
    ) -> None:
        if int(n_candidates) < 1:
            raise ValueError("n_candidates must be >= 1")
        if not (0.0 <= float(ts_weight) <= 1.0):
            raise ValueError("ts_weight must be in [0, 1]")
        if not (0.0 <= float(global_frac) <= 1.0):
            raise ValueError("global_frac must be in [0, 1]")
        if float(min_separation) < 0.0:
            raise ValueError("min_separation must be >= 0")
        self.n_candidates = int(n_candidates)
        self.ts_weight = float(ts_weight)
        self.global_frac = float(global_frac)
        self.min_separation = float(min_separation)

    def propose(
        self,
        surrogate: Any,
        bounds: Any,
        q: int = 5,
        *,
        X_train: Any = None,
        y_train: Any = None,
        pending: Any = None,
        fixed_features: Mapping[int, float] | None = None,
        inequality_constraints: Sequence[tuple[Sequence[int], Sequence[float], float]] | None = None,
        tr_state: TrustRegionState | None = None,
        incumbent: Any = None,
        maximize: bool = True,
        seed: int = 0,
    ) -> np.ndarray:
        """Propose a batch of up to ``q`` points inside ``bounds``."""
        lower, upper, orig_width, lower_eff, upper_eff, fixed, lin = _prepare_constraints(
            bounds, fixed_features, inequality_constraints
        )
        d = int(lower.shape[0])
        q_int = max(int(q), 0)
        if q_int == 0:
            return np.empty((0, d), dtype=float)

        X_tr, y_tr = _sanitize_xy(X_train, y_train, d)
        pend = _sanitize_pending_np(pending, lower, upper, [])
        base_length = float(tr_state.length) if tr_state is not None else 0.8
        lengthscales = _extract_lengthscales(surrogate, d)

        tr_lower, tr_upper, _ = _resolve_tr_center_and_sides(
            d=d,
            lower=lower,
            upper=upper,
            orig_width=orig_width,
            lower_eff=lower_eff,
            upper_eff=upper_eff,
            fixed=fixed,
            lin=lin,
            base_length=base_length,
            lengthscales=lengthscales,
            X_tr=X_tr,
            y_tr=y_tr,
            surrogate=surrogate,
            incumbent=incumbent,
            maximize=maximize,
        )

        n_gl = int(round(self.n_candidates * self.global_frac))
        n_tr_cand = max(self.n_candidates - n_gl, 0)
        cand_tr = _sample_feasible_box(n_tr_cand, d, tr_lower, tr_upper, fixed, lin, seed=seed)
        cand_gl = _sample_feasible_box(
            n_gl, d, lower_eff, upper_eff, fixed, lin, seed=int(seed) + 1_000_003
        )
        parts = [p for p in (cand_tr, cand_gl) if len(p) > 0]
        if not parts:
            return np.empty((0, d), dtype=float)

        X_pool = np.clip(np.vstack(parts), lower_eff, upper_eff)
        n_tr_pts = int(len(cand_tr))
        n_pool = int(X_pool.shape[0])

        X_eval = np.vstack([X_pool, pend]) if pend is not None else X_pool
        mu_eval, cov_eval, has_joint_cov = _query_surrogate_joint(
            surrogate, X_eval, lower, orig_width, lengthscales
        )
        if pend is not None:
            for p_idx in range(n_pool, int(X_eval.shape[0])):
                kb_noise = _KB_NOISE_REL * max(float(cov_eval[p_idx, p_idx]), _VAR_FLOOR)
                cov_eval = _schur_condition(cov_eval, p_idx, kb_noise)

        mu = mu_eval[:n_pool]
        cov = cov_eval[:n_pool, :n_pool]

        if y_tr is not None and len(y_tr) > 0:
            best_f = float(np.max(y_tr)) if maximize else float(np.min(y_tr))
        else:
            best_f = float(np.max(mu)) if maximize else float(np.min(mu))

        can_polish = (
            not lin
            and self.ts_weight < 1.0
            and callable(getattr(surrogate, "posterior_warped_and_grad", None))
            and callable(getattr(surrogate, "warped_posterior_mean_std", None))
        )
        best_std = 0.0
        if can_polish:
            ref_pts = X_tr if (X_tr is not None and len(X_tr) > 0) else X_pool
            mu_w, _ = surrogate.warped_posterior_mean_std(ref_pts)
            mu_w_arr = np.asarray(mu_w, dtype=float).reshape(-1)
            best_std = float(np.max(mu_w_arr)) if maximize else float(np.min(mu_w_arr))

        cont = list(range(d))
        refs_parts = [r for r in (X_tr, pend) if r is not None and len(r) > 0]
        if refs_parts:
            min_dist = _pairwise_separation(
                X_pool, np.vstack(refs_parts), cont, [], orig_width
            ).min(axis=1)
        else:
            min_dist = np.full(n_pool, np.inf, dtype=float)

        rng = np.random.default_rng(int(seed))
        selected: list[np.ndarray] = []
        used = np.zeros(n_pool, dtype=bool)

        for _ in range(q_int):
            eligible = ~used
            if self.min_separation > 0.0:
                eligible &= min_dist >= self.min_separation - 1e-12
            if not np.any(eligible):
                break

            std_cur = np.sqrt(np.maximum(np.diag(cov), _VAR_FLOOR))
            log_ei = _log_ei_gaussian(mu, std_cur, best_f, maximize=maximize)

            chol = _safe_cholesky(cov) if has_joint_cov else None
            if chol is not None:
                ts_sample = mu + chol @ rng.standard_normal(n_pool)
            else:
                ts_sample = mu + std_cur * rng.standard_normal(n_pool)
            ts_score = ts_sample if maximize else -ts_sample

            scores = np.full(n_pool, -np.inf, dtype=float)
            if self.ts_weight <= 0.0:
                scores[eligible] = log_ei[eligible]
            elif self.ts_weight >= 1.0:
                scores[eligible] = ts_score[eligible]
            else:
                z_ts = _zscore_1d(ts_score[eligible])
                z_ei = _zscore_1d(log_ei[eligible])
                scores[eligible] = self.ts_weight * z_ts + (1.0 - self.ts_weight) * z_ei

            best_idx = int(np.argmax(scores))
            chosen = np.clip(X_pool[best_idx].copy(), lower_eff, upper_eff)
            if can_polish:
                box_lo, box_hi = (
                    (tr_lower, tr_upper) if best_idx < n_tr_pts else (lower_eff, upper_eff)
                )
                repel_pts = list(selected)
                if pend is not None:
                    repel_pts.extend(pend[k] for k in range(len(pend)))
                refs_list = list(refs_parts) + ([np.vstack(selected)] if selected else [])
                refs_cur = np.vstack(refs_list) if refs_list else None
                chosen = _polish_slot_analytic_lbfgs(
                    surrogate=surrogate,
                    x_start=chosen,
                    box_lo=box_lo,
                    box_hi=box_hi,
                    orig_width=orig_width,
                    fixed=fixed,
                    best_std=best_std,
                    maximize=maximize,
                    repel_points=repel_pts,
                    lengthscales=lengthscales,
                    refs=refs_cur,
                    min_separation=self.min_separation,
                )
            selected.append(chosen)
            used[best_idx] = True

            d_new = _pairwise_separation(
                X_pool, chosen.reshape(1, -1), cont, [], orig_width
            ).reshape(-1)
            min_dist = np.minimum(min_dist, d_new)

            kb_noise = _KB_NOISE_REL * max(float(cov[best_idx, best_idx]), _VAR_FLOOR)
            cov = _schur_condition(cov, best_idx, kb_noise)

        if not selected:
            return np.empty((0, d), dtype=float)
        return np.clip(np.vstack(selected), lower_eff, upper_eff)


@dataclass(frozen=True)
class ActionSlot:
    """Single selected experiment slot in a Value-of-Information batch."""

    kind: Literal["explore", "replicate"]
    x: np.ndarray
    expected_voi: float
    predicted_mean: float
    predicted_std: float
    reason: str


@dataclass(frozen=True)
class ActionBatch:
    """Ordered batch of explore and replicate actions selected by ``VOIAllocator``."""

    slots: tuple[ActionSlot, ...]
    _dim: int = field(default=0, repr=False, compare=False)

    @property
    def X(self) -> np.ndarray:
        if not self.slots:
            return np.empty((0, self._dim), dtype=float)
        return np.vstack([np.asarray(slot.x, dtype=float).reshape(1, -1) for slot in self.slots])


class VOIAllocator:
    """One-Step Knowledge Gradient batch allocator for replicate-vs-explore decisions.

    Evaluates both unexplored Sobol/TuRBO candidates (``explore``) and unique
    observed recipes (``replicate``) using exact rank-1 posterior covariance
    updates and vectorized 15-point Gauss-Hermite quadrature, conditioning the
    fantasy posterior covariance sequentially across all ``q`` slots.
    """

    def __init__(
        self,
        n_candidates: int = 256,
        n_reference: int = 128,
        min_separation: float = 0.05,
        n_quadrature: int = 15,
        replicate_dedup_tol: float = 1e-4,
    ) -> None:
        if int(n_candidates) < 1:
            raise ValueError("n_candidates must be >= 1")
        if int(n_reference) < 1:
            raise ValueError("n_reference must be >= 1")
        if float(min_separation) < 0.0:
            raise ValueError("min_separation must be >= 0")
        if int(n_quadrature) < 3:
            raise ValueError("n_quadrature must be >= 3")
        self.n_candidates = int(n_candidates)
        self.n_reference = int(n_reference)
        self.min_separation = float(min_separation)
        self.n_quadrature = int(n_quadrature)
        self.replicate_dedup_tol = float(replicate_dedup_tol)

    @staticmethod
    def _resolve_noise_var(
        surrogate: Any,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        assay_noise_var: float | None,
    ) -> float:
        if assay_noise_var is not None:
            nv = float(assay_noise_var)
            if not math.isfinite(nv) or nv < 0.0:
                raise ValueError("assay_noise_var must be finite and non-negative")
            return nv

        nf = estimate_noise_floor(X_tr, y_tr)
        if math.isfinite(nf) and nf >= 0.0:
            return float(nf)

        diag = getattr(surrogate, "fit_diagnostics", None)
        if isinstance(diag, dict):
            if diag.get("noise_var") is not None:
                nv = float(diag["noise_var"])
                if math.isfinite(nv) and nv >= 0.0:
                    return nv
            if diag.get("noise_fitted") is not None:
                nf_std = float(diag["noise_fitted"])
                if math.isfinite(nf_std) and nf_std >= 0.0:
                    return nf_std * max(float(np.var(y_tr)), _VAR_FLOOR)

        return 1e-4 * max(float(np.var(y_tr)), _VAR_FLOOR)

    def _compute_kg(
        self,
        mu_A_eff: np.ndarray,
        base_max: float,
        cov_all: np.ndarray,
        idx_A: slice,
        cand_indices: np.ndarray,
        noise_var: float,
        z_nodes: np.ndarray,
        z_weights: np.ndarray,
        maximize: bool,
    ) -> np.ndarray:
        """Vectorized 15-point Gauss-Hermite One-Step Knowledge Gradient."""
        n_c = int(cand_indices.shape[0])
        if n_c == 0:
            return np.empty(0, dtype=float)

        diag_cov = np.diag(cov_all)
        var_c = np.maximum(diag_cov[cand_indices], _VAR_FLOOR)
        denom = np.sqrt(var_c + noise_var)
        cov_Ac = cov_all[idx_A][:, cand_indices]
        sigma_tilde = cov_Ac / denom[None, :]
        if not maximize:
            sigma_tilde = -sigma_tilde

        kg_out = np.empty(n_c, dtype=float)
        chunk = 64
        for start in range(0, n_c, chunk):
            end = min(start + chunk, n_c)
            # Shape: (M, C_chunk, K)
            val = (
                mu_A_eff[:, None, None]
                + sigma_tilde[:, start:end, None] * z_nodes[None, None, :]
            )
            max_z = val.max(axis=0)  # (C_chunk, K)
            kg_out[start:end] = np.maximum(max_z @ z_weights - base_max, 0.0)
        return kg_out

    def allocate(
        self,
        surrogate: Any,
        X_train: Any,
        y_train: Any,
        bounds: Any,
        q: int = 5,
        *,
        assay_noise_var: float | None = None,
        pending: Any = None,
        fixed_features: Mapping[int, float] | None = None,
        inequality_constraints: Sequence[tuple[Sequence[int], Sequence[float], float]] | None = None,
        maximize: bool = True,
        seed: int = 0,
    ) -> ActionBatch:
        """Allocate up to ``q`` slots across ``explore`` and ``replicate`` actions."""
        lower, upper, orig_width, lower_eff, upper_eff, fixed, lin = _prepare_constraints(
            bounds, fixed_features, inequality_constraints
        )
        d = int(lower.shape[0])
        q_int = max(int(q), 0)

        X_tr, y_tr = _sanitize_xy(X_train, y_train, d)
        if X_tr is None or y_tr is None:
            raise ValueError("X_train and y_train must be non-empty finite arrays matching bounds")
        if q_int == 0:
            return ActionBatch(slots=(), _dim=d)

        pend = _sanitize_pending_np(pending, lower, upper, [])
        noise_var = self._resolve_noise_var(surrogate, X_tr, y_tr, assay_noise_var)
        noise_sd = math.sqrt(max(noise_var, 0.0))
        lengthscales = _extract_lengthscales(surrogate, d)

        # Unique observed recipes for replicate candidates and reference grid.
        X_unique, y_mean, _, n_reps = aggregate_replicates(X_tr, y_tr)
        in_box = np.all((X_unique >= lower - 1e-9) & (X_unique <= upper + 1e-9), axis=1)
        feas_rep = in_box & _linear_ok(X_unique, lin)
        for j, v in fixed.items():
            feas_rep &= np.abs(X_unique[:, j] - v) <= 1e-6 * max(1.0, abs(v))

        X_rep_feas = np.clip(X_unique[feas_rep], lower_eff, upper_eff)
        y_rep_feas = y_mean[feas_rep]
        n_rep_feas = n_reps[feas_rep]
        if len(X_rep_feas) > 0:
            order = np.argsort(-y_rep_feas if maximize else y_rep_feas)
            X_rep_feas = X_rep_feas[order]
            y_rep_feas = y_rep_feas[order]
            n_rep_feas = n_rep_feas[order]

        cont = list(range(d))
        if pend is not None and len(X_rep_feas) > 0:
            d_pend = _pairwise_separation(X_rep_feas, pend, cont, [], orig_width).min(axis=1)
            keep_rep = d_pend > self.replicate_dedup_tol
            X_rep_cand = X_rep_feas[keep_rep]
            n_rep_cand = n_rep_feas[keep_rep]
        else:
            X_rep_cand = X_rep_feas
            n_rep_cand = n_rep_feas

        # Build TuRBO-style Sobol candidates for explore pool and reference set A.
        tr_lower, tr_upper, _ = _resolve_tr_center_and_sides(
            d=d,
            lower=lower,
            upper=upper,
            orig_width=orig_width,
            lower_eff=lower_eff,
            upper_eff=upper_eff,
            fixed=fixed,
            lin=lin,
            base_length=0.8,
            lengthscales=lengthscales,
            X_tr=X_tr,
            y_tr=y_tr,
            surrogate=surrogate,
            incumbent=None,
            maximize=maximize,
        )
        n_pool_target = max(self.n_candidates, self.n_reference)
        n_gl = int(round(n_pool_target * 0.15))
        n_tr_c = max(n_pool_target - n_gl, 0)
        cand_tr = _sample_feasible_box(n_tr_c, d, tr_lower, tr_upper, fixed, lin, seed=seed)
        cand_gl = _sample_feasible_box(
            n_gl, d, lower_eff, upper_eff, fixed, lin, seed=int(seed) + 1_000_003
        )
        cand_parts = [p for p in (cand_tr, cand_gl) if len(p) > 0]
        X_sobol = (
            np.clip(np.vstack(cand_parts), lower_eff, upper_eff)
            if cand_parts
            else np.empty((0, d), dtype=float)
        )

        # Reference discretization A = top unique feasible recipes + Sobol reference points.
        ref_parts: list[np.ndarray] = []
        if len(X_rep_feas) > 0:
            ref_parts.append(X_rep_feas[: min(len(X_rep_feas), 64)])
        if len(X_sobol) > 0:
            ref_parts.append(X_sobol[: self.n_reference])
        if not ref_parts:
            return ActionBatch(slots=(), _dim=d)
        X_ref = np.vstack(ref_parts)

        # Filter explore pool against X_train union pending.
        refs_init = [X_tr]
        if pend is not None:
            refs_init.append(pend)
        ref_init_arr = np.vstack(refs_init)
        if len(X_sobol) > 0:
            min_dist_exp = _pairwise_separation(
                X_sobol, ref_init_arr, cont, [], orig_width
            ).min(axis=1)
            if self.min_separation > 0.0:
                keep_exp = min_dist_exp >= self.min_separation - 1e-12
                X_exp_cand = X_sobol[keep_exp]
                min_dist_exp = min_dist_exp[keep_exp]
            else:
                X_exp_cand = X_sobol
        else:
            X_exp_cand = np.empty((0, d), dtype=float)
            min_dist_exp = np.empty(0, dtype=float)

        n_A = int(X_ref.shape[0])
        n_rep = int(X_rep_cand.shape[0])
        n_exp = int(X_exp_cand.shape[0])
        if n_rep == 0 and n_exp == 0:
            return ActionBatch(slots=(), _dim=d)

        eval_blocks = [X_ref]
        if n_rep > 0:
            eval_blocks.append(X_rep_cand)
        if n_exp > 0:
            eval_blocks.append(X_exp_cand)
        if pend is not None:
            eval_blocks.append(pend)
        X_all = np.vstack(eval_blocks)

        mu_all, cov_all, _ = _query_surrogate_joint(
            surrogate, X_all, lower, orig_width, lengthscales
        )
        idx_pend_start = n_A + n_rep + n_exp
        for p_idx in range(idx_pend_start, int(X_all.shape[0])):
            cov_all = _schur_condition(cov_all, p_idx, noise_var)

        idx_A = slice(0, n_A)
        idx_rep = np.arange(n_A, n_A + n_rep, dtype=int)
        idx_exp = np.arange(n_A + n_rep, n_A + n_rep + n_exp, dtype=int)

        gh_t, gh_w = np.polynomial.hermite.hermgauss(self.n_quadrature)
        z_nodes = np.sqrt(2.0) * gh_t
        z_weights = gh_w / math.sqrt(math.pi)

        mu_A_eff = mu_all[idx_A] if maximize else -mu_all[idx_A]
        base_max = float(np.max(mu_A_eff))

        eligible_rep = np.ones(n_rep, dtype=bool)
        eligible_exp = np.ones(n_exp, dtype=bool)
        slots: list[ActionSlot] = []

        for _ in range(q_int):
            active_rep_pos = np.flatnonzero(eligible_rep)
            active_exp_pos = np.flatnonzero(eligible_exp)
            if len(active_rep_pos) == 0 and len(active_exp_pos) == 0:
                break

            rep_cands = idx_rep[active_rep_pos]
            exp_cands = idx_exp[active_exp_pos]

            kg_rep = self._compute_kg(
                mu_A_eff, base_max, cov_all, idx_A, rep_cands, noise_var, z_nodes, z_weights, maximize
            )
            kg_exp = self._compute_kg(
                mu_A_eff, base_max, cov_all, idx_A, exp_cands, noise_var, z_nodes, z_weights, maximize
            )

            best_rep_local = int(np.argmax(kg_rep)) if len(kg_rep) > 0 else -1
            best_exp_local = int(np.argmax(kg_exp)) if len(kg_exp) > 0 else -1
            best_rep_kg = float(kg_rep[best_rep_local]) if best_rep_local >= 0 else -1.0
            best_exp_kg = float(kg_exp[best_exp_local]) if best_exp_local >= 0 else -1.0

            choose_replicate = False
            if best_rep_local >= 0 and (
                best_exp_local < 0 or (best_rep_kg > best_exp_kg and best_rep_kg > 0.0)
            ):
                choose_replicate = True

            if choose_replicate:
                pos = int(active_rep_pos[best_rep_local])
                chosen_idx = int(idx_rep[pos])
                chosen_x = X_rep_cand[pos].copy()
                voi = best_rep_kg
                mu_val = float(mu_all[chosen_idx])
                std_val = float(math.sqrt(max(float(cov_all[chosen_idx, chosen_idx]), 0.0)))
                n_r = int(n_rep_cand[pos])
                other_str = (
                    f"{best_exp_kg:.4g}" if best_exp_local >= 0 else "n/a"
                )
                reason = (
                    f"replicate observed recipe (n_reps={n_r}, mean={mu_val:.4g}, "
                    f"latent_std={std_val:.4g}, noise_sd={noise_sd:.4g}): "
                    f"KG={voi:.4g} > explore KG={other_str}"
                )
                kind: Literal["explore", "replicate"] = "replicate"

                d_rep = _pairwise_separation(
                    X_rep_cand, chosen_x.reshape(1, -1), cont, [], orig_width
                ).reshape(-1)
                eligible_rep &= d_rep > self.replicate_dedup_tol
            else:
                pos = int(active_exp_pos[best_exp_local])
                chosen_idx = int(idx_exp[pos])
                chosen_x = X_exp_cand[pos].copy()
                voi = best_exp_kg
                mu_val = float(mu_all[chosen_idx])
                std_val = float(math.sqrt(max(float(cov_all[chosen_idx, chosen_idx]), 0.0)))
                dist_val = float(min_dist_exp[pos])
                other_str = (
                    f"{best_rep_kg:.4g}" if best_rep_local >= 0 else "n/a"
                )
                reason = (
                    f"explore new region (mean={mu_val:.4g}, latent_std={std_val:.4g}, "
                    f"min_dist={dist_val:.3f}, noise_sd={noise_sd:.4g}): "
                    f"KG={voi:.4g} >= replicate KG={other_str}"
                )
                kind = "explore"

                eligible_exp[pos] = False
                if len(X_exp_cand) > 0:
                    d_exp = _pairwise_separation(
                        X_exp_cand, chosen_x.reshape(1, -1), cont, [], orig_width
                    ).reshape(-1)
                    min_dist_exp = np.minimum(min_dist_exp, d_exp)
                    if self.min_separation > 0.0:
                        eligible_exp &= min_dist_exp >= self.min_separation - 1e-12

            slots.append(
                ActionSlot(
                    kind=kind,
                    x=chosen_x,
                    expected_voi=float(max(voi, 0.0)),
                    predicted_mean=mu_val,
                    predicted_std=std_val,
                    reason=reason,
                )
            )
            cov_all = _schur_condition(cov_all, chosen_idx, noise_var)

        return ActionBatch(slots=tuple(slots), _dim=d)
