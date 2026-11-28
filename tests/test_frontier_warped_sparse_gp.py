"""Comprehensive tests for WarpedSparseGP first-principles surrogate."""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from kalos.core.frontier import WarpedSparseGP


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def test_import_and_fit_stay_torch_free() -> None:
    """Importing and running WarpedSparseGP must never import torch/botorch/gpytorch."""
    code = (
        "import sys, numpy as np; "
        "from kalos.core.frontier import WarpedSparseGP; "
        "rng = np.random.default_rng(0); "
        "X = rng.uniform(0, 1, (15, 3)); "
        "y = X[:, 0] ** 2 + 0.1 * rng.normal(size=15); "
        "bounds = np.array([[0.0]*3, [1.0]*3]); "
        "gp = WarpedSparseGP().fit(X, y, bounds, seed=0); "
        "m, s = gp.posterior(X[:4]); "
        "assert 'torch' not in sys.modules; "
        "assert 'botorch' not in sys.modules; "
        "assert 'gpytorch' not in sys.modules"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"


def test_sparse_25d_michaelis_menten_recovery_and_speed() -> None:
    """Recover the 3 active Michaelis-Menten dims in 25-D and predict accurately in ~ms."""
    rng = np.random.default_rng(42)
    d = 25
    n_train = 80
    n_test = 150
    bounds = np.vstack([np.zeros(d), np.full(d, 10.0)])

    def mm_response(X: np.ndarray) -> np.ndarray:
        # Three active dims with steep saturation (dims 2, 11, 20)
        x2 = X[:, 2] / 10.0
        x11 = X[:, 11] / 10.0
        x20 = X[:, 20] / 10.0
        return (
            8.0 * (x2 / (0.08 + x2))
            + 5.0 * (x11**2 / (0.12**2 + x11**2))
            + 3.0 * np.exp(-3.0 * x20)
        )

    X_train = rng.uniform(0.0, 10.0, size=(n_train, d))
    y_train = mm_response(X_train) + rng.normal(0.0, 0.15, size=n_train)
    X_test = rng.uniform(0.0, 10.0, size=(n_test, d))
    y_test = mm_response(X_test)

    t0 = time.perf_counter()
    gp = WarpedSparseGP(
        use_input_warp=True,
        use_output_warp=True,
        sparsity_strength=0.1,
        robust=True,
    ).fit(X_train, y_train, bounds, seed=42)
    elapsed = time.perf_counter() - t0

    mean, std = gp.posterior(X_test, observation_noise=False)
    r2 = _r2_score(y_test, mean)
    rho = float(spearmanr(y_test, mean).statistic)

    active = gp.active_dimensions(threshold_share=0.05)
    assert {2, 11, 20}.issubset(set(active)), f"Recovered active={active}, shares={gp.fit_diagnostics['relevance_shares']}"
    assert len(active) <= 5, f"Too many spurious active dims: {active}"
    assert r2 > 0.85, f"R2 too low: {r2:.3f}"
    assert rho > 0.90, f"Spearman too low: {rho:.3f}"
    assert np.all(std > 0.0)
    assert elapsed < 2.5, f"Fit took too long: {elapsed:.3f}s"

    diag = gp.fit_diagnostics
    for key in (
        "surrogate_kind",
        "output_lambda",
        "warp_alpha",
        "warp_beta",
        "lengthscales",
        "relevance_shares",
        "active_dims",
        "n_censored",
        "n_outliers_downweighted",
        "marginal_log_likelihood",
    ):
        assert key in diag
    assert diag["surrogate_kind"] == "warped_sparse_gp"


def test_output_warp_selects_log_lambda_and_beats_linear_gp() -> None:
    """Multiplicative log-normal targets select lambda < 0.4 and beat linear GP."""
    rng = np.random.default_rng(7)
    d = 4
    n_train = 60
    n_test = 120
    bounds = np.vstack([np.zeros(d), np.ones(d)])

    def latent_log(X: np.ndarray) -> np.ndarray:
        return (
            2.2 * np.sin(np.pi * X[:, 0])
            + 1.6 * (X[:, 1] - 0.5) ** 2 * 4.0
            - 1.2 * X[:, 2]
        )

    X_train = rng.uniform(0.0, 1.0, size=(n_train, d))
    y_train = np.exp(latent_log(X_train) + rng.normal(0.0, 0.12, size=n_train))

    X_test = rng.uniform(0.0, 1.0, size=(n_test, d))
    y_test = np.exp(latent_log(X_test))

    gp_warped = WarpedSparseGP(
        use_input_warp=False, use_output_warp=True, sparsity_strength=0.05, robust=False
    ).fit(X_train, y_train, bounds, seed=7)
    gp_linear = WarpedSparseGP(
        use_input_warp=False, use_output_warp=False, sparsity_strength=0.05, robust=False
    ).fit(X_train, y_train, bounds, seed=7)

    lam = gp_warped.fit_diagnostics["output_lambda"]
    assert lam < 0.4, f"Expected log-like lambda < 0.4, got {lam}"

    pred_warped, _ = gp_warped.posterior(X_test)
    pred_linear, _ = gp_linear.posterior(X_test)

    log_rmse_warped = float(np.sqrt(np.mean((np.log(np.maximum(pred_warped, 1e-6)) - np.log(y_test)) ** 2)))
    log_rmse_linear = float(np.sqrt(np.mean((np.log(np.maximum(pred_linear, 1e-6)) - np.log(y_test)) ** 2)))
    assert log_rmse_warped < log_rmse_linear * 0.75, (
        f"Expected warped log-RMSE ({log_rmse_warped:.3f}) to beat linear ({log_rmse_linear:.3f})"
    )

    # Also verify a purely linear-Gaussian target stays near lambda ~= 1.0
    y_lin_train = 10.0 + 3.0 * X_train[:, 0] - 2.0 * X_train[:, 1] + rng.normal(0.0, 0.2, size=n_train)
    gp_on_lin = WarpedSparseGP(use_input_warp=False, use_output_warp=True, robust=False).fit(
        X_train, y_lin_train, bounds, seed=7
    )
    assert gp_on_lin.fit_diagnostics["output_lambda"] > 0.65


def test_tobit_censored_lod_depresses_posterior_mean() -> None:
    """Dead < LOD wells must actively depress posterior mean compared to dropping them."""
    rng = np.random.default_rng(19)
    d = 2
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    X = rng.uniform(0.0, 1.0, size=(45, d))

    # True surface: high on x0 > 0.4, dead (< LOD=1.0) on x0 <= 0.35
    f_true = 10.0 * np.exp(-((X[:, 0] - 0.8) ** 2 + (X[:, 1] - 0.5) ** 2) / 0.08)
    # Force dead region for x0 <= 0.35
    dead = X[:, 0] <= 0.35
    lod_val = 1.0
    y_obs = f_true + rng.normal(0.0, 0.15, size=45)
    y_obs[dead] = 0.2  # stored below LOD; true censored region

    gp_tobit = WarpedSparseGP(
        use_input_warp=False, use_output_warp=False, sparsity_strength=0.0, robust=False
    ).fit(X, y_obs, bounds, censored_mask=dead, lod=lod_val, seed=19)

    gp_dropped = WarpedSparseGP(
        use_input_warp=False, use_output_warp=False, sparsity_strength=0.0, robust=False
    ).fit(X[~dead], y_obs[~dead], bounds, seed=19)

    assert gp_tobit.fit_diagnostics["n_censored"] == int(np.sum(dead))

    # Query inside the dead region
    X_dead_query = np.array([[0.15, 0.5], [0.20, 0.3], [0.25, 0.7]])
    mean_tobit, _ = gp_tobit.posterior(X_dead_query)
    mean_dropped, _ = gp_dropped.posterior(X_dead_query)

    assert np.all(mean_tobit < mean_dropped - 0.3), (
        f"Expected Tobit mean {mean_tobit} to be depressed below dropped-well mean {mean_dropped}"
    )
    assert np.all(mean_tobit < lod_val + 0.5)


def test_robust_loo_irls_downweights_extreme_outliers() -> None:
    """10% contaminated spikes are detected via Cholesky LOO and down-weighted cleanly."""
    rng = np.random.default_rng(123)
    d = 3
    n_train = 60
    bounds = np.vstack([np.zeros(d), np.ones(d)])

    def clean_fn(X: np.ndarray) -> np.ndarray:
        return 5.0 + 3.0 * np.sin(2.0 * np.pi * X[:, 0]) + 2.0 * X[:, 1] - 1.5 * X[:, 2]

    X_train = rng.uniform(0.0, 1.0, size=(n_train, d))
    y_clean = clean_fn(X_train) + rng.normal(0.0, 0.15, size=n_train)
    y_dirty = y_clean.copy()
    outlier_idx = np.array([2, 11, 23, 34, 47, 55])  # 10% contamination
    y_dirty[outlier_idx] += np.array([15.0, -14.0, 18.0, 12.0, -16.0, 15.0])

    gp_robust = WarpedSparseGP(
        use_input_warp=False, use_output_warp=False, sparsity_strength=0.05, robust=True
    ).fit(X_train, y_dirty, bounds, seed=123)
    gp_nonrobust = WarpedSparseGP(
        use_input_warp=False, use_output_warp=False, sparsity_strength=0.05, robust=False
    ).fit(X_train, y_dirty, bounds, seed=123)

    assert gp_robust.fit_diagnostics["n_outliers_downweighted"] >= 4

    X_test = rng.uniform(0.0, 1.0, size=(100, d))
    y_test = clean_fn(X_test)
    pred_robust, _ = gp_robust.posterior(X_test)
    pred_nonrobust, _ = gp_nonrobust.posterior(X_test)

    rmse_robust = float(np.sqrt(np.mean((pred_robust - y_test) ** 2)))
    rmse_nonrobust = float(np.sqrt(np.mean((pred_nonrobust - y_test) ** 2)))
    assert rmse_robust < rmse_nonrobust * 0.5, (
        f"Expected robust RMSE ({rmse_robust:.3f}) << non-robust ({rmse_nonrobust:.3f})"
    )
    assert rmse_robust < 0.7


def test_posterior_cov_quantiles_determinism_and_edge_cases() -> None:
    """Verify cov symmetry/PSD, quantile monotonicity, determinism, and zero-width bounds."""
    rng = np.random.default_rng(99)
    d = 4
    # Include a zero-width dimension (dim 3: [2.5, 2.5])
    bounds = np.array([[0.0, -1.0, 0.0, 2.5], [1.0, 1.0, 5.0, 2.5]])
    X = rng.uniform([0.0, -1.0, 0.0, 2.5], [1.0, 1.0, 5.0, 2.5], size=(30, d))
    y = np.exp(1.2 * X[:, 0] - 0.5 * X[:, 1]) + rng.normal(0.0, 0.05, size=30)

    gp1 = WarpedSparseGP().fit(X, y, bounds, seed=99)
    gp2 = WarpedSparseGP().fit(X, y, bounds, seed=99)

    X_q = rng.uniform([0.0, -1.0, 0.0, 2.5], [1.0, 1.0, 5.0, 2.5], size=(12, d))
    m1, cov1 = gp1.posterior_cov(X_q, observation_noise=True)
    m2, cov2 = gp2.posterior_cov(X_q, observation_noise=True)

    assert np.allclose(m1, m2, atol=1e-12)
    assert np.allclose(cov1, cov2, atol=1e-12)
    assert cov1.shape == (12, 12)
    assert np.allclose(cov1, cov1.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(cov1)
    assert np.all(eigs >= -1e-10), f"Negative eigenvalues: {eigs}"

    # Check posterior std matches sqrt(diag(cov))
    m_post, s_post = gp1.posterior(X_q, observation_noise=True)
    assert np.allclose(m_post, m1, atol=1e-10)
    assert np.allclose(s_post, np.sqrt(np.diag(cov1)), rtol=1e-4, atol=1e-6)

    # Quantile monotonicity
    qs = gp1.posterior_quantiles(X_q, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95))
    assert qs.shape == (12, 5)
    assert np.all(np.diff(qs, axis=1) >= 0.0)

    # Check stored attributes
    assert gp1._X is not None and gp1._X.shape == (30, d)
    assert gp1._y is not None and gp1._y.shape == (30,)
    assert gp1._bounds is not None and gp1._bounds.shape == (2, d)
    assert np.isclose(gp1.best_f, float(np.max(y)))


def test_posterior_warped_and_grad_matches_finite_differences() -> None:
    """Analytical posterior_warped_and_grad matches central FD for Cholesky+warp and Tobit Laplace."""
    rng = np.random.default_rng(77)
    d = 3
    bounds = np.array([[0.0, -2.0, 1.0], [2.0, 4.0, 5.0]])
    X = rng.uniform(bounds[0], bounds[1], size=(20, d))
    y = 2.0 + np.sin(1.5 * X[:, 0]) + 0.3 * (X[:, 1] ** 2) - 0.4 * X[:, 2] + rng.normal(0.0, 0.05, size=20)
    x_probe = np.array([0.85, 1.15, 2.75])
    eps = 1e-6

    # 1. Exact Cholesky + Kumaraswamy input warp
    gp_chol = WarpedSparseGP(
        use_input_warp=True, use_output_warp=True, sparsity_strength=0.05, robust=False
    ).fit(X, y, bounds, seed=77)
    assert gp_chol._active_input_warp is True
    mu, var, dmu_dx, dvar_dx = gp_chol.posterior_warped_and_grad(x_probe)
    mu_arr, std_arr = gp_chol.warped_posterior_mean_std(x_probe.reshape(1, -1))
    assert np.isclose(mu, float(mu_arr[0]), atol=1e-10)
    assert np.isclose(var, float(std_arr[0] ** 2), atol=1e-8)

    fd_mu = np.zeros(d, dtype=float)
    fd_var = np.zeros(d, dtype=float)
    for j in range(d):
        dx = np.zeros(d, dtype=float)
        dx[j] = eps
        m_p, v_p, _, _ = gp_chol.posterior_warped_and_grad(x_probe + dx)
        m_m, v_m, _, _ = gp_chol.posterior_warped_and_grad(x_probe - dx)
        fd_mu[j] = (m_p - m_m) / (2.0 * eps)
        fd_var[j] = (v_p - v_m) / (2.0 * eps)
    assert np.allclose(dmu_dx, fd_mu, rtol=1e-3, atol=1e-4)
    assert np.allclose(dvar_dx, fd_var, rtol=1e-3, atol=1e-4)

    # 2. Tobit Laplace censored mode
    cens = X[:, 0] < 0.7
    lod = 1.5
    y_cens = np.where(cens, lod, np.maximum(y, lod + 0.1))
    gp_lap = WarpedSparseGP(
        use_input_warp=False, use_output_warp=True, sparsity_strength=0.05, robust=False
    ).fit(X, y_cens, bounds, censored_mask=cens, lod=lod, seed=78)
    assert gp_lap._is_laplace is True
    _, _, dmu_lap, dvar_lap = gp_lap.posterior_warped_and_grad(x_probe)
    for j in range(d):
        dx = np.zeros(d, dtype=float)
        dx[j] = eps
        m_p, v_p, _, _ = gp_lap.posterior_warped_and_grad(x_probe + dx)
        m_m, v_m, _, _ = gp_lap.posterior_warped_and_grad(x_probe - dx)
        assert np.isclose(dmu_lap[j], (m_p - m_m) / (2.0 * eps), rtol=1e-3, atol=1e-4)
        assert np.isclose(dvar_lap[j], (v_p - v_m) / (2.0 * eps), rtol=1e-3, atol=1e-4)
