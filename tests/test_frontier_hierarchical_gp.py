"""Tests for pure NumPy/SciPy HierarchicalBatchGP and MultiScaleTransferGP."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr

from kalos.core.frontier.hierarchical_gp import (
    HierarchicalBatchGP,
    MultiScaleTransferGP,
    _matern52_ard_train,
    _mll_and_intermediates,
)


def _matern52_ground_truth(x: np.ndarray) -> np.ndarray:
    """Smooth 2-D bioprocess-like surface with interior peak."""
    return (
        2.0 * np.exp(-((x[:, 0] - 0.65) ** 2) / 0.08 - ((x[:, 1] - 0.4) ** 2) / 0.10)
        + 0.4 * np.sin(3.0 * x[:, 0])
        - 0.25 * x[:, 1]
    )


def test_module_is_torch_free_at_import_and_runtime() -> None:
    """Importing and running hierarchical_gp must never import torch/botorch/gpytorch."""
    code = (
        "import sys, numpy as np; "
        "from kalos.core.frontier.hierarchical_gp import HierarchicalBatchGP, MultiScaleTransferGP; "
        "assert 'torch' not in sys.modules; "
        "assert 'botorch' not in sys.modules; "
        "assert 'gpytorch' not in sys.modules; "
        "rng = np.random.default_rng(0); "
        "X = rng.uniform(0, 1, (12, 2)); "
        "b = np.array([[0.0, 0.0], [1.0, 1.0]]); "
        "y = np.sin(4 * X[:, 0]) + np.array([1.0 if i < 6 else -1.0 for i in range(12)]); "
        "g = ['r1' if i < 6 else 'r2' for i in range(12)]; "
        "gp = HierarchicalBatchGP().fit(X, y, g, b, seed=0); "
        "gp.posterior(X[:3]); "
        "tgp = MultiScaleTransferGP().fit(X, y, g, b, seed=0); "
        "tgp.posterior(X[:3], target_scale='r1'); "
        "assert 'torch' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_hierarchical_batch_gp_analytical_gradients_match_finite_diff() -> None:
    """Verify exact MLL analytical gradients against central finite differences."""
    rng = np.random.default_rng(123)
    n, d = 14, 2
    X_norm = rng.uniform(0.0, 1.0, (n, d))
    y_std = rng.normal(0.0, 1.0, n)
    groups = np.array([0] * 5 + [1] * 5 + [2] * 4)
    M_batch = (groups[:, None] == groups[None, :]).astype(np.float64)
    diff = X_norm[:, None, :] - X_norm[None, :, :]
    diff_sq = diff * diff
    design = np.ones((n, 1), dtype=np.float64)
    eye_n = np.eye(n, dtype=np.float64)
    jitter = 1e-8

    def eval_mll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        sf = float(np.exp(theta[0]))
        ls = np.exp(theta[1 : 1 + d])
        tau = float(np.exp(theta[1 + d]))
        sa = float(np.exp(theta[2 + d]))
        sf2, tau2, sa2 = sf * sf, tau * tau, sa * sa
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

    theta0 = np.array([-0.2, -0.7, -0.4, 0.3, -1.5], dtype=np.float64)
    _, analytic = eval_mll_grad(theta0)
    eps = 1e-6
    numeric = np.empty_like(theta0)
    for k in range(len(theta0)):
        tp = theta0.copy()
        tm = theta0.copy()
        tp[k] += eps
        tm[k] -= eps
        numeric[k] = (eval_mll_grad(tp)[0] - eval_mll_grad(tm)[0]) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_multiscale_transfer_gp_analytical_gradients_match_finite_diff() -> None:
    """Verify ICM MLL analytical gradients against central finite differences."""
    rng = np.random.default_rng(456)
    n, d, S, r = 15, 2, 3, 2
    X_norm = rng.uniform(0.0, 1.0, (n, d))
    y_std = rng.normal(0.0, 1.0, n)
    s_idx = np.array([0] * 6 + [1] * 5 + [2] * 4)
    D = np.zeros((n, S), dtype=np.float64)
    D[np.arange(n), s_idx] = 1.0
    diff = X_norm[:, None, :] - X_norm[None, :, :]
    diff_sq = diff * diff
    n_L = S * r
    jitter = 1e-8

    def eval_mll_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        ls = np.exp(theta[:d])
        L = theta[d : d + n_L].reshape(S, r)
        kappa = np.exp(theta[d + n_L : d + n_L + S])
        sigma = np.exp(theta[d + n_L + S : d + n_L + 2 * S])
        B = L @ L.T + np.diag(kappa)
        sigma2 = sigma * sigma
        K_x, s5r, exp_s5r = _matern52_ard_train(diff, ls)
        B_mat = B[s_idx[:, None], s_idx[None, :]]
        K = K_x * B_mat
        K[np.arange(n), np.arange(n)] += sigma2[s_idx] + jitter
        mll, W, _, _, _ = _mll_and_intermediates(K, D, y_std)

        grad = np.empty_like(theta)
        Q = (W * B_mat) * ((5.0 / 3.0) * (1.0 + s5r) * exp_s5r)
        grad[:d] = np.einsum("ij,ijk->k", Q, diff_sq) / (2.0 * ls * ls)
        G_B = 0.5 * (D.T @ (W * K_x) @ D)
        G_B = 0.5 * (G_B + G_B.T)
        grad[d : d + n_L] = (2.0 * (G_B @ L)).reshape(-1)
        grad[d + n_L : d + n_L + S] = kappa * np.diag(G_B)
        grad[d + n_L + S : d + n_L + 2 * S] = sigma2 * (D.T @ np.diag(W))
        return mll, grad

    theta0 = np.array(
        [-0.6, -0.3, 0.7, 0.2, 0.5, -0.3, -0.2, 0.6, -1.2, -0.9, -1.0, -1.8, -1.5, -1.3],
        dtype=np.float64,
    )
    _, analytic = eval_mll_grad(theta0)
    eps = 1e-6
    numeric = np.empty_like(theta0)
    for k in range(len(theta0)):
        tp = theta0.copy()
        tm = theta0.copy()
        tp[k] += eps
        tm[k] -= eps
        numeric[k] = (eval_mll_grad(tp)[0] - eval_mll_grad(tm)[0]) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_hierarchical_batch_gp_recovers_process_and_blups_under_large_batch_offsets() -> None:
    """With tau_batch = 2.0 corrupting raw y, unseen-batch predictions recover f_process."""
    rng = np.random.default_rng(7)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    true_offsets = {
        "round_1": -3.2,
        "round_2": 2.8,
        "round_3": -1.5,
        "round_4": 1.9,
        "round_5": 0.4,
        "round_6": -0.6,
    }
    # Deliberately vary batch quality across X so a naive pooled GP gets fooled by batch offsets.
    X_parts, y_parts, groups = [], [], []
    for g_name, offset in true_offsets.items():
        n_b = 8
        Xb = rng.uniform(0.0, 1.0, (n_b, 2))
        yb = _matern52_ground_truth(Xb) + offset + rng.normal(0.0, 0.08, n_b)
        X_parts.append(Xb)
        y_parts.append(yb)
        groups.extend([g_name] * n_b)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    gp = HierarchicalBatchGP().fit(X, y, groups, bounds, seed=42)

    # Evaluate on a dense test grid against true latent process f_process.
    gx = np.linspace(0.05, 0.95, 10)
    g0, g1 = np.meshgrid(gx, gx)
    X_test = np.column_stack([g0.ravel(), g1.ravel()])
    f_true = _matern52_ground_truth(X_test)

    mean_unseen, std_unseen = gp.posterior(X_test, group=None, observation_noise=False)
    rho_hgp = float(spearmanr(f_true, mean_unseen).statistic)
    assert rho_hgp > 0.90, f"expected high unseen-batch rank correlation, got {rho_hgp:.3f}"
    assert std_unseen.shape == (len(X_test),)
    assert np.all(std_unseen > 0.0)

    # Compare against a single-batch GP (ignoring batch offsets -> pooled fit).
    pooled = HierarchicalBatchGP().fit(X, y, ["all"] * len(y), bounds, seed=42)
    mean_pooled, _ = pooled.posterior(X_test, group=None, observation_noise=False)
    rmse_hgp = float(np.sqrt(np.mean((mean_unseen - np.mean(mean_unseen) - (f_true - np.mean(f_true))) ** 2)))
    rmse_pooled = float(np.sqrt(np.mean((mean_pooled - np.mean(mean_pooled) - (f_true - np.mean(f_true))) ** 2)))
    assert rmse_hgp < rmse_pooled

    # Check batch BLUPs accurately track centered true offsets.
    blups = gp.batch_effects()
    assert set(blups.keys()) == set(true_offsets.keys())
    est_means = np.array([blups[k]["mean"] for k in true_offsets])
    true_vals = np.array([true_offsets[k] for k in true_offsets])
    est_centered = est_means - np.mean(est_means)
    true_centered = true_vals - np.mean(true_vals)
    np.testing.assert_allclose(est_centered, true_centered, atol=0.35)
    assert all(blups[k]["std"] > 0.0 for k in true_offsets)

    # Seen-batch prediction adds the batch BLUP offset and reduces variance vs unseen-batch obs noise.
    mean_r2, std_r2_latent = gp.posterior(X_test[:5], group="round_2", observation_noise=False)
    mean_r1, _ = gp.posterior(X_test[:5], group="round_1", observation_noise=False)
    assert np.all(mean_r2 - mean_r1 > 4.5)
    _, std_unseen_obs = gp.posterior(X_test[:5], group=None, observation_noise=True)
    _, std_r2_obs = gp.posterior(X_test[:5], group="round_2", observation_noise=True)
    assert np.all(std_r2_obs < std_unseen_obs)
    assert np.all(std_r2_latent < std_r2_obs)


def test_hierarchical_batch_gp_variance_decomposition_and_observation_noise_identity() -> None:
    """Variance shares sum to 1.0 and observation_noise=True adds tau_batch^2 + sigma_assay^2."""
    rng = np.random.default_rng(11)
    bounds = np.array([[0.0, 0.0], [10.0, 10.0]])
    X = rng.uniform(0.0, 10.0, (36, 2))
    groups = [f"b{i // 6}" for i in range(36)]
    offsets = {f"b{k}": float(v) for k, v in enumerate([-2.5, -1.0, 0.0, 1.2, 2.3, 3.0])}
    y = (
        1.5 * np.sin(0.6 * X[:, 0])
        + 0.5 * np.cos(0.5 * X[:, 1])
        + np.array([offsets[g] for g in groups])
        + rng.normal(0.0, 0.15, 36)
    )

    gp = HierarchicalBatchGP().fit(X, y, groups, bounds, seed=3)
    vd = gp.variance_decomposition()
    assert set(vd.keys()) == {
        "process_var",
        "batch_var",
        "assay_var",
        "process_share",
        "batch_share",
        "assay_share",
    }
    assert vd["process_var"] > 0.0
    assert vd["batch_var"] > 1.0
    assert vd["assay_var"] > 0.0
    assert vd["process_share"] + vd["batch_share"] + vd["assay_share"] == pytest.approx(1.0, abs=1e-12)

    X_q = np.array([[2.5, 5.0], [7.5, 2.0], [5.0, 8.0]])
    mean_lat, std_lat = gp.posterior(X_q, group=None, observation_noise=False)
    mean_obs, std_obs = gp.posterior(X_q, group=None, observation_noise=True)
    np.testing.assert_allclose(mean_lat, mean_obs, atol=1e-12)
    expected_obs_var = std_lat**2 + vd["batch_var"] + vd["assay_var"]
    np.testing.assert_allclose(std_obs**2, expected_obs_var, rtol=1e-10, atol=1e-12)

    mean_cov, cov = gp.posterior_cov(X_q, group=None, observation_noise=False)
    np.testing.assert_allclose(mean_cov, mean_lat, atol=1e-12)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(cov), std_lat**2, rtol=1e-10, atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12

    diag = gp.fit_diagnostics
    assert diag["surrogate_kind"] == "hierarchical_batch_gp"
    assert len(diag["lengthscales"]) == 2
    assert diag["n_batches"] == 6
    assert np.isfinite(diag["marginal_log_likelihood"])
    assert gp.best_f == pytest.approx(float(y.max()))
    assert gp._X is not None and gp._y is not None and gp._bounds is not None


def test_hierarchical_batch_gp_edge_cases_and_seed_determinism() -> None:
    """Duplicate rows, constant target, single batch, and seed determinism work cleanly."""
    bounds = np.array([[0.0], [5.0]])
    X = np.array([[1.0], [1.0], [2.5], [4.0], [4.0], [4.5]])
    y = np.array([3.0, 3.1, 5.0, 2.0, 2.1, 1.5])
    groups = [0, 0, 1, 1, 1, 0]

    gp1 = HierarchicalBatchGP().fit(X, y, groups, bounds, seed=99)
    gp2 = HierarchicalBatchGP().fit(X, y, groups, bounds, seed=99)
    m1, s1 = gp1.posterior([[2.0], [3.5]], group=0)
    m2, s2 = gp2.posterior([[2.0], [3.5]], group=0)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(s1, s2)

    # Constant y should not produce NaNs and shares still sum to 1.0.
    gp_const = HierarchicalBatchGP().fit(X, np.full(len(X), 7.5), groups, bounds, seed=0)
    mc, sc = gp_const.posterior([[2.0]])
    assert np.allclose(mc, 7.5, atol=1e-4)
    assert np.all(np.isfinite(sc))
    vdc = gp_const.variance_decomposition()
    assert vdc["process_share"] + vdc["batch_share"] + vdc["assay_share"] == pytest.approx(1.0)


def test_multiscale_transfer_gp_high_correlation_with_shift_and_gain() -> None:
    """Correlated scales with 10x titer gain + large offset learn rho > 0.7."""
    rng = np.random.default_rng(21)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    X_bench = rng.uniform(0.0, 1.0, (24, 2))
    X_pilot = rng.uniform(0.0, 1.0, (12, 2))
    y_bench = _matern52_ground_truth(X_bench) + rng.normal(0.0, 0.04, len(X_bench))
    # 10x unit gain + 50.0 offset shift between shake flask (bench) and bioreactor (pilot).
    y_pilot = 10.0 * _matern52_ground_truth(X_pilot) + 50.0 + rng.normal(0.0, 0.4, len(X_pilot))

    X = np.vstack([X_bench, X_pilot])
    y = np.concatenate([y_bench, y_pilot])
    scales = ["bench"] * len(X_bench) + ["pilot"] * len(X_pilot)

    tgp = MultiScaleTransferGP().fit(X, y, scales, bounds, rank=2, seed=5)
    names, R = tgp.transfer_correlation_matrix()
    assert names == ["bench", "pilot"]
    assert R.shape == (2, 2)
    np.testing.assert_allclose(np.diag(R), [1.0, 1.0])
    assert R[0, 1] > 0.7, f"expected high transfer correlation (>0.7), got {R[0, 1]:.3f}"
    assert R[0, 1] == pytest.approx(R[1, 0])

    # Predictions for pilot scale should be back-transformed to pilot units (~50..70).
    X_test = np.array([[0.65, 0.40], [0.10, 0.90]])
    mean_p, std_p = tgp.posterior(X_test, target_scale="pilot", observation_noise=False)
    expected_p = 10.0 * _matern52_ground_truth(X_test) + 50.0
    np.testing.assert_allclose(mean_p, expected_p, atol=2.5)
    assert np.all(std_p > 0.0)


def test_multiscale_transfer_gp_low_correlation_on_uncorrelated_scales() -> None:
    """Uncorrelated scales yield |rho| < 0.4."""
    rng = np.random.default_rng(31)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    X_a = rng.uniform(0.0, 1.0, (25, 2))
    X_b = rng.uniform(0.0, 1.0, (25, 2))
    # Orthogonal spatial dependencies: scale A depends only on x0, scale B only on x1.
    y_a = np.sin(4.0 * X_a[:, 0]) + rng.normal(0.0, 0.05, len(X_a))
    y_b = np.cos(4.0 * X_b[:, 1]) + rng.normal(0.0, 0.05, len(X_b))

    X = np.vstack([X_a, X_b])
    y = np.concatenate([y_a, y_b])
    scales = ["scale_a"] * len(X_a) + ["scale_b"] * len(X_b)

    tgp = MultiScaleTransferGP().fit(X, y, scales, bounds, rank=2, seed=0)
    names, R = tgp.transfer_correlation_matrix()
    assert names == ["scale_a", "scale_b"]
    assert abs(R[0, 1]) < 0.4, f"expected low correlation (|rho|<0.4), got {R[0, 1]:.3f}"


def test_multiscale_transfer_gp_beats_target_only_gp_with_five_target_runs() -> None:
    """With 25 source runs and only 5 target runs, transfer GP beats target-only GP."""
    rng = np.random.default_rng(101)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    X_bench = rng.uniform(0.0, 1.0, (28, 2))
    y_bench = _matern52_ground_truth(X_bench) + rng.normal(0.0, 0.05, len(X_bench))

    # Only 5 pilot runs, with shifted + scaled response.
    X_pilot = np.array([
        [0.15, 0.20],
        [0.85, 0.80],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.50, 0.50],
    ])
    y_pilot = 4.0 * _matern52_ground_truth(X_pilot) + 12.0 + rng.normal(0.0, 0.1, len(X_pilot))

    X_all = np.vstack([X_bench, X_pilot])
    y_all = np.concatenate([y_bench, y_pilot])
    scales = ["bench"] * len(X_bench) + ["pilot"] * len(X_pilot)

    transfer_gp = MultiScaleTransferGP().fit(X_all, y_all, scales, bounds, rank=2, seed=0)
    target_only_gp = MultiScaleTransferGP().fit(
        X_pilot, y_pilot, ["pilot"] * len(X_pilot), bounds, rank=1, seed=0
    )

    gx = np.linspace(0.05, 0.95, 10)
    g0, g1 = np.meshgrid(gx, gx)
    X_test = np.column_stack([g0.ravel(), g1.ravel()])
    y_true_pilot = 4.0 * _matern52_ground_truth(X_test) + 12.0

    pred_transfer, std_transfer = transfer_gp.posterior(X_test, target_scale="pilot")
    pred_only, _ = target_only_gp.posterior(X_test, target_scale="pilot")

    rmse_transfer = float(np.sqrt(np.mean((pred_transfer - y_true_pilot) ** 2)))
    rmse_only = float(np.sqrt(np.mean((pred_only - y_true_pilot) ** 2)))
    rho_transfer = float(spearmanr(y_true_pilot, pred_transfer).statistic)
    rho_only = float(spearmanr(y_true_pilot, pred_only).statistic)

    assert rmse_transfer < rmse_only, f"expected transfer RMSE ({rmse_transfer:.3f}) < target-only ({rmse_only:.3f})"
    assert rho_transfer > rho_only, f"expected transfer rho ({rho_transfer:.3f}) > target-only ({rho_only:.3f})"
    assert rho_transfer > 0.85
    assert np.all(std_transfer > 0.0)

    # Also verify observation_noise=True strictly adds target-scale noise variance.
    _, std_obs = transfer_gp.posterior(X_test[:4], target_scale="pilot", observation_noise=True)
    _, std_lat = transfer_gp.posterior(X_test[:4], target_scale="pilot", observation_noise=False)
    noise_var = transfer_gp.fit_diagnostics["scale_noise_vars"]["pilot"]
    np.testing.assert_allclose(std_obs**2, std_lat**2 + noise_var, rtol=1e-10, atol=1e-12)
    assert transfer_gp.fit_diagnostics["surrogate_kind"] == "multi_scale_transfer_gp"

    with pytest.raises(ValueError, match="Unknown target_scale"):
        transfer_gp.posterior(X_test[:2], target_scale="production")
