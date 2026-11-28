"""Tests for torch-free AdaptiveTuRBO, TrustRegionState, and VOIAllocator."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from kalos.core.frontier.acquisition import (
    ActionBatch,
    ActionSlot,
    AdaptiveTuRBO,
    JointQLogNEI,
    TrustRegionState,
    VOIAllocator,
)
from kalos.core.space_filling import proposal_separation


class _ExactNumpyGP:
    """Minimal exact RBF GP in pure NumPy exposing .posterior, .posterior_cov, and lengthscales."""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        lengthscales: np.ndarray | float = 0.25,
        signal_var: float = 1.0,
        obs_noise_var: np.ndarray | float = 1e-4,
        prior_mean: float = 0.0,
    ) -> None:
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        d = self.X_train.shape[1]
        ls = np.broadcast_to(np.asarray(lengthscales, dtype=float), (d,)).copy()
        self.lengthscales = ls
        self.signal_var = float(signal_var)
        self.prior_mean = float(prior_mean)
        self.fit_diagnostics = {"lengthscales": ls.tolist()}

        n = self.X_train.shape[0]
        noise = np.broadcast_to(np.asarray(obs_noise_var, dtype=float), (n,)).copy()
        K_tt = self._kernel(self.X_train, self.X_train)
        K_noisy = K_tt + np.diag(noise + 1e-10)
        self._L = np.linalg.cholesky(K_noisy)
        centered = self.y_train - self.prior_mean
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, centered))

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = (X1[:, None, :] - X2[None, :, :]) / self.lengthscales[None, None, :]
        sq = np.sum(diff * diff, axis=2)
        return self.signal_var * np.exp(-0.5 * sq)

    def posterior_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        K_tx = self._kernel(self.X_train, Xa)
        K_xx = self._kernel(Xa, Xa)
        mean = self.prior_mean + K_tx.T @ self._alpha
        V = np.linalg.solve(self._L, K_tx)
        cov = K_xx - V.T @ V
        cov = 0.5 * (cov + cov.T)
        np.fill_diagonal(cov, np.maximum(np.diag(cov), 1e-12))
        return mean, cov

    def posterior(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, cov = self.posterior_cov(X)
        return mean, np.sqrt(np.maximum(np.diag(cov), 1e-12))


class _MarginalOnlySurrogate:
    """Surrogate exposing only .posterior(X) without .posterior_cov or lengthscales."""

    def __init__(self, target_fn) -> None:
        self._fn = target_fn

    def posterior(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        mean, std = self._fn(Xa)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def test_frontier_acquisition_is_torch_free():
    import textwrap

    code = textwrap.dedent("""\
        import sys
        import numpy as np
        from kalos.core.frontier.acquisition import TrustRegionState, AdaptiveTuRBO, VOIAllocator

        loaded_at_import = 'torch' in sys.modules
        bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        X_tr = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        y_tr = np.array([0.2, 1.0, 0.3])

        class S:
            def posterior(self, X):
                return np.ones(len(X)), np.full(len(X), 0.2)

        tr = TrustRegionState(bounds)
        tr.update(1.0)
        pts = AdaptiveTuRBO(n_candidates=64).propose(
            S(), bounds, q=2, X_train=X_tr, y_train=y_tr, tr_state=tr, seed=0
        )
        batch = VOIAllocator(n_candidates=32, n_reference=32).allocate(
            S(), X_tr, y_tr, bounds, q=2, assay_noise_var=0.1, seed=0
        )
        loaded_at_call = 'torch' in sys.modules
        print(loaded_at_import, loaded_at_call, pts.shape[0], batch.X.shape[0])
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False False 2 2"


def test_trust_region_state_expand_shrink_and_restart():
    bounds = np.array([[0.0, 0.0], [10.0, 10.0]])
    tr = TrustRegionState(
        bounds,
        base_length=0.4,
        min_length=0.15,
        max_length=1.0,
        success_tol=3,
        failure_tol=3,
    )
    assert tr.length == pytest.approx(0.4)
    assert tr.n_restarts == 0

    # 3 consecutive strict improvements -> expand by 1.6x
    tr.update(1.0)
    assert tr.success_count == 1
    tr.update(2.0)
    assert tr.success_count == 2
    ret = tr.update(3.0)
    assert ret is tr
    assert tr.success_count == 0
    assert tr.length == pytest.approx(0.64)
    assert tr.best_value == pytest.approx(3.0)

    # Cap at max_length=1.0 on next expansion
    tr.update(4.0).update(5.0).update(6.0)
    assert tr.length == pytest.approx(1.0)

    # Non-improvement resets success_count; 3 failures halve length
    tr.update(7.0)  # success 1
    assert tr.success_count == 1
    tr.update(7.0)  # equal is NOT a strict improvement -> failure 1
    assert tr.success_count == 0
    assert tr.failure_count == 1
    tr.update(6.5)  # failure 2
    tr.update(5.0)  # failure 3 -> 0.5
    assert tr.failure_count == 0
    assert tr.length == pytest.approx(0.5)

    # Halve to 0.25, then 0.125 < min_length=0.15 -> restart to base_length=0.4
    tr.update(6.0).update(6.0).update(6.0)
    assert tr.length == pytest.approx(0.25)
    assert tr.n_restarts == 0

    tr.update(6.0).update(6.0).update(6.0)
    assert tr.n_restarts == 1
    assert tr.length == pytest.approx(0.4)
    assert tr.success_count == 0
    assert tr.failure_count == 0


def test_trust_region_state_minimize_and_validation():
    bounds = np.array([[0.0], [1.0]])
    tr = TrustRegionState(bounds, base_length=0.5, min_length=0.1, success_tol=2, failure_tol=2, maximize=False)
    tr.update(5.0).update(4.0)
    assert tr.length == pytest.approx(0.8)
    tr.update(4.0).update(4.5)
    assert tr.length == pytest.approx(0.4)

    with pytest.raises(ValueError, match="finite"):
        tr.update(float("nan"))
    with pytest.raises(ValueError, match="min_length"):
        TrustRegionState(bounds, base_length=0.01, min_length=0.05)


def test_adaptive_turbo_ard_scaling_and_joint_cholesky_ts():
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    X_train = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]])
    y_train = np.array([2.0, 0.5, 0.4])
    # Strongly anisotropic lengthscales: dim 0 tight (0.05), dim 1 wide (2.0)
    gp = _ExactNumpyGP(X_train, y_train, lengthscales=np.array([0.05, 2.0]))
    cov_calls = 0
    orig_cov = gp.posterior_cov

    def spy_cov(X):
        nonlocal cov_calls
        cov_calls += 1
        return orig_cov(X)

    gp.posterior_cov = spy_cov  # type: ignore[method-assign]

    tr = TrustRegionState(bounds, base_length=0.4)
    turbo = AdaptiveTuRBO(n_candidates=256, ts_weight=0.5, global_frac=0.0, min_separation=0.05)
    pts1 = turbo.propose(gp, bounds, q=4, X_train=X_train, y_train=y_train, tr_state=tr, seed=12)
    pts2 = turbo.propose(gp, bounds, q=4, X_train=X_train, y_train=y_train, tr_state=tr, seed=12)

    assert cov_calls >= 2
    assert pts1.shape == (4, 2)
    assert np.array_equal(pts1, pts2)

    # Expected ARD side lengths:
    # geomean([0.05, 2.0]) = sqrt(0.1) ~= 0.31623
    # L_0 = clip(0.4 * 0.05 / 0.31623, 0.05, 1.0) = 0.06325 -> half-width ~ 0.03162
    # L_1 = clip(0.4 * 2.0 / 0.31623, 0.05, 1.0) = 1.0 -> full [0, 1] range
    assert np.all(np.abs(pts1[:, 0] - 0.5) <= 0.04)
    assert np.ptp(pts1[:, 1]) > 0.2


def test_adaptive_turbo_bounds_fixed_features_constraints_and_pending():
    bounds = np.array([[-2.0, 0.0, 10.0, -5.0], [4.0, 10.0, 20.0, 5.0]])
    X_train = np.array([
        [0.0, 2.5, 12.0, 0.0],
        [1.0, 2.5, 14.0, 1.0],
    ])
    y_train = np.array([1.0, 2.5])
    pending = np.array([
        [1.5, 2.5, 13.0, 1.0],
        [-0.5, 2.5, 16.0, -1.0],
    ])
    fixed = {1: 2.5, 3: -1.0}
    # x0 >= 0.0 and x0 + 0.1 * x2 <= 2.5  <=>  -x0 - 0.1 * x2 >= -2.5
    constraints = [
        ([0], [1.0], 0.0),
        ([0, 2], [-1.0, -0.1], -2.5),
    ]

    surrogate = _MarginalOnlySurrogate(
        lambda X: (X[:, 0] - 0.1 * X[:, 2], np.full(len(X), 0.3))
    )
    turbo = AdaptiveTuRBO(n_candidates=512, ts_weight=0.5, global_frac=0.15, min_separation=0.05)
    pts = turbo.propose(
        surrogate,
        bounds,
        q=5,
        X_train=X_train,
        y_train=y_train,
        pending=pending,
        fixed_features=fixed,
        inequality_constraints=constraints,
        seed=7,
    )

    assert 1 <= pts.shape[0] <= 5
    assert pts.shape[1] == 4
    assert np.all(pts >= bounds[0] - 1e-12)
    assert np.all(pts <= bounds[1] + 1e-12)
    assert np.allclose(pts[:, 1], 2.5)
    assert np.allclose(pts[:, 3], -1.0)
    assert np.all(pts[:, 0] >= -1e-6)
    assert np.all(pts[:, 0] + 0.1 * pts[:, 2] <= 2.5 + 1e-6)

    refs = np.vstack([X_train, pending])
    for i, row in enumerate(pts):
        for ref in refs:
            assert proposal_separation(row, ref, bounds[0], bounds[1]) >= 0.05 - 1e-9
        for j in range(i + 1, len(pts)):
            assert proposal_separation(row, pts[j], bounds[0], bounds[1]) >= 0.05 - 1e-9


def test_adaptive_turbo_25d_min_separation_no_corner_collapse():
    d = 25
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    rng = np.random.default_rng(42)
    X_train = rng.uniform(0.1, 0.9, size=(12, d))
    y_train = -np.sum((X_train - 0.5) ** 2, axis=1)
    pending = rng.uniform(0.1, 0.9, size=(4, d))

    gp = _ExactNumpyGP(X_train, y_train, lengthscales=np.linspace(0.2, 0.8, d))
    turbo = AdaptiveTuRBO(n_candidates=512, ts_weight=0.5, global_frac=0.15, min_separation=0.05)
    pts = turbo.propose(
        gp,
        bounds,
        q=8,
        X_train=X_train,
        y_train=y_train,
        pending=pending,
        seed=3,
    )

    assert pts.shape == (8, d)
    assert np.all((pts >= 0.0) & (pts <= 1.0))

    refs = np.vstack([X_train, pending])
    for i in range(len(pts)):
        for ref in refs:
            assert proposal_separation(pts[i], ref, bounds[0], bounds[1]) >= 0.05 - 1e-9
        for j in range(i + 1, len(pts)):
            assert proposal_separation(pts[i], pts[j], bounds[0], bounds[1]) >= 0.05 - 1e-9
        # Must not collapse onto hypercube corners (all coordinates near 0 or 1)
        interior_coords = np.sum((pts[i] > 0.05) & (pts[i] < 0.95))
        assert interior_coords >= 3


def test_adaptive_turbo_with_real_botorch_surrogate():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from kalos.core.surrogate import Surrogate

    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    rng = np.random.default_rng(9)
    X_train = rng.uniform(0.0, 1.0, size=(10, 3))
    y_train = np.sin(3.0 * X_train[:, 0]) + 0.5 * X_train[:, 1]
    surr = Surrogate().fit(X_train, y_train, bounds)

    pts = AdaptiveTuRBO(n_candidates=128).propose(
        surr, bounds, q=4, X_train=X_train, y_train=y_train, seed=1
    )
    assert pts.shape == (4, 3)
    assert np.all((pts >= 0.0) & (pts <= 1.0))


def test_voi_allocator_replicates_under_high_noise_and_uncertain_top():
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    # Recipe A (0.80, 0.80) is a single noisy measurement (high latent uncertainty),
    # while Recipe B (0.20, 0.20) is a well-replicated runner-up baseline.
    # Unexplored points far from A/B have lower prior mean, so replicating A has the
    # highest One-Step Knowledge Gradient to confirm whether A truly beats B.
    X_train = np.vstack([
        [[0.80, 0.80]],
        np.tile([0.20, 0.20], (10, 1)),
    ])
    y_train = np.concatenate([[3.5], np.full(10, 2.1)])
    noise_per_row = np.concatenate([[1.0], np.full(10, 0.5)])
    gp = _ExactNumpyGP(
        X_train,
        y_train,
        lengthscales=np.array([0.04, 0.04]),
        signal_var=2.0,
        obs_noise_var=noise_per_row,
        prior_mean=0.5,
    )

    alloc = VOIAllocator(n_candidates=128, n_reference=64, min_separation=0.05)
    batch = alloc.allocate(
        gp,
        X_train,
        y_train,
        bounds,
        q=3,
        assay_noise_var=0.8,
        seed=0,
    )

    assert isinstance(batch, ActionBatch)
    assert len(batch.slots) == 3
    assert batch.X.shape == (3, 2)
    assert batch.slots[0].kind == "replicate"
    assert batch.slots[0].expected_voi > 0.0
    assert np.allclose(batch.slots[0].x, [0.80, 0.80])
    assert "KG=" in batch.slots[0].reason
    assert "noise_sd=" in batch.slots[0].reason
    # Replicates in the same batch must be deduplicated
    rep_slots = [s for s in batch.slots if s.kind == "replicate"]
    for i in range(len(rep_slots)):
        for j in range(i + 1, len(rep_slots)):
            assert proposal_separation(rep_slots[i].x, rep_slots[j].x, bounds[0], bounds[1]) > 1e-4

    # Pending top recipe must be excluded from replicate candidates
    batch_pend = alloc.allocate(
        gp,
        X_train,
        y_train,
        bounds,
        q=2,
        assay_noise_var=0.8,
        pending=np.array([[0.80, 0.80]]),
        seed=0,
    )
    for slot in batch_pend.slots:
        assert not np.allclose(slot.x, [0.80, 0.80], atol=1e-4)


def test_voi_allocator_explores_when_noise_tiny_or_top_well_replicated():
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    # Case 1: Tiny assay noise + sparse training data -> explore wins
    X_sparse = np.array([[0.2, 0.2], [0.8, 0.8]])
    y_sparse = np.array([1.0, 0.9])
    gp_clean = _ExactNumpyGP(
        X_sparse,
        y_sparse,
        lengthscales=np.array([0.25, 0.25]),
        signal_var=1.0,
        obs_noise_var=1e-6,
        prior_mean=0.9,
    )
    alloc = VOIAllocator(n_candidates=128, n_reference=64, min_separation=0.05)
    batch_clean = alloc.allocate(
        gp_clean, X_sparse, y_sparse, bounds, q=3, assay_noise_var=1e-6, seed=1
    )
    assert len(batch_clean.slots) == 3
    assert all(isinstance(s, ActionSlot) for s in batch_clean.slots)
    assert batch_clean.slots[0].kind == "explore"
    assert batch_clean.slots[0].expected_voi > 0.0

    # Case 2: Top observed recipes are already heavily replicated (tiny latent std),
    # while unexplored regions have high latent uncertainty and high prior mean -> explore wins
    X_rep = np.vstack([
        np.tile([0.2, 0.2], (20, 1)),
        np.tile([0.3, 0.3], (20, 1)),
    ])
    y_rep = np.concatenate([np.full(20, 1.0), np.full(20, 0.9)])
    gp_well_rep = _ExactNumpyGP(
        X_rep,
        y_rep,
        lengthscales=np.array([0.2, 0.2]),
        signal_var=1.5,
        obs_noise_var=0.05,
        prior_mean=0.9,
    )
    batch_rep = alloc.allocate(
        gp_well_rep, X_rep, y_rep, bounds, q=3, assay_noise_var=0.5, seed=2
    )
    assert batch_rep.slots[0].kind == "explore"
    assert batch_rep.slots[0].expected_voi > 0.0


def test_voi_allocator_constraints_marginal_fallback_and_auto_noise():
    bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    X_train = np.array([
        [0.2, 0.5, 0.2],
        [0.2, 0.5, 0.2],
        [0.4, 0.5, 0.3],
        [0.4, 0.5, 0.3],
    ])
    y_train = np.array([1.0, 1.2, 1.4, 1.5])
    # Marginal-only surrogate (no posterior_cov)
    marginal = _MarginalOnlySurrogate(
        lambda X: (0.8 * X[:, 0] + 0.4 * X[:, 2], np.full(len(X), 0.35))
    )
    fixed = {1: 0.5}
    # x0 + x2 <= 1.2  <=>  -x0 - x2 >= -1.2
    constraints = [([0, 2], [-1.0, -1.0], -1.2)]

    alloc = VOIAllocator(n_candidates=128, n_reference=64, min_separation=0.05)
    batch = alloc.allocate(
        marginal,
        X_train,
        y_train,
        bounds,
        q=4,
        assay_noise_var=None,  # auto-estimated from replicated X_train/y_train
        fixed_features=fixed,
        inequality_constraints=constraints,
        seed=5,
    )

    bx = batch.X
    assert bx.shape == (4, 3)
    assert np.all(bx >= bounds[0] - 1e-12)
    assert np.all(bx <= bounds[1] + 1e-12)
    assert np.allclose(bx[:, 1], 0.5)
    assert np.all(bx[:, 0] + bx[:, 2] <= 1.2 + 1e-6)


def test_joint_qlognei_is_torch_free() -> None:
    """Importing and running JointQLogNEI must never import torch/botorch/gpytorch."""
    import textwrap

    code = textwrap.dedent("""\
        import sys
        import numpy as np
        from kalos.core.frontier import JointQLogNEI, WarpedSparseGP

        loaded_at_import = "torch" in sys.modules
        rng = np.random.default_rng(11)
        bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        X = rng.uniform(0.0, 1.0, size=(12, 2))
        y = np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0.0, 0.05, size=12)
        gp = WarpedSparseGP().fit(X, y, bounds, seed=11)
        pts = JointQLogNEI(n_mc=16, n_restarts=2, raw_samples=32).propose(
            gp, bounds, q=3, X_train=X, y_train=y, seed=11
        )
        loaded_at_call = "torch" in sys.modules
        print(loaded_at_import, loaded_at_call, pts.shape[0], pts.shape[1])
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False False 3 2"


def test_joint_qlognei_gradient_matches_finite_differences() -> None:
    """Reverse-mode Murray Cholesky pullback matches central FD in warp and Tobit modes."""
    from kalos.core.frontier.warped_sparse_gp import WarpedSparseGP

    rng = np.random.default_rng(31)
    d = 3
    bounds = np.array([[0.0, -1.0, 2.0], [2.0, 3.0, 6.0]])
    X = rng.uniform(bounds[0], bounds[1], size=(18, d))
    y = np.exp(0.4 * X[:, 0]) + np.sin(X[:, 1]) - 0.2 * X[:, 2] + rng.normal(0.0, 0.05, size=18)
    pending = np.array([[1.2, 0.5, 4.1]])
    fixed = {1: 0.8}
    acq = JointQLogNEI(n_mc=16, n_restarts=2, raw_samples=32)

    # 1. Exact Cholesky + Kumaraswamy input warp + Box-Cox output warp + pending + minimize
    gp_chol = WarpedSparseGP(
        use_input_warp=True, use_output_warp=True, sparsity_strength=0.05, robust=False
    ).fit(X, y, bounds, seed=31)
    assert gp_chol._active_input_warp is True
    ctx_c = gp_chol.qlognei_context(X)
    obj_c, _, free_c = acq._build_objective(
        ctx_c, q=2, pending=pending, fixed=fixed, maximize=False, seed=31
    )
    assert free_c == [0, 2]
    x0 = np.array([0.7, 3.3, 1.4, 4.8])
    _, g_analytic = obj_c(x0)
    eps = 1e-6
    g_fd = np.zeros_like(x0)
    for k in range(len(x0)):
        dx = np.zeros_like(x0)
        dx[k] = eps
        f_p, _ = obj_c(x0 + dx)
        f_m, _ = obj_c(x0 - dx)
        g_fd[k] = (f_p - f_m) / (2.0 * eps)
    assert np.allclose(g_analytic, g_fd, rtol=1e-3, atol=2e-4)

    # 2. Tobit Laplace censored mode + maximize
    cens = X[:, 0] < 0.7
    lod = 2.0
    y_c = np.where(cens, lod, np.maximum(y, lod + 0.1))
    gp_lap = WarpedSparseGP(
        use_input_warp=False, use_output_warp=True, sparsity_strength=0.05, robust=False
    ).fit(X, y_c, bounds, censored_mask=cens, lod=lod, seed=32)
    assert gp_lap._is_laplace is True
    ctx_l = gp_lap.qlognei_context()
    obj_l, _, free_l = acq._build_objective(
        ctx_l, q=2, pending=None, fixed={}, maximize=True, seed=32
    )
    assert free_l == [0, 1, 2]
    x1 = np.array([0.9, 0.2, 3.5, 1.5, 1.8, 5.1])
    _, g_ana_l = obj_l(x1)
    g_fd_l = np.zeros_like(x1)
    for k in range(len(x1)):
        dx = np.zeros_like(x1)
        dx[k] = eps
        f_p, _ = obj_l(x1 + dx)
        f_m, _ = obj_l(x1 - dx)
        g_fd_l[k] = (f_p - f_m) / (2.0 * eps)
    assert np.allclose(g_ana_l, g_fd_l, rtol=1e-3, atol=1e-4)


def test_joint_qlognei_bounds_fixed_separation_constraints_and_determinism() -> None:
    """Verify JointQLogNEI respects bounds, fixed_features, min_separation, constraints, and seed."""
    from kalos.core.frontier.warped_sparse_gp import WarpedSparseGP

    rng = np.random.default_rng(55)
    d = 4
    bounds = np.array([[-1.0, 0.0, 10.0, -2.0], [3.0, 5.0, 20.0, 2.0]])
    X = rng.uniform(bounds[0], bounds[1], size=(18, d))
    y = -(X[:, 0] - 1.2) ** 2 + 0.3 * X[:, 2] + rng.normal(0.0, 0.05, size=18)
    gp = WarpedSparseGP(sparsity_strength=0.05).fit(X, y, bounds, seed=55)

    acq = JointQLogNEI(n_mc=32, n_restarts=4, raw_samples=128, min_separation=0.05)
    fixed = {1: 2.5}
    pending = np.array([[0.5, 2.5, 14.0, 0.0]])

    p1 = acq.propose(gp, bounds, q=4, X_train=X, y_train=y, pending=pending, fixed_features=fixed, seed=5)
    p2 = acq.propose(gp, bounds, q=4, X_train=X, y_train=y, pending=pending, fixed_features=fixed, seed=5)
    assert p1.shape == (4, d)
    assert np.allclose(p1, p2, atol=1e-12)
    assert np.all(p1 >= bounds[0] - 1e-12)
    assert np.all(p1 <= bounds[1] + 1e-12)
    assert np.allclose(p1[:, 1], 2.5)

    refs = np.vstack([X, pending])
    for i, row in enumerate(p1):
        for r in refs:
            assert proposal_separation(row, r, bounds[0], bounds[1]) >= 0.05 - 1e-9
        for j in range(i + 1, len(p1)):
            assert proposal_separation(row, p1[j], bounds[0], bounds[1]) >= 0.05 - 1e-9

    # Linear constraint path delegates safely to AdaptiveTuRBO and respects constraint
    constraints = [([0, 2], [-1.0, -0.1], -2.0)]  # x0 + 0.1*x2 <= 2.0
    pc = acq.propose(
        gp, bounds, q=3, X_train=X, y_train=y, fixed_features=fixed, inequality_constraints=constraints, seed=9
    )
    assert 1 <= pc.shape[0] <= 3
    assert np.allclose(pc[:, 1], 2.5)
    assert np.all(pc[:, 0] + 0.1 * pc[:, 2] <= 2.0 + 1e-6)
