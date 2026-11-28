"""Tests for HybridSurrogate BoTorch Model bridge and torch-free import contract."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from kalos.core.mechanistic import HybridSurrogate, ODEModel
from kalos.core.optimize import propose, propose_batch


def test_mechanistic_module_import_is_torch_free():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kalos.core.mechanistic; "
            "print('torch' in sys.modules, 'botorch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False False"


def test_hybrid_posterior_equals_mechanistic_plus_residual_and_preserves_gradients():
    rng = np.random.default_rng(0)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)
    X = rng.uniform(0.0, 1.0, (16, 2))

    def mech(Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        return 5.0 * Z[:, 0] + 2.0 * Z[:, 1]

    y = mech(X) + np.sin(4.0 * X[:, 0]) + rng.normal(0.0, 0.05, len(X))
    hybrid = HybridSurrogate(mech)
    assert hybrid.model is None
    with pytest.raises(RuntimeError, match="fit"):
        hybrid.to_botorch_model()
    with pytest.raises(RuntimeError, match="fit"):
        hybrid.posterior(X)

    hybrid.fit(X, y, bounds=bounds)
    assert hybrid._X is not None and hybrid._bounds is not None
    assert hybrid.best_f == pytest.approx(float(y.max()))

    X_probe = rng.uniform(0.0, 1.0, (5, 2))
    mu_h, sd_h = hybrid.posterior(X_probe, observation_noise=False)
    mu_h_obs, sd_h_obs = hybrid.posterior(X_probe, observation_noise=True)
    mu_r, sd_r = hybrid.residual.posterior(X_probe, observation_noise=False)
    assert np.allclose(mu_h, mech(X_probe) + mu_r)
    assert np.allclose(sd_h, sd_r)
    assert np.all(sd_h_obs >= sd_h - 1e-12)
    assert np.allclose(mu_h_obs, mu_h, atol=1e-12)

    bt_model = hybrid.model
    assert bt_model is hybrid.to_botorch_model()
    assert bt_model.num_outputs == 1

    Xt = torch.tensor(X_probe, dtype=torch.double, requires_grad=True)
    post = bt_model.posterior(Xt.unsqueeze(1))  # (b=5, q=1, d=2)
    assert post.mean.shape == (5, 1, 1)
    assert np.allclose(post.mean.detach().numpy().reshape(-1), mu_h, atol=1e-9)
    assert np.allclose(post.variance.sqrt().detach().numpy().reshape(-1), sd_h, atol=1e-9)

    samples = post.rsample(torch.Size([4]))
    assert samples.shape == (4, 5, 1, 1)
    post.mean.sum().backward()
    assert Xt.grad is not None and torch.isfinite(Xt.grad).all()


def test_hybrid_surrogate_works_end_to_end_with_propose_batch():
    rng = np.random.default_rng(1)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)
    X = rng.uniform(0.0, 1.0, (14, 2))

    def mech(Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        return -((Z[:, 0] - 0.75) ** 2 + (Z[:, 1] - 0.25) ** 2)

    y = mech(X) + 0.1 * X[:, 0] + rng.normal(0.0, 0.02, len(X))
    hybrid = HybridSurrogate(mech).fit(X, y, bounds=bounds)

    batch = propose_batch(hybrid, bounds, q=2, num_restarts=3, raw_samples=32, seed=0)
    assert batch.X.shape == (2, 2)
    assert np.all(batch.X >= 0.0) and np.all(batch.X <= 1.0)

    # Also verify minimize + fixed_features + linear_constraints + pending
    nxt = propose(
        hybrid,
        bounds,
        q=2,
        num_restarts=3,
        raw_samples=32,
        seed=0,
        maximize=False,
        pending=np.array([[0.2, 0.5]]),
        fixed_features={1: 0.5},
        linear_constraints=[([0], [1.0], 0.1)],
    )
    assert nxt.shape == (2, 2)
    assert np.allclose(nxt[:, 1], 0.5)
    assert np.all(nxt[:, 0] >= 0.1 - 1e-6)


def test_hybrid_ode_nan_safe_during_fit_and_propose():
    ode = ODEModel(
        rhs=lambda t, s, p: [np.inf] if p[0] > 0.8 else [-p[0] * s[0]],
        x_to_params=lambda x: (float(x[0]),),
        x_to_y0=lambda x: [float(x[1]) + 1.0],
        readout=lambda t, y: float(y[0, -1]),
        t_end=1.0,
    )
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, (14, 2))
    raw_mech = ode.evaluate(X)
    assert np.isnan(raw_mech[X[:, 0] > 0.8]).any()

    y = np.where(np.isfinite(raw_mech), raw_mech, 0.5) + 0.1 * X[:, 1]
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)
    hybrid = HybridSurrogate(ode).fit(X, y, bounds=bounds)
    mu, sd = hybrid.posterior(np.array([[0.9, 0.5], [0.2, 0.5]]))
    assert np.isfinite(mu).all() and np.isfinite(sd).all()

    batch = propose_batch(hybrid, bounds, q=2, num_restarts=2, raw_samples=16, seed=3)
    assert batch.X.shape == (2, 2) and np.isfinite(batch.X).all()
