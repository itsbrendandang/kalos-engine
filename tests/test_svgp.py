"""Tests for SparseVariationalSurrogate (kalos.core.svgp)."""
from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest
from botorch.models.gpytorch import GPyTorchModel

from kalos.core.svgp import SparseVariationalSurrogate


def test_svgp_module_import_is_torch_free():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kalos.core.svgp; "
            "print('torch' in sys.modules, 'botorch' in sys.modules, 'gpytorch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False False False"


def test_svgp_2500_rows_fit_accuracy_speed_and_diagnostics():
    rng = np.random.default_rng(0)
    n, d = 2500, 3
    bounds = np.array([[-2.0, -2.0, -2.0], [4.0, 4.0, 4.0]], float)
    X = rng.uniform(-2.0, 4.0, (n, d))
    y_true = 20.0 + 3.0 * np.sin(X[:, 0]) + 1.5 * X[:, 1] - 0.5 * X[:, 2] ** 2
    y = y_true + rng.normal(0.0, 0.25, n)

    t0 = time.perf_counter()
    s = SparseVariationalSurrogate().fit(X, y, bounds=bounds, num_inducing=256, max_iter=200, lr=0.05, seed=0)
    elapsed = time.perf_counter() - t0

    assert isinstance(s.model, GPyTorchModel)
    assert elapsed < 20.0
    assert s.best_f == pytest.approx(float(y.max()))
    diag = s.fit_diagnostics
    assert diag["surrogate_kind"] == "svgp"
    assert diag["num_inducing"] == 256
    assert diag["n_train"] == 2500
    assert np.isfinite(diag["final_elbo"])

    X_test = rng.uniform(-2.0, 4.0, (200, d))
    y_test = 20.0 + 3.0 * np.sin(X_test[:, 0]) + 1.5 * X_test[:, 1] - 0.5 * X_test[:, 2] ** 2
    mu, sd = s.posterior(X_test, observation_noise=False)
    mu_obs, sd_obs = s.posterior(X_test, observation_noise=True)
    assert mu.shape == (200,) and sd.shape == (200,)
    assert np.corrcoef(mu, y_test)[0, 1] > 0.95
    assert np.allclose(mu, mu_obs, atol=1e-12)
    assert np.all(sd_obs >= sd - 1e-12)


def test_svgp_uncertainty_grows_away_from_data():
    rng = np.random.default_rng(1)
    n, d = 400, 2
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)
    X = rng.uniform(0.35, 0.65, (n, d))
    y = np.sin(5.0 * X[:, 0]) + X[:, 1] + rng.normal(0.0, 0.05, n)

    s = SparseVariationalSurrogate().fit(X, y, bounds=bounds, num_inducing=128, max_iter=200, lr=0.05, seed=4)
    _, sd_near = s.posterior(np.array([[0.5, 0.5]]))
    _, sd_far = s.posterior(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.all(sd_far > sd_near * 1.5)


def test_svgp_determinism_with_seed_and_fixed_noise():
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, (300, 2))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0.0, 0.1, 300)
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)

    a = SparseVariationalSurrogate().fit(X, y, bounds=bounds, noise=0.01, num_inducing=64, max_iter=80, seed=11)
    b = SparseVariationalSurrogate().fit(X, y, bounds=bounds, noise=0.01, num_inducing=64, max_iter=80, seed=11)
    mu_a, sd_a = a.posterior(X[:20], observation_noise=True)
    mu_b, sd_b = b.posterior(X[:20], observation_noise=True)
    _, sd_lat = a.posterior(X[:20], observation_noise=False)

    assert np.array_equal(mu_a, mu_b)
    assert np.array_equal(sd_a, sd_b)
    assert a.fit_diagnostics["final_elbo"] == b.fit_diagnostics["final_elbo"]
    assert np.all(sd_a > sd_lat)


def test_svgp_validation_and_before_fit_errors():
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]], float)
    s = SparseVariationalSurrogate()
    with pytest.raises(RuntimeError, match="fit"):
        s.posterior(np.array([[0.5, 0.5]]))
    with pytest.raises(RuntimeError, match="fit"):
        _ = s.best_f

    with pytest.raises(ValueError, match="non-empty"):
        s.fit(np.empty((0, 2)), np.array([]), bounds=bounds)
    with pytest.raises(ValueError, match="same number of rows"):
        s.fit(np.ones((4, 2)), np.ones(3), bounds=bounds)
    with pytest.raises(ValueError, match="finite"):
        s.fit(np.array([[0.5, np.nan], [0.2, 0.3]]), np.array([1.0, 2.0]), bounds=bounds)
    with pytest.raises(ValueError, match="finite"):
        s.fit(np.array([[0.5, 0.1], [0.2, 0.3]]), np.array([1.0, np.inf]), bounds=bounds)
    with pytest.raises(ValueError, match="non-negative"):
        s.fit(np.array([[0.5, 0.1], [0.2, 0.3]]), np.array([1.0, 2.0]), bounds=bounds, noise=-0.01)

    s.fit(np.array([[0.1, 0.2], [0.8, 0.9]]), np.array([1.0, 2.0]), bounds=bounds, num_inducing=8, max_iter=10)
    assert s.fit_diagnostics["num_inducing"] == 2
    with pytest.raises(ValueError, match="finite"):
        s.posterior(np.array([[0.5, np.nan]]))
