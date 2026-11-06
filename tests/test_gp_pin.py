"""The GP kernel and priors are pinned (kalos/core/gp_pin.py).

BoTorch's SingleTaskGP default is an ARD RBF kernel with a dimension-scaled
LogNormal lengthscale prior and no ScaleKernel; the docs used to say Matern.
These tests fail if a BoTorch upgrade changes any of it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")

from botorch.models.kernels.categorical import CategoricalKernel  # noqa: E402
from gpytorch.kernels import RBFKernel, ScaleKernel  # noqa: E402
from gpytorch.likelihoods import (  # noqa: E402
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch.priors import LogNormalPrior  # noqa: E402

from kalos.core.gp_pin import (  # noqa: E402
    KERNEL_CATEGORICAL,
    KERNEL_CONTINUOUS,
    KERNEL_MIXED,
    KernelPinMismatch,
    check_pinned_gp,
)
from kalos.core.surrogate import Surrogate  # noqa: E402

RNG = np.random.default_rng(0)


def _rbfs(model):
    return [m for m in model.covar_module.modules() if isinstance(m, RBFKernel)]


def _assert_rbf(k, n_cont):
    assert type(k) is RBFKernel and k.ard_num_dims == n_cont
    p = k.lengthscale_prior
    assert isinstance(p, LogNormalPrior)
    assert float(p.loc) == pytest.approx(math.sqrt(2) + math.log(n_cont) / 2)
    assert float(p.scale) == pytest.approx(math.sqrt(3))
    assert float(k.raw_lengthscale_constraint.lower_bound) == pytest.approx(0.025)


def _assert_noise(lik):
    assert type(lik) is GaussianLikelihood
    p = lik.noise_covar.noise_prior
    assert isinstance(p, LogNormalPrior)
    assert (float(p.loc), float(p.scale)) == pytest.approx((-4.0, 1.0))
    assert float(lik.noise_covar.raw_noise_constraint.lower_bound) == pytest.approx(1e-4)


def test_continuous_path_is_ard_rbf_without_scale_kernel():
    X = RNG.uniform(0, 1, (10, 3))
    s = Surrogate().fit(X, X.sum(1), bounds=np.array([[0, 0, 0], [1, 1, 1]]))
    assert s.fit_diagnostics["kernel"] == KERNEL_CONTINUOUS
    assert not isinstance(s.model.covar_module, ScaleKernel)
    _assert_rbf(s.model.covar_module, 3)
    _assert_noise(s.model.likelihood)


def test_mixed_path_is_rbf_plus_categorical():
    X = np.column_stack([RNG.uniform(0, 1, (12, 2)), RNG.integers(0, 3, 12)])
    s = Surrogate().fit(X, X[:, 0] + X[:, 2], bounds=np.array([[0, 0, 0], [1, 1, 2]]),
                        cat_dims=[2])
    assert s.fit_diagnostics["kernel"] == KERNEL_MIXED
    rbfs = _rbfs(s.model)
    assert len(rbfs) == 2
    for k in rbfs:
        _assert_rbf(k, 2)
    assert any(isinstance(m, CategoricalKernel) for m in s.model.covar_module.modules())
    _assert_noise(s.model.likelihood)


def test_all_categorical_and_fixed_noise_paths():
    X = RNG.integers(0, 3, (12, 1)).astype(float)
    s = Surrogate().fit(X, X[:, 0] + RNG.normal(0, 0.1, 12), bounds=np.array([[0], [2]]),
                        cat_dims=[0])
    assert s.fit_diagnostics["kernel"] == KERNEL_CATEGORICAL
    Xc = RNG.uniform(0, 1, (10, 2))
    f = Surrogate().fit(Xc, Xc.sum(1), bounds=np.array([[0, 0], [1, 1]]), noise=0.01)
    assert isinstance(f.model.likelihood, FixedNoiseGaussianLikelihood)
    assert f.fit_diagnostics["kernel"] == KERNEL_CONTINUOUS


def test_a_changed_prior_is_refused():
    X = RNG.uniform(0, 1, (10, 2))
    s = Surrogate().fit(X, X.sum(1), bounds=np.array([[0, 0], [1, 1]]))
    s.model.covar_module.register_prior(
        "lengthscale_prior", LogNormalPrior(0.0, 1.0), "lengthscale"
    )
    with pytest.raises(KernelPinMismatch):
        check_pinned_gp(s.model, d=2, cat_dims=[], fixed_noise=False)


def test_analysis_reports_the_kernel():
    from kalos.domains import ColumnRoles
    from kalos.portal.analysis import _analyze

    n = 12
    df = pd.DataFrame({"a": RNG.uniform(0, 1, n), "b": RNG.uniform(0, 1, n)})
    df["y"] = df.a * 2 + df.b
    r = _analyze(df, roles=ColumnRoles.from_dict({"target": "y", "features": ["a", "b"]}))
    assert r["model_diagnostics"]["kernel"] == KERNEL_CONTINUOUS
