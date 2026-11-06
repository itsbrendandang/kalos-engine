"""The GP kernel and priors, pinned by assertion.

What the surrogate uses is BoTorch's own default, checked here at every fit so
a BoTorch upgrade cannot change it silently (verified on BoTorch 0.18.1,
GPyTorch 1.15.2):

- Continuous design (`SingleTaskGP`): an ARD `RBFKernel`, with NO ScaleKernel
  (no output-scale parameter; the outcome is standardized). Lengthscale prior
  `LogNormal(loc = sqrt(2) + log(d) / 2, scale = sqrt(3))`, scaled by the
  input dimension d (Hvarfner et al. 2024). Lengthscale constraint
  `GreaterThan(0.025)`.
- Mixed design (`MixedSingleTaskGP`): the same ARD RBF over the continuous
  dims (d = number of continuous dims), plus a `CategoricalKernel` on the
  categorical dims, combined as `Scale(RBF + Scale(Cat)) + Scale(RBF x Cat)`.
  All categorical: `Scale(Cat)`. No output-scale prior.
- Inferred noise: `GaussianLikelihood` with a `LogNormal(-4, 1)` noise prior
  and a `GreaterThan(1e-4)` constraint (standardized units). Fixed noise
  (`train_Yvar`): `FixedNoiseGaussianLikelihood`, no prior.
- Mean: `ConstantMean`.

It is NOT Matern. Earlier comments said Matern; they were wrong.

Pinned by assertion, not by building the kernel by hand: constructing the same
kernel explicitly could still change numbers (parameter order and
initialization feed the fit and the acquisition), and every proposal would move.
The assertion leaves the model exactly as BoTorch builds it.
"""
from __future__ import annotations

import math
from typing import Any

KERNEL_CONTINUOUS = "rbf_ard"
KERNEL_MIXED = "rbf_ard+categorical"
KERNEL_CATEGORICAL = "categorical"

LENGTHSCALE_PRIOR_SCALE = math.sqrt(3.0)
LENGTHSCALE_LOWER_BOUND = 0.025
NOISE_PRIOR_LOC = -4.0
NOISE_PRIOR_SCALE = 1.0
NOISE_LOWER_BOUND = 1e-4


class KernelPinMismatch(RuntimeError):
    """The GP BoTorch built is not the pinned kernel/prior (an upgrade changed
    a default). Fails the fit loudly instead of changing proposals silently."""


def lengthscale_prior_loc(d: int) -> float:
    """The dimension-scaled LogNormal loc: sqrt(2) + log(d) / 2."""
    return math.sqrt(2.0) + math.log(d) / 2.0


def _close(a: Any, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise KernelPinMismatch(f"GP default changed: {what} (see kalos/core/gp_pin.py)")


def _check_rbf(kernel: Any, n_cont: int) -> None:
    from gpytorch.kernels import RBFKernel
    from gpytorch.priors import LogNormalPrior

    _check(type(kernel) is RBFKernel, f"continuous kernel is {type(kernel).__name__}, not RBFKernel")
    _check(int(kernel.ard_num_dims or 1) == n_cont, "RBF is not ARD over the continuous dims")
    prior: Any = getattr(kernel, "lengthscale_prior", None)
    _check(isinstance(prior, LogNormalPrior), "lengthscale prior is not LogNormal")
    _check(_close(prior.loc, lengthscale_prior_loc(n_cont)), "lengthscale prior loc")
    _check(_close(prior.scale, LENGTHSCALE_PRIOR_SCALE), "lengthscale prior scale")
    lb = getattr(kernel.raw_lengthscale_constraint, "lower_bound", None)
    _check(lb is not None and _close(lb, LENGTHSCALE_LOWER_BOUND), "lengthscale lower bound")


def _check_likelihood(likelihood: Any, *, fixed_noise: bool) -> None:
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
    from gpytorch.priors import LogNormalPrior

    if fixed_noise:
        _check(isinstance(likelihood, FixedNoiseGaussianLikelihood), "fixed-noise likelihood")
        return
    _check(type(likelihood) is GaussianLikelihood, "likelihood is not GaussianLikelihood")
    prior: Any = getattr(likelihood.noise_covar, "noise_prior", None)
    _check(isinstance(prior, LogNormalPrior), "noise prior is not LogNormal")
    _check(_close(prior.loc, NOISE_PRIOR_LOC), "noise prior loc")
    _check(_close(prior.scale, NOISE_PRIOR_SCALE), "noise prior scale")
    lb = getattr(likelihood.noise_covar.raw_noise_constraint, "lower_bound", None)
    _check(lb is not None and _close(lb, NOISE_LOWER_BOUND), "noise lower bound")


def check_pinned_gp(model: Any, *, d: int, cat_dims: list[int], fixed_noise: bool) -> str:
    """Assert `model` (unfitted or fitted) is the pinned kernel and priors and
    return the kernel name for `model_diagnostics.kernel`. Raises
    `KernelPinMismatch` otherwise."""
    from botorch.models.kernels.categorical import CategoricalKernel as BtCategorical
    from gpytorch.kernels import AdditiveKernel, ProductKernel, ScaleKernel
    from gpytorch.means import ConstantMean

    _check(isinstance(model.mean_module, ConstantMean), "mean is not ConstantMean")
    _check_likelihood(model.likelihood, fixed_noise=fixed_noise)
    covar = model.covar_module
    n_cont = d - len(set(cat_dims))
    if not cat_dims:
        _check_rbf(covar, d)
        return KERNEL_CONTINUOUS
    if n_cont == 0:
        _check(isinstance(covar, ScaleKernel) and isinstance(covar.base_kernel, BtCategorical),
               "all-categorical kernel is not Scale(Categorical)")
        return KERNEL_CATEGORICAL
    _check(isinstance(covar, AdditiveKernel) and len(covar.kernels) == 2, "mixed kernel layout")
    sum_part, prod_part = covar.kernels
    _check(isinstance(sum_part, ScaleKernel) and isinstance(sum_part.base_kernel, AdditiveKernel),
           "mixed sum term")
    rbf_a, cat_a = sum_part.base_kernel.kernels
    _check_rbf(rbf_a, n_cont)
    _check(isinstance(cat_a, ScaleKernel) and isinstance(cat_a.base_kernel, BtCategorical),
           "mixed sum term categorical part")
    _check(isinstance(prod_part, ScaleKernel) and isinstance(prod_part.base_kernel, ProductKernel),
           "mixed product term")
    rbf_b, cat_b = prod_part.base_kernel.kernels
    _check_rbf(rbf_b, n_cont)
    _check(isinstance(cat_b, BtCategorical), "mixed product term categorical part")
    return KERNEL_MIXED
