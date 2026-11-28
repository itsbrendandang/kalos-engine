"""Sparse / Variational GP surrogate for large datasets (>2,000 rows).

Uses GPyTorch `ApproximateGP` (`CholeskyVariationalDistribution` +
`VariationalStrategy` + `VariationalELBO`) wrapped as a BoTorch
`ApproximateGPyTorchModel` (`GPyTorchModel` subclass) with `Normalize` input
transform and `Standardize(m=1)` outcome transform on CPU float64.

Torch/BoTorch/GPyTorch are imported lazily inside methods so importing
`kalos.core.svgp` stays torch-free.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def _select_inducing_points(Xt_norm: Any, num_inducing: int, seed: int):
    """Deterministically select `m = min(num_inducing, n)` inducing points in
    the normalized input space."""
    import torch

    n = int(Xt_norm.shape[0])
    m = min(int(num_inducing), n)
    if m == n:
        return Xt_norm.clone()
    gen = torch.Generator(device=Xt_norm.device).manual_seed(int(seed))
    idx = torch.randperm(n, generator=gen, device=Xt_norm.device)[:m]
    return Xt_norm[idx].clone()


def _build_svgp_botorch_model(
    inducing_points: Any,
    input_transform: Any,
    outcome_transform: Any,
    *,
    yvar_std: Any = None,
    fixed_noise_std_val: float | None = None,
):
    """Build the GPyTorch `ApproximateGP` + BoTorch `ApproximateGPyTorchModel`."""
    import gpytorch
    import torch
    from botorch.models.approximate_gp import ApproximateGPyTorchModel
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )

    d = int(inducing_points.shape[-1])

    class _SVGP(ApproximateGP):
        def __init__(self, z: torch.Tensor) -> None:
            vd = CholeskyVariationalDistribution(z.size(0))
            vs = VariationalStrategy(self, z, vd, learn_inducing_locations=True)
            super().__init__(vs)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
            )

        def forward(self, x: torch.Tensor):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    class _BoTorchSVGP(ApproximateGPyTorchModel):
        def __init__(
            self,
            gp: ApproximateGP,
            likelihood: Any,
            fixed_std_var: float | None,
        ) -> None:
            super().__init__(model=gp, likelihood=likelihood, num_outputs=1)
            self._fixed_std_var = fixed_std_var

        @property
        def batch_shape(self) -> torch.Size:
            return torch.Size([])

        def posterior(
            self,
            X: torch.Tensor,
            output_indices: list[int] | None = None,
            observation_noise: bool = False,
            posterior_transform: Any = None,
        ):
            if observation_noise and self._fixed_std_var is not None:
                from botorch.posteriors.gpytorch import GPyTorchPosterior
                from gpytorch.distributions import MultivariateNormal

                self.eval()
                Xt = self.transform_inputs(X)
                dist = self.model(Xt)
                noise_t = torch.full(
                    Xt.shape[:-1],
                    self._fixed_std_var,
                    dtype=Xt.dtype,
                    device=Xt.device,
                )
                dist = self.likelihood(dist, noise=noise_t)
                post: Any = (
                    GPyTorchPosterior(distribution=dist)
                    if isinstance(dist, MultivariateNormal)
                    else super().posterior(X, output_indices=output_indices, observation_noise=False)
                )
                if hasattr(self, "outcome_transform"):
                    post = self.outcome_transform.untransform_posterior(post, X=Xt)
                if posterior_transform is not None:
                    post = posterior_transform(posterior=post, X=Xt)
                return post
            return super().posterior(
                X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

    gp = _SVGP(inducing_points)
    if yvar_std is not None:
        likelihood: Any = FixedNoiseGaussianLikelihood(noise=yvar_std, learn_additional_noise=False)
    else:
        likelihood = GaussianLikelihood()
    botorch_model = _BoTorchSVGP(gp, likelihood, fixed_noise_std_val)
    botorch_model.input_transform = input_transform
    botorch_model.outcome_transform = outcome_transform
    return botorch_model, gp, likelihood


class SparseVariationalSurrogate:
    """Sparse Variational GP (SVGP) matching `Surrogate`'s core interface."""

    def __init__(self) -> None:
        self.model: Any = None
        self._X: Any = None
        self._y: np.ndarray | None = None
        self._cat_dims: list[int] | None = None
        self.fit_diagnostics: dict[str, Any] = {}

    def fit(
        self,
        X,
        y,
        bounds,
        *,
        noise=None,
        num_inducing: int = 256,
        max_iter: int = 200,
        lr: float = 0.05,
        seed: int = 0,
    ) -> SparseVariationalSurrogate:
        """Fit an SVGP on `(X, y)` normalized to `bounds` `(2, d)`."""
        import torch
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import VariationalELBO

        from .surrogate import DEVICE, DTYPE, sanitize_bounds

        Xa = np.asarray(X, float)
        ya = np.asarray(y, float).reshape(-1)
        if Xa.ndim != 2 or Xa.shape[0] == 0:
            raise ValueError("X must be a non-empty 2-D array (n x d)")
        if Xa.shape[0] != ya.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
            raise ValueError("X and y must be finite (no NaN/inf)")
        if int(num_inducing) < 1:
            raise ValueError("num_inducing must be >= 1")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if not (float(lr) > 0.0 and np.isfinite(float(lr))):
            raise ValueError("lr must be positive and finite")

        n, d = Xa.shape
        lower, upper = sanitize_bounds(bounds)
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)

        yvar_t: torch.Tensor | None = None
        if noise is not None:
            noise_arr = np.broadcast_to(np.asarray(noise, float), (n,)).astype(float)
            if not np.isfinite(noise_arr).all() or (noise_arr < 0).any():
                raise ValueError("noise must be finite and non-negative")
            yvar_t = torch.as_tensor(noise_arr, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        t0 = time.perf_counter()
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        in_tf = Normalize(d=d, bounds=box).to(dtype=DTYPE, device=DEVICE)
        out_tf = Standardize(m=1).to(dtype=DTYPE, device=DEVICE)
        out_tf.train()
        yt_std, yvar_std = out_tf(yt, yvar_t)
        out_tf.eval()
        yt_std_1d = yt_std.squeeze(-1)
        yvar_std_1d = yvar_std.squeeze(-1) if yvar_std is not None else None
        fixed_std_val = float(yvar_std_1d.mean().item()) if yvar_std_1d is not None else None

        with torch.no_grad():
            Xt_norm = in_tf(Xt)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            inducing = _select_inducing_points(Xt_norm, int(num_inducing), int(seed))
            m_inducing = int(inducing.shape[0])
            botorch_model, gp, likelihood = _build_svgp_botorch_model(
                inducing,
                in_tf,
                out_tf,
                yvar_std=yvar_std_1d,
                fixed_noise_std_val=fixed_std_val,
            )
            botorch_model.to(dtype=DTYPE, device=DEVICE)
            gp.train()
            likelihood.train()

            params: list[dict[str, Any]] = [{"params": gp.parameters()}]
            if yvar_std_1d is None:
                params.append({"params": likelihood.parameters()})
            optimizer = torch.optim.Adam(params, lr=float(lr))
            mll = VariationalELBO(likelihood, gp, num_data=n)

            batch_size = min(256, n)
            gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
            final_elbo = float("nan")
            for _ in range(int(max_iter)):
                if batch_size < n:
                    b_idx = torch.randint(0, n, (batch_size,), generator=gen, device=DEVICE)
                    xb = Xt_norm[b_idx]
                    yb = yt_std_1d[b_idx]
                    nb = yvar_std_1d[b_idx] if yvar_std_1d is not None else None
                else:
                    xb = Xt_norm
                    yb = yt_std_1d
                    nb = yvar_std_1d
                optimizer.zero_grad()
                out = gp(xb)
                elbo = mll(out, yb, noise=nb) if nb is not None else mll(out, yb)
                loss = -elbo
                loss.backward()
                optimizer.step()
                final_elbo = float(elbo.detach().item())

        botorch_model.eval()
        self.model = botorch_model
        self._X = Xt
        self._y = ya
        self._cat_dims = None
        self.fit_diagnostics = {
            "surrogate_kind": "svgp",
            "num_inducing": m_inducing,
            "n_train": n,
            "final_elbo": final_elbo,
            "max_iter": int(max_iter),
            "lr": float(lr),
            "seed": int(seed),
            "fit_seconds": round(time.perf_counter() - t0, 3),
        }
        return self

    def posterior(self, X, *, observation_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mean, std)` in the target's original units."""
        import torch

        from .surrogate import DEVICE, DTYPE

        if self.model is None:
            raise RuntimeError("fit() first")
        Xa = np.atleast_2d(np.asarray(X, float))
        if not np.isfinite(Xa).all():
            raise ValueError("X must be finite (no NaN/inf)")
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt, observation_noise=observation_noise)
            mean = post.mean.squeeze(-1).cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        return mean, std

    @property
    def best_f(self) -> float:
        if self._y is None:
            raise RuntimeError("fit() first")
        return float(self._y.max())


__all__ = ["SparseVariationalSurrogate"]
