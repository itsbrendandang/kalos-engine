"""ml-surrogate candidate arms (PROTOCOL.md section 9, ml-surrogate list).

Every arm here is an ENGINE-PATCH arm: the production `_analyze` path runs
unchanged around an in-process swap of `kalos.core.surrogate.Surrogate` (the
proposal fit, the constraint model) AND `kalos.core.evaluation.Surrogate` (every
grouped-CV fold, so the trust verdict and the selection-aware intervals are
calibrated on the same model family that is served). Nothing in the engine is
edited.

    --arm kalos.bench.ml_v2.arms.surrogates:make_arm --arm-config '{"name": "...", ...}'

Configuration keys (all optional; the empty config is the reference surrogate
rebuilt through this module, a self-check):

- `kernel`: "rbf" (BoTorch 0.18 default: RBF, ARD, dimension-scaled LogNormal
  lengthscale prior) or "matern52" (same prior, Matern-5/2).
- `warp`: Kumaraswamy input warping (BoTorch `Warp`, LogNormal(0, 0.75^0.5)
  concentration priors) on the continuous inputs, after normalization to the
  design box; MAP-fitted jointly with the GP.
- `saas`: fully Bayesian SAAS GP (`SaasFullyBayesianSingleTaskGP`, NUTS via
  NumPyro; `nuts` = [warmup, samples, thinning]). Continuous designs only; a
  design with categorical dims falls back to the reference mixed GP (recorded).
- `noise_lb`: lower bound on the inferred noise variance in standardized units
  (reference: 1e-4). A larger floor stops small-n fits from interpolating noise.
- `hetero`: "resid" = most-likely heteroscedastic GP (Kersting et al. 2007, one
  EM step): fit the homoscedastic GP, fit a second GP to log squared in-sample
  residuals, refit with that noise fixed; the predictive sd at a new point adds
  the noise GP's prediction there. "rep" = on the replicate-aware path, replace
  the pooled assay floor sigma^2/n_reps by a per-recipe variance shrunk toward a
  log-variance-vs-mean fit over the replicated recipes (prior weight `rep_df`).
- `near_replicates`: rows within the engine's near-duplicate tolerance (0.05 of
  the column range, Chebyshev) are one recipe for the noise floor and the
  replicate-aware fit (the engine today uses exact input vectors there and
  clusters only for CV grouping).
- `name`: the arm name (unique; frozen arms are never renamed).
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from .kalos_ref import KalosEngineArm

_STASH = threading.local()


def _cfg_get(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    v = cfg.get(key, default)
    return default if v is None else v


def make_surrogate_class(cfg: Mapping[str, Any]) -> type:
    """A `kalos.core.surrogate.Surrogate` subclass with the configured model."""
    import torch
    from botorch.models import MixedSingleTaskGP, SingleTaskGP
    from botorch.models.transforms.input import ChainedInputTransform, Normalize, Warp
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.constraints import GreaterThan
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors import LogNormalPrior

    import kalos.core.surrogate as S

    kernel = str(_cfg_get(cfg, "kernel", "rbf"))
    warp = bool(_cfg_get(cfg, "warp", False))
    saas = bool(_cfg_get(cfg, "saas", False))
    nuts = tuple(int(v) for v in _cfg_get(cfg, "nuts", (256, 128, 16)))
    noise_lb = _cfg_get(cfg, "noise_lb", None)
    hetero = str(_cfg_get(cfg, "hetero", ""))
    rep_df = float(_cfg_get(cfg, "rep_df", 4.0))
    hetero_floor = float(_cfg_get(cfg, "hetero_floor", 1e-3))  # standardized variance floor

    def _input_tf(d: int, box: torch.Tensor, cont: list[int]) -> Any:
        if not cont:
            return None
        norm = Normalize(d=d, indices=cont if len(cont) < d else None, bounds=box[:, cont] if len(cont) < d else box)
        if not warp:
            return norm
        c1 = LogNormalPrior(0.0, 0.75 ** 0.5)
        c0 = LogNormalPrior(0.0, 0.75 ** 0.5)
        w = Warp(d=d, indices=cont, concentration1_prior=c1, concentration0_prior=c0)
        return ChainedInputTransform(normalize=norm, warp=w)

    def _likelihood() -> Any:
        if noise_lb is None:
            return None
        from botorch.models.utils.gpytorch_modules import (
            get_gaussian_likelihood_with_lognormal_prior,
        )

        lik = get_gaussian_likelihood_with_lognormal_prior()
        lik.noise_covar.register_constraint("raw_noise", GreaterThan(float(noise_lb)))
        return lik

    def _cont_kernel(d_cont: int, active: list[int] | None = None) -> Any:
        from botorch.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )

        return get_covar_module_with_dim_scaled_prior(
            ard_num_dims=d_cont, use_rbf_kernel=(kernel == "rbf"), active_dims=active)

    class CandidateSurrogate(S.Surrogate):
        candidate_config = dict(cfg)

        def __init__(self) -> None:
            super().__init__()
            self._noise_model: Any = None
            self._fully_bayesian = False

        # -- model construction ------------------------------------------------
        def _build(self, Xt: torch.Tensor, yt: torch.Tensor, box: torch.Tensor, cats: list[int],
                   yvar: torch.Tensor | None) -> Any:
            d = Xt.shape[-1]
            cont = [i for i in range(d) if i not in set(cats)]
            itf = _input_tf(d, box, cont)
            kw: dict[str, Any] = {"input_transform": itf, "outcome_transform": Standardize(m=1)}
            if yvar is not None:
                kw["train_Yvar"] = yvar
            elif _likelihood() is not None:
                kw["likelihood"] = _likelihood()
            if cats:
                if kernel != "rbf":
                    kw["cont_kernel_factory"] = lambda batch_shape, ard_num_dims, active_dims: _cont_kernel(
                        ard_num_dims, list(active_dims))
                return MixedSingleTaskGP(Xt, yt, cat_dims=cats, **kw)
            if kernel != "rbf":
                kw["covar_module"] = _cont_kernel(d)
            return SingleTaskGP(Xt, yt, **kw)

        def _fit_one(self, model: Any, restarts: int, restart_seed: int) -> tuple[dict, Any]:
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            if restarts > 1:
                return S._multi_start_fit(mll, restarts=int(restarts), seed=int(restart_seed))
            return dict(S._fit_mll_with_retry(mll)), model

        def _fit_saas(self, Xt: torch.Tensor, yt: torch.Tensor, box: torch.Tensor,
                      yvar: torch.Tensor | None, seed: int) -> tuple[dict, Any]:
            from botorch.fit import fit_fully_bayesian_model_nuts
            from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

            d = Xt.shape[-1]
            model = SaasFullyBayesianSingleTaskGP(
                Xt, yt, train_Yvar=yvar, input_transform=Normalize(d=d, bounds=box),
                outcome_transform=Standardize(m=1), use_input_warping=warp)
            w, s, t = nuts
            np_state = np.random.get_state()
            with torch.random.fork_rng():
                fit_fully_bayesian_model_nuts(model, warmup_steps=w, num_samples=s, thinning=t,
                                              disable_progbar=True, seed=int(seed))
            np.random.set_state(np_state)
            model.eval()
            return {"saas_nuts": [w, s, t], "jitter_rung": None}, model

        # -- heteroscedastic noise -------------------------------------------
        def _rep_noise(self, yvar: torch.Tensor) -> torch.Tensor:
            """Per-recipe variance shrunk toward a log-variance vs mean fit."""
            agg = getattr(_STASH, "agg", None)
            n = yvar.shape[0]
            if agg is None or len(agg[0]) != n:
                return yvar
            y_mean, y_var, n_reps = (np.asarray(a, float) for a in agg)
            pooled_mean_var = yvar.detach().cpu().numpy().reshape(-1) * n_reps  # sigma2 (same for all)
            sigma2 = float(np.median(pooled_mean_var))
            rep = (n_reps >= 2) & (y_var > 0)
            if rep.sum() < 4 or not np.isfinite(sigma2) or sigma2 <= 0:
                return yvar
            wts = n_reps[rep] - 1.0
            A = np.vstack([np.ones(rep.sum()), y_mean[rep]]).T
            coef, *_ = np.linalg.lstsq(A * np.sqrt(wts)[:, None], np.log(y_var[rep]) * np.sqrt(wts), rcond=None)
            # log s^2 of a normal sample is biased by digamma((n-1)/2) - log((n-1)/2); correct per recipe
            from scipy.special import digamma

            bias = digamma(wts / 2.0) - np.log(wts / 2.0)
            coef[0] -= float(np.average(bias, weights=wts))
            prior = np.exp(coef[0] + coef[1] * y_mean)
            # keep the prior within [1/10, 10] x the pooled floor (no wild extrapolation)
            prior = np.clip(prior, sigma2 / 10.0, sigma2 * 10.0)
            dfree = np.maximum(n_reps - 1.0, 0.0)
            post = (dfree * y_var + rep_df * prior) / (dfree + rep_df)
            out = post / np.maximum(n_reps, 1.0)
            self._rep_hetero = {"slope": float(coef[1]), "sigma2_pooled": sigma2}
            return torch.as_tensor(out, dtype=S.DTYPE).reshape(-1, 1)

        def _resid_noise_model(self, model: Any, Xt: torch.Tensor, yt: torch.Tensor, box: torch.Tensor,
                               cats: list[int]) -> Any:
            """Second GP on log squared in-sample residuals (original units)."""
            model.eval()
            with torch.no_grad():
                post = model.posterior(Xt)
                mu = post.mean.reshape(-1)
                var = post.variance.reshape(-1)
            r2 = (yt.reshape(-1) - mu) ** 2 + var
            ystd = float(yt.std()) or 1.0
            floor = hetero_floor * ystd ** 2
            z = torch.log(torch.clamp(r2, min=floor)).reshape(-1, 1)
            d = Xt.shape[-1]
            cont = [i for i in range(d) if i not in set(cats)]
            itf = Normalize(d=d, indices=cont if len(cont) < d else None,
                            bounds=box[:, cont] if len(cont) < d else box) if cont else None
            g = SingleTaskGP(Xt, z, input_transform=itf, outcome_transform=Standardize(m=1))
            S._fit_mll_with_retry(ExactMarginalLogLikelihood(g.likelihood, g))
            g.eval()
            return g, floor

        def _noise_at(self, Xt: torch.Tensor) -> torch.Tensor:
            g, floor = self._noise_model
            with torch.no_grad():
                p = g.posterior(Xt)
                # E[exp(Z)] for Z ~ N(m, v): exp(m + v / 2) (log-normal mean)
                v = torch.exp(p.mean.reshape(-1) + 0.5 * p.variance.reshape(-1))
            return torch.clamp(v, min=floor)

        # -- the Surrogate API -------------------------------------------------
        def fit(self, X, y, bounds, *, noise=None, cat_dims=None, check_conditioning: bool = False,  # type: ignore[override]
                restarts: int = 1, restart_seed: int = 0):
            Xa = np.asarray(X, float)
            ya = np.asarray(y, float).reshape(-1)
            if Xa.ndim != 2 or Xa.shape[0] == 0:
                raise ValueError("X must be a non-empty 2-D array (n x d)")
            if Xa.shape[0] != ya.shape[0]:
                raise ValueError("X and y must have the same number of rows")
            if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
                raise ValueError("X and y must be finite (no NaN/inf)")
            Xt = torch.as_tensor(Xa, dtype=S.DTYPE, device=S.DEVICE)
            yt = torch.as_tensor(ya, dtype=S.DTYPE, device=S.DEVICE).reshape(-1, 1)
            lower, upper = S.sanitize_bounds(bounds)
            box = torch.as_tensor(np.vstack([lower, upper]), dtype=S.DTYPE, device=S.DEVICE)
            yvar = None
            if noise is not None:
                noise_arr = np.broadcast_to(np.asarray(noise, float), (Xa.shape[0],)).astype(float)
                if not np.isfinite(noise_arr).all() or (noise_arr < 0).any():
                    raise ValueError("noise must be finite and non-negative")
                yvar = torch.as_tensor(noise_arr, dtype=S.DTYPE, device=S.DEVICE).reshape(-1, 1)
                if hetero == "rep":
                    yvar = self._rep_noise(yvar)
            cats = [int(i) for i in cat_dims] if cat_dims else []
            diag: dict[str, Any]
            if saas and not cats and Xa.shape[0] >= 3:
                diag, self.model = self._fit_saas(Xt, yt, box, yvar, restart_seed or 0)
                self._fully_bayesian = True
            else:
                self.model = self._build(Xt, yt, box, cats, yvar)
                diag, self.model = self._fit_one(self.model, restarts, restart_seed)
                if saas and cats:
                    diag["saas_fallback"] = "categorical dims: reference mixed GP"
                if hetero == "resid" and yvar is None and Xa.shape[0] >= 8:
                    try:
                        self._noise_model = self._resid_noise_model(self.model, Xt, yt, box, cats)
                        nv = self._noise_at(Xt).reshape(-1, 1)
                        m2 = self._build(Xt, yt, box, cats, nv)
                        d2, m2 = self._fit_one(m2, restarts, restart_seed)
                        self.model = m2
                        diag = {**d2, "hetero": "resid"}
                        yvar = nv
                    except Exception as exc:  # noqa: BLE001 - fall back to the homoscedastic fit
                        self._noise_model = None
                        diag["hetero_failed"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            if not self._fully_bayesian:
                diag.update(S.hyperparameter_report(self.model, fixed_noise=yvar is not None))
            if check_conditioning:
                try:
                    cond = S.kernel_condition_number(self.model)
                except Exception:  # noqa: BLE001 - batched (MCMC) models: max over samples
                    cond = _batched_condition(self.model)
                diag.update(S.condition_report(cond))
            if getattr(self, "_rep_hetero", None):
                diag["rep_hetero"] = self._rep_hetero
            self.fit_diagnostics = diag
            self._y = ya
            self._X = Xt
            self._cat_dims = cats or None
            return self

        def posterior(self, X, *, observation_noise: bool = False):
            assert self.model is not None, "fit() first"
            Xt = torch.as_tensor(np.asarray(X, float), dtype=S.DTYPE, device=S.DEVICE)
            self.model.eval()
            with torch.no_grad():
                if self._fully_bayesian:
                    post = self.model.posterior(Xt, observation_noise=observation_noise)
                    mean = post.mixture_mean.reshape(-1)
                    var = post.mixture_variance.reshape(-1)
                elif self._noise_model is not None:
                    post = self.model.posterior(Xt, observation_noise=False)
                    mean = post.mean.reshape(-1)
                    var = post.variance.reshape(-1)
                    if observation_noise:
                        var = var + self._noise_at(Xt)
                else:
                    post = self.model.posterior(Xt, observation_noise=observation_noise)
                    mean = post.mean.squeeze(-1)
                    var = post.variance.squeeze(-1)
                std = var.clamp_min(1e-12).sqrt()
            return mean.cpu().numpy(), std.cpu().numpy()

    CandidateSurrogate.__name__ = "CandidateSurrogate"
    return CandidateSurrogate


def _batched_condition(model: Any) -> float:
    import torch

    model.eval()
    with torch.no_grad():
        X0 = model.train_inputs[0]
        cov = model.likelihood(model.forward(X0)).covariance_matrix
        ev = torch.linalg.eigvalsh(cov)
    lo = ev.min(dim=-1).values
    hi = ev.max(dim=-1).values
    if bool((lo <= 0).any()):
        return float("inf")
    return float((hi / lo).max())


def _near_replicate_keys(frame: Any) -> list[str]:
    """Recipe keys where rows within the engine's near-duplicate tolerance of
    each other (Chebyshev, 0.05 of each column's observed range) are one recipe."""
    import pandas as pd

    from kalos.core.splits import row_hash_groups
    from kalos.portal.analysis_parts.cv_grouping import (
        NEAR_DUPLICATE_TOL,
        replicate_clusters,
    )

    df = pd.DataFrame(frame)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    lab = [c for c in df.columns if c not in num]
    if not num:
        return row_hash_groups(df)
    arr = df[num].to_numpy(float)
    widths = np.nanmax(arr, 0) - np.nanmin(arr, 0) if len(arr) else np.ones(len(num))
    ids = replicate_clusters(arr, widths, df[lab] if lab else None, tol=NEAR_DUPLICATE_TOL)
    return [f"nr{int(i)}" for i in ids]


class SurrogateArm(KalosEngineArm):
    """The production analysis with the configured surrogate swapped in."""

    shippable = True

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)
        self.name = str(self.config.get("name", "surr_selfcheck"))
        self._cls: type | None = None

    @contextlib.contextmanager
    def _patch(self) -> Iterator[None]:
        import kalos.core.evaluation as ev
        import kalos.core.surrogate as s_mod
        import kalos.portal.analysis as an

        if self._cls is None:
            self._cls = make_surrogate_class(self.config)
        saved = [(s_mod, "Surrogate", s_mod.Surrogate), (ev, "Surrogate", ev.Surrogate)]
        s_mod.Surrogate = self._cls  # type: ignore[misc]
        ev.Surrogate = self._cls  # type: ignore[misc]
        if self.config.get("hetero") == "rep":
            orig_agg = an.aggregate_replicates

            def agg(*a, **kw):  # type: ignore[no-untyped-def]
                out = orig_agg(*a, **kw)
                _STASH.agg = (out[1], out[2], out[3])
                return out

            saved.append((an, "aggregate_replicates", orig_agg))
            an.aggregate_replicates = agg  # type: ignore[assignment]
        if self.config.get("near_replicates"):
            saved.append((an, "row_hash_groups", an.row_hash_groups))
            an.row_hash_groups = _near_replicate_keys  # type: ignore[assignment]
        try:
            yield
        finally:
            for obj, attr, val in reversed(saved):
                setattr(obj, attr, val)
            _STASH.agg = None

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        return self._patch()


def make_arm(config: Mapping[str, Any] | None = None) -> SurrogateArm:
    return SurrogateArm(config)


__all__ = ["SurrogateArm", "make_arm", "make_surrogate_class"]
