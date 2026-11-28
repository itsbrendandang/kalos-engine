"""Multi-objective Bayesian optimization (e.g. titer AND purity together).

A bioprocess rarely optimizes one number — you want high titer without losing
purity. This fits one GP per objective (a `ModelListGP`) and proposes the next
batch by q-Log Noisy Expected Hypervolume Improvement (qLogNEHVI), the current
BoTorch state of the art for noisy multi-objective BO. Proposals expand the
Pareto front: the set of recipes where you can't improve one objective without
giving up another.

Objectives default to MAXIMIZED (`maximize=None`), or each objective can opt
into minimization via `maximize=[True, False, ...]`. Internally the surrogate
flips minimized objectives so BoTorch's maximization machinery applies
uniformly, while `pareto()`, `posterior()`, and `default_ref_point()` always
return values in the original units and original sign. Runs on CPU float64
(same as the single-objective surrogate).
"""
from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Sequence

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_mixed,
    optimize_acqf_mixed_alternating,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood

from .optimize import (
    MAX_MIXED_COMBOS,
    _resolve_cardinalities,
    _sanitize_pending,
    _validate_linear_constraints,
)
from .surrogate import (
    DEVICE,
    DTYPE,
    FitError,
    _build_mixed_gp,
    _fit_mll_with_retry,
    _randomize_hyperparameters,
    condition_report,
    hyperparameter_report,
    kernel_condition_number,
    sanitize_bounds,
)
from .torch_ext import clear_stale_extension_lock


def _sum_mll_value(mll: SumMarginalLogLikelihood) -> float:
    model = mll.model
    model.train()
    mll.train()
    with torch.no_grad():
        outs = model(*model.train_inputs)
        val = mll(outs, model.train_targets)
    model.eval()
    mll.eval()
    return float(val.sum())


def _multi_start_fit_list(
    mll: SumMarginalLogLikelihood,
    *,
    restarts: int,
    seed: int,
) -> tuple[dict, ModelListGP]:
    """Fit a `ModelListGP` from its default start plus `restarts - 1` seeded
    random starts, keeping the highest `SumMarginalLogLikelihood`."""
    model = mll.model
    pristine = copy.deepcopy(model)
    t0 = time.perf_counter()
    best = dict(_fit_mll_with_retry(mll))
    best_val = _sum_mll_value(mll)
    best_model = model
    values: list[float | None] = [round(best_val, 6)]
    chosen = 0
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        for k in range(1, restarts):
            cand = copy.deepcopy(pristine)
            cand_mll = SumMarginalLogLikelihood(cand.likelihood, cand)
            gen = torch.Generator().manual_seed(seed + k)
            for sub in cand.models:
                _randomize_hyperparameters(sub, gen)
            try:
                rec = dict(_fit_mll_with_retry(cand_mll))
            except FitError:
                values.append(None)
                continue
            val = _sum_mll_value(cand_mll)
            values.append(round(val, 6))
            if val > best_val + 1e-6:
                best, best_val, chosen, best_model = rec, val, k, cand
    np.random.set_state(np_state)
    best_model.eval()
    return {
        **best,
        "restarts": restarts,
        "restart_seed": seed,
        "restart_mll": values,
        "restart_chosen": chosen,
        "fit_seconds": round(time.perf_counter() - t0, 3),
    }, best_model


def _normalize_noise_matrix(noise, n: int, m: int) -> np.ndarray | None:
    if noise is None:
        return None
    arr = np.asarray(noise, float)
    if arr.ndim == 0:
        mat = np.full((n, m), float(arr), dtype=float)
    elif arr.ndim == 1:
        if arr.shape[0] != n:
            raise ValueError(f"1-D noise must have length {n}, got {arr.shape[0]}")
        mat = np.broadcast_to(arr[:, None], (n, m)).astype(float)
    elif arr.ndim == 2:
        if arr.shape == (n, 1):
            mat = np.broadcast_to(arr, (n, m)).astype(float)
        elif arr.shape == (n, m):
            mat = arr.astype(float)
        else:
            raise ValueError(f"2-D noise must have shape ({n}, {m}) or ({n}, 1), got {arr.shape}")
    else:
        raise ValueError("noise must be a scalar, 1-D (n,), or 2-D (n, m) array")
    if not np.isfinite(mat).all() or (mat < 0).any():
        raise ValueError("noise must be finite and non-negative")
    return mat


class MultiObjectiveSurrogate:
    """One GP per objective over a shared design space."""

    def __init__(self) -> None:
        self.model: ModelListGP | None = None
        self._X: torch.Tensor | None = None
        self._Y: torch.Tensor | None = None
        self._Y_signed: torch.Tensor | None = None
        self._signs: np.ndarray | None = None
        self._cat_dims: list[int] | None = None
        self.fit_diagnostics: dict = {}

    def fit(
        self,
        X,
        Y,
        bounds,
        *,
        noise=None,
        cat_dims=None,
        maximize: Sequence[bool] | None = None,
        check_conditioning: bool = False,
        restarts: int = 1,
        restart_seed: int = 0,
    ) -> MultiObjectiveSurrogate:
        """Fit one GP per objective. `bounds` (2 x d) is required so input
        normalization is always tied to the fixed design box, never the
        training-data envelope (the same footgun the single-objective fit closes)."""
        Xa = np.asarray(X, float)
        Ya = np.asarray(Y, float)
        if Xa.ndim != 2 or Ya.ndim != 2 or Xa.shape[0] == 0 or Ya.shape[1] == 0 or Xa.shape[0] != Ya.shape[0]:
            raise ValueError("X (n x d) and Y (n x m) must be non-empty and share n rows")
        if not np.isfinite(Xa).all() or not np.isfinite(Ya).all():
            raise ValueError("X and Y must be finite (no NaN/inf)")

        n, d = Xa.shape
        m = Ya.shape[1]
        if maximize is None:
            signs = np.ones(m, dtype=float)
        else:
            flags = list(maximize)
            if len(flags) != m:
                raise ValueError(f"maximize must have length {m}, got {len(flags)}")
            signs = np.array([1.0 if bool(v) else -1.0 for v in flags], dtype=float)

        noise_mat = _normalize_noise_matrix(noise, n, m)
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        Yt = torch.as_tensor(Ya, dtype=DTYPE, device=DEVICE)
        signs_t = torch.as_tensor(signs, dtype=DTYPE, device=DEVICE)
        Yt_signed = Yt * signs_t

        # sanitize_bounds widens zero-width (constant-feature) dims and repairs
        # non-finite bounds so Normalize never divides by a zero-width range and
        # NaN-poisons the fit. Mirrors the single-objective Surrogate.fit guard.
        lower, upper = sanitize_bounds(bounds)
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
        cats = [int(i) for i in cat_dims] if cat_dims else []

        # A fresh Normalize per model: Normalize is a stateful torch.nn.Module, and
        # sharing one instance across every GP in the ModelListGP means a future
        # learn_bounds=True (or any other transform mutation) on one objective
        # would silently leak into every other objective's model through the
        # shared object.
        models = []
        for i in range(m):
            yi = Yt_signed[:, i : i + 1]
            yvar_i = (
                None
                if noise_mat is None
                else torch.as_tensor(noise_mat[:, i : i + 1], dtype=DTYPE, device=DEVICE)
            )
            if cats:
                models.append(_build_mixed_gp(Xt, yi, box, cats, d, yvar_i))
            elif yvar_i is None:
                models.append(
                    SingleTaskGP(
                        Xt,
                        yi,
                        input_transform=Normalize(d=d, bounds=box),
                        outcome_transform=Standardize(m=1),
                    )
                )
            else:
                models.append(
                    SingleTaskGP(
                        Xt,
                        yi,
                        train_Yvar=yvar_i,
                        input_transform=Normalize(d=d, bounds=box),
                        outcome_transform=Standardize(m=1),
                    )
                )

        self.model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        if restarts > 1:
            diag, self.model = _multi_start_fit_list(mll, restarts=int(restarts), seed=int(restart_seed))
        elif restarts == 1 and restart_seed:
            diag = {
                **_fit_mll_with_retry(mll),
                "restarts": 1,
                "restart_seed": int(restart_seed),
                "restart_mll": None,
                "restart_chosen": 0,
            }
        else:
            diag = dict(_fit_mll_with_retry(mll))

        per_obj_hp = [hyperparameter_report(sub, fixed_noise=noise_mat is not None) for sub in self.model.models]
        diag["per_objective_diagnostics"] = per_obj_hp
        diag["boundary_flags"] = [
            f"obj[{i}]: {flag}" for i, hp in enumerate(per_obj_hp) for flag in hp["boundary_flags"]
        ]
        if check_conditioning:
            conds = [kernel_condition_number(sub) for sub in self.model.models]
            diag.update(condition_report(max(conds)))
            diag["per_objective_condition_numbers"] = conds

        self.fit_diagnostics = diag
        self._X = Xt
        self._Y = Yt
        self._Y_signed = Yt_signed
        self._signs = signs
        self._cat_dims = cats or None
        return self

    def posterior(self, X, *, observation_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return `(means, stds)` of shape `(k, m)` in original units & original sign."""
        assert self.model is not None, "fit() first"
        Xa = np.atleast_2d(np.asarray(X, float))
        if not np.isfinite(Xa).all():
            raise ValueError("X must be finite (no NaN/inf)")
        Xt = torch.as_tensor(Xa, dtype=DTYPE, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt, observation_noise=observation_noise)
            mean_signed = post.mean.cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().cpu().numpy()
        signs = self._signs if self._signs is not None else np.ones(mean_signed.shape[-1], dtype=float)
        return mean_signed * signs, std

    def pareto(self):
        """Return (X, Y) of the non-dominated observed points (original units & sign)."""
        assert self._Y is not None and self._X is not None
        y_signed = self._Y_signed if self._Y_signed is not None else self._Y
        mask = is_non_dominated(y_signed)
        return self._X[mask].cpu().numpy(), self._Y[mask].cpu().numpy()

    def default_ref_point(self, margin: float = 0.1) -> np.ndarray:
        """A reference point dominated by every observation in original units & sign."""
        assert self._Y is not None
        y_signed = self._Y_signed if self._Y_signed is not None else self._Y
        span = (y_signed.max(dim=0).values - y_signed.min(dim=0).values).clamp_min(1e-9)
        rp_signed = (y_signed.min(dim=0).values - margin * span).cpu().numpy()
        signs = self._signs if self._signs is not None else np.ones_like(rp_signed)
        return rp_signed * signs


def propose_multiobjective(
    surrogate: MultiObjectiveSurrogate,
    bounds,
    ref_point=None,
    q: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 128,
    mc_samples: int = 128,
    *,
    seed: int | None = None,
    pending=None,
    inequality_constraints=None,
    fixed_features: dict[int, float] | None = None,
    cat_dims: list[int] | None = None,
    cat_cardinalities: list[int] | None = None,
) -> np.ndarray:
    """Return q proposed points (q x d) that best expand the Pareto front.

    `seed` governs the stochastic work: the qLogNEHVI sampler's Sobol draws
    (`SobolQMCNormalSampler(seed=seed)`) and the multi-start acquisition
    optimization's raw-sample initial conditions inside `optimize_acqf`. `None`
    (the default) seeds nothing here, byte-identical to the historical
    behavior - reproducibility remains the caller's responsibility via seeding
    the global torch RNG beforehand, as the portal and bench already do. When
    given, the seed is applied inside `torch.random.fork_rng()`, which snapshots
    the caller's RNG state and restores it on exit, so a seeded call is
    reproducible without clobbering the caller's global RNG state as a side
    effect.

    Every returned coordinate is clamped into `[lower, upper]` per feature so a
    proposal can never fall outside the observed design box. This mirrors the
    single-objective `core.optimize.propose` guard against the historical
    out-of-range blow-up (a constant `Culture_Volume` proposing ~= 33,000,000)
    for a degenerate or near-constant feature. See `sanitize_bounds`.
    """
    assert surrogate.model is not None and surrogate._X is not None, "fit the surrogate first"
    lower, upper = sanitize_bounds(bounds)  # widen zero-width dims so optimize_acqf sees a valid box
    cats = list(cat_dims) if cat_dims is not None else list(surrogate._cat_dims or [])
    fixed = {int(k): float(v) for k, v in (fixed_features or {}).items()}
    for j, v in fixed.items():
        if not 0 <= j < len(lower):
            raise ValueError("fixed_features names a dimension outside the design")
        if not np.isfinite(v):
            raise ValueError("fixed_features values must be finite")
        if cats and j in set(cats):
            v = float(np.rint(v))
            fixed[j] = v
        lower[j] = upper[j] = v

    cards = _resolve_cardinalities(cats, cat_cardinalities, lower, upper) if cats else []
    free_cats = [j for j in cats if j not in fixed]
    free_cards = [int(k) for j, k in zip(cats, cards) if j not in fixed]

    lin = _validate_linear_constraints(inequality_constraints, lower, upper)
    ineq = [
        (torch.as_tensor(idx, dtype=torch.long), torch.as_tensor(coef, dtype=DTYPE), rhs)
        for idx, coef, rhs in lin
    ] or None
    X_pending = _sanitize_pending(pending, lower, upper, cats or None)

    b = torch.stack(
        [
            torch.as_tensor(lower, dtype=DTYPE, device=DEVICE),
            torch.as_tensor(upper, dtype=DTYPE, device=DEVICE),
        ]
    )
    rp = surrogate.default_ref_point() if ref_point is None else np.asarray(ref_point, float).reshape(-1)
    signs = surrogate._signs if surrogate._signs is not None else np.ones(len(rp), dtype=float)
    if rp.shape[0] != signs.shape[0]:
        raise ValueError(f"ref_point must have length {signs.shape[0]}, got {rp.shape[0]}")
    rp_t = torch.as_tensor(rp * signs, dtype=DTYPE, device=DEVICE)

    def _run_acqf_optimization() -> torch.Tensor:
        clear_stale_extension_lock()  # an abandoned JIT lock would block the fused-kernel load
        acq_kwargs: dict = {
            "model": surrogate.model,
            "ref_point": rp_t,
            "X_baseline": surrogate._X,
            "prune_baseline": True,
            "sampler": SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=seed),
        }
        if X_pending is not None:
            acq_kwargs["X_pending"] = X_pending
        acq = qLogNoisyExpectedHypervolumeImprovement(**acq_kwargs)

        if free_cats:
            n_combos = 1
            for k in free_cards:
                n_combos *= max(k, 1)
            if n_combos <= MAX_MIXED_COMBOS:
                fixed_features_list = [
                    {**fixed, **{dim: float(code) for dim, code in zip(free_cats, combo)}}
                    for combo in itertools.product(*(range(max(k, 1)) for k in free_cards))
                ]
                opt_kwargs: dict = {}
                if ineq:
                    opt_kwargs["inequality_constraints"] = ineq
                candidates, _ = optimize_acqf_mixed(
                    acq_function=acq,
                    bounds=b,
                    q=q,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    fixed_features_list=fixed_features_list,
                    **opt_kwargs,
                )
                return candidates
            cat_map = {
                dim: [float(code) for code in range(max(k, 1))]
                for dim, k in zip(free_cats, free_cards)
            }
            opt_kwargs = {}
            if fixed:
                opt_kwargs["fixed_features"] = fixed
            if ineq:
                opt_kwargs["inequality_constraints"] = ineq
            candidates, _ = optimize_acqf_mixed_alternating(
                acq_function=acq,
                bounds=b,
                cat_dims=cat_map,
                q=q,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                **opt_kwargs,
            )
            return candidates

        opt_kwargs = {}
        if ineq:
            opt_kwargs["inequality_constraints"] = ineq
        if fixed:
            opt_kwargs["fixed_features"] = fixed
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=b,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            **opt_kwargs,
        )
        return candidates

    if seed is None:
        candidates = _run_acqf_optimization()
    else:
        # devices=[] restricts the fork to CPU RNG state (DEVICE is always CPU
        # here), avoiding an unnecessary probe of CUDA RNG state on machines
        # without a GPU.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            candidates = _run_acqf_optimization()
    out = np.clip(candidates.detach().cpu().numpy(), lower, upper)
    for j in cats:
        out[:, j] = np.rint(out[:, j])
    return out


__all__ = ["MultiObjectiveSurrogate", "propose_multiobjective"]
