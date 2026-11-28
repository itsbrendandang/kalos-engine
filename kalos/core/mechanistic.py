"""Hybrid mechanistic + ML bridge.

A mechanistic model (typically an ODE system, e.g. Monod growth + product
formation) predicts the response from process settings using domain knowledge.
`HybridSurrogate` keeps that prediction as the mean function and lets a GP learn
only the *residual* `y - mechanistic(X)`, so the GP spends its data on what the
physics misses. A good mechanistic model helps; a poor one can leave the hybrid
no better, or WORSE, than a plain GP, because the GP must first undo the wrong
mean (in the simulated benchmark of examples/mechanistic_benchmark.py a badly wrong
ODE was no better than the plain GP in most seeds and worse in some). Judge a mechanistic model on held-out data before use.

OPT-IN, OFF BY DEFAULT. Nothing in the portal or `propose` uses this module; a
caller must construct a `HybridSurrogate` explicitly. Related work only: Richelle et
al. (CSBJ 2026, doi 10.34133/csbj.0078; cited from the project's literature notes,
not independently checked) is a single n=23 study of a hybrid ODE+ML CHO digital
twin. It concerns hybrid mechanistic+ML models in general and does not test a GP
prior mean. A simulated check (examples/mechanistic_benchmark.py; Monod fed-batch
truth with Haldane inhibition, 10 seeds, one fixed held-out split per seed) is saved in
examples/results/mechanistic_benchmark_10seeds.json: a correct or mildly wrong ODE
reached 95% of the best reachable response sooner than the plain GP in 9 of 10 seeds
(9-0, one tie, at both the oracle and mild levels; a two-sided sign test on the 9
non-tied seeds gives p = 0.004, about 0.016 after Bonferroni over the four levels;
"best reachable" excludes the held-out split no strategy can evaluate); a badly wrong or "adversarial" ODE showed no detectable
difference (counts of better/worse/tied are in the JSON). This is a noise-free,
2-D simulation whose truth shares the ODE's structure, so it says little about real
processes, and it is not a validation of the method. Read the exact counts,
reach rates and spreads from that file (regenerate with the script); judge your own
ODE on held-out data before turning it on.

Example (Monod fed-batch as the GP prior mean):

    from kalos.bench.mechanistic_benchmark import LEVELS
    hybrid = HybridSurrogate(LEVELS["mild"]).fit(X, y, bounds=bounds)
    mean, std = hybrid.predict(X_new)

Torch stays lazily imported (`Surrogate` is only touched inside `fit`, and
BoTorch `Model` wrapping is constructed lazily inside `to_botorch_model`),
keeping `kalos.core.mechanistic` 100% torch-free at import time.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from kalos.core.surrogate import Surrogate


@runtime_checkable
class MechanisticModel(Protocol):
    """Anything that maps a design matrix (n x d) to predicted responses (n,)."""

    def evaluate(self, X: np.ndarray) -> np.ndarray: ...


class ODEModel:
    """Wrap an ODE system as a `MechanisticModel`.

    rhs(t, state, params) -> d state/dt     (scipy `solve_ivp` convention)
    x_to_params(x) -> params                (one design row -> ODE parameters)
    x_to_y0(x) -> initial state             (one design row -> state at t=0)
    readout(t, states) -> float             (states: n_state x n_t) the response

    A row whose integration fails or yields a non-finite readout evaluates to
    NaN (never raises), so one bad region of the design space cannot take down a
    batch; `HybridSurrogate` treats NaN as "no mechanistic opinion" (0 offset).
    """

    def __init__(
        self,
        rhs: Callable,
        x_to_params: Callable[[np.ndarray], object],
        x_to_y0: Callable[[np.ndarray], Sequence[float]],
        readout: Callable[[np.ndarray, np.ndarray], float],
        t_end: float,
    ) -> None:
        if not t_end > 0:
            raise ValueError("t_end must be positive")
        self.rhs, self.x_to_params, self.x_to_y0 = rhs, x_to_params, x_to_y0
        self.readout, self.t_end = readout, float(t_end)

    def evaluate(self, X) -> np.ndarray:
        from scipy.integrate import solve_ivp

        def guarded(t, state, params):
            # A non-finite derivative can make LSODA spin forever instead of
            # failing; raise so the row is abandoned (-> NaN) right away.
            d = np.asarray(self.rhs(t, state, params), float)
            if not np.isfinite(d).all():
                raise FloatingPointError("non-finite ODE derivative")
            return d

        Xa = np.atleast_2d(np.asarray(X, float))
        out = np.full(len(Xa), np.nan)
        for i, x in enumerate(Xa):
            try:
                sol = solve_ivp(
                    guarded, (0.0, self.t_end), np.asarray(self.x_to_y0(x), float),
                    args=(self.x_to_params(x),), t_eval=np.linspace(0.0, self.t_end, 50),
                    method="LSODA",
                )
                if sol.success:
                    v = float(self.readout(sol.t, sol.y))
                    out[i] = v if np.isfinite(v) else np.nan
            except Exception:  # noqa: BLE001, S110 - user callbacks (rhs, mapping, readout) can raise anything
                # Leave NaN: this row has no mechanistic prediction (the contract is: never raises).
                pass
        return out


class _CallableModel:
    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(X), float)


class _NoMechanism:
    """Zero prior mean: the hybrid then behaves exactly like a plain GP."""

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(np.atleast_2d(X)))


def _build_hybrid_botorch_model(residual_model: Any, mech_fn: Callable[[np.ndarray], np.ndarray]):
    """Construct a BoTorch `Model` wrapping `residual_model` + detached `mech_fn` mean offset.

    Imported lazily so `kalos.core.mechanistic` remains torch-free at import time.
    """
    import torch
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal

    class _HybridBoTorchModel(Model):
        """BoTorch `Model` adding the detached mechanistic mean to a residual GP posterior."""

        def __init__(self, res_model: Any, mech_eval: Callable[[np.ndarray], np.ndarray]) -> None:
            super().__init__()
            self._residual_model = res_model
            self._mech_eval = mech_eval

        @property
        def num_outputs(self) -> int:
            return int(self._residual_model.num_outputs)

        @property
        def batch_shape(self) -> torch.Size:
            return self._residual_model.batch_shape

        @property
        def train_inputs(self):
            return self._residual_model.train_inputs

        @property
        def train_targets(self):
            return self._residual_model.train_targets

        def subset_output(self, idcs: list[int]) -> Model:
            if idcs == [0]:
                return self
            raise NotImplementedError("HybridBoTorchModel is single-output")

        def posterior(
            self,
            X: torch.Tensor,
            output_indices: list[int] | None = None,
            observation_noise: bool | torch.Tensor = False,
            posterior_transform: Any = None,
            **kwargs: Any,
        ) -> Any:
            if output_indices is not None and output_indices != [0]:
                raise NotImplementedError("HybridBoTorchModel is single-output")
            res_post = self._residual_model.posterior(
                X,
                observation_noise=observation_noise,
                posterior_transform=None,
                **kwargs,
            )
            with torch.no_grad():
                X_np = X.detach().cpu().numpy()
                flat = X_np.reshape(-1, X_np.shape[-1])
                mech_flat = self._mech_eval(flat)
                mech_t = torch.as_tensor(mech_flat, dtype=X.dtype, device=X.device).reshape(X.shape[:-1])

            mvn = res_post.distribution
            # Adding detached mech_t preserves autograd gradients through mvn.mean
            # and mvn.lazy_covariance_matrix so optimize_acqf climbs the residual GP.
            shifted_mvn = MultivariateNormal(mvn.mean + mech_t, mvn.lazy_covariance_matrix)
            posterior: Any = GPyTorchPosterior(shifted_mvn)
            if posterior_transform is not None:
                return posterior_transform(posterior=posterior, X=X)
            return posterior

    return _HybridBoTorchModel(residual_model, mech_fn)


class HybridSurrogate:
    """prediction(X) = mechanistic(X) + GP_residual(X).

    `mechanistic` may be a `MechanisticModel`, a plain callable X -> y (used as
    the GP prior mean), or None (falls back to the current plain-GP behavior).
    """

    def __init__(self, mechanistic: MechanisticModel | Callable | None = None) -> None:
        if mechanistic is None:
            mechanistic = _NoMechanism()
        elif not isinstance(mechanistic, MechanisticModel):
            if not callable(mechanistic):
                raise TypeError("mechanistic must be a MechanisticModel, a callable X -> y, or None")
            mechanistic = _CallableModel(mechanistic)
        self.mechanistic = mechanistic
        self.residual: Surrogate | None = None  # fitted GP on the residual
        self._X: Any = None
        self._y: np.ndarray | None = None
        self._cat_dims: list[int] | None = None
        self._bounds: np.ndarray | None = None
        self._botorch_model: Any = None

    def _mech(self, X) -> np.ndarray:
        Xa = np.atleast_2d(np.asarray(X, float))
        m = np.asarray(self.mechanistic.evaluate(Xa), float)
        # No silent broadcasting: a scalar or wrong-length output would shift every
        # row's prior mean by a wrong amount without any error.
        if m.ndim != 1 or m.shape[0] != Xa.shape[0]:
            raise ValueError(
                f"mechanistic model must return one value per row: expected shape ({Xa.shape[0]},), got {m.shape}"
            )
        return np.where(np.isfinite(m), m, 0.0)

    def fit(self, X, y, bounds, **kwargs) -> HybridSurrogate:
        from kalos.core.surrogate import Surrogate

        Xa = np.atleast_2d(np.asarray(X, float))
        ya = np.asarray(y, float).reshape(-1)
        self.residual = Surrogate().fit(Xa, ya - self._mech(Xa), bounds=bounds, **kwargs)
        self._X = self.residual._X
        self._y = ya
        self._cat_dims = self.residual._cat_dims
        self._bounds = np.asarray(bounds, float).copy()
        self._botorch_model = None
        return self

    def posterior(self, X, *, observation_noise: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) in the target's original units."""
        if self.residual is None:
            raise RuntimeError("fit() first")
        mu, sd = self.residual.posterior(X, observation_noise=observation_noise)
        return self._mech(X) + np.asarray(mu, float).reshape(-1), np.asarray(sd, float).reshape(-1)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) in the target's original units."""
        return self.posterior(X)

    def to_botorch_model(self):
        """Return a BoTorch `Model` combining the mechanistic mean with the residual GP."""
        if self.residual is None or self.residual.model is None:
            raise RuntimeError("fit() first")
        if self._botorch_model is None:
            self._botorch_model = _build_hybrid_botorch_model(self.residual.model, self._mech)
        return self._botorch_model

    @property
    def model(self):
        """Lazily constructed BoTorch `Model` (None before `fit()`)."""
        if self.residual is None or self.residual.model is None:
            return None
        return self.to_botorch_model()

    @property
    def best_f(self) -> float:
        if self._y is None:
            raise RuntimeError("fit() first")
        return float(self._y.max())

    @property
    def fit_diagnostics(self) -> dict:
        if self.residual is None:
            return {}
        return self.residual.fit_diagnostics


def monod_example(t_end: float = 24.0) -> ODEModel:
    """Tiny batch-culture example. Design row x = [mu_max, Ks, S0].

    state = [biomass X, substrate S]; dX/dt = mu X, dS/dt = -mu X / Y,
    mu = mu_max S / (Ks + S), Y fixed at 0.5. Readout: final biomass.
    """

    def rhs(t, state, p):
        mu_max, ks, _ = p
        biomass, s = state
        s = max(s, 0.0)
        mu = mu_max * s / (ks + s)
        return [mu * biomass, -mu * biomass / 0.5]

    return ODEModel(
        rhs=rhs,
        x_to_params=lambda x: tuple(x),
        x_to_y0=lambda x: [0.1, float(x[2])],
        readout=lambda t, y: y[0, -1],
        t_end=t_end,
    )
