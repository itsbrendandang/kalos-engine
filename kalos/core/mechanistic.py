"""Hybrid mechanistic + ML bridge (interface only; to be fleshed out).

A mechanistic model (typically an ODE system, e.g. Monod growth + product
formation) predicts the response from process settings using domain knowledge.
`HybridSurrogate` keeps that prediction as the mean function and lets a GP learn
only the *residual* `y - mechanistic(X)`, so the GP spends its data on what the
physics misses. A good mechanistic model helps; a poor one can leave the hybrid
no better, or WORSE, than a plain GP (a deliberately wrong ODE did worse in the
simulated benchmark of examples/mechanistic_benchmark.py), because the GP must
first undo the wrong mean. Judge a mechanistic model on held-out data before use.

Scope today: fit + predict (mean/std). Handing a hybrid straight to
`kalos.core.optimize.propose` is NOT supported yet - `propose` needs a BoTorch
model, and the residual GP alone ignores the mechanistic mean. Next step is to
wrap the sum as a BoTorch `Model`. Torch stays lazily imported (`Surrogate`
is only touched inside `fit`), keeping `kalos.core.mechanistic` cheap to import.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, Sequence, runtime_checkable

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
            except Exception:  # noqa: BLE001 - user callbacks (rhs, mapping, readout) can raise anything
                pass  # leave NaN: this row has no mechanistic prediction (contract: never raises)
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


class HybridSurrogate:
    """prediction(X) = mechanistic(X) + GP_residual(X).

    `mechanistic` may be a `MechanisticModel`, a plain callable X -> y (used as
    the GP prior mean), or None (falls back to the current plain-GP behavior).
    """

    def __init__(self, mechanistic: "MechanisticModel | Callable | None" = None) -> None:
        if mechanistic is None:
            mechanistic = _NoMechanism()
        elif not isinstance(mechanistic, MechanisticModel):
            if not callable(mechanistic):
                raise TypeError("mechanistic must be a MechanisticModel, a callable X -> y, or None")
            mechanistic = _CallableModel(mechanistic)
        self.mechanistic = mechanistic
        self.residual: Surrogate | None = None  # fitted GP on the residual

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

    def fit(self, X, y, bounds, **kwargs) -> "HybridSurrogate":
        from kalos.core.surrogate import Surrogate

        Xa = np.atleast_2d(np.asarray(X, float))
        ya = np.asarray(y, float).reshape(-1)
        self.residual = Surrogate().fit(Xa, ya - self._mech(Xa), bounds=bounds, **kwargs)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) in the target's original units."""
        if self.residual is None:
            raise RuntimeError("fit() first")
        mu, sd = self.residual.posterior(X)
        return self._mech(X) + np.asarray(mu, float).reshape(-1), np.asarray(sd, float).reshape(-1)


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
