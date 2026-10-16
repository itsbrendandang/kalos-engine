"""Hybrid mechanistic + ML bridge (interface only; to be fleshed out).

A mechanistic model (typically an ODE system, e.g. Monod growth + product
formation) predicts the response from process settings using domain knowledge.
`HybridSurrogate` keeps that prediction as the mean function and lets a GP learn
only the *residual* `y - mechanistic(X)`, so the GP spends its data on what the
physics misses. With a poor mechanistic model the residual GP simply absorbs the
error, so the hybrid degrades to a plain GP rather than below it.

Scope today: fit + predict (mean/std). Handing a hybrid straight to
`kalos.core.optimize.propose` is NOT supported yet - `propose` needs a BoTorch
model, and the residual GP alone ignores the mechanistic mean. Next step is to
wrap the sum as a BoTorch `Model`. Torch stays lazily imported (`Surrogate`
is only touched inside `fit`), keeping `kalos.core.mechanistic` cheap to import.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np


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
            except (ValueError, FloatingPointError, ArithmeticError):
                pass  # leave NaN: this row has no mechanistic prediction
        return out


class HybridSurrogate:
    """prediction(X) = mechanistic(X) + GP_residual(X)."""

    def __init__(self, mechanistic: MechanisticModel) -> None:
        if not isinstance(mechanistic, MechanisticModel):
            raise TypeError("mechanistic must provide evaluate(X) -> ndarray")
        self.mechanistic = mechanistic
        self.residual = None  # kalos.core.surrogate.Surrogate once fitted

    def _mech(self, X) -> np.ndarray:
        m = np.asarray(self.mechanistic.evaluate(np.atleast_2d(np.asarray(X, float))), float).reshape(-1)
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
