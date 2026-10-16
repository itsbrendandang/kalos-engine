"""Mechanistic model bridge — sketch interface for kinetic/ODE models in the BO loop.

This module defines a minimal interface so that mechanistic models (Monod growth,
substrate consumption, product formation) can be dropped into the kalos.optimize
loop in place of (or alongside) a Gaussian-process Surrogate.

Key concepts:
  - ``MechanisticModel``: abstract base defining the contract every kinetic model
    must implement (predict, jacobian, residuals).
  - ``MechanisticWrapper``: bridges ``MechanisticModel`` into the BO proposal loop
    by evaluating the model at proposed points and feeding synthetic observations
    back to the optimizer.
  - ``MonodKineticModel``: concrete Monod growth-kinetics implementation with
    parameters ``mu_max``, ``K_s``, ``Y_x/s``.

This is a *design sketch* — importable and type-checkable, but not a full
implementation.  Real integration with BoTorch/BoTorch acquisition would require
a proper likelihood layer and MCMC/APT calibration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from kalos.core.optimize import propose  # avoid hard import cycle


# ---------------------------------------------------------------------------
# Core interface
# ---------------------------------------------------------------------------

@dataclass
class MechanisticModel(ABC):
    """Interface that kinetic/ODE models must implement to plug into the BO loop.

    Every method accepts a 2-D array ``X`` of shape ``(n, d)`` where ``d`` is
    the number of design variables (e.g. temperature, pH, dissolved O₂) and
    returns arrays of shape ``(n,)`` or ``(n, m)``.

    Subclasses should also expose a ``.parameters`` attribute (dict-like) so
    the optimizer can inspect and, if desired, sweep kinetic parameters.
    """

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target variable(s) for design points ``X``.

        Parameters
        ----------
        X : np.ndarray
            Design points, shape ``(n, d)``.

        Returns
        -------
        np.ndarray
            Predicted values, shape ``(n,)`` for a single target or
            ``(n, m)`` for multiple targets.  Values should be on the same
            scale as what the BO objective maximises.
        """

    @abstractmethod
    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute the Jacobian ``dy/dx`` at each design point.

        Returns
        -------
        np.ndarray
            Shape ``(n, d, m)`` where ``m`` is the number of outputs.  For
            a single-target model this is ``(n, d)``.

        Notes
        -----
        Numerical Jacobians are acceptable; analytical forms are preferred
        when available (e.g. from an ODE solver's sensitivity module).
        """

    @abstractmethod
    def residuals(
        self, X: np.ndarray, y_observed: np.ndarray
    ) -> np.ndarray:
        """Compute residuals ``(y_predicted - y_observed)`` at ``X``.

        Parameters
        ----------
        X : np.ndarray
            Design points, shape ``(n, d)``.
        y_observed : np.ndarray
            Observed target values, shape ``(n,)`` or ``(n, m)``.

        Returns
        -------
        np.ndarray
            Residuals, shape matching ``y_observed``.
        """

    # ------------------------------------------------------------------
    # Optional / convention
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> dict[str, float]:
        """Human-readable kinetic parameters (e.g. ``mu_max``, ``K_s``)."""
        return {}

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls} params={self.parameters}>"


# ---------------------------------------------------------------------------
# Wrapper: bridging mechanistic models into the BO loop
# ---------------------------------------------------------------------------

class MechanisticWrapper:
    """Bridge ``MechanisticModel`` into ``kalos.optimize.propose``.

    The wrapper evaluates the mechanistic model at each proposed point,
    adds a configurable noise term to simulate measurement uncertainty, and
    returns a synthetic observation that the BO acquisition function can
    evaluate against.

    Parameters
    ----------
    model : MechanisticModel
        A concrete mechanistic model implementing the interface above.
    noise_scale : float
        Fraction of the predicted signal range to add as Gaussian noise.
        Set to ``0.0`` for a noiseless evaluation.
    """

    def __init__(
        self,
        model: MechanisticModel,
        noise_scale: float = 0.0,
    ) -> None:
        self.model = model
        self.noise_scale = noise_scale

    def evaluate(self, X: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the model at ``X`` and return ``(y_true, y_obs)``.

        Parameters
        ----------
        X : np.ndarray
            Design points, shape ``(n, d)``.
        rng : np.random.Generator | None
            Random number generator for noise.  Uses ``np.random.default_rng()``
            if ``None``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(y_true, y_obs)`` — noiseless predictions and noisy observations,
            both shape ``(n,)``.
        """
        y_true = self.model.predict(np.atleast_2d(X))
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1)

        if self.noise_scale > 0.0 and rng is not None:
            sigma = self.noise_scale * float(np.ptp(y_true)) if np.ptp(y_true) > 0 else 1.0
            y_obs = y_true + rng.normal(0.0, sigma, size=y_true.shape)
        else:
            y_obs = y_true.copy()

        return y_true, y_obs

    def jacobian_at(self, X: np.ndarray) -> np.ndarray:
        """Convenience wrapper around ``model.jacobian(X)``."""
        return self.model.jacobian(np.atleast_2d(X))

    def residuals_at(self, X: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        """Convenience wrapper around ``model.residuals(X, y_observed)``."""
        return self.model.residuals(np.atleast_2d(X), y_observed)


# ---------------------------------------------------------------------------
# Concrete model: Monod growth kinetics
# ---------------------------------------------------------------------------

@dataclass
class MonodKineticModel(MechanisticModel):
    """Monod growth-kinetics model for bioprocess optimization.

    The model predicts the specific growth rate ``mu`` from temperature, pH,
    and dissolved-oxygen conditions using the Monod equation:

        mu = mu_max * S/(K_s + S) * f_T(T) * f_pH(pH) * f_DO(DO)

    where:
        * ``mu_max`` : maximum specific growth rate (1/h)
        * ``K_s``    : half-saturation constant (g/L substrate)
        * ``Y_x/s``  : biomass yield on substrate (g/g)

    The ``predict`` method returns a proxy *yield* metric proportional to
    ``mu * duration``, suitable for maximisation in the BO loop.
    """

    # ------------------------------------------------------------------
    # Kinetic parameters
    # ------------------------------------------------------------------
    mu_max: float = 0.4       # 1/h — maximum specific growth rate
    K_s: float = 2.0          # g/L — half-saturation constant
    Y_x_s: float = 0.5        # g/g — biomass yield on substrate

    # ------------------------------------------------------------------
    # Operating-range defaults (for the design space)
    # ------------------------------------------------------------------
    T_ref: float = 28.0       # °C — reference temperature
    pH_ref: float = 6.0       # reference pH
    DO_ref: float = 30.0      # % — reference dissolved oxygen

    duration: float = 5.0     # days — fixed fermentation duration (used in predict)

    # ------------------------------------------------------------------
    # Predicted substrate concentration (for demonstration; in reality
    # this would come from an ODE solver).
    # ------------------------------------------------------------------
    S0: float = 10.0          # g/L — initial substrate concentration

    # ------------------------------------------------------------------
    # MechanisticModel interface
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> dict[str, float]:
        return {
            "mu_max": self.mu_max,
            "K_s": self.K_s,
            "Y_x_s": self.Y_x_s,
            "T_ref": self.T_ref,
            "pH_ref": self.pH_ref,
            "DO_ref": self.DO_ref,
            "S0": self.S0,
            "duration": self.duration,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict yield proxy from design conditions.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n, 3)`` — columns are ``[temperature (°C), pH, DO (%)]``.

        Returns
        -------
        np.ndarray
            Predicted yield proxy, shape ``(n,)``.
        """
        X = np.atleast_2d(X)
        T, pH, DO = X[:, 0], X[:, 1], X[:, 2]

        # Monod substrate term (assume substrate is at 50% of S0 for simplicity)
        S = self.S0 * 0.5
        mu_monod = self.mu_max * S / (self.K_s + S)

        # Temperature term (Gaussian around T_ref)
        sigma_T = 5.0
        f_T = np.exp(-0.5 * ((T - self.T_ref) / sigma_T) ** 2)

        # pH term (Gaussian around pH_ref)
        sigma_pH = 1.0
        f_pH = np.exp(-0.5 * ((pH - self.pH_ref) / sigma_pH) ** 2)

        # DO term (Gaussian around DO_ref)
        sigma_DO = 10.0
        f_DO = np.exp(-0.5 * ((DO - self.DO_ref) / sigma_DO) ** 2)

        # Yield proxy: mu * duration * Y_x/s * f_T * f_pH * f_DO
        yield_proxy = mu_monod * self.duration * self.Y_x_s * f_T * f_pH * f_DO

        return yield_proxy

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """Analytical Jacobian ``dy/d[T, pH, DO]``.

        Returns
        -------
        np.ndarray
            Shape ``(n, 3)`` — partial derivative of the yield proxy with
            respect to each input variable.
        """
        X = np.atleast_2d(X)
        T, pH, DO = X[:, 0], X[:, 1], X[:, 2]

        # Pre-compute shared terms
        S = self.S0 * 0.5
        mu_base = self.mu_max * S / (self.K_s + S)
        prefactor = mu_base * self.duration * self.Y_x_s

        sigma_T, sigma_pH, sigma_DO = 5.0, 1.0, 10.0

        f_T = np.exp(-0.5 * ((T - self.T_ref) / sigma_T) ** 2)
        f_pH = np.exp(-0.5 * ((pH - self.pH_ref) / sigma_pH) ** 2)
        f_DO = np.exp(-0.5 * ((DO - self.DO_ref) / sigma_DO) ** 2)

        # dy/dT = prefactor * f_pH * f_DO * df_T/dT
        dT = prefactor * f_pH * f_DO * f_T * (-(T - self.T_ref) / sigma_T**2)
        # dy/dpH = prefactor * f_T * f_DO * df_pH/dpH
        dpH = prefactor * f_T * f_DO * f_pH * (-(pH - self.pH_ref) / sigma_pH**2)
        # dy/dDO = prefactor * f_T * f_pH * df_DO/dDO
        dDO = prefactor * f_T * f_pH * f_DO * (-(DO - self.DO_ref) / sigma_DO**2)

        return np.column_stack([dT, dpH, dDO])

    def residuals(
        self, X: np.ndarray, y_observed: np.ndarray
    ) -> np.ndarray:
        """Residuals between model prediction and observed data."""
        X = np.atleast_2d(X)
        y_observed = np.atleast_1d(y_observed)
        y_predicted = self.predict(X)
        return y_predicted - y_observed


# ---------------------------------------------------------------------------
# Usage example (non-executable at import; shown for documentation)
# ---------------------------------------------------------------------------
#
#   from kalos.core.surrogate import Surrogate
#   from kalos.core.optimize import propose
#   from kalos.core.mechanistic_bridge import MonodKineticModel, MechanisticWrapper
#
#   # 1. Define the mechanistic model
#   model = MonodKineticModel(mu_max=0.4, K_s=2.0, Y_x_s=0.5)
#
#   # 2. Wrap it for the BO loop
#   wrapper = MechanisticWrapper(model, noise_scale=0.05)
#
#   # 3. Initial experimental design (e.g. LHS over [T, pH, DO])
#   n_init = 5
#   bounds = np.array([[20, 5.0, 10.0], [35, 7.5, 50.0]])
#   rng = np.random.default_rng(42)
#   X_init = rng.uniform(bounds[0], bounds[1], size=(n_init, 3))
#   _, y_init = wrapper.evaluate(X_init, rng=rng)
#
#   # 4. Fit a surrogate on the synthetic observations
#   surrogate = Surrogate().fit(X_init, y_init, bounds=bounds)
#
#   # 5. Propose next point (standard BO loop, but driven by the mechanistic model)
#   X_next = propose(surrogate, bounds, q=1)
#
#   # 6. Evaluate the mechanistic model at the proposed point
#   y_true, y_obs = wrapper.evaluate(X_next, rng=rng)
#
#   # 7. Accumulate and re-fit — the cycle repeats
#   X_all = np.vstack([X_init, X_next])
#   y_all = np.concatenate([y_init, y_obs])
#
