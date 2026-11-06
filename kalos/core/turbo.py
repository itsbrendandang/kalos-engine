"""TuRBO-1 trust region for high-dimensional proposals (opt-in, d >= 8).

Following the BoTorch TuRBO tutorial (Eriksson et al., NeurIPS 2019, "Scalable
Global Optimization via Local Bayesian Optimization"): keep a hyper-rectangle
centred on the best observation, side lengths shaped by the GP's ARD
lengthscales, that DOUBLES after `success_tolerance` consecutive improvements
and HALVES after `failure_tolerance` consecutive failures; when it shrinks below
`length_min` the region is spent and `restart_triggered` is set. The
acquisition (kalos's qLogNEI, unchanged) is optimized inside the region only.

Why only for d >= 8: in low dimension a global GP + qLogNEI already covers the
box; in higher dimension it over-explores the boundary and TuRBO's local focus
is what helps (the tutorial's regime is d = 20+). Below `TURBO_MIN_DIM`,
`propose_batch(..., trust_region=...)` ignores the region and says so in the
log, so a caller can pass a state unconditionally.

The idea of a trust region for kalos came from an earlier sklearn-GP prototype;
this is written from the BoTorch
tutorial, not from that code. Continuous designs only (no categoricals).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TURBO_MIN_DIM = 8


@dataclass
class TurboState:
    """TuRBO-1 state. `length` is in the unit cube (1.0 = the full box side)."""

    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(default=0)
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -math.inf
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        if self.failure_tolerance <= 0:
            self.failure_tolerance = math.ceil(max(4.0 / self.batch_size, float(self.dim) / self.batch_size))


def update_state(state: TurboState, y_next) -> TurboState:
    """Update the region after observing `y_next` (the batch's outcomes).

    Success = the batch's best beats the incumbent by 1e-3 of its magnitude."""
    y = np.asarray(y_next, dtype=float).reshape(-1)
    best = float(np.max(y)) if y.size else -math.inf
    if best > state.best_value + 1e-3 * abs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, best)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def _lengthscales(model, d: int) -> np.ndarray:
    """ARD lengthscales of a fitted kalos GP (normalized-input units), or ones."""
    try:
        ls = model.covar_module.lengthscale
    except AttributeError:
        try:
            ls = model.covar_module.base_kernel.lengthscale
        except AttributeError:
            return np.ones(d)
    arr = ls.detach().cpu().numpy().reshape(-1)
    return arr if arr.size == d and np.all(np.isfinite(arr)) and np.all(arr > 0) else np.ones(d)


def trust_region_bounds(surrogate, bounds, state: TurboState) -> np.ndarray:
    """`(2, d)` raw-unit box of the trust region: centred on the best observed
    point, side `length * w_i` in the unit cube with `w` the lengthscales
    normalised to geometric mean 1, clipped to `bounds`."""
    from .surrogate import (
        sanitize_bounds,  # deferred: keep this module torch-free at import
    )

    assert surrogate.model is not None and surrogate._X is not None and surrogate._y is not None
    lower, upper = sanitize_bounds(bounds)
    span = upper - lower
    X = surrogate._X.detach().cpu().numpy() if hasattr(surrogate._X, "detach") else np.asarray(surrogate._X)
    center = (X[int(np.argmax(surrogate._y))] - lower) / span
    d = len(lower)
    w = _lengthscales(surrogate.model, d)
    w = w / w.mean()
    w = w / np.prod(np.power(w, 1.0 / d))
    lo = np.clip(center - w * state.length / 2.0, 0.0, 1.0)
    hi = np.clip(center + w * state.length / 2.0, 0.0, 1.0)
    return np.vstack([lower + lo * span, lower + hi * span])


__all__ = ["TURBO_MIN_DIM", "TurboState", "trust_region_bounds", "update_state"]
