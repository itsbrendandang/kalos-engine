"""Joint feasibility: the chance that every declared spec is met at once.

Each spec (`lo <= value <= hi`, either side open) has its own fitted model.
For each candidate point we draw `n_samples` values from each model's
posterior, one model at a time and independently of the others, and count the
draws where the spec holds (`p_specs`) and the draws where every spec holds
together (`p_all`).

Because the models are fit separately and sampled independently, `p_all` is a
Monte Carlo estimate of the product of the per-spec probabilities: correlation
between outcomes (for example titer and purity moving together) is not
modelled. That is stated in the output (`METHOD`).

Pure numpy: callers pass each model's posterior mean and sd at the points
(a Gaussian posterior per point), and a seed. Advisory only: nothing here
changes which recipes are proposed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Posterior draws per spec per point. With 4096 draws the Monte Carlo standard
# error of a probability is at most 0.008.
N_SAMPLES = 4096

METHOD = (
    "independent_mc: draws from each spec's own model, sampled independently; "
    "the joint probability is the share of draws meeting every spec, an estimate "
    "of the product of the per-spec probabilities (outcome correlation not modelled)"
)


def sample_gaussian(
    mean: np.ndarray, std: np.ndarray, n_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """(n_samples, m) draws from independent N(mean[j], std[j]^2) per point."""
    mean = np.asarray(mean, float).reshape(-1)
    std = np.clip(np.asarray(std, float).reshape(-1), 0.0, None)
    z = rng.standard_normal((int(n_samples), mean.shape[0]))
    return mean[None, :] + z * std[None, :]


def joint_spec_probability(
    samples: Sequence[np.ndarray],
    limits: Sequence[tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray]:
    """From per-spec draws `samples[k]` of shape (n, m) and `limits[k] =
    (lo, hi)` (None = open), return `(p_specs, p_all)`: `p_specs` (k, m) is the
    share of draws meeting spec k, `p_all` (m,) the share meeting every spec in
    the same draw index. All arrays must share (n, m). No specs raises
    ValueError ("all of none" is not a useful answer)."""
    if not samples:
        raise ValueError("at least one spec is needed")
    if len(samples) != len(limits):
        raise ValueError("one (lo, hi) per spec")
    shape = np.asarray(samples[0]).shape
    met_all = np.ones(shape, dtype=bool)
    p_specs = []
    for draws, (lo, hi) in zip(samples, limits):
        d = np.asarray(draws, float)
        if d.shape != shape:
            raise ValueError("every spec needs the same (n_samples, n_points) draws")
        met = np.isfinite(d)
        if lo is not None:
            met &= d >= lo
        if hi is not None:
            met &= d <= hi
        p_specs.append(met.mean(axis=0))
        met_all &= met
    return np.vstack(p_specs), met_all.mean(axis=0)


def gaussian_spec_probability(
    mean: np.ndarray, std: np.ndarray, lo: float | None, hi: float | None,
) -> np.ndarray:
    """Exact P(lo <= value <= hi) under N(mean, std^2), for checking the draws."""
    from scipy.stats import norm

    mean = np.asarray(mean, float)
    std = np.clip(np.asarray(std, float), 1e-300, None)
    upper = norm.cdf((hi - mean) / std) if hi is not None else np.ones_like(mean)
    lower = norm.cdf((lo - mean) / std) if lo is not None else np.zeros_like(mean)
    return np.clip(upper - lower, 0.0, 1.0)


def joint_feasibility(
    means: Sequence[np.ndarray],
    stds: Sequence[np.ndarray],
    limits: Sequence[tuple[float | None, float | None]],
    *,
    seed: int,
    n_samples: int = N_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """`(p_specs, p_all)` from each spec model's posterior mean and sd at the
    points. Seeded: the same inputs and seed give the same numbers. One
    generator, specs drawn in order, so each spec's draws are independent."""
    rng = np.random.default_rng(int(seed))
    samples = [sample_gaussian(m, s, n_samples, rng) for m, s in zip(means, stds)]
    return joint_spec_probability(samples, limits)


__all__ = [
    "METHOD", "N_SAMPLES", "gaussian_spec_probability", "joint_feasibility",
    "joint_spec_probability", "sample_gaussian",
]
