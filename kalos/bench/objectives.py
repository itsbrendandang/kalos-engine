"""Synthetic bioprocess-like objectives with KNOWN optima.

Every objective is a MAXIMIZATION problem (titer / yield up is good), exposes its
true optimum so simple regret can be measured, and takes an observation-noise
argument so the benchmark can sweep the regime where a model's advantage over
random sampling appears or disappears. That noise sweep is the honest link to
real data: on the real media DoE the held-out signal is weak (grouped-CV
Spearman ~0.37-0.52), and the noise level is a large part of why.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Objective:
    name: str
    dim: int
    bounds: np.ndarray  # (2, d): [lower_row, upper_row]
    optimum: float  # max of the noiseless surface
    scale: float  # typical output spread, so a noise fraction is meaningful
    _f: Callable[[np.ndarray], np.ndarray]

    def evaluate(
        self, X, noise: float = 0.0, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_obs): the noiseless truth and a noisy observation.

        `noise` is a FRACTION of the objective's output scale, so `noise=0.15`
        means Gaussian measurement noise with sigma = 15% of the surface's spread
        - comparable across objectives with different units.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y_true = np.asarray(self._f(X), dtype=float).reshape(-1)
        if noise <= 0.0:
            return y_true, y_true.copy()
        if rng is None:
            rng = np.random.default_rng()
        sigma = noise * self.scale
        y_obs = y_true + rng.normal(0.0, sigma, size=y_true.shape)
        return y_true, y_obs


def gaussian_bump(dim: int = 4) -> Objective:
    """Smooth unimodal 'titer' surface: one Gaussian peak. Max = 10 at the center.

    The easy case. A model should help a little, but even random sampling does
    fine because there is a single wide basin of attraction.
    """
    center = np.full(dim, 0.6)
    amp, beta = 10.0, 4.0

    def f(X: np.ndarray) -> np.ndarray:
        d2 = ((X - center) ** 2).sum(axis=1)
        return amp * np.exp(-beta * d2)

    bounds = np.vstack([np.zeros(dim), np.ones(dim)])
    return Objective(f"gaussian_bump_{dim}d", dim, bounds, amp, amp, f)


def ackley(dim: int = 4) -> Objective:
    """Negated Ackley on [-2, 2]^d: rugged and multimodal. Max = 0 at the origin.

    The hard case. The surface has structure worth learning, so a model-based
    optimizer should beat space-filling by a wider margin here than on the bump -
    this is where BO is supposed to earn its keep.
    """
    a, b, c = 20.0, 0.2, 2.0 * np.pi

    def f(X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        s1 = (X ** 2).sum(axis=1)
        s2 = np.cos(c * X).sum(axis=1)
        ack = -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e
        return -ack  # negate: maximization, optimum 0 at the origin

    bounds = np.vstack([np.full(dim, -2.0), np.full(dim, 2.0)])
    # Typical spread of -Ackley over [-2,2]^d is a few units; 6 is a fair scale.
    return Objective(f"ackley_{dim}d", dim, bounds, 0.0, 6.0, f)


@dataclass
class MixedObjective:
    """A synthetic objective over a MIXED continuous + categorical design space.

    The design vector is continuous coordinates first, then integer-coded
    categorical levels (the same layout the engine's `cat_dims` assumes). Like
    `Objective` it is a maximization problem, exposes its true optimum for simple
    regret, and takes a noise fraction of `scale`. This is the categorical
    analogue used to check that the mixed BO loop beats random choice.
    """

    name: str
    cont_bounds: np.ndarray  # (2, n_cont): [lower_row, upper_row] over the continuous dims
    cat_cardinalities: tuple[int, ...]  # level count per categorical dim
    optimum: float
    scale: float
    _f: Callable[[np.ndarray], np.ndarray]

    @property
    def n_cont(self) -> int:
        return self.cont_bounds.shape[1]

    @property
    def cat_dims(self) -> list:
        return list(range(self.n_cont, self.n_cont + len(self.cat_cardinalities)))

    def bounds(self) -> np.ndarray:
        """Full `(2, d)` engine box: continuous ranges then `[0, card-1]` per cat."""
        lo = np.concatenate([self.cont_bounds[0], np.zeros(len(self.cat_cardinalities))])
        hi = np.concatenate([
            self.cont_bounds[1],
            np.array([c - 1 for c in self.cat_cardinalities], float),
        ])
        return np.vstack([lo, hi])

    def evaluate(
        self, X, noise: float = 0.0, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(y_true, y_obs)`; `noise` is a fraction of `scale` (see `Objective`)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y_true = np.asarray(self._f(X), dtype=float).reshape(-1)
        if noise <= 0.0:
            return y_true, y_true.copy()
        if rng is None:
            rng = np.random.default_rng()
        y_obs = y_true + rng.normal(0.0, noise * self.scale, size=y_true.shape)
        return y_true, y_obs


def mixed_bump(n_cont: int = 2, n_levels: int = 3) -> MixedObjective:
    """A Gaussian bump over `n_cont` continuous dims plus one categorical dim whose
    best level adds a bonus. Max = 13 at the bump center on the best level.

    A model that ignores the categorical dimension leaves the +3 bonus on the
    table, so a working mixed loop must both locate the continuous peak AND pick
    the right level - exactly what a discrete process choice (which resin, which
    catalyst) looks like.
    """
    center = np.full(n_cont, 0.6)
    amp, beta, cat_bonus = 10.0, 4.0, 3.0
    best_level = n_levels - 1

    def f(X: np.ndarray) -> np.ndarray:
        Xc = X[:, :n_cont]
        cat = np.rint(X[:, n_cont]).astype(int)
        bump = amp * np.exp(-beta * ((Xc - center) ** 2).sum(axis=1))
        return bump + np.where(cat == best_level, cat_bonus, 0.0)

    optimum = amp + cat_bonus
    cont_bounds = np.vstack([np.zeros(n_cont), np.ones(n_cont)])
    return MixedObjective(
        f"mixed_bump_{n_cont}c{n_levels}k", cont_bounds, (n_levels,), optimum, optimum, f
    )


def bioprocess_like() -> MixedObjective:
    """Synthetic bioprocess-like titer surface: 3 numeric factors plus 2 categorical.

    Numeric factors `u` are normalised to [0, 1] (think temperature, pH and feed
    rate). The categorical factors are a strain (4 levels, codes 0-3) and an
    inducer (3 levels, codes 0-2). Each strain has a baseline offset and its own
    peak location and width, and each inducer adds a gain:

        f = off[s] + gain[i] + 6 * exp(-0.5 * || (u - c[s]) / w[s] ||^2)

    The trap: the strain with the best offset (s=3) also has the narrowest peak,
    so a method has to find a small region. The optimum is exact:
    max over (s, i) of off + gain, plus the amplitude 6 (10.5 here).

    This is a made-up surface for the benchmark protocol in docs/BENCHMARK.md.
    It is NOT measured process data, and nothing here is calibrated to a real
    process.
    """
    off = np.array([0.0, 1.0, 0.5, 2.0])
    gain = np.array([0.0, 0.8, 2.5])
    centers = np.array([
        [0.3, 0.6, 0.4],
        [0.7, 0.3, 0.6],
        [0.5, 0.8, 0.2],
        [0.2, 0.2, 0.8],
    ])
    widths = np.array([0.35, 0.20, 0.15, 0.30])
    amp = 6.0

    def f(X: np.ndarray) -> np.ndarray:
        u = X[:, :3]
        s = np.rint(X[:, 3]).astype(int)
        i = np.rint(X[:, 4]).astype(int)
        d2 = (((u - centers[s]) / widths[s, None]) ** 2).sum(axis=1)
        return off[s] + gain[i] + amp * np.exp(-0.5 * d2)

    optimum = float(off.max() + gain.max() + amp)
    cont_bounds = np.vstack([np.zeros(3), np.ones(3)])
    return MixedObjective(
        "bioprocess_like_mixed", cont_bounds, (4, 3), optimum, amp, f
    )


def cho_fed_batch() -> Objective:
    """Mechanistic CHO fed-batch surface (`kalos.bench.cho_sim`): ODE kinetics,
    interior optimum in feed / glucose / pH. Lazy import: the ODE module loads
    only when this objective is built."""
    from .cho_sim import cho_objective

    return cho_objective()


# The benchmark objective registry: name -> zero-argument factory. Bench
# harnesses and CLIs look objectives up here instead of importing each builder.
OBJECTIVES: dict[str, Callable[[], Objective | MixedObjective]] = {
    "gaussian_bump_4d": lambda: gaussian_bump(4),
    "ackley_4d": lambda: ackley(4),
    "mixed_bump_2c3k": lambda: mixed_bump(2, 3),
    "bioprocess_like_mixed": bioprocess_like,
    "cho_fed_batch_4d": cho_fed_batch,
}


def get_objective(name: str) -> Objective | MixedObjective:
    """Build a registered objective by name (`OBJECTIVES`)."""
    try:
        factory = OBJECTIVES[name]
    except KeyError:
        raise KeyError(f"unknown objective {name!r}; registered: {sorted(OBJECTIVES)}") from None
    return factory()
