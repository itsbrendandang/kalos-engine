"""Closed-loop race for the TuRBO trust-region option (`kalos.core.turbo`).

Arms on the same seeded LHS init: `bo` (qLogNEI over the full box, the
default), `bo_turbo` (the same acquisition inside a TuRBO-1 trust region,
`propose(..., trust_region=state)`), and `random`. Problems: the sparse
media-DoE-shaped bump from `kalos.bench.saas_experiment.make_problem` (d=12,
4 active dims) and Ackley (d=10, rugged). Reports final simple regret
normalised by the optimum's scale. Synthetic surfaces only.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .benchmark import _lhs, _uniform

ARMS = ("bo", "bo_turbo", "random")


def _problem(name: str, seed: int):
    """(f, bounds, optimum, scale) for a named problem."""
    if name == "sparse_bump_12d":
        from .saas_experiment import make_problem

        p = make_problem(12, seed)
        return p.f, p.bounds, p.scale, p.scale
    if name == "ackley_10d":
        from .objectives import ackley

        o = ackley(10)
        return (lambda X: o.evaluate(X)[0]), o.bounds, o.optimum, o.scale
    raise KeyError(name)


def run_turbo_one(
    problem: str, arm: str, *, seed: int = 0, n_init: int = 10, budget: int = 30,
    noise: float = 0.02, num_restarts: int = 4, raw_samples: int = 128,
) -> float:
    """Final normalised simple regret of one trial."""
    import torch

    from kalos.core.optimize import propose
    from kalos.core.surrogate import FitError, Surrogate
    from kalos.core.turbo import TurboState, update_state

    if arm not in ARMS:
        raise ValueError(arm)
    f, bounds, optimum, scale = _problem(problem, seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = _lhs(n_init, bounds, rng)
    y_true = f(X)
    y_obs = y_true + rng.normal(0.0, noise * scale, len(y_true))
    state = TurboState(dim=X.shape[1], best_value=float(y_obs.max()))
    for _ in range(budget):
        if arm == "random":
            x = _uniform(1, bounds, rng)[0]
        else:
            if state.restart_triggered:
                state = TurboState(dim=X.shape[1], best_value=float(y_obs.max()))
            try:
                s = Surrogate().fit(X, y_obs, bounds=bounds)
                x = propose(s, bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                            trust_region=state if arm == "bo_turbo" else None)[0]
            except FitError:
                x = _uniform(1, bounds, rng)[0]
        yt = f(x[None, :])
        yo = yt + rng.normal(0.0, noise * scale, 1)
        if arm == "bo_turbo":
            update_state(state, yo)
        X = np.vstack([X, x])
        y_true = np.append(y_true, yt)
        y_obs = np.append(y_obs, yo)
    return float((optimum - y_true.max()) / scale)


def turbo_benchmark(
    problems: Iterable[str] = ("sparse_bump_12d", "ackley_10d"),
    seeds: Iterable[int] = range(3), **kw,
) -> dict:
    seeds = list(seeds)
    return {
        p: {a: float(np.mean([run_turbo_one(p, a, seed=s, **kw) for s in seeds])) for a in ARMS}
        for p in problems
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(turbo_benchmark(), indent=1))
