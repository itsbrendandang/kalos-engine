"""When does a mechanistic prior mean help (or hurt) a GP? A controlled simulation.

TRUTH is a Monod fed-batch with substrate inhibition (Haldane). The MODEL handed
to `HybridSurrogate` is the same family with deliberately wrong parameters / no
inhibition, at four misspecification levels. Design x = [S0 (g/L), F (feed, L/h)];
response = final product mass. Strategies are compared on pool-replay BO
(best-so-far after a fixed budget) and held-out prediction error, over >= 5 seeds.

This is a simulation of the interface's behaviour, not evidence about real
cultures. The mechanistic bridge stays opt-in; see docs in kalos/core/mechanistic.py.
"""
from __future__ import annotations

import numpy as np

from kalos.core.mechanistic import HybridSurrogate, ODEModel

SF = 200.0  # feed substrate concentration, g/L
T_END = 48.0
V0 = 1.0
BOUNDS = np.array([[5.0, 0.0], [60.0, 0.05]])  # S0, F


def fed_batch(mu_max: float, ks: float, ki: float | None, y_xs: float, qp: float) -> ODEModel:
    """ODE model; `ki=None` means no substrate inhibition (plain Monod)."""

    def rhs(t, state, p):
        x, s, prod, v = state
        s = max(s, 0.0)
        f = p[1]
        mu = mu_max * s / (ks + s + (s * s / ki if ki else 0.0))
        d = f / v
        return [mu * x - d * x, -mu * x / y_xs + d * (SF - s), qp * x - d * prod, f]

    return ODEModel(
        rhs=rhs,
        x_to_params=lambda x: (x[0], x[1]),
        x_to_y0=lambda x: [0.1, float(x[0]), 0.0, V0],
        readout=lambda t, y: y[2, -1] * y[3, -1],  # final product mass
        t_end=T_END,
    )


TRUTH = fed_batch(mu_max=0.35, ks=4.0, ki=45.0, y_xs=0.45, qp=0.25)
LEVELS = {
    "oracle": TRUTH,
    "mild": fed_batch(mu_max=0.30, ks=5.0, ki=70.0, y_xs=0.40, qp=0.22),  # ~15-50% off
    "severe": fed_batch(mu_max=0.60, ks=0.5, ki=None, y_xs=0.20, qp=0.05),  # no inhibition, wrong yield
    # strong inhibition at low substrate. NOTE: despite the name this prior is NOT adversarial:
    # its rank correlation with the truth over the pool grid is about +0.81 and its argmax is next
    # to the true one (Spearman +0.81, argmax [5, 0] vs true [5, 0.021]; reproduced locally); only its magnitude is far off. "severe" is the level
    # that is anti-correlated with the truth. Names kept so the committed results JSON stays valid.
    "adversarial": fed_batch(mu_max=0.35, ks=4.0, ki=3.0, y_xs=0.45, qp=0.25),
}


def make_pool(n_side: int = 13) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(0, 1, n_side)
    X = np.array([[BOUNDS[0, 0] + a * (BOUNDS[1, 0] - BOUNDS[0, 0]), BOUNDS[0, 1] + b * (BOUNDS[1, 1] - BOUNDS[0, 1])]
                  for a in g for b in g])
    return X, TRUTH.evaluate(X)


def _fit_predict(kind: str, Xtr, ytr, Xte, bounds):
    from kalos.core.surrogate import Surrogate

    if kind == "plain":
        s = Surrogate().fit(Xtr, ytr, bounds=bounds)
        mu, sd = s.posterior(Xte)
        return np.asarray(mu, float).reshape(-1), np.asarray(sd, float).reshape(-1)
    h = HybridSurrogate(LEVELS[kind]).fit(Xtr, ytr, bounds=bounds)
    return h.predict(Xte)


def run_seed(
    seed: int, X: np.ndarray, y: np.ndarray, *, n_init: int = 4, budget: int = 8, q: int = 1, test_frac: float = 0.3
) -> dict:
    """Pool-replay BO per strategy from the SAME initial design.

    One FIXED held-out split per seed (`test_frac` of the pool) is set aside before any
    strategy runs: no strategy may pick or train on it, and every strategy's prediction
    error is measured on exactly those rows, so the RMSEs are comparable.
    """
    import torch
    from scipy.stats import norm

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(test_frac * len(y))
    test = np.array(sorted(int(i) for i in order[:n_test]))
    cand = [int(i) for i in order[n_test:]]
    init = cand[:n_init]
    pool = set(cand)
    out: dict[str, dict] = {}
    for kind in ("plain", *LEVELS):
        ev = list(init)
        curve = [float(y[ev].max() / y.max())]
        while len(ev) < n_init + budget:
            rem = np.array(sorted(pool - set(ev)))
            mu, sd = _fit_predict(kind, X[ev], y[ev], X[rem], BOUNDS)
            sd = np.maximum(sd, 1e-9)
            best = y[ev].max()
            z = (mu - best) / sd
            ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
            picks = [int(rem[j]) for j in np.argsort(-ei)[:q]]
            ev += picks
            curve += [float(y[ev].max() / y.max())] * len(picks)
        mu, _ = _fit_predict(kind, X[ev], y[ev], X[test], BOUNDS)
        reached = next((n_init + i for i, c in enumerate(curve) if c >= 0.95), None)
        out[kind] = {
            "best_frac": curve[-1],
            "best_frac_early": curve[min(len(curve) - 1, 3)],  # after 3 BO evaluations
            "evals_to_95pct": reached,  # None = never reached within the budget (not a number)
            "rmse_norm": float(np.sqrt(np.mean((mu - y[test]) ** 2)) / y.std()),
        }
    return out


def _spread(v: list[float]) -> dict:
    a = np.asarray(v, float)
    if a.size == 0:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    return {"median": float(np.median(a)), "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75)), "n": int(a.size)}


def _compare_reach(h: int | None, p: int | None) -> int:
    """+1 hybrid reached sooner (or it alone reached), -1 slower (or only plain reached), 0 tie / both missed."""
    if h is None and p is None:
        return 0
    if p is None:
        return 1
    if h is None:
        return -1
    return (h < p) - (h > p)


def benchmark(seeds=range(5), **kw) -> dict:
    seeds = list(seeds)
    if len(set(seeds)) < 5:
        raise ValueError("use at least 5 distinct seeds")
    X, y = make_pool()
    per = [run_seed(s, X, y, **kw) for s in seeds]
    summary: dict = {}
    for kind in per[0]:
        for metric in ("best_frac", "best_frac_early", "rmse_norm"):
            summary.setdefault(kind, {})[metric] = _spread([p[kind][metric] for p in per])
        reached = [p[kind]["evals_to_95pct"] for p in per if p[kind]["evals_to_95pct"] is not None]
        summary[kind]["evals_to_95pct"] = {**_spread(reached), "reach_rate": len(reached) / len(per)}
    tallies = {}
    for kind in LEVELS:
        c = [_compare_reach(p[kind]["evals_to_95pct"], p["plain"]["evals_to_95pct"]) for p in per]
        tallies[kind] = {"better": c.count(1), "worse": c.count(-1), "tied_or_both_missed": c.count(0)}
    return {
        "summary": summary, "seeds": seeds, "hybrid_vs_plain_evals_to_95pct": tallies, "pool_size": int(len(y)),
        "notes": "evals_to_95pct spread is over seeds that reached 95%; see reach_rate. One fixed held-out split per seed for rmse_norm.",
    }
