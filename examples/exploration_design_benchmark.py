"""Exploration design vs the model's picks while the sheet has few recipes
(QUEUE round 3 item 6; engine-lit section 2 row 6).

Synthetic surfaces only (kalos.bench.objectives): not evidence about a real
process. Both arms put the model's best bet in every batch, as the engine
does; they differ in the other q-1 recipes while fewer than 2 x d distinct
recipes have been run: the exploration arm spreads them (maximin), the model
arm uses the batch acquisition. After that both use the acquisition.

    python examples/exploration_design_benchmark.py [--seeds 10] [--rounds 3]

Prints, per surface, the mean simple regret of the best TRUE value found and
a bootstrap 95% interval of the paired difference (exploration - model;
negative = exploration better).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from kalos.bench.objectives import ackley, gaussian_bump
from kalos.core.optimize import posterior_mean_optimum, propose_batch
from kalos.core.space_filling import maximin_batch
from kalos.core.surrogate import Surrogate


def run(obj, *, strategy: str, seed: int, n_init: int, q: int, rounds: int, noise: float) -> float:
    rng = np.random.default_rng(seed)
    lo, hi = obj.bounds
    X = lo + rng.uniform(size=(n_init, obj.dim)) * (hi - lo)
    y_true, y = obj.evaluate(X, noise=noise, rng=rng)
    for r in range(rounds):
        s = Surrogate().fit(X, y, bounds=obj.bounds)
        best = posterior_mean_optimum(s, obj.bounds, seed=seed + r).reshape(1, -1)
        if strategy == "exploration" and len(X) < 2 * obj.dim:
            rest = maximin_batch(X, obj.bounds, q - 1, seed=seed + r, pending=best)
        else:
            rest = propose_batch(s, obj.bounds, q=q - 1, pending=best, seed=seed + r).X
        new = np.vstack([best, rest])
        t, o = obj.evaluate(new, noise=noise, rng=rng)
        X, y, y_true = np.vstack([X, new]), np.concatenate([y, o]), np.concatenate([y_true, t])
    return float(obj.optimum - y_true.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    out = {}
    for obj in (gaussian_bump(4), ackley(4)):
        diffs, reg = [], {"exploration": [], "model": []}
        for seed in range(1000, 1000 + args.seeds):
            a = run(obj, strategy="exploration", seed=seed, n_init=4, q=4, rounds=args.rounds, noise=0.15)
            b = run(obj, strategy="model", seed=seed, n_init=4, q=4, rounds=args.rounds, noise=0.15)
            reg["exploration"].append(a)
            reg["model"].append(b)
            diffs.append(a - b)
        d = np.asarray(diffs)
        boot = np.random.default_rng(0).choice(d, (5000, len(d))).mean(1)
        out[obj.name] = {
            "regret_exploration": round(float(np.mean(reg["exploration"])), 3),
            "regret_model": round(float(np.mean(reg["model"])), 3),
            "paired_diff_mean": round(float(d.mean()), 3),
            "paired_diff_ci95": [round(float(np.quantile(boot, 0.025)), 3),
                                 round(float(np.quantile(boot, 0.975)), 3)],
            "seeds": args.seeds, "rounds": args.rounds, "n_init": 4, "q": 4, "noise": 0.15,
        }
        print(obj.name, json.dumps(out[obj.name]), flush=True)


if __name__ == "__main__":
    main()
