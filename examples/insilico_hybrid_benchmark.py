"""Benchmark on DataHow's public in-silico fed-batch runs (MIT; data not in
this repository; see benchmarks/INSILICO_HYBRID.md).

Two questions, both on SIMULATED runs:
1. Fixed split: fit the GP on the 20 train runs' endpoint table and rank the
   100 test runs (Spearman, with a bootstrap 95% interval).
2. Pool replay over all 120 runs: Kalos's pool BO (kalos.bench.pool) vs random
   picks; runs needed to reach 90% of the pool's best titer, per seed, and the
   paired difference with a bootstrap interval.

    KALOS_DATAHOW_DATA=/path/to/data python examples/insilico_hybrid_benchmark.py [--seeds 20]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from kalos.bench.insilico_hybrid import SOURCE, TARGET, load_runs, varying_inputs
from kalos.bench.pool import run_pool_one
from kalos.core.surrogate import Surrogate


def _boot_ci(vals: np.ndarray, seed: int = 0) -> list[float]:
    b = np.random.default_rng(seed).choice(vals, (5000, len(vals))).mean(1)
    return [round(float(np.quantile(b, 0.025)), 3), round(float(np.quantile(b, 0.975)), 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="noisy", choices=["noisy", "fixed"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    args = ap.parse_args()
    df = load_runs(variant=args.variant)
    feats = varying_inputs(df)
    X, y = df[feats].to_numpy(float), df[TARGET].to_numpy(float)
    train = (df["split"] == "train").to_numpy()
    bounds = np.vstack([X.min(0), X.max(0)])

    s = Surrogate().fit(X[train], y[train], bounds=bounds, restarts=3, restart_seed=1234)
    mu, _ = s.posterior(X[~train])
    rho = float(spearmanr(mu, y[~train]).statistic)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(~train)
    boots = []
    for _ in range(2000):
        b = rng.choice(len(idx), len(idx))
        boots.append(spearmanr(np.asarray(mu)[b], y[idx][b]).statistic)
    split = {"spearman_test": round(rho, 3),
             "ci95": [round(float(np.nanquantile(boots, q)), 3) for q in (0.025, 0.975)],
             "n_train": int(train.sum()), "n_test": int((~train).sum())}

    target = 0.9 * float(y.max())
    reach: dict[str, list[int]] = {"bo": [], "random": []}
    for seed in range(args.seeds):
        for strat in reach:
            traj = run_pool_one(X, y, strat, n_init=args.n_init, budget=args.budget, seed=seed)
            hit = np.flatnonzero(traj >= target)
            reach[strat].append(int(hit[0]) + 1 if len(hit) else args.n_init + args.budget + 1)
    diff = np.asarray(reach["bo"]) - np.asarray(reach["random"])
    pool = {"runs_to_90pct_bo": round(float(np.mean(reach["bo"])), 2),
            "runs_to_90pct_random": round(float(np.mean(reach["random"])), 2),
            "paired_diff_mean": round(float(diff.mean()), 2), "paired_diff_ci95": _boot_ci(diff),
            "censored_at": args.n_init + args.budget + 1,
            "seeds": args.seeds, "n_init": args.n_init, "budget": args.budget,
            "pool_size": len(y)}
    print(json.dumps({"source": SOURCE, "variant": args.variant, "inputs": feats,
                      "fixed_split": split, "pool_replay": pool}, indent=1))


if __name__ == "__main__":
    main()
