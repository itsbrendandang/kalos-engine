#!/usr/bin/env python3
"""Pool replay on the open datasets: experiments-to-90%-of-pool-max, BO vs random.

    KALOS_OPEN_DATA=... KALOS_AMII_DATA=... python examples/open_pool_replay.py \\
        --dataset all --seeds 20 --batch 4 --out /tmp/redesign/replay/pool

Protocol (retrospective, not validation): every arm starts from the SAME seeded
random initial rows, then spends the budget in batches of `--batch` on the
REMAINING measured rows. Arms come from `kalos.bench.amii.run_trial`: BO (top-q
expected improvement on the revealed labels), random, and LHS-style nearest
neighbour (space-filling). Categorical columns are one-hot encoded. Amii sets use
`kalos.bench.amii.load_amii_pool` (its registry, replicate-averaged rows).

Metric: the first evaluation at which the best-so-far reaches 90% of the pool max
(0.9 * max when every value is >= 0, else min + 0.9 * (max - min)). Seeds that never
get there are right-censored at n_init + budget + 1. Reported: per-arm mean, the
paired BO-minus-random mean difference with a 95% bootstrap CI over seeds, and the
exact paired sign test. With pools of 13 to 20 rows no result here is evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import binomtest  # noqa: E402

from kalos.bench.amii import load_amii_pool, run_trial  # noqa: E402
from kalos.bench.open_datasets import AMII_SETS, load_open_sheet  # noqa: E402

SMALL = {"candida_rsm_nr1": (5, 8), "trichoderma_ambr": (6, 14), "trichoderma_ambr_condition": (6, 14)}  # (n_init, budget) for tiny pools


def encode(name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(X numeric, y, feature names) for one dataset. Categoricals are one-hot."""
    if name in AMII_SETS:
        X, y, info = load_amii_pool(name)
        return X, y, list(info["features"])
    sheet = load_open_sheet(name)
    ids = list((sheet.roles or {}).get("ids", []))
    df = sheet.df.drop(columns=ids)
    cats = [c for c in df.columns if c != sheet.target and (not pd.api.types.is_numeric_dtype(df[c]) or c in (sheet.roles or {}).get("categoricals", []))]
    X = pd.get_dummies(df.drop(columns=[sheet.target]), columns=cats, dtype=float)
    return X.to_numpy(float), df[sheet.target].to_numpy(float), list(X.columns)


def runs_to(curve, thr: float) -> int | None:
    hit = np.nonzero(np.asarray(curve) >= thr - 1e-12)[0]
    return int(hit[0]) + 1 if hit.size else None


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def replay(name: str, *, seeds: int, batch: int, budget_cap: int) -> dict:
    X, y, _feats = encode(name)
    n = len(y)
    n_init, budget = SMALL.get(name, (10, budget_cap))
    budget = min(budget, n - n_init)
    thr = 0.9 * float(y.max()) if float(y.min()) >= 0 else float(y.min() + 0.9 * (y.max() - y.min()))
    cap = n_init + budget + 1
    per: dict[str, list[int]] = {"bo": [], "random": [], "lhs": []}
    curves: dict[str, list[np.ndarray]] = {"bo": [], "random": [], "lhs": []}
    fallbacks = 0
    for s in range(seeds):
        st: dict = {}
        out = run_trial(X, y, q=batch, budget=budget, n_init=n_init, seed=s, stats=st)
        fallbacks += st.get("fit_fallbacks", 0)
        for arm in per:
            c = out[arm]
            curves[arm].append(np.asarray(c))
            r = runs_to(c, thr)
            per[arm].append(cap if r is None else r)
    res = {"dataset": name, "n_rows": n, "n_features": X.shape[1], "y_min": float(y.min()),
           "y_max": float(y.max()), "threshold": thr, "n_init": n_init, "budget": budget,
           "batch": batch, "seeds": seeds, "censored_at": cap, "bo_fit_fallbacks": fallbacks,
           "arms": {}}
    for arm in per:
        cur = np.vstack(curves[arm])
        res["arms"][arm] = {
            "mean_runs_to_90pct": float(np.mean(per[arm])),
            "reached": int(sum(v < cap for v in per[arm])),
            "final_best_mean": float(cur[:, -1].mean()),
            # best-so-far after k evaluations (k counts the initial rows too)
            "best_at": {str(k): float(cur[:, k - 1].mean())
                        for k in sorted({n_init, min(n_init + 10, n_init + budget), n_init + budget})},
        }
    bo, rnd = np.asarray(per["bo"], float), np.asarray(per["random"], float)
    diff = bo - rnd  # negative = BO reached 90% in fewer experiments
    lo, hi = bootstrap_ci(diff)
    wins, losses = int((diff < 0).sum()), int((diff > 0).sum())
    p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    res["bo_minus_random"] = {"mean": float(diff.mean()), "ci95": [lo, hi], "bo_fewer": wins,
                              "bo_more": losses, "tied": int((diff == 0).sum()), "sign_p": p}
    lhs = np.asarray(per["lhs"], float)
    res["bo_minus_lhs"] = {"mean": float((bo - lhs).mean()),
                           "ci95": list(bootstrap_ci(bo - lhs))}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--out", default="/tmp/redesign/replay/pool")
    a = ap.parse_args()
    names = list(AMII_SETS) + ["pichia_HSA", "pichia_HerceptinA", "pichia_RBDJ", "trichoderma_ambr",
                               "trichoderma_ambr_condition", "candida_rsm_nr1"] if a.dataset == ["all"] else a.dataset
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        r = replay(name, seeds=a.seeds, batch=a.batch, budget_cap=a.budget)
        results.append(r)
        d = r["bo_minus_random"]
        print(f"{name:20s} n={r['n_rows']:4d} thr={r['threshold']:.3f} "
              f"BO {r['arms']['bo']['mean_runs_to_90pct']:.1f} rnd {r['arms']['random']['mean_runs_to_90pct']:.1f} "
              f"lhs {r['arms']['lhs']['mean_runs_to_90pct']:.1f} | BO-rnd {d['mean']:+.1f} "
              f"CI[{d['ci95'][0]:+.1f},{d['ci95'][1]:+.1f}] fewer {d['bo_fewer']} more {d['bo_more']} p={d['sign_p']:.3f}",
              flush=True)
        (out / f"{name}.json").write_text(json.dumps(r, indent=1))
    (out / "all.json").write_text(json.dumps(results, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
