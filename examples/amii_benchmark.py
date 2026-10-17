#!/usr/bin/env python3
"""BO vs random vs LHS (same initial design) on the six published media datasets.

Needs the paper's CSVs OUTSIDE this repo:  KALOS_AMII_DATA=/path/to/Data python examples/amii_benchmark.py
Reports median and IQR of evaluations needed to reach the pool's top 5%, plus the paired
per-seed BO-vs-other comparison (sign-test p, Holm-adjusted across datasets x batch sizes).
With few seeds nothing here is significant; use --seeds 20 or more for any claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kalos.bench.amii import REGISTRY, benchmark_dataset, load_amii_pool  # noqa: E402


N_INIT = 10


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [1.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="amii_benchmark.json")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--datasets", nargs="+", default=list(REGISTRY))
    a = ap.parse_args()

    results: dict = {}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in a.datasets:
        try:
            X, y, info = load_amii_pool(name, a.data)
        except FileNotFoundError as exc:
            print(f"{name}: SKIPPED ({exc})", file=sys.stderr)
            continue
        results[name] = {"info": info, "batches": {}}
        for q in a.batches:
            if len(y) < N_INIT + a.budget:
                results[name]["batches"][str(q)] = {"skipped": f"only {len(y)} unique recipes for budget {a.budget}"}
                print(f"{name} q={q}: SKIPPED, pool of {len(y)} < {N_INIT + a.budget}")
                continue
            r = benchmark_dataset(X, y, q=q, budget=a.budget, n_init=N_INIT, seeds=range(a.seeds))
            results[name]["batches"][str(q)] = r
            line = "  ".join(
                f"{arm}: med {v['median']:.0f} [{v['q25']:.0f}-{v['q75']:.0f}] hit {v['reached']}/{v['n_seeds']}"
                for arm, v in r["arms"].items()
            )
            print(f"{name:18s} n={len(y):4d} d={info['n_features']:2d} q={q}  {line}", flush=True)
            out.write_text(json.dumps(results, indent=2))
    cells = [(n, q, arm, c) for n, v in results.items() for q, b in v["batches"].items() if "arms" in b
             for arm, c in b["paired_vs_bo"].items()]
    for (n, q, arm, c), padj in zip(cells, holm([c["sign_test_p"] for *_, c in cells])):
        c["sign_test_p_holm"] = padj
        print(f"{n:18s} q={q} BO vs {arm:6s}: fewer {c['bo_fewer']} more {c['bo_more']} tied {c['tied']} "
              f"p={c['sign_test_p']:.3f} holm={padj:.3f}")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
