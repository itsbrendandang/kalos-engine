#!/usr/bin/env python3
"""BO vs random vs LHS (same initial design) on the six published media datasets.

Needs the paper's CSVs OUTSIDE this repo:  KALOS_AMII_DATA=/path/to/Data python examples/amii_benchmark.py
Reports median and IQR of evaluations needed to reach the pool's top 5%.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kalos.bench.amii import REGISTRY, benchmark_dataset, load_amii_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports/amii/amii_benchmark.json"))
    ap.add_argument("--seeds", type=int, default=5)
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
            if len(y) < 10 + a.budget:
                results[name]["batches"][str(q)] = {"skipped": f"only {len(y)} unique recipes for budget {a.budget}"}
                print(f"{name} q={q}: SKIPPED, pool of {len(y)} < {10 + a.budget}")
                continue
            r = benchmark_dataset(X, y, q=q, budget=a.budget, seeds=range(a.seeds))
            results[name]["batches"][str(q)] = r
            line = "  ".join(
                f"{arm}: med {v['median']:.0f} [{v['q25']:.0f}-{v['q75']:.0f}] hit {v['reached']}/{v['n_seeds']}"
                for arm, v in r["arms"].items()
            )
            print(f"{name:18s} n={len(y):4d} d={info['n_features']:2d} q={q}  {line}", flush=True)
            out.write_text(json.dumps(results, indent=2))
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
