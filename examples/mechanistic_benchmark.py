#!/usr/bin/env python3
"""Plain GP vs GP with a (possibly wrong) Monod fed-batch ODE prior mean, on a simulated surface."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kalos.bench.mechanistic_benchmark import benchmark  # noqa: E402

if __name__ == "__main__":
    r = benchmark(seeds=range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
    for kind, m in r["summary"].items():
        b, t, e = m["best_frac_early"], m["evals_to_95pct"], m["rmse_norm"]
        print(f"{kind:11s} best after 3 evals {b['median']:.2f} [{b['q25']:.2f}-{b['q75']:.2f}]  evals to 95% {t['median']:.0f} [{t['q25']:.0f}-{t['q75']:.0f}]  held-out RMSE/std {e['median']:.2f} [{e['q25']:.2f}-{e['q75']:.2f}]")
    print("hybrid vs plain, seeds faster/slower to 95%:", r["hybrid_vs_plain_evals_to_95pct"], "of", len(r["seeds"]))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(r, indent=2))
