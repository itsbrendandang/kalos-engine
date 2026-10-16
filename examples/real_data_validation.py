#!/usr/bin/env python3
"""Replay a 10-round, q=2 BO loop on the real fermentation sheet and compare its
convergence with the synthetic fixture. Writes a summary JSON and a Plotly HTML.

Run:  python examples/real_data_validation.py [--out reports/real_data] [--include-contaminated]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kalos.bench.pool import pool_from_frame  # noqa: E402
from kalos.bench.real_data import (  # noqa: E402
    DATASET_0, compare, frame_to_pool, load_real_frame, write_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "examples/datasets/dataset_0.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports/real_data"))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("-q", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--include-contaminated", action="store_true")
    ap.add_argument("--indpensim", nargs="?", const="env", default=None,
                    help="also benchmark IndPenSim (CSV path, or omit value to use KALOS_INDPENSIM_PATH/KALOS_MEDIA_DATA)")
    a = ap.parse_args()

    import pandas as pd

    raw_rows = len(pd.read_csv(a.data))
    df = load_real_frame(a.data, DATASET_0, exclude_contaminated=not a.include_contaminated)
    real = frame_to_pool(df, DATASET_0)
    syn_df = pd.read_csv(ROOT / "examples/synthetic_bioprocess/synthetic_bioprocess.csv")
    X, y, _ = pool_from_frame(syn_df, "titer_g_L")

    summary = compare(real, (X, y), rounds=a.rounds, q=a.q, seeds=list(range(a.seeds)))
    summary["real"]["rows_in_file"] = raw_rows
    summary["real"]["label"] = (
        "hybrid/semi-synthetic: only the first ~20 source runs are real, the rest are ML-synthesized"
    )
    summary["real"]["roles"] = {
        "titer": DATASET_0.titer, "timepoint": DATASET_0.timepoint, "drivers": list(DATASET_0.drivers),
    }
    summary["real"]["excluded_contaminated"] = not a.include_contaminated
    if a.indpensim:
        from kalos.bench.real_data import convergence_summary, load_indpensim_pool

        Xi, yi, _ = load_indpensim_pool(a.indpensim if a.indpensim != "env" else None)
        summary["indpensim"] = convergence_summary(
            Xi, yi, rounds=a.rounds, q=a.q, n_init=6, seeds=list(range(a.seeds))
        )
    jp, hp = write_report(summary, a.out)
    for k in ("real", "synthetic"):
        s = summary[k]
        print(f"{k:9s} n={s['n_rows']:4d} final BO {s['final_bo']:.3f}+/-{s['final_bo_std']:.3f} "
              f"random {s['final_random']:.3f} LHS {s['final_lhs']:.3f} "
              f"(95% of best at round {s['round_to_95pct_best']})")
    print(f"wrote {jp}\nwrote {hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
