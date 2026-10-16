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

    kw = dict(exclude_contaminated=not a.include_contaminated)
    all_rows = pd.read_csv(a.data)
    semi = load_real_frame(a.data, DATASET_0, **kw)
    real = load_real_frame(a.data, DATASET_0, real_only=True, **kw)
    syn_df = pd.read_csv(ROOT / "examples/synthetic_bioprocess/synthetic_bioprocess.csv")
    Xs, ys, _ = pool_from_frame(syn_df, "titer_g_L")

    arms = {
        # Only the rows flagged data_type == Real: the only measured data here.
        "real_only": frame_to_pool(real, DATASET_0),
        # Whole table incl. ML-synthesized rows: a semi-synthetic replay, not real data.
        "semi_synthetic_dataset_0": frame_to_pool(semi, DATASET_0),
        "synthetic_fixture": (Xs, ys),
    }
    summary = compare(arms, rounds=a.rounds, q=a.q, seeds=list(range(a.seeds)))
    summary["dataset_0"] = {
        "rows_in_file": int(len(all_rows)),
        "rows_flagged_real": int((all_rows["data_type"] == "Real").sum()),
        "rows_used_real_only": int(len(real)),
        "rows_used_semi_synthetic": int(len(semi)),
        "excluded_contaminated": not a.include_contaminated,
        "roles": {"titer": DATASET_0.titer, "timepoint": DATASET_0.timepoint, "drivers": list(DATASET_0.drivers)},
        "not_modelled": {"min_do_pct": "measured minimum, not a confirmed controllable setpoint"},
        "label": "semi_synthetic_dataset_0 is hybrid/semi-synthetic: only the rows flagged Real are measured",
    }
    jp, hp = write_report(summary, a.out)
    for name, st in summary["arms"].items():
        if "skipped" in st:
            print(f"{name:26s} SKIPPED: {st['skipped']}")
            continue
        print(f"{name:26s} n={st['n_rows']:4d} rounds={st['rounds']:2d} final BO {st['final_bo']:.3f}+/-{st['final_bo_std']:.3f} "
              f"random {st['final_random']:.3f} LHS {st['final_lhs']:.3f}")
    print(f"wrote {jp}\nwrote {hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
