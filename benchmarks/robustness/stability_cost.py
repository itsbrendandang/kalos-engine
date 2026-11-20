"""CPU and wall cost of the robustness checks on the 160-row example.

    python benchmarks/robustness/stability_cost.py mixed 5       # synthetic_bioprocess as is
    python benchmarks/robustness/stability_cost.py continuous 5  # numeric columns only

Prints one RESULT line (JSON): total CPU/wall seconds of the analysis and of the
two robustness parts. Serial unless `kalos.core.parallel` (perf/parallel-analysis)
is present; the pool workers' CPU is not counted in the parent's CPU time, so
compare serial runs by CPU seconds. Wall time depends on machine load (reported).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMES: dict[str, tuple[float, float]] = {}


def _wrap(mod, name: str) -> None:
    f = getattr(mod, name)

    def g(*a, **k):
        c, w = time.process_time(), time.perf_counter()
        r = f(*a, **k)
        TIMES[name] = (time.process_time() - c, time.perf_counter() - w)
        return r

    setattr(mod, name, g)


def main() -> None:
    import pandas as pd
    import torch

    import kalos.core.gp_relevance as gr
    import kalos.portal.analysis_parts.stability as st
    from kalos.portal.analysis import _analyze

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    which, k = sys.argv[1], int(sys.argv[2])
    df = pd.read_csv(ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv")
    if which == "continuous":
        df = df.select_dtypes("number")
    _wrap(gr, "subsample_relevance")
    _wrap(st, "proposal_stability_block")
    c, w = time.process_time(), time.perf_counter()
    r = _analyze(df, target="titer_g_L", stability_seeds=k)
    tot = (time.process_time() - c, time.perf_counter() - w)
    ps = r.get("proposal_stability") or {}
    out = {
        "sheet": which, "rows": len(df), "k": k,
        "total_cpu_s": round(tot[0], 1), "total_wall_s": round(tot[1], 1),
        "parts_cpu_wall_s": {n: [round(a, 1), round(b, 1)] for n, (a, b) in TIMES.items()},
        "workers": ps.get("workers"), "batch_stability": ps.get("batch_stability"),
        "n_unstable": ps.get("n_unstable"),
        "agreement": [p["agreement"] for p in ps.get("per_proposal", [])],
        "median_nearest_distance": [p["median_nearest_distance"] for p in ps.get("per_proposal", [])],
        "gp_reason": r["driver_second_opinion"]["gp"]["reason"],
        "flags": {f["name"]: f["flags"] for f in r["driver_second_opinion"]["features"] if f["flags"]},
        "load_average": os.getloadavg()[0],
    }
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
