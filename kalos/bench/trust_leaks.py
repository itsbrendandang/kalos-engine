"""False-trust rates of the reliability verdict on the ML audit's leak scenarios
(c1 batch offsets, c2 as-run replicates, f1 pure noise) - REORG F04/F05.

Every scenario has NO learnable input signal, so any "trusted" verdict is false
trust. Reported per scenario: the rate of `reliability.clears_floor`, the rate
of `verdict.state == "reliable"` when the analysis has one, the rate at which
proposals were withheld, and the median reported CV Spearman.

    python -m kalos.bench.trust_leaks --seeds 20 --procs 6 --out benchmarks/results/trust_leaks.json

Scenarios (generators follow /tmp/ml-audit leak.py and refusal.py of the audit):
  batch_text      6 batches x 8 runs clustered in space, batch offset sd 1.0,
                  noise sd 0.2, text `batch` column
  batch_numeric   the same with `batch` = 1..6
  batch_none      the same with no batch column (the structure is invisible)
  campaign_rounds the same, the batch passed as the campaign round index
  jitter_0.2pct   15 recipes x 4 replicates recorded as run (settings jitter
                  sd 0.002 of the range), recipe effect sd 1.0
  jitter_1pct     the same with jitter sd 0.01
  noise_n12/n20/n40  pure noise, 3 inputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd


def _batches(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, bs = [], [], []
    for b in range(6):
        c = rng.uniform(0.15, 0.85, 3)
        xs.append(np.clip(c + rng.normal(0, 0.12, (8, 3)), 0, 1))
        ys.append(rng.normal(0, 1.0) + rng.normal(0, 0.2, 8))
        bs += [b] * 8
    return np.vstack(xs), np.concatenate(ys), np.array(bs)


def make_sheet(name: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["feed_a", "feed_b", "temp_offset"]
    if name.startswith("batch") or name == "campaign_rounds":
        x, y, b = _batches(rng)
        df = pd.DataFrame(x.round(5), columns=cols)
        if name == "batch_text":
            df["batch"] = [f"B{i + 1}" for i in b]
        elif name == "batch_numeric":
            df["batch"] = b + 1
        elif name == "campaign_rounds":
            df.attrs["kalos_round_index"] = b.tolist()
        df["titer"] = y.round(5)
        return df
    if name.startswith("jitter"):
        sd = 0.002 if name == "jitter_0.2pct" else 0.01
        base = rng.uniform(0, 1, (15, 3))
        x = np.repeat(base, 4, axis=0) + rng.normal(0, sd, (60, 3))
        y = np.repeat(rng.normal(0, 1.0, 15), 4) + rng.normal(0, 0.2, 60)
        df = pd.DataFrame(x.round(6), columns=cols)
        df["titer"] = y.round(5)
        return df
    if name.startswith("noise_n"):
        n = int(name.split("_n")[1])
        df = pd.DataFrame(rng.uniform(0, 1, (n, 3)).round(5), columns=cols)
        df["titer"] = rng.normal(5, 1, n).round(5)
        return df
    raise KeyError(name)


SCENARIOS = ["batch_text", "batch_numeric", "batch_none", "campaign_rounds", "jitter_0.2pct", "jitter_1pct",
             "noise_n12", "noise_n20", "noise_n40"]


def run_one(name: str, seed: int) -> dict[str, Any]:
    import torch

    from kalos.portal.analysis import _analyze
    from kalos.portal.uploads import UploadRejected

    torch.set_num_threads(1)
    df = make_sheet(name, seed)
    try:
        r = _analyze(df, target="titer")
    except UploadRejected as exc:
        return {"scenario": name, "seed": seed, "refused_upload": getattr(exc, "code", "rejected")}
    verdict = r.get("verdict") or {}
    return {
        "scenario": name, "seed": seed,
        "cv": r["cv_spearman"], "ci95": r["cv_ci95"],
        "clears_floor": bool(r["reliability"]["clears_floor"]),
        "verdict": verdict.get("state"),
        "n_proposals": len(r["proposals"]),
        "trust_status": (r["proposals"][0].get("trust") or {}).get("status") if r["proposals"] else None,
        "batch_in_recipe": any("batch" in p["recipe"] for p in r["proposals"]),
        "cv_grouping_used": (r.get("cv_grouping") or {}).get("used"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in SCENARIOS:
        rs = [r for r in rows if r["scenario"] == name]
        if not rs:
            continue
        ok = [r for r in rs if "refused_upload" not in r]
        cvs = [r["cv"] for r in ok if r["cv"] is not None]
        out[name] = {
            "n": len(rs),
            "false_trust_clears_floor": round(float(np.mean([r["clears_floor"] for r in ok])), 3) if ok else None,
            "false_trust_verdict_reliable": (round(float(np.mean([r["verdict"] == "reliable" for r in ok])), 3)
                                             if ok and any(r["verdict"] for r in ok) else None),
            "proposals_withheld": round(float(np.mean([r["n_proposals"] == 0 for r in ok])), 3) if ok else None,
            "batch_proposed_as_setpoint": round(float(np.mean([r["batch_in_recipe"] for r in ok])), 3) if ok else None,
            "median_cv_spearman": round(float(np.median(cvs)), 3) if cvs else None,
            "upload_refused": int(len(rs) - len(ok)),
        }
    return out


def _worker(args: tuple[str, int]) -> dict[str, Any]:
    return run_one(*args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(nm, s) for nm in args.scenarios.split(",") for s in range(args.seeds)]
    if args.procs > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.procs) as pool:
            rows = pool.map(_worker, jobs, chunksize=1)
    else:
        rows = [_worker(j) for j in jobs]
    summary = summarize(rows)
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": {"seeds": args.seeds}, "summary": summary, "rows": rows}, fh)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
