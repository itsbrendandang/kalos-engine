"""Run several arms on the TUNING split in ONE worker pool (ml-acquire screening).

Same job function, seeds and output layout as `python -m kalos.bench.ml_v2 run`
(`harness._worker`, `harness.job_file`), so `report` and `compare` read the
output directories unchanged. Refuses the held-out split: held-out runs go
through the harness CLI and its freeze guard.

    python benchmarks/ml_v2/acq/multirun.py --arms arms.json --out DIR --suites closed_loop \
        --tasks bumps hartmann6 --workers 8

`arms.json` = [{"spec": "...:make_arm", "config": {...}}, ...]; each arm writes
DIR/<arm name>/ with a run_meta.json like the harness's.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from kalos.bench.ml_v2.adapter import config_hash, load_arm
from kalos.bench.ml_v2.harness import (
    _COST,
    _prefit_emulators,
    _stack,
    _worker,
    git_state,
    job_file,
    seeds_for,
)
from kalos.bench.ml_v2.protocol import PROTOCOL_VERSION
from kalos.bench.ml_v2.suites import SUITES, tasks_for


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="tuning", choices=["tuning", "smoke"])
    ap.add_argument("--suites", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, help="subset of the split's seeds (never other seeds)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    arms = json.loads(a.arms.read_text())
    jobs = []
    cache = a.out / "_cache"
    for spec_cfg in arms:
        spec, cfg = spec_cfg["spec"], spec_cfg.get("config", {})
        arm = load_arm(spec, cfg)
        d = a.out / arm.name
        d.mkdir(parents=True, exist_ok=True)
        meta = {"protocol": PROTOCOL_VERSION, "arm": arm.name, "spec": spec, "config": cfg,
                "config_hash": config_hash(spec, cfg), "shippable": bool(arm.shippable), "split": a.split,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stack": _stack(), "git": git_state(),
                "suites_requested": a.suites, "tasks_requested": a.tasks, "runner": "acq/multirun.py"}
        mp_ = d / "run_meta.json"
        if mp_.is_file():
            old = json.loads(mp_.read_text())
            meta["previous_runs"] = [*old.pop("previous_runs", []), old]
        mp_.write_text(json.dumps(meta, indent=1, default=str) + "\n")
        for suite in a.suites or list(SUITES):
            if suite == "internal":
                continue
            for task in tasks_for(suite, a.split):
                if a.tasks and task not in a.tasks and task.split(":")[-1] not in a.tasks:
                    continue
                for seed in seeds_for(suite, a.split):
                    if a.seeds is not None and seed not in a.seeds:
                        continue
                    f = job_file(d, suite, task, seed)
                    if f.is_file():
                        continue
                    f.parent.mkdir(parents=True, exist_ok=True)
                    jobs.append((spec, cfg, suite, task, seed, a.split, str(cache), None, None, str(f)))
    _prefit_emulators(jobs, cache, None, print)
    jobs.sort(key=lambda j: (_COST.get(j[2], 9), j[4], j[3]))
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(_worker, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            j = futs[fut]
            try:
                fut.result()
                print(f"[{k}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {Path(j[-1]).parts[-5]} {j[2]} {j[3]} s{j[4]}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[{k}/{len(jobs)}] FAILED {j[-1]}: {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
