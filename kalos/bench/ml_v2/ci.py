"""Stable entry point for a per-PR scoreboard (release-captain's `make scoreboard`).

    from kalos.bench.ml_v2.ci import ci_regret
    summary = ci_regret(Path("/tmp/ci-regret"), workers=4)

Runs the closed-loop suite on `protocol.CI_TASKS` x `protocol.CI_SEEDS` (a seed
set disjoint from tuning and held-out). A drift signal for CI, never a decision:
four jobs cannot resolve small regret differences.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import CI_SEEDS, CI_TASKS, PROTOCOL_VERSION

REFERENCE_ARM = "kalos.bench.ml_v2.arms.kalos_ref:make_arm"


def ci_regret(out_dir: Path, workers: int = 4, arm: str = REFERENCE_ARM) -> dict[str, Any]:
    from .harness import job_file, run

    out_dir = Path(out_dir)
    run(arm, {}, out_dir, "ci", workers=workers, log=lambda _m: None)
    per_job = []
    for task in CI_TASKS:
        for seed in CI_SEEDS["closed_loop"]:
            f = job_file(out_dir, "closed_loop", task, seed)
            rec = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
            last = (rec.get("rounds") or [{}])[-1]
            per_job.append({"task": task, "seed": seed, "n0": rec.get("n0"),
                            "regret_obs_final": last.get("regret_obs"), "regret_best_final": last.get("regret_best"),
                            "wall_s": sum(r.get("wall_s") or 0 for r in rec.get("rounds") or [])})

    def mean(k: str) -> float | None:
        v = [j[k] for j in per_job if j[k] is not None]
        return sum(v) / len(v) if v else None

    return {"protocol": PROTOCOL_VERSION, "arm": arm, "tasks": list(CI_TASKS), "seeds": list(CI_SEEDS["closed_loop"]),
            "per_job": per_job, "mean_regret_obs": mean("regret_obs_final"),
            "mean_regret_best": mean("regret_best_final")}
