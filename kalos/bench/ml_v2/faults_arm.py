"""The planted-fault suite (kalos/bench/faults, its own pre-registered PROTOCOL.md)
run with an arm's ENGINE PATCH applied in every worker (criterion C5).

    python -m kalos.bench.ml_v2 faults --arm SPEC --out DIR [--tier full]

The suite, catalogue, truth key and scorer are used unchanged; only the engine
the runner observes is patched. Compare two scorecards with
`kalos.bench.faults.score.regressions(reference_view, candidate_view)`
(done by `compare --faults-cand/--faults-ref`).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _observe(args: tuple[str, dict[str, Any], str, str, str]) -> str:
    spec, config, folder, tier, out_file = args
    os.environ.setdefault("KALOS_TORCH_THREADS", "1")
    from kalos.bench.faults.runner import observe_case

    from .adapter import load_arm

    arm = load_arm(spec, config)
    with arm.engine_patch():
        obs = observe_case(Path(folder), tier)
    Path(out_file).write_text(json.dumps(obs, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out_file


def run_faults(spec: str, config: dict[str, Any], out: Path, tier: str = "full", workers: int = 4,
               datasets: Path | None = None) -> Path:
    from kalos.bench.faults.cases import CASES_DIR, build_suite
    from kalos.bench.faults.runner import OBS_DIR
    from kalos.bench.faults.score import score_suite, write_scorecard

    from .adapter import load_arm

    arm = load_arm(spec, config)
    if not arm.shippable:
        print(f"note: {arm.name} is standalone; its engine patch is a no-op, so this scores the unpatched engine")
    out = Path(out)
    build_suite(out, datasets_dir=datasets)
    obs_dir = out / OBS_DIR / tier
    obs_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(spec, config, str(p), tier, str(obs_dir / f"{p.name}.json"))
            for p in sorted((out / CASES_DIR).iterdir()) if p.is_dir()]
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=mp.get_context("spawn")) as pool:
        for done in pool.map(_observe, jobs):
            print(f"{Path(done).stem} done", flush=True)
    card = score_suite(out, tier)
    card["arm"] = {"name": arm.name, "spec": spec, "config": config}
    jpath, _mpath = write_scorecard(out, tier, card)
    t = card["totals"]
    print(f"{arm.name}: {t['caught']} caught, {t['noticed']} noticed, {t['missed']} missed of {t['faults_scored']}; "
          f"false alarms on {t['clean_false_alarm']}/{t['clean_scored']} clean -> {jpath}")
    return jpath
