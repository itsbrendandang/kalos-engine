"""Parallel, resumable job runner with the held-out guard (PROTOCOL.md section 4).

Each job writes `<out>/raw/<suite>/<task>/s<seed>.json`; existing files are
skipped (resume). `<out>/run_meta.json` records the arm spec, config, config
hash, git commit and dirty state, the protocol version, the split, the seeds and
the library versions.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .adapter import config_hash, load_arm
from .protocol import (
    CI_SEEDS,
    CI_TASKS,
    HELDOUT_SEEDS,
    PROTOCOL_VERSION,
    SMOKE_SEEDS,
    TUNING_SEEDS,
)
from .suites import SUITES, run_job, tasks_for

REPO = Path(__file__).resolve().parents[3]
FREEZE_DIR = REPO / "benchmarks" / "ml_v2" / "freeze"
LEDGER = REPO / "benchmarks" / "ml_v2" / "results" / "heldout_ledger.jsonl"
# Job order: cheap, single-analysis suites first so a run interrupted on a busy
# machine still has broad coverage; the loops follow, one task at a time across seeds.
_COST = {"snapshot": 0, "internal": 1, "scale": 2, "backtest": 3, "snapshot_real": 4, "closed_loop": 5,
         "pool": 6, "emulator": 7}


def seeds_for(suite: str, split: str) -> tuple[int, ...]:
    table = {"tuning": TUNING_SEEDS, "heldout": HELDOUT_SEEDS, "smoke": SMOKE_SEEDS, "ci": CI_SEEDS}[split]
    return table.get(suite, ())


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,  # noqa: S603,S607
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "dirty_files": status.splitlines()[:20]}


def guard_heldout(arm_name: str, spec: str, config: dict[str, Any]) -> dict[str, Any]:
    """Refuse a held-out run unless the arm is frozen in a COMMITTED freeze file
    whose spec and config hash match, and the tracked tree is clean."""
    g = git_state()
    if g["dirty"]:
        raise SystemExit("held-out run refused: the working tree has uncommitted changes to tracked files "
                         f"({g['dirty_files'][:5]}); commit the frozen arm first")
    path = FREEZE_DIR / f"{arm_name}.json"
    rel = path.relative_to(REPO).as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO,  # noqa: S603,S607
                             capture_output=True, text=True).returncode == 0
    if not path.is_file() or not tracked:
        raise SystemExit(f"held-out run refused: no committed freeze file {rel} (see PROTOCOL.md section 4)")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    want = config_hash(spec, config)
    if frozen.get("spec") != spec or frozen.get("config_hash") != want:
        raise SystemExit(f"held-out run refused: {rel} freezes spec={frozen.get('spec')} "
                         f"hash={frozen.get('config_hash')}, this run is spec={spec} hash={want}")
    return {**g, "freeze_file": rel, "frozen": frozen}


def _stack() -> dict[str, Any]:
    try:
        from kalos.versions import stack_record

        return dict(stack_record())
    except Exception:  # noqa: BLE001 - versions are informative here
        return {"python": platform.python_version()}


def _worker(args: tuple[str, dict[str, Any], str, str, int, str, str | None, str | None, str | None, str]) -> str:
    spec, config, suite, task, seed, split, cache, user, lead, out_file = args
    os.environ.setdefault("KALOS_TORCH_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    arm = load_arm(spec, config)
    try:
        rec = run_job(arm, suite, task, seed, split, Path(cache) if cache else None,
                      Path(user) if user else None, Path(lead) if lead else None)
    except FileNotFoundError as exc:
        rec = {"suite": suite, "task": task, "seed": seed, "split": split, "arm": arm.name,
               "skipped": f"{type(exc).__name__}: {str(exc)[:200]}"}
    tmp = Path(out_file + ".tmp")
    tmp.write_text(json.dumps(rec, default=str) + "\n", encoding="utf-8")
    tmp.replace(out_file)
    return out_file


def job_file(out: Path, suite: str, task: str, seed: int) -> Path:
    return out / "raw" / suite / task.replace("/", "__").replace(":", "--") / f"s{seed}.json"


def run(spec: str, config: dict[str, Any], out: Path, split: str, *, suites: list[str] | None = None,
        tasks: list[str] | None = None, workers: int = 4, user_data: Path | None = None,
        leadgene_path: Path | None = None, log: Callable[[str], None] | None = None) -> Path:
    if log is None:
        def log(msg: str) -> None:
            print(msg, flush=True)
    if split not in ("tuning", "heldout", "smoke", "ci"):
        raise ValueError("split must be tuning, heldout, smoke or ci")
    if split == "ci":
        suites, tasks = ["closed_loop"], list(CI_TASKS)
    arm = load_arm(spec, config)
    meta: dict[str, Any] = {"protocol": PROTOCOL_VERSION, "arm": arm.name, "spec": spec, "config": config,
                            "config_hash": config_hash(spec, config), "shippable": bool(arm.shippable),
                            "split": split, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "stack": _stack(), "workers": workers, "host": platform.node()}
    if split == "heldout":
        meta["guard"] = guard_heldout(arm.name, spec, config)
    else:
        meta["git"] = git_state()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "_cache"
    jobs = []
    from . import datasets as dsm

    user_tasks: list[str] = []
    if user_data is not None:
        found, skipped = dsm.load_user_folder(user_data)
        user_tasks = [d.name for d in found]
        meta["user_data"] = {"folder": str(user_data), "datasets": user_tasks, "skipped": skipped}
    meta["leadgene"] = str(leadgene_path) if leadgene_path else None
    for suite in suites or list(SUITES):
        if suite == "internal" and leadgene_path is None:
            log("internal suite skipped: no KALOS_EXTERNAL_DATA / --leadgene")
            continue
        names = tasks_for(suite, split)
        if suite == "pool":
            names += user_tasks
        elif suite == "snapshot_real":
            names += [f"{v}:{u}" for u in user_tasks for v in ("true", "perm")]
        elif suite == "emulator":
            names += [f"{k}:{u}" for u in user_tasks for k in ("gp", "rf")]
        for task in names:
            if tasks and task not in tasks and task.split(":")[-1] not in tasks:
                continue
            for seed in seeds_for(suite, split):
                f = job_file(out, suite, task, seed)
                if f.is_file():
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((spec, config, suite, task, seed, split, str(cache),
                             str(user_data) if user_data else None,
                             str(leadgene_path) if leadgene_path else None, str(f)))
    meta_path = out / "run_meta.json"
    if meta_path.is_file():  # a resumed or partial re-run keeps the earlier runs' metadata
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["previous_runs"] = [*old.pop("previous_runs", []), old]
    meta["suites_requested"], meta["tasks_requested"] = suites, tasks
    meta_path.write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    _prefit_emulators(jobs, cache, user_data, log)
    jobs.sort(key=lambda j: (_COST.get(j[2], 9), j[4], j[3]))
    log(f"{len(jobs)} job(s) to run with {workers} worker(s) -> {out}")
    t0 = time.perf_counter()
    if workers <= 1:
        for j in jobs:
            _worker(j)
            log(f"done {j[2]} {j[3]} s{j[4]}")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = {pool.submit(_worker, j): j for j in jobs}
            for k, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                try:
                    fut.result()
                    log(f"[{k}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {j[2]} {j[3]} s{j[4]}")
                except Exception as exc:  # noqa: BLE001 - one broken job must not stop the run
                    log(f"[{k}/{len(jobs)}] FAILED {j[2]} {j[3]} s{j[4]}: {type(exc).__name__}: {exc}")
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    if split == "heldout":
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({k: meta.get(k) for k in ("arm", "spec", "config_hash", "started", "finished")}
                                | {"commit": meta["guard"]["commit"], "out": str(out)}) + "\n")
    return out


def _prefit_emulators(jobs: list[tuple], cache: Path, user_data: Path | None, log: Callable[[str], None]) -> None:
    """Fit every emulator once, in this process, before the workers start."""
    from .emulator import get_emulator
    from .suites import _load_ds, _prep_dataset

    seen = sorted({j[3] for j in jobs if j[2] == "emulator"})
    for task in seen:
        kind, dname = task.split(":", 1)
        try:
            ds = _prep_dataset(_load_ds(dname, user_data))
        except (FileNotFoundError, OSError, KeyError) as exc:
            log(f"emulator {task}: dataset unavailable ({exc}); its jobs will be skipped")
            continue
        t0 = time.perf_counter()
        em = get_emulator(ds, kind, cache)
        log(f"emulator {task}: f*={em.f_star:.4g} f_med={em.f_med:.4g} noise_sd={em.noise_sd:.3g} "
            f"({time.perf_counter() - t0:.0f}s)")
