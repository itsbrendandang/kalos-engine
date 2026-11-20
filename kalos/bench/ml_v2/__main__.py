"""CLI for the ML benchmark v2 (benchmarks/ml_v2/PROTOCOL.md, README.md).

    python -m kalos.bench.ml_v2 fetch                       # download open datasets (Zenodo, Olympus)
    python -m kalos.bench.ml_v2 datasets                    # what is available, with licences
    python -m kalos.bench.ml_v2 run --arm SPEC --split tuning --out DIR [--workers 8]
    python -m kalos.bench.ml_v2 report DIR                  # summary.json + summary.md for one arm
    python -m kalos.bench.ml_v2 compare CAND_DIR REF_DIR [--baseline DIR ...] [--faults-cand F --faults-ref F]
    python -m kalos.bench.ml_v2 faults --arm SPEC --out DIR # planted-fault suite with the arm's engine patch
    python -m kalos.bench.ml_v2 freeze --arm SPEC [--arm-config JSON]  # write the freeze file to commit

The reference arm is `kalos.bench.ml_v2.arms.kalos_ref:make_arm`; baselines are
`kalos.bench.ml_v2.arms.baselines:make_random` and `:make_lhs`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kalos.bench.ml_v2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    p_ds = sub.add_parser("datasets")
    p_ds.add_argument("--user-data", type=Path)
    p_run = sub.add_parser("run")
    p_run.add_argument("--arm", required=True)
    p_run.add_argument("--arm-config", default="{}")
    p_run.add_argument("--split", choices=["tuning", "heldout", "smoke", "ci"], default="tuning")
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--suites", nargs="*")
    p_run.add_argument("--tasks", nargs="*")
    p_run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p_run.add_argument("--user-data", type=Path)
    p_run.add_argument("--leadgene", type=Path, help="path to leadgene_cho_dynamics.csv (or KALOS_EXTERNAL_DATA)")
    p_rep = sub.add_parser("report")
    p_rep.add_argument("dir", type=Path)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("cand", type=Path)
    p_cmp.add_argument("ref", type=Path)
    p_cmp.add_argument("--baseline", type=Path, nargs="*", default=[])
    p_cmp.add_argument("--faults-cand", type=Path)
    p_cmp.add_argument("--faults-ref", type=Path)
    p_cmp.add_argument("--out", type=Path)
    p_f = sub.add_parser("faults")
    p_f.add_argument("--arm", required=True)
    p_f.add_argument("--arm-config", default="{}")
    p_f.add_argument("--out", type=Path, required=True)
    p_f.add_argument("--tier", choices=["fast", "full"], default="full")
    p_f.add_argument("--workers", type=int, default=4)
    p_f.add_argument("--datasets", type=Path, default=None)
    p_fr = sub.add_parser("freeze")
    p_fr.add_argument("--arm", required=True)
    p_fr.add_argument("--arm-config", default="{}")
    p_fr.add_argument("--note", default="")
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        from .datasets import fetch

        fetch()
        return 0
    if args.cmd == "datasets":
        from . import datasets as dsm
        from .leadgene import INTERNAL_LABEL, leadgene_path

        found, skipped = dsm.available(list(dsm.LOADERS))
        for d in found:
            print(json.dumps(d.info()))
        for k, v in skipped.items():
            print(json.dumps({"name": k, "skipped": v}))
        lp = leadgene_path()
        print(json.dumps({"name": "internal/leadgene", "available": bool(lp), "label": INTERNAL_LABEL}))
        if args.user_data:
            uf, us = dsm.load_user_folder(args.user_data)
            for d in uf:
                print(json.dumps(d.info()))
            for k, v in us.items():
                print(json.dumps({"name": k, "skipped": v}))
        return 0
    if args.cmd == "run":
        from .harness import run
        from .leadgene import leadgene_path

        run(args.arm, json.loads(args.arm_config), args.out, args.split, suites=args.suites, tasks=args.tasks,
            workers=args.workers, user_data=args.user_data, leadgene_path=leadgene_path(args.leadgene))
        return 0
    if args.cmd == "report":
        from .report import summarize_dir

        summarize_dir(args.dir)
        return 0
    if args.cmd == "compare":
        from .report import compare_dirs

        verdict = compare_dirs(args.cand, args.ref, args.baseline, args.faults_cand, args.faults_ref, args.out)
        return 0 if verdict.get("ships") else 1
    if args.cmd == "faults":
        from .faults_arm import run_faults

        run_faults(args.arm, json.loads(args.arm_config), args.out, args.tier, args.workers, args.datasets)
        return 0
    if args.cmd == "freeze":
        from .adapter import config_hash, load_arm
        from .harness import FREEZE_DIR, git_state

        cfg = json.loads(args.arm_config)
        arm = load_arm(args.arm, cfg)
        FREEZE_DIR.mkdir(parents=True, exist_ok=True)
        path = FREEZE_DIR / f"{arm.name}.json"
        if path.exists():
            print(f"{path} exists: a frozen arm is never re-frozen; use a new arm name", file=sys.stderr)
            return 1
        doc = {"arm": arm.name, "spec": args.arm, "config": cfg, "config_hash": config_hash(args.arm, cfg),
               "shippable": bool(arm.shippable), "frozen_from_commit": git_state()["commit"], "note": args.note}
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}: commit it, then run --split heldout")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
