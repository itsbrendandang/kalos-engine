"""CLI: `python -m kalos.bench.faults {make,run,score,all,gate}`.

    make   build the blinded suite (cases/, evaluator/truth.json, manifest.json)
    run    observe every case through the engine (blind: reads cases/ only)
    score  join observations with the truth key; write scorecard_<tier>.{json,md}
    all    make + run + score
    gate   score, then compare with a stored baseline (exit 1 on a regression)

Examples:
    python -m kalos.bench.faults all --tier fast --out /tmp/faults
    python -m kalos.bench.faults all --tier full --workers 4 --out /tmp/faults
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cases import build_suite
from .runner import run_suite
from .score import gate_view, regressions, score_suite, write_scorecard

BASELINE_FAST = Path(__file__).with_name("results") / "baseline_fast_gate.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kalos.bench.faults", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["make", "run", "score", "all", "gate"])
    ap.add_argument("--out", type=Path, default=Path("faults-suite"))
    ap.add_argument("--tier", choices=["fast", "full"], default="fast")
    ap.add_argument("--datasets", type=Path, default=None,
                    help="folder holding the optional Kaggle sheet (default ~/kalos/datasets)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None, help="case folder names to run (debugging)")
    ap.add_argument("--baseline", type=Path, default=BASELINE_FAST)
    args = ap.parse_args(argv)

    if args.command in ("make", "all"):
        info = build_suite(args.out, datasets_dir=args.datasets)
        print(f"built {info.n_cases} case(s) under {info.out} ({info.n_skipped} skipped)")
    if args.command in ("run", "all"):
        run_suite(args.out, args.tier, workers=args.workers, only=args.only,
                  log=lambda m: print(m, flush=True))
    if args.command in ("score", "all", "gate"):
        card = score_suite(args.out, args.tier)
        jpath, mpath = write_scorecard(args.out, args.tier, card)
        t = card["totals"]
        print(f"{t['caught']} caught, {t['noticed']} noticed, {t['missed']} missed of "
              f"{t['faults_scored']}; false alarms on {t['clean_false_alarm']}/{t['clean_scored']} clean")
        print(f"wrote {jpath} and {mpath}")
        if args.command == "gate":
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            problems = regressions(baseline, gate_view(card))
            for p in problems:
                print(f"REGRESSION {p}", file=sys.stderr)
            return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
