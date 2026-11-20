"""Serial vs pooled analyses: full response JSON and wall time (docs/PERFORMANCE.md).

    # one worker setting per process; writes {results, seconds, stats}
    python scripts/parallel_identity.py run --workers 1    --out serial.json
    python scripts/parallel_identity.py run --workers auto --out pooled.json
    python scripts/parallel_identity.py compare serial.json pooled.json

`--sets` picks the sheets: `trust` (the 8 trust-golden sheets), `strain` (the 5
strain-golden sheets), `examples` (examples/synthetic_bioprocess, 160 rows, and
examples/synthetic_scaleup, through `compute_upload_analysis`, the /api/run path).
`--seed` overrides the analysis seed (`ANALYZE_SEED`). `--threads` sets torch's
thread count (the pool runs only at 1). Every key is compared except wall clock
(`timestamp`, `fit_seconds`, `created_at`).

To time another checkout (e.g. `git archive origin/main | tar -x -C /tmp/base`), set
`KALOS_ROOT=/tmp/base`; that code is imported instead of this one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

VOLATILE = {"timestamp", "fit_seconds", "created_at"}
EXAMPLES = (
    ("synthetic_bioprocess", "examples/synthetic_bioprocess/synthetic_bioprocess.csv", "titer_g_L"),
    ("synthetic_scaleup", "examples/synthetic_scaleup/synthetic_scaleup.csv", "titer_g_per_L"),
)


def _strip(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(args: argparse.Namespace) -> int:
    os.environ["KALOS_ANALYSIS_WORKERS"] = args.workers
    os.environ["KALOS_TORCH_THREADS"] = args.threads
    root = Path(os.environ.get("KALOS_ROOT", Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(root))

    import torch

    torch.set_num_threads(int(args.threads))
    import kalos.portal.analysis as analysis
    from kalos.portal.pipeline import compute_upload_analysis

    if args.seed is not None:
        analysis.ANALYZE_SEED = args.seed
    jobs: list[tuple[str, Any]] = []
    sets = args.sets.split(",")
    if "trust" in sets:
        tg = _load(root / "scripts" / "trust_golden.py", "trust_golden")
        for name, make in tg.SHEETS.items():
            def _t(make: Any = make) -> dict:
                df, kw = make()
                kw = dict(kw)
                target = kw.pop("target")
                return analysis._analyze(df, target=target, **kw)
            jobs.append((f"trust/{name}", _t))
    if "strain" in sets:
        sg = _load(root / "tests" / "test_strain_golden_no_clone.py", "strain_golden")
        for name, make in sg.DATASETS.items():
            def _s(make: Any = make) -> dict:
                df, target, kw = make()
                return analysis._analyze(df, target, **kw)
            jobs.append((f"strain/{name}", _s))
    if "examples" in sets:
        for name, rel, target in EXAMPLES:
            raw = (root / rel).read_bytes()
            def _e(raw: bytes = raw, target: str = target, rel: str = rel) -> dict:
                return compute_upload_analysis(raw, target, False, Path(rel).name)[1]
            jobs.append((f"examples/{name}", _e))
    try:
        from kalos.core import parallel
    except ImportError:  # a checkout from before the pool
        parallel = None  # type: ignore[assignment]
    out: dict[str, Any] = {"workers": args.workers, "threads": args.threads, "seed": args.seed,
                           "active_workers": parallel.active_workers() if parallel else 1,
                           "results": {}, "seconds": {}}
    if parallel:
        parallel.reset_stats()
    for name, fn in jobs:
        t0 = time.perf_counter()
        res = fn()
        out["seconds"][name] = round(time.perf_counter() - t0, 2)
        out["results"][name] = json.loads(json.dumps(_strip(res), default=str, sort_keys=True))
        print(f"{name}: {out['seconds'][name]:.1f}s", flush=True)
    out["stats"] = dict(parallel.stats) if parallel else None
    Path(args.out).write_text(json.dumps(out, sort_keys=True))
    return 0


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    ok = True
    for k in sorted(set(a["results"]) | set(b["results"])):
        x = json.dumps(a["results"].get(k), sort_keys=True)
        y = json.dumps(b["results"].get(k), sort_keys=True)
        same = x == y
        ok &= same
        print(f"{'IDENTICAL' if same else 'DIFFERENT'} {k} ({len(x)} bytes) "
              f"{a['seconds'].get(k)}s vs {b['seconds'].get(k)}s")
    print(f"workers {a.get('active_workers')} vs {b.get('active_workers')}; pool stats {b.get('stats')}")
    print("ALL IDENTICAL" if ok else "SOME DIFFER")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", required=True, help="KALOS_ANALYSIS_WORKERS: 1, N or auto")
    r.add_argument("--threads", default="1")
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--sets", default="trust,strain,examples")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(args)
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
