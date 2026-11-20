#!/usr/bin/env python3
"""Golden oracle (reorg guard R02): record and check the engine's output.

    python scripts/golden.py check                       # compare with tests/golden/<env key>/
    python scripts/golden.py check --require --actual D  # CI: missing goldens fail; keep the actual output in D
    python scripts/golden.py diff                        # show every changed leaf
    python scripts/golden.py record --reason "F04: ..."  # (re)record for THIS platform and stack
    python scripts/golden.py record --reason "..." --from D   # adopt a CI run's output (artifact `golden-actual`)

The scenario (tests/golden/oracle.py) runs in a FRESH interpreter (one torch
thread, a temporary state directory, deterministic ids) and its records are
compared byte for byte with `tests/golden/<env key>/`. A behaviour-preserving
change (every reorg move) must leave them identical. A change that is meant to
move numbers or wording records new goldens with `--reason`, and the reason is
appended to the manifest's change log, so the diff and the why ship together.

Records are valid only for the platform and pinned stack in the env key
(decision D7: byte-exact, no tolerance). CI's linux key gates merges
(.github/workflows/golden.yml); a developer machine may keep its own key for a
fast local check. To refresh CI's records: run the `golden` workflow (it runs
on every PR), download its `golden-actual` artifact, and `record --from` it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
MANIFEST = "MANIFEST.json"


def _oracle() -> Any:
    """tests/golden/oracle.py, loaded by path (tests/ is not a package)."""
    import importlib.util

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if "kalos_golden_oracle" in sys.modules:
        return sys.modules["kalos_golden_oracle"]
    spec = importlib.util.spec_from_file_location("kalos_golden_oracle", GOLDEN_DIR / "oracle.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kalos_golden_oracle"] = module
    spec.loader.exec_module(module)
    return module


def compute(out: Path) -> None:
    """Run the scenario in a fresh interpreter and write one file per case."""
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "KALOS_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_compute", str(out)], cwd=ROOT,  # noqa: S603 - this script, by path
                          env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"the golden scenario failed (exit {proc.returncode})")


def _compute_here(out: Path) -> None:
    oracle = _oracle()
    records = oracle.run_scenario()
    cases: dict[str, str] = {}
    import hashlib

    for name, text in oracle.iter_cases(records):
        path = out / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        cases[name] = hashlib.sha256(text.encode()).hexdigest()
    manifest = {"env": oracle.env_details(), "reasons": [], "cases": cases}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def _load(dir_: Path) -> dict[str, Any]:
    return dict(json.loads((dir_ / MANIFEST).read_text(encoding="utf-8")))


def _leaf_diff(old: Any, new: Any, path: str = "", out: list[str] | None = None, limit: int = 40) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(old, dict) and isinstance(new, dict):
        if list(old) != list(new) and set(old) == set(new):
            out.append(f"{path or '/'}: key ORDER changed")
        for k in list(old) + [k for k in new if k not in old]:
            if k not in new:
                out.append(f"{path}/{k}: removed")
            elif k not in old:
                out.append(f"{path}/{k}: added")
            else:
                _leaf_diff(old[k], new[k], f"{path}/{k}", out, limit)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            out.append(f"{path}: length {len(old)} -> {len(new)}")
        for i, (a, b) in enumerate(zip(old, new)):
            _leaf_diff(a, b, f"{path}[{i}]", out, limit)
    elif old != new or type(old) is not type(new):
        out.append(f"{path}: {json.dumps(old)[:80]} -> {json.dumps(new)[:80]}")
    return out


def compare(expected: Path, actual: Path, *, verbose: bool, summary: dict[str, Any] | None = None) -> int:
    exp, act = _load(expected), _load(actual)
    if exp["env"]["key"] != act["env"]["key"]:
        print(f"env key differs: recorded {exp['env']['key']}, current {act['env']['key']}")
        return 1
    names = sorted(set(exp["cases"]) | set(act["cases"]))
    bad = 0
    summary = {} if summary is None else summary
    summary.update({"changed": [], "new": [], "missing": []})
    for name in names:
        if name not in act["cases"]:
            print(f"MISSING  {name} (the scenario no longer produces it)")
            summary["missing"].append(name)
            bad += 1
        elif name not in exp["cases"]:
            print(f"NEW      {name} (not recorded)")
            summary["new"].append(name)
            bad += 1
        elif exp["cases"][name] != act["cases"][name]:
            bad += 1
            summary["changed"].append(name)
            print(f"CHANGED  {name}")
            if verbose:
                old = json.loads((expected / f"{name}.json").read_text(encoding="utf-8"))
                new = json.loads((actual / f"{name}.json").read_text(encoding="utf-8"))
                for line in _leaf_diff(old, new):
                    print(f"    {line}")
    print(f"golden oracle: {len(names) - bad}/{len(names)} cases byte-identical ({exp['env']['key']})")
    summary.update({"total": len(names), "identical": len(names) - bad})
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "diff"):
        p = sub.add_parser(name)
        p.add_argument("--require", action="store_true", help="fail when no goldens exist for this env")
        p.add_argument("--actual", type=Path, help="keep the computed output here")
        p.add_argument("--summary-json", type=Path, help="also write {env_key, total, identical, changed, new, "
                       "missing, recorded_for_env} here (for the scoreboard)")
    rec = sub.add_parser("record")
    rec.add_argument("--reason", required=True, help="why the goldens change (kept in the manifest)")
    rec.add_argument("--from", dest="src", type=Path, help="adopt this computed output (a CI artifact)")
    internal = sub.add_parser("_compute")
    internal.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "_compute":
        _compute_here(args.out)
        return 0

    if args.cmd == "record":
        if not args.reason.strip():
            parser.error("--reason must say why the goldens change")
        if args.src is not None:
            src = args.src
        else:
            src = Path(tempfile.mkdtemp(prefix="kalos-golden-out-"))
            compute(src)
        manifest = _load(src)
        dest = GOLDEN_DIR / manifest["env"]["key"]
        reasons = _load(dest)["reasons"] if (dest / MANIFEST).exists() else []
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        manifest["reasons"] = [*reasons, args.reason.strip()]
        (dest / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"recorded {len(manifest['cases'])} cases in {dest.relative_to(ROOT)}")
        return 0

    actual = args.actual or Path(tempfile.mkdtemp(prefix="kalos-golden-out-"))
    if actual.exists():
        shutil.rmtree(actual)
    compute(actual)
    key = _load(actual)["env"]["key"]
    expected = GOLDEN_DIR / key
    summary: dict[str, Any] = {"env_key": key, "recorded_for_env": (expected / MANIFEST).exists()}
    if not summary["recorded_for_env"]:
        recorded = sorted(p.name for p in GOLDEN_DIR.iterdir() if (p / MANIFEST).exists())
        print(f"no goldens recorded for {key} (recorded: {recorded}). The computed output is in {actual};"
              " adopt it with: scripts/golden.py record --reason '...' --from <that dir>")
        code = 1 if args.require else 0
    else:
        code = compare(expected, actual, verbose=args.cmd == "diff" or args.actual is not None, summary=summary)
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
