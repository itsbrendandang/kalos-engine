#!/usr/bin/env python3
"""Golden oracle (reorg guard R02): record and check the engine's output.

Two checks, both byte-exact:

    python scripts/golden.py differential --base-tree DIR   # CI gate: this tree vs DIR, same machine
    python scripts/golden.py check                          # this tree vs tests/golden/<env key>/

`differential` is the merge gate (.github/workflows/golden.yml). It runs the
same scenario (this tree's harness) on the base commit and on the PR head, on
the SAME runner, and compares the outputs byte for byte. CI runners are not
all the same CPU, and float results move with the CPU even with every BLAS and
torch kernel pinned (measured: AMD EPYC 9V74, 9V45 and Intel Xeon 8573C gave
three different sets of proposals for identical code), so committed linux
records cannot gate merges; a same-machine comparison can. A difference fails
the gate unless the PR DECLARES it: it edits tests/golden/ (a new line in
tests/golden/CHANGES.md, written by `record --reason`). The diff is printed and
uploaded as an artifact either way.

`check` compares with committed records for THIS machine's platform and pinned
stack (tests/golden/<env key>/), a fast local check before pushing a move:

    python scripts/golden.py check                       # compare
    python scripts/golden.py diff                        # show every changed leaf
    python scripts/golden.py record --reason "F04: ..."  # (re)record this env and log the reason

The scenario (tests/golden/oracle.py) always runs in a fresh interpreter: one
torch thread, a temporary state directory, deterministic ids.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
CHANGES = GOLDEN_DIR / "CHANGES.md"
MANIFEST = "MANIFEST.json"
# The harness files copied onto the base tree, so both sides run the same scenario.
HARNESS = ("scripts/golden.py", "tests/golden/oracle.py", "tests/golden/assist_docs")

# Pin the BLAS/LAPACK and torch kernel paths to one every x86-64 CPU has (AVX2):
# this removes the run-to-run differences on one runner (memory alignment, MKL
# dispatch), though not the cross-CPU ones (see the module docstring). arm64
# ignores them. Applied to the oracle's process only; the engine is unchanged.
KERNEL_PINS = {
    "MKL_CBWR": "AVX2,STRICT",
    "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "OPENBLAS_CORETYPE": "Haswell",
    "ATEN_CPU_CAPABILITY": "avx2",
} if platform.machine().lower() in ("x86_64", "amd64") else {}


def _oracle(root: Path = ROOT) -> Any:
    """tests/golden/oracle.py, loaded by path (tests/ is not a package)."""
    import importlib.util

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "kalos_golden_oracle" in sys.modules:
        return sys.modules["kalos_golden_oracle"]
    spec = importlib.util.spec_from_file_location("kalos_golden_oracle", root / "tests" / "golden" / "oracle.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kalos_golden_oracle"] = module
    spec.loader.exec_module(module)
    return module


def compute(out: Path, root: Path = ROOT) -> None:
    """Run the scenario on the engine in `root`, in a fresh interpreter."""
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONHASHSEED": "0", "KALOS_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(root), **KERNEL_PINS}
    script = root / "scripts" / "golden.py"
    proc = subprocess.run([sys.executable, str(script), "_compute", str(out)], cwd=root,  # noqa: S603 - fixed argv
                          env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"the golden scenario failed on {root} (exit {proc.returncode})")


def _compute_here(out: Path) -> None:
    import hashlib

    import kalos

    oracle = _oracle()
    engine_root = Path(kalos.__file__).resolve().parents[1]
    if engine_root != ROOT:
        raise SystemExit(f"imported kalos from {engine_root}, expected {ROOT}")
    records = oracle.run_scenario()
    cases: dict[str, str] = {}
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


def compare(expected: Path, actual: Path, *, verbose: bool, summary: dict[str, Any] | None = None,
            label: str = "") -> int:
    exp, act = _load(expected), _load(actual)
    if exp["env"]["key"] != act["env"]["key"]:
        print(f"env key differs: {exp['env']['key']} vs {act['env']['key']}")
        return 1
    names = sorted(set(exp["cases"]) | set(act["cases"]))
    summary = {} if summary is None else summary
    summary.update({"changed": [], "new": [], "missing": []})
    for name in names:
        if name not in act["cases"]:
            print(f"MISSING  {name}")
            summary["missing"].append(name)
        elif name not in exp["cases"]:
            print(f"NEW      {name}")
            summary["new"].append(name)
        elif exp["cases"][name] != act["cases"][name]:
            summary["changed"].append(name)
            print(f"CHANGED  {name}")
            if verbose:
                old = json.loads((expected / f"{name}.json").read_text(encoding="utf-8"))
                new = json.loads((actual / f"{name}.json").read_text(encoding="utf-8"))
                for line in _leaf_diff(old, new):
                    print(f"    {line}")
    bad = len(summary["changed"]) + len(summary["new"]) + len(summary["missing"])
    print(f"golden oracle{label}: {len(names) - bad}/{len(names)} cases byte-identical ({act['env']['key']})")
    summary.update({"total": len(names), "identical": len(names) - bad})
    return 1 if bad else 0


def _log_change(reason: str, key: str) -> None:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    head = CHANGES.read_text(encoding="utf-8") if CHANGES.exists() else (
        "# Golden output changes\n\nOne line per intended change to the engine's output (scripts/golden.py "
        "record --reason). A PR whose output differs from its base must add a line here; the CI gate "
        "(golden.yml) fails an undeclared difference.\n\n")
    CHANGES.write_text(head.rstrip("\n") + f"\n- {stamp} ({key}): {reason}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "diff"):
        p = sub.add_parser(name)
        p.add_argument("--require", action="store_true", help="fail when no records exist for this env")
        p.add_argument("--actual", type=Path, help="keep the computed output here")
        p.add_argument("--summary-json", type=Path, help="also write {env_key, recorded_for_env, total, "
                       "identical, changed, new, missing} here (for the scoreboard)")
    dif = sub.add_parser("differential")
    dif.add_argument("--base-tree", type=Path, required=True, help="a checkout of the base commit")
    dif.add_argument("--actual", type=Path, required=True, help="outputs go to <actual>/{base,head}")
    dif.add_argument("--declared", action="store_true", help="the PR declares an output change (edits tests/golden/)")
    dif.add_argument("--summary-json", type=Path)
    rec = sub.add_parser("record")
    rec.add_argument("--reason", required=True, help="why the output changes (logged in tests/golden/CHANGES.md)")
    rec.add_argument("--from", dest="src", type=Path, help="adopt this computed output instead of computing")
    rec.add_argument("--log-only", action="store_true",
                     help="only log the reason (no records for this machine's stack)")
    internal = sub.add_parser("_compute")
    internal.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "_compute":
        _compute_here(args.out)
        return 0

    if args.cmd == "record":
        reason = args.reason.strip()
        if not reason:
            parser.error("--reason must say why the output changes")
        if args.log_only:
            _log_change(reason, "no local record")
            return 0
        src = args.src if args.src is not None else Path(tempfile.mkdtemp(prefix="kalos-golden-out-"))
        if args.src is None:
            compute(src)
        manifest = _load(src)
        dest = GOLDEN_DIR / manifest["env"]["key"]
        reasons = _load(dest)["reasons"] if (dest / MANIFEST).exists() else []
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        manifest["reasons"] = [*reasons, reason]
        (dest / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        _log_change(reason, manifest["env"]["key"])
        print(f"recorded {len(manifest['cases'])} cases in {dest.relative_to(ROOT)}")
        return 0

    if args.cmd == "differential":
        base = args.base_tree.resolve()
        for rel in HARNESS:  # the same scenario on both sides
            src, dst = ROOT / rel, base / rel
            if dst.is_dir():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        out_base, out_head = args.actual / "base", args.actual / "head"
        for d in (out_base, out_head):
            if d.exists():
                shutil.rmtree(d)
        compute(out_base, base)
        compute(out_head, ROOT)
        summary: dict[str, Any] = {"env_key": _load(out_head)["env"]["key"], "mode": "differential"}
        code = compare(out_base, out_head, verbose=True, summary=summary, label=" (head vs base, same machine)")
        summary["declared"] = bool(args.declared)
        if code and args.declared:
            print("output differs from the base and the PR declares it (tests/golden/ edited): allowed. "
                  "Review the diff above.")
            code = 0
        elif code:
            print("output differs from the base and the PR does not declare it. A move must leave every byte "
                  "unchanged; an intended change logs its reason: scripts/golden.py record --reason '...' "
                  "(or --log-only).")
        if args.summary_json is not None:
            args.summary_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        return code

    actual = args.actual or Path(tempfile.mkdtemp(prefix="kalos-golden-out-"))
    if actual.exists():
        shutil.rmtree(actual)
    compute(actual)
    key = _load(actual)["env"]["key"]
    expected = GOLDEN_DIR / key
    summary = {"env_key": key, "recorded_for_env": (expected / MANIFEST).exists()}
    if not summary["recorded_for_env"]:
        recorded = sorted(p.name for p in GOLDEN_DIR.iterdir() if (p / MANIFEST).exists())
        print(f"no records for {key} (recorded: {recorded}). The computed output is in {actual}.")
        code = 1 if args.require else 0
    else:
        code = compare(expected, actual, verbose=args.cmd == "diff" or args.actual is not None, summary=summary)
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
