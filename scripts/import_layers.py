#!/usr/bin/env python3
"""Import-layer contract for the engine (reorg guard R03).

    .venv/bin/python scripts/import_layers.py                    # report
    .venv/bin/python scripts/import_layers.py --write-baseline   # re-seed the baseline

Every module under kalos/ is placed in one LAYER (by longest module-prefix
match in `PLACEMENT`, which lists today's paths and the target paths of
~/.kalos-dev/REORG_PLAN.md section 2.1 side by side, so the same rules hold
before, during and after the Phase 2 moves). `ALLOWED` says which layers each
layer may import. Every `import kalos...` / `from kalos... import` statement is
read with the AST (module level, inside functions and under TYPE_CHECKING
alike; nothing is imported or executed), and an edge from layer A to layer B
not in `ALLOWED[A]` is a violation.

The layers, lowest first:

    util      ids, JSON helpers                 (kalos.util; today portal.tenancy, portal.serialization)
    data      anonymizer                        (kalos.data)
    config    env readers, flags, deploy config (kalos.config; today envconfig, portal.config)
    core      the ML layer, unchanged           (kalos.core, kalos.features, kalos.versions)
    libs      domain libraries                  (normalize, domains, validation, scale, providers)
    runtime   heavy-work admission, body caps, the state-dir lock   (kalos.runtime; today portal.busy/limits/engine_lock)
    ingest    upload parsing and cell typing    (kalos.ingest; today portal.uploads, portal.cells)
    storage   SQLite connection and migrations  (kalos.storage; today portal.migrate)
    analysis  the analysis pipeline, no HTTP    (kalos.analysis; today portal.analysis*, setup_settings, ...)
    campaigns campaign rules and records        (kalos.campaigns; today portal.campaign*, stages, clone_registry)
    jobs      background jobs                   (kalos.jobs; today portal.jobs, job_runner, ...)
    assist    the grounded assistant            (kalos.assist except its routers)
    api       HTTP only: app, routers, auth     (kalos.api; today portal.app, *_api, *_routes, auth*)
    bench     benchmarks and the fault suite    (kalos.bench)
    kit       torch-free public facade          (kalos.kit)
    compat    the kalos.portal shim package (after the moves) and the kalos root

Imports of the bare root package (`from kalos import __version__`) are allowed
from every layer: any `import kalos.x` executes it anyway.

Rules: nothing imports `api` except `api` (and compat); nothing imports `bench`;
`core` imports nothing above config. The test (tests/contract/test_import_layers.py)
is REPORT-ONLY for now: it prints the violations, compares them with the
checked-in baseline (tests/contract/import_layers_baseline.json) and fails only
on a malformed placement (a module in no layer). It becomes enforcing (new
violations fail; the baseline may only shrink) after the API move (R14), and
the baseline must be empty after the assist decoupling (R19).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "kalos"
BASELINE = ROOT / "tests" / "contract" / "import_layers_baseline.json"

LAYER_ORDER = (
    "util", "data", "config", "core", "libs", "runtime", "ingest", "storage", "analysis",
    "campaigns", "jobs", "assist", "api", "bench", "kit", "compat",
)

# Longest prefix wins. Today's paths first, then the REORG_PLAN 2.1 target paths.
PLACEMENT: dict[str, str] = {
    "kalos": "compat",  # the root package __init__ (re-exports core + data)
    # util
    "kalos.util": "util",
    "kalos.portal.tenancy": "util",
    "kalos.portal.serialization": "util",
    # data / config / core
    "kalos.data": "data",
    "kalos.envconfig": "config",
    "kalos.config": "config",
    "kalos.portal.config": "config",
    "kalos.core": "core",
    "kalos.features": "core",
    "kalos.versions": "core",
    # libs
    "kalos.normalize": "libs",
    "kalos.domains": "libs",
    "kalos.validation": "libs",
    "kalos.scale": "libs",
    "kalos.providers": "libs",
    # runtime
    "kalos.runtime": "runtime",
    "kalos.portal.busy": "runtime",
    "kalos.portal.limits": "runtime",
    "kalos.portal.engine_lock": "runtime",
    # ingest
    "kalos.ingest": "ingest",
    "kalos.portal.uploads": "ingest",
    "kalos.portal.cells": "ingest",
    # storage
    "kalos.storage": "storage",
    "kalos.portal.migrate": "storage",
    # analysis
    "kalos.analysis": "analysis",
    "kalos.portal.analysis": "analysis",
    "kalos.portal.analysis_parts": "analysis",
    "kalos.portal.recipe_values": "analysis",
    "kalos.portal.validate": "analysis",
    "kalos.portal.viz_utils": "analysis",
    "kalos.portal.pipeline": "analysis",
    "kalos.portal.run_prediction": "analysis",
    "kalos.portal.run_record": "analysis",
    "kalos.portal.setup_settings": "analysis",
    "kalos.portal.operating_ranges": "analysis",
    "kalos.portal.verdict": "analysis",
    # campaigns
    "kalos.campaigns": "campaigns",
    "kalos.portal.campaign": "campaigns",
    "kalos.portal.campaign_list": "campaigns",
    "kalos.portal.campaign_settings": "campaigns",
    "kalos.portal.campaign_stage": "campaigns",
    "kalos.portal.stages": "campaigns",
    "kalos.portal.clone_registry": "campaigns",
    "kalos.portal.prediction_history": "campaigns",  # arriving with feat/prediction-history
    "kalos.portal.track_record": "campaigns",  # arriving with feat/prediction-history
    "kalos.portal.result_referee": "campaigns",  # arriving with feat/data-checks
    "kalos.records": "campaigns",  # projects, factors, lineage, outcomes (feat/voyager-records)
    # jobs
    "kalos.jobs": "jobs",
    "kalos.portal.jobs": "jobs",
    "kalos.portal.job_runner": "jobs",
    "kalos.portal.job_worker": "jobs",
    "kalos.portal.reanalyze_job": "jobs",
    # assist (its two routers are api)
    "kalos.assist": "assist",
    "kalos.assist.routes": "api",
    "kalos.assist.sop_routes": "api",
    # api
    "kalos.api": "api",
    "kalos.portal": "api",  # every other portal module: app, routers, auth, __main__
    # bench / kit
    "kalos.bench": "bench",
    "kalos.kit": "kit",
}

_BELOW_API = ("util", "data", "config", "core", "libs", "runtime", "ingest", "storage", "analysis",
              "campaigns", "jobs", "assist")
ALLOWED: dict[str, frozenset[str]] = {
    "util": frozenset({"util"}),
    "data": frozenset({"util", "data"}),
    "config": frozenset({"util", "data", "config"}),
    "core": frozenset({"util", "data", "config", "core"}),
    "libs": frozenset({"util", "data", "config", "core", "libs"}),
    "runtime": frozenset({"util", "data", "config", "runtime"}),
    "ingest": frozenset({"util", "data", "config", "core", "libs", "runtime", "ingest"}),
    "storage": frozenset({"util", "data", "config", "storage"}),
    "analysis": frozenset({"util", "data", "config", "core", "libs", "ingest", "analysis"}),
    "campaigns": frozenset({"util", "data", "config", "core", "libs", "ingest", "storage", "analysis",
                            "campaigns"}),
    "jobs": frozenset({"util", "data", "config", "core", "libs", "runtime", "ingest", "storage", "analysis",
                       "campaigns", "jobs"}),
    "assist": frozenset({"util", "data", "config", "core", "libs", "ingest", "storage", "analysis",
                         "campaigns", "assist"}),
    "api": frozenset({*_BELOW_API, "api"}),
    "bench": frozenset({"util", "data", "config", "core", "libs", "ingest", "storage", "analysis",
                        "campaigns", "bench"}),
    "kit": frozenset({"util", "data", "config", "core", "kit"}),
    "compat": frozenset(LAYER_ORDER),
}


def layer_of(module: str) -> str | None:
    if module == "kalos":
        return PLACEMENT["kalos"]
    parts = module.split(".")
    for i in range(len(parts), 1, -1):  # the bare root matches only itself
        hit = PLACEMENT.get(".".join(parts[:i]))
        if hit is not None:
            return hit
    return None


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    base = module.split(".") if is_package else module.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, *([target] if target else [])])


def imports_of(path: Path) -> list[dict[str, Any]]:
    """Every kalos import in one file: target module, line, and whether it runs
    at import time (module level) or lazily (inside a function)."""
    module = module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[dict[str, Any]] = []

    def visit(node: ast.AST, lazy: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                out.extend({"target": a.name, "line": child.lineno, "lazy": lazy}
                           for a in child.names if a.name == "kalos" or a.name.startswith("kalos."))
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    base = _resolve_relative(module, path.name == "__init__.py", child.level, child.module)
                else:
                    base = child.module or ""
                if base == "kalos" or base.startswith("kalos."):
                    # `from kalos.portal import jobs` imports the submodule kalos.portal.jobs.
                    for a in child.names:
                        sub = f"{base}.{a.name}"
                        is_sub = (PACKAGE.parent / Path(*sub.split("."))).with_suffix(".py").exists() or (
                            PACKAGE.parent / Path(*sub.split(".")) / "__init__.py").exists()
                        out.append({"target": sub if is_sub else base, "line": child.lineno, "lazy": lazy})
            visit(child, inner)

    visit(tree, False)
    return out


def scan() -> dict[str, Any]:
    modules: dict[str, str | None] = {}
    edges: list[dict[str, Any]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        mod = module_name(path)
        modules[mod] = layer_of(mod)
        for imp in imports_of(path):
            edges.append({"src": mod, "dst": imp["target"], "line": imp["line"], "lazy": imp["lazy"],
                          "file": str(path.relative_to(ROOT))})
    violations = []
    for e in edges:
        src_layer, dst_layer = layer_of(e["src"]), layer_of(e["dst"])
        if src_layer is None or dst_layer is None:
            continue
        if e["dst"] == "kalos":
            # The bare root (`from kalos import __version__`): every `import kalos.x`
            # executes it anyway, so naming it adds no dependency.
            continue
        if dst_layer not in ALLOWED[src_layer]:
            violations.append({**e, "src_layer": src_layer, "dst_layer": dst_layer})
    return {"modules": modules, "edges": edges, "violations": violations}


def violation_keys(violations: list[dict[str, Any]]) -> list[str]:
    """Baseline keys: module-to-module, without line numbers (so an unrelated edit
    above an import does not churn the baseline)."""
    return sorted({f"{v['src']} -> {v['dst']}" for v in violations})


def package_cycles(edges: list[dict[str, Any]]) -> list[tuple[str, str]]:
    def pkg(m: str) -> str:
        parts = m.split(".")
        return ".".join(parts[:2]) if len(parts) > 1 else m

    pairs = {(pkg(e["src"]), pkg(e["dst"])) for e in edges if pkg(e["src"]) != pkg(e["dst"])}
    return sorted((a, b) for a, b in pairs if (b, a) in pairs and a < b)


def report(result: dict[str, Any], baseline: list[str] | None) -> str:
    v = result["violations"]
    keys = violation_keys(v)
    by_pair = Counter(f"{x['src_layer']} -> {x['dst_layer']}" for x in v)
    lines = [f"import-layer report: {len(result['modules'])} modules, {len(result['edges'])} kalos imports, "
             f"{len(v)} violating import statements ({len(keys)} module edges)"]
    unplaced = sorted(m for m, layer in result["modules"].items() if layer is None)
    if unplaced:
        lines.append(f"  UNPLACED modules (add them to PLACEMENT): {unplaced}")
    for pair, n in sorted(by_pair.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {n:4d}  {pair}")
    cycles = package_cycles(result["edges"])
    lines.append(f"  package cycles: {['<->'.join(c) for c in cycles] or 'none'}")
    if baseline is not None:
        new = sorted(set(keys) - set(baseline))
        fixed = sorted(set(baseline) - set(keys))
        lines.append(f"  vs baseline ({len(baseline)} edges): {len(new)} new, {len(fixed)} fixed")
        lines.extend(f"    NEW   {k}" for k in new)
        lines.extend(f"    FIXED {k} (remove it from the baseline)" for k in fixed)
    by_src: dict[str, list[str]] = defaultdict(list)
    for x in v:
        lazy = " (lazy)" if x["lazy"] else ""
        by_src[x["src_layer"]].append(f"{x['file']}:{x['line']} {x['src']} -> {x['dst']} [{x['dst_layer']}]{lazy}")
    for layer in LAYER_ORDER:
        if by_src.get(layer):
            lines.append(f"  [{layer}]")
            lines.extend(f"    {s}" for s in sorted(by_src[layer]))
    return "\n".join(lines)


def load_baseline() -> list[str]:
    if not BASELINE.exists():
        return []
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return list(data["violations"])


def write_baseline(result: dict[str, Any]) -> None:
    payload = {
        "about": "Known import-layer violations (module edges), seeded from main. Report-only until R14; "
                 "may only shrink after that. Regenerate: scripts/import_layers.py --write-baseline",
        "violations": violation_keys(result["violations"]),
    }
    BASELINE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    result = scan()
    if args.write_baseline:
        write_baseline(result)
        print(f"wrote {BASELINE.relative_to(ROOT)}")
    print(report(result, load_baseline()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
