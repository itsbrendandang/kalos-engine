#!/usr/bin/env python3
"""Write docs/routes.json: every HTTP route the engine serves, with its owner.

    .venv/bin/python scripts/export_routes.py          # rewrite the file
    .venv/bin/python scripts/export_routes.py --check  # exit 1 on drift

One entry per (method, path): the endpoint function and the module that defines
it, the scopes and auth mode its dependency tree enforces (the same values as
docs/route-scopes.json, from kalos.portal.route_scopes.build_manifest), whether
it is in the OpenAPI schema, its declared success status and response model.

Reorg guard R04 (~/.kalos-dev/REORG_PLAN.md). A behaviour-preserving move
changes only the `module` of the routes it moves (which is the point: the diff
of this file shows reviewers exactly what moved); method, path, scopes, auth
and status stay identical. tests/contract/test_route_inventory.py fails when
the file is stale or disagrees with docs/route-scopes.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "docs" / "routes.json"


def build() -> list[dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from kalos.portal.app import app
    from kalos.portal.route_scopes import _walk, build_manifest

    scopes = {(e["method"], e["path"]): e for e in build_manifest(app)}
    app_deps = tuple(d.dependency for d in (app.router.dependencies or []) if d.dependency is not None)
    out: list[dict[str, Any]] = []
    for path, route, _extra in _walk(app.router.routes, "", app_deps):
        endpoint = route.endpoint
        model = route.response_model
        for method in sorted(route.methods or ()):
            manifest = scopes[(method, path)]
            entry: dict[str, Any] = {
                "method": method,
                "path": path,
                "module": endpoint.__module__,
                "endpoint": endpoint.__qualname__,
                "scopes": manifest["scopes"],
                "auth": manifest["auth"],
                "in_schema": bool(route.include_in_schema),
                "status_code": route.status_code,
                "response_model": None if model is None else getattr(model, "__name__", repr(model)),
            }
            if "conditional_scopes" in manifest:
                entry["conditional_scopes"] = manifest["conditional_scopes"]
            out.append(entry)
    out.sort(key=lambda e: (e["path"], e["method"]))
    return out


def render(inventory: list[dict[str, Any]] | None = None) -> str:
    return json.dumps(build() if inventory is None else inventory, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale; write nothing")
    args = parser.parse_args(argv)
    text = render()
    rel = INVENTORY.relative_to(ROOT)
    if args.check:
        current = INVENTORY.read_text(encoding="utf-8") if INVENTORY.exists() else ""
        if current != text:
            print(f"{rel} is stale; run scripts/export_routes.py", file=sys.stderr)
            return 1
        return 0
    INVENTORY.write_text(text, encoding="utf-8")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
