#!/usr/bin/env python3
"""Write docs/route-scopes.json: the scope(s) each portal API route requires.

    .venv/bin/python scripts/export_scope_manifest.py          # rewrite the file
    .venv/bin/python scripts/export_scope_manifest.py --check  # exit 1 on drift

The web (kalos-web) should mint each engine token with exactly the scopes
listed here for the route it calls, rather than deriving them from the HTTP
method (security review S1). tests/test_route_scopes.py fails when this file
is stale. See kalos/portal/route_scopes.py for how scopes are discovered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "route-scopes.json"


def render() -> str:
    sys.path.insert(0, str(ROOT))
    from kalos.portal.app import app
    from kalos.portal.route_scopes import build_manifest

    return json.dumps(build_manifest(app), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale; write nothing")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = MANIFEST.read_text() if MANIFEST.exists() else ""
        if current != text:
            print(f"{MANIFEST.relative_to(ROOT)} is stale; run scripts/export_scope_manifest.py", file=sys.stderr)
            return 1
        return 0
    MANIFEST.write_text(text)
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
