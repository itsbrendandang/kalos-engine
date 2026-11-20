#!/usr/bin/env python3
"""Write docs/openapi.json: the engine's OpenAPI document, the HTTP contract.

    .venv/bin/python scripts/export_openapi.py          # rewrite the file
    .venv/bin/python scripts/export_openapi.py --check  # exit 1 on drift

The live app serves `/openapi.json` only in development (docs/HARDENING.md), so
the checked-in copy is what the web (kalos-web) and reviewers read. It is
rendered from `app.openapi()` with sorted keys and a trailing newline, so the
file is byte-stable. tests/contract/test_openapi_drift.py fails when it is
stale: a route, parameter or request/response schema changed without the file
being regenerated in the same commit.

Reorg guard R01 (~/.kalos-dev/REORG_PLAN.md): every behaviour-preserving move
must leave this file byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCUMENT = ROOT / "docs" / "openapi.json"


def build() -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from kalos.portal.app import app

    return app.openapi()


def render(document: dict[str, Any] | None = None) -> str:
    return json.dumps(build() if document is None else document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale; write nothing")
    args = parser.parse_args(argv)
    text = render()
    rel = DOCUMENT.relative_to(ROOT)
    if args.check:
        current = DOCUMENT.read_text(encoding="utf-8") if DOCUMENT.exists() else ""
        if current != text:
            print(f"{rel} is stale; run scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    DOCUMENT.write_text(text, encoding="utf-8")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
