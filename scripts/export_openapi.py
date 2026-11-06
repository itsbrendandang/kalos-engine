#!/usr/bin/env python3
"""Write the engine's OpenAPI spec to docs/openapi.json (or check it).

    python scripts/export_openapi.py           # rewrite docs/openapi.json
    python scripts/export_openapi.py --check   # exit 1 when it is out of date

The live `/openapi.json` is off outside development (kalos/portal/app.py), so
the spec is exported here and committed; the web copies it and checks its
contract number (kalos-web `lib/contract.ts`). `tests/test_openapi_drift.py`
fails when the committed file differs from what this script would write.

Output is deterministic: sorted keys, two-space indent, trailing newline. The
contract number is recorded as `info.x-kalos-contract`, the value every
response sends as the `X-Kalos-Contract` header.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "openapi.json"


def build_spec() -> dict[str, Any]:
    """The spec of `kalos.portal.app.app` plus the contract number."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from kalos.portal.app import app
    from kalos.portal.contract import CONTRACT_VERSION

    spec = json.loads(json.dumps(app.openapi()))  # a deep copy, not FastAPI's cached dict
    spec.setdefault("info", {})["x-kalos-contract"] = CONTRACT_VERSION
    return spec


def render(spec: dict[str, Any] | None = None) -> str:
    return json.dumps(build_spec() if spec is None else spec, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if docs/openapi.json is out of date")
    parser.add_argument("--out", type=Path, default=SPEC_PATH, help="where to write (default docs/openapi.json)")
    args = parser.parse_args(argv)
    # Importing the app must never read or create a real state directory.
    os.environ["KALOS_STATE_DIR"] = tempfile.mkdtemp(prefix="kalos-openapi-")
    os.environ.pop("KALOS_DESKTOP", None)
    text = render()
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is out of date: run python scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
