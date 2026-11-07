"""Write an EMPTY, header-only CSV run-sheet template for the blood-reactor profile.

Schema only: the file has the column names declared in `roles.json` and no data
rows. Fill it with your own measured runs; nothing here contains or implies any
wet-lab result. Do not commit the generated file.

Usage:
    python examples/blood_reactor_profile/make_template.py /path/to/runs_template.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def template_columns(roles: dict) -> list[str]:
    """Header order: ids, grouping column, features, then the target."""
    cols: list[str] = [*roles.get("ids", [])]
    if roles.get("groups"):
        cols.append(roles["groups"])
    cols += list(roles.get("features", []))
    cols.append(roles["target"])
    return list(dict.fromkeys(cols))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="where to write the header-only CSV")
    ap.add_argument("--roles", type=Path, default=HERE / "roles.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args(argv)
    if args.out.exists() and not args.force:
        ap.error(f"{args.out} exists; pass --force to overwrite")
    roles = json.loads(args.roles.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        csv.writer(fh).writerow(template_columns(roles))
    print(f"wrote header-only template ({len(template_columns(roles))} columns) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
