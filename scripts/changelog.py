"""Fold the entries in changes/*.md into CHANGELOG.md under "## Unreleased", then
delete them. Newest file (by git commit time, else mtime) first. `--check` only lists
what would be folded."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGES = ROOT / "changes"
CHANGELOG = ROOT / "CHANGELOG.md"
MARKER = "## Unreleased\n"


def _added_at(path: Path) -> float:
    git = shutil.which("git")
    if git is None:
        return path.stat().st_mtime
    # Fixed argument list; `path` is a file this script globbed from changes/.
    out = subprocess.run([git, "log", "-1", "--format=%ct", "--", str(path)],  # noqa: S603
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return float(out) if out else path.stat().st_mtime


def main(argv: list[str]) -> int:
    entries = sorted((p for p in CHANGES.glob("*.md") if p.name != "README.md"),
                     key=_added_at, reverse=True)
    if not entries:
        print("no entries in changes/")
        return 0
    for p in entries:
        print(p.relative_to(ROOT))
    if "--check" in argv:
        return 0
    text = CHANGELOG.read_text()
    if MARKER not in text:
        print(f"{CHANGELOG.name} has no {MARKER.strip()!r} section", file=sys.stderr)
        return 1
    block = "".join(p.read_text().strip() + "\n\n" for p in entries)
    CHANGELOG.write_text(text.replace(MARKER, MARKER + "\n" + block, 1))
    for p in entries:
        p.unlink()
    print(f"folded {len(entries)} entries into {CHANGELOG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
