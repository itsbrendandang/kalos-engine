"""The README's configuration table and the code agree on every `KALOS_*` setting.

Two directions, both fail the build:

- a `KALOS_*` name the code reads (a whole string literal in `kalos/`, or a
  `${KALOS_*}` expansion in `scripts/*.sh`) that the README table omits;
- a `KALOS_*` name in the README table that the code does not read, unless the
  row says it lives on an unmerged branch (`not yet merged`) or is read only by
  tests (`tests only`).

String literals are collected with `ast`, so a name mentioned in a comment or a
docstring does not count as "read": only a constant that is exactly the name,
which is how every `os.environ.get("KALOS_...")`, `*_VAR = "KALOS_..."` and
`env_int("KALOS_...", ...)` in the package spells it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NAME = re.compile(r"^KALOS_[A-Z0-9_]+$")
EXEMPT = ("not yet merged", "tests only")


def names_in_code() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "kalos").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and NAME.match(node.value):
                found.add(node.value)
    for path in (ROOT / "scripts").glob("*.sh"):
        found.update(re.findall(r"\$\{(KALOS_[A-Z0-9_]+)", path.read_text(encoding="utf-8")))
    return found


def readme_table_rows() -> list[str]:
    """The rows of the `### Engine settings` table: one line each, starting with `|`."""
    text = README.read_text(encoding="utf-8")
    start = text.index("### Engine settings")
    end = text.index("\n### ", start + 1)
    section = text[start:end]
    rows = [ln for ln in section.splitlines() if ln.startswith("| `KALOS_")]
    assert rows, "the Engine settings table has no rows"
    return rows


def names_in_readme() -> tuple[set[str], set[str]]:
    """(names the code must read, names allowed to be absent from the code)."""
    required: set[str] = set()
    exempt: set[str] = set()
    for row in readme_table_rows():
        first_cell = row.split("|")[1]
        names = set(re.findall(r"`(KALOS_[A-Z0-9_]+)`", first_cell))
        assert names, f"table row without a KALOS_ name: {row[:60]}"
        if any(marker in row.lower() for marker in EXEMPT):
            exempt |= names
        else:
            required |= names
    return required, exempt


def test_every_setting_the_code_reads_is_in_the_readme() -> None:
    required, exempt = names_in_readme()
    missing = names_in_code() - required - exempt
    assert not missing, f"read by the code, missing from README.md 'Engine settings': {sorted(missing)}"


def test_every_readme_setting_is_read_by_the_code() -> None:
    required, _ = names_in_readme()
    stale = required - names_in_code()
    assert not stale, (
        f"in README.md 'Engine settings' but not read by kalos/ or scripts/: {sorted(stale)}. "
        "Remove the row, or mark it 'not yet merged' / 'tests only'."
    )


def test_exempt_rows_are_really_absent_from_the_code() -> None:
    """A row marked 'not yet merged' or 'tests only' must not describe something the
    code already reads; once a branch merges, the marker has to go."""
    _, exempt = names_in_readme()
    merged = exempt & names_in_code()
    assert not merged, f"marked exempt in README.md but now read by the code: {sorted(merged)}"
