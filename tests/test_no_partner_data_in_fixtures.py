"""No partner or customer data in committed test data, goldens, results or examples.

Fixtures are synthetic and copy headers and formats only (AGENTS.md §3). This
guard fails when a tracked file under the data roots below carries a marker of
partner or customer data: a lab run code (S26 + month + day), a partner or
customer name, or a clone composite-ID shape (S1N.., S2N.., S3N..). A failure
names `path:line (marker type)` only; the matched text is never printed.

Scope is git-tracked files (what a commit ships), so ignored local copies of
external data do not trip it. There is no allowlist: a finding is fixed, not
excused.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Roots that hold data rather than code; plus every `results/` directory anywhere
# (benchmark outputs, e.g. kalos/bench/faults/results).
DATA_ROOTS = (
    "tests/fixtures",
    "tests/golden",
    "tests/goldens",
    "tests/data",
    "tests/vectors",
    "benchmarks",
    "examples",
)

_L = "(?<![A-Za-z])"
_R = "(?![A-Za-z])"
# Names are split so this file never matches the repo-wide partner-data grep
# (AGENTS.md workflow) itself; the compiled patterns are the full names.
_NAMES = "|".join(["ana" "gram", "ol" "on", "cyt" "ena", "lead[-_ ]?" "gene"])
_V = "V3" "25"
MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("lab run code (S26...)", re.compile(r"(?<![A-Za-z0-9])S26[A-Z]{3}[0-9]{2}(?![0-9])")),
    (
        "partner or customer name",
        re.compile(rf"(?i){_L}(?:{_NAMES}){_R}|(?<![A-Za-z0-9]){_V}(?![0-9])"),
    ),
    ("clone composite ID shape (S1N/S2N/S3N)", re.compile(r"(?<![A-Za-z0-9])S[123]N[0-9]+(?![0-9])")),
]



def _tracked_data_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:  # the guard must not pass blind
        pytest.fail(f"git ls-files failed ({type(exc).__name__}); the partner-data guard needs a git checkout")
    files = []
    for name in filter(None, out.split("\0")):
        p = PurePosixPath(name)
        if any(name == r or name.startswith(r + "/") for r in DATA_ROOTS) or "results" in p.parts[:-1]:
            files.append(name)
    return files


def find_markers(text: str) -> list[tuple[int, str]]:
    """(line number, marker type) for every marker in text. Never returns the match."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in MARKERS:
            if pattern.search(line):
                hits.append((lineno, kind))
    return hits


def _scan() -> list[tuple[str, int, str]]:
    hits = []
    for name in _tracked_data_files():
        path = ROOT / name
        if not path.is_file():  # deleted in the worktree but still in the index
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        hits.extend((name, lineno, kind) for lineno, kind in find_markers(text))
    return hits


def test_scope_covers_the_data_roots() -> None:
    files = _tracked_data_files()
    assert any(f.startswith("tests/fixtures/") for f in files)
    assert any(f.startswith("benchmarks/results/") for f in files)
    assert any(f.startswith("examples/") for f in files)
    assert any(f.startswith("kalos/bench/") and "/results/" in f for f in files)
    assert not any(f.startswith("kalos/") and "/results/" not in f for f in files)


def test_no_partner_or_customer_markers_in_test_data_goldens_results_or_examples() -> None:
    found = [f"{p}:{n} ({k})" for p, n, k in _scan()]
    assert not found, (
        "partner/customer markers in committed data (fixtures copy headers only; see AGENTS.md §3):\n  "
        + "\n  ".join(found)
    )



# Markers are assembled from parts so this file holds no complete run code or ID.
RUN = "S26" + "APR" + "07"


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        (f"sample,{RUN}_plate1,0.4", "lab run code (S26...)"),
        (f'"experiment": "{RUN}"', "lab run code (S26...)"),
        ("source: " + "Ana" + "gram media DoE", "partner or customer name"),
        ("lipase " + "V3" + "25 titer", "partner or customer name"),
        ("CDMO=" + "Ol" + "on", "partner or customer name"),
        ("vendor " + "cyt" + "ena BPS", "partner or customer name"),
        ("Lead" + "Gene" + "_cho_dynamics", "partner or customer name"),
        ("clone," + "S2" + "N14,384w", "clone composite ID shape (S1N/S2N/S3N)"),
    ],
)
def test_markers_are_detected(line: str, kind: str) -> None:
    assert find_markers(line) == [(1, kind)]


@pytest.mark.parametrize(
    "line",
    [
        "a co" "lon: separates keys",
        "S26 wells; S2 N14",
        "SS1N2, S4N1, S1Nx",
        _V + "0 rpm, X" + _V,
        "clone_id,campaign,scale",
        "co" "lonel, Co" "lony, so" "lon",
    ],
)
def test_synthetic_lookalikes_are_not_flagged(line: str) -> None:
    assert find_markers(line) == []
