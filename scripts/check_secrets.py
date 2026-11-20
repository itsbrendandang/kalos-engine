#!/usr/bin/env python3
"""Secret guard: scan the lines a change adds for secret-shaped content.

    python3 scripts/check_secrets.py                       # commits not yet pushed (@{u}..HEAD)
    python3 scripts/check_secrets.py --range origin/main..HEAD --paths
    python3 scripts/check_secrets.py --staged --paths      # what `git commit` would record

Only ADDED lines are scanned, so existing files are judged when they change, not
on every run. A hit is reported as `path:line (type)`; the matched value is
never printed, logged or echoed (a leaked secret is rotated, not re-displayed).
`--paths` also flags newly added files whose name marks them as secret-bearing
(.env*, *.pem, *.key, ...); those are reported as `path:0 (type)`.

Exit codes: 0 clean, 1 hits, 2 git failed. The guard fails closed: a gate that
cannot read the diff must not pass it. With no upstream, `@{u}` falls back to
the last commit (HEAD~1..HEAD, or the whole tree for a root commit).

A line that is a deliberate, reviewed exception carries `secret-guard: allow`.

Stdlib only. Canonical copy: kalos-engine/scripts/check_secrets.py; kalos-web
keeps a byte-identical copy. Idea credit: claude-tools-marketplace@08fff7e
plugins/bioqore-team/hooks/check-secrets.py by Kai Suematsu for Bioqore;
re-implemented, no code copied.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
UPSTREAM_RANGE = "@{u}..HEAD"
ALLOW_PRAGMA = "secret-guard: allow"

# (type, pattern, placeholder_exempt). Patterns are matched against one added
# line; the first match names the line's type. placeholder_exempt=False means a
# placeholder-looking line still counts (the setting itself is the problem).
# A secret-ish name, but not one that names where a secret lives (`*_FILE`,
# `*_ENV_VAR`, `token_id`): those hold paths and identifiers, not values.
_NAME = (
    r"[A-Za-z0-9_]*(?:api[_-]?key|secret|token|passw(?:or)?d|access[_-]?key|private[_-]?key)"
    r"(?![A-Za-z0-9_]*_(?:file|path|dir|env|var|id|ids|name|url|count|len|length|type|kind|ttl|field|header|aud|audience|iss|issuer|kid|alg|prefix|scope|endpoint)\b)"
    r"[A-Za-z0-9_]*"
)
PATTERNS: list[tuple[str, re.Pattern[str], bool]] = [
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"), True),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}"), True),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"), True),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"), True),
    ("GitLab token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), True),
    ("Slack token", re.compile(r"\bxox[abeprs]-[A-Za-z0-9-]{10,}"), True),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), True),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), True),
    ("bearer token", re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]{16,}"), True),
    ("private JWK component", re.compile(r"\"d\"\s*:\s*\"[A-Za-z0-9_-]{32,}\""), True),
    ("credentialed database URL", re.compile(r"\bpostgres(?:ql)?(?:\+\w+)?://[^:\s/@]+:[^@\s/]+@"), True),
    (
        "named Kalos secret",
        re.compile(r"(?i)\b(?:BETTER_AUTH_SECRET|KALOS_ANON_SALT|GOOGLE_CLIENT_SECRET)['\"]?\s*[:=]\s*['\"]?[^\s'\"$<{]{8,}"),
        True,
    ),
    ("secret behind NEXT_PUBLIC_", re.compile(r"(?i)\bNEXT_PUBLIC_[A-Z0-9_]*(?:SECRET|KEY|TOKEN|PASSWORD)\b"), False),
    (
        "KALOS_DEV_NO_AUTH enabled",
        re.compile(r"(?i)\bKALOS_DEV_NO_AUTH['\"]?\s*[:=]\s*['\"]?(?:1|true|yes|on)\b"),
        False,
    ),
    # Generic assignments: a quoted literal, or an unquoted env/YAML value that
    # ends the line. Unquoted Python (`token = make_token(...)`) is not a hit.
    ("secret assignment", re.compile(rf"(?i)\b{_NAME}['\"]?\s*[:=]\s*(['\"])[^'\"\s]{{12,}}\1"), True),
    (
        "secret assignment",
        re.compile(rf"(?i)^\s*(?:export\s+|-\s+)?{_NAME}\s*[:=]\s*[A-Za-z0-9+_=-][A-Za-z0-9/+_=-]{{11,}}\s*$"),
        True,
    ),
]

# Documentation and template values, not secrets.
PLACEHOLDER = re.compile(
    r"(?i)(?:your[-_ ]|replace[-_ ]?(?:me|with)|change[-_ ]?me|<[^<>\s]+>|example|placeholder|"
    r"dummy|sample|fake|not[-_ ]a[-_ ]real|redacted|x{4,}|\*{4,}|\.\.\.|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)"
)

SENSITIVE_NAMES = (
    ".env*",
    "*.pem",
    "*.key",
    "credentials*",
    "secrets.*",
    "service-account*.json",
    "tenants*.json",
    "*jwks*private*",
)
TEMPLATE_SUFFIXES = (".example", ".sample", ".template")


class GitError(RuntimeError):
    """git could not produce what the guard needs; the gate fails closed."""


@dataclass(frozen=True, order=True)
class Hit:
    path: str
    line: int
    kind: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} ({self.kind})"


def classify(text: str) -> str | None:
    """The secret type of one added line, or None. Never returns the value."""
    if ALLOW_PRAGMA in text:
        return None
    placeholder = PLACEHOLDER.search(text) is not None
    for kind, pattern, exempt in PATTERNS:
        if exempt and placeholder:
            continue
        if pattern.search(text):
            return kind
    return None


def _diff_path(header: str) -> str | None:
    """`+++ b/path` -> `path`; `+++ /dev/null` -> None (file deleted)."""
    raw = header[4:].rstrip("\n").split("\t", 1)[0]
    if raw == "/dev/null":
        return None
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return raw.removeprefix("b/")


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(diff: str) -> Iterator[tuple[str, int, str]]:
    """Yield (path, new line number, text) for every added line of a unified diff.

    `+++ ` names the file only in a file header (before its first hunk); inside a
    hunk, an added line that itself starts with `++` is content, not a header.
    """
    path: str | None = None
    lineno = 0
    in_header = True
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, in_header = None, True
        elif in_header and line.startswith("+++ "):
            path = _diff_path(line)
        elif (m := _HUNK.match(line)) is not None:
            lineno, in_header = int(m.group(1)), False
        elif in_header:
            continue
        elif line.startswith("+"):
            if path is not None:
                yield path, lineno, line[1:]
            lineno += 1
        elif line.startswith(" "):
            lineno += 1


def scan_diff(diff: str) -> list[Hit]:
    return [Hit(p, n, kind) for p, n, text in added_lines(diff) if (kind := classify(text)) is not None]


def sensitive_name(path: str) -> bool:
    name = PurePosixPath(path).name
    if name.endswith(TEMPLATE_SUFFIXES):
        return False
    return any(fnmatch.fnmatchcase(name, pat) for pat in SENSITIVE_NAMES)


def scan_paths(paths: Iterable[str]) -> list[Hit]:
    return [Hit(p, 0, "secret-bearing file name") for p in paths if sensitive_name(p)]


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "-C", repo, "-c", "core.quotePath=false", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)  # noqa: S603
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(f"git {args[0]} failed: {type(exc).__name__}") from exc
    if check and proc.returncode != 0:
        first = (proc.stderr.strip().splitlines() or ["no output"])[0]
        raise GitError(f"git {args[0]} failed: {first}")
    return proc


def _resolves(repo: str, rev: str) -> bool:
    return _git(repo, "rev-parse", "--verify", "--quiet", rev, check=False).returncode == 0


def resolve_range(repo: str, rng: str) -> str | None:
    """`@{u}..HEAD` without an upstream becomes the last commit. None = no commits yet."""
    if "@{u}" not in rng or _resolves(repo, "@{u}"):
        return rng
    if _resolves(repo, "HEAD~1"):
        return "HEAD~1..HEAD"
    if _resolves(repo, "HEAD"):
        return f"{EMPTY_TREE}..HEAD"
    return None


_DIFF = ("diff", "--no-color", "--no-ext-diff", "--unified=0", "-M")


def scan(repo: str, rng: str | None, staged: bool, paths: bool) -> list[Hit]:
    sources: list[tuple[str, ...]] = []
    if rng is not None and (resolved := resolve_range(repo, rng)) is not None:
        sources.append((resolved,))
    if staged:
        sources.append(("--cached",))
    hits: set[Hit] = set()
    for src in sources:
        hits.update(scan_diff(_git(repo, *_DIFF, *src).stdout))
        if paths:
            names = _git(repo, *_DIFF, "--name-only", "--diff-filter=A", "-z", *src).stdout
            hits.update(scan_paths(n for n in names.split("\0") if n))
    return sorted(hits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--range", dest="rng", metavar="RANGE", help=f"git range to scan (default {UPSTREAM_RANGE})")
    ap.add_argument("--staged", action="store_true", help="scan the index (what `git commit` records)")
    ap.add_argument("--paths", action="store_true", help="also flag newly added secret-bearing file names")
    ap.add_argument("-C", dest="repo", default=".", help="repository to scan (default: cwd)")
    args = ap.parse_args(argv)
    rng = args.rng if args.rng is not None or args.staged else UPSTREAM_RANGE
    try:
        hits = scan(args.repo, rng, args.staged, args.paths)
    except GitError as exc:
        print(f"check_secrets: {exc}; failing closed", file=sys.stderr)
        return 2
    for hit in hits:
        print(hit)
    if hits:
        print(
            f"check_secrets: {len(hits)} secret-shaped addition(s); values not shown. Remove them from the "
            f"change (a real leaked secret is rotated first), or mark a reviewed false positive "
            f"with '{ALLOW_PRAGMA}'.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
