"""Word lists the assistant checks its own output against.

`BANNED` words never appear in anything the assistant says, in any context.
`CLAIM` words state more than a record can hold (a future, a promise, a
cause, a best). The assistant may only use one when the source it cites uses
the same word.
"""
from __future__ import annotations

import re

# Never said by the assistant. Lower case; matched on word boundaries.
BANNED: tuple[str, ...] = (
    "validated",
    "compliant",
    "compliance",
    "gmp",
    "gxp",
    "part 11",
    "tamper-proof",
    "design space",
    "digital twin",
    "real-time",
    "certified",
)

# Stronger than any record. Each entry is a stem pattern; the list covers the
# owner's set (will, guarantees, proves, causes, optimal) and their forms.
CLAIM: tuple[str, ...] = (
    r"will",
    r"won't",
    r"guarantee[sd]?",
    r"prove[sdn]?",
    r"proving",
    r"cause[sd]?",
    r"causing",
    r"optimal",
    r"optimum",
)

_BANNED_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(w) for w in BANNED) + r")(?![a-z0-9])",
    re.IGNORECASE,
)
_CLAIM_RE = re.compile(r"(?<![a-z0-9'])(" + "|".join(CLAIM) + r")(?![a-z0-9'])", re.IGNORECASE)


def banned_in(text: str) -> list[str]:
    """Banned words found in `text`, lower case, in order of appearance."""
    return [m.group(1).lower() for m in _BANNED_RE.finditer(text)]


def claims_in(text: str) -> list[str]:
    """Claim words found in `text`, lower case, in order of appearance."""
    return [m.group(1).lower() for m in _CLAIM_RE.finditer(text)]


def unsupported_claims(sentence: str, sources: list[str]) -> list[str]:
    """Claim words in `sentence` that no text in `sources` contains."""
    found = set(claims_in(sentence))
    if not found:
        return []
    supported: set[str] = set()
    for text in sources:
        supported.update(claims_in(text))
    return sorted(found - supported)


__all__ = ["BANNED", "CLAIM", "banned_in", "claims_in", "unsupported_claims"]
