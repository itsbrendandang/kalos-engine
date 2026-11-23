"""`target_unit`: the display unit of the analysis target (VIZ_SPEC R6).

The web formats every target number with a unit ("4.2 g/L"). Before this block it
had to guess the unit from the column header client-side. Here the engine says it
once, with where it came from:

- `source: "user"`: the unit the user typed in the campaign setup. Shown as given.
- `source: "header"`: parsed from the target column header, ONLY for well-defined
  patterns (a trailing `(unit)` / `[unit]`, a trailing unit token after a space,
  or a trailing `_unit` / `_num_den` suffix whose every token is a known unit).
- `source: "none"`: no unit. An unrecognized suffix is NOT guessed at: `od600`,
  `yield_A`, `titer (replicate 2)` all give `unit: null`.

Nothing here converts values; this is a label. Bare one-letter suffixes (`_L`,
`_g`, `_C`, `_M`, `_h`) are refused on their own because they collide with
vessel labels ("5L"), letters and initials; they are accepted inside an explicit
ratio (`titer_g_L`) or inside brackets (`titer (g)`), where the intent is clear.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Callable
from typing import Any

from kalos.normalize.header_units import parse_header_unit

MAX_USER_UNIT_CHARS = 32

def _clean_user_unit(user_unit: object) -> str | None:
    if not isinstance(user_unit, str):
        return None
    text = user_unit.strip()
    if not text or len(text) > MAX_USER_UNIT_CHARS:
        return None
    if any(unicodedata.category(ch).startswith("C") for ch in text):
        return None
    return text


def target_unit_block(target: str | None, user_unit: str | None) -> dict[str, Any]:
    """`{unit, source, header, note}` for the analysis target.

    `user_unit` (campaign setup) wins when it is a non-empty printable string of
    at most `MAX_USER_UNIT_CHARS`; it is reported as typed, never normalized.
    Otherwise the header is parsed (`parse_header_unit`); else `unit: null`.
    """
    header = str(target) if target is not None else None
    parsed = parse_header_unit(header) if header else None
    user = _clean_user_unit(user_unit)
    if user is not None:
        note = "set by the user in the campaign setup"
        if parsed is not None and parsed != user:
            note += f"; the column header suggests {parsed}"
        return {"unit": user, "source": "user", "header": header, "note": note}
    if parsed is not None:
        return {"unit": parsed, "source": "header", "header": header,
                "note": "parsed from the target column header"}
    return {"unit": None, "source": "none", "header": header,
            "note": ("no target column" if header is None else
                     "the target column header states no recognized unit; "
                     "set one in the campaign setup")}


def anonymize_target_unit(block: dict[str, Any], alias: Callable[[str], str]) -> dict[str, Any]:
    """`block` under `anonymize=True`. When the target header is pseudonymized, the
    header-derived unit (a piece of the raw header) is dropped and `header` is the
    alias; a user-supplied unit is kept, minus any header-derived hint in the note.
    A header that is not pseudonymized is shown verbatim anyway, so its unit stays."""
    header = block.get("header")
    if not isinstance(header, str):
        return dict(block)
    aliased = alias(header)
    if aliased == header:
        return dict(block)
    out = {**block, "header": aliased}
    if out.get("source") == "user":
        out["note"] = "set by the user in the campaign setup"
    else:
        out.update(unit=None, source="none",
                   note="not derived: the target column header is anonymized")
    return out


__all__ = ["MAX_USER_UNIT_CHARS", "anonymize_target_unit", "parse_header_unit",
           "target_unit_block"]
