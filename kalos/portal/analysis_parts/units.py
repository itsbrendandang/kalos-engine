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

import re
import unicodedata
from collections.abc import Callable
from typing import Any

MAX_USER_UNIT_CHARS = 32

# token -> canonical spelling. Case matters where it is meaningful (mM vs MM);
# a few common lowercase spellings of volume units are folded (ml -> mL).
_UNIT_ATOMS: dict[str, str] = {
    # mass
    "g": "g", "mg": "mg", "ug": "µg", "µg": "µg", "μg": "µg", "ng": "ng", "kg": "kg",
    # volume
    "L": "L", "l": "L", "mL": "mL", "ml": "mL", "uL": "µL", "ul": "µL", "µL": "µL",
    "μL": "µL", "µl": "µL",
    # amount / concentration
    "mol": "mol", "mmol": "mmol", "umol": "µmol", "µmol": "µmol",
    "M": "M", "mM": "mM", "uM": "µM", "µM": "µM", "μM": "µM", "nM": "nM",
    "mOsm": "mOsm",
    # time
    "h": "h", "hr": "h", "hrs": "h", "min": "min", "s": "s", "sec": "s", "d": "d",
    "day": "d", "days": "d",
    # temperature
    "°C": "°C", "degC": "°C", "C": "°C", "K": "K",
    # percent
    "%": "%", "pct": "%", "percent": "%",
    # misc bioprocess
    "cells": "cells", "U": "U", "IU": "IU", "rpm": "rpm", "vvm": "vvm",
    "Pa": "Pa", "kPa": "kPa", "bar": "bar", "mbar": "mbar",
}
# Compact two-unit tokens seen in run sheets ("titer_gL").
_COMPACT: dict[str, str] = {
    "gL": "g/L", "gl": "g/L", "mgL": "mg/L", "mgmL": "mg/mL", "mgml": "mg/mL",
    "ugmL": "µg/mL", "ugml": "µg/mL",
}
# Units accepted as a BARE trailing token (`yield_pct`, `glucose mM`). Ambiguous
# words are left out on purpose: `min` (minimum), `cells`, `day`, `bar`, and every
# one-letter unit.
_BARE_OK = frozenset({
    "pct", "percent", "%", "mL", "ml", "uL", "ul", "µL", "μL", "mg", "ug", "µg", "μg",
    "ng", "kg", "mM", "uM", "µM", "μM", "nM", "mmol", "umol", "µmol", "mOsm", "degC",
    "°C", "rpm", "vvm", "kPa", "mbar", "hr", "hrs", "IU",
})
_BRACKET_RE = re.compile(r"^(?P<name>.*\S)\s*[(\[]\s*(?P<unit>[^()\[\]]+?)\s*[)\]]\s*$")
_SCALE_RE = re.compile("^(?:[x\u00d7]\\s*)?" r"(?:10\^?-?\d{1,2}|1e-?\d{1,2})\s*")


def _atom(token: str) -> str | None:
    return _UNIT_ATOMS.get(token)


def _parse_expression(text: str) -> str | None:
    """A unit expression written by a person: optional power-of-ten scale, then
    unit atoms joined by `/` (e.g. `g/L`, `mg/L/h`, `10^6 cells/mL`). None if any
    atom is unknown."""
    text = text.strip()
    scale = ""
    m = _SCALE_RE.match(text)
    if m:
        scale = m.group(0).strip()
        text = text[m.end():].strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split("/")]
    atoms = [_atom(p) for p in parts]
    if any(a is None for a in atoms):
        if len(parts) == 1 and parts[0] in _COMPACT:
            return f"{scale} {_COMPACT[parts[0]]}".strip()
        return None
    return f"{scale} {'/'.join(a for a in atoms if a)}".strip()


def _parse_suffix(header: str) -> str | None:
    """`name_num_den` / `name_unit` / `name_num_per_den` suffixes."""
    tokens = header.split("_")
    if len(tokens) < 2:
        return None
    body = tokens[1:]  # the first token is always the quantity name
    # name_X_per_Y
    if len(body) >= 3 and body[-2].lower() == "per":
        num, den = _atom(body[-3]), _atom(body[-1])
        if num and den:
            return f"{num}/{den}"
    for k in (3, 2):  # ratio chains: titer_g_L, rate_mg_L_h
        if len(body) >= k:
            atoms = [_atom(t) for t in body[-k:]]
            if all(atoms):
                return "/".join(a for a in atoms if a)
    last = body[-1]
    if last in _COMPACT:
        return _COMPACT[last]
    if last in _BARE_OK:
        return _atom(last)
    return None


def parse_header_unit(header: str) -> str | None:
    """The unit a target header states, or None when it states none we recognize."""
    text = unicodedata.normalize("NFC", header).strip()
    if not text:
        return None
    m = _BRACKET_RE.match(text)
    if m:
        return _parse_expression(m.group("unit"))
    if " " in text:
        last = text.rsplit(None, 1)[1]
        # A bare word after a space must be an unambiguous unit ("titer A" and
        # "temp min" are names); a `/` expression is explicit.
        if "/" in last or last in _BARE_OK or last in _COMPACT:
            parsed = _parse_expression(last)
            if parsed is not None:
                return parsed
    return _parse_suffix(text)


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
