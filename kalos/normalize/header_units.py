"""Units stated in a column HEADER (`titer_g_L`, `DO (%)`, `glucose mM`).

Moved here from `kalos.portal.analysis_parts.units` (which re-exports
`parse_header_unit`) so the torch-free validation layer can read a header's unit
for its physical bounds without importing the portal.

Only well-defined patterns are read: a trailing `(unit)` / `[unit]`, a trailing
unit token after a space, or a trailing `_unit` / `_num_den` suffix whose every
token is a known unit. An unrecognized suffix is NOT guessed at: `od600`,
`yield_A`, `titer (replicate 2)` all give None. Bare one-letter suffixes (`_L`,
`_g`, `_C`, `_M`, `_h`) are refused on their own because they collide with
vessel labels ("5L"), letters and initials; they are accepted inside an explicit
ratio (`titer_g_L`) or inside brackets (`titer (g)`), where the intent is clear.
"""
from __future__ import annotations

import re
import unicodedata

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
    "ugmL": "µg/mL", "ugml": "µg/mL", "mLh": "mL/h", "mlh": "mL/h", "mLmin": "mL/min",
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


def unit_atoms(unit: str) -> list[str]:
    """The canonical atoms of a unit returned by `parse_header_unit`, without a
    power-of-ten scale: "10^6 cells/mL" -> ["cells", "mL"], "g/L" -> ["g", "L"]."""
    text = _SCALE_RE.sub("", unit.strip()).strip()
    return [p.strip() for p in text.split("/") if p.strip()]


__all__ = ["parse_header_unit", "unit_atoms"]
