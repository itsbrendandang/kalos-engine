"""Nominal working volumes for plate and flask formats, as labelled assumptions.

Screening sheets name the vessel format ("384w", "96w", "24w", "SF125")
instead of a volume in liters. The scale readout needs liters. A format label
does not say how much liquid was in the well or flask, so every volume taken
from this table is an ASSUMPTION, printed as one: the readout lists each format
it converted, the liters used, how many rows it applies to, and whether the
value came from this default table or from the user. It never converts a label
it does not know, and never converts silently.

Defaults (typical working volumes; override them with what the lab used):

| key | format | liters | note |
|---|---|---|---|
| 384w | 384-well plate | 5e-5 (50 uL) | typical 30-80 uL |
| 96w | 96-well plate (standard, not deep-well) | 2e-4 (200 uL) | typical 100-250 uL |
| 24w | 24-well deep-well plate | 2e-3 (2 mL) | a standard shallow 24-well plate holds about 1 mL |
| SF125 | 125 mL shake flask | 0.03 (30 mL) | typical 25-30 mL; 125 mL is the flask size, not the liquid |

Formats outside this table (96 deep-well, 6-well, 250 mL flasks, ...) are left
unresolved and reported by label, with the override that would resolve them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

SOURCE_DEFAULT = "assumption (default table)"
SOURCE_USER = "assumption (user-supplied)"

NOMINAL_WORKING_VOLUME_L: dict[str, float] = {
    "384w": 5e-5,
    "96w": 2e-4,
    "24w": 2e-3,
    "SF125": 0.03,
}
FORMAT_NOTES: dict[str, str] = {
    "384w": "384-well plate; typical working volume 30-80 uL",
    "96w": "96-well plate (standard, not deep-well); typical working volume 100-250 uL",
    "24w": "24-well DEEP-WELL plate; a standard shallow 24-well plate holds about 1 mL",
    "SF125": "125 mL shake flask; typical working volume 25-30 mL (125 mL is the flask size)",
}
# Ordered: the first pattern that matches the whole label wins. Deep-well 96
# plates and other formats are deliberately absent (unresolved, never guessed).
_FORMAT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^384\s*[-_ ]?\s*(w|wp|well|wells|well\s*plate|well[-_ ]plate)$"), "384w"),
    (re.compile(r"^96\s*[-_ ]?\s*(w|wp|well|wells|well\s*plate|well[-_ ]plate)$"), "96w"),
    (re.compile(r"^24\s*[-_ ]?\s*(w|wp|well|wells|dw|dwp|deep[-_ ]?well(\s*plate)?)$"), "24w"),
    (re.compile(r"^(sf|shake[-_ ]?flask)\s*[-_ ]?\s*125(\s*ml)?$"), "SF125"),
    (re.compile(r"^125\s*ml\s*(shake[-_ ]?)?flask$"), "SF125"),
)


def format_key(label: object) -> str | None:
    """The table key a vessel-format label refers to, or None if unknown."""
    text = str(label).strip().lower()
    for pattern, key in _FORMAT_PATTERNS:
        if pattern.match(text):
            return key
    return None


def _is_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class VolumeResolution:
    """`values`: the column with every resolved label replaced by liters (numbers
    and blanks unchanged, unresolved labels unchanged). `assumptions`: one entry
    per format used. `unresolved`: labels that are neither numbers nor known
    formats."""

    values: list[object]
    assumptions: dict[str, dict[str, Any]]
    unresolved: list[str]

    @property
    def used_defaults(self) -> bool:
        return any(a["source"] == SOURCE_DEFAULT for a in self.assumptions.values())


def resolve_scale_labels(values: list[object], overrides: dict[str, float] | None = None) -> VolumeResolution:
    """Replace vessel-format labels with nominal working volumes in liters.

    `overrides` maps a table key (`"SF125"`) or a label exactly as written in the
    sheet (`"SF250"`) to liters; a user value always wins over the default table.
    Raises `ValueError` for a non-positive or non-finite override.
    """
    user = {str(k).strip(): float(v) for k, v in (overrides or {}).items()}
    for k, v in user.items():
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"working volume override for '{k}' must be a positive number of liters, got {v}")
    user_lower = {k.lower(): v for k, v in user.items()}
    out: list[object] = []
    assumptions: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for raw in values:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == "" or _is_number(raw):
            out.append(raw)
            continue
        label = str(raw).strip()
        key = format_key(label)
        if label.lower() in user_lower:
            name, liters, source = label, user_lower[label.lower()], SOURCE_USER
        elif key is not None and key in user:
            name, liters, source = key, user[key], SOURCE_USER
        elif key is not None:
            name, liters, source = key, NOMINAL_WORKING_VOLUME_L[key], SOURCE_DEFAULT
        else:
            if label not in unresolved:
                unresolved.append(label)
            out.append(raw)
            continue
        entry = assumptions.setdefault(
            name,
            {
                "value": liters,
                "unit": "L",
                "source": source,
                "basis": "assumption",
                "note": FORMAT_NOTES.get(key or "", "user-supplied working volume"),
                "labels_as_written": [],
                "n_rows": 0,
            },
        )
        if label not in entry["labels_as_written"]:
            entry["labels_as_written"].append(label)
        entry["n_rows"] += 1
        out.append(liters)
    return VolumeResolution(values=out, assumptions=assumptions, unresolved=unresolved)


def assumption_sentence(assumptions: dict[str, dict[str, Any]]) -> str:
    """One plain sentence naming every assumed volume (for the readout's reasons)."""
    parts = [
        f"{name} = {a['value']:g} L ({a['source']}, {a['n_rows']} row(s))" for name, a in sorted(assumptions.items())
    ]
    return (
        "scale volumes for plate/flask formats are nominal working-volume assumptions, not measured: "
        + "; ".join(parts)
        + ". Pass working_volume_overrides with the volumes actually used."
    )


__all__ = [
    "FORMAT_NOTES",
    "NOMINAL_WORKING_VOLUME_L",
    "SOURCE_DEFAULT",
    "SOURCE_USER",
    "VolumeResolution",
    "assumption_sentence",
    "format_key",
    "resolve_scale_labels",
]
