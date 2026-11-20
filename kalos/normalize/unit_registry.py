"""The units registry: which units a factor, a target or a stage may DECLARE.

Ported from Voyager's ``DataUnit`` table (a writable list of unit names). In
Kalos the list is code, not data: a unit is a fact about physics, it is the same
for every tenant, and a typo in a shared table would silently change every
factor that used it. So the registry is read-only over HTTP (`GET /api/units`).

One table, three views, so the unit vocabularies already in the engine stay
consistent instead of growing a fourth:

- ``symbol``: the canonical DISPLAY spelling. It is the spelling a header parser
  gives (`kalos.normalize.header_units.parse_header_unit` on branch
  feat/data-checks: 'g/L', '°C', 'µM', '10^6 cells/mL'), so a factor declared in
  'g/L' and a column headed ``titer_g_L`` name the same unit.
- ``dimension``: what it measures, in the words of `kalos.validation.bounds`
  (``concentration``, ``temperature``, ``percent``, ``time``, ...), so the
  physical-bounds table can judge a declared range.
- ``cell_token``: the token `kalos.normalize.units` converts in a messy CELL
  ("34.6 C"), or None when that module deliberately does not convert the unit
  (volume, for example: "5L" is a vessel label there). Conversion formulas live
  ONLY in `kalos.normalize.units`; this module never converts a number.

`lookup` canonicalizes what a person typed ("mg/ml", "ug/mL", "degC", "pct",
"hr") to one registered unit, or None: an unknown unit is refused with a reason
by the caller, never guessed. tests/test_unit_registry.py checks this table
against `kalos.normalize.units` (and against `header_units` when present), so
the vocabularies cannot drift apart silently.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisteredUnit:
    symbol: str
    dimension: str
    name: str
    aliases: tuple[str, ...] = ()
    cell_token: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol, "dimension": self.dimension, "name": self.name,
            "aliases": list(self.aliases), "converts_in_cells": self.cell_token is not None,
        }


UNITS: tuple[RegisteredUnit, ...] = (
    # concentration
    RegisteredUnit("g/L", "concentration", "grams per litre", ("g/l", "gL", "gl"), "g/l"),
    RegisteredUnit("mg/mL", "concentration", "milligrams per millilitre", ("mg/ml", "mgmL", "mgml"), "mg/ml"),
    RegisteredUnit("mg/L", "concentration", "milligrams per litre", ("mg/l", "mgL")),
    RegisteredUnit("µg/mL", "concentration", "micrograms per millilitre", ("ug/mL", "ug/ml", "μg/mL", "ugmL")),
    RegisteredUnit("M", "concentration", "molar"),
    RegisteredUnit("mM", "concentration", "millimolar"),
    RegisteredUnit("µM", "concentration", "micromolar", ("uM", "μM")),
    RegisteredUnit("nM", "concentration", "nanomolar"),
    RegisteredUnit("mOsm/kg", "osmolality", "milliosmoles per kilogram"),
    RegisteredUnit("mOsm", "osmolality", "milliosmoles (as written on the sheet; usually per kg)"),
    RegisteredUnit("mOsm/L", "osmolarity", "milliosmoles per litre"),
    RegisteredUnit("U/mL", "activity concentration", "enzyme units per millilitre", ("U/ml",)),
    # cell density
    RegisteredUnit("10^6 cells/mL", "cell density", "million cells per millilitre",
                   ("1e6 cells/mL", "x10^6 cells/mL", "10^6 cells/ml", "e6 cells/mL")),
    # temperature
    RegisteredUnit("°C", "temperature", "degrees Celsius", ("degC", "C", "celsius"), "c"),
    RegisteredUnit("K", "temperature", "kelvin", ("kelvin",), "k"),
    # percent (share of a whole or relative: the bounds table tells them apart by name)
    RegisteredUnit("%", "percent", "percent", ("pct", "percent"), "%"),
    # time
    RegisteredUnit("h", "time", "hours", ("hr", "hrs", "hour", "hours"), "h"),
    RegisteredUnit("min", "time", "minutes", ("minute", "minutes"), "min"),
    RegisteredUnit("d", "time", "days", ("day", "days")),
    RegisteredUnit("s", "time", "seconds", ("sec", "second", "seconds")),
    # agitation, aeration, flow
    RegisteredUnit("rpm", "agitation", "revolutions per minute", (), "rpm"),
    RegisteredUnit("vvm", "gas flow per volume", "gas volume per liquid volume per minute"),
    RegisteredUnit("mL/h", "flow rate", "millilitres per hour", ("ml/h", "mLh", "mlh"), "ml/h"),
    RegisteredUnit("L/h", "flow rate", "litres per hour", ("l/h",), "l/h"),
    RegisteredUnit("mL/min", "flow rate", "millilitres per minute", ("ml/min", "mLmin"), "ml/min"),
    RegisteredUnit("L/min", "flow rate", "litres per minute", ("l/min",), "l/min"),
    # pressure
    RegisteredUnit("bar", "pressure", "bar", (), "bar"),
    RegisteredUnit("mbar", "pressure", "millibar"),
    RegisteredUnit("kPa", "pressure", "kilopascal"),
    RegisteredUnit("Pa", "pressure", "pascal"),
    # volume (a scale; never converted inside a cell, see kalos/normalize/units.py)
    RegisteredUnit("L", "volume", "litres", ("l", "litre", "liter")),
    RegisteredUnit("mL", "volume", "millilitres", ("ml",)),
    RegisteredUnit("µL", "volume", "microlitres", ("uL", "ul", "μL")),
    # mass
    RegisteredUnit("g", "mass", "grams", (), "g"),
    RegisteredUnit("kg", "mass", "kilograms", (), "kg"),
    RegisteredUnit("mg", "mass", "milligrams"),
    RegisteredUnit("µg", "mass", "micrograms", ("ug", "μg")),
    RegisteredUnit("ng", "mass", "nanograms"),
    # amount of substance, activity, counts
    RegisteredUnit("mol", "amount", "moles"),
    RegisteredUnit("mmol", "amount", "millimoles"),
    RegisteredUnit("µmol", "amount", "micromoles", ("umol", "μmol")),
    RegisteredUnit("U", "activity", "enzyme units"),
    RegisteredUnit("IU", "activity", "international units"),
    RegisteredUnit("cells", "count", "cells"),
    # specific productivity and yield rates
    RegisteredUnit("pg/cell/day", "specific productivity", "picograms per cell per day", ("pcd",)),
    RegisteredUnit("g/L/d", "volumetric productivity", "grams per litre per day", ("g/L/day",)),
    # dimensionless scales: pH and optical density have no unit symbol
    RegisteredUnit("pH", "ph", "pH units (dimensionless)", ("ph",), "ph"),
    RegisteredUnit("OD", "od", "optical density (dimensionless)", ("od", "OD600", "od600")),
)

_BY_KEY: dict[str, RegisteredUnit] = {}


def _key(text: str) -> str:
    """NFC, the micro sign and Greek mu folded to one character, spaces trimmed."""
    t = unicodedata.normalize("NFC", text).strip().replace("\u03bc", "\u00b5")
    return " ".join(t.split())


for _u in UNITS:
    for _spelling in (_u.symbol, *_u.aliases):
        _k = _key(_spelling)
        if _k in _BY_KEY and _BY_KEY[_k] is not _u:  # pragma: no cover - table invariant
            raise RuntimeError(f"unit spelling {_spelling!r} is registered twice")
        _BY_KEY[_k] = _u

MAX_UNIT_CHARS = 32


def lookup(text: object) -> RegisteredUnit | None:
    """The registered unit `text` spells (case kept where it matters: 'M' is
    molar, 'm' is not a unit here), or None."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_UNIT_CHARS:
        return None
    return _BY_KEY.get(_key(text))


def all_units() -> list[RegisteredUnit]:
    return list(UNITS)


def dimensions() -> list[str]:
    return sorted({u.dimension for u in UNITS})


__all__ = ["MAX_UNIT_CHARS", "UNITS", "RegisteredUnit", "all_units", "dimensions", "lookup"]
