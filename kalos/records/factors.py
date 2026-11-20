"""Factors: a project's input definitions, mapped onto campaign setup bounds.

Voyager's ``Factor`` (name, description, lower/upper bound, data unit) defined
the search space of a formulation. In Kalos the search space of a campaign is its
SETUP (`kalos.portal.setup_settings`: goal, ``bounds: {column: {lo, hi}}``,
linear constraints, target unit). So a factor is a PROJECT-level definition that a
campaign's setup can be filled from (`setup_bounds`); it never changes a campaign
by itself, and the campaign setup keeps its own validation (strict ``lo < hi``,
feasibility against the data).

Changes from Voyager, each deliberate:

- bounds are real numbers, not integers (a pH of 6.8 or 0.5 g/L is common);
- ``lower < upper`` strictly, as the campaign setup requires (Voyager allowed
  equal bounds, which describe a constant, not a factor);
- the unit must be a registered unit (`kalos.normalize.unit_registry`), or none
  for a dimensionless count; an unknown unit is refused, never guessed;
- a range outside what is physically possible for the unit or the name
  (DO above 100 %, a negative concentration, pH outside 0-14) is refused, using
  the engine's physical-bounds table (`kalos.validation.bounds`, passed in by
  the caller as `physical_bounds`, so this module stays free of the table);
- a factor has a role: ``setting`` (proposed by the optimizer) or ``covariate``
  (measured, fitted, never proposed), the two roles of `ColumnRoles` a factor
  can be. Outputs and outcomes are not factors.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from kalos.normalize.unit_registry import lookup as lookup_unit
from kalos.records.projects import RecordsValidationError, clean_text
from kalos.storage.migrations.v9 import FACTOR_ROLES

MAX_FACTOR_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 300
MAX_FACTORS_PER_PROJECT = 64
ROLES: tuple[str, ...] = FACTOR_ROLES


@dataclass(frozen=True)
class PhysicalBounds:
    """What the physical-bounds table says about a (name, unit): values outside
    [hard_lo, hard_hi] are impossible. `source` is plain words for the reason."""

    hard_lo: float
    hard_hi: float
    source: str


# (factor name, canonical unit symbol or None) -> bounds or None (nothing known)
PhysicalBoundsFn = Callable[[str, str | None], PhysicalBounds | None]


@dataclass(frozen=True)
class FactorFields:
    name: str
    description: str | None
    lower: float
    upper: float
    unit: str | None
    dimension: str | None
    role: str


def check_factor_name(name: object) -> str:
    """A factor name as it appears as a run-sheet column: trimmed, 1..64
    characters, no control characters."""
    clean = clean_text(name, "factor name", MAX_FACTOR_NAME_CHARS, required=True)
    assert clean is not None
    return clean


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RecordsValidationError("invalid_bounds", f"{what} must be a finite number")
    return float(value)


def factor_fields(
    *, name: object, lower: object, upper: object, unit: object = None, description: object = None,
    role: object = "setting", physical_bounds: PhysicalBoundsFn | None = None,
) -> FactorFields:
    """Validate one factor definition. Raises `RecordsValidationError` with a
    stable code: invalid_body, invalid_bounds, unknown_unit, impossible_bounds,
    invalid_role."""
    clean_name = check_factor_name(name)
    lo, hi = _finite(lower, "lower_bound"), _finite(upper, "upper_bound")
    if not lo < hi:
        raise RecordsValidationError(
            "invalid_bounds", "lower_bound must be strictly less than upper_bound"
        )
    if role not in ROLES:
        raise RecordsValidationError("invalid_role", f"role must be one of: {', '.join(ROLES)}")
    symbol: str | None = None
    dimension: str | None = None
    if unit is not None and not (isinstance(unit, str) and not unit.strip()):
        registered = lookup_unit(unit)
        if registered is None:
            raise RecordsValidationError(
                "unknown_unit",
                "the unit is not in the units registry (GET /api/units lists them); "
                "leave it empty for a dimensionless factor",
            )
        symbol, dimension = registered.symbol, registered.dimension
    if physical_bounds is not None:
        pb = physical_bounds(clean_name, symbol)
        if pb is not None and (lo < pb.hard_lo or hi > pb.hard_hi):
            raise RecordsValidationError(
                "impossible_bounds",
                f"the range is outside what is physically possible ({pb.source}: "
                f"{_fmt(pb.hard_lo)} to {_fmt(pb.hard_hi)})",
            )
    return FactorFields(
        name=clean_name,
        description=clean_text(description, "description", MAX_DESCRIPTION_CHARS, required=False,
                               allow_newlines=True),
        lower=lo, upper=hi, unit=symbol, dimension=dimension, role=str(role),
    )


def _fmt(x: float) -> str:
    if math.isinf(x):
        return "no limit" if x > 0 else "no lower limit"
    return f"{x:g}"


def setup_bounds(
    factors: Iterable[dict[str, Any]], *, columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Map a project's factors onto a campaign setup's `bounds` (the body of
    `PUT /api/campaigns/{id}/setup` takes it as is). Only `setting` factors are
    proposed, so only they become bounds; covariates are listed as skipped.

    `columns`: the campaign's analysed columns when known. A factor that is not a
    column of the campaign is skipped with the reason, so the preview never
    produces a bound the setup would refuse. Nothing is written: applying the
    bounds is the caller's explicit `PUT .../setup`, which re-validates them."""
    known = set(columns) if columns is not None else None
    bounds: dict[str, dict[str, float]] = {}
    units: dict[str, str | None] = {}
    skipped: list[dict[str, str]] = []
    for f in factors:
        name = str(f["name"])
        if f.get("role", "setting") != "setting":
            skipped.append({"factor": name, "reason": "covariate: measured and fitted, never proposed"})
            continue
        if known is not None and name not in known:
            skipped.append({"factor": name, "reason": "not a continuous setting in this campaign's latest analysis"})
            continue
        bounds[name] = {"lo": float(f["lower_bound"]), "hi": float(f["upper_bound"])}
        units[name] = f.get("unit")
    return {
        "bounds": bounds,
        "units": units,
        "skipped": skipped,
        "columns_checked": known is not None,
        "applied": False,
        "how_to_apply": "send these bounds in PUT /api/campaigns/{id}/setup (write scope); "
                        "the setup validates them against the campaign's data",
    }


__all__ = [
    "MAX_FACTORS_PER_PROJECT", "ROLES", "FactorFields", "PhysicalBounds", "PhysicalBoundsFn",
    "check_factor_name", "factor_fields", "setup_bounds",
]
