"""Where a row of data came from: `measured`, `simulated` or `synthetic`.

Origin is a hard type in Kalos, not a free-text tag (docs/VIRTUAL_RUNS.md):

- `measured`: entered or uploaded from the lab, with an attributable user. This
  is the only origin the analysis and the models accept by default.
- `simulated`: output of a calibrated process model (twin) version for a design.
  Phase 1 has no calibrated process model, so no generator produces this yet; the
  value exists so the storage, the export and the guard already handle it.
- `synthetic`: output of a generator that is not tied to the lab's process model
  (benchmark surfaces, demo datasets, test fixtures).

When data of several origins is combined, the result carries the "highest" one
(`synthetic` > `simulated` > `measured`, see `highest_origin`).

Virtual (non-measured) data is marked in two ways, and `check_origin` reads both:

- a DataFrame column named `origin` (case and surrounding spaces ignored) whose
  cells say `simulated` or `synthetic`. The same column is written into every
  virtual export, so a re-uploaded export is refused too;
- `DataFrame.attrs["kalos_origin"]`, or a `kalos_origin` attribute on any other
  object (for example `OriginArray`, the ndarray subclass `VirtualBatch.X`/`.y`
  return).

Limit, on purpose and documented: a caller who converts a marked frame to a plain
`numpy` array (`df.to_numpy()`, `np.asarray(arr)`, `np.vstack`) or renames the
`origin` column removes the marker. The guard stops accidents and every path the
service exposes; it cannot stop deliberate laundering inside a Python process.

Torch-free and portal-free: `kalos.core` imports this module.
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

ORIGIN_COLUMN = "origin"
ORIGIN_ATTR = "kalos_origin"


class Origin(StrEnum):
    MEASURED = "measured"
    SIMULATED = "simulated"
    SYNTHETIC = "synthetic"


VIRTUAL_ORIGINS: frozenset[Origin] = frozenset({Origin.SIMULATED, Origin.SYNTHETIC})
_RANK = {Origin.MEASURED: 0, Origin.SIMULATED: 1, Origin.SYNTHETIC: 2}


class VirtualDataRefused(ValueError):
    """A fit or analysis entry point was handed simulated or synthetic rows
    without an explicit `allow_origins=`. Nothing was fitted. The message names
    the entry point and the origins found; it never quotes data."""

    code = "virtual_data_refused"

    def __init__(self, where: str, found: Iterable[Origin]) -> None:
        names = ", ".join(sorted(str(o) for o in found))
        super().__init__(
            f"{where} refuses non-measured data (found origin: {names}). Simulated and "
            "synthetic rows are kept apart from lab measurements and are not used to fit "
            "models or analyse a campaign. Pass allow_origins= explicitly to use them on "
            "purpose (see docs/VIRTUAL_RUNS.md)."
        )
        self.where = where
        self.found = frozenset(found)


def parse_origin(value: object) -> Origin:
    """`Origin` from an `Origin` or its string value (case and spaces ignored).
    Anything else raises ValueError."""
    if isinstance(value, Origin):
        return value
    if isinstance(value, str):
        try:
            return Origin(value.strip().lower())
        except ValueError:
            pass
    raise ValueError(
        f"unknown origin; expected one of {', '.join(o.value for o in Origin)}"
    )


def highest_origin(origins: Iterable[Origin | str]) -> Origin:
    """The taint of combined data: synthetic > simulated > measured. An empty
    input is `measured` (nothing virtual was read)."""
    best = Origin.MEASURED
    for o in origins:
        po = parse_origin(o)
        if _RANK[po] > _RANK[best]:
            best = po
    return best


class OriginArray(np.ndarray):
    """A float ndarray that remembers its origin (`kalos_origin`). Slicing and
    elementwise arithmetic keep it; `np.asarray`, `np.vstack` and friends return a
    plain ndarray, which is the documented limit in this module's docstring."""

    kalos_origin: Origin

    def __new__(cls, data: Any, origin: Origin | str) -> OriginArray:
        obj = np.asarray(data, dtype=float).view(cls)
        obj.kalos_origin = parse_origin(origin)
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is not None:
            self.kalos_origin = getattr(obj, "kalos_origin", Origin.MEASURED)

    def __reduce__(self) -> Any:  # keep the origin through pickle / copy
        ctor, args, state = super().__reduce__()  # type: ignore[misc]
        return ctor, args, (state, str(self.kalos_origin))

    def __setstate__(self, state: Any) -> None:
        inner, origin = state
        super().__setstate__(inner)
        self.kalos_origin = parse_origin(origin)


def _is_origin_column(name: object) -> bool:
    return isinstance(name, str) and name.strip().lower() in (ORIGIN_COLUMN, ORIGIN_ATTR)


def _origins_in_values(values: Iterable[object]) -> set[Origin]:
    found: set[Origin] = set()
    for v in values:
        if isinstance(v, str):
            key = v.strip().lower()
            for o in VIRTUAL_ORIGINS:
                if key == o.value:
                    found.add(o)
    return found


def origins_of(obj: object) -> set[Origin]:
    """The non-measured origins `obj` is marked with (empty = treated as measured).
    Reads DataFrame/Series attrs and `origin` columns, and a `kalos_origin`
    attribute on anything else. Mappings (a recipe dict) are read by key."""
    found: set[Origin] = set()
    if obj is None:
        return found
    attrs = getattr(obj, "attrs", None)
    if isinstance(attrs, dict) and attrs.get(ORIGIN_ATTR) is not None:
        found |= _origins_in_values([str(attrs[ORIGIN_ATTR])])
    marker = getattr(obj, ORIGIN_ATTR, None)
    if marker is not None and not isinstance(obj, (pd.DataFrame, pd.Series)):
        found |= _origins_in_values([str(marker)])
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            if _is_origin_column(col):
                found |= _origins_in_values(obj[col].dropna().unique().tolist())
    elif isinstance(obj, pd.Series):
        if _is_origin_column(obj.name):
            found |= _origins_in_values(obj.dropna().unique().tolist())
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if _is_origin_column(k):
                found |= _origins_in_values([v])
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, dict):
                found |= origins_of(item)
    return found


def check_origin(
    *data: object,
    where: str,
    allow_origins: Iterable[Origin | str] | None = None,
) -> Origin:
    """Refuse non-measured data at a fit or analysis entry point.

    Returns the taint of the inputs (`highest_origin`) when they are allowed:
    `measured` for ordinary data. Raises `VirtualDataRefused` when any input is
    marked simulated/synthetic and that origin is not in `allow_origins`.
    `allow_origins=None` (the default) allows measured data only."""
    allowed = {parse_origin(o) for o in (allow_origins or ())}
    found: set[Origin] = set()
    for d in data:
        found |= origins_of(d)
    refused = found - allowed
    if refused:
        raise VirtualDataRefused(where, refused)
    return highest_origin(found)


__all__ = [
    "ORIGIN_ATTR",
    "ORIGIN_COLUMN",
    "VIRTUAL_ORIGINS",
    "Origin",
    "OriginArray",
    "VirtualDataRefused",
    "check_origin",
    "highest_origin",
    "origins_of",
    "parse_origin",
]
