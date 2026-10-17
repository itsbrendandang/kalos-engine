"""Type-check a started recipe's values against the campaign's own dataset.

`POST /api/campaign/start` bounds each recipe value's SHAPE (a finite JSON
number, a string of at most 512 characters, or null; see
`kalos.portal.campaign_routes`), but a recipe is folded back into the tenant's
persisted dataset as `{**recipe, target: result}`, so its values also have to
fit the COLUMN they land in. The column's kind is read from the campaign's
`base_rows` (the uploaded sheet) with THE ANALYSIS'S OWN TYPING, so a value the
engine itself proposed is never refused:

- a column is **numeric** when the analysis would treat it as numeric:
  single-unit columns are converted first (`"34.6 C"` -> 34.6, exactly as
  `_analyze` does via `check_units_consistency` + `apply_unit_conversions`),
  then `kalos.portal.uploads.numeric_columns` (at least 3 non-blank cells,
  >= 80% of them numbers). A pH column with one `"ND"` cell is numeric. A
  column with fewer than 3 values is numeric when every value is a number.
- every other column with values is **categorical**; a column with no value
  at all is **empty**.

Accepted values:

- numeric column: a finite JSON number within `MAGNITUDE_FACTOR` x the
  column's observed span of the observed range (a sanity bound against
  `1e300`-style junk, not a design-box check: the engine's own proposals stay
  inside the observed box). A string is refused, including a numeric-looking
  one (`"7.3"`), except a string that is EXACTLY the label of an observed cell
  (`"200"` for an observed `200`): that is how the engine proposes a numeric
  column declared categorical through `roles` (`DesignSpace.decode_row`
  returns `str(value)` labels), and it is stored as the observed cell.
- categorical column: a string naming one of the observed levels
  (`str(value).strip()` of the non-blank cells, the analysis's labelling), or
  null (emitted for a categorical with no levels). New levels are refused.
- empty column: only null.

A campaign holds at most `MAX_PENDING_RUNS` runs in flight (started, or
measured but not yet re-analyzed); `pending_capacity_problem` says so.

Messages never echo the caller's key or value.
Torch- and FastAPI-free, so the store can import it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

NUMERIC = "numeric"
CATEGORICAL = "categorical"
EMPTY = "empty"

MAGNITUDE_FACTOR = 1e6
# A numeric recipe value must lie within MAGNITUDE_FACTOR x span of the
# column's observed [min, max], span = max(max - min, |min|, |max|, 1). Six
# orders of magnitude is far beyond any unit slip a scientist could intend
# (mg vs kg is 1e6), while still refusing values that would wreck the column's
# scale for every later fit.

MAX_PENDING_RUNS = 256
# Runs in flight per campaign (per tenant). Each one is a row the next
# re-analysis folds in and the acquisition treats as pending; 256 is many
# rounds of a 96-well plate, and it bounds the persisted state a caller can grow
# without ever recording a result.


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _label(value: Any) -> str:
    return str(value).strip()


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


@dataclass
class ColumnKind:
    kind: str
    # label -> the observed cell it came from
    levels: dict[str, Any] = field(default_factory=dict)
    # numeric columns: the observed numeric range (after unit conversion)
    low: float | None = None
    high: float | None = None

    def admits_magnitude(self, value: float) -> bool:
        if self.low is None or self.high is None:
            return True
        span = max(self.high - self.low, abs(self.low), abs(self.high), 1.0)
        margin = MAGNITUDE_FACTOR * span
        return self.low - margin <= value <= self.high + margin


def column_kinds(
    base_rows: Iterable[Mapping[str, Any]], columns: Iterable[str]
) -> dict[str, ColumnKind]:
    """The kind, observed levels and numeric range of each of `columns`."""
    # local imports: keep this module import-light for the store
    from kalos.portal.uploads import numeric_columns
    from kalos.validation import apply_unit_conversions
    from kalos.validation.checks import check_units_consistency

    wanted = sorted(set(columns))
    rows = list(base_rows)
    frame = pd.DataFrame(
        {c: [row.get(c) for row in rows] for c in wanted}, columns=wanted, dtype=object
    )
    _, conversions = check_units_consistency(frame)
    converted = apply_unit_conversions(frame, conversions) if conversions else frame
    numeric = set(numeric_columns(converted))

    out: dict[str, ColumnKind] = {}
    for col in wanted:
        cells = [v for v in frame[col].tolist() if not _blank(v)]
        if not cells:
            out[col] = ColumnKind(EMPTY)
            continue
        levels: dict[str, Any] = {}
        for v in cells:
            levels.setdefault(_label(v), v)
        is_numeric = col in numeric or (len(cells) < 3 and all(_is_number(v) for v in cells))
        if not is_numeric:
            out[col] = ColumnKind(CATEGORICAL, levels)
            continue
        values = pd.to_numeric(converted[col], errors="coerce").to_numpy(dtype=float)
        values = values[pd.notna(values) & (values != float("inf")) & (values != float("-inf"))]
        low = float(values.min()) if values.size else None
        high = float(values.max()) if values.size else None
        out[col] = ColumnKind(NUMERIC, levels, low, high)
    return out


def pending_capacity_problem(n_pending: int, n_new: int) -> str | None:
    """A safe message when starting `n_new` more runs would exceed
    `MAX_PENDING_RUNS` in flight, else None."""
    if n_pending + n_new <= MAX_PENDING_RUNS:
        return None
    return (
        f"this campaign already has {n_pending} run(s) in flight and at most "
        f"{MAX_PENDING_RUNS} are allowed; log results and re-analyze before starting "
        f"{n_new} more"
    )


def check_recipe_values(
    recipe: Mapping[str, Any], kinds: Mapping[str, ColumnKind]
) -> tuple[dict[str, Any], str | None]:
    """`(recipe with any accepted label replaced by its observed cell, problem)`;
    `problem` is None when every value fits its column, else a safe message."""
    out: dict[str, Any] = {}
    bad_numeric = bad_magnitude = bad_level = bad_empty = 0
    for key, value in recipe.items():
        kind = kinds.get(key)
        if kind is None:  # not a dataset column; the key check reports it
            out[key] = value
            continue
        if kind.kind == NUMERIC:
            if _is_number(value) and math.isfinite(float(value)):
                if kind.admits_magnitude(float(value)):
                    out[key] = value
                else:
                    bad_magnitude += 1
            elif isinstance(value, str) and _label(value) == value and value in kind.levels:
                out[key] = kind.levels[value]
            else:
                bad_numeric += 1
        elif kind.kind == CATEGORICAL:
            if value is None or (isinstance(value, str) and _label(value) in kind.levels):
                out[key] = value if value is None else _label(value)
            else:
                bad_level += 1
        else:
            if value is None:
                out[key] = value
            else:
                bad_empty += 1
    problems = []
    if bad_numeric:
        problems.append(
            f"{bad_numeric} value(s) for numeric columns are not finite JSON numbers "
            "(numeric strings are refused)"
        )
    if bad_magnitude:
        problems.append(
            f"{bad_magnitude} value(s) for numeric columns are more than "
            f"{MAGNITUDE_FACTOR:.0e} times the column's observed span away from its "
            "observed range"
        )
    if bad_level:
        problems.append(
            f"{bad_level} value(s) for categorical columns are not one of that column's "
            "levels in this campaign's dataset (new levels are refused; upload a sheet "
            "that contains the level to add it)"
        )
    if bad_empty:
        problems.append(
            f"{bad_empty} value(s) are set for columns that have no values in this "
            "campaign's dataset (only null is accepted)"
        )
    if problems:
        return out, "recipe has " + "; ".join(problems)
    return out, None
