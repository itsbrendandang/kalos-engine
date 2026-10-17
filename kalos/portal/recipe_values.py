"""Type-check a started recipe's values against the campaign's own dataset.

`POST /api/campaign/start` bounds each recipe value's SHAPE (a finite JSON
number, a string of at most 512 characters, or null; see
`kalos.portal.campaign_routes`), but a recipe is folded back into the tenant's
persisted dataset as `{**recipe, target: result}`, so its values also have to
fit the COLUMN they land in. The column's kind is read from the campaign's
`base_rows` (the uploaded sheet), the only record of the dataset the store has:

- **numeric** column (every observed value is a finite number): the value must
  be a finite JSON number. A string is refused, including a numeric-looking one
  (`"7.3"`), with one exception: a string that is EXACTLY the label of an
  observed value (`"2"` for an observed `2`) is accepted and stored as that
  number. That is the shape the engine's own proposals take for a numeric
  column declared categorical through `roles` (`DesignSpace.decode_row`
  returns level labels, which are `str(value)`), and storing the number keeps
  the column's type intact.
- **categorical** column (any observed value is a string): the value must be a
  string naming one of the column's observed levels (same labelling as the
  analysis: `str(value).strip()` of the non-blank cells). An unknown level is
  refused: the model has never seen it, so it cannot be a proposal, and storing
  it would quietly add a level to the dataset. `null` is accepted (the engine
  emits it for a categorical with no levels).
- a column with no observed value at all accepts only `null`.

Messages never echo the caller's key or value.
Torch- and FastAPI-free, so the store can import it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

NUMERIC = "numeric"
CATEGORICAL = "categorical"
EMPTY = "empty"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _label(value: Any) -> str:
    return str(value).strip()


@dataclass
class ColumnKind:
    kind: str
    # label -> the observed value it came from (numeric columns: the number)
    levels: dict[str, Any] = field(default_factory=dict)


def column_kinds(
    base_rows: Iterable[Mapping[str, Any]], columns: Iterable[str]
) -> dict[str, ColumnKind]:
    """The kind and observed levels of each of `columns` over `base_rows`."""
    wanted = set(columns)
    seen_str: set[str] = set()
    levels: dict[str, dict[str, Any]] = {c: {} for c in wanted}
    for row in base_rows:
        for col in wanted:
            value = row.get(col)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            if isinstance(value, str):
                if not value.strip():
                    continue
                seen_str.add(col)
            levels[col].setdefault(_label(value), value)
    out: dict[str, ColumnKind] = {}
    for col in wanted:
        if not levels[col]:
            out[col] = ColumnKind(EMPTY)
        elif col in seen_str or not all(_is_number(v) for v in levels[col].values()):
            out[col] = ColumnKind(CATEGORICAL, levels[col])
        else:
            out[col] = ColumnKind(NUMERIC, levels[col])
    return out


def check_recipe_values(
    recipe: Mapping[str, Any], kinds: Mapping[str, ColumnKind]
) -> tuple[dict[str, Any], str | None]:
    """`(recipe with any accepted label coerced to its number, problem)`;
    `problem` is None when every value fits its column, else a safe message."""
    out: dict[str, Any] = {}
    bad_numeric = bad_level = bad_empty = 0
    for key, value in recipe.items():
        kind = kinds.get(key)
        if kind is None:  # not a dataset column; the key check reports it
            out[key] = value
            continue
        if kind.kind == NUMERIC:
            if _is_number(value) and math.isfinite(float(value)):
                out[key] = value
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
