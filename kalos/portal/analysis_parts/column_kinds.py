"""What kind of data each uploaded column holds, and what the engine did with it.

`columns` in the analysis response (docs/CAMPAIGNS_API.md, "Column kinds"): one
entry per uploaded column, in sheet order, so a client never has to guess
whether a column is a number or a label (a text input such as CarbonSource =
src_14 must not be drawn on a numeric axis).

`kind` describes the values:
  numeric       numbers that vary (the target is numeric too)
  category      text labels that repeat (a medium, a carbon source, a recipe label)
  text_id       text (or an identifier-named column) with about one value per row
  row_number    a headerless index column, or a near-dense up/down counter
  outcome_like  something measured during or after the run (not an input)
  constant      one value, or no values, on the rows the model is fit on
  date          dates or times (never binned as numbers)

`role` says what the engine did with it: target, input, categorical_input,
group (a recipe label that groups replicates), fixed (a near-constant input
held at its median), excluded. `used_in_fit` is true for target, input and
categorical_input.

A category column is fitted as a categorical input when it has at most
AUTO_CATEGORICAL_MAX_LEVELS levels or was declared categorical in `roles`; the
GP is then a mixed categorical + continuous model and proposals carry the label
text. A category with more levels that was not declared is excluded, with the
finding code `category_not_modeled`: never silently, and never as numbers.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from kalos.portal.analysis_parts.roles import REASON_CODES
from kalos.portal.cells import NON_NUMERIC, classify_cells
from kalos.validation.report import Finding

# Levels listed per category column; `n_levels` always has the full count.
MAX_LEVELS_LISTED = 50
# Text with at least this share of distinct values among its cells is an id.
TEXT_ID_DISTINCT_SHARE = 0.9

_ROW_NUMBER_REASONS = {"headerless", "monotone_row_number", "unique_per_row_integer"}

# The closed list of `columns[].reason` codes (docs/CAMPAIGNS_API.md, "Column
# kinds"): the provenance reasons plus the ones this module adds.
COLUMN_KIND_REASONS: tuple[str, ...] = (
    *REASON_CODES,
    "too_many_levels", "mixed_text_and_numbers", "too_few_repeats",
    "unique_per_row", "date_or_time",
)
FINDING_CODES: tuple[str, ...] = (
    "category_not_modeled", "possibly_measured_after_run", "outcome_like_name",
    "after_target_in_sheet", "row_number_column", "headerless_column",
)
# The finding each role-safety reason comes with (kalos/portal/analysis_parts/roles.py),
# so a client can show the column unticked with its reason and its finding.
_REASON_FINDING = {
    "possibly_measured": "possibly_measured_after_run",
    "outcome_like_name": "outcome_like_name",
    "after_target_in_sheet": "after_target_in_sheet",
    "monotone_row_number": "row_number_column",
    "headerless": "row_number_column",
}


def _looks_like_dates(series: pd.Series, text_cells: int) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if text_cells == 0:
        return False
    labels = series[classify_cells(series) == NON_NUMERIC].astype(str).str.strip()
    if not labels.str.contains(r"\d", regex=True).all():
        return False
    if not labels.str.contains(r"[-/:.T ]", regex=True).all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(labels, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.8)
_CONSTANT_STATUSES = {"dropped_constant", "dropped_constant_on_fitted_rows", "dropped_all_blank"}


def _levels(series: pd.Series) -> list[dict[str, Any]]:
    kinds = classify_cells(series)
    labels = series[kinds == NON_NUMERIC].astype(str).str.strip()
    counts = labels.value_counts()
    rows = sorted(((str(k), int(v)) for k, v in counts.items()), key=lambda kv: (-kv[1], kv[0]))
    return [{"label": k, "count": v} for k, v in rows]


def column_kinds(
    df: pd.DataFrame,
    provenance: Iterable[Mapping[str, Any]],
    *,
    kept_cont: Iterable[str],
    kept_cats: Iterable[str],
    max_auto_levels: int,
    declared_cats: Iterable[str] = (),
    is_identifier: Any = None,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    """The `columns` block and the findings for excluded category columns."""
    prov = {str(row["name"]): row for row in provenance}
    cont = {str(c) for c in kept_cont}
    cats = {str(c) for c in kept_cats}
    declared = {str(c) for c in declared_cats}
    out: list[dict[str, Any]] = []
    findings: list[Finding] = []
    for col in df.columns:
        name = str(col)
        row = prov.get(name, {})
        status = str(row.get("status", ""))
        reason = str(row.get("reason") or "")
        series = df[col]
        n_values = int((series.notna() & (series.astype(str).str.strip() != "")).sum())
        entry: dict[str, Any] = {
            "name": name, "kind": "numeric", "role": "excluded", "used_in_fit": False,
            "n_values": n_values, "n_levels": None, "levels": None,
            "finding_code": None, "reason": reason or None,
        }
        text_cells = int((classify_cells(series) == NON_NUMERIC).sum())
        is_text = n_values > 0 and text_cells >= 0.5 * n_values
        if status == "target":
            entry.update(kind="numeric", role="target", used_in_fit=True)
        elif name not in cont and name not in cats and _looks_like_dates(series, text_cells):
            entry.update(kind="date", reason="date_or_time")
        elif status == "fixed_near_constant":
            entry.update(kind="numeric", role="fixed")
        elif reason in _ROW_NUMBER_REASONS and not is_text:
            entry.update(kind="row_number")
        elif reason == "headerless" and is_text:
            # a text column with no header: listed with its labels, never an input
            levels = _levels(series)
            entry.update(
                kind="category" if len(levels) < TEXT_ID_DISTINCT_SHARE * max(text_cells, 1)
                else "text_id",
                n_levels=len(levels), levels=levels[:MAX_LEVELS_LISTED],
            )
            if entry["kind"] == "text_id":
                entry["levels"] = None
        elif status == "dropped_output":
            entry.update(kind="outcome_like")
        elif status in _CONSTANT_STATUSES:
            entry.update(kind="constant")
        elif name in cats or status == "dropped_recipe_label":
            levels = _levels(series)
            entry.update(
                kind="category", n_levels=len(levels), levels=levels[:MAX_LEVELS_LISTED],
                role="categorical_input" if name in cats else "group",
                used_in_fit=name in cats,
            )
        elif name in cont:
            entry.update(kind="numeric", role="input", used_in_fit=True)
        elif is_text:
            levels = _levels(series)
            n_levels = len(levels)
            identifier = bool(is_identifier and is_identifier(name.strip()))
            if n_levels <= 1:
                entry.update(kind="constant")
            elif identifier or n_levels >= TEXT_ID_DISTINCT_SHARE * max(text_cells, 1):
                entry.update(
                    kind="text_id", n_levels=n_levels,
                    reason="name_matches_identifier" if identifier else "unique_per_row",
                )
            else:
                why = (
                    "too_many_levels" if n_levels > max_auto_levels and name not in declared
                    else "mixed_text_and_numbers" if text_cells < n_values
                    else "too_few_repeats"
                )
                entry.update(
                    kind="category", n_levels=n_levels, levels=levels[:MAX_LEVELS_LISTED],
                    finding_code="category_not_modeled", reason=why,
                )
                detail = {"n_levels": n_levels, "max_auto_levels": max_auto_levels, "why": why}
                if why == "too_many_levels":
                    msg = (
                        f"Column {name!r} holds {n_levels} text labels. With no declared roles "
                        f"the model uses text columns with at most {max_auto_levels} labels, so "
                        "it was left out of the model. To include it, declare it categorical "
                        "in the column roles."
                    )
                elif why == "mixed_text_and_numbers":
                    msg = (
                        f"Column {name!r} mixes text labels and numbers, so it was left out of "
                        "the model. Make every cell a label (or every cell a number) to use it."
                    )
                else:
                    msg = (
                        f"Column {name!r} holds text labels that repeat too rarely to learn "
                        "from, so it was left out of the model. Declare it categorical in the "
                        "column roles to include it anyway."
                    )
                findings.append(Finding(
                    check="category_not_modeled", code="category_not_modeled",
                    severity="warning", column=name, message=msg, detail=detail,
                ))
        elif status == "dropped_id":
            entry.update(kind="text_id")
        if entry["finding_code"] is None and entry["reason"] in _REASON_FINDING:
            code = _REASON_FINDING[entry["reason"]]
            if entry["reason"] == "headerless" and entry["kind"] != "row_number":
                code = "headerless_column"
            entry["finding_code"] = code
        if entry["reason"] not in COLUMN_KIND_REASONS:
            entry["reason"] = None  # never an undocumented code
        out.append(entry)
    return out, findings


def with_blank_columns(
    columns: list[dict[str, Any]], sheet_names: list[str], blank: list[bool],
) -> list[dict[str, Any]]:
    """Put the sheet's entirely blank columns back into `columns`, in sheet order.
    The analysis drops them before anything else, so `columns` (built from the
    remaining ones, in the same order) gets one `constant` / `all_blank` entry
    per blank column. They have no `provenance` row."""
    rest = iter(columns)
    out: list[dict[str, Any]] = []
    for name, is_blank in zip(sheet_names, blank, strict=True):
        if is_blank:
            out.append({
                "name": name, "kind": "constant", "role": "excluded", "used_in_fit": False,
                "n_values": 0, "n_levels": None, "levels": None,
                "finding_code": None, "reason": "all_blank",
            })
        else:
            out.append(next(rest))
    out.extend(rest)  # defensive: never drop an entry
    return out


def anonymize_columns(columns: list[dict[str, Any]], alias: Any, hash_level: Any) -> list[dict[str, Any]]:
    """`columns` under `anonymize=True`: identifier-like names are aliased, and
    the level labels of an aliased column are replaced too (a level can be a
    person's name). Other names and labels stay, as in `provenance` and the
    proposal recipes."""
    out = []
    for c in columns:
        name = str(c["name"])
        aliased = alias(name)
        entry = {**c, "name": aliased}
        if aliased != name and isinstance(c.get("levels"), list):
            entry["levels"] = [
                {**lv, "label": hash_level(name, str(lv["label"]))} for lv in c["levels"]
            ]
        out.append(entry)
    return out
