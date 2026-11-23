"""Ingestion preflight + per-column provenance for the /api/run analyze path.

A client uploads an arbitrary run sheet and the engine silently keeps some
columns as features, treats one as the target, and drops the rest (ids, other
measured outputs, constants, sparse columns). That silent drop is a
product-readiness hole: the client cannot see what the model actually used.

`column_provenance` reproduces the exact feature-selection decisions made in
`kalos.portal.app._analyze` and returns a typed, per-column report so the UI can
show, for every column, whether it was kept, why it was dropped, and how many of
its cells had to be coerced from non-numeric text (units like "34.6 C", "88 %").

This module is pure and side-effect free: it inspects a dataframe and the same
regexes/thresholds `_analyze` uses, and never logs raw column names or values.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from re import Pattern
from typing import Literal

import pandas as pd

ColumnStatus = Literal[
    "kept_feature",
    "target",
    "dropped_id",
    "dropped_output",
    "dropped_constant",
    "dropped_constant_on_fitted_rows",
    "dropped_sparse",
    "dropped_non_numeric",
    "dropped_all_blank",
    "dropped_cv_group",
]


RoleSource = Literal["declared", "inferred"]


@dataclass(frozen=True)
class ColumnProvenance:
    """Why one uploaded column was kept or dropped, and its coercion count.

    `source` records whether the column's role came from an explicit declaration
    (`ColumnRoles`) or was inferred from the domain profile's hint patterns, so
    the audit trail is honest about which decisions the caller made.
    """

    name: str
    status: ColumnStatus
    coerced_cells: int  # non-blank cells that did not parse as a plain number
    non_null: int  # count of non-blank cells
    source: RoleSource = "inferred"
    # Why the column was not used, in plain words ("" when it is used). Never
    # echoes the column name (the anonymized response aliases `name` only).
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _coerced_count(series: pd.Series) -> tuple[int, int]:
    """Return (non_blank_count, coerced_count) for one column.

    A cell is "coerced" when it is non-blank but does not parse as a plain number
    on its own (e.g. "34.6 C", "88 %", "n/a"). We count these so the client can
    see that a units-in-cells column was silently cleaned.
    """
    s = series
    nonblank_mask = s.notna() & (s.astype(str).str.strip() != "")
    nonblank = int(nonblank_mask.sum())
    if nonblank == 0:
        return 0, 0
    parsed = pd.to_numeric(s[nonblank_mask], errors="coerce")
    coerced = int(parsed.isna().sum())
    return nonblank, coerced


def column_provenance(
    df: pd.DataFrame,
    *,
    target: str,
    features: list[str],
    numeric_cols: list[str],
    is_identifier: Callable[[str], bool],
    outcome_hint: Pattern[str],
    constant_on_fitted_rows: list[str] | None = None,
    declared: set[str] | None = None,
    declared_ids: set[str] | None = None,
    declared_features: set[str] | None = None,
    cv_group_columns: set[str] | None = None,
    derived_outputs: Mapping[str, str] | set[str] | None = None,
    is_outcome: Callable[[str], bool] | None = None,
) -> list[ColumnProvenance]:
    """Build a per-column provenance report mirroring `_analyze`'s selection.

    Parameters map to the exact objects `_analyze` computes so this report can
    never disagree with what the model actually used:
      - `target`: the chosen target column name.
      - `features`: the columns kept as model inputs.
      - `numeric_cols`: columns that passed the >=80% numeric-parse test.
      - `is_identifier` / `outcome_hint`: the predicate `_analyze` uses to drop
        identifier-like columns (`analysis.is_identifier_name`) and the compiled
        regex for output-like columns.
      - `constant_on_fitted_rows`: columns that vary over the full sheet but are
        constant on the target-present rows the GP actually fits, so their design
        box would collapse to zero width. `_analyze` drops these; they are flagged
        here (not silently pinned) so the client is not misled about that column.
      - `declared_ids`: columns the caller explicitly declared as ids to ignore,
        so they are reported as `dropped_id` (source `declared`) rather than
        mislabeled `dropped_sparse` by the generic non-numeric branch.
      - `declared_features`: columns the caller declared as features; if such a
        column is non-numeric (and not a categorical the engine used) it is
        reported as `dropped_non_numeric` - an honest "you likely meant to mark
        this categorical" - rather than the misleading `dropped_sparse`.
      - `derived_outputs`: columns kept out of the inputs as outputs, mapped to
        the reason code (`declared`, `observed`, `target_leak`, `outcome_name`);
        reported as `dropped_output` with that reason.
      - `is_outcome`: the outcome-NAME predicate `_analyze` uses
        (`columns.is_outcome_name`); defaults to `outcome_hint.search`.

    Every row carries `reason`: why the column is not used, in plain words.
      - `cv_group_columns`: batch, lot, plate, block, date, week or round
        columns (F05) that are the run's circumstances, not settings: they are
        not fitted or proposed and serve as cross-validation group candidates
        (`dropped_cv_group`).

    Status precedence per column:
      target -> kept_feature -> dropped_id(declared) -> dropped_constant_on_fitted_rows
      -> dropped_all_blank -> dropped_non_numeric(declared) -> dropped_id -> dropped_output
      -> dropped_constant -> dropped_sparse.
    """
    feat_set = set(features)
    num_set = set(numeric_cols)
    constant_fitted = set(constant_on_fitted_rows or [])
    declared_set = declared or set()
    declared_id_set = declared_ids or set()
    declared_feat_set = declared_features or set()
    cv_group_set = cv_group_columns or set()
    derived_map: dict[str, str] = (
        dict(derived_outputs) if isinstance(derived_outputs, Mapping)
        else {c: "outcome_name" for c in (derived_outputs or set())}
    )
    outcome_name = is_outcome if is_outcome is not None else (lambda n: bool(outcome_hint.search(n)))
    report: list[ColumnProvenance] = []
    for col in df.columns:
        name = str(col)
        non_null, coerced = _coerced_count(df[col])
        status: ColumnStatus
        if name == str(target):
            status = "target"
        elif col in feat_set:
            status = "kept_feature"
        elif name in cv_group_set:
            status = "dropped_cv_group"
        elif name in derived_map:
            status = "dropped_output"
        elif name in declared_id_set:
            # the caller explicitly excluded this column; its drop is an
            # instruction, not a data-quality inference.
            status = "dropped_id"
        elif name in constant_fitted:
            status = "dropped_constant_on_fitted_rows"
        elif non_null == 0:
            status = "dropped_all_blank"
        elif name in declared_feat_set and col not in num_set:
            # declared as a feature but not numeric and not used as a categorical:
            # honestly say it could not be modeled as a number, not "sparse".
            status = "dropped_non_numeric"
        elif is_identifier(name.strip()):
            status = "dropped_id"
        elif outcome_name(name):
            status = "dropped_output"
        elif col not in num_set:
            # non-numeric and not an id/output: too sparse or non-parseable to model
            status = "dropped_sparse"
        else:
            # numeric, not target/feature/id/output -> excluded as constant
            # (zero variance is the only remaining reason `_analyze` drops it)
            col_num = pd.to_numeric(df[col], errors="coerce")
            std = col_num.std(skipna=True)
            status = "dropped_constant" if (std != std or std <= 1e-9) else "dropped_sparse"
        source: RoleSource = "declared" if name in declared_set else "inferred"
        report.append(
            ColumnProvenance(
                name=name, status=status, coerced_cells=coerced, non_null=non_null,
                source=source, reason=_reason(status, name, derived_map, declared_id_set,
                                              declared_feat_set),
            )
        )
    return report


_OUTPUT_REASONS = {
    "declared": "declared in roles.outputs",
    "observed": "the name says it is observed during or after a run, not set before it",
    "target_leak": "its values follow the target almost exactly, so it looks computed from it",
    "outcome_name": "the name reads like a measured output (titer, yield, concentration, product, score)",
}
_OVERRIDE = " If it is a setting, declare it in roles.features (roles.categoricals for text)."
_STATUS_REASONS = {
    "target": "the target: the value the analysis models",
    "kept_feature": "",
    "dropped_cv_group": (
        "a batch, lot, plate, block, date, week or round column: the run's circumstances, "
        "used as a cross-validation group, never fitted or proposed"
    ),
    "dropped_constant_on_fitted_rows": "the same value on every run that has a measured target",
    "dropped_all_blank": "no values",
    "dropped_constant": "the same value in every run",
    "dropped_sparse": (
        "too few usable numbers, or text that is not a categorical factor "
        "(more than 12 levels, or levels that do not repeat)"
    ),
}


def _reason(
    status: str, name: str, derived: Mapping[str, str], declared_ids: set[str],
    declared_features: set[str],
) -> str:
    if status == "dropped_output":
        code = derived.get(name, "outcome_name")
        text = _OUTPUT_REASONS.get(code, code)
        return text + ("." if code == "declared" else "." + _OVERRIDE)
    if status == "dropped_id":
        if name in declared_ids:
            return "declared in roles.ids."
        return "the name reads like an identifier (run, sample or batch id)." + _OVERRIDE
    if status == "dropped_non_numeric":
        return "declared a feature, but its cells are not numbers; declare it in roles.categoricals if it is a label."
    text = _STATUS_REASONS.get(status, "")
    return text + "." if text else ""


def provenance_dicts(report: list[ColumnProvenance]) -> list[dict]:
    """Serialize a provenance report to plain dicts for the JSON response."""
    return [row.to_dict() for row in report]


__all__ = [
    "ColumnProvenance",
    "ColumnStatus",
    "RoleSource",
    "column_provenance",
    "provenance_dicts",
]
