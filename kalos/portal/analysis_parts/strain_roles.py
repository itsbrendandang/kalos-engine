"""Clone / strain and covariate roles for one run sheet (docs/CLONES.md).

Decides, before any model is fit:

- which column (if any) is the production clone / strain, from (in order) the
  campaign setup (`strain_column`), the declared `roles.strain`, or, with no
  declared roles, a column NAME matching `STRAIN_NAME_HINT` (`clone`, `strain`,
  `cell line`). The clone is ALWAYS an unordered label, even when its values
  look numeric: a numeric `clone` column used to be fit as a continuous input and
  proposed as `clone: 329.44`, a clone that does not exist.
- the clone policy: `fixed` (the default once a clone column is DECLARED: every
  proposal holds the clone at one level) or `factor` (the proposals choose among
  the clones; the behavior of a detected-but-unconfirmed clone column, which is
  what text clone columns got before).
- which columns are COVARIATES (passage, generation, PDL, thaw / vial / cell
  bank): recorded run properties, never setpoints. Numeric covariates stay model
  inputs when every usable run has a value, and are HELD at a reference value in
  every proposal (never proposed). Label covariates (a thaw ID) are not model
  inputs: the next run uses a new thaw the model has never seen. They are used
  for the confounding checks.

Pure pandas/numpy, torch-free. Labels are only ever compared for equality and
counted; nothing here parses or interprets them.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kalos.domains import ColumnRoles
from kalos.portal.analysis_parts.columns import AUTO_CATEGORICAL_MAX_LEVELS
from kalos.portal.cells import is_null_label
from kalos.validation.report import Finding

STRAIN_NAME_HINT = re.compile(r"(?:^|[^a-z])(?:clone|strain|cell[ _.-]?line)s?(?![a-z])", re.I)
COVARIATE_NAME_HINT = re.compile(
    r"(?:^|[^a-z])(?:passage|generation|gen[ _.-]?(?:no|num|number)|pdl|"
    r"population[ _.-]?doublings?|thaw|vial|cell[ _.-]?bank)(?![a-z])",
    re.I,
)
# Label columns used for the confounding checks (round, thaw, scale, lot...).
BLOCK_NAME_HINT = re.compile(
    r"(?:^|[^a-z])(?:round|batch|lot|thaw|vial|scale|vessel|bioreactor|plate|day|date|bank)s?(?![a-z])",
    re.I,
)
SCALE_NAME_HINT = re.compile(r"(?:^|[^a-z])(?:scale|vessel|bioreactor)(?![a-z])", re.I)
ROUND_NAME_HINT = re.compile(r"(?:^|[^a-z])round(?:[ _.-]?(?:no|num|number|id))?$", re.I)
MAX_BLOCK_LEVELS = 50
_INT_FLOAT = re.compile(r"^-?\d+\.0+$")


def normalize_label(value: Any) -> str | None:
    """One clone label as a string, or None for blank / null tokens. Integer-valued
    numbers lose a trailing `.0`, so `101`, `101.0` and `"101"` are one clone."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            return None
        return str(int(f)) if f.is_integer() else repr(f)
    text = str(value).strip()
    if not text or is_null_label(text):
        return None
    if _INT_FLOAT.match(text):
        text = text.split(".")[0]
    return text


def normalized_labels(series: pd.Series) -> pd.Series:
    return pd.Series([normalize_label(v) for v in series.tolist()], index=series.index, dtype=object)


@dataclass
class StrainPlan:
    """What the analysis does with the clone and covariate columns."""

    column: str | None = None
    source: str | None = None            # "setup" | "declared" | "detected"
    policy: str | None = None            # "fixed" | "factor"
    requested_level: str | None = None
    requested_level_source: str | None = None  # "setup" | "declared" | "registry_lead"
    modeled: bool = False                # the clone is a categorical model input
    n_levels: int = 0
    numeric_labels: bool = False
    covariates_numeric: list[str] = field(default_factory=list)
    covariates_label: list[str] = field(default_factory=list)
    covariates_unfitted: dict[str, str] = field(default_factory=dict)
    covariate_source: str | None = None  # "setup" | "declared" | "detected"
    findings: list[Finding] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    unavailable_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.column is not None


def _levels(df: pd.DataFrame, col: str, keep: np.ndarray | None) -> list[str]:
    labels = normalized_labels(df[col])
    if keep is not None:
        labels = labels[keep]
    return sorted({v for v in labels.tolist() if v is not None})


def _is_numeric_column(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce")
    non_blank = series.notna() & (series.astype(str).str.strip() != "")
    return bool(non_blank.any()) and bool(vals[non_blank].notna().mean() >= 0.8)


def resolve_strain(
    df: pd.DataFrame,
    *,
    target: str,
    roles: ColumnRoles | None,
    setup: Mapping[str, Any] | None,
    cont_feats: list,
    cat_feats: list,
    keep: np.ndarray,
) -> tuple[pd.DataFrame, list, list, StrainPlan]:
    """Apply the clone and covariate roles to the resolved feature lists.

    Returns `(df, cont_feats, cat_feats, plan)`. `df` is a copy whose clone column
    holds normalized string labels (so every downstream cast agrees), or the
    input frame unchanged when there is no clone column. With no clone and no
    covariate column the feature lists come back exactly as given (the no-clone
    path is byte-identical; tests/test_strain_golden_no_clone.py)."""
    plan = StrainPlan()
    setup = setup or {}
    columns = [str(c) for c in df.columns]
    colset = set(columns)

    # --- the clone column --------------------------------------------------- #
    col: str | None = None
    if setup.get("strain_column"):
        col, plan.source = str(setup["strain_column"]), "setup"
    elif roles is not None and roles.strain:
        col, plan.source = str(roles.strain), "declared"
    elif roles is None:
        hits = [c for c in columns if c != str(target) and STRAIN_NAME_HINT.search(c)]
        if hits:
            col, plan.source = hits[0], "detected"
    if col is not None and col not in colset:
        plan.unavailable_reason = "the declared clone column is not in this sheet"
        plan.refusals.append({
            "code": "strain_column_not_found", "level": None,
            "reason": "the clone column named in the campaign setup is not a column of this sheet",
            "what_would_fix_it": "set strain_column to one of the sheet's columns, or clear it",
        })
        col, plan.source = None, None
    if col is not None and col == str(target):
        col, plan.source = None, None

    if col is not None:
        plan.column = col
        if plan.source == "detected":
            plan.policy = "factor"
        else:
            raw_policy = setup.get("strain_policy") or (roles.strain_policy if roles else None)
            plan.policy = raw_policy if raw_policy in ("fixed", "factor") else "fixed"
        lvl = setup.get("strain_level")
        if lvl:
            plan.requested_level, plan.requested_level_source = normalize_label(lvl), "setup"
        elif roles is not None and roles.strain_level and plan.source != "setup":
            plan.requested_level, plan.requested_level_source = normalize_label(roles.strain_level), "declared"
        elif setup.get("strain_lead"):
            plan.requested_level, plan.requested_level_source = normalize_label(setup["strain_lead"]), "registry_lead"
        plan.numeric_labels = _is_numeric_column(df[col])
        df = df.copy()
        df[col] = normalized_labels(df[col])
        levels = _levels(df, col, keep)
        plan.n_levels = len(levels)
        cont_feats = [c for c in cont_feats if str(c) != col]
        cat_feats = [c for c in cat_feats if str(c) != col]
        if len(levels) > AUTO_CATEGORICAL_MAX_LEVELS:
            plan.refusals.append({
                "code": "too_many_clones", "level": None,
                "reason": (
                    f"{len(levels)} clone labels exceed the {AUTO_CATEGORICAL_MAX_LEVELS}-level "
                    "limit for a categorical model input; the clone is not modeled and the "
                    "per-clone measured summary is still reported"
                ),
                "what_would_fix_it": (
                    f"analyze at most {AUTO_CATEGORICAL_MAX_LEVELS} clones at a time "
                    "(filter the sheet to the clones still in play)"
                ),
            })
        elif len(levels) >= 2:
            plan.modeled = True
            wanted = {str(c) for c in cat_feats} | {col}
            cat_feats = [c for c in df.columns if str(c) in wanted]
        if plan.source == "detected":
            plan.findings.append(Finding(
                check="strain_column_detected", severity="info", column=col,
                message=(
                    f"Column {col!r} was read as the production clone / strain by its name and "
                    "treated as an unordered label"
                    + (" (its values look numeric, but a clone ID is never a number)" if plan.numeric_labels else "")
                    + ". Proposals may choose among the clones until the role is confirmed; declare "
                    "it (strain_column in the campaign setup, or roles.strain) to hold proposals at "
                    "one clone (policy 'fixed')."
                ),
                detail={"n_levels": len(levels), "numeric_looking": bool(plan.numeric_labels),
                        "policy": plan.policy},
            ))

    # --- covariates ----------------------------------------------------------- #
    covs: list[str] = []
    if setup.get("covariates"):
        covs, plan.covariate_source = [str(c) for c in setup["covariates"]], "setup"
    elif roles is not None and roles.covariates:
        covs, plan.covariate_source = [str(c) for c in roles.covariates], "declared"
    elif roles is None:
        covs = [
            c for c in columns
            if c != str(target) and c != col and COVARIATE_NAME_HINT.search(c)
        ]
        plan.covariate_source = "detected" if covs else None
    covs = [c for c in covs if c in colset and c != str(target) and c != col]
    if covs:
        cont_names = {str(c) for c in cont_feats}
        for c in covs:
            if _is_numeric_column(df[c]):
                plan.covariates_numeric.append(c)
            else:
                plan.covariates_label.append(c)
        cat_feats = [c for c in cat_feats if str(c) not in set(covs)]
        # A numeric covariate is a model input only when every run with a usable
        # target has a value: nothing is imputed, and dropping those runs instead
        # would throw away every campaign run logged without it (a folded round's
        # recipe never carries a passage).
        fitted: list[str] = []
        for c in plan.covariates_numeric:
            vals = pd.to_numeric(df.loc[keep, c], errors="coerce")
            n_missing = int(vals.isna().sum())
            if n_missing:
                plan.covariates_unfitted[c] = (
                    f"{n_missing} run(s) with a measured target have no numeric value; a covariate "
                    "is not imputed, so it is recorded but not used as a model input"
                )
            elif float(vals.max() - vals.min()) <= 1e-9:
                plan.covariates_unfitted[c] = "constant over the measured runs"
            else:
                fitted.append(c)
        plan.covariates_numeric = fitted
        # Numeric covariates are model inputs (held in proposals); a declared one
        # the profile did not pick up is added back in sheet order.
        cont_names -= set(covs)
        wanted = cont_names | set(plan.covariates_numeric)
        cont_feats = [c for c in df.columns if str(c) in wanted]
        if plan.covariate_source == "detected":
            plan.findings.append(Finding(
                check="covariate_columns_detected", severity="info",
                message=(
                    f"{len(covs)} column(s) were read as covariates by name (passage, generation, "
                    "PDL, thaw / vial / cell bank). Covariates are recorded run properties and are "
                    "never proposed; numeric ones are model inputs held at a reference value in "
                    "every proposal."
                ),
                detail={"columns": list(covs)},
            ))
    return df, cont_feats, cat_feats, plan


def block_columns(
    df: pd.DataFrame, *, target: str, plan: StrainPlan, kept_cats: list, features: list,
) -> dict[str, pd.Series]:
    """Label columns the confounding check compares the clone against: label
    covariates, other kept categorical inputs, and non-feature columns whose name
    marks a block (round, batch, lot, thaw, scale, plate, day...) with 2 to
    `MAX_BLOCK_LEVELS` levels. Values as normalized strings (None for blank)."""
    out: dict[str, pd.Series] = {}
    feat = {str(c) for c in features}
    for c in df.columns:
        name = str(c)
        if name in (str(target), plan.column) or (name in feat and c not in kept_cats):
            continue
        if not (name in plan.covariates_label or c in kept_cats or BLOCK_NAME_HINT.search(name)):
            continue
        series = df[c]
        if isinstance(series, pd.DataFrame):
            continue
        labels = normalized_labels(series)
        n = len({v for v in labels.tolist() if v is not None})
        if 2 <= n <= MAX_BLOCK_LEVELS:
            out[name] = labels
    return out


def first_matching(columns: Mapping[str, Any], pattern: re.Pattern[str]) -> str | None:
    return next((c for c in columns if pattern.search(c)), None)


__all__ = [
    "BLOCK_NAME_HINT",
    "COVARIATE_NAME_HINT",
    "ROUND_NAME_HINT",
    "SCALE_NAME_HINT",
    "STRAIN_NAME_HINT",
    "StrainPlan",
    "block_columns",
    "first_matching",
    "normalize_label",
    "normalized_labels",
    "resolve_strain",
]
