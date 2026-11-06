"""Column-role safety: guards on top of `_resolve_columns` for columns that must
not be model inputs by default, and the per-column `reason` codes the
provenance report carries.

Rules (each one narrow on purpose, so a normal sheet is not changed):

1. Row-number columns are never inputs unless declared.
   - `headerless`: a column with no header (named by its position) whose values
     are unique integers, one per row. This is a saved row index.
   - `monotone_row_number`: a named column of at least `ROW_NUMBER_MIN_ROWS`
     rows of unique whole numbers that run up or down in sheet order (any
     start: 0..n-1, 1..n, Excel's 2..n+1, n..1) and fill their range nearly
     densely (rows / (span + 1) >= `ROW_NUMBER_MIN_DENSITY`, so a few deleted
     rows still count). A design factor that is unique, whole, sorted and
     step-1 across the sheet is rare; declaring roles keeps it.
   Status `dropped_id`.
2. Outcome-like columns are not inputs unless declared (owner decision: a
   leaky input makes the model look better than it is, so the default is to
   leave it out and say so). The name must read like a measurement taken during
   or after the run (`looks_outcome_like`: `OUTCOME_LIKE_NAME`, i.e. OD, optical
   density, biomass, VCD, viability, DCW, final, harvest, measured, lactate,
   ammonia, ..., or the profile's outcome hint other than a bare concentration
   word). After the target the reason is `after_target_in_sheet`; anywhere else
   it is `outcome_like_name`; each has a warning finding with the same code.
   Status `dropped_output`. Position alone is not enough: many sheets put
   inputs after the target. A strong correlation alone is not enough either: it
   would drop a real strong driver. A non-matching input after the target that
   correlates strongly with it (|Spearman| >= `AFTER_TARGET_RHO`) is kept and
   flagged.
3. A concentration input before a given target is kept. The profile's outcome
   hint matches `conc`, so `Carbon_conc` was dropped as a possible output. When
   the caller named the target, a column whose only outcome-hint match is the
   concentration token, that sits before the target, and that does not match
   `OUTCOME_LIKE_NAME`, is kept as an input (`concentration_before_target`),
   with an info finding.

Declared features (`roles.features`) are never excluded by these rules. A
declared input whose name looks outcome-like gets a warning
(`feature_looks_outcome_like`).

The `recipe_label` reason belongs to a separate rule in `columns.py`; it is
listed here so the provenance reason vocabulary is in one place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kalos.domains import ColumnRoles, DomainProfile
from kalos.portal.analysis_parts.identifiers import is_identifier_name
from kalos.validation.report import Finding

ROW_NUMBER_MIN_ROWS = 8
# Rows per value in the column's range: 1.0 is a gapless counter; 0.9 lets a
# 20-row sheet miss 2 numbers (deleted rows). A dose ladder 10, 20, 30 is 0.1.
ROW_NUMBER_MIN_DENSITY = 0.9
AFTER_TARGET_RHO = 0.9

# Measurement words for rule 2 and the outcome-like flag. Short tokens (OD,
# VCD, DCW) must stand alone (not inside a word: "Methanol_dose" is not OD).
OUTCOME_LIKE_NAME = re.compile(
    r"(?<![a-z])(od|vcd|dcw|wcw|abs)(?![a-z])|optical[ _\-]?density|absorbance|biomass"
    r"|viab|cell[ _\-]?density|final|harvest|measured|readout|lactate|ammoni|residual",
    re.IGNORECASE,
)
_CONC_TOKEN = re.compile(r"conc(entration)?", re.IGNORECASE)

# Provenance reason codes, by status, when no specific rule fired.
_DEFAULT_REASON = {
    "kept_feature": "numeric_input",
    "dropped_id": "name_matches_identifier",
    "dropped_output": "name_matches_target_candidate",
    "dropped_constant": "constant",
    "dropped_constant_on_fitted_rows": "constant_on_fitted_rows",
    "dropped_sparse": "sparse_or_unparsed",
    "dropped_non_numeric": "declared_non_numeric",
    "dropped_all_blank": "all_blank",
    "dropped_recipe_label": "recipe_label",
    "fixed_near_constant": "near_constant",
}
# The role the engine ended up giving each status.
STATUS_ROLE = {
    "target": "target",
    "kept_feature": "input",
    "dropped_id": "identifier",
    "dropped_output": "outcome",
    "dropped_recipe_label": "group",
    "fixed_near_constant": "fixed_input",
}

REASON_CODES = (
    "declared", "requested_target", "name_matches_target_candidate", "last_numeric_column",
    "numeric_input", "categorical_text", "concentration_before_target", "headerless",
    "monotone_row_number", "after_target_in_sheet", "name_matches_identifier", "constant",
    "constant_on_fitted_rows", "sparse_or_unparsed", "declared_non_numeric", "all_blank",
    "recipe_label", "near_constant", "not_declared", "outcome_like_name",
)


def default_reason(status: str, *, numeric: bool) -> str:
    """The reason code for a status when no rule recorded a specific one."""
    if status == "kept_feature" and not numeric:
        return "categorical_text"
    return _DEFAULT_REASON.get(status, status.removeprefix("dropped_"))


def role_for_status(status: str) -> str:
    return STATUS_ROLE.get(status, "unused")


@dataclass
class RoleSafety:
    cont_feats: list
    cat_feats: list
    # column -> (status, reason) for columns these rules excluded
    overrides: dict[str, tuple[str, str]] = field(default_factory=dict)
    # column -> reason for columns kept or chosen with a specific reason
    reasons: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def _row_number_reason(series: pd.Series, *, headerless: bool) -> str | None:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or len(values) < 2:
        return None
    arr = values.to_numpy(float)
    if not np.all(np.isfinite(arr)) or not np.all(arr == np.round(arr)):
        return None
    if len(np.unique(arr)) != len(arr):
        return None
    if headerless:
        return "headerless"
    if len(arr) < ROW_NUMBER_MIN_ROWS:
        return None
    steps = np.diff(arr)
    monotone = bool(np.all(steps > 0) or np.all(steps < 0))
    density = len(arr) / (abs(arr[-1] - arr[0]) + 1.0)
    if monotone and density >= ROW_NUMBER_MIN_DENSITY:
        return "monotone_row_number"
    return None


def _conc_only(name: str, profile: DomainProfile) -> bool:
    """The profile's outcome hint matches `name` only through `conc`."""
    if not profile.outcome_hint.search(name) or not _CONC_TOKEN.search(name):
        return False
    return profile.outcome_hint.search(_CONC_TOKEN.sub(" ", name)) is None


def looks_outcome_like(name: str, profile: DomainProfile) -> bool:
    if OUTCOME_LIKE_NAME.search(name):
        return True
    return bool(profile.outcome_hint.search(name)) and not _conc_only(name, profile)


def column_role_safety(
    df: pd.DataFrame,
    num: list,
    target: str,
    cont_feats: list,
    cat_feats: list,
    *,
    declared: set,
    roles: ColumnRoles | None,
    profile: DomainProfile,
    headerless: list[str],
    target_given: bool,
) -> RoleSafety:
    """Apply the module's rules to the resolved feature lists."""
    cols = list(df.columns)
    rows = df.dropna(axis=0, how="all")
    pos = {c: i for i, c in enumerate(cols)}
    t_pos = pos.get(target, len(cols))
    declared_feats = set(roles.features) if roles is not None else set()
    headerless_set = {str(c) for c in headerless}
    out = RoleSafety(cont_feats=list(cont_feats), cat_feats=list(cat_feats))

    # target reason
    if roles is not None:
        out.reasons[str(target)] = "declared"
    elif target_given:
        out.reasons[str(target)] = "requested_target"
    elif profile.outcome_hint.search(str(target)):
        out.reasons[str(target)] = "name_matches_target_candidate"
    else:
        out.reasons[str(target)] = "last_numeric_column"

    # Rule 3: a concentration input before a given target is kept.
    features_inferred = roles is None or not roles.features
    if features_inferred and target_given:
        cat_set = set(cat_feats)
        ids = set(roles.ids) if roles is not None else set()
        current = set(out.cont_feats)
        for c in num:
            name = str(c)
            if (
                c in current or c == target or c in cat_set or c in ids
                or pos.get(c, len(cols)) >= t_pos
                or is_identifier_name(name, profile.id_hint)
                or not _conc_only(name, profile)
                or OUTCOME_LIKE_NAME.search(name)
            ):
                continue
            current.add(c)
            out.reasons[name] = "concentration_before_target"
            out.findings.append(Finding(
                check="concentration_before_target", code="concentration_before_target",
                severity="info", column=name,
                message=(
                    f"Column {name!r} has a concentration name, which can also mean a measured "
                    "output. It sits before the target in the sheet, so it was kept as an "
                    "input. If it was measured during the run, declare roles to leave it out."
                ),
            ))
        out.cont_feats = [c for c in cols if c in current]

    # Rules 1 and 2: excluded unless declared as a feature.
    kept: list = []
    for c in out.cont_feats:
        name = str(c)
        if c in declared_feats:
            kept.append(c)
            continue
        reason = _row_number_reason(rows[c], headerless=name in headerless_set)
        if reason is not None:
            out.overrides[name] = ("dropped_id", reason)
            out.findings.append(Finding(
                check="row_number_column", code="row_number_column", severity="warning",
                column=name,
                message=(
                    f"Column {name!r} looks like a row number (unique whole numbers, one per "
                    "row), so it was not used as an input. Declare it in roles.features if it "
                    "is a real input."
                ),
                detail={"reason": reason},
            ))
            continue
        if out.reasons.get(name) != "concentration_before_target" and looks_outcome_like(
            name, profile,
        ) and pos.get(c, -1) <= t_pos:
            out.overrides[name] = ("dropped_output", "outcome_like_name")
            out.findings.append(Finding(
                check="outcome_like_name", code="outcome_like_name", severity="warning",
                column=name,
                message=(
                    f"Column {name!r} has a name that reads like a measurement taken during or "
                    "after the run, so it was not used as an input. Declare it in "
                    "roles.features if it is set before the run."
                ),
            ))
            continue
        if pos.get(c, -1) > t_pos and looks_outcome_like(name, profile):
            out.overrides[name] = ("dropped_output", "after_target_in_sheet")
            out.findings.append(Finding(
                check="after_target_outcome", code="after_target_in_sheet", severity="warning",
                column=name,
                message=(
                    f"Column {name!r} sits after the target and its name reads like a "
                    "measurement taken during the run, so it was not used as an input. Declare "
                    "it in roles.features if it is set before the run."
                ),
            ))
            continue
        kept.append(c)
    out.cont_feats = kept

    # Flags (no exclusion).
    y = pd.to_numeric(rows[target], errors="coerce")
    for c in [*out.cont_feats, *out.cat_feats]:
        name = str(c)
        if out.reasons.get(name) == "concentration_before_target":
            continue
        if looks_outcome_like(name, profile):
            out.findings.append(Finding(
                check="feature_looks_outcome_like", code="feature_looks_outcome_like",
                severity="warning", column=name,
                message=(
                    f"Input {name!r} has a name that reads like a measured outcome. If it is "
                    "measured during or after the run, it should not be an input: the model "
                    "would look better than it is."
                ),
                detail={"declared": c in declared},
            ))
            continue
        if c in out.cont_feats and c not in declared and pos.get(c, -1) > t_pos:
            x = pd.to_numeric(rows[c], errors="coerce")
            ok = x.notna() & y.notna()
            if int(ok.sum()) >= 6 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
                rho = float(x[ok].rank().corr(y[ok].rank()))
                if rho == rho and abs(rho) >= AFTER_TARGET_RHO:
                    out.findings.append(Finding(
                        check="after_target_high_correlation",
                        code="after_target_high_correlation", severity="warning", column=name,
                        message=(
                            f"Input {name!r} sits after the target and tracks it closely "
                            f"(Spearman {rho:.2f}). Check it is set before the run, not "
                            "measured during it."
                        ),
                        detail={"spearman": round(rho, 3)},
                    ))
    for c in out.cont_feats:
        if str(c) in declared:
            out.reasons.setdefault(str(c), "declared")
    return out


__all__ = [
    "AFTER_TARGET_RHO",
    "OUTCOME_LIKE_NAME",
    "REASON_CODES",
    "ROW_NUMBER_MIN_DENSITY",
    "ROW_NUMBER_MIN_ROWS",
    "STATUS_ROLE",
    "RoleSafety",
    "column_role_safety",
    "default_reason",
    "looks_outcome_like",
    "role_for_status",
]
