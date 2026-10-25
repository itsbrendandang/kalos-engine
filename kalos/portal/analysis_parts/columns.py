"""Column-role resolution for one run sheet: which columns are the target, the
continuous and categorical features, and the CV group. Also the numeric-column
rule and duplicate-header handling.
"""
from __future__ import annotations

import pandas as pd

from kalos.core.splits import row_hash_groups
from kalos.domains import ColumnRoles, DomainProfile
from kalos.portal.analysis_parts.identifiers import is_identifier_name
from kalos.portal.cells import (
    CENSORED,
    NON_NUMERIC,
    NUMERIC,
    classify_cells,
    is_null_label,
)
from kalos.portal.uploads import numeric_columns


def _numeric_cols(df: pd.DataFrame) -> list:
    # Numeric if >=80% of NON-BLANK cells parse as numbers. Blanks are treated as
    # "absent" (filled with 0 later), so a sparse component column still counts.
    # One definition, shared with the upload layer's default-sheet rule.
    return numeric_columns(df)


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Suffix duplicate column labels (`X`, `X.1`, ...) so every column is a Series.

    A run sheet with a repeated header would otherwise make `df[label]` return a
    2-D frame and break the analysis. pandas already does this on CSV read; we do
    it here too so a directly-built frame (or an xlsx with duplicate headers) is
    handled identically, and the duplicate stays visible in the provenance report.
    """
    if not df.columns.duplicated().any():
        return df
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        key = str(col)
        if key in seen:
            seen[key] += 1
            new_cols.append(f"{key}.{seen[key]}")
        else:
            seen[key] = 0
            new_cols.append(key)
    out = df.copy()
    out.columns = new_cols
    return out


def _resolve_columns(
    df: pd.DataFrame,
    num: list,
    target: str | None,
    roles: ColumnRoles | None,
    profile: DomainProfile,
) -> tuple[str, list, list, str | None, set]:
    """Decide the column roles for one run sheet.

    Returns `(target, cont_feats, cat_feats, gcol, declared)`, where the two
    feature lists are in sheet-column order and `declared` is the set of columns
    whose role came from an explicit `ColumnRoles` (empty in inference mode).

    Two modes:
      - `roles` given: use the declared target, features, ids, group, and
        categoricals. When `roles.features` is empty the continuous features are
        still inferred from the `profile`, honoring the declared ids/categoricals.
      - `roles` None (default): infer everything from `profile` (with the
        bioprocess profile this reproduces the original behavior exactly).
    """
    num_set = set(num)
    if roles is not None:
        if roles.target not in df.columns:
            raise ValueError(f"declared target {roles.target!r} is not a column")
        # A declared feature / categorical / group / id whose name does not match
        # a header is a silent data-loss trap (typo, case mismatch): the real
        # column would never be excluded/used and provenance would not flag it.
        # Fail loudly so the caller is never told a declared column was honored
        # when it was not - ids included (a typo'd id leaves the real id in as a
        # feature otherwise).
        declared_names = [*roles.features, *roles.categoricals, *roles.ids, *roles.sparse]
        if roles.groups:
            declared_names.append(roles.groups)
        unknown = sorted({c for c in declared_names if c not in df.columns})
        if unknown:
            raise ValueError(f"declared columns not found in the sheet: {unknown}")
        # A column cannot be both the target and an id/categorical - a
        # self-contradictory schema is a caller error, not something to resolve
        # silently by dropping one role.
        if roles.target in set(roles.ids) or roles.target in set(roles.categoricals):
            raise ValueError(
                f"declared target {roles.target!r} cannot also be declared an id or categorical"
            )
        target = roles.target
        cat_feats = [c for c in roles.categoricals if c in df.columns and c != target]
        cat_set = set(cat_feats)
        ids = set(roles.ids)
        # `declared` drives the provenance `source` field, so declared ids belong
        # in it too (their exclusion was the caller's instruction, not an inference).
        declared: set = {target, *cat_feats, *ids}
        if roles.features:
            declared.update(roles.features)
            # Declared continuous features must clear the same >=80% numeric-parse
            # gate inference mode uses (`num`). A text column declared as a feature
            # (the caller likely meant categorical) is left out here and reported
            # as `dropped_non_numeric` by provenance, not silently zero-filled in.
            cont_feats = [
                c for c in roles.features
                if c != target and c not in ids and c not in cat_set and c in num_set
            ]
        else:
            cont_feats = [
                c for c in num
                if c != target and c not in cat_set and c not in ids
                and not is_identifier_name(c, profile.id_hint)
                and not profile.outcome_hint.search(str(c))
            ]
        gcol = roles.groups if (roles.groups and roles.groups in df.columns) else None
        # normalize both feature lists to sheet-column order
        cont_feats = [c for c in df.columns if c in set(cont_feats)]
        cat_feats = [c for c in df.columns if c in cat_set]
        if gcol is None:
            gcol = _auto_group_column(df, profile, [*cont_feats, *cat_feats])
        return target, cont_feats, cat_feats, gcol, declared

    # inference mode (default profile = bioprocess): unchanged legacy behavior
    outcomes = [c for c in num if profile.outcome_hint.search(str(c))]
    if target is None or target not in df.columns:
        target = next(
            (c for c in outcomes if profile.target_pref.search(str(c))),
            outcomes[0] if outcomes else num[-1],
        )
    cont_feats = [
        c for c in num
        if c != target
        and not is_identifier_name(c, profile.id_hint)
        and not profile.outcome_hint.search(str(c))
    ]
    cat_feats = _auto_categorical(df, num, target, profile)
    gcol = _auto_group_column(df, profile, [*cont_feats, *cat_feats])
    return target, cont_feats, cat_feats, gcol, set()


def _auto_group_column(df: pd.DataFrame, profile: DomainProfile, features: list) -> str | None:
    """The CV group column detected from the sheet's names, or None.

    A name matching `profile.group_hint` (medium, strain, batch, lot, ...) is NOT
    automatically a leakage barrier. Medium is usually a process FACTOR: grouping
    folds by it hides the very effect the model should learn, and proposals never
    vary it (the `Medium in {A,B,C}` repro). So it is used as the group only when
    every one of its levels is confounded with a single recipe - each level's rows
    are all the same feature settings, so the level IS the recipe and grouping by
    it cannot hide a factor. Otherwise it stays a candidate factor and the caller
    reports it. A declared `roles.groups` never reaches this function.
    """
    names = [str(c) for c in features]
    for c in df.columns:
        if not profile.group_hint.search(str(c)):
            continue
        # The recipe is the other features: the candidate itself is what is tested.
        feats = [f for f in names if f != str(c)]
        if not feats:
            continue
        labels = df[c].fillna("").astype(str).str.strip()
        levels = [v for v in labels.unique().tolist() if not is_null_label(v)]
        if len(levels) < 2:
            continue
        recipe = row_hash_groups(df[feats])
        per_level = pd.Series(recipe).groupby(labels.to_numpy()).nunique()
        if all(int(per_level.get(v, 0)) == 1 for v in levels):
            return c
    return None


# A text column is treated as a categorical factor (with no declared roles) when it
# has between 2 and this many distinct values, each repeated across runs.
AUTO_CATEGORICAL_MAX_LEVELS = 12


def _auto_categorical(df: pd.DataFrame, num: list, target: str, profile: DomainProfile) -> list:
    """Text columns that look like categorical factors, for a sheet with no declared
    roles. Without this, a text factor (a resin, a medium) was silently dropped as
    `dropped_sparse` and no proposal ever varied it.

    A column qualifies when it is not numeric, not the target, not an identifier or
    outcome by name, holds no numeric-looking cells (a mixed column is ambiguous),
    and has 2..AUTO_CATEGORICAL_MAX_LEVELS distinct values that repeat: at least two
    runs per value on average. Each such column is reported by `_analyze` as a
    finding, so the uploader can confirm it or override it with `roles`.
    """
    num_set = set(num)
    found: list = []
    for c in df.columns:
        if c in num_set or c == target:
            continue
        name = str(c)
        if is_identifier_name(name, profile.id_hint) or profile.outcome_hint.search(name):
            continue
        series = df[c]
        if isinstance(series, pd.DataFrame):  # duplicate header; analysis dedupes later
            continue
        kinds = classify_cells(series)
        if kinds.isin([NUMERIC, CENSORED]).any():
            continue
        labels = series[kinds == NON_NUMERIC].astype(str).str.strip()
        n_levels = int(labels.nunique())
        enough_repeats = 2 * n_levels <= len(labels)
        if len(labels) >= 3 and 2 <= n_levels <= AUTO_CATEGORICAL_MAX_LEVELS and enough_repeats:
            found.append(c)
    return found
