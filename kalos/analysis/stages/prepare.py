"""Stage 1 (prepare): validate, type cells, resolve columns/strain/setup, build design matrix.

Torch-free at import time. When `preflight_only=True`, returns `None` after all
pre-fit refusals succeed without importing torch or allocating design matrices.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kalos.domains import ColumnRoles, DesignSpace, Dimension, DomainProfile
from kalos.portal.analysis_parts.columns import (
    _dedupe_columns,
    _numeric_cols,
    _resolve_columns,
    derived_output_columns,
    is_outcome_name,
    name_inferred_outputs,
)
from kalos.portal.analysis_parts.cv_grouping import ROUND_ATTR, is_batch_like
from kalos.portal.analysis_parts.policy import _ERR_VALIDATION_FAILED, validation_mode
from kalos.portal.analysis_parts.ranges import _absent_is_zero_columns, _physical_range
from kalos.portal.analysis_parts.refusals import (
    _ERR_NO_PROCESS_INPUTS,
    _refuse_unusable_target,
    _too_few_rows,
)
from kalos.portal.analysis_parts.strain_roles import StrainPlan, resolve_strain
from kalos.portal.analysis_parts.strain_state import StrainState
from kalos.portal.cells import (
    BLANK,
    CENSORED,
    MISSING,
    NON_NUMERIC,
    NUMERIC,
    REASON_CENSORED,
    REASON_MISSING,
    REASON_NON_NUMERIC,
    cell_labels,
    classify_cells,
    is_null_label,
    to_numbers,
)
from kalos.portal.setup_settings import SetupPlan, plan_setup
from kalos.portal.uploads import (
    _ERR_TOO_MANY_FIT_ROWS,
    MAX_FIT_ROWS,
    MIN_FIT_ROWS,
    UploadRejected,
    clip_name,
)
from kalos.validation import apply_unit_conversions, validate_frame


@dataclass
class PreparedAnalysis:
    """Outputs of stage 1 (`prepare_analysis`), consumed by downstream stages."""

    df: pd.DataFrame
    target: Any
    num: list[Any]
    cont_feats: list[Any]
    kept_cats: list[Any]
    constant_cats: list[Any]
    constant_on_fitted: list[Any]
    cat_dims_map: dict[Any, tuple[str, ...]]
    code_maps: dict[Any, dict[str, int]]
    gcol: Any
    declared: set[Any]
    batch_cols: list[Any]
    derived_outputs: dict[str, dict[str, Any]]
    auto_cats: list[Any]
    candidate_targets: list[Any]
    absent_zero: set[str]
    header_info: dict[str, Any]
    round_attr: Any
    frame_index: pd.Index
    keep_index: pd.Index
    validation: Any
    by_column: dict[str, dict[str, int]]
    n_rows_dropped: int
    n_censored_rows: int
    zero_filled: dict[str, int]
    y: np.ndarray
    Xc_raw: pd.DataFrame
    X: np.ndarray
    design: DesignSpace
    bounds: np.ndarray
    cat_dims: list[int] | None
    cat_cardinalities: list[int] | None
    box_exclusions: list[dict[str, Any]]
    strain_plan: StrainPlan
    strain: StrainState
    setup_plan: SetupPlan | None
    minimize: bool


def prepare_analysis(
    df: pd.DataFrame,
    target: str | None,
    *,
    roles: ColumnRoles | None,
    profile: DomainProfile,
    constraint: Mapping[str, Any] | None,
    preflight_only: bool,
    setup: Mapping[str, Any] | None,
    max_fit_rows: int = MAX_FIT_ROWS,
) -> PreparedAnalysis | None:
    """Run all preflight checks and build the design matrix for fitting.

    Returns `None` when `preflight_only=True` and every pre-fit check passes.
    """
    header_info: dict[str, Any] = dict(df.attrs.get("kalos_header") or {})
    round_attr = df.attrs.get(ROUND_ATTR)
    frame_index = df.index
    original_columns = {str(c) for c in df.columns}
    n_header_columns = int(df.shape[1])
    df = _dedupe_columns(df.dropna(axis=1, how="all"))

    n_columns = int(df.shape[1]) if len(df) else n_header_columns
    if n_columns < 2:
        raise UploadRejected(
            f"The sheet has {n_columns} column{'s' if n_columns != 1 else ''}; "
            "a run sheet needs at least 2 (process inputs and a measured target). If "
            "this is a text file, check its values are separated by commas or tabs.",
            code="too_few_columns",
            context={"n_columns": n_columns},
        )
    n_data_rows = len(df.dropna(axis=0, how="all"))
    if n_data_rows < MIN_FIT_ROWS:
        raise _too_few_rows(n_data_rows)
    requested_target = roles.target if roles is not None else target
    if requested_target is not None and str(requested_target) not in {str(c) for c in df.columns}:
        if str(requested_target) in original_columns:
            raise UploadRejected(
                f"The target column {clip_name(requested_target)!r} has no values. No model "
                "can be fit and no proposals were generated. Fill in the measured "
                "outcome, or choose another target column.",
                code="target_empty",
                context={"target": str(requested_target)},
            )
        available = [
            str(c) for c in _numeric_cols(df) if is_outcome_name(str(c), profile)
        ] or [str(c) for c in _numeric_cols(df)]
        raise UploadRejected(
            f"The requested target column {clip_name(requested_target)!r} is not in this sheet. "
            "Choose one of the sheet's columns as the target.",
            code="target_not_found",
            context={"target": str(requested_target), "available_targets": available},
        )

    mode = validation_mode()
    validation = validate_frame(df, target=target, profile=profile, mode=mode)
    if mode == "strict" and validation.status == "fail":
        raise UploadRejected(_ERR_VALIDATION_FAILED)
    if validation.conversions:
        df = apply_unit_conversions(df, list(validation.conversions))

    num = _numeric_cols(df)
    if not num:
        raise UploadRejected(
            "No numeric columns were found. A run sheet needs numeric process-input "
            "columns (for example pH or temperature) and a numeric target column "
            "(for example a titer), one run per row.",
            code="no_numeric_inputs",
        )
    resolved_target, cont_feats, cat_feats, gcol, declared = _resolve_columns(
        df, num, target, roles, profile
    )
    df, cont_feats, cat_feats, strain_plan = resolve_strain(
        df, target=str(resolved_target), roles=roles, setup=setup,
        cont_feats=list(cont_feats), cat_feats=list(cat_feats),
        keep=classify_cells(df[resolved_target]).to_numpy(object) == NUMERIC,
    )
    _strain_cols = {
        c for c in (strain_plan.column, *strain_plan.covariates_numeric, *strain_plan.covariates_label)
        if c is not None
    }
    batch_cols = [
        c for c in [*cont_feats, *cat_feats]
        if is_batch_like(str(c)) and c not in declared and str(c) not in _strain_cols
    ]
    if batch_cols:
        cont_feats = [c for c in cont_feats if c not in batch_cols]
        cat_feats = [c for c in cat_feats if c not in batch_cols]
    derived_outputs = derived_output_columns(
        df, str(resolved_target), [c for c in [*cont_feats, *cat_feats] if str(c) not in _strain_cols],
        roles, profile,
    )
    if derived_outputs:
        cont_feats = [c for c in cont_feats if str(c) not in derived_outputs]
        cat_feats = [c for c in cat_feats if str(c) not in derived_outputs]
    derived_outputs.update(name_inferred_outputs(
        df, str(resolved_target), profile, roles,
        already={*derived_outputs, *(str(c) for c in [*cont_feats, *cat_feats])}
        | {str(c) for c in batch_cols} | _strain_cols,
    ))
    auto_cats = (
        [c for c in cat_feats if str(c) != strain_plan.column] if roles is None else []
    )
    if constraint is not None:
        _constraint_col_name = str(constraint["column"])
        cont_feats = [c for c in cont_feats if str(c) != _constraint_col_name]
        cat_feats = [c for c in cat_feats if str(c) != _constraint_col_name]
    if roles is None:
        outcomes = [c for c in num if is_outcome_name(str(c), profile)]
        candidate_targets = outcomes or [resolved_target]
    else:
        candidate_targets = [resolved_target]

    _refuse_unusable_target(df, resolved_target, num, profile)

    varying = []
    for c in cont_feats:
        col = to_numbers(df[c])
        if col.std(skipna=True) and col.std() > 1e-9:
            varying.append(c)
    cont_feats = varying
    if not cont_feats and not cat_feats:
        raise UploadRejected(_ERR_NO_PROCESS_INPUTS, code="no_process_inputs")

    tgt_kinds = classify_cells(df[resolved_target]).to_numpy(object)
    row_blank = np.ones(len(df), dtype=bool)
    for c in df.columns:
        row_blank &= classify_cells(df[c]).to_numpy(object) == BLANK
    active = ~row_blank
    y_all = to_numbers(df[resolved_target]).to_numpy(float)
    keep = tgt_kinds == NUMERIC
    feat_kind = {c: classify_cells(df[c]).to_numpy(object) for c in cont_feats}
    Xc_all = df[cont_feats].apply(to_numbers) if cont_feats else pd.DataFrame(index=df.index)

    kept_cats: list[Any] = []
    constant_cats: list[Any] = []
    cat_dims_map: dict[Any, tuple[str, ...]] = {}
    for c in cat_feats:
        labels = cell_labels(df.loc[keep, c])
        levels = tuple(sorted({v for v in labels.tolist() if not is_null_label(v)}))
        if len(levels) < 2:
            constant_cats.append(c)
            continue
        kept_cats.append(c)
        cat_dims_map[c] = levels
    cat_ok = {
        c: cell_labels(df[c]).isin(cat_dims_map[c]).to_numpy()
        for c in kept_cats
    }

    absent_zero = _absent_is_zero_columns(cont_feats, roles)

    def _row_flags(features: list[Any], cats: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        incomplete = np.isin(tgt_kinds, [BLANK, MISSING, NON_NUMERIC])
        censored = tgt_kinds == CENSORED
        for c in features:
            bad = [MISSING, NON_NUMERIC] if str(c) in absent_zero else [BLANK, MISSING, NON_NUMERIC]
            incomplete |= np.isin(feat_kind[c], bad)
            censored |= feat_kind[c] == CENSORED
        for c in cats:
            incomplete |= ~cat_ok[c]
        return incomplete, censored

    inc1, cens1 = _row_flags(cont_feats, kept_cats)
    fit1 = keep & ~inc1 & ~cens1
    constant_on_fitted: list[Any] = []
    if cont_feats and fit1.any():
        block = Xc_all[fit1].fillna(0.0)
        spread = block.max(axis=0) - block.min(axis=0)
        constant_on_fitted = [c for c in cont_feats if float(spread[c]) <= 1e-9]
        if constant_on_fitted:
            cont_feats = [c for c in cont_feats if c not in set(constant_on_fitted)]
            Xc_all = Xc_all.drop(columns=constant_on_fitted)
    inc, cens = _row_flags(cont_feats, kept_cats)
    fit = keep & ~inc & ~cens
    fit_pos = np.flatnonzero(fit)
    if not cont_feats and not kept_cats:
        raise UploadRejected(_ERR_NO_PROCESS_INPUTS, code="no_process_inputs")

    n_rows_dropped = int((inc & active).sum())
    n_censored_rows = int((cens & ~inc & active).sum())
    if len(fit_pos) < MIN_FIT_ROWS:
        if int(keep.sum()) < MIN_FIT_ROWS:
            raise _too_few_rows(int(keep.sum()), str(resolved_target))
        raise UploadRejected(
            f"Only {len(fit_pos)} run{'s have' if len(fit_pos) != 1 else ' has'} complete data "
            f"for the model. {n_rows_dropped} were excluded for missing or non-numeric cells "
            f"and {n_censored_rows} for censored values (for example <LOD or >100), which are "
            f"not imputed. At least {MIN_FIT_ROWS} complete runs are needed to fit a model. "
            "The per-column counts are in `validation.findings`.",
            code="too_few_complete_rows",
            context={
                "min_rows": MIN_FIT_ROWS,
                "n_rows": len(fit_pos),
                "n_dropped_incomplete": n_rows_dropped,
                "n_censored_excluded": n_censored_rows,
            },
        )
    if len(fit_pos) > max_fit_rows:
        raise UploadRejected(
            _ERR_TOO_MANY_FIT_ROWS,
            code="too_many_fit_rows",
            context={"max_fit_rows": int(max_fit_rows), "n_rows": int(len(fit_pos))},
        )

    by_column: dict[str, dict[str, int]] = {}

    def _tally(column: str, reason: str, n: int) -> None:
        if n:
            row = by_column.setdefault(column, {})
            row[reason] = row.get(reason, 0) + int(n)

    def _tally_kinds(column: str, kinds: np.ndarray, *, blank_is_missing: bool) -> None:
        for kind, reason in (
            (MISSING, REASON_MISSING),
            (NON_NUMERIC, REASON_NON_NUMERIC),
            (CENSORED, REASON_CENSORED),
        ):
            _tally(column, reason, int(((kinds == kind) & active).sum()))
        if blank_is_missing:
            _tally(column, REASON_MISSING, int(((kinds == BLANK) & active).sum()))

    _tally_kinds(str(resolved_target), tgt_kinds, blank_is_missing=True)
    for c in cont_feats:
        _tally_kinds(str(c), feat_kind[c], blank_is_missing=str(c) not in absent_zero)
    for c in kept_cats:
        _tally(str(c), REASON_MISSING, int(((~cat_ok[c]) & active).sum()))
    zero_filled = {
        str(c): int((feat_kind[c][fit_pos] == BLANK).sum())
        for c in cont_feats
        if str(c) in absent_zero and int((feat_kind[c][fit_pos] == BLANK).sum())
    }

    if preflight_only:
        return None

    keep_index = df.index[fit_pos]
    y = y_all[fit_pos]
    Xc_raw = Xc_all.iloc[fit_pos]
    Xc_zf = Xc_raw.copy()
    for c in cont_feats:
        if str(c) in absent_zero:
            Xc_zf[c] = Xc_raw[c].fillna(0.0)
    if n_rows_dropped or n_censored_rows:
        _refuse_unusable_target(df.loc[keep_index], resolved_target, num, profile)

    dims: list[Dimension] = []
    cont_arr = Xc_zf.to_numpy(float) if cont_feats else np.empty((len(y), 0))
    box_exclusions: list[dict[str, Any]] = []
    for c in cont_feats:
        raw_col = Xc_raw[c].to_numpy(float)
        observed = raw_col[~np.isnan(raw_col)]
        n_absent = int(np.isnan(raw_col).sum())
        lower, upper, n_excluded = _physical_range(str(c), observed, absent=n_absent)
        if n_excluded:
            box_exclusions.append(
                {"column": str(c), "n_excluded": n_excluded, "lower": lower, "upper": upper}
            )
        dims.append(Dimension(str(c), "continuous", lower=lower, upper=upper))
    for c in kept_cats:
        dims.append(Dimension(str(c), "categorical", levels=cat_dims_map[c]))
    design = DesignSpace(tuple(dims))

    code_maps = {c: {lvl: i for i, lvl in enumerate(cat_dims_map[c])} for c in kept_cats}
    if kept_cats:
        cat_cols = [
            cell_labels(df.loc[keep_index, c]).map(code_maps[c]).to_numpy(float)
            for c in kept_cats
        ]
        X = np.column_stack([cont_arr, *cat_cols]) if cont_feats else np.column_stack(cat_cols)
    else:
        X = cont_arr
    bounds = design.bounds()
    cat_dims = design.cat_dims or None
    cat_cardinalities = design.cat_cardinalities or None
    strain = StrainState.build(
        strain_plan, df=df, keep_index=keep_index, X=X, bounds=bounds,
        cont_feats=cont_feats, kept_cats=kept_cats, cat_dims_map=cat_dims_map,
    )
    if setup is None:
        setup_plan = None
    elif strain.active:
        from kalos.core.optimize import MAX_MIXED_COMBOS

        setup_plan = plan_setup(
            setup, names=design.names, categorical=[str(c) for c in kept_cats], bounds=bounds,
            target=str(resolved_target), converted=[str(c.column) for c in validation.conversions],
            held=strain.held_names, mixed_constraints_ok=strain.mixed_constraints_ok(
                cat_dims or [], cat_cardinalities or [], MAX_MIXED_COMBOS),
        )
    else:
        setup_plan = plan_setup(
            setup, names=design.names, categorical=[str(c) for c in kept_cats], bounds=bounds,
            target=str(resolved_target), converted=[str(c.column) for c in validation.conversions],
        )
    minimize = setup_plan is not None and setup_plan.minimize

    return PreparedAnalysis(
        df=df,
        target=resolved_target,
        num=num,
        cont_feats=cont_feats,
        kept_cats=kept_cats,
        constant_cats=constant_cats,
        constant_on_fitted=constant_on_fitted,
        cat_dims_map=cat_dims_map,
        code_maps=code_maps,
        gcol=gcol,
        declared=declared,
        batch_cols=batch_cols,
        derived_outputs=derived_outputs,
        auto_cats=auto_cats,
        candidate_targets=candidate_targets,
        absent_zero=absent_zero,
        header_info=header_info,
        round_attr=round_attr,
        frame_index=frame_index,
        keep_index=keep_index,
        validation=validation,
        by_column=by_column,
        n_rows_dropped=n_rows_dropped,
        n_censored_rows=n_censored_rows,
        zero_filled=zero_filled,
        y=y,
        Xc_raw=Xc_raw,
        X=X,
        design=design,
        bounds=bounds,
        cat_dims=cat_dims,
        cat_cardinalities=cat_cardinalities,
        box_exclusions=box_exclusions,
        strain_plan=strain_plan,
        strain=strain,
        setup_plan=setup_plan,
        minimize=minimize,
    )
