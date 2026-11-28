"""Kalos portal — the science: run the engine on an arbitrary uploaded run sheet.

Outcome-like columns are candidate TARGETS and are excluded from input features
so the model never "predicts" titer from another measured output (leakage).

The analysis pipeline lives in `kalos.analysis.stages` (`prepare`, `trust`,
`propose`, `explain`, `assemble`). This module owns the seeded execution scope
and re-exports legacy helpers imported by callers and tests.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos.analysis.stages import (
    assemble_result,
    compute_drivers,
    compute_post_fit_diagnostics,
    evaluate_trust,
    explain_surrogate,
    prepare_analysis,
    run_proposals,
)
from kalos.core.gp_shape import gp_shape_report
from kalos.domains import BIOPROCESS_PROFILE, ColumnRoles, DomainProfile
from kalos.portal.analysis_parts.anonymize import (
    _anonymize_result,
    anonymize_parsed,
    anonymize_rejection,
    anonymize_sheet_rejection,
    sheet_alias,
)
from kalos.portal.analysis_parts.columns import (
    AUTO_CATEGORICAL_MAX_LEVELS,
    _dedupe_columns,
    _numeric_cols,
    _resolve_columns,
)
from kalos.portal.analysis_parts.diagnostics import (
    _data_drops_block,
    _data_findings,
    _with_data_findings,
)
from kalos.portal.analysis_parts.identifiers import (
    identifier_pattern,
    is_identifier_name,
)
from kalos.portal.analysis_parts.noise_blocks import (
    _alternative_scale_block,
    _noise_block,
)
from kalos.portal.analysis_parts.policy import (
    _ERR_VALIDATION_FAILED,
    ANALYZE_SEED,
    CONFORMAL_ALPHA,
    CONSTRAINT_MIN_ROWS,
    CV_N_REPEATS,
    CV_TOPK_K,
    DRIVER_FDR_Q,
    DRIVER_TOP_K,
    GATE_MIN_N_INFEASIBLE,
    GROUP_MEAN_BASELINE_MIN_GROUPS,
    LOGO_MAX_GROUPS,
    RELIABILITY_SPEARMAN_FLOOR,
    validation_mode,
)
from kalos.portal.analysis_parts.proposals import _annotate, _encode_pending
from kalos.portal.analysis_parts.ranges import _absent_is_zero_columns, _physical_range
from kalos.portal.analysis_parts.refusals import (
    _ERR_NO_PROCESS_INPUTS,
    _refuse_unusable_target,
    _too_few_rows,
    _validate_constraint,
)
from kalos.portal.uploads import MAX_FIT_ROWS

# Re-exported: each name below was defined in this module before the split into
# `kalos.portal.analysis_parts`; callers and tests keep importing it from here.
__all__ = [
    "ANALYZE_SEED", "AUTO_CATEGORICAL_MAX_LEVELS", "CONFORMAL_ALPHA", "CONSTRAINT_MIN_ROWS",
    "CV_N_REPEATS", "CV_TOPK_K", "DRIVER_FDR_Q", "DRIVER_TOP_K", "GATE_MIN_N_INFEASIBLE",
    "GROUP_MEAN_BASELINE_MIN_GROUPS", "LOGO_MAX_GROUPS", "RELIABILITY_SPEARMAN_FLOOR",
    "_ERR_NO_PROCESS_INPUTS", "_ERR_VALIDATION_FAILED", "_absent_is_zero_columns",
    "_alternative_scale_block", "_annotate", "_anonymize_result", "_data_drops_block",
    "_data_findings", "_dedupe_columns", "_encode_pending", "_noise_block", "_numeric_cols",
    "_physical_range", "_refuse_unusable_target", "_resolve_columns", "_too_few_rows",
    "_validate_constraint", "_with_data_findings", "anonymize_parsed", "anonymize_rejection",
    "anonymize_sheet_rejection", "identifier_pattern", "is_identifier_name", "sheet_alias",
    "validation_mode",
]

# NOTE: `kalos.core.evaluation`, `kalos.core.optimize`, `kalos.core.surrogate`,
# and `torch` are intentionally NOT imported at module level here or in
# `kalos.analysis.stages.*` so importing `kalos.portal.analysis` stays torch-free.
_RNG_LOCK = threading.RLock()
PROPOSAL_FIT_RESTARTS = 3


@contextlib.contextmanager
def _seeded_analysis(seed: int = ANALYZE_SEED) -> Iterator[None]:
    """Run one analysis with torch and numpy seeded to `seed`, then restore them."""
    import torch  # local: deferred to keep portal boot torch-free

    with _RNG_LOCK:
        np_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                np.random.seed(seed)
                yield
        finally:
            np.random.set_state(np_state)


def _analyze(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    anonymize: bool = False,
    roles: ColumnRoles | None = None,
    profile: DomainProfile = BIOPROCESS_PROFILE,
    pending: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    constraint: Mapping[str, Any] | None = None,
    progress: Callable[[str, float], None] | None = None,
    preflight_only: bool = False,
    user_target_unit: str | None = None,
    setup: Mapping[str, Any] | None = None,
) -> dict:
    """See `_analyze_body`. Every fitting analysis runs inside `_seeded_analysis`."""
    _validate_constraint(constraint)
    kwargs: dict[str, Any] = dict(
        anonymize=anonymize, roles=roles, profile=profile, pending=pending,
        constraint=constraint, progress=progress, preflight_only=preflight_only,
        user_target_unit=user_target_unit,
    )
    if setup is not None:
        kwargs["setup"] = setup
    if preflight_only:
        return _analyze_body(df, target, **kwargs)
    with _seeded_analysis(ANALYZE_SEED):
        return _analyze_body(df, target, **kwargs)


def _analyze_body(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    anonymize: bool = False,
    roles: ColumnRoles | None = None,
    profile: DomainProfile = BIOPROCESS_PROFILE,
    pending: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    constraint: Mapping[str, Any] | None = None,
    progress: Callable[[str, float], None] | None = None,
    preflight_only: bool = False,
    user_target_unit: str | None = None,
    setup: Mapping[str, Any] | None = None,
) -> dict:
    """Orchestrate the five analysis stages in fixed RNG-preserving order."""
    _progress = progress if progress is not None else (lambda _stage, _fraction: None)
    prep = prepare_analysis(
        df, target,
        roles=roles, profile=profile, constraint=constraint,
        preflight_only=preflight_only, setup=setup, max_fit_rows=MAX_FIT_ROWS,
    )
    if prep is None:
        return {}
    trust = evaluate_trust(
        prep, roles=roles, constraint=constraint, progress=_progress,
        gate_min_n_infeasible=GATE_MIN_N_INFEASIBLE,
    )
    drivers = compute_drivers(prep, trust)
    prop = run_proposals(
        prep, trust, drivers,
        pending=pending, progress=_progress, proposal_fit_restarts=PROPOSAL_FIT_RESTARTS,
    )
    explanation = explain_surrogate(
        prep, trust, prop, drivers, gp_shape_fn=gp_shape_report,
    )
    post_fit = compute_post_fit_diagnostics(
        prep, trust, prop, explanation,
        anonymize=anonymize, user_target_unit=user_target_unit, setup=setup,
    )
    return assemble_result(
        prep, trust, drivers, prop, explanation, post_fit,
        roles=roles, profile=profile, anonymize=anonymize,
        user_target_unit=user_target_unit, setup=setup,
    )
