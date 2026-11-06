"""Kalos portal — the science: run the engine on an arbitrary uploaded run sheet.

Outcome-like columns are candidate TARGETS and are excluded from input features
so the model never "predicts" titer from another measured output (leakage).
"""
from __future__ import annotations

import contextlib
import gc
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from kalos import __version__ as ENGINE_VERSION
from kalos.core.conformal import q_from_residuals
from kalos.core.drivers import (
    benjamini_hochberg,
    bootstrap_spearman,
    spearman_driver_matrix,
)
from kalos.core.gp_shape import gp_shape_report
from kalos.core.replicates import (
    aggregate_replicates,
    heteroscedasticity_report,
    noise_report,
)
from kalos.core.splits import row_hash_groups
from kalos.domains import (
    BIOPROCESS_PROFILE,
    ColumnRoles,
    DesignSpace,
    Dimension,
    DomainProfile,
    InvalidDirection,
    parse_direction,
)
from kalos.portal.analysis_parts.anonymize import (
    _anonymize_result,
    anonymize_parsed,
    anonymize_rejection,
    anonymize_sheet_rejection,
    sheet_alias,
)
from kalos.portal.analysis_parts.applicability import (
    applicability_block,
    proposal_applicability,
)
from kalos.portal.analysis_parts.best_bet import (
    label_proposals,
    place_best_bet,
    predicted_vs_best,
)
from kalos.portal.analysis_parts.column_kinds import column_kinds, with_blank_columns
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
from kalos.portal.analysis_parts.near_constant import (
    near_constant_findings,
    near_constant_inputs,
    with_fixed_inputs,
)
from kalos.portal.analysis_parts.noise_blocks import (
    _alternative_scale_block,
    _noise_block,
)
from kalos.portal.analysis_parts.policy import (
    _ERR_VALIDATION_FAILED,
    ANALYSIS_CONTRACT_VERSION,
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
from kalos.portal.analysis_parts.proposals import (
    _annotate,
    _encode_pending,
    nearest_training_distance,
    training_nn_distance_median,
)
from kalos.portal.analysis_parts.ranges import (
    _absent_is_zero_columns,
    _physical_range,
)
from kalos.portal.analysis_parts.refusals import (
    _ERR_NO_PROCESS_INPUTS,
    _refuse_unusable_target,
    _too_few_rows,
    _validate_constraint,
)
from kalos.portal.analysis_parts.roles import column_role_safety
from kalos.portal.analysis_parts.sanitize import sanitize_non_finite
from kalos.portal.cells import (
    BLANK,
    CENSORED,
    MISSING,
    NON_NUMERIC,
    NUMERIC,
    REASON_CENSORED,
    REASON_MISSING,
    REASON_NON_NUMERIC,
    classify_cells,
    is_null_label,
)
from kalos.portal.uploads import (
    _ERR_TOO_MANY_FIT_ROWS,
    MAX_FIT_ROWS,
    MIN_FIT_ROWS,
    UploadRejected,
    clip_name,
)
from kalos.portal.validate import column_provenance, provenance_dicts
from kalos.validation import apply_unit_conversions, report_dict, validate_frame
from kalos.validation.report import Finding

# Re-exported: each name below was defined in this module before the split into
# `kalos.portal.analysis_parts`; callers and tests keep importing it from here.
__all__ = [
    "ANALYSIS_CONTRACT_VERSION", "ANALYZE_SEED", "AUTO_CATEGORICAL_MAX_LEVELS", "CONFORMAL_ALPHA", "CONSTRAINT_MIN_ROWS",
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

# NOTE: `kalos.core.evaluation` (which imports `kalos.core.surrogate`),
# `kalos.core.optimize`, and `torch` itself are intentionally NOT imported at
# module level. This module is imported by `kalos.portal.app` (the portal),
# so a top-level torch import here would tax portal boot with the whole
# torch/botorch/gpytorch stack (~220 MB) before any analysis ever runs. They
# are imported lazily inside `_analyze`/`_analyze_body`, the only places
# that actually need them. `kalos.domains` above is torch-free by contract.
#
# This is load-bearing and easy to break by accident: it was broken, by a single
# top-level `from kalos.core.evaluation import producer_only_spearman` added for
# one call inside `_analyze`. `evaluation` imports `surrogate`, so importing this
# module pulled the whole stack (measured: 1.2s and `torch in sys.modules` right
# after `import kalos.portal.analysis`) and the poller paid the tax it was
# written to avoid. `tests/test_portal.py` now asserts the property instead of
# trusting this comment. Every name from `kalos.core.evaluation` belongs in the
# deferred block inside `_analyze`.


# torch's default generator and numpy's global RNG are PROCESS-wide. Two analyses
# running on two threads (busy.py admits two) would reseed each other mid-fit, so
# the proposals depended on which thread got there first (review C3). Every
# analysis holds this lock for its seeded scope, so the RNG is never shared.
_RNG_LOCK = threading.RLock()


# Hyperparameter starts for the PROPOSAL surrogate only (default start + 2
# seeded random starts; best marginal likelihood wins), 1 above
# MULTI_START_MAX_ROWS rows (`proposal_restarts`). CV folds keep one start.
# Measured cost: see docs/CAMPAIGNS_API.md "Model diagnostics".
PROPOSAL_FIT_RESTARTS = 3

# Proposals per analysis (the batch size handed to `propose_batch`).
PROPOSAL_BATCH_Q = 5


class _SignFlipped:
    """A fitted surrogate seen through `y -> -y`: `posterior` returns the
    negated mean (same sd); every other attribute is the wrapped surrogate's.
    Used on a minimize target so shapes and surfaces are in the original sign."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def posterior(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        mean, sd = self._inner.posterior(X)
        return -np.asarray(mean, float), sd

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@contextlib.contextmanager
def _seeded_analysis(seed: int = ANALYZE_SEED) -> Iterator[None]:
    """Run one analysis with torch and numpy seeded to `seed`, then restore them.

    The seed is applied inside `torch.random.fork_rng` and `np.random.get_state`
    snapshots, so the caller's global RNG state is put back on exit. The lock is
    held for the whole scope, so two concurrent analyses run one after the other
    on the RNG and give exactly the proposals they would give sequentially.
    Nothing here depends on the global RNG having been seeded beforehand.
    """
    import torch  # local: deferred, see module-level note above

    with _RNG_LOCK:
        np_state = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                np.random.seed(seed)
                yield
        finally:
            np.random.set_state(np_state)


def _recipe_label_columns(roles: ColumnRoles | None, gcol: Any, cat_feats: list) -> list:
    """The inferred categoricals that are recipe labels (see the note at the call
    in `_analyze_body`): the inferred group column, when it was also picked as a
    text factor. Empty when roles are declared."""
    if roles is None and gcol is not None and gcol in cat_feats:
        return [gcol]
    return []


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
    direction: str | None = None,
    applicability_distance: float | None = None,
) -> dict:
    """See `_analyze_body`. Every fitting analysis runs inside `_seeded_analysis`,
    so its proposals depend only on its inputs, not on other analyses in flight."""
    _validate_constraint(constraint)
    kwargs: dict[str, Any] = dict(
        anonymize=anonymize, roles=roles, profile=profile, pending=pending,
        constraint=constraint, progress=progress, preflight_only=preflight_only,
        direction=direction, applicability_distance=applicability_distance,
    )
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
    direction: str | None = None,
    applicability_distance: float | None = None,
) -> dict:
    """Run the engine on an arbitrary run sheet: pick the target (the value to
    maximize), use the process INPUTS as features (other measured outputs are
    excluded to avoid leakage), then honest grouped-CV, signed drivers, and a
    proposed next batch.

    Column roles come from an explicit `roles` schema when given, else are
    inferred from `profile` (defaulting to the bioprocess profile, which
    reproduces the original behavior). `roles.categoricals` marks feature columns
    whose values are unordered labels; the engine fits a mixed GP over them and
    proposals decode back to labels.

    `pending` carries the recipes that are already RUNNING but not yet measured
    (the campaign's awaiting runs). They have no target value, so they cannot
    join the fit; they are encoded into the same design and handed to the
    acquisition as in-flight points, so the proposed batch does not spend budget
    re-running an experiment currently in the incubator. Omit it (the default)
    and the proposal path is unchanged.

    `constraint` requests a constrained single-objective proposal: maximize
    `target` subject to a SECOND outcome column clearing a floor, e.g.
    `{"column": "purity_pct", "floor": 95.0}`. This is per-run intent, not a
    declared schema role (`ColumnRoles` is deliberately not the place - a run
    can ask for a constraint on one upload and not the next, on the same
    sheet). `None` (the default) leaves the proposal unconstrained. Whether
    the constraint could actually be applied (column present, enough rows,
    the constraint model fit) is reported in the response's `constraint`
    block regardless of the outcome - a constraint that could not be honored
    on this sheet falls back to an unconstrained proposal rather than
    raising; see PART 2 below for the policy. Wiring this parameter through
    the portal's HTTP surface is out of scope here.

    Seeded: torch + numpy are seeded up front. That makes small sheets repeat,
    but identical runs CAN give different proposals (measured on an 80x6 sheet,
    even in one process; tests/test_golden_determinism.py, docs/CAMPAIGNS_API.md
    "Determinism"). Returns a per-column `provenance` report (what was kept/dropped and
    why) plus `seed`, `timestamp`, and `engine_version` for audit. When
    `anonymize` is True, identifier-type column names are replaced with stable
    pseudonyms in the response (the owner UI keeps real names when False).

    `progress(stage, fraction)` (background jobs, docs/JOBS.md) is called at the
    stage boundaries: `fitting` (cross-validation fits), `calibrating`
    (interval calibration and diagnostics), `proposing` (final fit + batch).

    `preflight_only=True` runs only the refusals that precede the first model
    fit (shape, target, constant target, process inputs, row caps), then
    returns `{}` - without importing torch, seeding any RNG, or fitting. The
    job API uses it to refuse bad input with the sync route's 400 BEFORE
    queueing; it is the same code, stopped early, so the two cannot drift.

    `direction` ("maximize"/"minimize") is the plain request field; a direction
    stated in `roles` takes precedence, and two different stated directions
    are refused (`direction_conflict`). Neither stated: `profile.direction`
    (default "maximize"). Minimize fits the proposal surrogate on -y, so the
    acquisition, incumbent and explore/exploit labels look for LOW values.
    Everything reported (predictions, `best`, drivers, `gp_shapes`, CV) stays
    in the original units and sign. Feasibility gating (P(y > 0)) is not
    applied when minimizing.

    `applicability_distance` is a lab-set `far_from_data` threshold (the
    reanalyze passes the one the upload's roles set); `roles.applicability_distance`
    wins. Neither: `KALOS_APPLICABILITY_DISTANCE`, else the product default
    (kalos/portal/analysis_parts/applicability.py)."""
    try:
        stated_direction = parse_direction(direction)
    except InvalidDirection as exc:
        raise UploadRejected(str(exc), code=exc.code) from None
    roles_direction = roles.direction if roles is not None else None
    if stated_direction and roles_direction and stated_direction != roles_direction:
        raise UploadRejected(
            "The request states two different directions (roles.direction and direction). "
            "State one.",
            code="direction_conflict",
        )
    direction = roles_direction or stated_direction or profile.direction
    minimize = direction == "minimize"
    # The proposal surrogate is fit on sign * y, so "maximize" in the
    # acquisition means "minimize y" when sign is -1.
    sign = -1.0 if minimize else 1.0
    if not preflight_only:
        # Deferred: these transitively import torch/botorch/gpytorch (see the
        # module-level note above). This is the one place in this module that
        # actually needs them, so this is where the torch tax is paid.
        # `kalos.core.gates` and `kalos.core.feasibility` do not themselves
        # import torch, but they are only ever used from inside this function,
        # so they are kept in this same deferred block rather than opening a
        # second import site. Skipped by a preflight, which returns before any
        # of these names is used.
        from kalos.core.evaluation import (
            group_mean_baseline_spearman,
            group_mean_margin,
            grouped_cv_report,
            interval_calibration,
            logo_report,
            producer_only_spearman,
            top_k_overlap,
        )
        from kalos.core.feasibility import (
            FeasibilityClassifier,
            feasibility_cv_report,
            feasible_labels,
        )
        from kalos.core.gates import (
            NOT_APPLICABLE_WHEN_MINIMIZING,
            GatesConfig,
            check_gates,
        )
        from kalos.core.optimize import (
            NEAR_DUPLICATE_TOL,
            mixed_strategy,
            posterior_mean_optimum,
            proposal_separation,
            propose_batch,
        )
        from kalos.core.surrogate import FitError, Surrogate, proposal_restarts
        from kalos.portal.viz_utils import (
            _correlation_spearman,
            _embedding_pca,
            _response_surface_gp,
        )

    _progress = progress if progress is not None else (lambda _stage, _fraction: None)
    # Header decision from the upload parser (`uploads.choose_header`), read before
    # any frame operation below could drop the attrs.
    header_info: dict = dict(df.attrs.get("kalos_header") or {})
    shape_as_written = df.attrs.get("kalos_shape_as_written")
    original_columns = {str(c) for c in df.columns}
    n_header_columns = int(df.shape[1])
    # Which uploaded columns are entirely blank, in sheet order: they are dropped
    # here but still listed in `columns` (kind constant, reason all_blank).
    sheet_blank = [bool(df.iloc[:, i].isna().all()) for i in range(df.shape[1])]
    sheet_names = [str(c) for c in df.columns]
    df = _dedupe_columns(df.dropna(axis=1, how="all"))

    # Shape problems that no model can work around get a specific, actionable
    # refusal (never the parser's or numpy's own text). Checked before anything
    # else so a one-column or four-row upload says exactly that. Columns are
    # counted once empty ones are dropped - except on a sheet with no data rows
    # at all (a header line, or one line of prose in a .txt), where every column
    # is "empty" and the header's own width is the useful number.
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
        raise _too_few_rows(n_data_rows, as_written=shape_as_written)
    requested_target = roles.target if roles is not None else target
    if requested_target is not None and str(requested_target) not in {str(c) for c in df.columns}:
        if str(requested_target) in original_columns:
            # It was there, but every cell was blank (dropped above).
            raise UploadRejected(
                f"The target column {clip_name(requested_target)!r} has no values. No model "
                "can be fit and no proposals were generated. Fill in the measured "
                "outcome, or choose another target column.",
                code="target_empty",
                context={"target": str(requested_target)},
            )
        available = [
            str(c) for c in _numeric_cols(df) if profile.outcome_hint.search(str(c))
        ] or [str(c) for c in _numeric_cols(df)]
        raise UploadRejected(
            f"The requested target column {clip_name(requested_target)!r} is not in this sheet. "
            "Choose one of the sheet's columns as the target.",
            code="target_not_found",
            context={"target": str(requested_target), "available_targets": available},
        )

    # The validation gate runs FIRST, on the sheet as uploaded, before any
    # column is typed or dropped - it has to see the raw cells to catch a mixed
    # g/L-and-mg/mL column or a "34.6 C" string, both of which are invisible
    # once `_numeric_cols` has already discarded them as non-numeric.
    mode = validation_mode()
    validation = validate_frame(df, target=target, profile=profile, mode=mode)
    if mode == "strict" and validation.status == "fail":
        raise UploadRejected(_ERR_VALIDATION_FAILED)
    # Convert every single-unit column to its base unit before feature
    # selection. This is not merely cosmetic: a temperature column written
    # "34.6 C" fails the >=80% numeric-parse test and is silently DROPPED as
    # non-numeric today, so converting first recovers real process features
    # that were being thrown away. Only units the registry actually knows are
    # converted (see `check_units_consistency`).
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
    target, cont_feats, cat_feats, gcol, declared = _resolve_columns(
        df, num, target, roles, profile
    )
    # Column-role safety (analysis_parts/roles.py): row-number columns and
    # outcome-like columns after the target are not inputs unless declared.
    role_safety = column_role_safety(
        df, num, target, cont_feats, cat_feats, declared=declared, roles=roles,
        profile=profile, headerless=list(header_info.get("named_by_position") or []),
        target_given=requested_target is not None,
    )
    cont_feats, cat_feats = role_safety.cont_feats, role_safety.cat_feats
    # A RECIPE LABEL is not a factor. An inferred group column is chosen only when
    # each of its levels is one recipe of the other inputs (`_auto_group_column`),
    # so with no declared roles a text group column such as `recipe` = R1..R10
    # next to pH and temperature names the recipe: it groups replicates and CV
    # folds, but as a model input it only repeats what pH and temperature already
    # say, and a proposal cannot set it ("R3 at pH 7.9" is not a recipe). Used as
    # a categorical it also made the fit a mixed GP whose acquisition enumerates
    # every label (77-88 s on a 30-row sheet; docs/DESKTOP.md "Fit time").
    # Declared categoricals are never removed.
    recipe_labels = _recipe_label_columns(roles, gcol, cat_feats)
    cat_feats = [c for c in cat_feats if c not in recipe_labels]
    # With no declared roles, text factors are auto-detected (see `_auto_categorical`);
    # they are reported as findings so the uploader can confirm or override them.
    auto_cats = list(cat_feats) if roles is None else []
    # PART 2 leakage guard: the constraint column (e.g. "purity_pct") is
    # normally already excluded from the feature set by `profile.outcome_hint`
    # (the same mechanism that excludes the target itself), same as the
    # comment at `_resolve_columns` describes. But an explicit `roles.features`
    # list bypasses that hint entirely (see `_resolve_columns`'s declared-mode
    # branch), so a caller who both declares the constraint column as a
    # feature AND asks to constrain on it would otherwise leak that outcome
    # into the model it is meant to be held out from. Stripped here
    # unconditionally so the leakage rule holds regardless of how the column
    # got into the feature list.
    if constraint is not None:
        _constraint_col_name = str(constraint["column"])
        cont_feats = [c for c in cont_feats if str(c) != _constraint_col_name]
        cat_feats = [c for c in cat_feats if str(c) != _constraint_col_name]
    if roles is None:
        outcomes = [c for c in num if profile.outcome_hint.search(str(c))]
        candidate_targets = outcomes or [target]
    else:
        candidate_targets = [target]

    # A target that does not vary (every titer 5.0, or 5.0 +- float noise), or
    # has no numeric values at all, cannot support ANY model. Refuse here -
    # before a single surrogate is fit or `propose` is called - rather than
    # hand back a batch ranked by rounding error. Covers every caller of
    # `_analyze`: a fresh upload, a re-upload with a different `target`, and the
    # campaign re-analyze.
    _refuse_unusable_target(df, target, num, profile)

    # Keep only continuous features that actually vary (a constant column carries
    # no signal and would collapse its normalization range). Matches the legacy
    # full-column variance filter.
    varying = []
    for c in cont_feats:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.std(skipna=True) and col.std() > 1e-9:
            varying.append(c)
    cont_feats = varying
    if not cont_feats and not cat_feats:
        raise UploadRejected(_ERR_NO_PROCESS_INPUTS, code="no_process_inputs")

    # --- row accounting (D6) ------------------------------------------------- #
    # Every cell the model reads is classified over ALL rows first, so each drop
    # is reported by its real reason and column (`kalos.portal.cells`). Nothing is
    # imputed. A missing or non-numeric cell in the target or a fitted feature
    # drops its run. A censored value (<LOD, >100, <x, >x) excludes its run from
    # the fit and is counted separately. A blank feature cell keeps the
    # sparse-component convention (absent = 0): the run stays in, and the count is
    # reported. An entirely blank sheet row is not data and is not counted.
    tgt_kinds = classify_cells(df[target]).to_numpy(object)
    row_blank = np.ones(len(df), dtype=bool)
    for c in df.columns:
        row_blank &= classify_cells(df[c]).to_numpy(object) == BLANK
    active = ~row_blank
    y_all = pd.to_numeric(df[target], errors="coerce").to_numpy(float)
    keep = tgt_kinds == NUMERIC                      # rows with a usable target
    feat_kind = {c: classify_cells(df[c]).to_numpy(object) for c in cont_feats}
    Xc_all = df[cont_feats].apply(pd.to_numeric, errors="coerce") if cont_feats else pd.DataFrame(index=df.index)

    # Categorical levels come from the target-present rows. A blank or null-token
    # label is not a proposable level (it means "unknown"), so it cannot define one.
    kept_cats: list = []
    constant_cats: list = []
    cat_dims_map: dict = {}
    for c in cat_feats:
        labels = df.loc[keep, c].fillna("").astype(str).str.strip()
        levels = tuple(sorted({v for v in labels.tolist() if not is_null_label(v)}))
        if len(levels) < 2:
            constant_cats.append(c)
            continue
        kept_cats.append(c)
        cat_dims_map[c] = levels
    cat_ok = {
        c: df[c].fillna("").astype(str).str.strip().isin(cat_dims_map[c]).to_numpy()
        for c in kept_cats
    }

    # Blank feature cells are absent (0) only for sparse / concentration columns
    # (see `_absent_is_zero_columns`); for every other feature a blank is missing.
    absent_zero = _absent_is_zero_columns(cont_feats, roles)

    def _row_flags(features: list, cats: list) -> tuple[np.ndarray, np.ndarray]:
        """(incomplete, censored) per row, over every row of the sheet."""
        incomplete = np.isin(tgt_kinds, [BLANK, MISSING, NON_NUMERIC])
        censored = tgt_kinds == CENSORED
        for c in features:
            bad = [MISSING, NON_NUMERIC] if str(c) in absent_zero else [BLANK, MISSING, NON_NUMERIC]
            incomplete |= np.isin(feat_kind[c], bad)
            censored |= feat_kind[c] == CENSORED
        for c in cats:
            incomplete |= ~cat_ok[c]
        return incomplete, censored

    # Pass 1 decides which features are constant over the rows that will be fitted
    # (the honest varying-feature check, see the note below), pass 2 recomputes
    # the drops without the constant features.
    inc1, cens1 = _row_flags(cont_feats, kept_cats)
    fit1 = keep & ~inc1 & ~cens1
    constant_on_fitted: list = []
    if cont_feats and fit1.any():
        block = Xc_all[fit1].fillna(0.0)
        spread = block.max(axis=0) - block.min(axis=0)
        constant_on_fitted = [c for c in cont_feats if float(spread[c]) <= 1e-9]
        if constant_on_fitted:
            cont_feats = [c for c in cont_feats if c not in set(constant_on_fitted)]
            Xc_all = Xc_all.drop(columns=constant_on_fitted)
    # Inputs whose spread is within sensor precision are fixed at their median
    # (analysis_parts/near_constant.py), not searched.
    fixed_inputs: dict[str, dict] = {}
    if cont_feats and fit1.any():
        fixed_inputs = near_constant_inputs(Xc_all[fit1], cont_feats)
        if fixed_inputs:
            cont_feats = [c for c in cont_feats if str(c) not in fixed_inputs]
            Xc_all = Xc_all.drop(columns=[c for c in Xc_all.columns if str(c) in fixed_inputs])
    inc, cens = _row_flags(cont_feats, kept_cats)
    fit = keep & ~inc & ~cens
    fit_pos = np.flatnonzero(fit)
    if not cont_feats and not kept_cats:
        raise UploadRejected(_ERR_NO_PROCESS_INPUTS, code="no_process_inputs")

    n_rows_dropped = int((inc & active).sum())
    n_censored_rows = int((cens & ~inc & active).sum())
    if len(fit_pos) < MIN_FIT_ROWS:
        if int(keep.sum()) < MIN_FIT_ROWS:
            raise _too_few_rows(int(keep.sum()), str(target))
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
    # Cap the rows the O(n^2) exact GP is fit on, separately from the raw-upload
    # cap. Reject rather than subsample: silent subsampling would be invisible,
    # non-deterministic data loss and contradict the reproducibility contract.
    if len(fit_pos) > MAX_FIT_ROWS:
        # The message text is unchanged (the web matches its sentence); the code
        # and the two numbers let a client branch without parsing prose.
        raise UploadRejected(
            _ERR_TOO_MANY_FIT_ROWS,
            code="too_many_fit_rows",
            context={"max_fit_rows": int(MAX_FIT_ROWS), "n_rows": int(len(fit_pos))},
        )

    # Per-column tallies over the rows of the sheet that carry data, by reason.
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

    _tally_kinds(str(target), tgt_kinds, blank_is_missing=True)
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
        # Every refusal that does not need a fitted model has run.
        return {}

    keep_index = df.index[fit_pos]
    y = y_all[fit_pos]
    Xc_raw = Xc_all.iloc[fit_pos]                    # NaN preserved (for grouping)
    # Only absent-zero columns can still hold a blank here (every other feature's
    # incomplete rows were excluded above), so this zero-fill is the declared
    # absence convention and nothing else.
    Xc_zf = Xc_raw.copy()
    for c in cont_feats:
        if str(c) in absent_zero:
            Xc_zf[c] = Xc_raw[c].fillna(0.0)
    if n_rows_dropped or n_censored_rows:
        # Dropping rows can leave only rows that share one target value.
        _refuse_unusable_target(df.loc[keep_index], target, num, profile)

    # Assemble the design space with continuous dims first, categorical dims last.
    # BoTorch's mixed models accept arbitrary cat_dims indices, so this order is not
    # required; it just keeps X and the continuous driver slice `X[:, :n_cont]` simple.
    dims: list[Dimension] = []
    cont_arr = Xc_zf.to_numpy(float) if cont_feats else np.empty((len(y), 0))
    box_exclusions: list[dict] = []
    coded_columns = {f.column for f in validation.findings if f.code == "coded_column"}
    for c in cont_feats:
        # The box is built from OBSERVED cells only: blanks (absent-zero columns
        # only) are counted, not read as values.
        raw_col = Xc_raw[c].to_numpy(float)
        observed = raw_col[~np.isnan(raw_col)]
        n_absent = int(np.isnan(raw_col).sum())
        lower, upper, n_excluded = _physical_range(
            str(c), observed, absent=n_absent, coded=str(c) in coded_columns
        )
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
            df.loc[keep_index, c].fillna("").astype(str).str.strip().map(code_maps[c]).to_numpy(float)
            for c in kept_cats
        ]
        X = np.column_stack([cont_arr, *cat_cols]) if cont_feats else np.column_stack(cat_cols)
    else:
        X = cont_arr
    bounds = design.bounds()
    cat_dims = design.cat_dims or None
    cat_cardinalities = design.cat_cardinalities or None

    # PART 2: constrained single-objective proposals. Fit a SECOND `Surrogate`
    # over the constraint column, on the SAME design/bounds/cat_dims the target
    # surrogate uses, so `propose` can hand both to a `ModelListGP` (see
    # `kalos.core.optimize.propose`'s `constraint_surrogate` argument). This
    # never crashes and never blocks the (unconstrained) proposal: a missing
    # column, too few rows, or a `FitError` from an ill-conditioned constraint
    # fit all fall back to reporting why, not raising.
    #
    # Fit on rows where the constraint value is PRESENT, which is not
    # necessarily every row `X`/`y` cover - the constraint column can be
    # sparser than the target (e.g. purity was only measured on a subset of
    # runs). `X`'s rows are 1:1 with `keep_index` at this point (both were
    # finalized before `X` was assembled above), so a boolean mask read off
    # `df.loc[keep_index, constraint_col]` lines up with `X`'s rows directly.
    constraint_surrogate: Surrogate | None = None
    constraint_col: str | None = None
    constraint_floor_val: float | None = None
    constraint_applied = False
    if constraint is None:
        constraint_report: dict[str, Any] = {
            "column": None, "floor": None, "n_rows_with_value": 0,
            "applied": False, "reason": "no constraint requested",
        }
    else:
        constraint_col = str(constraint["column"])
        constraint_floor_val = float(constraint["floor"])
        if constraint_col not in df.columns:
            constraint_report = {
                "column": constraint_col, "floor": constraint_floor_val,
                "n_rows_with_value": 0, "applied": False,
                "reason": f"constraint column {constraint_col!r} not found in the uploaded sheet",
            }
        else:
            cy_all = pd.to_numeric(df.loc[keep_index, constraint_col], errors="coerce")
            c_mask = cy_all.notna().to_numpy()
            n_rows_with_value = int(c_mask.sum())
            if n_rows_with_value < CONSTRAINT_MIN_ROWS:
                constraint_report = {
                    "column": constraint_col, "floor": constraint_floor_val,
                    "n_rows_with_value": n_rows_with_value, "applied": False,
                    "reason": (
                        f"only {n_rows_with_value} rows have a numeric "
                        f"{constraint_col!r} value; need at least {CONSTRAINT_MIN_ROWS} "
                        "to fit a constraint model"
                    ),
                }
            else:
                Xc = X[c_mask]
                yc = cy_all.to_numpy(float)[c_mask]
                try:
                    constraint_surrogate = Surrogate().fit(Xc, yc, bounds=bounds, cat_dims=cat_dims)
                    constraint_applied = True
                    constraint_report = {
                        "column": constraint_col, "floor": constraint_floor_val,
                        "n_rows_with_value": n_rows_with_value, "applied": True, "reason": None,
                    }
                except FitError as exc:
                    constraint_report = {
                        "column": constraint_col, "floor": constraint_floor_val,
                        "n_rows_with_value": n_rows_with_value, "applied": False,
                        "reason": f"constraint model could not be fit: {exc}",
                    }

    # RECIPE identity, on the RAW values (NaN preserved) so rows missing
    # different components are never merged by the zero-fill. Built from the one
    # leakage-checked, deterministic grouper; categorical labels join the key so
    # identical recipes stay one replicate group.
    #
    # This is computed ALWAYS and kept SEPARATE from the CV group below, because
    # the two answer different questions and conflating them is a bug in either
    # direction. A declared group column is a LEAKAGE BARRIER and is deliberately
    # coarser than a recipe - every run sharing a medium lot goes in one fold, but
    # those runs are not replicates of each other. Using it as a recipe key would
    # collapse genuinely different recipes into one and fabricate within-recipe
    # variance; using a recipe key as the CV group would let a lot straddle folds.
    recipe_gf = Xc_raw.copy()
    for c in kept_cats:
        recipe_gf[c] = df.loc[keep_index, c].fillna("").astype(str)
    recipe_key = row_hash_groups(recipe_gf)

    if gcol:
        # fillna("") before astype(str), matching every other cast of this
        # column family in this file (recipe_gf just above, and the label
        # casts below) - NOT redundant here. Under pandas 3.x's default
        # string-backed dtype, .astype(str) on a Series that mixes real
        # strings with NaN no longer stringifies the NaN (legacy object-dtype
        # behavior); it leaves it as a bare float, so `groups` silently
        # carries one float among strings. That is invisible until
        # np.unique's sort at line ~1200 (LOGO group counting) raises
        # `TypeError: '<' not supported between instances of 'float' and
        # 'str'` - found live via a campaign reanalyze, since a folded
        # campaign row has no value for a declared group column like
        # `medium_base` (it is not a proposable numeric feature).
        groups = df.loc[keep_index, gcol].fillna("").astype(str).tolist()
    else:
        # No declared barrier, so the recipe itself is the safest CV group: two
        # replicates of one recipe must never straddle a fold.
        groups = recipe_key

    # honest grouped cross-validation: pooled out-of-fold predictions + a
    # group-level bootstrap CI, all through the single leakage-checked splitter.
    _progress("fitting", 0.15)
    rep = grouped_cv_report(
        X, y, groups=groups, n_splits=5, bounds=bounds, cat_dims=cat_dims, n_repeats=CV_N_REPEATS,
    )
    rho = rep["spearman"]
    # Repeat-0-only: the point estimate and the OOF pools below stay anchored to
    # the SAME unshuffled partition `grouped_cv_report` always uses for repeat 0,
    # so calibration/conformal/producer-only ranking all read off one consistent
    # set of predictions (see `grouped_cv_report`'s docstring for why pooling
    # across repeats here would double-count rows with correlated errors instead
    # of adding independent evidence). Only `cv_ci95` below draws on every
    # repeat's partition.
    oof_a, oof_p, oof_s = rep["oof_actual"], rep["oof_pred"], rep["oof_std"]

    # distribution-free +/- band from the pooled out-of-fold residuals (approximate
    # coverage under grouped CV). Honest alternative to the surrogate's own std,
    # which is often overconfident on small bioprocess datasets.
    resid = np.asarray(oof_a, float) - np.asarray(oof_p, float)
    conformal_q = round(q_from_residuals(resid, alpha=CONFORMAL_ALPHA), 4) if len(resid) else None

    # Honest reliability verdict: only what this path can actually assess. The
    # spearman floor mirrors GatesConfig.min_spearman (kalos/core/gates.py); we do
    # NOT assert the feasibility gate, which is not measured here.
    # The CI needs >= 3 CV groups for the group bootstrap; with fewer it comes
    # back NaN. Sent as None with a reason, never as NaN (replay F1: a NaN here
    # made the reanalyze response a 500).
    _ci_lo, _ci_hi = (float(v) for v in rep["ci95"])
    ci95 = (
        [round(_ci_lo, 3), round(_ci_hi, 3)]
        if rho == rho and np.isfinite(_ci_lo) and np.isfinite(_ci_hi)
        else None
    )
    ci95_reason = None if ci95 is not None else (
        f"The 95% interval for the CV Spearman needs at least 3 CV groups; this sheet has "
        f"{int(rep['n_groups'])}. It is reported as null."
        if rho == rho
        else "The CV Spearman could not be computed on this sheet, so it has no 95% interval."
    )

    # CALIBRATION. A model can rank held-out runs correctly and still state every
    # uncertainty at half its true width, and the scientist reading "4.2 +/- 0.3"
    # is acting on the 0.3. The held-out posterior sd needed to check that was
    # already computed inside the CV and then thrown away, so calibration was
    # reported as unmeasured for want of a value the engine had already paid for.
    # It is now kept and scored: `ece` is the mean gap between nominal and
    # empirical coverage across four central intervals, on the same 0-to-1 scale
    # `GatesConfig.max_ece` is written against, and `z_std` says which way a
    # miscalibrated model errs (above 1.0 = bands too narrow).
    #
    # Reported, NOT folded into `clears_floor`. Tightening the verdict changes
    # which uploads the API accepts, which is a product decision - the same line
    # `producer_clears_floor` sits on.
    _progress("calibrating", 0.55)
    calibration = interval_calibration(oof_a, oof_p, oof_s)

    # PRODUCER-ONLY ranking. The pooled `rho` above is taken over every held-out
    # row, producers and non-producers together, so a model can score well on it
    # by separating zeros from non-zeros - a feasibility classifier, not a ranking
    # of recipes. The client's question is "which of my producing recipes is
    # best", and BENCHMARK.md shows the two can diverge badly: on the real media
    # DoE the pooled score looked like 0.37-0.52 while feasibility was never the
    # bottleneck, so most of that agreement was the easy half of the problem.
    #
    # The threshold is 0.0, meaning "any non-zero measurement", which is a PROXY.
    # The assay LOD is the correct value - below it a reading is censored rather
    # than zero - and is not available until an SOP supplies it.
    prod = producer_only_spearman(oof_a, oof_p)
    reliability = {
        "spearman": None if rho != rho else round(rho, 3),
        "ci95": ci95,
        "spearman_floor": RELIABILITY_SPEARMAN_FLOOR,
        "clears_floor": bool(rho == rho and rho >= RELIABILITY_SPEARMAN_FLOOR),
        # Reported alongside, not folded into `clears_floor`: making the gate
        # stricter changes which uploads the API accepts, which is a product
        # decision rather than a bug fix. Surfaced so the divergence is visible
        # and the gate can be tightened deliberately.
        "producer_spearman": None if prod["spearman"] != prod["spearman"] else round(float(prod["spearman"]), 3),
        "producer_clears_floor": bool(
            prod["evaluable"] and float(prod["spearman"]) >= RELIABILITY_SPEARMAN_FLOOR
        ),
        "n_producers": int(prod["n_producers"]),
        "producer_threshold": float(prod["threshold"]),
        "ci_excludes_zero": bool(ci95 is not None and ci95[0] > 0),
        # Held-out coverage of the GP's own Gaussian bands. `available` is false
        # on a sheet with too few out-of-fold points for a coverage rate to mean
        # anything, and the matching "calibration" line stays in `unmodeled`.
        "calibration": calibration,
        "unmodeled": [
            "feasibility probability",
            *(
                []
                if calibration["available"]
                else ["calibration (ECE): " + str(calibration["reason"])]
            ),
            "scale-up transfer",
            *(
                []
                if prod["evaluable"]
                else ["producer ranking (too few producing runs to score it separately)"]
            ),
        ],
    }

    # PROMOTION VERDICT. `check_gates` (kalos/core/gates.py) has existed since
    # the lean-engine port but had ZERO callers on this path, and
    # `feasibility_cv_auc` had zero callers outside its own tests - the
    # fail-closed gate the codebase already built was never actually asked
    # anything. This wires it up, WITHOUT changing what the API accepts: the
    # verdict is REPORTED, never used to reject an upload. Gating uploads on a
    # promotion verdict is a product decision (same line `producer_clears_floor`
    # and `clears_floor` already sit on above) that this change does not make.
    #
    # feasibility_cv_report is cheap - a handful of sklearn LogisticRegression
    # fits under CV, no GP involved - unlike the grouped surrogate CV above, so
    # computing it here costs essentially nothing next to the GP fits this
    # function already pays for.
    feas = feasibility_cv_report(X, y, groups=recipe_key)

    # PART 1: feasibility-gated production proposals. `feas` above already
    # cross-validates the classifier for the promotion verdict below; this
    # section decides, from that same CV measurement plus a fresh full-data
    # fit, whether the PROPOSAL acquisition itself should be gated by
    # P(feasible) - a separate question from "is this classifier fit to
    # promote" (the two verdicts can legitimately disagree: a promotable
    # classifier on a producer-only sheet still should not gate, because there
    # is nothing zero-inflated to gate against).
    #
    # GATE POLICY - all three required, each independently reported (never
    # silent about why a run was not gated):
    #
    # (a) the classifier actually FIT on this sheet, not the cold-start
    #     fallback (`FeasibilityClassifier.fitted`, a clean public accessor -
    #     see its docstring) - gating on the fallback (a uniform P(feasible)=1)
    #     would multiply every candidate's EI by 1 and change nothing while
    #     still claiming to gate.
    # (b) the sheet is zero-inflated enough: `n_infeasible >=
    #     GATE_MIN_N_INFEASIBLE` (see that constant's comment above for the
    #     5-row justification).
    # (c) the CV feasibility AUC (`feas["auc"]`, reused rather than a second CV
    #     run) clears `GatesConfig().min_feasibility_auc` (0.65) - the SAME bar
    #     already used for the promotion verdict below (see
    #     `GATE_MIN_N_INFEASIBLE`'s comment above for why no second number is
    #     defined for this).
    #
    # When any condition fails, `gated=False` and `propose()` below is called
    # exactly as it always was (no `feasibility_classifier` argument at all) -
    # this is the no-regression guarantee: a producer-only or too-small sheet's
    # proposed batch is byte-identical to before this change.
    feas_clf = FeasibilityClassifier().fit(X, feasible_labels(y))
    n_infeasible = int((feasible_labels(y) == 0).sum())
    feas_auc = feas["auc"]
    _gate_auc_floor = GatesConfig().min_feasibility_auc
    _gate_fail_reasons: list[str] = []
    if not feas_clf.fitted:
        _gate_fail_reasons.append(
            "feasibility classifier did not fit (cold-start fallback: too few "
            "non-producer examples to train on)"
        )
    if n_infeasible < GATE_MIN_N_INFEASIBLE:
        _gate_fail_reasons.append(
            f"not zero-inflated enough: n_infeasible={n_infeasible} < {GATE_MIN_N_INFEASIBLE}"
        )
    _auc_ok = feas_auc == feas_auc and feas_auc >= _gate_auc_floor  # NaN-safe
    if not _auc_ok:
        _gate_fail_reasons.append(
            "feasibility CV AUC not measurable on this sheet"
            if feas_auc != feas_auc
            else f"feasibility CV AUC too low to trust: auc={feas_auc:.3f} < {_gate_auc_floor}"
        )
    if minimize:
        _gate_fail_reasons.append(
            "not applied when the direction is minimize: P(feasible) scores y > 0, "
            "a maximize notion"
        )
    gated = (
        feas_clf.fitted and n_infeasible >= GATE_MIN_N_INFEASIBLE and _auc_ok and not minimize
    )
    gate_reason = (
        "; ".join(_gate_fail_reasons)
        if _gate_fail_reasons
        else (
            f"zero-inflated (n_infeasible={n_infeasible}) with a trustworthy "
            f"feasibility classifier (auc={feas_auc:.3f} >= {_gate_auc_floor}); "
            "gating acquisition by P(feasible)"
        )
    )

    # gate_stats assembles the four keys `check_gates` reads by name
    # (kalos/core/gates.py: surrogate_spearman, feasibility_auc, ece, brier).
    #
    # DELIBERATE CHOICE, stated here because it is easy to get backwards: the
    # `ece` fed to the gate is `feas["ece"]` - the FEASIBILITY CLASSIFIER's
    # expected calibration error (is the stated P(feasible) itself honest) -
    # NOT `calibration["ece"]` from `reliability` above, which is the
    # REGRESSION interval calibration ECE (are the titer error bars honest).
    # Both are legitimately named "ece", both live on the same 0-to-1 scale,
    # and they answer different questions. `GatesConfig.max_ece` was written
    # in the lean engine against the classifier metric - promotion asks "is
    # this model's feasibility judgment trustworthy enough to gate
    # acquisition on", not "are this model's regression error bars the right
    # width" - so the classifier's `ece` is what belongs here. Naming both in
    # this comment is the whole point: nothing downstream should ever swap
    # them without knowing it changed the question being asked.
    # GROUP-MEAN FLOOR AS A GATE (compmodel PLAN E13): the surrogate must beat
    # the leave-one-row-out recipe mean by more than a paired bootstrap's
    # uncertainty (lower 95% bound of the paired Spearman difference > 0).
    # Not judged when too few recipes are replicated to form the floor.
    _n_rep_groups = int((pd.Series(recipe_key).value_counts() >= 2).sum())
    if _n_rep_groups < GROUP_MEAN_BASELINE_MIN_GROUPS:
        gm_margin: dict[str, Any] = {
            "evaluable": False, "passed": None,
            "reason": (
                f"only {_n_rep_groups} recipes have 2+ rows; need at least "
                f"{GROUP_MEAN_BASELINE_MIN_GROUPS} to form a group-mean floor"
            ),
        }
    else:
        gm_margin = group_mean_margin(y, recipe_key, rep["oof_index"], oof_p)
    gate_stats = {
        "surrogate_spearman": rho,  # the pooled CV rho already computed above
        "group_mean_margin": gm_margin,
        "feasibility_auc": feas["auc"],
        "ece": feas["ece"],
        # How many pooled out-of-fold runs the classifier ECE was scored on:
        # below GatesConfig.min_ece_n (20) the ECE is reported, not judged.
        "ece_n": feas["n"] if feas.get("evaluable") else None,
        "brier": feas["brier"],
    }
    if minimize:
        # P(feasible) scores y > 0, a maximize notion: its AUC/ECE/Brier gates are
        # reported as not judged rather than passing or failing a minimize run.
        gate_stats["feasibility_not_applicable"] = NOT_APPLICABLE_WHEN_MINIMIZING
    promotion_result = check_gates(gate_stats)
    promotion = {
        "passed": promotion_result.passed,
        "failures": promotion_result.failures,
        # `check_gates` only ever inserts a value into its `metrics` dict when
        # `_finite` accepts it (isinstance float/int, not bool, math.isfinite),
        # so this is strict-JSON safe by construction - no NaN can reach here.
        "metrics": promotion_result.metrics,
        "summary": promotion_result.summary,
        # Gates neither passed nor failed, e.g. "ece: not enough runs to judge
        # calibration (n=14, need 20)". Additive.
        "not_judged": promotion_result.not_judged,
        # Additive: `passed`, `failed` (a measured gate missed), or
        # `not_measurable` (every blocking gate was unmeasured, e.g. no
        # non-producers to score feasibility). Same verdict as `passed`.
        "status": promotion_result.status,
        "unmeasured": list(promotion_result.unmeasured),
        # FAIL-CLOSED IS CORRECT, BUT ILLEGIBLE WITHOUT THIS. On an all-producer
        # sheet (no non-producers to build a feasibility classifier from) or
        # any sheet too small to measure one of the four gates,
        # `feasibility_auc`/`ece`/`brier` come back `nan`, `check_gates` fails
        # closed with "missing/NaN (unmeasured)" failures, and `passed` reads
        # `False` - which is the RIGHT answer (an unmeasured gate must block,
        # not silently pass), but reads exactly like a rejection if nothing
        # explains it. This fixed string is that explanation. It does not
        # soften the verdict - `passed` and `failures` are unchanged - it only
        # states what a `False` here does and does not mean.
        "meaning": (
            'passed=false means "not yet shown fit to promote", which '
            "includes the case where a required metric is unmeasurable on "
            "this data (e.g. an all-producer sheet with no non-producers to "
            "score feasibility against). It is not a rejection of the analysis."
        ),
    }

    # LEAVE-ONE-GROUP-OUT. Grouped K-fold above tests generalization to a FEW
    # unseen groups among many familiar ones; LOGO is the harder question -
    # generalization to the NEXT group the model has seen nothing like - and
    # the two can diverge sharply (see `logo_report`'s docstring: -0.12 vs 0.91
    # on the owner's real clone-selection data). Only run when there is a
    # DECLARED/detected group column (`gcol` is not None): the recipe-hash
    # fallback groups by feature identity, so leaving one recipe-hash group out
    # would just be a costlier copy of the CV already computed above, not a
    # harder question. Also capped at `LOGO_MAX_GROUPS` groups, since LOGO
    # costs one GP fit per group (see that constant's comment for the measured
    # timing this cap is tuned against) - absence is always explained via
    # `reason`, never silent.
    cv_logo: dict[str, Any]
    if gcol is None:
        cv_logo = {
            "spearman": None,
            "n_groups": None,
            "n_oof": None,
            "reason": (
                "no declared/detected group column; leave-one-group-out over "
                "the recipe-hash fallback would just be a costlier copy of "
                "the grouped CV already reported as cv_spearman"
            ),
        }
    else:
        _grp = pd.Series(groups).astype(str).to_numpy()
        n_logo_groups = len(np.unique(_grp))
        # The group column labels exactly the recipes (one group per recipe and
        # one recipe per group): leaving one group out is then the grouped CV
        # above with smaller folds, one GP fit per recipe for no new question.
        _same_as_recipes = (
            n_logo_groups == len(set(recipe_key))
            == len(set(zip(_grp.tolist(), list(recipe_key), strict=True)))
        )
        if n_logo_groups > LOGO_MAX_GROUPS:
            cv_logo = {
                "spearman": None,
                "n_groups": n_logo_groups,
                "n_oof": None,
                "reason": (
                    f"{n_logo_groups} groups exceeds the leave-one-group-out cap "
                    f"of {LOGO_MAX_GROUPS} groups (one GP fit per held-out group)"
                ),
            }
        elif _same_as_recipes:
            cv_logo = {
                "spearman": None,
                "n_groups": n_logo_groups,
                "n_oof": None,
                "reason": (
                    "the group column labels exactly the recipes, so leaving one "
                    "group out would repeat the grouped CV already reported as "
                    "cv_spearman"
                ),
            }
        else:
            logo = logo_report(X, y, groups, bounds=bounds, cat_dims=cat_dims)
            logo_rho = logo["spearman"]
            cv_logo = {
                "spearman": None if logo_rho != logo_rho else round(float(logo_rho), 3),
                "n_groups": logo["n_groups"],
                "n_oof": logo["n_oof"],
                "reason": None,
            }

    # TOP-K OVERLAP. The client's actual decision is "which k recipes do I
    # advance", not "how well-ranked is the whole held-out set" - Spearman can
    # look fine while the model gets exactly the recipes that matter out of
    # order (see `top_k_overlap`'s docstring). Computed on the same repeat-0
    # pooled OOF everything else in this function reads off. Under minimize the
    # recipes that matter are the LOWEST k, so both sides are ranked on sign * y.
    topk = top_k_overlap(sign * np.asarray(oof_a, float), sign * np.asarray(oof_p, float), CV_TOPK_K)
    cv_topk: dict[str, Any] = {
        "overlap": None if not topk["evaluable"] else round(float(topk["overlap"]), 3),
        "k": topk["k"],
        "n": topk["n"],
        "evaluable": bool(topk["evaluable"]),
        "reason": topk["reason"],
    }

    # GROUP-MEAN BASELINE FLOOR. How much of the apparent CV signal is just the
    # model recognizing which recipe a row belongs to, rather than modeling the
    # process? (See `group_mean_baseline_spearman`'s docstring: on the owner's
    # real data this floor alone captured ~0.75 of a trained model's ~0.86
    # headline.) Computed over `recipe_key` - the same replicate grouping the
    # noise floor and driver panel already use - only when enough groups are
    # actually replicated to make the leave-one-row-out means a real signal
    # rather than noise from a couple of pairs.
    n_groups_with_reps = int((pd.Series(recipe_key).value_counts() >= 2).sum())
    cv_group_mean_baseline: dict[str, Any]
    if n_groups_with_reps < GROUP_MEAN_BASELINE_MIN_GROUPS:
        cv_group_mean_baseline = {
            "spearman": None,
            "n_evaluable": 0,
            "n_groups_used": 0,
            "interpretation": None,
            "reason": (
                f"only {n_groups_with_reps} recipes have 2+ rows; need at "
                f"least {GROUP_MEAN_BASELINE_MIN_GROUPS} to compute a "
                "group-mean floor"
            ),
        }
    else:
        gmb = group_mean_baseline_spearman(y, recipe_key)
        if gmb["evaluable"]:
            cv_group_mean_baseline = {
                "spearman": round(float(gmb["spearman"]), 3),
                "n_evaluable": gmb["n_evaluable"],
                "n_groups_used": gmb["n_groups_used"],
                "interpretation": (
                    "A model whose CV Spearman does not clearly beat this "
                    "floor may be recognizing recipes, not modeling the "
                    "process."
                ),
                "reason": None,
            }
        else:
            cv_group_mean_baseline = {
                "spearman": None,
                "n_evaluable": gmb["n_evaluable"],
                "n_groups_used": gmb["n_groups_used"],
                "interpretation": None,
                "reason": gmb["reason"],
            }

    # signed drivers, each with a bootstrap 95% CI so the client can tell a real
    # driver from noise. A driver whose CI straddles zero is NOT distinguishable
    # from no-correlation at this sample size; the UI must not present it as a
    # finding. (Same honesty contract as `reliability.ci_excludes_zero` above.)
    # `rho` is the sample Spearman POINT estimate (consistent with the point
    # estimate `reliability.spearman` reports); the bootstrap supplies only the CI.
    # Drivers are computed over the CONTINUOUS features only: a signed Spearman
    # rank correlation on an integer-coded nominal category is not a meaningful
    # "driver", so categorical dims are deliberately excluded here. Their
    # continuous indices (0..len(cont_feats)-1) align with the leading columns of X.
    # MULTIPLICITY. Every continuous feature is tested, then the strongest are
    # selected and shipped. With an uncorrected per-feature 95% CI that is a
    # machine for printing false process insights: over 400 simulated reports on
    # 30 pure-noise features against an independent target, 78.8% contained at
    # least one "significant" driver. Under Benjamini-Hochberg at q=0.05 that
    # falls to 3.5%. Selection on the same data makes it worse than the raw rate
    # suggests, because ranking by |rho| preferentially surfaces exactly the
    # flukes. So `significant` now requires BOTH tests to agree:
    #   - the bootstrap CI excludes zero (is it distinguishable from no-effect?)
    #   - the feature survives BH across ALL tested features (is it still a
    #     finding once we account for how many features we looked at?)
    # BH runs over every tested feature BEFORE the top-k cut, never after -
    # correcting for 8 tests when 30 were performed would understate the very
    # multiplicity it exists to control.
    #
    # THE UNIT OF ANALYSIS IS THE RECIPE, NOT THE ROW. Both tests above ask "how
    # surprising is this association, given how much independent evidence there
    # is?", and on a replicated sheet a row is not independent evidence: three
    # wells of one recipe carry one recipe's worth of information about the
    # process, plus three draws of assay noise. Testing rows counts them as three,
    # so the p-value that feeds BH is computed against an `n` the sheet does not
    # have, and BH stops correcting anything.
    #
    # That is not a rounding error. Re-running the 30-noise-feature simulation on
    # 20 recipes x 3 replicates (300 reports, the replication depth a media DoE
    # actually ships with) put a "significant" driver in 87.0% of reports, against
    # 7.3% for the same 60 rows drawn independently. Collapsing replicates to
    # recipe means first brings it back to 8.0%; a cluster bootstrap alone does
    # not (82.5%), because the row-level p-values are what BH is reading.
    #
    # So the panel is computed on replicate-averaged rows, keyed by the same
    # `recipe_key` the CV grouping and the noise floor already use. On a sheet
    # with no replicates every group is a singleton and `aggregate_replicates` is
    # an exact no-op, so unreplicated uploads are unchanged.
    drv: list[dict[str, Any]] = []
    n_tested = len(cont_feats)
    n_driver_units = len(y)
    if cont_feats:
        names = [str(c) for c in cont_feats]
        Xr, yr, _rv, _rn = aggregate_replicates(X, y, groups=recipe_key)
        n_driver_units = int(Xr.shape[0])
        Xcont = Xr[:, : len(cont_feats)]
        matrix = spearman_driver_matrix(Xcont, yr, feature_names=names)
        point = np.asarray(matrix["rho"]).astype(float)
        pvals = np.asarray(matrix["pvals"]).astype(float)
        bh_survives = benjamini_hochberg(pvals, q=DRIVER_FDR_Q)
        boot = bootstrap_spearman(Xcont, yr, feature_names=names)
        boot_lo = np.asarray(boot["lo"], dtype=float)
        boot_hi = np.asarray(boot["hi"], dtype=float)
        for j, c in enumerate(cont_feats):
            lo, hi = round(float(boot_lo[j]), 3), round(float(boot_hi[j]), 3)
            ci_excludes_zero = bool(lo > 0 or hi < 0)
            drv.append(
                {
                    "_idx": j,
                    "name": str(c),
                    "rho": round(float(point[j]), 3),
                    "ci95": [lo, hi],
                    "p": round(float(pvals[j]), 6),
                    # Both components are reported, not just the verdict, so a
                    # reviewer can see WHICH test a borderline feature failed.
                    "ci_excludes_zero": ci_excludes_zero,
                    "survives_fdr": bool(bh_survives[j]),
                    # significance uses the SAME rounded bounds the client sees, so
                    # the flag never disagrees with the displayed CI. Two-sided: a
                    # strong negative driver is a finding too.
                    "significant": bool(ci_excludes_zero and bh_survives[j]),
                }
            )
    drv.sort(key=lambda d: -abs(float(d["rho"])))
    drv = drv[:DRIVER_TOP_K]

    # Replicate structure + assay noise floor (the "SNR lever", BENCHMARK.md).
    # Media DoE sheets are frequently heavily replicated because the ASSAY is
    # noisy, not the process. Scoring/optimizing single measurements rewards
    # lucky noise spikes - which is why undirected search can out-score BO on
    # such data. When replicates are present we instead fit the PROPOSAL
    # surrogate on the reproducible (replicate-averaged) titer and hand the GP
    # the measured assay noise floor as fixed per-recipe observation variance
    # (sigma^2 / n_reps, the variance of each recipe mean), so it stops chasing
    # spikes. This is the configuration the benchmark shows actually beats random
    # on the real data. Diagnostics (grouped-CV reliability, drivers) stay on the
    # raw rows - they are already replicate-grouped for leakage and describe the
    # as-measured signal; only the proposed batch switches to the reproducible
    # objective.
    nr = noise_report(X, y, groups=recipe_key)
    # Is the homoscedastic assumption behind that noise floor actually true? A
    # diagnostic only: nothing is transformed here, because silently changing the
    # target's scale would change every number the engine reports.
    het = heteroscedasticity_report(X, y, groups=recipe_key)
    best_single = float(y.min()) if minimize else float(y.max())
    best_reproducible: float | None = None
    replicate_aware = False
    incumbent = best_single
    # The rows the PROPOSAL surrogate is fit on. They stay the raw rows unless the
    # replicate-aware swap below happens, and they are what the shape report has
    # to be handed - see the `gp_shape_report` call for why that matters.
    X_fit, y_fit = X, y
    if nr["n_replicated"] >= 1:
        Xf, yf, _yvar_g, n_reps_g = aggregate_replicates(X, y, groups=recipe_key)
        best_reproducible = float(yf.min()) if minimize else float(yf.max())
        sigma2 = nr["noise_var"]
        # Only swap in the reproducible objective when there are enough distinct
        # recipes to fit a meaningful GP and a real, positive noise floor to feed;
        # otherwise fall through to the unchanged raw fit (still reported honestly).
        if nr["n_recipes"] >= 6 and np.isfinite(sigma2) and sigma2 > 0:
            replicate_aware = True
            per_point_var = float(sigma2) / np.maximum(n_reps_g, 1).astype(float)
            s = Surrogate().fit(
                Xf, sign * yf, bounds=bounds, noise=per_point_var, cat_dims=cat_dims,
                check_conditioning=True,
                restarts=proposal_restarts(len(yf), Xf.shape[1], PROPOSAL_FIT_RESTARTS),
                restart_seed=ANALYZE_SEED,
            )
            incumbent = best_reproducible
            X_fit, y_fit = Xf, yf

    # proposed next batch, with predicted target + uncertainty + a why per row
    _progress("proposing", 0.8)
    if not replicate_aware:
        s = Surrogate().fit(
            X, sign * y, bounds=bounds, cat_dims=cat_dims, check_conditioning=True,
            restarts=proposal_restarts(len(y), X.shape[1], PROPOSAL_FIT_RESTARTS),
            restart_seed=ANALYZE_SEED,
        )
    # In-flight recipes, encoded into this design so the acquisition treats them
    # as taken. Encoding can drop rows (a recipe naming an unknown categorical
    # level cannot be placed in the design at all), so the count actually used is
    # reported rather than assumed equal to what the caller passed.
    X_pending, n_pending_dropped = _encode_pending(
        pending, cont_feats, kept_cats, code_maps, X.shape[1],
        absent_zero=absent_zero & set(map(str, cont_feats)),
    )
    n_pending = 0 if X_pending is None else int(X_pending.shape[0])
    # `propose_batch` never returns two near-duplicate recipes (see its
    # docstring); on a design space with fewer than q distinct candidates the
    # batch comes back SHORT, and `proposal_notice` below says so.
    proposal = propose_batch(
        s,
        bounds,
        q=PROPOSAL_BATCH_Q,
        cat_dims=cat_dims,
        cat_cardinalities=cat_cardinalities,
        pending=X_pending,
        # Explicit, not the global RNG: the seed the run record reports is the seed
        # the acquisition actually used (review C3).
        seed=ANALYZE_SEED,
        # PART 1: only ever passed when GATE POLICY (above) cleared all three
        # conditions - `None` here is what keeps an ungated run's batch
        # byte-identical to the pre-gating acquisition.
        feasibility_classifier=feas_clf if gated else None,
        # PART 2: only ever passed when the constraint model actually fit.
        constraint_surrogate=constraint_surrogate if constraint_applied else None,
        constraint_floor=constraint_floor_val if constraint_applied else None,
    )
    batch = proposal.X
    # BEST BET (first-time-user study, Finding 1; analysis_parts/best_bet.py): the
    # recipe with the highest predicted value inside the box is always in the
    # batch, first, unless it is already running. Computed after the batch so the
    # batch acquisition and its seeds are unchanged.
    optimum_x = posterior_mean_optimum(
        s, bounds, cat_dims=cat_dims, cat_cardinalities=cat_cardinalities, seed=ANALYZE_SEED,
    )
    if len(batch):
        _batch_model_mean, _ = s.posterior(batch)
    else:
        _batch_model_mean = np.empty(0)
    batch, best_bet_idx, optimum_running = place_best_bet(
        batch, optimum_x, _batch_model_mean, bounds, cat_dims, X_pending, PROPOSAL_BATCH_Q,
    )
    opt_mean, opt_std = s.posterior(optimum_x.reshape(1, -1))
    _best_row = X[int(np.argmin(y) if minimize else np.argmax(y))]
    near_best_run = proposal_separation(
        optimum_x, _best_row, bounds[0], bounds[1], list(cat_dims or []),
    ) < NEAR_DUPLICATE_TOL
    show = [d["name"] for d in drv[:4]]
    show_idx = [d["_idx"] for d in drv[:4]]  # carry the feature index, not a name lookup
    for d in drv:
        d.pop("_idx", None)  # internal-only; not part of the returned API surface
    p_mean, p_std = s.posterior(batch)
    p_mean = sign * np.asarray(p_mean, float)  # back to original units and sign
    # Shapes and the response surface read the posterior in the original sign.
    s_report = _SignFlipped(s) if minimize else s

    # PART 1 per-proposal P(feasible), read off the SAME classifier that gated
    # (or did not gate) the batch above - `feas_clf` was fit on the full sheet
    # regardless of `gated`, but its probability is only reported when the gate
    # was actually active (see `_annotate`'s docstring for why the ungated case
    # is `nan` here rather than a value nobody asked to trust).
    p_feasible_arr = (
        feas_clf.predict_proba(batch) if gated else np.full(len(batch), np.nan)
    )
    # PART 2 predicted constraint value +/- sd for each proposed recipe, read
    # off the constraint surrogate before it is released below - `None` (both
    # here and in `_annotate`) unless the constraint was actually applied.
    constraint_pred: tuple[str, np.ndarray, np.ndarray] | None = None
    if constraint_applied and constraint_surrogate is not None:
        c_mean, c_std = constraint_surrogate.posterior(batch)
        assert constraint_col is not None  # constraint_applied implies this was set
        constraint_pred = (constraint_col, c_mean, c_std)

    # Response SHAPES, read off this same fitted GP before it is released. This is
    # what closes the Spearman blind spot: a titer peaking at pH 7.0 gives a rank
    # correlation near zero, so the driver panel reports "no signal" for the most
    # important variable on the sheet, and the rho sign points the wrong way.
    #
    # Deliberately read from THIS surrogate rather than from a second model. The
    # shapes then describe the posterior that actually produced `proposals`, the
    # posterior's own standard deviation gates the interior-optimum claim, and ARD
    # lengthscales supply per-feature relevance already paid for during the fit.
    # Conditioned on `rho` (the out-of-fold Spearman the reliability verdict
    # already uses), so a model that cannot predict held-out runs reports no
    # shapes at all rather than describing the shape of its own overfitting.
    #
    # Continuous features only, matching the driver panel: a swept axis has to be
    # a measurement, not an integer category code.
    #
    # `X_fit`/`y_fit` are the rows this surrogate was actually fit on, which is
    # the contract `gp_shape_report` states and, on the replicate-aware path, is
    # NOT the raw sheet. It sweeps each feature through the incumbent - the row
    # with the best measured target - and on raw rows that is `best_single`, the
    # luckiest single well. Anchoring the shapes there re-introduces exactly the
    # noise spike the replicate-averaged fit exists to ignore, and it does it
    # inside a GP that never saw that row. The observed spread it compares peak
    # height against would be the raw spread too, assay noise included, which
    # makes a real bump read as flat.
    gp_shapes = gp_shape_report(
        s_report,
        X_fit[:, : len(cont_feats)] if cont_feats else np.empty((len(y_fit), 0)),
        y_fit,
        feature_names=[str(c) for c in cont_feats],
        cv_spearman=None if rho != rho else float(rho),
        rho_floor=RELIABILITY_SPEARMAN_FLOOR,
        maximize=not minimize,
    ).to_dict()

    # VISUALIZATION PAYLOADS: embedding (PCA), correlation (Spearman), and
    # response surface from the fitted GP. Computed while the surrogate is alive
    # (before `del s`) and discarded immediately after — no serialization, no
    # persistence. Uses the null-with-reason pattern: `None` with a `reason`
    # when unavailable, otherwise the data structure with `reason=None`.
    #
    # These are torch-free (except the response surface which reads from `s`
    # while it is still alive): PCA and Spearman use only scipy/numpy.
    # The response surface evaluates the existing GP at a 21x21 grid — fast
    # because it is a single posterior call, not a second fit.
    viz_reasons: dict[str, str | None] = {
        "embedding": None, "correlation": None, "response_surface": None,
    }

    # PCA embedding
    _emb = _embedding_pca(X, y, cont_feats, kept_cats, cat_dims_map if kept_cats else None)
    embedding: dict | None = None
    if _emb is not None:
        if "reason" in _emb:
            viz_reasons["embedding"] = _emb["reason"]
        else:
            embedding = _emb

    # Spearman correlation on recipe means
    _corr = _correlation_spearman(X, y, cont_feats)
    correlation: dict | None = None
    if _corr is not None:
        if "reason" in _corr:
            viz_reasons["correlation"] = _corr["reason"]
        else:
            correlation = _corr

    # Response surface: evaluate the fitted GP on a 21x21 grid over the top
    # two continuous drivers, fixing all other dims at the incumbent recipe.
    _surf = _response_surface_gp(
        s_report, X_fit, y_fit, cont_feats, kept_cats,
        cat_dims_map or {}, code_maps, bounds,
        drv, cat_dims, incumbent, replicate_aware, maximize=not minimize,
    )
    response_surface: dict | None = None
    if _surf is not None:
        if "reason" in _surf:
            viz_reasons["response_surface"] = _surf["reason"]
        else:
            response_surface = _surf

    # Release the fitted GP(s) (hold torch/gpytorch tensors + parameter/prior
    # back-references that can form reference cycles refcounting alone won't
    # break) as soon as their last use is done, rather than waiting on
    # `_analyze` to return. Keeps a long-lived process (the portal, the
    # `--watch` poller) from accumulating fit memory across repeated analyses.
    # CV folds (the trust verdict) fit with ONE start; the served model may use
    # more. Stated here so the mismatch is visible in the record.
    model_diagnostics = {**s.fit_diagnostics, "cv_fit_restarts": 1}
    del s, s_report
    if constraint_surrogate is not None:
        del constraint_surrogate
    gc.collect()

    # The explicit, LABELED alternative to modeling on the wrong scale: when
    # `het` (computed above, alongside `nr`) says a log transform would
    # materially help, actually run it - as a second, clearly-marked
    # evaluation pass, not a silent switch. Uses the SAME `groups` the raw
    # `rep`/`cv_spearman` above used (not `recipe_key`), so the two CV passes
    # are apples-to-apples and only differ by the target's scale. See
    # `_alternative_scale_block`'s docstring for the offset-semantics gotcha
    # and why this must not be `np.log1p(y + offset)`.
    #
    # DELIBERATELY LAST, after every other torch-consuming step (the
    # proposal's `Surrogate.fit`, `propose`, `gp_shape_report`) has already
    # run and its results are fixed: this block fits its OWN GP internally
    # (inside `grouped_cv_report`), which draws from the same seeded global
    # torch RNG everything else in this function shares. Running it any
    # earlier would advance that RNG state before the proposal fit and
    # `propose()` consume it, silently changing the proposed batch depending
    # on whether this diagnostic happened to run - i.e. this block would stop
    # being purely additive. Placed here, its extra draws can only affect
    # itself.
    alt_scale = _alternative_scale_block(
        het, X, y, groups=groups, bounds=bounds, cat_dims=cat_dims,
        grouped_cv_report=grouped_cv_report, interval_calibration=interval_calibration,
    )

    # per-column provenance: what was kept as a feature, used as the target, or
    # dropped (id / other output / constant / sparse), so the client is never left
    # guessing about a silently dropped column. Mirrors the selection logic above.
    kept_features = [str(c) for c in cont_feats] + [str(c) for c in kept_cats]
    provenance = provenance_dicts(
        column_provenance(
            df,
            target=str(target),
            features=kept_features,
            numeric_cols=[str(c) for c in num],
            is_identifier=lambda name: is_identifier_name(name, profile.id_hint),
            outcome_hint=profile.outcome_hint,
            constant_on_fitted_rows=[str(c) for c in constant_on_fitted]
            + [str(c) for c in constant_cats],
            recipe_labels={str(c) for c in recipe_labels},
            declared={str(c) for c in declared},
            declared_ids={str(c) for c in roles.ids} if roles is not None else set(),
            declared_features={str(c) for c in roles.features} if roles is not None else set(),
            status_overrides={
                **role_safety.overrides,
                **{c: ("fixed_near_constant", "near_constant") for c in fixed_inputs},
            },
            reasons=role_safety.reasons,
        )
    )

    # What kind of data each column holds and what the engine did with it, so a
    # client never draws a text input on a numeric axis (analysis_parts/column_kinds.py).
    columns_block, kind_findings = column_kinds(
        df, provenance, kept_cont=[str(c) for c in cont_feats],
        kept_cats=[str(c) for c in kept_cats], max_auto_levels=AUTO_CATEGORICAL_MAX_LEVELS,
        declared_cats=[str(c) for c in roles.categoricals] if roles is not None else [],
        is_identifier=lambda name: is_identifier_name(name, profile.id_hint),
    )
    columns_block = with_blank_columns(columns_block, sheet_names, sheet_blank)

    # Which acquisition optimizer actually served these proposals, so the audit
    # trail is explicit (mirrors seed / timestamp / engine_version): a large
    # categorical space silently switches from exact enumeration to the
    # alternating heuristic (kalos/core/optimize.py), and the client should know.
    if not cat_dims:
        proposal_optimizer = "continuous"
    else:
        proposal_optimizer = mixed_strategy(cat_cardinalities, PROPOSAL_BATCH_Q)

    # Applicability flags per proposal (analysis_parts/applicability.py): far
    # from the data, and at the box edge, as two separate flags.
    nn_median = training_nn_distance_median(X, bounds, cat_dims)
    applicability = applicability_block(
        nn_median,
        roles.applicability_distance
        if roles is not None and roles.applicability_distance is not None
        else applicability_distance,
    )
    distances = nearest_training_distance(batch, X, bounds, cat_dims)
    proposals = _annotate(
        batch, p_mean, p_std, incumbent, cols=show_idx, design=design,
        p_feasible=p_feasible_arr, constraint=constraint_pred, maximize=not minimize,
        distance=distances,
    )
    for row, flags in zip(
        proposals, proposal_applicability(batch, distances, design, applicability)
    ):
        row["applicability"] = flags
    likely_half_width = conformal_q
    label_proposals(
        proposals, best_bet_idx, best=best_single, half_width=likely_half_width,
        maximize=not minimize, target=str(target),
    )
    opt_pred = float(sign * np.asarray(opt_mean, float)[0])
    opt_hw = (
        float(likely_half_width) if likely_half_width is not None
        else 1.96 * float(np.asarray(opt_std, float)[0])
    )
    opt_recipe = dict(zip(design.names, design.decode_row(optimum_x)))
    for _c, _v in fixed_inputs.items():
        opt_recipe.setdefault(_c, _v["value"])
    model_optimum: dict[str, Any] = {
        "recipe": opt_recipe,
        "predicted": round(opt_pred, 4),
        "likely_range": [round(opt_pred - opt_hw, 4), round(opt_pred + opt_hw, 4)],
        "likely_range_source": "conformal" if likely_half_width is not None else "model_sd",
        # Index in `proposals` (0 = P1), or None when it is already running.
        "in_batch": best_bet_idx,
        "already_running": bool(optimum_running),
        "predicted_vs_best": predicted_vs_best(
            opt_pred, best_single, likely_half_width, maximize=not minimize,
            target=str(target), is_best_bet=True, mode="exploit",
        ),
        # True when the best bet is (within the batch tolerance) the recipe of the
        # best measured run: the model's best guess is to repeat it.
        "near_best_run": bool(near_best_run),
    }
    if near_best_run:
        model_optimum["predicted_vs_best"]["plain"] += (
            " It is close to the recipe of your best run, so running it again checks "
            "whether that result repeats."
        )

    result = {
        "n": len(y), "d": len(kept_features), "target": str(target), "group_col": gcol,
        # "maximize" or "minimize": what the proposals optimize. `best` is the
        # max or the min accordingly.
        "direction": direction,
        "targets": [str(c) for c in candidate_targets], "features": kept_features,
        "categorical_features": [str(c) for c in kept_cats],
        "n_dropped_incomplete": n_rows_dropped,
        "n_censored_excluded": n_censored_rows,
        "data_drops": _data_drops_block(by_column, n_rows_dropped, n_censored_rows, zero_filled),
        # Text columns the engine treated as categorical with no declared roles
        # (see `_auto_categorical`); empty when roles were declared.
        "auto_categorical": [str(c) for c in auto_cats if c in kept_cats],
        "header_row": header_info.get("header_row"),
        "proposal_optimizer": proposal_optimizer,
        # In-flight recipes the acquisition was told about. 0 means the batch was
        # computed as if nothing were running - true for a one-shot upload, and
        # the thing to check first if a round re-proposes work already started.
        "n_pending_considered": n_pending,
        # In-flight recipes dropped because a continuous component (or a kept
        # categorical level) was missing: no coordinate, so not zero-filled.
        "n_pending_dropped": int(n_pending_dropped),
        "cv_spearman": None if rho != rho else round(rho, 3),
        "cv_ci95": ci95,
        # Why `cv_ci95` is null (None when it was formed).
        "cv_ci95_reason": ci95_reason,
        "cv_n_groups": rep["n_groups"],
        # How many partitions actually went into `cv_ci95` above - the requested
        # ceiling (`CV_N_REPEATS`) collapses to 1 when the sheet has too few
        # groups for a second, genuinely different partition to exist (see
        # `grouped_cv_report`'s docstring), so this is the EFFECTIVE count, not
        # the requested one.
        "cv_n_repeats": rep["n_repeats"],
        # Per-repeat pooled Spearman, so the partition-to-partition spread that
        # widens `cv_ci95` is visible on its own rather than only as a width.
        # NaN -> None: this dict is serialized straight into an API response.
        "cv_spearman_per_repeat": [
            None if s != s else round(float(s), 3) for s in rep["spearman_per_repeat"]
        ],
        "conformal_q": conformal_q,
        # The band's ACTUAL coverage, so no consumer has to hardcode it.
        "conformal_coverage": round(1.0 - CONFORMAL_ALPHA, 4),
        # What `conformal_coverage` does NOT promise. The band is one +/- width
        # from held-out residuals; no proposal carries its own conformal interval.
        "conformal_caveat": {
            "code": "coverage_not_guaranteed",
            "message": (
                "The +/- conformal_q band's coverage was measured on held-out rows "
                "near the data. It is not guaranteed for proposals, especially those "
                "flagged far_from_data or at_box_edge."
            ),
        },
        # How the driver panel was selected and corrected, so the client can see
        # that many features were tested and only the strongest are shown.
        "driver_selection": {
            "n_tested": n_tested,
            "n_reported": len(drv),
            "top_k": DRIVER_TOP_K,
            "fdr_method": "benjamini_hochberg",
            "fdr_q": DRIVER_FDR_Q,
            "n_bootstrap": 200,
            "ranked_by": "abs_rho",
            # The unit of analysis, stated rather than assumed. Replicates of one
            # recipe are averaged before the panel runs, so `n_units` (not the row
            # count) is the independent evidence every p-value and CI is computed
            # against. On an unreplicated sheet the two are equal.
            "unit": "recipe",
            "n_units": n_driver_units,
            "n_rows": len(y),
        },
        "reliability": reliability,
        # Fail-closed promotion verdict, REPORTED not enforced (see the block
        # above where `promotion` is assembled for why this must never change
        # what the API accepts).
        "promotion": promotion,
        # PART 1: whether/why the proposal acquisition was gated by P(feasible)
        # (see the GATE POLICY block above `promotion` is assembled near, for
        # the full three-condition reasoning). `threshold` is the label
        # threshold `feasible_labels` used to define "feasible" (strictly > 0
        # here, its own default), not the AUC/n_infeasible policy floors -
        # reported so a consumer never has to hardcode what "feasible" meant.
        "proposal_gating": {
            "gated": bool(gated),
            "reason": gate_reason,
            "n_infeasible": n_infeasible,
            "feasibility_auc": None if feas_auc != feas_auc else round(float(feas_auc), 3),
            "threshold": 0.0,
            # True when the gate was not applied because the target is minimized
            # (P(feasible) scores y > 0); `reason` says so too.
            "off_for_minimize": bool(minimize),
        },
        # PART 2: whether/why the requested constraint (if any) was applied to
        # the proposal below. Always present (even with no `constraint`
        # argument) so a consumer never has to special-case its absence.
        "constraint": constraint_report,
        # Leave-one-group-out, top-k overlap, and the group-mean baseline floor
        # - three evaluation-hygiene checks that ask harder or narrower
        # questions than the pooled grouped-CV Spearman above. Each is `None`
        # with a stated `reason` when it cannot be computed on this sheet,
        # rather than silently absent.
        "cv_logo": cv_logo,
        "cv_topk": cv_topk,
        # `margin` (additive): the paired surrogate-vs-floor comparison the
        # promotion gate reads (kalos/core/evaluation.py `group_mean_margin`).
        "cv_group_mean_baseline": {**cv_group_mean_baseline, "margin": gm_margin},
        "best": round(best_single, 4),
        "noise": _noise_block(nr, best_single, best_reproducible, replicate_aware, het),
        # Explicit, labeled second evaluation pass on the log scale, run only
        # when `noise.scale.suggests_transform` is True (see
        # `_alternative_scale_block`'s docstring). Evaluation-only: proposals
        # and every other number in this response stay on the raw scale.
        "alternative_scale": alt_scale,
        "drivers": drv,
        # Response shape per feature, read off the same GP that proposed the
        # batch. Closes the interior-optimum blind spot in the rank drivers.
        "gp_shapes": gp_shapes,
        "proposal_features": show,
        # The SEARCH BOX of every continuous input (`design.dims`: the observed range
        # widened to include 0 for absent-as-zero columns, see `_physical_range`), so a
        # client can flag a proposal sitting on the edge of what was tried. Categorical
        # inputs are omitted.
        "feature_ranges": {
            dim.name: [round(float(dim.lower), 6), round(float(dim.upper), 6)]
            for dim in design.dims
            if not dim.is_categorical
        },
        # None when the full batch came back; otherwise
        # `{code: "fewer_distinct_candidates", requested, returned, min_separation,
        # message}` explaining why `proposals` is shorter than requested.
        "proposal_notice": proposal.notice,
        # How distinct the batch is and how many collapsed (near-duplicate) rows
        # of the joint optimization were replaced by sequential selection.
        "batch_diversity": proposal.diversity_report(),
        "proposals": with_fixed_inputs(proposals, fixed_inputs),
        # The model's best bet (analysis_parts/best_bet.py): the recipe with the
        # highest predicted value inside the search box, its likely range, and
        # where it is in `proposals`.
        "model_optimum": model_optimum,
        # Inputs fixed at their median because their spread is within sensor
        # precision (analysis_parts/near_constant.py). Not searched.
        "fixed_inputs": {c: v["value"] for c, v in fixed_inputs.items()},
        # Context for `proposals[].nearest_training_distance` (same metric): the
        # median distance from each training row to its nearest other row.
        "training_nn_distance_median": nn_median,
        # The `far_from_data` threshold used, where it came from, and the rule.
        "applicability": applicability,
        "oof": [[round(a, 4), round(p, 4)] for a, p in zip(oof_a, oof_p)],
        "provenance": provenance,
        # One entry per uploaded column: kind (numeric, category, text_id,
        # row_number, outcome_like, constant), role, levels for categories.
        "columns": columns_block,
        # Orientation pre-pass fact from the upload path (kalos/portal/uploads.py):
        # present when the sheet was confidently detected as transposed and
        # normalized before analysis, so the client can see the frame they sent is
        # not byte-for-byte the frame that was modeled. None for a standard sheet.
        "orientation": df.attrs.get("kalos_orientation"),
        "validation": report_dict(
            _with_data_findings(
                validation,
                _data_findings(
                    by_column, n_rows_dropped, n_censored_rows, zero_filled,
                    auto_cats=[str(c) for c in auto_cats if c in kept_cats],
                    header=header_info,
                ) + role_safety.findings + near_constant_findings(fixed_inputs)
                + kind_findings + (
                    [Finding(
                        check="cv_ci_unavailable", severity="info", message=ci95_reason,
                        detail={"cv_n_groups": int(rep["n_groups"])},
                    )]
                    if ci95_reason is not None
                    else []
                ),
            )
        ),
        # Features whose search range was narrowed because some observed cells
        # were physically impossible. Reported, never silent: the client needs to
        # know the box they are being optimized over is not their full data range.
        "design_box_exclusions": box_exclusions,
        "embedding": embedding,
        "correlation": correlation,
        "response_surface": response_surface,
        "visualization_reasons": viz_reasons,
        # Numerical record of the proposal surrogate's fit (kalos/core/surrogate.py
        # `Surrogate.fit_diagnostics`): the Cholesky jitter rung and the largest
        # diagonal addition actually made, with the cap.
        "model_diagnostics": model_diagnostics,
        "seed": ANALYZE_SEED,
        "timestamp": int(time.time()),
        "engine_version": ENGINE_VERSION,
        # Response-shape version (docs/CAMPAIGNS_API.md, "Contract version").
        "contract_version": ANALYSIS_CONTRACT_VERSION,
    }
    if anonymize:
        result = _anonymize_result(
            result,
            categorical_levels={str(c): cat_dims_map[c] for c in kept_cats},
        )
    # Last step: no NaN/inf may reach the JSON encoder (see analysis_parts/sanitize.py).
    result, _sanitized = sanitize_non_finite(result)
    return result
