"""Analysis policy: the named constants, the validation-mode policy and the
default (bioprocess) column-role hints. Every number here is named so it can be
argued with in one place; see the comment on each.
"""
from __future__ import annotations

from kalos.domains import BIOPROCESS_PROFILE
from kalos.portal.config import validation_mode_setting

# The default (bioprocess) column-role hints, kept as module-level names because
# `_anonymize_result` and the provenance defaults reference them. Sourced from
# BIOPROCESS_PROFILE so a run with no declared roles behaves exactly as before.
_ERR_VALIDATION_FAILED = (
    "The uploaded run sheet failed data validation. Fix the errors listed in "
    "`validation.findings` and upload again, or set KALOS_VALIDATION_MODE=warn "
    "to analyze it anyway and receive the findings as warnings."
)


def validation_mode() -> str:
    """Policy for what an error-severity validation finding does to an upload.

    - "warn" (the DEFAULT): analyze the sheet anyway and return the findings in
      the response. Chosen as the default deliberately, so adding this gate
      cannot start rejecting data that clients push through today - a validation
      gate that silently changes what the API accepts is its own outage.
    - "strict": refuse the upload with `UploadRejected` when the report's status
      is "fail". This is what a regulated workflow should run, and what a client
      who wants the sheet vetted before it reaches a model should ask for.

    Note that "warn" does NOT mean impossible values can reach a recommendation:
    the design box is built from physically valid observations regardless of
    mode (see `_physical_range`). Mode governs reporting, not that safety
    property. Read at call time, not import time, so a deployment can change it
    without a restart and tests can monkeypatch it.
    """
    # Fail closed: an unknown value raises (kalos.envconfig) instead of
    # silently selecting "warn" (review H3).
    return validation_mode_setting()


_OUTCOME_HINT = BIOPROCESS_PROFILE.outcome_hint
_TARGET_PREF = BIOPROCESS_PROFILE.target_pref
_ID_HINT = BIOPROCESS_PROFILE.id_hint
_GROUP_HINT = BIOPROCESS_PROFILE.group_hint


# Version of the analysis response shape. Every analysis carries it as
# `contract_version`. It goes up by one when fields are added in a round of
# changes (docs/CAMPAIGNS_API.md, "Contract version"); no field is removed or
# changes meaning without a new version. 1 = every response before this field.
ANALYSIS_CONTRACT_VERSION = 2

# Deterministic seed for the analyze path. The same upload -> the same GP fit and
# the same proposed batch, which matters for client reproducibility and audit.
ANALYZE_SEED = 1234

# Driver-panel policy. Named here so the numbers can be argued with in one place.
DRIVER_FDR_Q = 0.05
# Benjamini-Hochberg false-discovery rate for the driver panel. 0.05 keeps the
# expected proportion of false findings among reported drivers at 1 in 20, which
# is the right error rate when a scientist will act on several of them.

DRIVER_TOP_K = 8
# How many drivers are shipped after ranking by |rho|. The cut is REPORTED
# (`driver_selection` below) rather than silent, because selecting the strongest
# of many tested features is itself a statistical act the client needs to know
# happened.

CONFORMAL_ALPHA = 0.1
# Miscoverage for the conformal band, so the response can state its own coverage
# (1 - alpha = 90%) instead of leaving every consumer to hardcode a number. The
# frontend had drifted to labelling this same band "95%" on one surface and
# "90%" on another; a stated coverage that is wrong on screen is an overclaim,
# not a hedge, so the engine now ships the number it actually computed.

CV_N_REPEATS = 2
# How many partitions `grouped_cv_report` draws for the reliability CI. On a
# continuous target the grouped CV takes the unshuffled GroupKFold branch, so
# a SINGLE partition's group-bootstrap CI only ever covers group-resampling
# variance, not how much the estimate moves under a different, equally valid
# partition - on the real media DoE the pooled Spearman moved 0.44 to 0.71
# across n_splits 3 to 8 alone. Repeated CV over shuffled partitions is what
# makes the CI honest about that.
#
# Repeat 0 is always the free unshuffled partition (the point estimate's
# anchor, already paid for today), so each ADDITIONAL repeat costs one more
# round of n_splits GP fits - not cheap at this module's per-fit cost. Timed
# end-to-end on a 60-row, 13-column sheet on the dev machine this was tuned
# on: n_repeats=1 (today's cost) ~5.3-6.1s, n_repeats=2 ~7.3-8.6s,
# n_repeats=3 ~10.3-10.6s - already over the ~10s budget before counting any
# slower CI hardware or a larger upload. 2 is the largest value that stayed
# comfortably under budget with margin to spare: one extra shuffled partition
# beyond today's, which is enough to turn "the CI is conditional on one fixed
# split" into "the CI has seen the estimate move at least once," without
# betting the request's latency on it.

RELIABILITY_SPEARMAN_FLOOR = 0.20
# The one out-of-fold Spearman floor this module uses, for the reliability verdict
# AND for gating the GP shape report. It was previously a bare 0.20 literal in two
# places here; a single name means the verdict and the shapes can never be held to
# different bars, and a future change to the bar cannot move one without the other.

LOGO_MAX_GROUPS = 12
# Leave-one-group-out (`kalos.core.evaluation.logo_report`) costs one Surrogate GP
# fit PER GROUP - unlike `grouped_cv_report`'s fixed `n_splits` folds, a LOGO run
# grows with however many groups the sheet declares, so an unbounded sheet could
# turn one request into dozens of GP fits. Timed end-to-end on this dev machine,
# a 60-row, 2-continuous-feature sheet (12 recipes x 5 reps): without a declared
# group column (`cv_logo` skipped) `_analyze` ran 1.46-2.05s across 4 runs
# (min 1.46s); with a declared 12-group column - LOGO_MAX_GROUPS exactly, the
# most expensive case this cap still allows - it ran 2.02-2.59s (min 2.02s), a
# delta of roughly 0.5-0.6s for 12 extra GP fits on this machine, each fit on a
# fold nearly as large as the full sheet (leave-ONE-group-out trains on
# n_groups-1 of n_groups groups, a much bigger training split per fit than the
# 5-fold CV's 4/5). That stays comfortably under a ~12s budget even with
# slower CI hardware or a larger upload; a sheet with more groups than this cap
# skips LOGO instead of letting the cost scale unbounded with group count, and
# says so via `cv_logo`'s `reason` field below rather than staying silent about
# why the number is missing.

CV_TOPK_K = 5
# How many of the top-ranked recipes `cv_topk` (kalos.core.evaluation.top_k_overlap)
# scores. 5 mirrors `q=5`, the size of the batch this module actually proposes
# below (`propose(..., q=5, ...)`) - the top-k question this metric answers
# ("of the ones I'd actually advance, how many does the model get right") is
# most honest when k matches the real decision size, not an arbitrary round number.

GROUP_MEAN_BASELINE_MIN_GROUPS = 5
# `cv_group_mean_baseline` (kalos.core.evaluation.group_mean_baseline_spearman)
# needs enough REPLICATED groups for the leave-one-row-out group mean to be a
# real prediction rather than noise from 1-2 groups' worth of pairs. 5 is a
# convention, chosen for the same reason `MIN_REPLICATED_FOR_HETERO` (see
# kalos.core.replicates) picks a small integer over a formula: below it, a
# Spearman computed against so few independent group means would read as
# confident nonsense, so the baseline declines rather than reports one.

GATE_MIN_N_INFEASIBLE = 5
# PART 1 (feasibility-gated proposals) POLICY constant: the sheet must have at
# least this many non-producer rows before the acquisition is gated by
# P(feasible) at all - "zero-inflated enough" to be the pathology BENCHMARK.md
# documents, not a producer-only sheet with a couple of stray zeros.
# `FeasibilityClassifier.fit` already refuses to fit sklearn below 3
# minority-class examples (its own cold-start guard), but 3-4 examples give an
# unstable decision boundary with no room to spare; 5 is the smallest count
# that lets a 5-fold CV split (the fold count `feasibility_cv_report` below
# uses) reserve at least one non-producer per held-out fold, which is the
# regime the classifier and its CV AUC below both implicitly assume.

# No separate GATE_MIN_FEASIBILITY_AUC constant is defined here. PART 1's gate
# policy reuses `kalos.core.gates.GatesConfig.min_feasibility_auc` (0.65)
# directly at the point it is used below, rather than duplicating that number
# under a second name: it is already the bar this codebase uses to decide "is
# this classifier's feasibility ranking trustworthy" for the promotion
# verdict, and a second, separately-tunable ~0.6 floor living next to it would
# fracture the single-number-single-meaning discipline
# `RELIABILITY_SPEARMAN_FLOOR`'s comment above states for the reliability
# verdict. A classifier too weak to trust for promotion is, by construction,
# too weak to trust for gating acquisition - not by coincidence.

CONSTRAINT_MIN_ROWS = 6
# PART 2 (constrained proposals) POLICY constant: the constraint column needs
# at least this many present values before a second `Surrogate` is fit on it.
# Mirrors the >=6 row floor `_analyze` already requires to fit the TARGET
# surrogate (see the `len(y) < 6` check above) - a GP fit on fewer points than
# that is not a floor this module considers meaningful for the target, and the
# constraint model is held to the identical bar rather than a laxer one.
