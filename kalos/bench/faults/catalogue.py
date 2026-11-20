"""The pre-registered fault catalogue: what is planted, and what counts as caught.

This module is the machine-readable half of `PROTOCOL.md`. Both were committed
before the first scored run, in a commit of their own, so the expected outcome
of every case was fixed before anyone saw what the engine does with it. Change
either only through a dated amendment in `PROTOCOL.md` (section "Amendments"),
never retroactively to make a number look better.

Vocabulary
----------
- A **case** is one engine input: an uploaded sheet (surface `sheet`), a sheet
  plus result entries for started runs (surface `result`), or a scale-up readout
  request (surface `scale`).
- The runner turns whatever the engine returns into **signals**: flat records
  `{source, code, severity, column, rows, message}` (see `signals.py`). A refusal
  is a signal with severity `error`.
- A **matcher** is a predicate over one signal. A fault is **caught** when any of
  its `caught_if` matchers matches a signal of the faulted case.
- **noticed**: not caught, but the case raised a warning-or-worse signal that the
  clean parent case did not raise (by `(source, code, column)`), or a
  warning-or-worse signal whose rows include a planted row, or a `noticed_if`
  matcher matched.
- **missed**: neither. A crash (any exception that is not a refusal) is missed.
- `harm_if` matchers do not change the score. They record concrete damage done by
  a miss (for example a proposal that sets `passage = 13.9`).

Matchers name engine codes that exist on `feat/web-redesign-apis` today, plus a
few codes the plans name for future checks (`planned=True`, from
CRUX_PORT_PLAN Rank 1). A future code that is not listed here counts only after
a protocol amendment, never retroactively.

Clean-room note: written from `~/.kalos-dev/CRUX_PORT_PLAN.md` (Rank 4) and the
Kalos code base only; no Crux code, fixture or text.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = "1"
SUITE_SEED = 20261119

Severity = Literal["info", "warning", "error"]
Surface = Literal["sheet", "result", "scale"]
Tier = Literal["fast", "full"]
SEVERITY_RANK: dict[str, int] = {"info": 0, "warning": 1, "error": 2}

# Column placeholder: "any of the columns the truth file lists for this case".
FAULT_COLUMN = "$fault"
ANY_CODE = "*"


@dataclass(frozen=True)
class Matcher:
    """A predicate over one signal.

    `codes` lists accepted codes (`"*"` accepts any code from `source`).
    `column` is a literal column name, `"$fault"` (any planted column) or None
    (no column constraint). `rows=True` requires the signal to carry rows and at
    least one of them to be a planted row. `message` is a regex the signal's
    message must match (case-insensitive). `planned` marks a code the engine does
    not emit yet; it is listed so a planned fix scores without a protocol change.
    """

    source: str
    codes: tuple[str, ...]
    column: str | None = None
    rows: bool = False
    min_severity: Severity = "info"
    message: str | None = None
    planned: bool = False


@dataclass(frozen=True)
class Fault:
    """One planted fault: where it is planted, what is expected, who owns a miss."""

    id: str
    family: Literal["sheet", "result", "model", "scale"]
    title: str
    base: str
    surface: Surface
    tiers: tuple[Tier, ...]
    parent: str
    expected: str
    caught_if: tuple[Matcher, ...]
    owner_module: str
    owner_plan: str
    noticed_if: tuple[Matcher, ...] = ()
    harm_if: tuple[Matcher, ...] = ()
    detectable_from_input: bool = True
    requires: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CleanCase:
    """An unmodified input, scored only for false alarms."""

    id: str
    base: str
    surface: Surface
    tiers: tuple[Tier, ...]
    expected: str
    requires: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)


BOTH: tuple[Tier, ...] = ("fast", "full")
FULL: tuple[Tier, ...] = ("full",)
KAGGLE = "kaggle"  # the base needs ~/kalos/datasets/kaggle_fermentation_dataset_0.csv

# --- reusable matchers --------------------------------------------------------- #


def _v(codes: tuple[str, ...], *, column: str | None = FAULT_COLUMN, rows: bool = True,
       min_severity: Severity = "warning") -> Matcher:
    return Matcher("validation", codes, column=column, rows=rows, min_severity=min_severity)


_REFUSAL_TOO_FEW = Matcher("refusal", ("too_few_rows", "too_few_complete_rows"))
_NOT_TRUSTED = Matcher("reliability", ("does_not_clear_floor",), min_severity="warning")
_NOT_JUDGED = Matcher("reliability", ("not_judged",), planned=True)
_PROPOSES = Matcher("proposals", ("proposes",), column=FAULT_COLUMN)
_DROPPED_OUTPUT = Matcher("columns", ("dropped_output",), column=FAULT_COLUMN)
_ANY_DROPPED = Matcher(
    "columns",
    ("dropped_id", "dropped_output", "dropped_constant", "dropped_sparse",
     "dropped_non_numeric", "dropped_constant_on_fitted_rows"),
    column=FAULT_COLUMN,
)
# A result entry the engine refuses or holds (today set_result only refuses
# non-finite values; CRUX_PORT_PLAN Rank 1 names the referee verdict codes).
_RESULT_HELD = Matcher("result_entry", ("rejected", "invalid", "needs_review"), rows=True)
_REFEREE_MAGNITUDE = Matcher("result_entry", ("result_magnitude_jump",), rows=True, planned=True)
_REFEREE_IMPOSSIBLE = Matcher("result_entry", ("result_physically_impossible",), rows=True, planned=True)
_REFEREE_DUPLICATE = Matcher("result_entry", ("result_duplicate_entry",), rows=True, planned=True)
_SCALE_STOP = Matcher("scale_readout", ("refusal",))
_SCALE_WARN_RANGE = Matcher(
    "scale_readout", ("warning",), min_severity="warning",
    message=r"ratio|range|physic|tip speed|p/v|kla|extrapolat",
)

# --- the catalogue -------------------------------------------------------------- #

FAULTS: tuple[Fault, ...] = (
    # Sheet faults: a bad upload. Visible to the torch-free validation surface.
    Fault(
        id="mixed_units_labeled", family="sheet", title="Mixed units in one column (g/L and mg/L labels)",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="units_consistency flags glucose_gL (mixed or unconvertible unit tokens)",
        caught_if=(_v(("units_consistency",), rows=False),),
        owner_module="kalos/validation/checks.py (check_units_consistency)",
        owner_plan="existing check (no plan item)",
        params={"column": "glucose_gL", "fraction": 0.20, "unit": "g/L", "alt_unit": "mg/L", "factor": 1000.0},
    ),
    Fault(
        id="mixed_units_unlabeled", family="sheet", title="A block of target rows entered in mg/L as bare numbers",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="the x1000 block is flagged on titer_g_L (units or outliers), naming planted rows",
        caught_if=(_v(("units_consistency", "outliers", "physical_bounds")),),
        owner_module="kalos/validation/checks.py (check_outliers / check_units_consistency)",
        owner_plan="unplanned for sheets (Crux #1 covers result entries only)",
        params={"column": "titer_g_L", "block": "last_quarter_producers", "factor": 1000.0},
    ),
    Fault(
        id="fahrenheit_rows_bare", family="sheet", title="Fahrenheit rows in a Celsius column (bare numbers)",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="physical_bounds or units_consistency flags temperature_C on the planted rows",
        caught_if=(_v(("physical_bounds", "units_consistency")),),
        noticed_if=(_v(("outliers",)),),
        owner_module="kalos/validation/checks.py (check_physical_bounds)",
        owner_plan="existing check (no plan item)",
        params={"column": "temperature_C", "fraction": 0.15},
    ),
    Fault(
        id="fahrenheit_rows_labeled", family="sheet", title="Fahrenheit rows in a Celsius column (labelled 'F')",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="units_consistency recognises the F token on temperature_C (and converts it)",
        caught_if=(_v(("units_consistency",), rows=False, min_severity="info"),),
        owner_module="kalos/validation/checks.py (check_units_consistency)",
        owner_plan="existing check (no plan item)",
        params={"column": "temperature_C", "fraction": 0.15},
    ),
    Fault(
        id="transposed_sheet", family="sheet", title="Transposed sheet (runs as columns), with a run-id row",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="orientation pre-pass detects 'transposed' and re-orients",
        caught_if=(Matcher("orientation", ("transposed",)),),
        owner_module="kalos/normalize/orientation.py (detect_orientation)",
        owner_plan="existing pre-pass (no plan item)",
    ),
    Fault(
        id="transposed_sheet_no_id", family="sheet", title="Transposed sheet with no run-id row (scale-up sheet)",
        base="scaleup90", surface="sheet", tiers=BOTH, parent="clean_scaleup90",
        expected="orientation pre-pass detects 'transposed' and re-orients",
        caught_if=(Matcher("orientation", ("transposed",)),),
        owner_module="kalos/normalize/orientation.py (detect_orientation)",
        owner_plan="existing pre-pass (no plan item)",
    ),
    Fault(
        id="typo_x10_target", family="sheet", title="x10 typo in two target cells",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="outliers (or bounds) flags titer_g_L naming the planted rows",
        caught_if=(_v(("outliers", "physical_bounds")),),
        owner_module="kalos/validation/checks.py (check_outliers)",
        owner_plan="existing check; Crux #1 referee for result entries",
        params={"column": "titer_g_L", "n_rows": 2, "factor": 10.0},
    ),
    Fault(
        id="typo_x1000_target", family="sheet", title="x1000 typo in one target cell",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="outliers (or bounds) flags titer_g_L naming the planted row",
        caught_if=(_v(("outliers", "physical_bounds")),),
        owner_module="kalos/validation/checks.py (check_outliers)",
        owner_plan="existing check; Crux #1 referee for result entries",
        params={"column": "titer_g_L", "n_rows": 1, "factor": 1000.0},
    ),
    Fault(
        id="typo_x10_feature", family="sheet", title="x10 typo in two glucose cells",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="outliers (or bounds) flags glucose_gL naming the planted rows",
        caught_if=(_v(("outliers", "physical_bounds")),),
        owner_module="kalos/validation/checks.py (check_outliers)",
        owner_plan="existing check (no plan item)",
        params={"column": "glucose_gL", "n_rows": 2, "factor": 10.0},
    ),
    Fault(
        id="typo_x1000_feature", family="sheet", title="x1000 typo in one agitation cell",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="outliers (or bounds) flags agitation_rpm naming the planted row",
        caught_if=(_v(("outliers", "physical_bounds")),),
        owner_module="kalos/validation/checks.py (check_outliers)",
        owner_plan="existing check (no plan item)",
        params={"column": "agitation_rpm", "n_rows": 1, "factor": 1000.0},
    ),
    Fault(
        id="swapped_columns", family="sheet", title="pH and temperature swapped in six rows",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="physical_bounds flags pH (or temperature_C) on the planted rows",
        caught_if=(_v(("physical_bounds",)),),
        noticed_if=(_v(("outliers",)),),
        owner_module="kalos/validation/checks.py (check_physical_bounds)",
        owner_plan="existing check (no plan item)",
        params={"columns": ["pH", "temperature_C"], "n_rows": 6},
    ),
    Fault(
        id="swapped_levels", family="sheet", title="Medium labels rotated in a block of 30 runs",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected=(
            "no sheet-only check can see a consistent relabelling; expected miss, kept as a "
            "measured blind spot (would need an as-run record or a bridging control)"
        ),
        caught_if=(_v((ANY_CODE,)),),
        owner_module="none today (as-run record / bridging controls)",
        owner_plan="Crux #1 recipe_deviation (with as-run recipes, F11); DOMAIN H4 bridging controls (unplanned)",
        detectable_from_input=False,
        params={"column": "medium_base", "block_rows": 30,
                "rotation": {"CD-CHO": "ProCHO", "ProCHO": "BalanCD", "BalanCD": "CD-CHO"}},
    ),
    Fault(
        id="duplicated_rows_exact", family="sheet", title="Eight rows pasted twice (identical, run id included)",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="duplicate_rows warns and names the duplicated rows",
        caught_if=(_v(("duplicate_rows",), column=None),),
        owner_module="kalos/validation/checks.py (check_duplicate_rows)",
        owner_plan="existing check (no plan item)",
        params={"n_rows": 8},
    ),
    Fault(
        id="duplicated_rows_new_id", family="sheet", title="Eight rows pasted again under new run ids",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="duplicate_rows warns (identical on every column except the run id)",
        caught_if=(_v(("duplicate_rows",), column=None),),
        noticed_if=(_v(("duplicate_rows",), column=None, min_severity="info"),),
        owner_module="kalos/validation/checks.py (check_duplicate_rows ignores nothing but outcome columns)",
        owner_plan="unplanned (exclude identifier columns from the duplicate comparison)",
        params={"n_rows": 8},
    ),
    Fault(
        id="duplicate_run_id", family="sheet", title="Two different runs share one run id",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="a finding on run_id names the two rows (duplicate identifier)",
        caught_if=(_v((ANY_CODE,), column="run_id"),),
        owner_module="kalos/validation/checks.py (no duplicate-id check)",
        owner_plan="unplanned (DOMAIN_AUDIT LOW: duplicate run ids are not flagged)",
        params={"column": "run_id"},
    ),
    Fault(
        id="too_few_rows_5", family="sheet", title="Five rows",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess30",
        expected="refusal too_few_rows",
        caught_if=(_REFUSAL_TOO_FEW, _v(("replicate_adequacy",), column=None, rows=False, min_severity="error")),
        owner_module="kalos/portal/analysis_parts/refusals.py (_too_few_rows)",
        owner_plan="existing refusal (no plan item)",
        params={"n_rows": 5},
    ),
    Fault(
        id="too_few_rows_8", family="sheet", title="Eight rows, eleven inputs",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess30",
        expected="refusal, or a reliability verdict of not_judged (too few rows to judge trust)",
        caught_if=(_REFUSAL_TOO_FEW, _NOT_JUDGED),
        noticed_if=(_NOT_TRUSTED,),
        owner_module="kalos/portal/analysis.py (reliability verdict)",
        owner_plan="F04 (not_judged below about 12 rows; ML_AUDIT N3)",
        params={"n_rows": 8},
    ),
    Fault(
        id="constant_target", family="sheet", title="Constant target",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="refusal target_constant",
        caught_if=(Matcher("refusal", ("target_constant",)),),
        owner_module="kalos/portal/analysis_parts/refusals.py (_refuse_unusable_target)",
        owner_plan="existing refusal (no plan item)",
        params={"column": "titer_g_L", "value": 1.0},
    ),
    Fault(
        id="impossible_value", family="sheet", title="Dissolved oxygen of 140 % in one row",
        base="bioprocess160", surface="sheet", tiers=BOTH, parent="clean_bioprocess160",
        expected="physical_bounds error on DO_pct naming the row",
        caught_if=(_v(("physical_bounds",), min_severity="error"),),
        owner_module="kalos/validation/checks.py (check_physical_bounds)",
        owner_plan="existing check (no plan item)",
        params={"column": "DO_pct", "value": 140.0},
    ),
    # Result-entry faults: a bad number typed for a started run. Scored on what
    # the engine does at entry (set_result) and on the re-analysis validation of
    # the folded sheet; rows are fold rows (base rows first, then the entries).
    Fault(
        id="result_x1000", family="result", title="Result entered x1000 (mg/L typed as g/L)",
        base="results155", surface="result", tiers=BOTH, parent="clean_results_plain",
        expected="held at entry (referee) or flagged on the folded row at re-analysis",
        caught_if=(_RESULT_HELD, _REFEREE_MAGNITUDE, _v(("outliers", "physical_bounds"), column="titer_g_L")),
        owner_module="kalos/portal/campaign.py (set_result accepts any finite value)",
        owner_plan="Crux #1 result referee (result_magnitude_jump; F11)",
        params={"entry": 2, "factor": 1000.0},
    ),
    Fault(
        id="result_x10", family="result", title="Result entered x10 (decimal slip)",
        base="results155", surface="result", tiers=BOTH, parent="clean_results_plain",
        expected="held at entry (referee) or flagged on the folded row at re-analysis",
        caught_if=(_RESULT_HELD, _REFEREE_MAGNITUDE, _v(("outliers", "physical_bounds"), column="titer_g_L")),
        owner_module="kalos/portal/campaign.py (set_result accepts any finite value)",
        owner_plan="Crux #1 result referee (result_magnitude_jump; F11)",
        params={"entry": 3, "factor": 10.0},
    ),
    Fault(
        id="result_negative", family="result", title="Negative titer entered",
        base="results155", surface="result", tiers=BOTH, parent="clean_results_plain",
        expected="refused at entry, or a physical_bounds error on the folded row",
        caught_if=(_RESULT_HELD, _REFEREE_IMPOSSIBLE,
                   _v(("physical_bounds",), column="titer_g_L", min_severity="error")),
        owner_module="kalos/portal/campaign.py (set_result accepts any finite value)",
        owner_plan="Crux #1 result referee (result_physically_impossible; F11)",
        params={"entry": 1},
    ),
    Fault(
        id="result_copy_paste", family="result", title="Same value pasted into two runs of one round",
        base="results155", surface="result", tiers=BOTH, parent="clean_results_plain",
        expected="held for review as a copy-paste duplicate",
        caught_if=(_RESULT_HELD, _REFEREE_DUPLICATE, _v(("duplicate_rows",), column=None)),
        owner_module="kalos/portal/campaign.py (set_result accepts any finite value)",
        owner_plan="Crux #1 result referee (result_duplicate_entry; F11)",
        params={"entry": 4, "copy_of": 0},
    ),
    # Model faults: data that is clean cell by cell but cannot support the claim
    # the engine would make. Visible only to the full analysis (torch).
    *(
        Fault(
            id=f"pure_noise_target_s{k}", family="model", title=f"Target shuffled across runs (seed {k})",
            base="bioprocess30", surface="sheet", tiers=FULL, parent="clean_bioprocess30",
            expected="trust is refused: reliability does not clear the floor (or a refusal)",
            caught_if=(_NOT_TRUSTED, Matcher("refusal", (ANY_CODE,))),
            noticed_if=(Matcher("reliability", ("space_filling_advice",)),),
            owner_module="kalos/portal/analysis.py (clears_floor on the CV point estimate)",
            owner_plan="F04 (clears_floor needs the CV CI lower bound > 0)",
            params={"column": "titer_g_L", "shuffle_seed": k},
        )
        for k in (1, 2, 3)
    ),
    Fault(
        id="leaked_column_named", family="model", title="Target-derived yield_tier kept in the sheet (Kaggle)",
        base="kaggle150", surface="sheet", tiers=FULL, parent="clean_kaggle150",
        expected="yield_tier is excluded from the inputs as an outcome (dropped_output)",
        caught_if=(_DROPPED_OUTPUT,), harm_if=(_PROPOSES,),
        owner_module="kalos/portal/analysis_parts/columns.py (role inference)",
        owner_plan="F06 column roles (derived role)",
        requires=(KAGGLE,), params={"column": "yield_tier"},
    ),
    Fault(
        id="leaked_column_renamed", family="model", title="Target tertile label under a neutral name (qc_grade)",
        base="bioprocess160", surface="sheet", tiers=FULL, parent="clean_bioprocess160",
        expected="qc_grade is excluded or flagged as target-derived",
        caught_if=(_DROPPED_OUTPUT, Matcher("validation", ("target_leak",), column=FAULT_COLUMN, planned=True)),
        harm_if=(_PROPOSES,),
        owner_module="kalos/portal/analysis_parts/columns.py (_auto_categorical)",
        owner_plan="F06 column roles (derived role); a content-based leak check is unplanned",
        params={"column": "qc_grade"},
    ),
    Fault(
        id="id_like_numeric_categorical", family="model", title="Numeric clone ids (101/204/377) with real offsets",
        base="bioprocess160", surface="sheet", tiers=FULL, parent="clean_bioprocess160",
        expected="clone is treated as a label (categorical), never proposed as a fractional number",
        caught_if=(Matcher("columns", ("categorical",), column=FAULT_COLUMN),),
        noticed_if=(Matcher("columns", ("dropped_id",), column=FAULT_COLUMN),),
        harm_if=(Matcher("proposals", ("proposes_non_level",), column=FAULT_COLUMN),),
        owner_module="kalos/portal/analysis_parts/columns.py (numeric columns default to continuous)",
        owner_plan="strain-engine P1 (roles.strain) + F06 (id role)",
        params={"column": "clone", "levels": [101, 204, 377], "offsets": [0.0, 0.15, 0.30]},
    ),
    Fault(
        id="covariate_as_setpoint", family="model", title="Passage number (a covariate) in the sheet",
        base="bioprocess160", surface="sheet", tiers=FULL, parent="clean_bioprocess160",
        expected="passage is recognised as a covariate: kept out of the design box and never proposed",
        caught_if=(Matcher("columns", ("covariate", "dropped_covariate"), column=FAULT_COLUMN, planned=True),),
        noticed_if=(_ANY_DROPPED,), harm_if=(_PROPOSES,),
        owner_module="kalos/portal/analysis_parts/columns.py (no covariate role)",
        owner_plan="strain-engine P1 (roles.covariates) + F06 (covariate role)",
        params={"column": "passage", "low": 5, "high": 30, "slope": -0.01},
    ),
    Fault(
        id="batch_offset_with_column", family="model", title="Batch offsets, no input signal, batch column present",
        base="bioprocess60", surface="sheet", tiers=FULL, parent="clean_bioprocess60",
        expected="CV is grouped by batch or trust is refused; batch is never proposed",
        caught_if=(_NOT_TRUSTED, Matcher("cv", ("group_col",), column=FAULT_COLUMN)),
        harm_if=(_PROPOSES,),
        owner_module="kalos/portal/analysis.py + analysis_parts/columns.py (CV groups)",
        owner_plan="F05 leak-proof CV grouping (+ F06: never propose a batch level)",
        params={"column": "batch", "n_batches": 6, "per_batch": 8, "offset_sd": 0.30, "noise_sd": 0.03},
    ),
    Fault(
        id="batch_offset_no_column", family="model", title="Batch offsets, no input signal, batch not recorded",
        base="bioprocess60", surface="sheet", tiers=FULL, parent="clean_bioprocess60",
        expected="trust is refused (CV cannot see the batches, so a trusted verdict is false trust)",
        caught_if=(_NOT_TRUSTED,),
        owner_module="kalos/portal/analysis.py (CV groups)",
        owner_plan="F05 leak-proof CV grouping (round index on folded rows) + F04",
        params={"n_batches": 6, "per_batch": 8, "offset_sd": 0.30, "noise_sd": 0.03},
    ),
    Fault(
        id="jittered_replicates", family="model", title="Replicates recorded as-run with 0.2 % jitter, no input signal",
        base="bioprocess60", surface="sheet", tiers=FULL, parent="clean_bioprocess60",
        expected="replicates are recognised (grouped) or trust is refused",
        caught_if=(_NOT_TRUSTED, Matcher("noise", ("replicates_detected",))),
        owner_module="kalos/core/splits.py + kalos/core/replicates.py (exact-match replicates)",
        owner_plan="F05 (replicates grouped with a tolerance)",
        params={"n_recipes": 15, "replicates": 4, "jitter_frac": 0.002, "effect_sd": 0.25, "noise_sd": 0.05},
    ),
    Fault(
        id="outcome_as_factor", family="model", title="Contamination (an observed outcome) as a text column (Kaggle)",
        base="kaggle150", surface="sheet", tiers=FULL, parent="clean_kaggle150",
        expected="contamination is excluded as an outcome and never proposed",
        caught_if=(_DROPPED_OUTPUT,), harm_if=(_PROPOSES,),
        owner_module="kalos/portal/analysis_parts/columns.py (_auto_categorical)",
        owner_plan="F06 column roles (response role)",
        requires=(KAGGLE,), params={"column": "contamination"},
    ),
    # Scale faults: a scale-up readout asked to extrapolate.
    Fault(
        id="scale_physics_extrapolation", family="scale", title="2,000 L at 240 rpm from a ladder that stops at 300 L",
        base="scaleup_le300", surface="scale", tiers=FULL, parent="clean_scale_in_ladder",
        expected="refusal, or a number with a warning that names the ratio / physics range",
        caught_if=(_SCALE_STOP, _SCALE_WARN_RANGE),
        owner_module="kalos/scale/readout.py (decide; derived physics not range-checked)",
        owner_plan="F10 scale physics guardrails",
        params={"max_scale_L": 300.0, "target_scale_L": 2000.0, "agitation_rpm": 240.0, "airflow_L_per_min": 100.0},
    ),
    Fault(
        id="scale_far_extrapolation", family="scale", title="5,000 L from a ladder that stops at 10 L",
        base="scaleup_le10", surface="scale", tiers=FULL, parent="clean_scale_in_ladder",
        expected="refusal (or a warning naming the ratio)",
        caught_if=(_SCALE_STOP, _SCALE_WARN_RANGE),
        owner_module="kalos/scale/readout.py (decide)",
        owner_plan="F10 scale physics guardrails",
        params={"max_scale_L": 10.0, "target_scale_L": 5000.0, "agitation_rpm": 150.0, "airflow_L_per_min": 250.0},
    ),
)

CLEAN: tuple[CleanCase, ...] = (
    CleanCase("clean_bioprocess160", "bioprocess160", "sheet", BOTH,
              "no refusal and no catch detector fires (examples/synthetic_bioprocess, all 160 rows)"),
    CleanCase("clean_bioprocess60", "bioprocess60", "sheet", BOTH,
              "no refusal and no catch detector fires (60-row seeded subsample)"),
    CleanCase("clean_bioprocess30", "bioprocess30", "sheet", BOTH,
              "no refusal and no catch detector fires (30-row seeded subsample)"),
    CleanCase("clean_scaleup90", "scaleup90", "sheet", BOTH,
              "no refusal and no catch detector fires (examples/synthetic_scaleup as an upload)"),
    CleanCase("clean_kaggle150", "kaggle150", "sheet", BOTH,
              "no refusal and no catch detector fires (Kaggle sheet minus outcome/derived/provenance columns)",
              requires=(KAGGLE,)),
    CleanCase("clean_results_plain", "results155", "result", BOTH,
              "five true held-out results: none refused, none flagged on their fold rows"),
    CleanCase("clean_results_surprise", "results155", "result", BOTH,
              "a surprising but plausible new best (1.10 x the best measured): not refused, not flagged",
              params={"entry": 2, "factor_of_best": 1.10}),
    CleanCase("clean_scale_in_ladder", "scaleup_le1000", "scale", FULL,
              "a target at the largest trained scale: a number, no refusal and no warning",
              params={"max_scale_L": 1000.0, "target_scale_L": 1000.0}),
)


def fault_by_id(fault_id: str) -> Fault:
    for f in FAULTS:
        if f.id == fault_id:
            return f
    raise KeyError(fault_id)


def clean_by_id(case_id: str) -> CleanCase:
    for c in CLEAN:
        if c.id == case_id:
            return c
    raise KeyError(case_id)


def catch_detectors() -> list[Matcher]:
    """Every non-planned `caught_if` matcher, with column/row constraints removed.

    On a clean case a detector "fires" when a signal matches it at warning or
    worse (and at its own minimum severity): that is a false alarm."""
    seen: dict[tuple[str, tuple[str, ...], str, str | None], Matcher] = {}
    for f in FAULTS:
        for m in f.caught_if:
            if m.planned or m.codes == (ANY_CODE,):
                continue
            sev: Severity = m.min_severity if SEVERITY_RANK[m.min_severity] >= 1 else "warning"
            key = (m.source, m.codes, sev, m.message)
            seen.setdefault(key, Matcher(m.source, m.codes, min_severity=sev, message=m.message))
    return list(seen.values())


def protocol_dict() -> dict[str, Any]:
    """The catalogue as plain data (written next to each scored run)."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "suite_seed": SUITE_SEED,
        "faults": [asdict(f) for f in FAULTS],
        "clean": [asdict(c) for c in CLEAN],
    }


__all__ = [
    "ANY_CODE", "BOTH", "CLEAN", "FAULTS", "FAULT_COLUMN", "FULL", "KAGGLE", "PROTOCOL_VERSION",
    "SEVERITY_RANK", "SUITE_SEED", "CleanCase", "Fault", "Matcher", "catch_detectors",
    "clean_by_id", "fault_by_id", "protocol_dict",
]
