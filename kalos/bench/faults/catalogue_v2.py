"""Planted-fault catalogue v2, pre-registered: the faults lab scientists make.

This module is the machine-readable half of `PROTOCOL_V2.md`. Both were
committed before any v2 detector existed and before the first scored v2 run, in
a commit of their own. Change either only through a dated amendment at the end
of `PROTOCOL_V2.md`. The v1 catalogue (`catalogue.py`, `PROTOCOL.md`) is frozen
and is not touched by v2.

v2 reuses the v1 vocabulary unchanged (`Matcher`, `Fault`, `CleanCase`, the
`$fault` column placeholder, the signal format and the caught / noticed /
missed rule). What differs, and why, is in PROTOCOL_V2.md section 5:

- the false-alarm detectors of v2 include its *planned* codes, so a new detector
  counts against the clean cases from the first run that emits it;
- the v2 false-alarm detectors also include every v1 catch detector, so a v2
  clean sheet is judged against the whole engine, not only the new checks.

Each fault is modelled on a failure seen in real lab data, described
generically. Every case is generated from repo-owned synthetic data
(`bases_v2.py`, `examples/`). No partner file, value, sample id or name is used.

Planned codes (`planned=True`) are the finding codes the WS-C detectors will
emit through `kalos/validation/runner.py`, in the data-checks finding format
(`check` = code, `column`, `rows`, `severity`). A code that is not listed here
counts only after an amendment, never retroactively.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .catalogue import (
    ANY_CODE,
    FAULT_COLUMN,
    SEVERITY_RANK,
    CleanCase,
    Fault,
    Matcher,
    Severity,
    Tier,
)
from .catalogue import (
    catch_detectors as v1_catch_detectors,
)

PROTOCOL_VERSION_V2 = "2"
SUITE_SEED_V2 = 20261120
BOTH: tuple[Tier, ...] = ("fast", "full")

# --- planned finding codes (all from kalos/validation/<module>.py) ----------------- #

CENSORED_ZERO = "censored_zero"                    # censoring.py: exact 0 in a positive concentration outcome
REPEATED_CONSTANT = "repeated_constant"            # censoring.py: one value repeated far beyond chance
VOLUME_NOT_CONCENTRATION = "volume_not_concentration"  # units (PR 1): an amount added, not a concentration
STACKED_REGIMES = "stacked_regimes"                # regimes.py: >100x span and a split outcome
PLATE_EFFECT = "plate_effect"                      # controls.py: plate-level offset beyond replicate noise
CONTROL_OUT_OF_RANGE = "control_out_of_range"      # controls.py: a plate's controls fail acceptance
GROUP_KEY_CONFLICT = "group_key_conflict"          # leaks.py: one key, several recipes or campaigns
WELL_POSITION_CONFLICT = "well_position_conflict"  # plate ingest (WS-E): duplicate or impossible wells
DERIVED_INCONSISTENT = "derived_column_inconsistent"  # derived.py: a derived column breaks its formula
RUN_FAILURE_NOTE = "run_failure_note"              # controls.py: notes record a failed or contaminated run
TARGET_LINEAR_COMBINATION = "target_linear_combination"  # leaks.py: target = linear combo of inputs
LABEL_LEAK = "label_leak"                          # leaks.py: one column / missingness predicts a label
ID_TOKEN_FEATURE = "id_token_feature"  # noqa: S105  # leaks.py: an input parsed out of an id (not a secret)
CONFOUNDED_GROUPING = "confounded_grouping"        # regimes.py: a grouping nested in a process level
MEASURED_AFTER_OUTCOME = "measured_after_outcome"  # leaks.py: an input recorded at or after harvest
PLATE_POSITION_EFFECT = "plate_position_effect"    # plate layout (WS-F): row / column / edge effect

PLANNED_CODES: tuple[str, ...] = (
    CENSORED_ZERO, REPEATED_CONSTANT, VOLUME_NOT_CONCENTRATION, STACKED_REGIMES, PLATE_EFFECT,
    CONTROL_OUT_OF_RANGE, GROUP_KEY_CONFLICT, WELL_POSITION_CONFLICT, DERIVED_INCONSISTENT,
    RUN_FAILURE_NOTE, TARGET_LINEAR_COMBINATION, LABEL_LEAK, ID_TOKEN_FEATURE, CONFOUNDED_GROUPING,
    MEASURED_AFTER_OUTCOME, PLATE_POSITION_EFFECT,
)


def _p(code: str, *, column: str | None = FAULT_COLUMN, rows: bool = True,
       min_severity: Severity = "warning") -> Matcher:
    """A planned validation finding."""
    return Matcher("validation", (code,), column=column, rows=rows, min_severity=min_severity, planned=True)


def _v(codes: tuple[str, ...], *, column: str | None = FAULT_COLUMN, rows: bool = True,
       min_severity: Severity = "warning") -> Matcher:
    """A validation finding the engine can emit today (main after feat/data-checks)."""
    return Matcher("validation", codes, column=column, rows=rows, min_severity=min_severity)


_DROPPED_OUTPUT = Matcher("columns", ("dropped_output",), column=FAULT_COLUMN)
_DROPPED_ID = Matcher("columns", ("dropped_id",), column=FAULT_COLUMN)
_COVARIATE = Matcher("columns", ("covariate", "dropped_covariate"), column=FAULT_COLUMN)
_PROPOSES = Matcher("proposals", ("proposes",), column=FAULT_COLUMN)
_OUTLIERS_ROWS = _v(("outliers",))

WSC = "WS-C feat/lab-data-checks"

# --- the catalogue ------------------------------------------------------------------ #

FAULTS_V2: tuple[Fault, ...] = (
    # Censored and filled results (item 5).
    Fault(
        id="nondetects_as_zero", family="sheet",
        title="Non-detects typed as 0 in a positive concentration outcome",
        base="clone64", surface="sheet", tiers=BOTH, parent="clean_clone64",
        expected=(
            "the exact zeros in titer_g_L are reported as probable non-detects (below the assay's "
            "limit), naming the rows, with a request to declare LOD/LOQ; the zero rows are not fitted "
            "as measured 0 g/L"
        ),
        caught_if=(_p(CENSORED_ZERO),),
        owner_module="kalos/validation/censoring.py",
        owner_plan=f"{WSC} PR 2 (censoring, BRE-127)",
        params={"column": "titer_g_L", "censored_token": "<0.05", "replacement": 0.0},
    ),
    Fault(
        id="placeholder_fill_constant", family="sheet",
        title="Unmeasured wells filled with one placeholder value (half the reporting limit)",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "the repeated value in titer_g_L is reported as a probable placeholder, naming the rows; "
            "the rows are not fitted as measurements"
        ),
        caught_if=(_p(REPEATED_CONSTANT), _v(("result_duplicate_entry",))),
        owner_module="kalos/validation/censoring.py",
        owner_plan=f"{WSC} PR 2 (censoring, BRE-127)",
        params={"column": "titer_g_L", "n_rows": 12, "value": 0.025},
    ),
    Fault(
        id="mean_imputed_target", family="sheet",
        title="Missing results filled with the column mean",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "the repeated value in titer_g_L (equal to the mean of the other rows) is reported as a "
            "probable imputation, naming the rows; the rows are not fitted as measurements"
        ),
        caught_if=(_p(REPEATED_CONSTANT), _v(("result_duplicate_entry",))),
        owner_module="kalos/validation/censoring.py",
        owner_plan=f"{WSC} PR 2 (censoring, BRE-127)",
        params={"column": "titer_g_L", "n_rows": 10, "decimals": 3},
    ),
    # Units, regimes, missing context (items 8, 16).
    Fault(
        id="volumes_not_concentrations", family="sheet",
        title="Feed recorded as mL added per vessel instead of % of working volume, across scales",
        base="scaleup_feed90", surface="sheet", tiers=BOTH, parent="clean_scaleup_feed90",
        expected=(
            "feed_added_mL is reported as an amount (a volume), not a concentration: it tracks the "
            "vessel scale over a >1000x span, so equal mL are not equal doses; the report asks for "
            "the concentration or % of working volume"
        ),
        caught_if=(_p(VOLUME_NOT_CONCENTRATION, rows=False),),
        noticed_if=(_v(("units_consistency",), rows=False, min_severity="info"),),
        owner_module="kalos/normalize/units.py + header_units.py + unit_registry.py (one registry)",
        owner_plan=f"{WSC} PR 1 (unit vocabulary, BRE-130)",
        params={"source_column": "feed_pct", "column": "feed_added_mL", "scale_column": "scale_L"},
    ),
    Fault(
        id="stacked_regimes", family="sheet",
        title="Bench bioreactor runs stacked under a deep-well plate screen in one sheet",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "a stacked-regime warning: working volume spans 2000x and titer splits into two groups "
            "that follow it; the report suggests separate analyses or a regime column"
        ),
        caught_if=(_p(STACKED_REGIMES, rows=False),),
        owner_module="kalos/validation/regimes.py",
        owner_plan=f"{WSC} PR 4 (regimes)",
        params={"column": "working_volume_mL", "plate_volume_mL": 1.0, "reactor_volume_mL": 2000.0,
                "n_runs": 24, "n_recipes": 12, "regime_factor": 3.2, "noise_sd": 0.08},
    ),
    Fault(
        id="unrecorded_harvest_time", family="sheet",
        title="One plate per round harvested days earlier, with no harvest-time column",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "a plate-level offset on the early plates is reported (controls and replicate pairs on the "
            "other plate disagree), naming their rows; the report asks whether harvest time or vessel "
            "differed and suggests recording it. The cause itself is not visible in the sheet"
        ),
        caught_if=(_p(PLATE_EFFECT), _p(CONTROL_OUT_OF_RANGE)),
        owner_module="kalos/validation/controls.py",
        owner_plan=f"{WSC} PR 4 (controls per plate)",
        params={"plates": ["P2", "P4"], "factor": 0.55},
    ),
    # Plates, controls, identifiers (items 7e, 13, 16).
    Fault(
        id="failed_control_plate_kept", family="sheet",
        title="A plate whose reference controls failed is kept in the sheet",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "the plate's controls are reported out of range against the other plates' controls, naming "
            "the plate's rows; the plate is kept unless the user records a decision to exclude it"
        ),
        caught_if=(_p(CONTROL_OUT_OF_RANGE),),
        noticed_if=(_p(PLATE_EFFECT),),
        owner_module="kalos/validation/controls.py; kalos/campaigns/audit.py (exclusion decision)",
        owner_plan=f"{WSC} PR 4 (controls per plate)",
        params={"plate": "P3", "control_factor": 0.30, "sample_noise_sd": 0.35},
    ),
    Fault(
        id="sample_id_not_recipe", family="model",
        title="Sample ids used as the recipe key; some ids cover two different recipes",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "the key sample_id is reported as linking rows with different recipes, naming the rows; "
            "it is not used as the replicate or CV group as it stands"
        ),
        caught_if=(_p(GROUP_KEY_CONFLICT),),
        noticed_if=(_v(("duplicate_ids",)),),
        owner_module="kalos/validation/leaks.py",
        owner_plan=f"{WSC} PR 3 (leak detectors v2, BRE-128)",
        params={"source_column": "recipe_id", "column": "sample_id", "plate": "P2", "n_conflicts": 6},
    ),
    Fault(
        id="well_name_mismatch", family="sheet",
        title="Well names that collide on one plate or do not exist on a 96-well plate",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "the duplicate and impossible well positions are reported, naming the rows; nothing is "
            "re-joined or moved automatically"
        ),
        caught_if=(_p(WELL_POSITION_CONFLICT),),
        noticed_if=(_v(("duplicate_ids",)),),
        owner_module="kalos/ingest/plates.py (plate-map mismatch report)",
        owner_plan="WS-E feat/lab-ingest (item 13); scored here because the fault is in the sheet",
        params={"column": "well", "plate": "P2", "n_duplicates": 3, "impossible": ["A13", "I5"]},
    ),
    Fault(
        id="derived_column_inconsistent", family="sheet",
        title="A derived column (glucose in mM) not updated after its source was corrected",
        base="plate368_derived", surface="sheet", tiers=BOTH, parent="clean_plate368_derived",
        expected=(
            "glucose_mM is reported as inconsistent with glucose_gL on the planted rows (the relation "
            "holds on the other rows); neither value is changed"
        ),
        caught_if=(_p(DERIVED_INCONSISTENT),),
        owner_module="kalos/validation/derived.py",
        owner_plan=f"{WSC} PR 4 (derived-column consistency)",
        params={"column": "glucose_mM", "source_column": "glucose_gL", "n_rows": 6, "stale_factor": 1.5},
    ),
    Fault(
        id="contamination_in_notes", family="sheet",
        title="Contaminated wells kept as measured; the contamination is only in the notes",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "the rows whose notes record contamination are reported, naming them; they are kept "
            "unless the user records a decision to exclude them"
        ),
        caught_if=(_p(RUN_FAILURE_NOTE),),
        noticed_if=(_OUTLIERS_ROWS,),
        owner_module="kalos/validation/controls.py (run-failure notes)",
        owner_plan=f"{WSC} PR 4 (controls, run-failure notes)",
        params={"column": "notes", "n_rows": 7, "titer_factor": 0.2},
    ),
    # Leaks (item 7).
    Fault(
        id="derived_target_linear_combo", family="model",
        title="The target is the sum of two fraction columns that are offered as inputs",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "a linear-combination leak: titer_g_L = supernatant_g_L + cell_assoc_g_L (R^2 >= 0.999, "
            "formula shown); the two fractions are not used as inputs or proposed"
        ),
        caught_if=(_p(TARGET_LINEAR_COMBINATION, rows=False), _v(("target_leak",), rows=False),
                   _DROPPED_OUTPUT),
        harm_if=(_PROPOSES,),
        owner_module="kalos/validation/leaks.py",
        owner_plan=f"{WSC} PR 3 (leak detectors v2, BRE-128)",
        params={"columns": ["supernatant_g_L", "cell_assoc_g_L"], "low": 0.60, "high": 0.95},
    ),
    Fault(
        id="label_encoded_by_column", family="model",
        title="A status column set from the flask result (terminated / expanded) offered as an input",
        base="clone64", surface="sheet", tiers=BOTH, parent="clean_clone64",
        expected=(
            "expansion_status alone predicts the producer / non-producer label at > 0.95 balanced "
            "accuracy and is reported as a label leak; it is not used as an input or proposed"
        ),
        caught_if=(_p(LABEL_LEAK, rows=False), _v(("target_leak",), rows=False), _DROPPED_OUTPUT),
        harm_if=(_PROPOSES,),
        owner_module="kalos/validation/leaks.py",
        owner_plan=f"{WSC} PR 3 (leak detectors v2, BRE-128)",
        params={"column": "expansion_status", "n_flipped": 2},
    ),
    Fault(
        id="id_token_feature", family="model",
        title="A number parsed out of the run id, assigned in pre-screen rank order, offered as an input",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "tube_no is reported as a token of run_id (identifier-derived), not a setting; it is not "
            "used as an input or proposed"
        ),
        caught_if=(_p(ID_TOKEN_FEATURE, rows=False), _DROPPED_ID),
        harm_if=(_PROPOSES,),
        owner_module="kalos/validation/leaks.py",
        owner_plan=f"{WSC} PR 3 (leak detectors v2, BRE-128)",
        params={"column": "tube_no", "id_column": "run_id", "rank_noise_sd": 0.15},
    ),
    Fault(
        id="pair_id_across_campaigns", family="model",
        title="Recipe keys restart in the second campaign, so one key spans two campaigns",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "recipe_id is reported as linking rows of different recipes in different campaigns, "
            "naming the rows; it is not used as the replicate or CV group as it stands"
        ),
        caught_if=(_p(GROUP_KEY_CONFLICT),),
        owner_module="kalos/validation/leaks.py",
        owner_plan=f"{WSC} PR 3 (leak detectors v2, BRE-128)",
        params={"column": "recipe_id", "campaign": "C2"},
    ),
    # Design structure and time order (items 6, 7, 18).
    Fault(
        id="flask_media_mixed", family="model",
        title="Half the clones grown only in one flask medium, half only in the other",
        base="clone64", surface="sheet", tiers=BOTH, parent="clean_clone64",
        expected=(
            "clone is reported as nested in flask_medium (each clone seen in one medium only), so "
            "clone differences cannot be separated from the medium effect; clones are not ranked "
            "across media as if comparable"
        ),
        caught_if=(_p(CONFOUNDED_GROUPING, rows=False), _v(("constant_within_group",), rows=False)),
        owner_module="kalos/validation/regimes.py; kalos/core/funnel.py (never average across flask media)",
        owner_plan=f"{WSC} PR 4 (regimes); WS-D feat/clone-funnel (item 6)",
        params={"medium_column": "flask_medium", "clone_column": "clone",
                "first_half_medium": "medium_A", "second_half_medium": "medium_B"},
    ),
    Fault(
        id="feature_after_outcome", family="model",
        title="Residual glucose measured at harvest offered as an input",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "residual_glucose_harvest_gL is reported as recorded at or after the outcome (not a "
            "setting); it is not used as an input or proposed"
        ),
        caught_if=(_p(MEASURED_AFTER_OUTCOME, rows=False), _DROPPED_OUTPUT, _COVARIATE),
        harm_if=(_PROPOSES,),
        owner_module="kalos/validation/leaks.py",
        owner_plan=f"{WSC} PR 3 (leak detectors v2, BRE-128)",
        params={"column": "residual_glucose_harvest_gL", "feed_glucose_frac": 0.4,
                "uptake_per_titer": 3.1, "noise_sd": 0.4},
    ),
    Fault(
        id="edge_well_effect", family="sheet",
        title="Outer-ring wells read higher (evaporation), every plate",
        base="plate368", surface="sheet", tiers=BOTH, parent="clean_plate368",
        expected=(
            "a plate-position (edge) effect on titer_g_L is reported with its size and interval, "
            "naming edge rows; no value is corrected"
        ),
        caught_if=(_p(PLATE_POSITION_EFFECT),),
        noticed_if=(_OUTLIERS_ROWS,),
        owner_module="kalos/analysis/plate_layout.py (position-effect check)",
        owner_plan="WS-F feat/bench-prep (item 18); scored here because the fault is in the sheet",
        params={"factor": 1.20},
    ),
)

CLEAN_V2: tuple[CleanCase, ...] = (
    CleanCase("clean_plate368", "plate368", "sheet", BOTH,
              "no refusal and no catch detector fires (two-round 96-deep-well media screen, controls "
              "on every plate, replicates on different plates, benign notes)"),
    CleanCase("clean_plate368_derived", "plate368_derived", "sheet", BOTH,
              "no refusal and no catch detector fires (the plate screen plus a consistent glucose_mM)"),
    CleanCase("clean_clone64", "clone64", "sheet", BOTH,
              "no refusal and no catch detector fires (crossed clone x flask-medium design; non-producers "
              "written as '<0.05', a correctly labelled censored value: info at most)"),
    CleanCase("clean_scaleup_feed90", "scaleup_feed90", "sheet", BOTH,
              "no refusal and no catch detector fires (1-5000 L ladder with feed in % v/v; the scale "
              "span is >100x but titer does not split, so no stacked-regime warning)"),
)

# Cases that belong in this catalogue's scope but are built on other branches.
# Referenced, not duplicated: they are scored by those branches' own tests.
REFERENCED_ELSEWHERE: dict[str, str] = {
    "assay_method_changes_with_scale": (
        "fix/scale-readout-honesty: kalos/scale/assay_confound.py (the assay method changes "
        "with scale and is confounded with it)"
    ),
    "transposed_funnel_composite_ids": (
        "fix/orientation-composite-ids: kalos/normalize/orientation.py (a transposed funnel "
        "sheet whose run ids are composite stage tokens)"
    ),
}


def fault_by_id(fault_id: str) -> Fault:
    for f in FAULTS_V2:
        if f.id == fault_id:
            return f
    raise KeyError(fault_id)


def clean_by_id(case_id: str) -> CleanCase:
    for c in CLEAN_V2:
        if c.id == case_id:
            return c
    raise KeyError(case_id)


def catch_detectors_v2() -> list[Matcher]:
    """The v2 false-alarm detectors: every `caught_if` matcher of v2 (planned
    included) plus every v1 catch detector, with column and row constraints
    removed, at warning or worse (PROTOCOL_V2.md section 5)."""
    seen: dict[tuple[str, tuple[str, ...], str, str | None], Matcher] = {}
    for m in [*(m for f in FAULTS_V2 for m in f.caught_if), *v1_catch_detectors()]:
        if m.codes == (ANY_CODE,):
            continue
        sev: Severity = m.min_severity if SEVERITY_RANK[m.min_severity] >= 1 else "warning"
        key = (m.source, m.codes, sev, m.message)
        seen.setdefault(key, Matcher(m.source, m.codes, min_severity=sev, message=m.message))
    return list(seen.values())


def protocol_dict_v2() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION_V2,
        "suite_seed": SUITE_SEED_V2,
        "faults": [asdict(f) for f in FAULTS_V2],
        "clean": [asdict(c) for c in CLEAN_V2],
        "planned_codes": list(PLANNED_CODES),
        "referenced_elsewhere": dict(REFERENCED_ELSEWHERE),
    }


__all__ = [
    "BOTH", "CLEAN_V2", "FAULTS_V2", "PLANNED_CODES", "PROTOCOL_VERSION_V2", "REFERENCED_ELSEWHERE",
    "SUITE_SEED_V2", "catch_detectors_v2", "clean_by_id", "fault_by_id", "protocol_dict_v2",
]
