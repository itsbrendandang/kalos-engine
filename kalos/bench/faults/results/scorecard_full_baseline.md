# Planted-fault scorecard (full tier): synthetic planted faults

Protocol v1, seed 20261119, commit `bdd2d974c3` on `feat/planted-faults`, 2026-11-19T20:14:07Z.

- Caught: 20/36 (56%; Wilson 95% 0.40-0.70)
- Caught or noticed: 24/36 (67%; Wilson 95% 0.50-0.80)
- Caught, faults detectable from the input: 20/35 (57%; Wilson 95% 0.41-0.72)
- Clean cases with a false alarm: 6/8 (75%; Wilson 95% 0.41-0.93)
- Missed: 12; crashed: 0; skipped (base unavailable): 0

## Faults

| Fault | Family | Outcome | Evidence | Harm on a miss | Owner (module / plan) |
|---|---|---|---|---|---|
| `batch_offset_no_column` | model | **missed** |  |  | kalos/portal/analysis.py (CV groups) / F05 leak-proof CV grouping (round index on folded rows) + F04 |
| `batch_offset_with_column` | model | **missed** |  | proposals:proposes [batch] (info): 5 proposal(s), e.g. 'B4' | kalos/portal/analysis.py + analysis_parts/columns.py (CV groups) / F05 leak-proof CV grouping (+ F06: never propose a batch level) |
| `covariate_as_setpoint` | model | **missed** |  | proposals:proposes [passage] (info): 5 proposal(s), e.g. 5.0 | kalos/portal/analysis_parts/columns.py (no covariate role) / strain-engine P1 (roles.covariates) + F06 (covariate role) |
| `id_like_numeric_categorical` | model | **missed** |  | proposals:proposes_non_level [clone] (info): proposed 325.298; observed levels [101.0, 204.0, 377.0] | kalos/portal/analysis_parts/columns.py (numeric columns default to continuous) / strain-engine P1 (roles.strain) + F06 (id role) |
| `jittered_replicates` | model | **missed** |  |  | kalos/core/splits.py + kalos/core/replicates.py (exact-match replicates) / F05 (replicates grouped with a tolerance) |
| `leaked_column_named` | model | **caught** | columns:dropped_output [yield_tier] (info) |  |  |
| `leaked_column_renamed` | model | **missed** |  | proposals:proposes [qc_grade] (info): 5 proposal(s), e.g. 'A' | kalos/portal/analysis_parts/columns.py (_auto_categorical) / F06 column roles (derived role); a content-based leak check is unplanned |
| `outcome_as_factor` | model | **noticed** | reliability:does_not_clear_floor (warning) (not on the clean parent); reliability:space_filling_advice (warning) (not on the clean parent) | proposals:proposes [contamination] (info): 5 proposal(s), e.g. 'Y' | kalos/portal/analysis_parts/columns.py (_auto_categorical) / F06 column roles (response role) |
| `pure_noise_target_s1` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `pure_noise_target_s2` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `pure_noise_target_s3` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `result_copy_paste` | result | **missed** |  |  | kalos/portal/campaign.py (set_result accepts any finite value) / Crux #1 result referee (result_duplicate_entry; F11) |
| `result_negative` | result | **caught** | validation:physical_bounds [titer_g_L] (error) |  |  |
| `result_x10` | result | **missed** | right reason fired but its 20-row list omits the planted rows |  | kalos/portal/campaign.py (set_result accepts any finite value) / Crux #1 result referee (result_magnitude_jump; F11) |
| `result_x1000` | result | **missed** | right reason fired but its 20-row list omits the planted rows |  | kalos/portal/campaign.py (set_result accepts any finite value) / Crux #1 result referee (result_magnitude_jump; F11) |
| `scale_far_extrapolation` | scale | **caught** | scale_readout:refusal (error) |  |  |
| `scale_physics_extrapolation` | scale | **missed** |  |  | kalos/scale/readout.py (decide; derived physics not range-checked) / F10 scale physics guardrails |
| `constant_target` | sheet | **caught** | refusal:target_constant (error) |  |  |
| `duplicate_run_id` | sheet | **missed** |  |  | kalos/validation/checks.py (no duplicate-id check) / unplanned (DOMAIN_AUDIT LOW: duplicate run ids are not flagged) |
| `duplicated_rows_exact` | sheet | **caught** | validation:duplicate_rows (warning) |  |  |
| `duplicated_rows_new_id` | sheet | **noticed** | validation:outliers [antifoam_mgL] (warning) (names a planted row); validation:outliers [titer_g_L] (warning) (names a planted row) |  | kalos/validation/checks.py (check_duplicate_rows ignores nothing but outcome columns) / unplanned (exclude identifier columns from the duplicate comparison) |
| `fahrenheit_rows_bare` | sheet | **caught** | validation:physical_bounds [temperature_C] (warning) |  |  |
| `fahrenheit_rows_labeled` | sheet | **caught** | validation:units_consistency [temperature_C] (info) |  |  |
| `impossible_value` | sheet | **missed** |  |  | kalos/validation/checks.py (check_physical_bounds) / existing check (no plan item) |
| `mixed_units_labeled` | sheet | **caught** | validation:units_consistency [glucose_gL] (warning) |  |  |
| `mixed_units_unlabeled` | sheet | **caught** | validation:outliers [titer_g_L] (warning); validation:physical_bounds [titer_g_L] (warning) |  |  |
| `swapped_columns` | sheet | **caught** | validation:physical_bounds [pH] (error); validation:physical_bounds [temperature_C] (warning) |  |  |
| `swapped_levels` | sheet | **noticed (blind spot)** | validation:outliers [titer_g_L] (warning) (names a planted row) |  | none today (as-run record / bridging controls) / Crux #1 recipe_deviation (with as-run recipes, F11); DOMAIN H4 bridging controls (unplanned) |
| `too_few_rows_5` | sheet | **caught** | refusal:too_few_rows (error) |  |  |
| `too_few_rows_8` | sheet | **noticed** | reliability:does_not_clear_floor (warning) |  | kalos/portal/analysis.py (reliability verdict) / F04 (not_judged below about 12 rows; ML_AUDIT N3) |
| `transposed_sheet` | sheet | **caught** | orientation:transposed (info) |  |  |
| `transposed_sheet_no_id` | sheet | **caught** | orientation:transposed (info) |  |  |
| `typo_x1000_feature` | sheet | **caught** | validation:outliers [agitation_rpm] (warning) |  |  |
| `typo_x1000_target` | sheet | **caught** | validation:outliers [titer_g_L] (warning) |  |  |
| `typo_x10_feature` | sheet | **caught** | validation:outliers [glucose_gL] (warning) |  |  |
| `typo_x10_target` | sheet | **caught** | validation:outliers [titer_g_L] (warning) |  |  |

## Clean cases (false alarms)

| Clean case | Surface | False alarm | Detectors fired / refusals | Standing warnings (not counted) |
|---|---|---|---|---|
| `clean_bioprocess160` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess30` | sheet | **yes** | reliability:does_not_clear_floor; validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | reliability:space_filling_advice; validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess60` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |
| `clean_kaggle150` | sheet | **yes** | validation:outliers/physical_bounds; validation:physical_bounds; validation:physical_bounds/units_consistency; +1 more | validation:controls_present; validation:provenance_metadata |
| `clean_results_plain` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_results_surprise` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_scale_in_ladder` | scale | **yes** | scale_readout:refusal |  |
| `clean_scaleup90` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |

Synthetic planted faults only; see PROTOCOL.md section 8 for what this does not claim.
