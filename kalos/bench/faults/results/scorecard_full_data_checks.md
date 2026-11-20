# Planted-fault scorecard (full tier): synthetic planted faults

Protocol v1, seed 20261119, commit `e864b3efbd` on `feat/data-checks`, 2026-11-20T09:50:04Z.

- Caught: 34/36 (94%; Wilson 95% 0.82-0.98)
- Caught or noticed: 34/36 (94%; Wilson 95% 0.82-0.98)
- Caught, faults detectable from the input: 34/35 (97%; Wilson 95% 0.85-0.99)
- Clean cases with a false alarm: 2/8 (25%; Wilson 95% 0.07-0.59)
- Missed: 2; crashed: 0; skipped (base unavailable): 0

## Faults

| Fault | Family | Outcome | Evidence | Harm on a miss | Owner (module / plan) |
|---|---|---|---|---|---|
| `batch_offset_no_column` | model | **missed** |  |  | kalos/portal/analysis.py (CV groups) / F05 leak-proof CV grouping (round index on folded rows) + F04 |
| `batch_offset_with_column` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `covariate_as_setpoint` | model | **caught** | columns:covariate [passage] (info) [planned code] |  |  |
| `id_like_numeric_categorical` | model | **caught** | columns:categorical [clone] (info) |  |  |
| `jittered_replicates` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `leaked_column_named` | model | **caught** | columns:dropped_output [yield_tier] (info) |  |  |
| `leaked_column_renamed` | model | **caught** | columns:dropped_output [qc_grade] (info); validation:target_leak [qc_grade] (warning) [planned code] |  |  |
| `outcome_as_factor` | model | **caught** | columns:dropped_output [contamination] (info) |  |  |
| `pure_noise_target_s1` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `pure_noise_target_s2` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `pure_noise_target_s3` | model | **caught** | reliability:does_not_clear_floor (warning) |  |  |
| `result_copy_paste` | result | **caught** | result_entry:result_duplicate_entry (warning) [planned code] |  |  |
| `result_negative` | result | **caught** | result_entry:result_physically_impossible (warning) [planned code]; validation:physical_bounds [titer_g_L] (error) |  |  |
| `result_x10` | result | **caught** | result_entry:result_magnitude_jump (warning) [planned code]; validation:outliers [titer_g_L] (warning) |  |  |
| `result_x1000` | result | **caught** | result_entry:result_magnitude_jump (warning) [planned code]; validation:outliers [titer_g_L] (warning) |  |  |
| `scale_far_extrapolation` | scale | **caught** | scale_readout:refusal (error) |  |  |
| `scale_physics_extrapolation` | scale | **caught** | scale_readout:refusal (error) |  |  |
| `constant_target` | sheet | **caught** | refusal:target_constant (error) |  |  |
| `duplicate_run_id` | sheet | **caught** | validation:duplicate_ids [run_id] (warning) |  |  |
| `duplicated_rows_exact` | sheet | **caught** | validation:duplicate_rows (warning) |  |  |
| `duplicated_rows_new_id` | sheet | **caught** | validation:duplicate_rows (warning) |  |  |
| `fahrenheit_rows_bare` | sheet | **caught** | validation:physical_bounds [temperature_C] (warning) |  |  |
| `fahrenheit_rows_labeled` | sheet | **caught** | validation:units_consistency [temperature_C] (info) |  |  |
| `impossible_value` | sheet | **caught** | validation:physical_bounds [DO_pct] (error) |  |  |
| `mixed_units_labeled` | sheet | **caught** | validation:units_consistency [glucose_gL] (warning) |  |  |
| `mixed_units_unlabeled` | sheet | **caught** | validation:outliers [titer_g_L] (warning); validation:physical_bounds [titer_g_L] (warning) |  |  |
| `swapped_columns` | sheet | **caught** | validation:physical_bounds [pH] (error); validation:physical_bounds [temperature_C] (warning) |  |  |
| `swapped_levels` | sheet | **missed (blind spot)** |  |  | none today (as-run record / bridging controls) / Crux #1 recipe_deviation (with as-run recipes, F11); DOMAIN H4 bridging controls (unplanned) |
| `too_few_rows_5` | sheet | **caught** | refusal:too_few_rows (error) |  |  |
| `too_few_rows_8` | sheet | **caught** | reliability:not_judged (warning) [planned code] |  |  |
| `transposed_sheet` | sheet | **caught** | orientation:transposed (info) |  |  |
| `transposed_sheet_no_id` | sheet | **caught** | orientation:transposed (info) |  |  |
| `typo_x1000_feature` | sheet | **caught** | validation:outliers [agitation_rpm] (warning) |  |  |
| `typo_x1000_target` | sheet | **caught** | validation:outliers [titer_g_L] (warning) |  |  |
| `typo_x10_feature` | sheet | **caught** | validation:outliers [glucose_gL] (warning) |  |  |
| `typo_x10_target` | sheet | **caught** | validation:outliers [titer_g_L] (warning) |  |  |

## Clean cases (false alarms)

| Clean case | Surface | False alarm | Detectors fired / refusals | Standing warnings (not counted) |
|---|---|---|---|---|
| `clean_bioprocess160` | sheet | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess30` | sheet | **yes** | reliability:does_not_clear_floor | reliability:space_filling_advice; validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess60` | sheet | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_kaggle150` | sheet | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_results_plain` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_results_surprise` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_scale_in_ladder` | scale | **yes** | scale_readout:refusal |  |
| `clean_scaleup90` | sheet | no |  | validation:controls_present; validation:provenance_metadata |

Synthetic planted faults only; see PROTOCOL.md section 8 for what this does not claim.
