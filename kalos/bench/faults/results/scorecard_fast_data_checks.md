# Planted-fault scorecard (fast tier): synthetic planted faults

Protocol v1, seed 20261119, commit `e864b3efbd` on `feat/data-checks`, 2026-11-20T09:50:59Z.

- Caught: 21/23 (91%; Wilson 95% 0.73-0.98)
- Caught or noticed: 21/23 (91%; Wilson 95% 0.73-0.98)
- Caught, faults detectable from the input: 21/22 (96%; Wilson 95% 0.78-0.99)
- Clean cases with a false alarm: 0/6 (0%; Wilson 95% 0.00-0.39)
- Missed: 2; crashed: 0; skipped (base unavailable): 1

## Faults

| Fault | Family | Outcome | Evidence | Harm on a miss | Owner (module / plan) |
|---|---|---|---|---|---|
| `result_copy_paste` | result | **caught** | result_entry:result_duplicate_entry (warning) [planned code] |  |  |
| `result_negative` | result | **caught** | result_entry:result_physically_impossible (warning) [planned code]; validation:physical_bounds [titer_g_L] (error) |  |  |
| `result_x10` | result | **caught** | result_entry:result_magnitude_jump (warning) [planned code]; validation:outliers [titer_g_L] (warning) |  |  |
| `result_x1000` | result | **caught** | result_entry:result_magnitude_jump (warning) [planned code]; validation:outliers [titer_g_L] (warning) |  |  |
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
| `too_few_rows_8` | sheet | **missed** |  |  | kalos/portal/analysis.py (reliability verdict) / F04 (not_judged below about 12 rows; ML_AUDIT N3) |
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
| `clean_bioprocess30` | sheet | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess60` | sheet | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_results_plain` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_results_surprise` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_scaleup90` | sheet | no |  | validation:controls_present; validation:provenance_metadata |

Skipped: `clean_kaggle150` (base kaggle150 unavailable)

Synthetic planted faults only; see PROTOCOL.md section 8 for what this does not claim.
