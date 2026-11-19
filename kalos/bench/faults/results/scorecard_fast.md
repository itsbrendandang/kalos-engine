# Planted-fault scorecard (fast tier): synthetic planted faults

Protocol v1, seed 20261119, commit `bdd2d974c3` on `feat/planted-faults`, 2026-11-19T20:15:10Z.

- Caught: 15/23 (65%; Wilson 95% 0.45-0.81)
- Caught or noticed: 17/23 (74%; Wilson 95% 0.54-0.88)
- Caught, faults detectable from the input: 15/22 (68%; Wilson 95% 0.47-0.84)
- Clean cases with a false alarm: 5/7 (71%; Wilson 95% 0.36-0.92)
- Missed: 6; crashed: 0; skipped (base unavailable): 0

## Faults

| Fault | Family | Outcome | Evidence | Harm on a miss | Owner (module / plan) |
|---|---|---|---|---|---|
| `result_copy_paste` | result | **missed** |  |  | kalos/portal/campaign.py (set_result accepts any finite value) / Crux #1 result referee (result_duplicate_entry; F11) |
| `result_negative` | result | **caught** | validation:physical_bounds [titer_g_L] (error) |  |  |
| `result_x10` | result | **missed** | right reason fired but its 20-row list omits the planted rows |  | kalos/portal/campaign.py (set_result accepts any finite value) / Crux #1 result referee (result_magnitude_jump; F11) |
| `result_x1000` | result | **missed** | right reason fired but its 20-row list omits the planted rows |  | kalos/portal/campaign.py (set_result accepts any finite value) / Crux #1 result referee (result_magnitude_jump; F11) |
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
| `clean_bioprocess160` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess30` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |
| `clean_bioprocess60` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |
| `clean_kaggle150` | sheet | **yes** | validation:outliers/physical_bounds; validation:physical_bounds; validation:physical_bounds/units_consistency; +1 more | validation:controls_present; validation:provenance_metadata |
| `clean_results_plain` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_results_surprise` | result | no |  | validation:controls_present; validation:provenance_metadata |
| `clean_scaleup90` | sheet | **yes** | validation:outliers/physical_bounds; validation:units_consistency/outliers/physical_bounds | validation:controls_present; validation:provenance_metadata |

Synthetic planted faults only; see PROTOCOL.md section 8 for what this does not claim.
