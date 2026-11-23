# Production clones and strains

Kalos optimizes the PROCESS (media, pH, temperature, feed) for production hosts
that are already chosen: CHO clones, E. coli or Pichia production strains. A
clone or strain is an **opaque label**. Kalos compares labels for equality and
counts them; it never reads, designs, ranks or scores a sequence, construct or
variant, and it runs no protein language model. Design: `STRAIN_VIEWS.md`
(approved: P1, P2, P3; default policy `fixed`).

Code: `kalos/portal/analysis_parts/strain_roles.py` (roles),
`strain_state.py` (per-analysis state), `strain_block.py` (the blocks),
`kalos/core/strain.py` (statistics, numpy/scipy), `kalos/core/optimize.py`
(`fixed_features`), `kalos/portal/clones_api.py` + `clone_registry.py`
(registry, schema v7). Tests: `tests/test_strain_*.py`, `tests/test_clones_api.py`,
`tests/test_schema_v7.py`.

## Honesty rules

- **Measured, derived and predicted never share a field.** Every block carries
  `basis`: `measured` (per-clone summaries, `by_round`, `categorical_effects`,
  titer history), `derived` (the process-adjusted clone effect), `predicted`
  (`gp_at_reference`, `slices_by_level`, `predicted_optimum_by_clone`).
- **Intervals carry their coverage and kind.** Measured means: 95% t-interval,
  only when `n_runs >= 3`. Adjusted effects: 95%, `approximate: true`, `ci_method`.
  GP at the reference process: 90% normalized conformal (`calibrated_on:
  grouped_cv_oof`), or a labeled `gp_posterior` band when conformal is refused.
  Slices: `band_kind: "gp_posterior"`, `coverage`, `calibrated: false`.
- **Refusals carry `code`, `reason`, `what_would_fix_it`.** Thin data is
  `insufficient`, never "no interaction" and never a guessed number.
- **No clone ranking score.** Levels are listed in label order (the registry
  summary orders lead, backup, candidate, retired, then unregistered by name).
  Wording: "clone effect at these process conditions", "associated with".
- **Kalos never predicts an unseen clone** (`clone_not_in_training`): the
  categorical kernel has nothing to say about a new label.

## Declaring the clone and covariates

| Where | Keys |
|---|---|
| Upload `roles` JSON | `strain`, `strain_policy` (`fixed`\|`factor`), `strain_level`, `covariates` (list) |
| Campaign setup (`PUT /api/campaigns/{id}/setup`) | `strain_column`, `strain_policy`, `strain_level`, `covariates` |

The setup wins when both are present (campaigns do not store upload roles, so a
campaign declares its clone in the setup). The keys are stored only when set:
a setup without them keeps its hash (`settings_in_sync` is unchanged for every
existing campaign).

- **Policy `fixed`** (the default once a clone column is declared): the GP is
  fitted on every clone (the other clones lend strength to the shape of the
  process response through the shared continuous kernel), and every proposal
  holds the clone at one level. The level is `strain_level`, else the
  registry's single `status: lead` clone (when it has runs), else the clone with
  the most runs (ties: label order). `strain_summary.fixed_level_source` says
  which, and `fixed_level_note` says the default is a run count (label order on
  a tie), never a ranking. A named level with no runs is refused: proposals withheld,
  `proposal_notice.code = clone_not_in_training`; `PUT setup` refuses it too.
- **Policy `factor`**: the proposals choose among the clones (a clone-by-process
  screen). A proposal on a clone with fewer than 3 runs is marked `mode:
  "explore"` with `clone_refusal.code = clone_too_few_runs`.
- **Detected, not declared**: with no roles, a column named `clone`, `strain` or
  `cell line` (any case, `_id` suffixes included) is the clone, policy `factor`
  (what a text clone column got before), plus the finding
  `strain_column_detected` asking the user to declare it. Numeric-looking IDs
  (`101`, `204`) are labels: `101`, `101.0` and `"101"` are one clone.
- **More than 12 clones** (`too_many_clones`): the clone is not a model input;
  the measured summary is still produced.

**Covariates** (passage, generation, PDL, thaw ID, vial, cell bank; detected by
name with no roles) are recorded run properties and are **never proposed**:

- A numeric covariate is a model input when every run with a measured target has
  a value. Proposals hold it at a reference value (the median over the fixed
  clone's runs, else over all fitted runs), returned as `proposals[].held`
  (`{value, basis, proposed: false}`) and never in `recipe`. In-flight runs are
  given the held value so they still count as taken.
- A numeric covariate missing on some runs (campaign runs logged without a
  passage) is recorded but not fitted (`covariates.not_fitted` says why). Nothing
  is imputed and no run is dropped for it.
- A label covariate (thaw ID) is not a model input (the next run uses a thaw the
  model has never seen); it feeds the confounding checks.
- Bounds or linear constraints on a covariate are refused
  (`covariate_not_proposable`).

**Linear constraints with a clone** (ML_AUDIT N2): with the clone fixed, the
optimizer holds it and runs the continuous path with the constraints. In
`factor` mode the constraints go through the exact mixed enumeration
(`optimize_acqf_mixed` with `inequality_constraints`) when the free categorical
combinations number at most 64; above that they are refused
(`constraints_with_categoricals`), as they still are for campaigns with
categorical inputs and no clone column.

## Analysis fields (additive; present on every analysis)

| Field | Content |
|---|---|
| `strain_summary` | The per-clone block below. `available: false` with `reason` when there is no clone column. |
| `slices_by_level` | `{<clone column>: {<label>: [slice, ...]}}`: the R2 slices of the same GP with the clone set to each level, at the same reference process as `slices`. Each slice has the `slices[]` fields plus `level`, `band_kind: "gp_posterior"`, `coverage` (0.9), `calibrated: false`, `basis: "predicted"`. Covariates are not swept. Measured runs are never in the payload (the web draws them as a rug). `{}` without a modeled clone. |
| `slices_by_level_skipped` | `{<clone column>: [{level, reason, n_runs, message}]}`, e.g. `clone_too_few_runs` ("slice not shown: n=2 < 3 runs"). |
| `oof_levels` | `{<clone column>: [label, ...]}` aligned to `oof` rows (the parity-plot filter). |
| `covariates` | `{columns, source, fitted, held: {name: {value, basis}}, not_fitted: {name: reason}, note}`. |
| `drivers_skipped`, `correlation_skipped` | `[{feature, reason}]` for every categorical input: a signed rank correlation is not defined for unordered labels. |
| `categorical_effects` | `{unit: "recipe", method: "kruskal_wallis", effect_size: "epsilon_squared", basis: "measured", features: [{name, n_levels, n_units, levels: [{level, n_recipes, mean}], h, p, epsilon_sq, survives_fdr, reason}]}`; `null` with no categorical input. An association across levels, not adjusted for the other inputs. |
| `proposals[].held` | Covariates held in this proposal (never in `recipe`). |
| `proposals[].clone` | `{column, level, policy, n_runs}`. G5: the chip is hollow while the run is proposed/awaiting, filled once measured. |
| `proposals[].nearest_training_distance_same_clone`, `extrapolated_for_clone`, `extrapolated_for_clone_features` | The distance metric of `nearest_training_distance` over this clone's runs only, and the inputs where the proposal leaves this clone's observed range. |
| `drivers[].role` | `"covariate"` for a fitted covariate. |
| `provenance[].role` | `"strain"` or `"covariate"`. |
| `gp_shapes` | Now available on mixed fits (categorical codes held at the incumbent; stated in `unmodeled`). |

### `strain_summary`

```jsonc
{
  "available": true, "reason": null,
  "column": "clone", "source": "setup",          // setup | declared | detected
  "policy": "fixed", "modeled": true, "numeric_looking_labels": false,
  "fixed_level": "CHO-C33", "fixed_level_source": "setup",  // setup | declared | registry_lead | most_runs
  "reference_level": "CHO-C33",                  // effects are relative to it
  "n_levels": 3, "n_levels_with_3_runs": 3, "target": "titer_g_L",
  "outcome_units": {"titer_g_L": "g/L"}, "outcome_direction": {"titer_g_L": "higher_is_better"},
  "levels": [{
    "label": "CHO-B07", "n_runs": 8, "n_recipes": 8, "n_replicated_recipes": 0, "n_fitted": 8,
    "rounds": ["0", "1"], "scales": null,
    "outcomes": {"titer_g_L": {"basis": "measured", "n_runs": 8, "mean": 2.33, "sd": 0.44,
      "ci95": [1.96, 2.69], "coverage": 0.95, "ci_reason": null, "median": 2.3, "min": 1.7,
      "max": 3.0, "best": 3.0, "best_run_id": "17", "replicate_sd": null, "...": "..."}},
    "covariates": {"passage": {"min": 9, "max": 27, "n": 8}},
    "adjusted_effect": {"basis": "derived", "estimate": -0.55, "ci95": [-1.0, -0.1],
      "coverage": 0.95, "approximate": true, "shrinkage_w": 0.93, "mostly_pooled": false,
      "relative_to": "CHO-C33", "refusal": null},
    "gp_at_reference": {"basis": "predicted", "pred": 3.09, "lo": 2.72, "hi": 3.47,
      "coverage": 0.9, "method": "conformal_normalized", "calibrated_on": "grouped_cv_oof",
      "at": {"temp_C": 35.0, "feed_pct": 5.8}, "refusal": null},
    "oof_coverage": {"n": 8, "nominal": 0.9, "empirical": 0.875, "reason": null},
    "flags": []        // clone_too_few_runs, mostly_pooled, clone_confounded_with:<col>,
                        // clone_process_imbalance:<input>, clone_effect_methods_disagree
  }],
  "held_in_proposals": {"passage": {"value": 17.0, "basis": "median of the fitted runs on the fixed clone"}},
  "adjusted_method": "oof_residual_empirical_bayes", "tau": 0.6, "tau2": 0.36, "i2": 0.93,
  "adjusted_note": "clone effect at these process conditions ...",
  "process_balance": {"flag": false, "by_input": {"temp_C": {"smd_max": 0.3, "coverage_min": 0.8,
      "flag": false, "by_level": {"CHO-B07": {"n": 8, "min": 33.1, "max": 36.9, "mean": 35.0,
      "smd": 0.1, "coverage": 0.95}}}}, "messages": []},
  "confounded_with": [{"column": "round", "cramers_v": 0.62, "fully": false,
      "confined": {"CHO-A12": "0"}, "aliased": []}],
  "scale": {"column": null, "levels": [], "mixed": false, "note": null},
  "interaction": {"rows": {"temp_C": {"status": "tested", "Q": 4.1, "df": 2, "p": 0.13,
      "p_bh": 0.2, "i2": 0.51, "flag": false, "n_qualifying": 3, "reason": null}},
    "cells": {"CHO-B07": {"temp_C": {"status": "estimated", "slope": -0.1, "se": 0.05,
      "ci95": [-0.2, 0.0], "n": 8, "adjusted_for_other_inputs": false, "reason": null}}},
    "thin_rule": {"min_runs": 6, "min_levels": 3, "min_range_frac": 0.5}, "flags": [],
    "note": "insufficient means the data cannot test it, not that there is no interaction"},
  "predicted_optimum_by_clone": {"basis": "predicted", "by_input": {"temp_C": {"CHO-B07":
      {"argmax": 35.1, "mean": 3.1, "lo": 2.8, "hi": 3.4, "coverage": 0.9}}},
    "flags": [], "shift_threshold_frac": 0.25},
  "leave_one_clone_out": {"spearman": 0.17, "n_clones": 3, "n_oof": 24, "reason": null, "note": "..."},
  "by_round": {"available": false, "column": null, "reason": "no round column in the sheet ..."},
  "oof_coverage_method": "...",
  "refusals": [{"code": "clone_too_few_runs", "level": "CHO-A12", "reason": "...", "what_would_fix_it": "..."}]
}
```

How each part is computed:

- **Measured** (`levels[].outcomes`): every outcome column in `targets`, over runs
  with a value and a clone label. Mean, SD (n >= 2), median, range, best (target
  only: it needs the goal's direction), `replicate_sd` (pooled within-recipe SD
  with >= 2 replicated recipes), 95% t-interval only when `n_runs >= 3`
  (`ci_reason: "n<3"` otherwise). With a scale column holding 2+ scales, the
  pooled mean and interval are withheld (`ci_reason: "scales_mixed"`) and each
  scale is summarized in `by_scale`: never pooled across scales.
- **Adjusted effect** (`adjusted_effect`, derived): a GP on the continuous
  process inputs only (no clone, no covariates), cross-validated on the same CV
  groups as the main fit; its out-of-fold residuals per clone feed a
  random-intercept empirical-Bayes estimate (DerSimonian-Laird tau^2, shrinkage
  weight `shrinkage_w` = "data weight"; below 0.5 the clone is `mostly_pooled`).
  Effects are relative to `reference_level`. The 95% interval is approximate
  (`ci_method`). Withheld per clone with `clone_too_few_runs` (n < 3),
  `clone_effect_not_separable` (confounded), and for all clones with
  `clone_effect_needs_two_clones` (< 2 clones with 3+ runs).
- **Predicted** (`gp_at_reference`): the served GP at the incumbent's
  continuous settings with the clone set to each level (n >= 3), with the 90%
  normalized conformal band. When the derived and predicted differences from the
  reference clone disagree in sign, the clone is flagged
  `clone_effect_methods_disagree` (both are shown; neither overrides the other).
- **Confounding**: Cramer's V between the clone and every label covariate,
  other categorical input and block column by name (round, batch, lot, thaw,
  vial, scale, vessel, plate, day, date, bank). A clone whose runs share no level
  of such a column with any other clone (for example the only clone in round 2)
  is not separable from it: its adjusted effect is withheld
  (`clone_effect_not_separable`, flag `clone_confounded_with:<column>`).
  `confined` lists clones confined to one level that is shared (separable).
- **Process balance**: per clone and continuous input, the share of the
  observed range covered and the standardized mean difference;
  `clone_process_imbalance:<input>` when |SMD| > 0.8 or coverage < 0.3 for a clone
  with 3+ runs. Raw means of such clones partly reflect the process.
- **Interaction screen**: per clone and process input, the within-clone slope
  (OLS, adjusted for the other inputs when n >= p + 4) with a percentile
  bootstrap CI (B = 500, seeded). A cell is `insufficient` unless the clone has
  >= 6 runs, >= 3 distinct settings and covers >= 50% of the input's range; a row
  is `insufficient` unless 2+ clones qualify. Tested rows report Cochran's Q, p
  (Welch's small-sample correction for Q; the chi-squared p is `p_chi2`, which
  rejected 9.5% of null simulations at a nominal 5%), BH-adjusted `p_bh` and I^2; `flag` (clone-dependent response) needs p < 0.05
  and I^2 >= 50%.
- **Predicted optimum by clone**: the argmax of each per-clone slice with its
  band. `predicted_optimum_differs_by_clone` when two clones' argmaxes are more
  than 25% of the range apart and each clone's band at its own argmax sits above
  its band at the other's.
- **Leave-one-clone-out**: Spearman over held-out clones, only with >= 3 clones
  of >= 5 runs. Reported, never a gate.
- **`by_round`**: best measured target per clone per round when the sheet has a
  round column (campaign runs carry no round label, so this is often unavailable
  with a reason).

Refusal codes: `strain_column_not_found`, `too_many_clones`,
`clone_not_in_training`, `clone_too_few_runs`, `fixed_clone_too_few_runs`,
`clone_effect_needs_two_clones`, `reference_clone_too_few_runs`,
`clone_effect_not_separable`, `clone_effect_not_estimated`,
`interaction_insufficient`.

**Anonymize**: with `anonymize=True` every clone label (values, dict keys and
inside messages) becomes `level_<hash>` in `strain_summary`, `slices_by_level`,
`oof_levels`, `categorical_effects`, `proposals[].recipe` and `.clone`, whatever
the column is called; `best_run_id` is null. The registry keeps real labels and
is never joined to an anonymized analysis.

## Graphics the web renders from these fields (no client-side statistics)

| | Fields |
|---|---|
| G1 clone x outcome heatmap | `levels[].outcomes[col].{mean, ci95, n_runs, n_recipes, ci_reason}`, `outcome_units`, `outcome_direction`, `process_balance.flag`, `confounded_with`; hatch when `ci95` is null |
| G2 clone x input interaction | `interaction.cells[clone][input]`, `interaction.rows[input]`, `predicted_optimum_by_clone` |
| G3a per-clone small multiples | `slices_by_level`, `slices_by_level_skipped`; measured points from the runs as a rug |
| G3b best by round | `by_round` |
| G3c titer history | `GET /api/campaigns/{id}/clones/{label}/titer-history` |
| G4 clone comparison table | `levels[]` (measured, `adjusted_effect`, `gp_at_reference`, `covariates`, `flags`), `reference_level`, registry fields from `GET .../clones/summary` |
| G5 clone chip | `proposals[].clone`, `proposals[].recipe[<clone column>]`, run recipes; `strain_summary.policy`/`fixed_level` for "All proposals use X (fixed)" |

## Clone registry (labels only, schema v7)

Code: `kalos/portal/clone_registry.py` (storage), `kalos/portal/clones_api.py`
(routes). Tests: `tests/test_clones_api.py`, `tests/test_schema_v7.py`. The
registry's single `status: lead` clone is the default fixed level
(`setup_settings.setup_for_analysis` passes it to the analysis as `strain_lead`;
not part of the setup hash). Schema v8 is the prediction history
(docs/MIGRATIONS.md); the other Crux items take v9 or later.

A campaign can keep a small registry of the production clones (or strains) it
runs: a status and a few text labels per clone. Kalos stores, shows and filters
these values. It never parses them, never computes from them except counts and
equality, and never sends them anywhere. **Kalos stores labels only**: no
sequence, plasmid map, structure or database accession is accepted.

Registry edits live outside the campaign state blob, so they never mint a new
campaign generation and never put an analysis out of sync.

### Storage (schema v7)

- `campaign_clones(tenant, campaign_id, label, fields, status, updated_at,
  updated_by)`, primary key `(tenant, campaign_id, label)`. `label` is 1..64
  characters and must equal the run-sheet value exactly (it is not trimmed).
  `status` is one of `lead`, `backup`, `retired`, `candidate`. `fields` is a JSON
  object of the allowlisted keys below.
- `clone_titer_history(id, tenant, campaign_id, label, recorded_at, measured_on,
  basis, value, unit, scale_label, age_kind, age_value, source_note, actor,
  previous_id, reason)`: append-only (UPDATE and DELETE are rejected by triggers).
  A correction is a new row that names the row it supersedes in `previous_id` and
  carries a non-blank `reason`.
- Limits: 200 clones per campaign (`too_many_clones`), 20,000 titer rows per
  campaign (`titer_history_limit`).

### Fields (all optional except `status`)

| Key | Type | Limit | Example |
|---|---|---|---|
| `status` | `lead` \| `backup` \| `retired` \| `candidate` | required | `lead` |
| `host` | string | 64 | `CHO-K1 GS-KO`, `K. phaffii X-33` |
| `parent_cell_bank` | string | 64 | `RCB-2024-03` |
| `construct_id` | string (opaque) | 64 | `pK-mAb7-v3` |
| `phenotype_label` | string | 64 | `MutS`, `MTX 50 nM` |
| `product` | string | 64 | `mAb7 (IgG1)` |
| `passage_at_banking` | finite number or string | 64 | `12`, `P12` |
| `note` | string (newlines allowed) | 500 | free text, shown escaped |

Strings are trimmed; a blank or `null` value means "not set". Control
characters are refused (newline and tab are allowed in `note`).

### Routes

All routes follow the campaign API rules: the scope is checked before the id is
looked at; the tenant is always the caller's; a malformed, missing or another
tenant's campaign id gets the byte-identical `404 {"error": "campaign not
found", "code": "not_found"}`; errors never echo a submitted label, key or value;
an archived campaign answers `409 campaign_archived` on writes; the JSON body cap
is `KALOS_MAX_REQUEST_BYTES` (default 1 MiB, `413 payload_too_large`). Every
write records an audit row (`campaign.clone.put`,
`campaign.clone.titer_append`) with the label, status / basis, the keys sent and
the row id, never field values.

The `{label}` path segment is the URL-encoded clone label: 1..64 characters, not
blank, no control characters (`400 invalid_label` otherwise). A label that
contains `/` cannot be addressed.

#### `GET /api/campaigns/{id}/clones` (read)

```jsonc
{
  "campaign_id": "cmp_…",
  "clones": [{"label": "CHO-B07", "status": "lead",
              "fields": {"host": "CHO-K1 GS-KO"}, "updated_at": 1730000000.0,
              "updated_by": "bob@acme"}],
  "seen_in_data": ["CHO-A12", "CHO-B07"],     // strain_summary.levels[].label of the latest analysis
  "unregistered_labels": ["CHO-A12"],         // seen_in_data minus the registry, by name
  "max_clones": 200,
  "seen_in_data_reason": "…"                  // only when seen_in_data is empty because there is
                                              // no analysis, no/unavailable strain_summary, or the
                                              // analysis was anonymized (labels are aliases)
}
```

#### `PUT /api/campaigns/{id}/clones/{label}` (write)

Creates or replaces the whole row (omitted keys are cleared). Body: the fields
table above, `extra="forbid"`. `sequence`, `fasta`, `smiles`, `pdb`, `uniprot` and
any other key are refused with `400 invalid_body` ("Kalos stores clone labels
only; …"). Over 200 clones: `409 too_many_clones` (with `max_clones` and
`what_would_fix_it`); editing an existing clone is always allowed.

Response: `{"campaign_id", "clone": {"label", "status", "fields", "updated_at",
"updated_by", "created": bool}}`.

#### `POST /api/campaigns/{id}/clones/{label}/titer-history` (write)

Appends one MEASURED value to a registered clone (`404 clone_not_registered`
otherwise).

| Key | Type | Notes |
|---|---|---|
| `basis` | `harvest_titer` \| `qP` \| `space_time_yield` | required |
| `value` | finite number | required |
| `unit` | `g/L` \| `mg/L` \| `pg/cell/day` \| `g/L/day` | required; must match the basis: `harvest_titer` g/L or mg/L, `qP` pg/cell/day, `space_time_yield` g/L/day (`unit_basis_mismatch`) |
| `measured_on` | `YYYY-MM-DD` | optional (`invalid_date`) |
| `scale_label` | string ≤ 64 | optional, e.g. `ambr15` |
| `age_kind` | `passage` \| `generation` \| `pdl` | optional |
| `age_value` | finite number | optional |
| `source_note` | string ≤ 200 | optional |
| `previous_id` | integer ≥ 1 | correction only: an existing row of the same campaign and label (`previous_not_found`) |
| `reason` | string ≤ 500 | required with `previous_id` (`reason_required`); refused without it (`previous_id_required`) |

The actor and `recorded_at` come from the token and the server clock. Response:
`{"campaign_id", "row": {"id", "label", "recorded_at", …the fields…, "actor",
"basis_kind": "measured", "is_correction": bool}}`.

#### `GET /api/campaigns/{id}/clones/{label}/titer-history` (read)

Query: `limit` (1..500, default 100), `before_id` (cursor). Newest first, the
same page shape as `/results/history`:

```jsonc
{"campaign_id": "cmp_…", "label": "CHO-B07", "limit": 100,
 "history": {"items": [{"id": 7, "label": "CHO-B07", "recorded_at": …, "measured_on": "2025-03-01",
                        "basis": "harvest_titer", "value": 3.2, "unit": "g/L", "scale_label": "ambr15",
                        "age_kind": "passage", "age_value": 14.0, "source_note": null,
                        "actor": "bob@acme", "previous_id": null, "reason": null,
                        "basis_kind": "measured", "is_correction": false}],
             "has_more": false, "next_before_id": null},
 "notes": "…"}
```

#### `GET /api/campaigns/{id}/clones/summary` (read)

The latest stored analysis's `strain_summary`, joined with the registry.

- No analysis: `409 {"code": "no_analysis", "what_would_fix_it": …}`.
- No `strain_summary`, or `available: false`: `200 {"campaign_id", "available":
  false, "reason"}`.
- Anonymized analysis (`run.anonymize` true): the block is returned unchanged
  plus `registry_join: {"applied": false, "reason": "anonymized analysis: labels
  are aliases"}`. No level gets a `registry` key and there is no
  `registry_order` (aliases never match real labels).
- Otherwise: the block plus
  - `levels[i].registry`: `{"status", "fields"}` or `null` (unregistered);
  - `unregistered_labels`: level labels without a registry row, by name;
  - `registry_order`: the level labels ordered lead, backup, candidate, retired
    (by name within a status), then unregistered by name. Never ordered by a
    computed score;
  - `registry_join: {"applied": true, "reason": null}`.

The route is registered before the `/clones/{label}` routes. Only GET exists on
`/clones/summary` and no GET exists on `/clones/{label}`, so it is never read as
a label; a `PUT /clones/summary` registers a clone literally named `summary`.

### Python helpers (`kalos.portal.clone_registry`)

- `list_clones(store, *, tenant, campaign_id) -> list[dict]`
- `get_clone(store, *, tenant, campaign_id, label) -> dict | None`
- `put_clone(store, *, tenant, campaign_id, label, status, fields, updated_by, auth_method=None) -> dict`
- `append_titer(store, *, tenant, campaign_id, label, row, actor, auth_method=None) -> dict`
- `list_titer_history(store, *, tenant, campaign_id, label, before_id=None, limit=100) -> dict`
- `lead_clone_label(store, *, tenant, campaign_id) -> str | None`: the single
  `status='lead'` clone; `None` when there is none or more than one.

All raise `CampaignNotFound` for a missing or another tenant's campaign.
