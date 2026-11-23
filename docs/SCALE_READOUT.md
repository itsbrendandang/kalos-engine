# Scale-Up Readout

A one-page answer to one question: what will this process do at the next scale, and should you believe the number?

Upload a multi-scale run sheet and a target scale.
Kalos backtests itself on your own scale ladder, compares against two naive baselines, and either issues a prediction with an interval or refuses and says why.
Refusing is a feature: the readout never claims more than your data supports.

Intended use: development decision support only.
Not a GMP or regulatory record.
Not validated under 21 CFR Part 11.

## Request

`POST /api/scale/readout` (requires the `write` scope), multipart form:

- `file`: the run sheet (CSV or XLSX), parsed by the same hardened uploader as `/api/run`.
  One row per run with the process parameters, `scale_L`, the response column, and - optionally - `agitation_rpm` and `airflow_L_per_min`.
- `target`: JSON with `scale_L`, `target_column`, `process_params` (a dict whose keys are the process columns), optional `physics_overrides` (any `ScaleFeatureConfig` field), and optional `agitation_rpm` / `airflow_L_per_min`.
  Whether `agitation_rpm` and `airflow_L_per_min` are actually required depends on the sheet, not the target JSON: see "Scale-only fallback" below.
  When they are required and missing, the response is a `422` `invalid_target` naming which one.

Responses:

- `200 text/html`: the readout page. Save it as PDF from the browser; page 1 holds the decision, and a provenance appendix follows.
- `422`: `{"error": ..., "code": ..., "failed_check": ..., "detail": ...}` when the sheet or target fails the gate or the target JSON is malformed. `code` and `failed_check` carry the same check name (docs/HARDENING.md, "Error envelope").
- `400`: upload rejected (same rules as `/api/run`).
- `503` + `Retry-After`: another analysis is running (one analysis slot, shared with `/api/run`).

JSON instead of HTML (same multipart inputs, `write` scope, size caps, analysis slot, and the same 400/422/503 envelopes): `POST /api/scale/readout?format=json`, `POST /api/scale/readout` with `Accept: application/json` ranked above `text/html`, or `POST /api/scale/readout.json`. HTML stays the default and is byte-identical; see "JSON readout" below.

Nothing is stored on the server.
To regenerate a readout, upload the same sheet again; the page prints the SHA-256 of the raw upload so anyone holding the file can verify it.

### Scale-only fallback

Real run sheets often do not record `agitation_rpm` or `airflow_L_per_min`.
Kalos reads the uploaded sheet's own columns to choose a feature set.
If the sheet has both `agitation_rpm` and `airflow_L_per_min`, it uses the "physics" feature set, unchanged from before this fallback existed.
The target JSON must then supply both values, or the response is a `422` `invalid_target` naming whichever is missing.
If either column is absent, it falls back to the "scale_only" feature set and names which column(s) are missing.

What scale_only models: the two physics features computable from scale alone, `log_volume_ratio` and `hydrostatic_pressure_mmHg`.
These carry the volume and hydrostatic-pressure scale effects.

What scale_only does not model: mixing (power per volume) and oxygen transfer (kLa), since both depend on agitation and airflow.
The page shows a neutral note under the decision banner naming exactly which inputs are missing.

How it is chosen: automatically, from the sheet, never from the target JSON.
In scale_only mode, `agitation_rpm` and `airflow_L_per_min` in the target are optional.
A value supplied anyway is accepted but ignored, since the sheet gives the model nothing to relate it to, and the page marks it "ignored: sheet does not record them" (or "not recorded" when the field was left unset).
The physics assumptions section only reports the assumptions scale_only actually uses (the geometry fields `log_volume_ratio` and `hydrostatic_pressure_mmHg` need, plus `reference_scale_L`); the power-number and van't Riet kLa constants are marked "not used (scale-only)".

## How the decision is made

1. **Gate.** At least 3 distinct scales, at most 20% non-finite rows, a target strictly larger than every trained scale, and a target at most 20x the largest trained scale.
   The first failing check is named in the 422.
   There is deliberately no minimum run count at any scale and no minimum row count at the rung-eligible scales.
   Real tech-transfer sheets often have dozens of bench runs and only one or two runs at each large scale, and that shape is exactly what this readout exists to read honestly.
   Whether there is enough evidence to license a number is decided later, and too little evidence is a refusal, not a 422.
2. **Ladder backtest.** For each scale whose smaller scales give at least 2 distinct scales and 4 rows, fit on the smaller scales only and predict that scale (`leave_one_scale_out_report(..., include_oof=True, held_out_scales=[k])`).
   The held-out scale itself may have any number of runs, including just one.
   The one-scale first rung is skipped and listed.
   Every rung uses the same explicit bounds as the final model, so the backtest evaluates the model that produces the number.
   Those bounds span the training scales and the target, so the same sheet can backtest differently for different targets: on the demo sheet the 10 L rung's MAE is 0.126 for a 7,500 L target and 0.603 for a 40,000 L target.
   For a fixed sheet and target the readout is deterministic.
   A rung with fewer than 3 runs (`MIN_RUNG_N_FOR_LICENSE`) still contributes its residuals to the pooled evidence, but it is marked "too few to judge": its own "beats both baselines" verdict is shown as "too few to judge" rather than yes or no, and it can never set the reference step ratio below.
3. **Decision.**
   The reference step ratio is the largest step ratio among rungs with at least 3 runs (`MIN_RUNG_N_FOR_LICENSE`) whose own MAE beats both naive baselines, and whose held-out scale is within one decade (`LICENSE_WINDOW_DECADES`) of the largest trained scale.
   The window matters because a step ratio alone ignores the scale regime: a 5x step won at 0.05 to 0.25 L says little about a 5x step at plant scale, where mixing and oxygen transfer behave differently.
   A winning rung below the window is shown as "yes (below license window)".
   A thin rung (fewer than 3 runs) is never allowed to license the reference, even if it happens to beat both baselines.
   - *Prediction issued*: the pooled ladder MAE beats both baselines, there are at least 10 pooled residuals, at least one rung licenses a reference, the target ratio is at most 2x that reference, and every process parameter is inside the trained range.
   - *Prediction issued with warnings*: as above, but a process parameter is out of range or the target ratio is 2x to 5x the reference.
   - *No prediction (refusal)*: no rung licenses a reference, the pooled ladder does not beat both baselines (judged only with at least 10 residuals), fewer than 10 pooled residuals, or the target ratio exceeds 5x the reference. Every condition that applies is listed, not just the first.
     A refusal is a normal 200 HTML page, not a 422: it still shows every section that could be computed (the rung table, the pooled baseline comparison when any residuals exist, the skipped rungs), plus a "What it would take" section (see below).
   - *Mixing and aeration range (F10, `kalos/scale/physics_range.py`)*: in the physics feature set, the target's power per volume, impeller tip speed, superficial gas velocity, gas flow per volume (vvm) and kLa proxy are computed with the same geometry and correlations as the model's features and compared with the measured runs' range. More than 5 % outside (`PHYSICS_EDGE_TOLERANCE`): the decision becomes at least *prediction issued with warnings*, with one reason per quantity naming the target value, the measured range and, where one setting controls it, the agitation or airflow that would keep it inside. More than 2x outside (`PHYSICS_REFUSE_FACTOR`) on P/V, vvm or kLa: *refusal*. Tip speed and gas velocity rise with scale under the usual constant-P/V and constant-vvm rules, so they only ever warn. Example: 2,000 L at 240 rpm from a ladder that stops at 300 L puts P/V at 3,127 W/m3, 3.1x the highest measured (995 W/m3), and is refused, with "about 164 rpm" as the setting that stays inside. These are proxies compared with the measured runs only, not engineering limits for a vessel. `physics_range` in the readout carries every quantity; `extrapolation` states how far beyond the measured scales the target is.
4. **Interval.** Split-conformal on the pooled ladder residuals, alpha 0.1, labeled *approximate coverage*: exchangeability does not hold for an unseen larger scale, so it is not called calibrated.
   A number is never issued without an interval.

All thresholds are named constants in `kalos/scale/readout.py` and are printed on every readout.

### What it would take

A refusal, or a prediction issued with a ratio warning, carries a "What it would take" section: plain, specific lines computed straight from the rules above, never an invented statistic.

- **Too few pooled residuals**: "N more run(s) at any scale of X L or larger", where N is how many more residuals are needed to reach 10, and X is the smallest scale that could become a rung target (in practice almost always the third-smallest distinct scale). A second line notes that runs at the largest scales also build the evidence that licenses larger steps.
- **No rung licenses a reference**: "at least 3 runs at a single scale of X L or larger whose backtest beats both baselines", using the same X.
- **Target ratio too far from the reference**: with reference ratio r and target scale T, "for a clean prediction, add runs at T/(2r) L or larger; to avoid refusal, T/(5r) L or larger (assuming the reference step holds)".

These lines only appear when the underlying condition actually applies, and a refusal page can show more than one of them at once.

## JSON readout

`?format=json` (or `format=html` to force the page; any other value is `400 invalid_format`), `Accept: application/json` (only when it outranks `text/html`; a tie, a wildcard, or no header keeps HTML), or the `/api/scale/readout.json` path return `200 application/json` built by `kalos.scale.readout_json.readout_to_json`.
Responses on `/api/scale/readout` carry `Vary: Accept`.
Every number is the engine's own; non-finite values are `null`, never a placeholder.

| field | meaning |
|---|---|
| `kind`, `schema_version` | `"scale_readout"`, `1` |
| `readout_id` | SHA-256 (64 hex) of the raw uploaded sheet, the canonicalized target (key order and whitespace do not matter), and the engine version. Deterministic, so a gate decision can cite it; nothing is stored |
| `decision`, `reasons[]` | `NUMBER`, `NUMBER_WITH_WARNING`, or `REFUSAL`, and why |
| `prediction`, `prediction_basis` | the model's number at the target (`"predicted"`), `null` on a refusal |
| `interval`, `interval_half_width` | `[lo, hi]` = prediction +/- the split-conformal half-width, `null` on a refusal |
| `interval_coverage`, `interval_method`, `interval_label` | nominal `1 - ALPHA` (0.9), `"split_conformal_ladder_residuals"`, and the honest wording: approximate, nominal, not calibrated at the target scale. `null` coverage/method on a refusal |
| `rungs[]`, `skipped_rungs[]`, `reference_ratio`, `requested_ratio`, `out_of_range_params[]`, `baseline_comparison`, `data_plan[]`, `target_inputs`, `target_column`, `target_column_unit`, `feature_set`, `missing_physics_inputs`, `physics_assumptions`, `provenance` | exactly the fields the HTML page renders |
| `scale_points[]` | one per measured row, ascending by scale: `{scale_L, actual, basis: "measured", oof_pred, oof_pred_basis}`. `oof_pred` is the ladder backtest prediction for that row (model fit only on strictly smaller scales; `oof_pred_basis: "predicted"`), `null` where the scale was never held out (the smallest two scales, skipped rungs) |
| `backtest` | `{method: "nested_leave_last_scale_out", method_description, per_scale[], skipped[], pooled}`: each rung's `n`, `mae`, `naive_mean_mae`, `naive_nn_mae`, `beats_both`, `too_few_to_judge`, `in_license_window`, `step_ratio`, plus `measured_mean` and `backtest_pred_mean`; `pooled` has the pooled MAEs, `beats_both`, `n_residuals`, `too_few_to_judge` (< `MIN_RESIDUALS`) |
| `divergence` | `{computed: true, rule, flags[{scale_L, codes}], per_scale[]}` or `{computed: false, reason, rule}` when no rung exists. A held-out scale is flagged `did_not_beat_baselines` (>= 3 runs and its backtest MAE does not beat both baselines) and/or `outside_backtest_band` (its measured mean is outside the mean backtest prediction +/- a split-conformal half-width built from the residuals of strictly smaller held-out scales only, with at least `MIN_RESIDUALS` of them; otherwise `outside_band: null` and `band_unavailable_reason`) |
| `max_trained_scale_L`, `warn_ratio_mult`, `refuse_ratio_mult`, `hard_cap_ratio` | the largest scale in the cleaned sheet and the engine constants (`WARN_RATIO_MULT`, `REFUSE_RATIO_MULT`, `HARD_CAP_RATIO`) |
| `extrapolation` | `{target_scale_L, max_measured_scale_L, factor, message}`: how far beyond the largest measured scale the target is, in plain words; `null` on a gate failure |
| `physics_range` | `{checked, decision: none\|warn\|refuse, reasons[], quantities[{quantity, label, unit, target, measured_min, measured_max, factor_outside, status: inside\|outside\|far_outside, refuses, setting_to_stay_inside}], refuse_factor, basis}`; `checked: false` with a `reason` in the scale_only feature set; `null` on a gate failure |
| `zones` | derived scale-axis zones, `null` without a reference ratio: `licensed_max_L` = M x warn x r, `warn_max_L` = M x refuse x r, both capped at `hard_cap_L` = M x hard cap; `basis: "derived"`. The process-parameter and pooled checks still apply to any target |

A gate failure is still a `422` with `{error, code, failed_check, detail}`, never a JSON readout.

## What it does not claim

It predicts a level at the target scale, not a ranking of recipes there.
Ranking at the target scale is unmeasured at current replication; see `docs/SCALE_EVIDENCE.md`.

## Demo

```bash
.venv/bin/python examples/synthetic_scaleup/make_synthetic.py   # regenerates the committed CSV
```

`examples/synthetic_scaleup/synthetic_scaleup.csv` is synthetic (simulated scale effects, 9 scales from 1 to 5000 L, 10 runs each).
With a 1.5x target (7500 L) it issues a prediction: the pooled ladder MAE is 0.263 vs 0.699 and 0.729 for the naive baselines over 70 residuals, and the reference step ratio is 3.33x.
