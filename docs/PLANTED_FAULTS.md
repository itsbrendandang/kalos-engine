# Planted-fault honesty suite

Kalos says it refuses, flags and holds instead of guessing. This suite measures
how often that actually happens. It plants known faults into clean bioprocess
sheets, runs the engine on them blind, and counts what was caught. It also counts
alarms raised on clean sheets.

- Code: `kalos/bench/faults/`
- Pre-registered protocol: `kalos/bench/faults/PROTOCOL.md` and `catalogue.py`.
  Both were committed on their own (`fb6a0a4`) before the first scored run.
- Recorded results: `kalos/bench/faults/results/`
- Tests and the CI gate: `tests/test_planted_faults.py` (marker `planted_faults`)
- Origin: CRUX_PORT_PLAN Rank 4 and REORG_PLAN F03. Clean-room: built from the
  plan only. No Crux code, fixture or text was used.

Every number below comes from **synthetic planted faults**: 36 faults on 3 sheets
that are synthetic or mostly synthetic. Treat the results as a regression gate and
a to-do list. They are not a catch rate for real client data.

## Running it

```bash
# fast tier: torch-free, about 5 s (parser, pre-fit refusals, validation gate, result entry)
python -m kalos.bench.faults all --tier fast --out /tmp/faults

# full tier: the real upload pipeline, the scale readout and result entry
# (about 45 min with 4 workers on a loaded 14-core machine)
python -m kalos.bench.faults all --tier full --workers 4 --out /tmp/faults \
    --datasets ~/kalos/datasets      # optional: adds the 3 Kaggle-based cases

# the steps one at a time
python -m kalos.bench.faults make  --out /tmp/faults    # blinded cases + truth key
python -m kalos.bench.faults run   --out /tmp/faults --tier full
python -m kalos.bench.faults score --out /tmp/faults --tier full   # scorecard_full.{json,md}
```

**CI.**
- The fast tier's regression gate is `test_fast_tier_regression_gate`. It is not
  marked `slow`, so the existing fast CI job runs it on every push.
  - It rebuilds the suite from `examples/` only, so the Kaggle cases are skipped.
  - It compares the result with `results/baseline_fast_gate.json`.
  - It fails if any fault's outcome drops (caught → noticed → missed), or if a
    clean case raises a new false alarm.
- `pytest -m planted_faults` runs every suite test. The one `slow` test pushes four
  small cases through the full tier and checks them against the recorded full
  scorecard.

**After an engine fix improves a case,** regenerate the gate baseline in the same PR:

```bash
python -m kalos.bench.faults all --tier fast --datasets /nonexistent --out /tmp/f
python -m kalos.bench.faults baseline --tier fast --out /tmp/f
```

Re-record the full scorecard in the same way, and say in the PR which cases moved.

## How it works

1. **Bases.** Deterministic clean frames:
   - `examples/synthetic_bioprocess` (160, 60 and 30 rows; and 155 rows plus 5
     held-out runs for result entry);
   - `examples/synthetic_scaleup` (the full sheet and ladders capped at 10, 300 and
     1,000 L);
   - optionally, a cleaned 150-row slice of the local Kaggle sheet.

   PROTOCOL.md section 2 has the details.
2. **Planting.** `plant.py` changes only the cells its fault names. A test checks
   this cell by cell.
3. **Blinding.**
   - Each case gets a folder `cases/case-NN/`. NN comes from a seeded permutation.
   - The folder holds `sheet.csv` with its rows shuffled, and `case.json` with the
     surface, file name and target. Result cases add `entries.json`; scale cases
     add `request.json`.
   - The truth key sits in `evaluator/truth.json`, outside the case folders.
   - The runner reads `cases/` only. A test checks that it never imports the
     catalogue or reads the truth key, and that no fault name appears in any case
     file.
   - Two builds give byte-identical output.
4. **Running.**
   - **Fast tier** (torch-free):
     - `parse_upload`, which includes the orientation pre-pass;
     - `_analyze(preflight_only=True)`;
     - `validate_frame`, called the way the analysis path calls it;
     - result entry through `CampaignStore.seed/start/set_result`, then validation
       of the folded sheet.
   - **Full tier:**
     - `pipeline.compute_upload_analysis`, the upload route's code;
     - the same result-entry path;
     - `kalos.scale.readout.build_readout`.
   - Engine output becomes flat signals: `{source, code, severity, column, rows}`.
5. **Scoring.** These are the PROTOCOL.md section 5 rules.
   - **caught**: a pre-registered matcher fires. That means the right refusal, or
     the right check on the right column, naming a planted row where the matcher
     requires one.
   - **noticed**: not caught, but some new warning appeared that the clean parent
     case did not raise, or a warning names a planted row.
   - **missed**: neither. A crash also counts as missed.
   - **False alarm**: on a clean case, any refusal or crash, or any catch detector
     firing at warning level or above. For result cases, only the new entries' rows
     are judged.
   - Rates come with Wilson 95 % intervals.

## Baseline scorecard (first scored run)

**Full tier.**
- Run at `bdd2d97` on `feat/planted-faults`. That branch carries no engine change
  over `feat/web-redesign-apis` @ `0fc05c3`; the one existing-module edit is a
  lazy import in `kalos/bench/__init__.py`.
- The Kaggle base was present.
- Library versions: torch 2.13.0, BoTorch 0.18.1, GPyTorch 1.15.2.
- Full table: `results/scorecard_full.md`.

| | Count | Wilson 95 % |
|---|---|---|
| Caught | **20 / 36** (56 %) | 0.40–0.70 |
| Caught or noticed | 24 / 36 (67 %) | 0.50–0.80 |
| Caught, excluding the declared blind spot | 20 / 35 (57 %) | 0.41–0.72 |
| Clean cases with a false alarm | **6 / 8** (75 %) | 0.41–0.93 |
| Crashes | 0 | |

By family (caught / noticed / missed):

| Family | Caught | Noticed | Missed |
|---|---|---|---|
| sheet (19) | 14 | 3 | 2 |
| result (4) | 1 | 0 | 3 |
| model (11) | 4 | 1 | 6 |
| scale (2) | 1 | 0 | 1 |

**Fast tier** (the same suite, torch-free surfaces only): 15 caught, 2 noticed and 6
missed of 23. False alarms on 5 of 7 clean cases. On the sheet surface the fast and
full tiers agree on every case except `too_few_rows_8`, which only the full tier
notices, through the reliability verdict.

### Caught (20)

`constant_target`, `too_few_rows_5` (refusals), `duplicated_rows_exact`,
`fahrenheit_rows_bare`, `fahrenheit_rows_labeled`, `mixed_units_labeled`,
`mixed_units_unlabeled`, `swapped_columns`, `transposed_sheet`,
`transposed_sheet_no_id`, `typo_x10_feature`, `typo_x1000_feature`,
`typo_x10_target`, `typo_x1000_target`, `result_negative`,
`leaked_column_named` (excluded as an outcome by its name), `pure_noise_target_s1..s3`,
`scale_far_extrapolation` (refused: ratio 500× > 20× hard cap).

Read some of these catches with care (see "What the numbers hide").

### Misses and near-misses mapped to owners

| Fault | Outcome | What happened | Module that should catch it | Planned item |
|---|---|---|---|---|
| `result_x1000` | missed | `set_result` accepted 1,000× the value. On the fold, the target outlier check fired, but its 20-row list (lowest row numbers first) never reaches the new rows | `kalos/portal/campaign.py` `set_result`; result referee | **Crux #1** result referee (`result_magnitude_jump`, F11) |
| `result_x10` | missed | same as above | same | **Crux #1** (F11) |
| `result_copy_paste` | missed | the same value was accepted for two runs of one round | same | **Crux #1** (`result_duplicate_entry`, F11) |
| `impossible_value` | missed | DO 140 % passed silently: `infer_dimension("DO_pct")` returns None, because `\bdo\b` does not match `DO_pct` (`_` is a word character), so the bundled sheet's DO column gets no bounds check at all | `kalos/validation/bounds.py` (`_DIMENSION_HINTS`) | **unplanned** (a one-line regex fix; add a planted test) |
| `duplicate_run_id` | missed | two different runs share one id and nothing is said | `kalos/validation/checks.py` (no duplicate-id check) | **unplanned** (DOMAIN_AUDIT LOW) |
| `duplicated_rows_new_id` | noticed | rows pasted under new ids are not duplicates to the check, because the id column is compared too. Only unrelated outlier warnings named the rows | `check_duplicate_rows` | **unplanned** (exclude identifier columns from the comparison) |
| `swapped_levels` | noticed (declared blind spot) | a consistent relabelling of medium in 30 runs is invisible on the sheet | as-run record / bridging controls | **Crux #1** `recipe_deviation` with as-run recipes (F11); DOMAIN H4 controls (unplanned) |
| `too_few_rows_8` | noticed | 8 rows and 11 inputs: CV ρ −0.79, untrusted, but a verdict is still issued and 5 proposals are returned | `kalos/portal/analysis.py` (reliability verdict) | **F04** (`not_judged` below about 12 rows) |
| `batch_offset_no_column` | missed | batch offsets with no input signal: CV 0.75 [0.49, 0.88], trusted | `kalos/portal/analysis.py` CV groups; `kalos/core/splits.py` | **F05** (round index, leave-one-round-out) + **F04** |
| `batch_offset_with_column` | missed | CV 0.93 [0.87, 0.96], trusted; `batch` fitted as a categorical and proposed (`'B4'`) | `analysis_parts/columns.py` (`_auto_categorical`, `_auto_group_column`) | **F05** (batch columns as CV-group candidates) + **F06** (never propose a batch level) |
| `jittered_replicates` | missed | 15 recipes × 4 jittered replicates, no input signal: CV 0.78 [0.61, 0.97], trusted, and no replicates detected | `kalos/core/splits.py`, `kalos/core/replicates.py` | **F05** (replicates grouped with a tolerance) |
| `leaked_column_renamed` | missed | a target tertile label named `qc_grade` was fitted (CV 0.79) and proposed (`'A'`) | `analysis_parts/columns.py` (`_auto_categorical`) | **F06** (derived role). A content-based leak check is **unplanned** |
| `outcome_as_factor` | noticed | Kaggle `contamination` was fitted and proposed (`'Y'`). Only a low trust score was raised, which says nothing about the column | `analysis_parts/columns.py` | **F06** (response role) |
| `id_like_numeric_categorical` | missed | numeric clone ids 101/204/377 were treated as continuous; proposal `clone = 325.298` | `analysis_parts/columns.py` (numeric defaults to continuous) | **strain-engine P1** (`roles.strain`) + **F06** (id role) |
| `covariate_as_setpoint` | missed | `passage` was fitted and proposed as a setpoint (`5.0`) | `analysis_parts/columns.py` (no covariate role) | **strain-engine P1** (`roles.covariates`) + **F06** |
| `scale_physics_extrapolation` | missed | 2,000 L at 240 rpm from a ladder that stops at 300 L: `NUMBER` 5.58 [5.09, 6.07] g/L with no warning. The requested ratio 6.67 is exactly 2× the reference 3.33, and the warning needs `>`. Derived P/V and tip speed are not range-checked (DOMAIN H6) | `kalos/scale/readout.py` (`decide`) | **F10** scale physics guardrails |

Note on the strain-engine rows: `feat/strain-clones` fixes clone and passage when
the roles are **declared**. This suite uploads with no declared roles, which is the
default path. The inference side therefore still needs F06's auto-flag findings for
id-like codes and passage/generation columns.

F07 (provenance) has no case in v1. Its fault, Kaggle `data_type`, has no
existing engine signal to match yet. It is a candidate for protocol v2.

### False alarms on clean sheets (6 of 8)

| Clean case | Why it counts | Comment |
|---|---|---|
| `clean_bioprocess160`, `clean_bioprocess60`, `clean_scaleup90` | `outliers` warnings: on the 160-row sheet the target outlier check flags **60 of 160** rows; `antifoam_mgL` adds 2 more | The zero-inflated titer (44 % zeros) gives a linear-scale MAD near 0, so most producers look like outliers. This is the noisiest detector in the engine |
| `clean_bioprocess30` | target outliers, and trust refused (CV ρ −0.32) | At 30 rows and 11 inputs, refusing trust is defensible, but the rule counts it |
| `clean_kaggle150` | `physical_bounds` warning for pH below 4 on 5 rows, plus target outliers | The Kaggle synthetic rows really do go down to pH 2.6 (DOMAIN_AUDIT LOW), so the pH flag is arguably true. Also: this sheet is **trusted with a CV CI that includes 0** (0.27 [−0.06, 0.38]), the F04 pattern |
| `clean_scale_in_ladder` | readout refused: "interpolation is out of scope for this readout" | A protocol design flaw, not an engine fault: the clean case asked for the largest trained scale (ratio 1.0), which the readout refuses by design. It still counts under v1. Protocol v2 should use a licensed small extrapolation |

Result entry raised no false alarm. Five true results and one surprising but
plausible new best (1.10× the best measured) were all accepted, and none was
flagged on its fold row.

## What the numbers hide

- **The target-typo catches lean on a noisy detector.**
  - `typo_x10_target` and `typo_x1000_target` count as caught. The outlier finding
    lists one of the planted rows among its first 20 rows.
  - But the same finding flags 60 of 160 rows on the clean sheet. Whether a planted
    row is listed depends on where the shuffle put it.
  - The same 20-row cap hides result-entry rows entirely, because they are appended
    at the end of the fold. That is why `result_x10` and `result_x1000` are misses.
  - The scorecard flags these cases: "right reason fired but its 20-row list omits
    the planted rows".
  - The fix belongs in `check_outliers`: judge a zero-inflated target on its
    producers or on a log scale, and list the most extreme rows first.
- **The transposed sheet is "caught" but damaged.** The orientation pre-pass
  detects it and re-orients it. While doing so it silently drops the text row
  (`medium_base`) and the run-id row, so the analysis runs on 10 of 11 inputs with
  no ids, and no finding says so (`kalos/normalize/orientation.py`; unplanned).
- **The pure-noise catches do not discriminate.** All three shuffled-target sheets
  (n = 30) are refused trust. So is the clean 30-row parent (CV ρ −0.32). At n = 30
  this suite cannot tell noise from signal. ML_AUDIT f1 is the better estimate here:
  15.8 % false trust over 400 noise sheets.
- **Some catches only hold at the point of re-analysis.** `result_negative` is
  caught by a `physical_bounds` error on the folded sheet. The value was still
  accepted at entry and would still be folded in `warn` mode. Only the referee
  (Crux #1) would hold it.
- **One seed per fault.** The intervals are wide on purpose (Wilson 95 %, n = 36).
  A single case moving is not a trend.

## Changing the suite

- The protocol is versioned. Adding a fault, changing a matcher or fixing the
  clean-scale case means:
  - a dated amendment in PROTOCOL.md;
  - `PROTOCOL_VERSION = "2"`;
  - a new scored run.

  Never edit v1 results.
- An engine fix that makes a planned code real (for example the referee's
  `result_magnitude_jump`) scores without a protocol change. Those codes are listed
  in `catalogue.py` with `planned=True`.
