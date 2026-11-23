# Planted-fault honesty suite: pre-registered protocol (v1)

Committed on 2026-11-19, in a commit of its own, **before the first scored run**.
The machine-readable half is `catalogue.py` (same commit). Nothing in either file
may change after the first scored run except through a dated entry under
"Amendments" at the end of this file.

Clean-room: designed from `~/.kalos-dev/CRUX_PORT_PLAN.md` (Rank 4),
`REORG_PLAN.md` (F03), `ML_AUDIT.md`, `DOMAIN_AUDIT.md` and the Kalos code. No
Crux code, fixture or text was opened or copied.

## 1. Question

How often does the Kalos engine's honesty layer (refusals, the validation gate,
the reliability verdict, column-role inference, the scale readout's decision,
and result entry) **catch** a planted fault, and how often does it raise an
alarm on clean data? The suite measures the engine as it is. It does not fix it.

## 2. Inputs (clean bases)

All bases are built deterministically from files that ship with the repo, plus one
optional local file. Seed: `SUITE_SEED = 20261119`.

| Base | Source | Preparation |
|---|---|---|
| `bioprocess160` | `examples/synthetic_bioprocess/synthetic_bioprocess.csv` | all 160 rows, unchanged |
| `bioprocess60` | same | 60 rows sampled without replacement (seeded) |
| `bioprocess30` | same | 30 rows sampled without replacement (seeded, independent of the 60) |
| `results155` | same | 155 rows as the uploaded sheet; 5 held-out producer rows (titer > 0.3 g/L) become started runs whose true titers are the clean results |
| `scaleup90` | `examples/synthetic_scaleup/synthetic_scaleup.csv` | all 90 rows |
| `scaleup_le300`, `scaleup_le10`, `scaleup_le1000` | same | rows with `scale_L` at or below 300 / 10 / 1,000 L |
| `kaggle150` (optional) | `~/kalos/datasets/kaggle_fermentation_dataset_0.csv` (not in the repo) | duplicate run ids dropped (first kept); columns `min_do_pct` (a response), `total_media_vol_ml` (derived), `contamination` (an outcome), `yield_tier` (target-derived) and `data_type` (provenance) removed; all 19 `Real` rows plus seeded `Synthetic` rows up to 150 |

Cases whose base is unavailable (Kaggle in CI) are reported as **skipped**, never
as caught or missed.

## 3. Cases and blinding

- One case per catalogue entry: 36 faults and 8 clean cases (`catalogue.py`).
- Case folders are `cases/case-NN/`. NN is a seeded random permutation, so the
  number says nothing about the fault. Every folder holds only what a user would
  supply: `sheet.csv` (row order shuffled with a seed) and `case.json` (surface,
  file name, target column), plus `entries.json` (result cases) or `request.json`
  (scale cases). No fault name, planted row or expected outcome appears in a case
  folder.
- The truth key is `evaluator/truth.json`, outside `cases/`. It lists, per case,
  the fault id (or `clean`), the parent clean case, the planted columns and the
  planted rows in the case's own (shuffled) row order.
- The runner reads only `cases/` and writes `observations/`. The scorer joins
  observations with the truth key. The runner never opens `evaluator/`.
- Generation is deterministic: two builds with the same seed are byte-identical.

## 4. What the engine is asked

| Surface | Fast tier (torch-free, CI) | Full tier (slow) |
|---|---|---|
| `sheet` | `parse_upload` (orientation pre-pass) + `_analyze(preflight_only=True)` (pre-fit refusals) + `validate_frame` with the analysis path's arguments | `pipeline.compute_upload_analysis` (the upload route's code path), default settings (`KALOS_VALIDATION_MODE` unset = `warn`, no declared roles) |
| `result` | `CampaignStore.seed` with the sheet, `start` one run per entry, `set_result` per entry, then `validate_frame` on `plan_fold`'s folded sheet (what a re-analysis validates) | same as fast |
| `scale` | not run | `kalos.scale.readout.build_readout` with `request.json` |

The case's target column is passed explicitly, as a user picking the target would.

## 5. Scoring rule

Per fault case (matchers in `catalogue.py`):

- **caught**: a `caught_if` matcher matches a signal: the right refusal, or a
  finding with the right reason on the right column (and, where the matcher says
  so, naming at least one planted row).
- **noticed**: not caught, but (a) a `noticed_if` matcher matches, or (b) a
  warning-or-worse signal appears whose `(source, code, column)` the clean parent
  case did not raise, or (c) a warning-or-worse signal names a planted row.
- **missed**: otherwise. A crash (an exception other than a refusal) is missed.
- `harm_if` matchers are reported as evidence and do not change the score.

Per clean case:

- **false alarm** if the engine refuses or crashes, or if any catch detector
  (every non-planned `caught_if` matcher with its column/row constraint removed,
  at warning or worse) fires. For result cases only the entries' fold rows and
  entry-time signals are judged. Warnings that are not catch detectors (for
  example "no date column") are listed as standing warnings, not counted.

Rates are reported as counts with Wilson 95 % intervals: caught / scored faults,
caught-or-noticed / scored faults, and the same over faults with
`detectable_from_input = true`; false-alarm cases / clean cases, and per detector.
Results are labelled **synthetic planted faults** and say nothing about real
sheets beyond that.

## 6. Expected outcomes (normative)

The `expected` text and `caught_if` matchers of each entry in `catalogue.py` are
the expected outcomes: what an honest engine should do, not a prediction of what
this one does. Misses are expected and are the point. Each fault names the module
that should catch it and the planned item that covers it (Crux #1 result referee
/ F11, F04, F05, F06, F07, F10, strain-engine P1, or "unplanned").

## 7. Runs

- **Baseline (first scored run):** the full tier at the tip of `feat/planted-faults`
  (which carries no engine change over `feat/web-redesign-apis`), with the Kaggle
  base present. Written to `kalos/bench/faults/results/` as JSON and markdown,
  with the commit, seed, tier and library versions.
- **CI regression gate:** the fast tier on the repo-only bases. Against the stored
  fast baseline, no fault's outcome may drop (caught > noticed > missed), and no
  clean case may gain a false-alarm detector. An improvement is allowed; it is
  recorded by regenerating the baseline in the same PR as the engine fix.

## 8. What this suite does not claim

- It covers 36 synthetic faults on 3 synthetic or mostly synthetic sheets. A catch
  rate here is not a catch rate on real client data.
- One seed per fault (three for the pure-noise target). Rates are case counts,
  not probabilities; the intervals are wide on purpose.
- Model faults are scored on one analysis each. ML_AUDIT's multi-seed rates
  (15.8 % false trust on noise, 95-100 % under batch effects) are the better
  estimate of those failure rates; this suite only checks that they still occur.

## Amendments

None.
