# Open-dataset replay (retrospective, not validation)

This is a retrospective replay of the portal loop on published open datasets.
It asks two questions and nothing more:

1. **Does the system run end to end on real data?** `examples/open_replay_e2e.py`
   drives the portal's HTTP API (upload, start, result, reanalyze) for several rounds
   per dataset and logs every step.
2. **In a pool replay, does BO reach 90% of the pool max in fewer experiments than
   random or space-filling selection?** `examples/open_pool_replay.py`.

The replay is **not validation**. A "lab" reveals the value of the nearest unused
measured row, the pools are small or enriched by earlier campaigns, and no claim
about compliance, regulatory fitness, or a specific number of experiments saved
follows from it.

## Data is not in this repository

No dataset file is committed. Loaders read files from folders named by environment
variables. Each dataset's licence is set by its publisher. Check the DOI page before
redistributing anything derived from it.

| Set | Env var | Source (DOI) | Licence string (as reported by the host) |
| --- | --- | --- | --- |
| Pichia CoCaBO, HSA / HerceptinA / RBDJ (3 files) | `KALOS_OPEN_DATA` | Zenodo 15466161, doi [10.5281/zenodo.15466161](https://doi.org/10.5281/zenodo.15466161) (GitHub NHarini-1995/CellCultureBayesianOptimization) | Zenodo: CC BY 4.0. The GitHub repo has no licence file; the Zenodo record governs the archived copy |
| Trichoderma sp. ambr250 (Tamminen 2022) | `KALOS_OPEN_DATA` | Zenodo 5846028, doi [10.5281/zenodo.5846028](https://doi.org/10.5281/zenodo.5846028) | CC BY 4.0 |
| Candida utilis RSM Nr1 | `KALOS_OPEN_DATA` | Zenodo 22277916, doi [10.5281/zenodo.22277916](https://doi.org/10.5281/zenodo.22277916) | CC BY 4.0 |
| Amii DBO / MOBO rat myoblast, HeLa regular / time-saving, T-cell, TF-1 | `KALOS_AMII_DATA` | GitHub Amii-Applied-AI/amii-cell-ag-tools `active-learning-for-cell-media/Data`; source articles doi [10.1002/elsc.202300005](https://doi.org/10.1002/elsc.202300005), [10.1038/s41540-023-00284-7](https://doi.org/10.1038/s41540-023-00284-7), [10.1038/s42003-019-0296-7](https://doi.org/10.1038/s42003-019-0296-7) | Repo root MIT; the data folder's `Data/SOURCES.md` says the datasets keep the original papers' terms (source articles CC BY 4.0 per Crossref). See `kalos/bench/amii.py` |

Not used (see `/tmp` research notes, not in this repo): the erythromycin batch set
(zenodo 14619074) has no recipe inputs that a proposal can set, and its target is
ambiguous (`hx` in the loader vs "chemical potency" in the record). The
Bacillus OFAT, ART, and Kaggle sets are not media designs.

### Folder layout expected

`KALOS_OPEN_DATA` points to a folder that contains the following, found by file name
anywhere underneath it:

- `*_CoCaBO_Prod_AllData.csv` (`HSA`, `HerceptinA`, `RBDJ`), from the `AllMolecule` folder
  of the Pichia repository
- `*Tamminen2022_ambr*.xlsx` (the ambr250 workbook)
- `*Cutilis*.xlsx` (the Candida workbook; sheet `6.  RSM Nr1`)

`KALOS_AMII_DATA` points to the folder holding `DBO_dataset_rat_myocyte.csv`,
`MOBO_dataset_rat_myocyte.csv`, `df_Human_Hela_regular_mode.csv`,
`df_Human_Hela_timesaving_mode.csv`, `df_Human_T_Cell_Expanded.csv`, and
`df_Human_TF_Cell_Expanded.csv`.

## What each loader does

`kalos/bench/open_datasets.py` returns a run sheet (inputs plus target) and a
`roles` object for the upload.

- **Pichia**: `CarbonSource` (an integer code 0-18) becomes a text level and is declared
  categorical. `Carbon_conc`, `Gly`, `Met`, `SP` are inputs. `Titer` is the target.
  `OD_OG` and `OD_Prod` are measured outputs and are dropped. The index column is dropped.
- **Trichoderma ambr250**: one row per culture sheet. The endpoint is the last offline
  `Mucic acid (g/L)` value. Two variants: `trichoderma_ambr` (strain and condition as two
  categoricals) and `trichoderma_ambr_condition` (condition only, with strain declared an id).
  YE, glucose, and pH are not parsed, because the baseline medium of the pH-only and
  YE/glucose-only sheets is not stated in the workbook.
- **Candida RSM Nr1**: molasses and ammonium are inputs, `biomass_g_L` is the target. The
  run order and protein responses are dropped. Five centre points give the replicate
  noise estimate.
- **Amii**: the same feature and target choices as `kalos/bench/amii.py` (`REGISTRY`), but
  one row per measurement (no averaging). DBO's `IS_indicator` (information source, four
  values) is not an input, so the sources are mixed and the result is unresolved. MOBO uses
  the highest fidelity only.

## Reproduce

```bash
export KALOS_OPEN_DATA=/path/to/open          # Pichia, Trichoderma, Candida files
export KALOS_AMII_DATA=/path/to/amii-Data     # the six Amii CSVs
export KALOS_STATE_DIR=/tmp/kalos-replay      # never point a replay at a real state dir

# 1. portal (loopback, open mode, no tokens)
KALOS_PORT=8061 python -m kalos.portal &

# 2. end-to-end HTTP run: seed upload, then 3 rounds of start / measure / reanalyze
python examples/open_replay_e2e.py --base http://127.0.0.1:8061 --dataset all --rounds 3 \
    --out /tmp/kalos-replay/e2e --full-upload-probe

# 3. pool replay (BO vs random vs LHS, paired seeds, 90%-of-max metric)
python examples/open_pool_replay.py --dataset all --seeds 20 --batch 4 --out /tmp/kalos-replay/pool

# 4. the existing Amii pool replay, unchanged
python examples/amii_benchmark.py --data "$KALOS_AMII_DATA" --seeds 20 --budget 60
```

Tests use tiny fake files written to a temporary folder and never read real data:
`python -m pytest tests/test_open_datasets.py`.

## Method notes

**End-to-end driver** (`examples/open_replay_e2e.py`). A seeded random subset of each
dataset (8 rows for the 20-row Trichoderma and 6 for the 13-row Candida, 24 otherwise) is
uploaded as the client's existing data. The rest is the held-out "lab". Each round starts
the proposed batch, reveals the held-out value of each recipe, records it, and calls
reanalyze. A proposal rarely sits on a measured grid. The revealed value is therefore the
closest **unused** measured row (categorical levels must match exactly, continuous
features are compared in min-max normalised space). The match distance is recorded. The
driver stops a dataset when nothing can be matched. `--full-upload-probe` also uploads the
whole sheet once, to test the fit-size limit.

**Pool replay** (`examples/open_pool_replay.py`). It reuses `kalos.bench.amii.run_trial`,
where every arm (BO, random, LHS-style nearest neighbour) starts from the same seeded
initial rows. `run_pool` in `kalos/bench/pool.py` is not used: it returns only aggregated
curves, and it has no LHS arm. Categorical columns are one-hot encoded. The metric is the
first evaluation at which best-so-far reaches 90% of the pool max (0.9 x max when all values
are non-negative, otherwise min + 0.9 x range). Seeds that never get there are
right-censored. The report gives the paired BO-minus-random mean with a bootstrap 95% CI
over seeds and an exact paired sign test. With pools of 13 to 20 rows, no result is
evidence.

## Known limits

- Proposals are matched to the nearest measured row, not to a new measurement. Pool
  results describe the optimizer's ranking of existing rows, not a new experiment.
- Torch and BoTorch versions change the GP fits. The lock file pins torch 2.13.0 and
  botorch 0.18.1 for the runs reported here.
- The Amii pool replay uses replicate-averaged rows and the paper's feature choices. It is
  not the paper's surrogate-landscape protocol (see `kalos/bench/amii.py`).
