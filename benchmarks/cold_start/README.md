# Cold-start design benchmark

Which starting design, and how many starting runs, leave a campaign best off?
The results pick the defaults of `POST /api/campaigns/{id}/design`
(`kalos/portal/design_plan.py`: `DEFAULT_METHOD`, `DEFAULT_N`) and are summarised in
docs/COLD_START_DESIGN.md.

## Protocol

The tasks, arms, budget, seeds and scores were fixed in `run_benchmark.py` before the
full run; the decision rule below was written while the full run was in progress,
after only a 2-seed smoke run had been seen.

- **Budget.** N = 32 runs in total. A starting round of n0 in {8, 12, 20} runs from
  the design, then model-guided rounds of q = 4 until N (6, 5 and 3 rounds).
- **Arms.** `sobol` (scrambled Sobol), `lhs_maximin` (Latin hypercube, best of 32
  starts plus swaps that raise the smallest pairwise distance), `random` (i.i.d.
  uniform, categoricals i.i.d.), `lhs` (plain Latin hypercube: BayOp-style without
  any refinement) and `lhs_cd` (Latin hypercube refined for centered discrepancy,
  scipy `optimization="random-cd"`: BayOp's actual default). Every arm except
  `random` runs through `kalos.core.design.generate_design`, so it snaps to the
  declared steps, rejects constraint violators and balances categorical levels.
- **Model-guided rounds.** Synthetic tasks: the production path, `Surrogate().fit`
  then `kalos.core.optimize.propose` (q = 4, seeded), identical for every arm.
  Pool replays: the same surrogate scores every unrun row of the sheet by expected
  improvement and runs the top 4; the starting design is mapped to the nearest
  unused rows (standardized inputs).
- **Tasks.**
  - `hartmann6`: Hartmann-6 on [0,1]^6, maximized, noise sd 0.1.
  - `bioprocess_sheet_sim`: the generator of `examples/synthetic_bioprocess`
    (10 settings, zero-inflated, 12% multiplicative assay noise), with declared
    setpoint steps (pH 0.05, temperature 0.5, glucose 0.5, glutamine 0.5, feed 0.1,
    DO 5, rpm 10, inoculum 0.1, antifoam 5, osmolality 5).
  - `bioprocess_like_mixed`: `kalos.bench.objectives.bioprocess_like` (3 settings
    plus a 4-level strain and a 3-level inducer), noise 15% of its scale.
  - `gaussian4d_constrained`: a 4-D Gaussian bump under x0 + x1 <= 1 (the
    unconstrained peak is outside; the constrained optimum 9.2312 is exact).
  - `pool_synthetic_bioprocess`, `pool_synthetic_scaleup`: replays on the example
    sheets in `examples/` (synthetic, committed).
  - `pool_kaggle_fermentation`: replay on the Kaggle fermentation sheet
    (semi-synthetic: 696 of 715 rows are flagged synthetic by the uploader; listed
    as Apache-2.0 in the uploader's metadata, the rights to the underlying runs are
    not verified; the file is never committed, see examples/datasets/README.md).
    Runs only when `KALOS_DATASET_DIR` points at it. Target `max_yield_fpu_ml`;
    the target-derived columns `yield_tier`, `contamination` and `data_type` are
    dropped.
- **Scores.** Normalized simple regret at N: (best possible - best found) /
  (best possible - median), on the TRUE (noiseless) value for synthetic tasks and on
  the measured value for pool replays; lower is better. Trust: the engine's own
  verdict (`kalos.portal.verdict.model_verdict` on `grouped_cv_report`, 5 folds,
  2 repeats, floor 0.20, recipe groups) after the starting round and after every
  model-guided round; "trust reached" means the verdict `reliable`, which by
  construction needs at least 20 runs and 8 groups.
- **Statistics.** Seeds 1000-1019 (20) per cell. Means with 95% percentile
  bootstrap CIs over seeds (10,000 resamples); paired differences against `sobol`
  on the same seeds. "Pooled" averages each seed's regret over the tasks (each task
  weighted equally).
- **Decision rule for the default.** The default method is the arm with the
  lowest pooled final regret unless another arm's paired CI shows it is worse on no
  task and better on at least one; ties (overlapping CIs) go to the simpler
  `sobol`. The default size is the smallest n0 whose pooled regret is not worse
  (overlapping CI) than the best n0, with its trust-reached rate reported.

## Run

```bash
KALOS_DATASET_DIR=/path/to/folder/with/kaggle_fermentation_dataset_0.csv \
KALOS_TORCH_THREADS=1 .venv/bin/python benchmarks/cold_start/run_benchmark.py --workers 10
.venv/bin/python benchmarks/cold_start/augment_study.py   # gap filling next to existing runs
```

Outputs: `results/RESULTS.md`, `results/summary.json`, `results/units.json` (one
record per task, arm, n0 and seed), `results/augment_study.{json,md}`.
