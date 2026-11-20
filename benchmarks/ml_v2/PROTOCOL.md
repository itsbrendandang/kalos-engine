# ML benchmark v2: pre-registered protocol (ml_v2.0)

Committed on 2026-11-20 in a commit of its own, **before any candidate arm is run**
and before the reference arm's tuning results. The machine-readable half is
`kalos/bench/ml_v2/protocol.py` (same commit): every seed, size, budget, level and
threshold named here is a constant there. Nothing in either file may change after
this commit except through a dated entry under "Amendments" at the end, which names
the constant, the reason, and whether any result had been seen.

Runs before this commit, disclosed: harness smoke tests on pilot/smoke seed 9002
(random, LHS and XGBoost arms, all suites, for debugging and timing), and two
reference-arm (`kalos_ref`) debugging jobs, closed-loop `mixed` seed 1 and snapshot
`null_batch_rounds` seed 1 (tuning seeds). No candidate arm has been run.

## 1. Question and rules

Which changes to the Kalos ML (surrogate, acquisition, trust layer) make it better,
measured so that an improvement cannot be tuned on the test? A change **ships** only
if it meets every criterion of section 8 on **held-out** seeds and datasets, against
the reference arm (current Kalos, `main` at the commit a held-out run names).

- Candidates tune on TUNING seeds and TUNING datasets only, as often as they like.
- A candidate is FROZEN before its held-out run (`python -m kalos.bench.ml_v2 freeze`,
  commit `benchmarks/ml_v2/freeze/<arm>.json`). The harness refuses a held-out run
  without a committed freeze file whose spec and configuration hash match, or with
  uncommitted changes to tracked files. Each held-out run is appended to
  `benchmarks/ml_v2/results/heldout_ledger.jsonl`; every held-out run is reported,
  including failures. A frozen arm is never re-frozen; a changed arm is a new arm.
- At most **3 frozen candidates per agent** reach a held-out run (selection on
  held-out results is otherwise unbounded).
- Predicted and measured values are never mixed: regret and coverage are judged on
  measurements (simulated with fresh noise, or recorded rows), never on the arm's
  own predictions.

## 2. Arms

| Arm | Spec | Role |
|---|---|---|
| `kalos_ref` | `kalos.bench.ml_v2.arms.kalos_ref:make_arm` | Reference: the production `kalos.portal.analysis._analyze`, as a campaign re-analysis calls it |
| `random` | `kalos.bench.ml_v2.arms.baselines:make_random` | Baseline (pool replay: uniform over not-yet-run rows) |
| `lhs` | `kalos.bench.ml_v2.arms.baselines:make_lhs` | Baseline: one Latin hypercube of the whole budget per instance |
| `rf_conformal`, `xgb_conformal` | `kalos.bench.ml_v2.arms.trees:make_rf` / `:make_xgb` | Tree baselines, section 6.3 |
| `rf_spread`, `xgb_spread` | same, `--arm-config '{"interval": "spread"}'` | Tree baselines with seed-ensemble spread bands |
| candidates | new modules under `kalos/bench/ml_v2/arms/` | section 6 |
| external | BayBE, Ax default, GP+UCB as in voyager-brain | Standalone baselines (ml-acquire), never shippable |

## 3. Tasks and data

### 3.1 Synthetic families (randomized instances)

One seed draws one INSTANCE: dimension, optimum location, rotation, noise and
categorical effects are random, so a result is an average over many surfaces
(`kalos/bench/ml_v2/problems.py`). Every instance knows f* (exact or multi-start
local optimization, recorded in `meta.f_star_method`) and f_med (median of the truth
over 4,096 uniform feasible points).

| Family | Randomized per instance |
|---|---|
| `hartmann6` | Hartmann-6 embedded in d = 6 or 10 (inactive dims), random axis permutation and reflections, noise 2/10/20 % of (f* - f_med) |
| `bumps` | mixture of 1-3 Gaussian bumps, d = 2/4/6/10, random centers, widths and **rotations** (anisotropic), noise 2/10/20 % |
| `bioprocess` | the `examples/synthetic_bioprocess` generator (10 inputs, zero-inflated viability region) with shifted pH/temperature/DO/glutamine/feed optima, multiplicative noise CV 6/12/20 % |
| `heteroscedastic` | bump, d = 2/3/5, random center and sharpness, noise sd = 0.05 + b f, b = 0.1/0.2/0.3 |
| `mixed` | 3 continuous + strain (3-5 levels) + inducer (3 levels); per-strain offset, peak and width random; declared categoricals |
| `constrained` | bump mixture d = 3/5 with one linear constraint over 2-3 inputs that cuts off the unconstrained optimum (passed as a campaign setup) |
| `clones` | 3 clones with numeric-looking IDs, per-clone offsets and clone x process optimum shifts, a passage covariate with a small effect; historical runs use every clone, proposals pinned to one (declared roles: strain, fixed policy, covariate) |
| `batch` | bump mixture d = 3/6, every round adds its own offset (sd 10/30 % of the range); the round index reaches the arm |
| null families (snapshot only) | `null_noise` (d = 3/6), `null_batch_column` (clustered batches, offsets, text batch column), `null_batch_rounds` (same, round index), `null_batch_hidden` (same, no column), `null_jitter` (replicates recorded with 0.5 % input jitter) |

### 3.2 Real datasets (pool replay and real snapshots)

Licences, provenance and the datasets considered and rejected are in
`benchmarks/ml_v2/DATASETS.md`. No data file is committed.

| Dataset | Group | Split | Notes |
|---|---|---|---|
| Amii `DBO_rat_myocyte`, `HeLa_timesaving`, `Tcell_expansion` | real | tuning | published media datasets of VALIDATION.md |
| Amii `MOBO_rat_myocyte`, `HeLa_regular`, `TF1_expansion` | real | **held-out only** | same |
| Zenodo 18270843 inclusion-body refolding (DoE1 + BO rounds, 5 inputs) | real | **held-out only** | CC-BY-4.0; small (snapshot and emulator; pool when >= 47 recipes) |
| Olympus `hplc` (6-D process), `lnp3` (formulation, mixed) | ood (out of domain) | `hplc` tuning, `lnp3` held-out only | MIT; not bioprocess |
| `fixture/synthetic_bioprocess`, `fixture/synthetic_scaleup` | fixture | both | generated; repo files |
| Kaggle fermentation | info_only | both | mostly machine-generated, unclear licence: **information only**, never pooled, never decides |
| user folder (`--user-data DIR`) | user | both | every `*.csv` plus optional `<stem>.json` sidecar; report-only until the user decides |

### 3.3 Emulators (Olympus-style)

Pool replay can only run recipes that exist in the pool. For every real and ood
dataset two emulators are fitted on all rows (`emulator.py`): a scikit-learn GP
(Matern-5/2 ARD + white noise, <= 1,000 rows) and a random forest (300 trees), both
with the GP's fitted noise sd. Two kinds so that no surrogate family is favoured by
construction. f* = max of the emulator mean over 20,000 box points and the dataset
rows (approximate; a negative regret is reported as is). **Caveat stated with every
emulator result:** an emulator is a model of the data, not the lab; far from the rows
it reverts to its prior (GP) or nearest leaf (RF), and in 15-29 dimensions most of the
box is far from every row. Emulator results are reported in their own groups
(`real_emulator`, `ood_emulator`) and never pooled with pool replay.

### 3.4 Internal data (rights unconfirmed): report-only, held-out only

Read only through `KALOS_EXTERNAL_DATA` or a path argument; never committed, nor any
row-level derived file (`benchmarks/ml_v2/.gitignore`). Every result is labelled
**"internal, data rights unconfirmed"**, carries aggregate metrics only, and decides
no criterion of section 8.

- **LeadGene/Cytena CHO clones** (`leadgene_cho_dynamics.csv`, Voyager repo;
  transposed, 504 clones in series S1 96, S2 384, S3 24; 384w -> 96w -> 24w -> SF125).
  A series is a campaign. Only clones the lab ranked highest advance, so every
  larger-scale outcome is for a selected subset (range restriction).
  - `rank_96to24` (clone ranking): 384w confluence, 96w confluence and titer signal
    predict the 24w titer signal; Spearman, top-8 precision, AUC for "top third at the
    next scale", interval coverage and width on held-out clones.
  - `transfer_loso` (scale transfer): within-series percentiles of the smaller-scale
    readouts predict the next-scale percentile; leave-one-transition-out
    (384w->96w, 96w->24w, 24w->SF125).
  - `flask_small` (384w..24w -> SF125 mg/L, 8-11 clones): too few for a model; an
    honest arm refuses or reports not trusted.
  - **Mandatory baseline:** the previous-scale readout (carry-forward rank).
    **Splits grouped by campaign:** tuning uses S1 and S2 only (train on one, test on
    the other, plus seeded 4-fold clone splits within them); held-out is
    leave-one-campaign-out over S1, S2, S3. **Coverage under campaign shift** (the
    90/95 % coverage of the arm's intervals on the held-out campaign) is a trust
    metric. Known evidence before this protocol (voyager-port, internal): previous-scale
    rank rho = 0.53 [0.31, 0.70], Kalos GP rho = 0.30, GP 90 % coverage 0.79 under
    leave-one-campaign-out.
- **Anagram / Bioqore Pichia lipase** (`anagram_compiled.csv` S26JAN06 bench rows;
  `Combined_S26APR07_and_S26APR29_...tsv`). Customer/partner data. The normative
  definitions for these are `~/.kalos-dev/REAL_BACKTEST_PLAN.md` (frozen by agent
  real-backtest) for the backtest; zeros and 0.0005 g/L are censored non-detects; the
  APR07 8-hour plate is excluded (harvest time is not recorded); S26APR07 vs S26APR29 is
  an experiment group. The 5 large-scale fed-batch rows (3 L to 35 m3) are the
  `internal/pichia_2L_to_35m3` scale task, where refusal is the expected answer.
  The Bioqore `mab_scaleup_dataset.csv` is GENERATED (its own README says so) and is
  not used as evidence.

### 3.5 Low-data regimes

Closed loop, emulator and pool instances start from n0 in {12, 20, 40} (seed % 3)
and run 5 rounds of q = 5 (n0 + 25 runs). Snapshots use n in {12, 20, 40, 80}
(seed % 4). Results are reported per regime and pooled.

## 4. Seeds and splits

| Suite | Tuning seeds | Held-out seeds |
|---|---|---|
| closed_loop | 0-11 | 1000-1029 |
| pool | 0-5 | 1000-1015 |
| emulator | 0-3 | 1000-1009 |
| snapshot | 0-11 | 1000-1039 |
| snapshot_real | 0-5 | 1000-1015 |
| backtest | 0-5 (seeded orderings; real orders run once) | 1000-1029 |
| scale, internal | 0 (internal: 0-2 clone splits) | 1000 |
| ci (scoreboard drift only) | closed_loop 5001, 5004 on `heteroscedastic`, `bumps` | never decides |

Tuning runs use TUNING datasets only (3.2). A held-out run uses every dataset
(tuning and held-out-only) with held-out seeds; results are reported separately for
the two dataset sets. Internal data are held-out only (except LeadGene S1/S2 as
above). Seeds feed `numpy.random.default_rng([seed, task, stream])`: the initial
design, the measurement noise and the held-out points are separate streams, so all
arms see the same initial design and the same noise draws for the same index.

## 5. Suites (one job = suite x task x seed x arm)

1. **closed_loop / emulator**: LHS initial design of n0 (all clones for `clones`),
   then 5 rounds: the arm proposes q = 5; proposals are clipped to the box (constraint
   violations are counted, not repaired); a short or withheld batch is filled with
   uniform feasible points (counted as `n_filled`); each run is measured once with
   fresh noise (round offsets for `batch`, random passage for `clones`). The campaign
   round index of every row is passed (the engine's fold history carries it).
2. **pool**: n0 random rows, 5 rounds; each proposal is snapped to the nearest
   not-yet-run row (unit-scaled continuous distance + 1 per categorical mismatch) and
   reveals its recorded value. Coverage at proposals is the arm's band at the snapped
   row (`band_at`).
3. **snapshot**: one analysis of an LHS design of n (signal families) or a null design;
   proposals measured once; 200 fresh in-box held-out runs.
4. **snapshot_real**: one analysis of n random recipes of a real dataset (`true`) and
   of the same rows with the target permuted (`perm`: a null on real inputs); held-out
   runs are 200 other rows.
5. **backtest** (top priority; retrospective, pool-restricted; never a wet-lab win):
   hide later experiments, propose from earlier ones, score against what performed
   best. Real order where it exists (HeLa `Round`; T-cell `gen #`; TF-1 (`exp #`,
   `gen #`) lexicographic; Anagram S26APR07 -> S26APR29); otherwise seeded random
   orderings of recipes (`default_rng(seed).permutation`), cuts at 25/50/75 % of
   recipes (real orders: the step boundaries nearest to 25/50/75 % of rows). History is
   capped at the latest 400 recipes. Picks: q = 5, snapped to the pool (arms may pick
   pool rows directly). Metrics M1-M4 exactly as REAL_BACKTEST_PLAN.md: M1 =
   (max pool truth - max picked truth) / (max pool truth - min pool truth) on recipe
   means; M2 rank of the true best in the arm's ordering; M3 Spearman of the arm's
   mean vs truth over the pool; M4 distance of the unsnapped proposals to the true top-3
   relative to random box points. Baselines: random (200 draws), OFAT proxy (pool
   recipes ordered by the number of inputs differing from the incumbent by > 0.5 SD,
   then standardized distance), the lab's actual sequence (real orders: the first q
   pool recipes in recorded order). CI: bootstrap over cuts and orderings, B = 4000,
   seed 20261120; Win / Tie / Loss = the 95 % CI of the paired difference entirely
   below / containing / entirely above 0.
6. **scale**: leave-one-scale-out on `fixture/synthetic_scaleup` (synthetic) with the
   direction of each held-out scale (`up`, `down`, `interior`), which exposes the known
   extrapolate-up ranking failure (`kalos/scale/candidates.py`); and
   `internal/pichia_2L_to_35m3` (5 single runs): refusal or an untrusted verdict is the
   correct answer.
7. **internal**: section 3.4.

## 6. The adapter interface (`kalos/bench/ml_v2/adapter.py`)

```python
class Arm(Protocol):
    name: str
    shippable: bool                       # True only for engine-patch arms
    def propose(self, ctx: RoundContext) -> ArmOutput: ...
    def engine_patch(self) -> ContextManager[None]: ...   # in-process engine patch

RoundContext(suite, task, seed, round, sheet: DataFrame, target, space: Space, q, levels,
             roles, round_index, goal, candidates)        # read-only; ctx.arm_seed for randomness
ArmOutput(proposals: list[Proposal(recipe, pred, bands: {level: Band(lo, hi)}, label)],
          verdict: Verdict(trusted, state, withheld, cv_spearman, clears_floor),
          band_at: (DataFrame, level) -> (lo, hi) | None,   # intervals at arbitrary recipes
          predict: DataFrame -> mean,                        # ranking at arbitrary recipes
          info: dict)
```

6.1 **Engine-patch candidates (the only ones that can ship).** Subclass
`arms.kalos_ref.KalosEngineArm` and return a context manager from `engine_patch()`
that swaps an engine internal for one analysis (e.g. `kalos.core.surrogate.Surrogate`,
`kalos.core.optimize.propose_batch`, `kalos.portal.analysis.calibrate_selection_aware`,
`kalos.portal.analysis.PROPOSAL_FIT_RESTARTS`; patch the binding the caller looks up
at call time). The production `_analyze` path runs around it, so verdicts, refusals and
the planted-fault suite see the candidate. Template: `arms/example_patch.py`.
6.2 **Standalone arms** (BayBE, Ax, GP+UCB, trees) implement `Arm` directly;
`shippable = False`.
6.3 **Tree baselines** (`arms/trees.py`): seeded ensembles (5 members) of random
forests or XGBoost (`bench` extra; never a core dependency). Uncertainty: seed-ensemble
spread (`spread`), or grouped split-conformal (`conformal`, default: 5 folds grouped by
recipe, or by campaign round when there are >= 3 rounds; normalized scores; rank
ceil((n + 1) level)). Trust uses the engine's rule (grouped-CV Spearman >= 0.2, lower
CI > 0, >= 20 runs, >= 8 groups). Acquisition: EI on (mean, ensemble sd) over 4,096
uniform feasible candidates (pool: the not-yet-run rows), greedy batch with a 0.1
Chebyshev exclusion radius. Hyperparameters fixed in the module; HPO (as in
voyager-ml-pipeline) only on tuning seeds, frozen as an arm config.

## 7. Metrics (all with 95 % CIs over instances)

- **Simple regret** (closed loop, emulator, pool): (f* - f(x_best)) / (f* - f_med),
  x_best = the feasible recipe with the best MEASURED value (pinned clone only), judged
  on the noise-free truth (pool: recorded value; f* = pool max, f_med = pool median).
  Primary: after the last round; also per round and per n0. Secondary: best true value
  among all queried (`regret_best`).
- **Experiments to target**: runs until regret <= 0.25 and <= 0.10 (pool: the pool's
  top 5 %), right-censored at budget + 1; median and fraction reached. Compared against
  `kalos_ref`, `random` and `lhs`.
- **Coverage at proposals** (90 and 95 %): fraction of proposals whose fresh measurement
  falls in the band shown next to it; CI = bootstrap over instances (proposals within an
  instance are correlated). **Coverage at held-out runs**: 200 fresh in-box points
  (pool: unrun rows), same CI.
- **Interval width**: median (hi - lo) / (f* - f_med) (pool and real: / (max - median)),
  at proposals and at held-out runs.
- **False trust**: rate of `trusted` on null snapshots (synthetic nulls + permuted real
  targets), Wilson CI, per scenario and pooled. **Power**: rate of `trusted` on signal
  snapshots, per family and n, with the prospective Spearman (arm mean vs truth on
  held-out points) as the warrant.
- **Planted-fault scorecard**: `kalos.bench.faults` full tier (its own pre-registered
  protocol, unchanged) with the arm's engine patch (`python -m kalos.bench.ml_v2 faults`).
- **Wall time and CPU time** per arm call (median, p90); arm errors counted (an error
  spends the round on random proposals and is reported).
- Backtest M1-M4 (5.5); scale/clone Spearman, top-k precision, AUC, coverage under
  campaign shift (3.4).

Everything is reported **per family / dataset and pooled** (`report` command).
A method must not win on average while losing on one dataset without that being shown:
the report lists every task, and criterion C6 blocks it.

## 8. Success criteria (held-out seeds and datasets; `compare` command)

A candidate SHIPS only if all hold, against `kalos_ref` on the same held-out jobs:

- **C1 improvement, at least one of:**
  - C1a regret: paired mean difference in final regret (candidate - reference) over the
    synthetic closed-loop instances has a 95 % CI entirely below 0, and the pooled
    real pool-replay difference is <= 0;
  - C1b width: ratio of median interval width at proposals (90 %, synthetic closed loop)
    has a 95 % CI entirely below 1.
- **C2 coverage at proposals** (90 and 95 %, synthetic closed loop + emulator): pooled
  coverage >= nominal, and no family or dataset whose CI upper bound is below nominal.
- **C3 held-out coverage guard**: pooled coverage at held-out runs >= nominal - 0.02.
- **C4 no worse false trust**: pooled false-trust rate <= the reference's, and no null
  scenario more than 0.05 above the reference.
- **C5 no planted-fault regression**: `regressions(reference, candidate)` is empty.
- **C6 no hidden loss**: no family or dataset (synthetic, pool, emulator) where the
  candidate is worse on final regret, Wilcoxon signed-rank, Holm-corrected over all
  families and datasets at alpha 0.05, by more than 0.05.
- **C7 non-inferiority on the other axis**: shipping on regret, the width-ratio CI must
  not lie entirely above 1.10; shipping on width, the regret-difference CI must not lie
  entirely above 0.
- **C8 backtest**: no Loss (M1, 5.5) against the reference on the pooled real backtests.
- Wall time is reported; a median per-call time above 2x the reference needs the user's
  sign-off.
- Only a held-out run of a shippable (engine-patch) arm can ship. Report-only groups
  (internal, info_only, user) inform but never decide.

**Tree rule.** RF/XGBoost are adopted (as a surrogate, a second opinion, or for the
scale/clone tasks) only where they beat `kalos_ref` on the relevant metric with
nominal coverage maintained: (a) prospective accuracy (Spearman, and |error| / range)
and interval coverage and width on held-out runs at n = 12/20/40/80 and on the real
sets; (b) final regret in closed loop and pool replay; (c) AUC and top-k precision on
the scale-transfer and clone-ranking tasks. Adoption then needs an engine-patch
implementation that passes C1-C8.

**Scale and clone rule.** On `rank_96to24`, `transfer_loso` and the `scale` tasks, a
method is better only if it beats the previous-scale readout (carry-forward) baseline,
with a CI, under the campaign-grouped split; coverage under campaign shift below
nominal is reported as a trust failure. These tasks are report-only while the data
rights are unconfirmed.

## 9. Pre-registered candidate lists (the arms start after this protocol)

**ml-acquire** (worktree `~/kalos/kalos-engine-mlacq`, branch `research/acquisition`):
qUCB with beta fixed in advance at {1, 4, 9}; Thompson sampling via BoTorch
`MaxPosteriorSampling` on 4,096 Sobol candidates; mixed batches of q = 5 as
4 qLogNEI + 1 qUCB(beta = 4) and 3 qLogNEI + 2 qUCB(beta = 4); batch construction and
diversity (qLogNEI variants, fantasies vs local penalization, qNIPV for exploration when
untrusted); cost-aware acquisition; multi-fidelity / multi-task across scale stages;
multi-objective (qLogNEHVI, constrained NEI); a labelled step outside the observed range
(trust region, TuRBO) vs inside only; optimizer settings (restarts, raw samples,
iteration cap) jointly with proposal quality; BayBE and a well-configured Ax default as
external baselines; and a **fair rerun of the reported voyager-brain GP+UCB vs Kalos tie
(6.73 vs 6.72, p = 0.66, 30 campaigns)**: GP+UCB implemented as a standalone arm with
voyager-brain's configuration documented, run on the held-out closed-loop seeds,
paired against `kalos_ref`, plus the original 30-campaign setup reproduced to explain it.

**ml-surrogate** (`~/kalos/kalos-engine-mlsurr`, `research/surrogates`): SAAS fully
Bayesian GP; heteroscedastic noise (replicate variance); Kumaraswamy input warping;
priors and kernels for small n; replicate-aware noise including near-replicates;
mixed categorical handling; TabPFN (licence per version first; not adoptable unless the
licence allows commercial use in an Apache-2.0 product). **Trees as a second-opinion
diagnostic**: flag "GP may be missing structure" when the tree ensemble's grouped-CV
Spearman exceeds the GP's by >= 0.15 and the paired bootstrap CI of the difference is
above 0; scored on signal snapshots by the flag rate and its precision (flagged
instances must have a worse GP prospective Spearman than the tree's) and on null
snapshots by the false-flag rate. Port the calibration of voyager-ml-pipeline's
20-member XGBoost ensemble (check the code licence first). A **sparse / variational GP**
(GPyTorch `ApproximateGP`) beyond the 2,000-row exact cap, scored on wall time,
held-out Spearman, NLL and coverage at 2k, 5k and 20k rows (synthetic `bumps` d = 6, and
the T-cell / TF-1 sets), against exact GP at 2k and an exact GP on a 2k subsample.

**ml-trust** (`~/kalos/kalos-engine-mltrust`, `research/trust-v2`): tighter
selection-aware conformal (localized / weighted, model-sd scores, Mondrian by region,
cross-conformal / jackknife+, conformal BO methods); CV folds that match the served model
(audit S2); hidden batch or drift detection from run order or date (residual test); a
better-powered trust gate with false trust at or below the reference; MAPIE only as an
independent cross-check. For the scale and clone tasks: **(a) the previous-scale rank is
the mandatory baseline, (b) splits are grouped by campaign, (c) coverage under campaign
shift is a trust metric** (section 3.4).

**Scale transfer (all arms)**: Voyager 384w -> SF125 (`flask_small`, `transfer_loso`),
the Pichia 2 L -> 35 m3 runs (refusal expected), and the extrapolate-up ranking failure
(`scale` suite, direction `up`), with `kalos/scale/candidates.py`'s PhysicsMean and
MultiFidelity surrogates as existing candidates.

## 10. What this protocol does not claim

- Synthetic and emulator results are simulations; pool replay and backtests are
  retrospective and pool-restricted. None is a wet-lab result.
- Internal data have unconfirmed rights and tiny n; they inform, never decide.
- Seeds protect against tuning to a split, not to a dataset; the dataset-level split
  (3.2) is the guard for real data, and it is only as strong as the number of datasets.

## Amendments

None.
