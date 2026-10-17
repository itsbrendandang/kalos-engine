# Kalos optimiser benchmark: preregistered protocol

> **Status: DRAFT, NOT YET FROZEN.** Nothing in this repository is a benchmark
> result. No confirmatory seed has been run. No performance claim of any kind may
> be made from this branch.

| Field | Value |
|---|---|
| Protocol version | 1.0-draft (adapted 2026-10-17 from the research draft `PROTOCOL.md`) |
| Preregistration commit | `<PREREG_COMMIT_SHA — fill in when frozen>` |
| Preregistration tag | `bench-prereg-v1` (**not yet created**) |
| Harness | `benchmarks/protocol/` (this repo) |
| Analysis script | `benchmarks/protocol/analysis.py` (run unchanged after the tag) |
| Problem constants | `benchmarks/results/problem_constants.json` |

## How this protocol becomes binding (do this before any real run)

1. Finish the pre-run checklist in section 13 (pilot, adapter smoke tests, pinned
   peer environments, problem constants committed).
2. Replace `<PREREG_COMMIT_SHA ...>` above with the SHA of the commit that freezes
   this file, the harness, the adapters, `analysis.py`, `problem_constants.json`
   and the seed lists (`benchmarks/protocol/seeds.py`).
3. Tag that commit: `git tag -a bench-prereg-v1 -m "Kalos benchmark preregistration v1" <sha> && git push origin bench-prereg-v1`.
4. Only then run confirmatory seeds. The harness refuses `--phase confirmatory`
   unless `HEAD` is the tagged commit (or a descendant whose diff against the tag
   touches nothing under `benchmarks/protocol/` or this file) and the working tree is clean.
5. Every change after the tag goes into the deviation log (section 15) with a
   date and a reason. Silent edits are not allowed.

---

## 0. Why this protocol exists

Earlier Kalos benchmarks had specific, documented flaws:

- 5-10 seeds, and comparisons of means or medians with no paired test (paired sign tests then came out at p ~ 0.4-0.6).
- A shared random start that already reached the target in 40-44% of seeds.
- An information-source column (`IS_indicator`) fed in as a design variable.
- A pool-EI arm that is not the production qLogNEI path.
- Enriched retrospective pools.
- An LHS snapping tie bug.
- Silent random fallbacks when the GP failed.
- A target defined on single noisy measurements.
- An objective switched after seeing results.

This protocol fixes each of these in advance. Every analysis choice is declared here before the data exist.

## 1. Questions and hypotheses

**Q1 (primary).** With a fixed evaluation budget and batch size, on noisy problems
with known optima, does the Kalos production proposal path end with a better
*recommended* recipe than non-adaptive designs and classical DoE+RSM? Is it no
worse than established open-source BO tools?

Per comparator `j`, define `d_s = P_kalos,s - P_j,s`, the paired difference of the
primary endpoint (section 7) over confirmatory seeds `s`. Lower regret is better,
so `d < 0` favours Kalos.

| ID | Comparator | Hypothesis type | H0 | H1 |
|---|---|---|---|---|
| H1 | Random | superiority | median(d) = 0 | median(d) != 0 (two-sided) |
| H2 | Sobol | superiority | median(d) = 0 | median(d) != 0 |
| H3 | DoE+RSM | superiority | median(d) = 0 | median(d) != 0 |
| H4 | Ax default | non-inferiority, margin 0.05 | median(d) >= 0.05 | median(d) < 0.05 |
| H5 | BayBE default | non-inferiority, margin 0.05 | same | same |
| H6 | Optuna TPE | non-inferiority, margin 0.05 | same | same |
| H7 | SMAC3 (RF) | non-inferiority, margin 0.05 | same | same |

H1-H7 form one family with Holm correction at familywise alpha = 0.05.
Superiority over the H4-H7 comparators is **exploratory**: it may only be called
significant if the two-sided test also survives Holm within the same 7-test
family re-run as superiority tests, and it is reported as exploratory either way.

**Q2 (SECONDARY, own family).** The same comparisons on oracle landscapes fitted
to published media and titer data (section 9). Decision (2026-10-17): the oracle
track is **secondary**. It is not implemented in the first harness PR.

**Q3 (descriptive only).** Pool replay on measured rows (section 10). No
hypothesis test from Q3 may be used in a claim. Not implemented in the first harness PR.

## 2. Arms (exactly 8)

Every arm runs through one adapter interface (`benchmarks/protocol/arms/base.py`):
`ask(n) -> X`, `tell(X, y)`, `recommend() -> x` (the arm's *own* recommendation,
used only for a secondary endpoint). Peer arms run in their own virtual
environment as subprocess workers (`benchmarks/envs/`). Each arm is used at its
documented default in the primary analysis, plus a fixed tuning budget (section 8).

| Arm id | Package (pinned at freeze) | Default configuration used in the primary analysis |
|---|---|---|
| `kalos_production` | kalos-engine at tag `bench-prereg-v1`; botorch 0.18.1, gpytorch 1.15.2, torch as locked in `uv.lock` at the tag | **Production path, full design box.** Exactly the `kalos/portal/analysis.py` proposal path: `noise_report` on the observed rows; if ≥1 recipe is replicated, ≥6 distinct recipes and a positive finite noise floor, fit `Surrogate()` on replicate means with fixed noise `sigma^2/n_reps`, otherwise `Surrogate().fit(X, y, bounds=B, cat_dims=C)` with inferred noise; then `propose(s, B, q=q, cat_dims=C, cat_cardinalities=K, seed=arm_seed)`. **B is the declared problem box** (decision 2026-10-17). The production observed-range box (`design.bounds()` = observed min/max) is a **tuning variant** (`kalos_production_observed_box`), not the primary arm. Synthetic problems have no replicates, so the inferred-noise branch is what runs; the replicate branch is implemented and unit-tested for parity. |
| `random` | numpy | IID uniform over the box. Categorical levels uniform. |
| `sobol` | scipy.stats.qmc | Continues the same scrambled Sobol sequence that generated the shared initial design, from index `n0` onward (categoricals continue the balanced cyclic assignment). |
| `doe_rsm` | numpy + scipy (no pyDOE needed for the shared-init track) | Rounds 1-4: D-optimal augmentation of the existing data for a full quadratic model (greedy sequential selection + one Fedorov exchange pass over 4096 Sobol candidates, ridge 1e-6). Rounds ≥5: fit a full quadratic by OLS (ridge 1e-6; categoricals as dummies with their linear interactions), take the box-constrained maximiser (20-start L-BFGS-B per categorical level), and fill the batch with that point plus `q-1` axial points at ±10% of each range around it, clipped and chosen in order of predicted value. Not run for d > 8. |
| `ax` | ax-platform 1.3.1 | `ax.api.Client` with the default generation strategy. Shared initial data are attached as completed trials. Then `get_next_trials(max_trials=q)`. No SEM, so Ax infers noise. Every round logs the generator name. |
| `baybe` | baybe 0.15.0 | `Campaign` with `SingleTargetObjective(NumericalTarget(..., mode MAX))` and the default recommender. Shared initial data added as measurements, then `recommend(batch_size=q)`. Continuous → `NumericalContinuousParameter`, categorical → `CategoricalParameter` (default encoding). |
| `optuna_tpe` | optuna 5.0.0 | `TPESampler(seed=arm_seed)`, all other arguments default. Shared initial data go in via `study.add_trial`. Batches come from `q` asks before the tells. |
| `smac3_rf` | smac 2.4.1 | `HyperparameterOptimizationFacade` (random forest), `Scenario(deterministic=True, n_trials=N)`. Shared initial data are told to SMAC before the first ask, with zero extra initial configurations. Batches come from `q` asks before the tells. |

Alternates, used only if an arm cannot be installed or fails the pilot gate, in
this order: ProcessOptimizer 1.1.2, then BoFire 0.5.0, then Optuna GPSampler.
Substitutions are logged in section 15. HEBO, Dragonfly, GPyOpt, EDBO/EDBO+,
Summit and Atlas are excluded (fail to install/import on CPython 3.12 macOS
arm64, or unmaintained; verified 2026-10-17).

## 3. Problems (synthetic track: primary)

All problems are maximisation problems: `g = -f` for the standard minimisation
forms. Constants are computed once with seed 12345 by
`python -m benchmarks.protocol constants` and committed to
`benchmarks/results/problem_constants.json`:

- `g*`: the known optimum. For constructed variants it is computed by exhaustive
  enumeration over the discrete dimensions × 64-start L-BFGS-B over the continuous ones.
- `g_med`: the median of `g` under 2^16 scrambled Sobol uniform samples.
- `s_g`: the standard deviation of `g` under the same samples.

| ID | Problem | d | Domain | Notes |
|---|---|---|---|---|
| P1 | Branin | 2 | x1 in [-5, 10], x2 in [0, 15] | 3 global optima, none at the box centre |
| P2 | Hartmann-3 | 3 | [0, 1]^3 | |
| P3 | Hartmann-6 | 6 | [0, 1]^6 | |
| P4 | Shifted Ackley-4 | 4 | [-2, 2]^4, optimum at s = (0.73, -1.12, 0.41, 1.29) | `a=20, b=0.2, c=2*pi`; optimum off the box centre and off CCD/BBD lattice points |
| P5 | Mixed Hartmann-6 | 6 | x1..x4 continuous in [0, 1]; x5 ordinal on {0, 0.25, 0.5, 0.75, 1}; x6 categorical with 3 levels mapped to {0.20, 0.65, 0.95} | process-choice dimension + grid dimension |
| P6 | Zero-inflated Hartmann-6 | 6 | [0, 1]^6 | `g_ZI(x) = max(g_H6(x) - q30, 0)`, q30 = 30th percentile of g_H6 under uniform |

(The research REPORT summary called P6 "ZeroInflated-Hartmann-4"; the protocol
table, which governs, is Hartmann-6.)

**Observation noise.** `y = g(x) + eps`, `eps ~ N(0, (eta * s_g)^2)`. Primary
eta in {0.10, 0.25}; secondary eta in {0, 0.50}. Common random numbers: the k-th
evaluation (k = 1..N, counting the initial design) of every arm in a given
(seed, cell) receives the same `eps_k`, and the shared initial design receives
identical observations in every arm.

**Ordinal and grid handling (one rule for all arms).** Ordinal dimensions are
exposed to every arm as continuous [0, 1]. The harness snaps each proposed
coordinate to the nearest grid level before evaluation (exact ties go up) and
the arm is told the **snapped** point.

**Categorical dimensions** are given natively (integer codes `0..K-1`) to every
arm that supports them. Categorical codes are rounded and clipped by the harness.
Categoricals are placed last in the design vector, as production does.

**Primary cell set.** 6 problems × eta in {0.10, 0.25} = 12 cells, at q = 5, shared-init track.

## 4. Budget, batch size, initial design

- `n0 = max(6, 2d)`: P1 6, P2 6, P3 12, P4 8, P5 12, P6 12.
- R = 8 rounds of q = 5, so N = n0 + 40 evaluations per run. (The q = 1 secondary track uses 40 sequential evaluations.)
- **Shared-init track (primary).** Per (seed, problem), the first `n0` points of a
  scrambled Sobol sequence over the continuous + ordinal dimensions, seeded from the
  INIT stream (section 5). Ordinals then snap. Categoricals get a seeded balanced
  cyclic assignment (a seeded permutation of the levels, cycled). The init stream
  does not depend on eta or q, so a given (seed, problem) has the same initial X in
  every noise cell.
- **Not-solved rule.** If any initial point has noiseless normalised regret
  r ≤ 0.10 (section 7), redraw with the next scramble seed (redraw index k+1), up to
  100 redraws. The redraw count is logged per run. If no acceptable design exists
  after 100 redraws the run aborts (it is never silently accepted). If the pilot
  estimate of P(redraw needed) for a cell exceeds 50%, the cell's `n0` is halved and
  logged as a deviation *before* the confirmatory run.
- All arms receive the identical initial X and y (asserted by unit test).
- **Independent-init track (secondary).** Each arm uses its own default initial
  design. Not implemented in the first harness PR.

## 5. Seeds and reproducibility

- Confirmatory seeds: 0-49 (synthetic); 0-29 (oracle). Pilot seeds: 9001-9005.
  Tuning seeds: 8001-8010. The three sets are disjoint and the harness refuses to
  run a seed outside the set of the requested phase. Pilot results are never pooled
  with confirmatory results; every log row carries its `phase`.
- Every random stream is `numpy.random.SeedSequence([seed, problem_id, noise_id, q, track_id, stream_id, k])`
  (implementation: `benchmarks/protocol/seeds.py`). `stream_id`: 0 = INIT (with
  `noise_id = 0` and `q = 0` so the init design is shared across noise levels and
  batch sizes; `k` = redraw index), 1 = NOISE, 2 = FALLBACK (labelled Sobol fill
  stream), 3 = TIE (tie-breaking), 100 + arm index = each arm's internal RNG
  (`k` = retry index; a retry after a failed step uses `k + 1`, the protocol's "seed + 1").
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`, one run per worker process.
- Record `pip freeze` per arm venv, the git SHA, the OS version, the chip and the
  Python build (`<out>/env/`).
- Reproducibility gate: rerun 3 randomly selected confirmatory (seed, cell, arm)
  triples. The endpoint must reproduce within 1e-6, or within 1e-3 for arms whose
  libraries are documented as non-deterministic (named in the report).

## 6. Instrumentation and failure handling (no silent fallbacks)

Every `ask` is wrapped (`benchmarks/protocol/runner.py`, `instrument.py`) and logs
one JSONL row per round:

- wall-clock time of the ask, and for Kalos the fit and acquisition parts separately;
- exceptions by type and message;
- for Kalos: every `fit_gpytorch_mll` attempt, the Cholesky jitter of each attempt,
  which jitter rung succeeded, and whether `FitError` was raised. This wraps
  `kalos.core.surrogate.fit_gpytorch_mll` from the harness; production code is not changed;
- warnings captured by category (botorch `OptimizationWarning`, `NumericalWarning`,
  `InputDataWarning`, `BotorchWarning`, gpytorch `NumericalWarning`, and everything else);
- NaN or Inf in the posterior mean or variance at the proposed points;
- proposals outside the box (clipped, counted) or duplicating an evaluated point
  (distance < 1e-6 in the normalised box);
- the generator name for Ax/BayBE.

**Failed step.** An exception escapes the arm's `ask`, the proposal has the wrong
shape, or the proposal is non-finite. The harness then (1) records `failed=True`,
(2) retries once with the arm's retry stream (`k + 1`), (3) if that also fails,
fills the batch from a *separate* labelled Sobol stream (FALLBACK) and records
`fallback=True` for those points. There is no other fallback path; there is no
`except` in the harness that selects a point without setting `fallback=True`
(enforced by a unit test).

**Crashed run.** The worker process dies or exits without a `run_end` row. It is
rerun once. If it crashes again, the run is recorded as `imputed_loss=True`, and
its endpoint is set at scoring time to the worst r observed by any arm in that
(seed, cell).

**Reporting.** For every arm and cell: failed-step, fallback, warning and
imputed-loss rates. The primary analysis is intention-to-treat (fallbacks and
imputed losses stay in). Sensitivity: (a) exclude runs with any fallback, (b)
count those runs as losses.

**Reliability gate.** An arm with fallback steps above 5% in any cell, or above
1% overall, is labelled UNRELIABLE in every table. If that arm is Kalos, the result is NO-GO.

## 7. Endpoints

**Normalised regret** of a point x: `r(x) = (g* - g(x)) / (g* - g_med)` on the
**noiseless** g. r = 0 is the optimum, r = 1 a median random point. No clipping.

**Recommendation rules** (applied after rounds 0..R):

- **Common rule (primary)** (`benchmarks/protocol/recommend.py`, run in the scoring step, never inside an arm):
  1. Fit a reference BoTorch GP on the arm's own observed (X, y): `SingleTaskGP`
     with default priors, inferred noise, `Normalize` to the problem box,
     `Standardize`. **Clarification:** for problems with categorical dimensions the
     reference model is BoTorch's `MixedSingleTaskGP` (default kernels; continuous
     dims normalised to the box), because a `SingleTaskGP` would treat level codes
     as ordered. The reference GP does not use any Kalos code.
  2. `x_hat` = argmax of the posterior mean over the arm's **evaluated** points.
  3. If the reference fit fails or the posterior is non-finite, `x_hat` = argmax
     observed y, and `ref_fit_failed=True` is logged.
  4. At eta = 0, `x_hat` = argmax observed y.
- Secondary rules: (i) the arm's own `recommend()` (final round only); (ii) best
  observed noisy y; (iii) in-sample best *true* value, labelled "oracle-selected".

**Primary endpoint.** `P_arm,s` = mean over the 12 primary cells of `r(x_hat)`
after N evaluations (shared-init, q = 5, default configs, common rule). One
number per arm per seed. A seed missing any primary cell for any arm is reported
and the analysis refuses to produce a primary result.

**Secondary endpoints.** S1 regret curve and `AUC = mean over rounds 1..R`; S2
time to target (first evaluation with in-sample best true r ≤ 0.10, censored at
N + 1); S3 per-cell r(x_hat) at N; S4 q = 1 track; S5 eta in {0, 0.50}; S6
independent-init track; S7 tuned configurations; S8 wall-clock per proposal
(median, p95) and total CPU time per arm; S9 failure, fallback and warning rates.

## 8. Tuning budget (fixed, disjoint suite)

- At most 4 configurations per arm, including the default, listed here before the run.
- Tuning suite: Rosenbrock-4, Levy-5, Styblinski-Tang-4, Mixed Levy-4 (one 3-level
  categorical), all at eta = 0.25, q = 5, seeds 8001-8010, shared-init rules.
  (Not implemented in the first harness PR.)
- Selection: lowest mean common-rule r at N. Frozen, committed, then used in the S7 analysis only.

| Arm | Configurations |
|---|---|
| Kalos | default (full box); **observed-range box** (`design.bounds()` rule on the data so far; implemented as `kalos_production_observed_box`); `num_restarts=20, raw_samples=1024`; `prune_baseline=False` |
| Ax | default; explicit BoTorch modular generator with SingleTaskGP + qLogNEI + 20 restarts |
| BayBE | default; `BotorchRecommender` with qLogNEI; random initial recommender |
| Optuna TPE | default; `multivariate=True`; `multivariate=True, group=True`; `n_startup_trials=n0` |
| SMAC3 | HPO facade default; BlackBox facade (GP); HPO facade with random-design interleaving probability 0 |
| DoE+RSM | default; 3 D-optimal rounds; ridge 1e-3 |
| Random, Sobol | none |

## 9. Oracle track (Q2, SECONDARY): semi-synthetic landscapes from real data

Not implemented in the first harness PR. Design kept as drafted:

- Datasets: O1-O4 Narayanan et al. 2025 *K. phaffii* (HSA, RBDJ, Trastuzumab,
  TL_HSA); O5 Hashizume 2023 HeLa regular; O6/O7 Kim & Audet 2019 TF-1 / T-cell;
  O8 Cosenza 2023 MOBO (max fidelity); O9 Kalos private media DoE (local only,
  aggregate statistics only). **Excluded:** DBO rat myocyte (`IS_indicator` mixes
  assays; licence possibly CC BY-NC-ND 4.0).
- Inputs: design variables only; replicates aggregated to recipe means; log1p if skewness > 1.
- Oracles: ExtraTrees (500 trees, `min_samples_leaf=3`, headline) and a
  scikit-learn GP (Matern-5/2 ARD + WhiteKernel, sensitivity).
- Fidelity gate (committed before any optimiser runs): grouped 5-fold CV × 5;
  include only if CV Spearman ≥ 0.40 **and** CV R^2 ≥ 0.15.
- Domain = observed box + observed levels; optimum = best of 2^17 Sobol points
  refined from the top 20; flag "optimum in unsupported region" if it is further
  from the data than the 95th percentile NN distance.
- Noise from replicates where available, else eta in {0.10, 0.25}.
- `n0 = min(max(6, 2d), 20)`, R = 8, q = 5, 30 seeds; same analysis, own Holm family.

### Data licensing rules (binding)

| Data | Rule |
|---|---|
| Synthetic problems | Code only; committed. |
| Narayanan et al. 2025 (figshare 10.6084/m9.figshare.27715134, **MIT**) | Fetched at runtime from the pinned URL and verified by SHA-256 (`benchmarks/data/manifest.json`). May be committed only if small **and** the MIT licence text + citation are committed next to it. Not committed in this PR. |
| Hashizume 2023, Kim & Audet 2019, Cosenza 2023 (Amii-derived CSVs) | **Never committed.** Fetch at runtime from a pinned commit URL with a SHA-256 check, or skip the dataset. |
| DBO rat myocyte / Cosenza et al. 2022 (CC BY-NC-ND 4.0) | **Never committed, never used for a claim.** |
| Any Amii or Kaggle dataset file | **Never committed.** |
| `examples/datasets/dataset_0/1.csv` (#81 stack) | Not used. Unknown provenance. |
| Kalos private media data (`KALOS_MEDIA_DATA`) | Never committed; local runs only; aggregates only. |

## 10. Pool replay (Q3, descriptive)

Not implemented in the first harness PR. Design kept as drafted: same pool for all
arms; [0, 1] normalisation by pool min/max; nearest-*unevaluated*-row snap with
seeded tie-breaking; arms Kalos-production-snap, Kalos-pool (`optimize_acqf_discrete`,
joint q), Random, Sobol-snap; replicate-mean ground truth; top-5% target over
recipes with n_rep ≥ 2 when ≥ 20 exist (else labelled "single-measurement target
(noisy)"); initial design only from recipes below the 80th percentile; n0 = 10;
DBO excluded or stratified by `IS_indicator` with the indicator removed from the
inputs; 30 seeds; W/T/L and HL with CIs, no p-values in headline text. Every
report states the pools are enriched, random is a strong baseline, and results do
not transfer to open design spaces.

## 11. Statistical analysis plan

Implemented in `benchmarks/protocol/analysis.py` and run unchanged after the tag.

1. **Paired differences.** Per comparator, `d_s = P_kalos,s - P_j,s`.
2. **Superiority (H1-H3).** `scipy.stats.wilcoxon(d, zero_method="pratt", alternative="two-sided")`.
3. **Non-inferiority (H4-H7).** One-sided Wilcoxon on `d - 0.05`, `alternative="less"`.
   Non-inferiority also requires the upper bound of the 95% paired-bootstrap CI of
   the Hodges-Lehmann shift to be < 0.05.
4. **Multiplicity.** Holm across the 7 primary p-values at alpha = 0.05.
   Secondary families get their own Holm correction and are labelled secondary.
5. **Effect sizes.** Hodges-Lehmann paired shift (median of Walsh averages) with a
   95% percentile CI from 10,000 paired bootstrap resamples of seeds (bootstrap seed
   20261017); mean difference with a BCa CI; matched-pairs rank-biserial correlation.
6. **Wins/ties/losses** per seed, tie = |d_s| < 0.01. Reported next to every p-value.
7. **Power.** With 50 paired seeds a Wilcoxon test has about 80% power for dz = 0.5
   at alpha 0.05/7. A non-significant result is reported as "not detected", not "no difference".
8. **Curves.** Mean and median regret per round with 95% bootstrap bands (CSV; PNG
   when matplotlib is installed via the `bench` extra). Descriptive only.
9. **Reliability gate** (section 6) evaluated and printed with every table.
10. **Pre-declared exclusions:** none, except the oracle fidelity gate and pilot `n0`
    changes, both recorded before the confirmatory run.

## 12. Compute budget

Planning numbers from the research draft (M3 Pro, single-threaded): Kalos
continuous fit 0.03-0.4 s, continuous propose 0.04-0.9 s, mixed GP fit ~2.4 s;
~2 s per model-based q = 5 round. Primary block (12 cells × 8 arms × 50 seeds) ≈
4,800 runs ≈ 13 CPU-h; whole programme ≈ 130 CPU-h, plan 200. The pilot in
`benchmarks/pilot/` re-measures per-step timing on this harness; see its README for the
measured numbers (PILOT, not evidence about performance).

## 13. Go / no-go checklist

**Before the confirmatory run (all must be true):**

- [ ] This protocol is committed at tag `bench-prereg-v1`, together with the harness, the adapters, `analysis.py`, `problem_constants.json` and the seed lists, and the SHA is written at the top of this file.
- [ ] Every arm installs from its requirements file in `benchmarks/envs/` (locked with `pip freeze` at freeze time) and passes an adapter smoke test (Branin, 2 seeds, q = 1 and q = 5).
- [ ] Ax and BayBE generator names are logged, and each arm's default configuration matches its documentation.
- [ ] Unit tests prove: the not-solved rule; common random numbers; identical shared initial data across arms; the snap rule; failure instrumentation (forced `FitError`, forced NaN proposal, forced worker crash); reproducibility of a run.
- [ ] No `except` in the harness selects a point without setting `fallback=True` (unit test).
- [ ] Pilot finished on pilot seeds only: timing measured, no arm above the reliability gate, cell `n0` adjustments logged.
- [ ] (Secondary track only) the oracle fidelity table is committed.
- [ ] Data licence check passed (section 9 table).

**After the run (GO for publishing results):**

- [ ] Kalos fallback rate ≤ 5% in every cell and ≤ 1% overall; every arm's rate reported.
- [ ] `analysis.py` ran unchanged; its output hash is recorded.
- [ ] Reproducibility gate passed.
- [ ] Every primary and secondary result is reported, including losses and nulls. No cherry-picked cells.
- [ ] Deviations, if any, are in section 15 with a reason and a date.

**NO-GO conditions:** any unlogged deviation; a Kalos reliability failure; mixing
pilot and confirmatory seeds; changing the endpoint, recommendation rule, tie
threshold, NI margin or cell set after seeing confirmatory data.

## 14. Claims policy

### Claims allowed (only after a GO, with numbers, CI and Holm-adjusted p)

- "On the preregistered synthetic suite (12 cells, 50 seeds, q = 5), the Kalos production proposal path reached a lower recommended-point regret than random / Sobol / DoE+RSM (HL shift X, 95% CI [a, b], W/T/L n/n/n, Holm p = ...)." — or the opposite, if that is what the data show.
- "Kalos was non-inferior (margin 0.05 normalised regret) to Ax / BayBE / Optuna TPE / SMAC3 defaults on this suite." — or that non-inferiority was not shown.
- "On oracle landscapes fitted to published media/titer data that passed a fidelity gate (list), ..." (secondary, labelled as such).
- Workflow claims backed by their own tests (e.g. the pending-duplication result).
- "The harness is implemented and the pilot ran end to end" (engineering statement, not a performance claim).

### Claims NOT allowed

- Any performance claim before the tag exists and the confirmatory run passed the GO checklist. **This includes this PR, its README and its pilot outputs.**
- "Scientifically validated", "clinically/industrially validated", "proven".
- Any claim about real-lab experiment savings ("saves N experiments", "X% faster in your lab").
- Generalising pool-replay results to open design spaces.
- Citing pilot, tuning, exploratory or secondary results as confirmatory.
- "Beats Ax/BoTorch/BayBE" unless the superiority test in section 1 survives Holm, and even then only "on this suite".
- Any claim from the DBO dataset, or from private data beyond aggregate statistics.
- Reporting medians or means without the paired analysis, or dropping null results from the README.

## 15. Deviation log

| Date | Section | Change | Reason | Decided before seeing confirmatory data? |
|---|---|---|---|---|
| 2026-10-17 | 2, 8 | Kalos primary arm uses the declared problem box; observed-range box is a tuning variant. | Owner decision before freeze. | Yes (pre-freeze) |
| 2026-10-17 | 1, 9 | Oracle track confirmed SECONDARY; not in the first harness PR. | Owner decision before freeze. | Yes (pre-freeze) |
| 2026-10-17 | 7 | Reference GP for categorical problems is `MixedSingleTaskGP`. | `SingleTaskGP` would treat level codes as ordered. | Yes (pre-freeze) |
| 2026-10-17 | 2 | SMAC3 shared init is told before the first ask instead of via `additional_configs`. | Gives identical told data to every arm without SMAC re-evaluating the init configs. | Yes (pre-freeze) |

## 16. References

- Parsaee et al. 2026, *Bioinformatics*, doi:10.1093/bioinformatics/btag607. Code: github.com/Amii-Applied-AI/amii-cell-ag-tools at commit 5a095fd.
- Narayanan et al. 2025, *Nat Commun*, doi:10.1038/s41467-025-61113-5. Data: doi:10.6084/m9.figshare.27715134 (MIT).
- Hashizume, Ozawa, Ying 2023, *npj Syst Biol Appl*, doi:10.1038/s41540-023-00284-7 (CC BY 4.0).
- Kim & Audet 2019, *Commun Biol*, doi:10.1038/s42003-019-0296-7 (CC BY 4.0).
- Cosenza et al. 2023, *Eng Life Sci*, doi:10.1002/elsc.202300005 (CC BY 4.0).
- Cosenza et al. 2022, *Biotechnol Bioeng*, doi:10.1002/bit.28132 (CC BY-NC-ND 4.0). Possible DBO source, not used.
- Liang et al. 2021, *npj Comput Mater*, doi:10.1038/s41524-021-00656-9.
- Siska et al. 2026, *Biotechnol Bioeng*, doi:10.1002/bit.70129.
- Bayesmark / NeurIPS 2020 BBO challenge; HPOBench; COCO/BBOB; Olympus. Methodology only.
