# Head-to-head benchmark: Kalos vs other optimizers (pre-registered protocol)

| Field | Value |
| --- | --- |
| Status | **Pre-registered** (this file is committed before any run of the head-to-head harness) |
| Protocol written | 2026-10-18 (UTC) |
| Scope | Synthetic objectives with known optima. No real wet-lab data. |
| Harness | `kalos/bench/vs_baselines.py`, run with `python -m kalos.bench --vs-baselines` |
| Results | `benchmarks/results/vs_baselines.csv`, `benchmarks/results/vs_baselines.json` |
| Deviations | Section 12, dated, with the reason. Nothing is changed silently. |

This is a fixed, written-down comparison, not a tuned one. Kalos runs with its production defaults. Every other method runs with its library defaults plus only the arguments listed below. No method is tuned on the benchmark seeds.

## 1. Questions

- **Q1 (primary).** On the synthetic tasks in section 3, with a fixed evaluation budget and the batch and pending settings in section 4, what is the best *queried* value after the budget, relative to the known optimum, for each method?
- **Q2.** How many experiments does each method need to get 90% (and 75%) of the way from the random-design median to the optimum? Experiments include the shared initial design.
- **Q3.** How much wall-clock time does each method use?

Descriptive comparisons. Each cell reports mean regret, 95% bootstrap intervals, paired win/tie/loss counts against Kalos, and Holm-adjusted exploratory Wilcoxon p-values. A method is reported as worse or better than Kalos in a cell only when the 95% paired bootstrap interval of the mean difference excludes zero. Otherwise it is reported as a tie or unresolved. Losses are reported in the same tables as wins.

## 2. Methods (arms)

| Arm id | Implementation | Configuration used |
| --- | --- | --- |
| `kalos` | `kalos.core.surrogate.Surrogate` + `kalos.core.optimize.propose` (this repo) | Production defaults. Inferred noise. `propose(..., q, cat_dims, cat_cardinalities, pending, seed)`. No tuning. |
| `random` | numpy | Uniform over the box; categorical levels uniform. |
| `lhs` | scipy `qmc.LatinHypercube` (as in `kalos.bench`) | Shared init, then the remaining N - n0 points of a fresh seeded LHS design (drawn up front, evaluated in order). Categorical levels uniform (as in `kalos.bench`). |
| `optuna_tpe` | `optuna` (bench extra) `TPESampler(seed=...)` | Library defaults (`constant_liar=True` is the default in optuna 5.0.0; it is what gives pending handling). Shared init via `add_trial`. Batches via `q` `ask` calls before the matching `tell` calls. |
| `ax` | `ax-platform` (bench extra) `ax.api.client.Client` | Default generation strategy (`method="fast"`) with its Sobol initialisation seeded (`initialization_random_seed`); BoTorch RNG seeded before each ask. Shared init attached as completed trials. `get_next_trials(max_trials=q)`. Ax reads RUNNING trials as pending observations (checked in the installed source). Noise inferred (no SEM). |
| `botorch_gp_ei` | BoTorch directly (`SingleTaskGP`, `Normalize` to the box, `Standardize`, `fit_gpytorch_mll`, `qExpectedImprovement` with `X_pending`, `optimize_acqf`) | "Vanilla" reference: the textbook GP-EI, with no Kalos code. Categorical codes are passed as ordinal numbers and the acquisition output is rounded to the nearest level (no categorical kernel). |

Not run, and why:

- **scikit-optimize** (`gp_minimize`): last release 0.10.2 and effectively unmaintained. It has no native pending-run handling beyond the constant-liar trick, so it is not a like-for-like comparator here.
- **HEBO, Dragonfly, GPyOpt, EDBO, Summit, Atlas**: not evaluated for this run.

Any method that raises during a step gets a uniform-random fallback for that step, and the fallback is counted per run and reported. It is not silently hidden.

Optional dependencies live in the `bench` extra (`pip install -e ".[ml,bench]"`), not in core.

## 3. Tasks

All objectives are maximisation. `f*` is the known optimum. `f_med` is the median of `f` over 2^16 scrambled Sobol points in the box (seed 12345). The noise level is a fraction `eta` of the objective's `scale`, the same convention as `kalos.bench`.

| Task id | Definition | Box | f* | scale | Noise cells |
| --- | --- | --- | --- | --- | --- |
| `gaussian_4d` | `10 exp(-4 ||x - 0.6||^2)` (existing `gaussian_bump(4)`) | [0,1]^4 | 10 | 10 | eta = 0, 0.15 |
| `ackley_4d` | `-Ackley` (existing `ackley(4)`) | [-2,2]^4 | 0 | 6 | eta = 0, 0.15 |
| `bioprocess_like_mixed` | 3 numeric factors `u` in [0,1]^3 (temperature, pH, feed rate, normalised) and 2 categorical factors: strain s in {0..3}, inducer i in {0..2}. `f = off[s] + gain[i] + 6 exp(-0.5 ||(u - c[s]) / w[s]||^2)` with off = [0, 1.0, 0.5, 2.0], gain = [0, 0.8, 2.5], c = [[0.3,0.6,0.4],[0.7,0.3,0.6],[0.5,0.8,0.2],[0.2,0.2,0.8]], w = [0.35, 0.20, 0.15, 0.30]. | [0,1]^3 x levels | 10.5 (closed form: max over s,i of off+gain, plus 6) | 6 | eta = 0.15 |

Notes:

- Scale for `bioprocess_like_mixed` is the amplitude 6. This is a stand-in with a deliberate trap: the best offset strain (s=3) has the narrowest peak. It is **a synthetic function written for this benchmark, not measured process data.**
- Noise (common random numbers): for a given (task, eta, seed), the noise vector `eps_1..eps_N` is `numpy.random.default_rng([seed, task_index, eta_index]).standard_normal(N)` times `eta * scale`. The k-th evaluation of every method gets `eps_k`, whatever point it proposes.
- Categorical codes are integers `0..K-1`. Every method gets them natively, except `botorch_gp_ei` (section 2).

## 4. Budgets, batches and pending runs

| Task | Shared initial design | Batch size q | Rounds | Total evaluations N | Pending runs |
| --- | --- | --- | --- | --- | --- |
| Synthetic (`gaussian_4d`, `ackley_4d`) | 5 (LHS, seeded) | 1 | 18 | 23 | Not applicable (q = 1, each result told before the next ask) |
| `bioprocess_like_mixed` | 8 (LHS on numeric factors, uniform categories, seeded) | 4 | 5 | 28 | **Pipeline depth 2.** Of each batch's 4 points, the first 2 are measured before the next proposal and the last 2 stay in flight (not yet measured) while the next batch is proposed. In-flight points are passed to each method that supports it (see below). |

- The shared initial design is identical for all methods at a given seed and task (asserted in tests).
- Pending support: `kalos` via `propose(pending=...)`; `botorch_gp_ei` via `X_pending`; `optuna_tpe` via RUNNING trials (constant liar); `ax` via RUNNING trials. `random` and `lhs` do not use observed data, so pending points cannot change what they propose.
- At the end of the run, the in-flight points are measured, so every method has measured exactly N points.

## 5. Seeds

- **Confirmatory seeds: 1000-1029 (30 seeds).** Fixed in `kalos/bench/vs_baselines.py` as `CONFIRMATORY_SEEDS`. This is the default for `python -m kalos.bench --vs-baselines`.
- **Pilot seed: 9001.** Used only to measure runtime (section 10). Its results are never reported. Pilot runs are not written to `benchmarks/results/`.
- Seeds 0-9 are the seeds of the earlier `python -m kalos.bench` sweep. They are not used here.
- Every method at a given (task, eta, seed) gets the same shared initial design and the same noise stream. Method-internal RNG is `torch.manual_seed(seed)` plus a seeded `numpy` generator.

## 6. Endpoints

Let `g_t` be the true (noiseless) value of the best point queried after `t` evaluations, and `p_t = (g_t - f_med) / (f* - f_med)` the normalised progress (0 = median random point, 1 = optimum).

- **Primary.** `R_N = f* - g_N`, the simple regret on the best queried value after N evaluations. This is the "regret = best queried value" definition: the best queried point's *true* value is used, which is an oracle selection. Reported in raw units, and normalised as `r = 1 - p_N`.
- **Secondary S1 (recommendation without the oracle).** `R^obs_N = f* - f(argmax_i y_obs_i)`: the true value of the point with the best *noisy* observation. This is the regret a practitioner who only sees measurements would get. It is reported for every cell, and it matters most for the noisy cells.
- **Secondary S2 (experiments to reach X%).** The first evaluation index `t` (counting the shared init) with `p_t >= X`, for X = 0.90 (headline) and 0.75. Censored at N+1 when not reached. Reported as reach rate, and as mean censored experiments with a bootstrap CI.
- **Secondary S3.** Mean best-so-far curves `p_t`, with 95% bootstrap bands (descriptive).
- **Secondary S4 (cost).** Wall-clock time per run, per proposal (median and p95), and fallback counts per method.

## 7. Statistics

- 95% intervals: percentile bootstrap over seeds, 10,000 resamples, bootstrap seed 20261018.
- Paired comparison with Kalos per cell: `d_s = r_kalos,s - r_method,s`, with `r = 1 - p_N`. Negative values favour Kalos. Mean `d` and its 95% bootstrap CI. Also W/T/L per seed with tie = |d_s| < 0.01 (1% of the normalised range).
- Exploratory: two-sided Wilcoxon signed-rank test on `d_s`, Holm-adjusted across the comparators within each cell. These p-values are reported only as supplementary and do not change the "better/worse/unresolved" call above.
- Power: with 30 paired seeds, the bootstrap CI can separate methods whose mean normalised difference is larger than about 0.05. Smaller differences are reported as unresolved, not as equal.

## 8. Tuning policy

- Kalos runs with defaults: `propose` default restarts and raw samples, default priors, inferred noise. No parameter was changed after seeing benchmark results.
- Other arms: library defaults. The only exceptions are the explicit arguments named in section 2 (seeds, categorical levels, batch sizes, pending handling).
- No arm is tuned on the confirmatory seeds. Any tuning must be done on pilot seeds, logged in section 12, and reported.

## 9. Reproducibility and failures

- Every run emits a JSON record: method, task, eta, seed, N, trajectory, fallback count, failure messages, wall-clock time, library versions.
- Reproducibility check: three randomly selected (method, task, eta, seed) triples are rerun after the main run. Their final regret must match within 1e-6 (the Kalos and BoTorch arms are seeded; any mismatch is reported).
- Failures are never hidden. A method that fails on a step uses the uniform fallback for that step; the fallback count is reported per method and cell.

## 10. Compute budget

- Target: under about one hour on an 11-core laptop (Apple M3 Pro). Runs are parallelised across processes with one thread each (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`).
- If the pilot (seed 9001, timing only) predicts more than about one hour for 30 seeds, the seed count is reduced to fit and the change is logged in section 12. Cells are not dropped.

## 11. Environment record

The results JSON records: Python, OS, CPU, torch, botorch, gpytorch, numpy, scipy, optuna, ax-platform versions, and the git commit SHA of the harness.

## 12. Deviation log

| Date (UTC) | Section | Change | Reason | Before seeing confirmatory data? |
| --- | --- | --- | --- | --- |
| 2026-10-18 | all | Protocol committed before any harness code or run. | Pre-registration. | Yes |
| 2026-10-18 | 3 | Reference median `f_med` for `bioprocess_like_mixed` is over 2^16 scrambled Sobol points on the numeric factors, with uniform categorical levels. Section 3 only said "Sobol" and "uniform" for this task. | Clarification of an omitted detail, written before any run. | Yes |
| 2026-10-18 | 2 | The Ax arm sets `objective="y"` (maximise). In the Ax version pinned by the `bench` extra, a bare metric name is maximised; `"-y"` minimises. The sign was found wrong during a pilot on seed 9001 (the arm proposed the corners) and was corrected before the confirmatory run. The fix was decided from Ax's own best point, not from a regret comparison. Pilot numbers were printed; they are not reported. | Implementation bug, found by a direction check before confirmatory data existed. | Yes (pilot printed; not used for any decision except this bug fix) |
| 2026-10-18 | 5 | `--quick` smoke seeds are 9002-9003 (not 1000-1001), so no confirmatory seed is used in any smoke test. Quick output is never written. | Keeps the confirmatory seeds untouched. | Yes |
| 2026-10-18 | 2 | Pilot seed 9001 gave a runtime of about 57 s wall for one seed across all arms, on 10 workers. That projects to roughly 12 min for 30 seeds, within the section 10 target, so the seed count stays at 30. | Timing check. | Yes |
| 2026-10-18 | 2, 9 | **Ax arm now seeded** (`configure_generation_strategy(method="fast", initialization_random_seed=seed)` and `torch.manual_seed` before each ask). The first confirmatory run failed the reproducibility gate for Ax: the same (gaussian_4d, eta 0.15, seed 1007) run gave regret 0.0789, then 0.0121. The other arms passed. Fixed by seeding, and the **full** confirmatory suite was rerun on the fixed harness. Run 1 is kept under `benchmarks/results/superseded/run1_ax_unseeded/` and reported in the README. The fix was triggered by the reproducibility gate, not by the Ax numbers. Seeds, cells, budgets, endpoints and statistics are unchanged. | Reproducibility gate (section 9) failed. | **No** (made after seeing run-1 results; disclosed here) |

## 13. Limitations (fixed in advance)

- All objectives are synthetic. Nothing here says anything about real assays, real media, or real process outcomes.
- Regret uses the best *queried* true value, which is an oracle selection. S1 gives the practitioner's view.
- One batch size (q = 1 for synthetic, q = 4 for the bioprocess-like task), one noise level for the bioprocess-like task, and 4 dimensions for the synthetic tasks. Results do not generalise to other dimensions or budgets.
- Arms are at library defaults, not at each library's best configuration. A tuned Optuna, Ax or BoTorch could do better. Kalos is not tuned either.
- The `bioprocess_like_mixed` surface was designed by the author. Its difficulty is not calibrated against any real process.
