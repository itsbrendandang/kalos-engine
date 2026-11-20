# Analysis performance

Where an analysis spends its time, what runs in parallel, and why the results do
not change. Measurements are on an Apple M4 Max (14 cores) shared with other work;
absolute times vary run to run with the machine's load, so every table states the
load it was taken under.

## Where the time goes

`examples/synthetic_bioprocess` (160 rows, 10 continuous inputs and the 3-level
`medium_base` factor), before this change:

| Stage | Share | Why |
|---|---|---|
| Acquisition (`propose_batch` -> `optimize_acqf_mixed`) | ~83% | q=5 greedy slots x 3 levels = 15 multi-start L-BFGS solves; every one stops at botorch's `maxiter=2000` |
| Grouped CV (`grouped_cv_report`) | ~11% | 2 repeats x 5 folds = 10 GP fits |
| Proposal fit | ~5% | 3 hyperparameter starts |
| Everything else | <1% | parse, drivers, surfaces, report |

The work is dominated by Python overhead on tiny matrices, so more torch threads
do not help (4 threads were slower than 1).

## What runs in parallel

`kalos/core/parallel.py` runs independent tasks in a pool of worker processes
(spawn context, one torch thread each) and returns exactly what running them in
order in the analysis process returns:

| Work | Tasks | Where |
|---|---|---|
| CV fold fits: both repeats of the grouped CV, every leak-free grouping candidate, LOGO, log-scale CV | one per fold; all repeats in one batch | `kalos/core/evaluation.py` (`_fit_folds`) |
| The proposal fit's extra starting points | `restarts - 1` (2); they run while the default start is fit in the analysis process | `kalos/core/surrogate.py` (`_multi_start_fit`) |
| The mixed acquisition's per-level solves | one per categorical level, per greedy slot | `kalos/core/mixed_parallel.py` |

Not parallelized: the continuous acquisition. Its 10 restarts are one joint
L-BFGS problem (botorch sums them), so splitting them would change the result.

### Why the results are identical

- Worker processes, never threads: gpytorch/linear_operator settings and
  `warnings.catch_warnings` are process-global; a thread pool changed the batch.
- Each worker runs one torch thread, and the pool is used only when the analysis
  process also runs one (`KALOS_TORCH_THREADS=1`, the default).
- The global RNGs. Every task starts from the RNG state the serial loop would have
  started it from. A task that draws from the RNG (a botorch fallback) is detected;
  the analysis process then adopts its end state and computes the rest serially.
- The acquisition: the initial conditions of every level are drawn in the analysis
  process, in level order, by the same botorch call, so they consume the RNG as the
  serial loop does; only the RNG-free solves go to the pool, and the best is picked
  with botorch's rule. A solve that hit botorch's OptimizationWarning retry (which
  would draw new initial conditions) makes that slot recompute serially from the
  saved RNG state. Tied to botorch 0.18.1 (`BOTORCH_VERSION`); any other version runs
  botorch's own function.
- A task that raises, cannot be pickled, or meets a broken pool is recomputed in
  the analysis process, so the caller sees the serial result or the serial error.
- Models come back from workers pickled, which is a deep copy, as the serial path
  makes.

`tests/test_parallel_identity.py` enforces it: the full `_analyze` response (every
key but `timestamp`, `fit_seconds`, `created_at`) compared as a JSON string, serial
vs a 2-process pool, on the eight trust-golden sheets and a 40-row mixed slice of
the example, at several seeds; it also asserts the pool really ran each kind of
task, and covers each fallback.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `KALOS_ANALYSIS_WORKERS` | `auto` | Worker processes per analysis. `auto` = `cpus // KALOS_JOB_WORKERS`, at most 5, so the analyses the engine runs at once use about every CPU and no more (a 4-vCPU host: 2 per analysis). `0`/`1` = serial. A malformed value refuses to start. |
| `KALOS_TORCH_THREADS` | `1` (bare metal), `0` in deploy/compose.yaml | The pool runs only at 1. |

The pool is serial (and gives the same result) inside any multiprocessing child
(benchmark harnesses with their own pools, the pool's own workers), when the
analysis process runs more than one torch thread, and when the pool cannot start.
It starts on the first analysis of a process (2-4 s, overlapping the first CV
fits), is stopped after 5 minutes idle, and its workers exit when their parent
dies. A background job's pool is inside the job's process group, so a cancel or
timeout kills it too. Each worker holds about 300 MB once torch is loaded.

Scripts that run analyses must keep their top-level code under
`if __name__ == "__main__":` (a spawned worker imports the main script).

## Measured

See the PR description and `CHANGELOG.md` for the before/after table on this Mac.
