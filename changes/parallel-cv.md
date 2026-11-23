### Changed - cross-validation fits run in parallel, alongside the proposal fit

- Grouped-CV fold fits run in a pool of worker processes (`KALOS_CV_WORKERS`, default one per CPU; `1` fits them in the engine process). The analysis starts them before the proposal surrogate is fit, so the two overlap.
- Every fold fit runs on one torch thread with the torch RNG reset per fold, so the CV numbers are the same for any worker count and any `KALOS_TORCH_THREADS`. They can differ slightly from earlier releases, which fit folds on the engine's thread budget (on the example sheet, held-out predictions by up to 0.02 and the CV Spearman by 0.003).
- On `examples/synthetic_bioprocess` (8 CPUs), two-repeat grouped CV takes 38 s instead of 88 s, and a full analysis 185-294 s instead of 361-384 s (two runs each), with the same numbers.
