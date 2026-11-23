### Changed - cross-validation fits run in parallel, alongside the proposal fit

- Grouped-CV fold fits run in a pool of worker processes (`KALOS_CV_WORKERS`, default the CPUs divided by `KALOS_TORCH_THREADS`; `1` fits them in the engine process). The analysis starts them before the proposal surrogate is fit, so the two overlap; a benchmark's daemon worker runs them after the proposal fit, in its own process.
- Each worker fits on the engine's torch thread count from the engine's torch RNG state, so the outputs are byte-identical to fitting the folds one after another: `scripts/trust_golden.py` output matches main on all 8 sheets at 1 and 4 torch threads.
- On `examples/synthetic_bioprocess` (8 CPUs, one run each), a full analysis took 65 s instead of 68 s at 1 torch thread and 155 s instead of 166 s at 4; the 8 trust-golden sheets took 40 s instead of 43 s.
