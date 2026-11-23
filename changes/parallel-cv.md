### Changed - cross-validation fits run in parallel, alongside the proposal fit

- Grouped-CV fold fits run in a pool of worker processes (`KALOS_CV_WORKERS`, default the CPUs divided by `KALOS_TORCH_THREADS`; `1` fits them in the engine process). While the analysis fits its proposal surrogate, the pool precomputes the primary CV's folds; the CV then runs after the proposal fit, as before, and uses a precomputed fold only when it started from the same torch RNG state. A benchmark's daemon worker fits the folds in its own process.
- Each worker fits on the engine's torch thread count from the engine's torch RNG state, so the outputs are byte-identical to fitting the folds one after another: `scripts/trust_golden.py` output and a full analysis of `examples/synthetic_bioprocess` match main at 1 and 4 torch threads.
- On `examples/synthetic_bioprocess` (8 CPUs, one run each, noisy), a full analysis took 114 s instead of 155 s at 1 torch thread and 206 s instead of 242 s at 4.
