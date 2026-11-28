### Changed - Phase 2 ML pipeline upgrades, efficiency improvements, and audit bugfixes

- **Narrowed `_RNG_LOCK` (`kalos/portal/analysis.py`)**:
  - Moved `_seeded_analysis` inside `_analyze_body` so `prepare_analysis` and `assemble_result` run lock-free while stages 2–4 remain deterministically seeded.
- **Visualization, caching & CV efficiency (`kalos/portal/viz_utils.py`, `kalos/portal/operating_ranges.py`, `kalos/portal/analysis_parts/cv_grouping.py`, `kalos/core/model_cache.py`, `kalos/core/evaluation.py`, `kalos/analysis/stages/explain.py`)**:
  - Batched `_response_surface_gp` into a single `(441, d)` `Surrogate.posterior` call, fixed `_embedding_pca` categorical code comparisons, and aggregated `_correlation_spearman` on `recipe_key` groups.
  - Added content-addressed `SurrogateCache` (`weights_only=True` `torch.load` + JSON metadata sidecar, atomic writes, bounded LRU eviction) and wired `operating_ranges.py` to support caching, `check_conditioning=True`, and replicate-aware fixed noise.
  - Deduplicated collapsed partitions in `leak_free_cv` before fitting and added opt-in `n_workers` (`KALOS_CV_WORKERS`, default `1`) parallel fold evaluation to `_oof` / `grouped_cv_report`.
- **Mixed-GP shapes, flagged one-hot feasibility & flagged small-N DoE (`kalos/core/gp_shape.py`, `kalos/core/feasibility.py`, `kalos/core/optimize.py`, `kalos/core/space_filling.py`, `kalos/portal/config.py`, `kalos/analysis/stages/{trust,propose}.py`)**:
  - Extended `ard_lengthscales` to read the continuous ARD sub-kernel on `MixedSingleTaskGP`.
  - Added opt-in `KALOS_FEAS_ONEHOT_CAT` (default off) for one-hot categorical encoding in `FeasibilityClassifier` and `_FeasibilityGate`.
  - Added torch-free `propose_space_filling` gated behind `KALOS_SPACE_FILLING_SMALL_N` (default off) for constrained Sobol/maximin proposals when `n_distinct_recipes < 2 * d` and `verdict != "reliable"`.
- **Multi-objective, Hybrid ODE+GP & SVGP tier (`kalos/core/multiobjective.py`, `kalos/core/mechanistic.py`, `kalos/core/svgp.py`)**:
  - Upgraded `MultiObjectiveSurrogate` & `propose_multiobjective` with per-objective `maximize`/`minimize`, fixed `noise`, `MixedSingleTaskGP`, `X_pending`, `inequality_constraints`, and `fixed_features`.
  - Added BoTorch `Model` wrapper (`_HybridBoTorchModel`) to `HybridSurrogate` while keeping `kalos.core.mechanistic` torch-free at import.
  - Added `SparseVariationalSurrogate` (`kalos/core/svgp.py`) as an inducing-point variational GP tier for `> 2,000`-row workloads.
- **ML audit fixes (`kalos/core/optimize.py`, `kalos/core/evaluation.py`, `kalos/portal/analysis_parts/{noise_blocks,strain_block}.py`, `kalos/analysis/stages/assemble.py`)**:
  - Preserved full pre-`fixed_features` bounds when sanitizing `pending` runs in `propose_batch`.
  - Fixed strict-JSON `cv_ci95: [nan, nan]` handling in `_alternative_scale_block`, zero-fitted-run clone refusal in `choose_fixed_level`, mixed-type group string coercion in `_oof`, negative OOF index guard in `group_mean_margin`, and free-categorical `proposal_optimizer` reporting in `assemble_result`.
