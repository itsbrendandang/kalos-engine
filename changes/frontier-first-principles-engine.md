# Frontier First-Principles Bioprocess ML Core (`feat/frontier-first-principles`)

Branched from `feat/ml-pipeline-phase2` (`c76445d`). Default `Surrogate`, `propose`, and `calibrate_selection_aware` paths remain unchanged behind existing guardrails.

## 1. Greenfield `kalos/core/frontier/` Architecture

All four core modules are implemented in pure NumPy / SciPy `float64` (zero top-level or runtime `torch`, `botorch`, or `gpytorch` imports), obeying `kalos` `"core"` layer contracts (`0` import violations):

- **`kalos/core/frontier/__init__.py`** — Lazy `TYPE_CHECKING` / `__getattr__` exports for all 10 public frontier symbols: `ActionBatch`, `ActionSlot`, `AdaptiveConformalState`, `AdaptiveTuRBO`, `HierarchicalBatchGP`, `MultiScaleTransferGP`, `ShiftWeightedConformal`, `TrustRegionState`, `VOIAllocator`, and `WarpedSparseGP`.
- **`kalos/core/frontier/warped_sparse_gp.py`** — `WarpedSparseGP`:
  - Kumaraswamy input warping (`w = 1 - (1 - u^alpha)^beta`, auto-enabled when `n_unc >= max(16, 2 * d)` so low-`N` high-`D` fits avoid overparameterization).
  - Exact-Jacobian shifted Box-Cox output warping with 11-point Gauss-Hermite back-transformation.
  - Dimension- and sample-regime-aware lengthscale priors: Hvarfner-style LogNormal regularization in underdetermined high-`D` (`d > 6` and `n_unc < 2 * d`), smoothly ramping to Horseshoe / Cauchy ARD sparsity when `n_unc > d`.
  - Tobit / Laplace-Newton censored `< LOD` likelihood and Cholesky LOO-IRLS robust outlier down-weighting.
  - Analytical standardized warped posterior mean/variance and gradients (`warped_posterior_mean_std`, `posterior_warped_and_grad`) verified against central finite differences (`eps=1e-6`, `rtol=1e-3`, `atol=1e-4`) in both Cholesky+warp and Tobit Laplace modes.
- **`kalos/core/frontier/hierarchical_gp.py`** — `HierarchicalBatchGP` and `MultiScaleTransferGP`:
  - Separable latent GP kernel plus compound-symmetric random batch effects (`sigma_batch^2`), analytic log-marginal-likelihood with block-structured Woodbury / Schur-complement inference for unseen-batch extrapolation under drift and multi-scale (e.g., bench -> bioreactor) transfer.
- **`kalos/core/frontier/acquisition.py`** — `TrustRegionState`, `AdaptiveTuRBO`, `ActionSlot`, `ActionBatch`, and `VOIAllocator`:
  - ARD-scaled trust region (upper-capping inert lengthscales at `2.0` during TR geometry sizing), scrambled Sobol TR + global candidate pools, Joint Cholesky Thompson Sampling blended with Analytic Log-Expected Improvement (`ts_weight=0.25`), Kriging Believer Schur repulsion + Chebyshev `min_separation` enforcement, and analytical L-BFGS-B local polishing (`_polish_slot_analytic_lbfgs`) for surrogates exposing `posterior_warped_and_grad`.
  - `VOIAllocator`: vectorized 15-point Gauss-Hermite Knowledge Gradient allocating replicate vs exploration slots.
- **`kalos/core/frontier/shift_conformal.py`** — `ShiftWeightedConformal` and `AdaptiveConformalState`:
  - Prediction-tilted (`w_tilt = exp(beta * z)`) and logistic-density-ratio (`sqrt(w_dens)`) covariate-shift weighted split conformal calibration using unfloored OOF scores (`|actual - pred| / std`) with prediction-time `sigma_floor_quantile=0.60` protection and fail-closed `n_eff` / `p_new` gating, plus online adaptive `alpha_t` tracking under variance drift.

## 2. Benchmark Evolution (`kalos.bench.frontier_race`)

Baseline arms compare against unmodified default `kalos.core.surrogate.Surrogate`, `kalos.core.optimize.propose`, and `kalos.portal.analysis_parts.intervals.calibrate_selection_aware` (`KALOS_NO_PAID_CALLS=1`, `KALOS_TORCH_THREADS=1`).

### `--seeds 5 --quick` (Round 1 -> Round 2 -> Round 3)

| Arena / Scenario | Baseline | Frontier Round 1 | Frontier Round 2 | Frontier Round 3 |
|---|---|---|---|---|
| **Arena 1** (25D sparse, opt `7.8431`) | mean `0.6630` / med `0.5863` (`6.33–6.54s`) | mean `1.2556` / med `0.6163` (`0.079s`) | mean `0.6086` / med `0.5837` (`0.110s`) | mean `0.7558` / med `0.6825` (`0.093s`) |
| **Arena 2** (6D log-normal + outliers, opt `10.5`) | mean `4.0991` / med `4.8106` (`5.01–5.54s`) | mean `3.7466` / med `4.3167` (`0.092s`) | mean `3.7897` / med `3.8904` (`0.108s`) | **mean `2.8888` / med `2.9877`** (`0.117s`) |
| **Arena 3** (5D `< LOD` cliff, opt `10.0`) | mean `4.5102` / med `5.5632` (`4.50–5.08s`) | mean `6.2565` / med `6.6364` (`0.503s`) | mean `3.3120` / med `3.3931` (`1.073s`) | **mean `2.6853` / med `2.5974`** (`1.055s`) |
| **Arena 4** (batch drift unseen-batch CV) | Spear `0.3441` / RMSE `1.6464` (`0.164–0.201s`) | **Spear `0.8791` / RMSE `1.1454`** (`0.054s`) | **Spear `0.8791` / RMSE `1.1454`** (`0.055s`) | **Spear `0.8791` / RMSE `1.1454`** (`0.055s`) |
| **Arena 5a** (`hartmann6`, 90% nominal) | cov `1.0000`, width `1.6702`, avail `1.0000` | cov `0.8400`, width `0.9905`, avail `1.0000` | **cov `1.0000`, width `1.4282`**, avail `1.0000` | **cov `1.0000`, width `1.4282`**, avail `1.0000` |
| **Arena 5b** (`bioprocess`, 90% nominal) | cov `1.0000`, width `2.1217`, avail `1.0000` | cov `0.8400`, width `1.4416`, avail `1.0000` | **cov `0.9600`, width `1.8346`**, avail `1.0000` | **cov `0.9600`, width `1.8346`**, avail `1.0000` |

### `--seeds 10 --quick` Generalization (Round 2 -> Round 3)

| Arena / Scenario | Baseline (`10 seeds`) | Frontier Round 2 (`10 seeds`) | Frontier Round 3 (`10 seeds`) | LHS (`10 seeds`) |
|---|---|---|---|---|
| **Arena 1** (25D sparse, `n_init=12`) | mean `0.6294` / med `0.5364` (`6.728–6.915s`) | mean `0.8192` / med `0.7945` (`0.113s`) | mean `0.7474` / med `0.8101` (`0.095s`, 71x faster) | mean `1.4810` / med `1.3849` |
| **Arena 2** (6D log-normal + outliers) | mean `3.5299` / med `4.1969` (`5.143–5.322s`) | mean `3.8506` / med `3.7547` (`0.108s`) | **mean `3.0987` / med `3.0267`** (`0.126s`, 41x faster) | mean `4.5159` / med `4.7768` |
| **Arena 3** (5D `< LOD` cliff) | mean `4.7222` / med `5.7431` (`3.880–4.128s`) | mean `3.3448` / med `3.3751` (`1.013s`) | **mean `2.8003` / med `2.4434`** (`1.005s`, 3.9x faster) | mean `5.2094` / med `5.2477` |
| **Arena 4** (batch drift unseen-batch CV) | Spear `0.3109` / RMSE `1.7225` / Obs `-0.1269` (`0.163–0.164s`) | **Spear `0.8454` / RMSE `1.0954` / Obs `0.1932`** (`0.052s`) | **Spear `0.8454` / RMSE `1.0954` / Obs `0.1932`** (`0.053s`, 3.1x faster) | — |
| **Arena 5a** (`hartmann6`, 90% nominal) | cov `0.9200`, width `1.4847`, avail `1.0000` | **cov `0.9400`, width `1.3300`**, avail `1.0000` | **cov `0.9400`, width `1.3300`**, avail `1.0000` | — |
| **Arena 5b** (`bioprocess`, 90% nominal) | cov `0.9600`, width `2.1205`, avail `1.0000` | **cov `0.9800`, width `1.9840`**, avail `1.0000` | **cov `0.9800`, width `1.9840`**, avail `1.0000` | — |

On 10 seeds, Round 3 decisively beats Baseline on Arenas 2, 3, 4, and 5 (including higher empirical coverage at tighter relative widths on both Arena 5 scenarios) while cutting Arena 1's Round 1 regret gap from `1.2556` to `0.7474` mean (1.5% of the `7.8431` optimum behind BoTorch `Surrogate`) at 71x faster wall-clock speed (`0.095s` vs `6.728s`).

## 3. Verification Gate

- `python scripts/import_layers.py` -> exited `0`: `196 modules, 1509 kalos imports, 0 violating import statements (0 module edges)`.
- `ruff check kalos tests` -> exited `0`: `All checks passed!`.
- `mypy kalos` -> exited `0`: `Success: no issues found in 196 source files`.
- `pytest tests/test_core_torch_contract.py tests/test_surrogate_guardrail.py tests/contract/ tests/test_frontier_*.py -q` -> exited `0`: `73 passed, 1 skipped` in `6.82s`.
- `python scripts/trust_golden.py diff` -> exited `0` (only pre-existing `provenance[*].reason` additions and `mixed_medium` PCA embedding sign flip; default `Surrogate` behavior unchanged).

## 4. Branch Inventory

- `feat/arch-phase1-layers-stages`: `2945236`
- `feat/ml-pipeline-phase2`: `c76445d`
- `feat/frontier-first-principles`: this commit
