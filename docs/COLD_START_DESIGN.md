# Cold-start experiment design

A campaign with no measured runs, or too few for the model to be judged, gets a
**space-filling plan** instead of model-guided proposals: recipes chosen only to
cover the declared region. No model is used to choose them, so no row carries a
predicted value or an interval, and every row says so.

Code: `kalos/core/design.py` (the generator, torch-free), `kalos/portal/design_plan.py`
(campaign to design, labels, CSV), `kalos/portal/design_api.py` (the route). API
contract: docs/CAMPAIGNS_API.md, "Exploration design (cold start)". Benchmark:
`benchmarks/cold_start/`. Idea credit: the Latin-hypercube mode of the user's earlier
BayOp service (NOTICE).

## When it is used

| Situation | What the user gets |
|---|---|
| A campaign with no data | `POST /api/campaigns/{id}/design` with bounds (campaign setup or the request), setpoint steps and categorical levels: a plan and a run-sheet CSV to fill in and upload. |
| An analysis whose model is withheld (verdict `refused`) | `proposals: []` and the refusal as before, plus `exploration_design`: 5 recipes that fill the gaps between the runs already made. |
| An analysis whose batch is labelled exploration (verdict `not_yet`; `reliability.space_filling_advice` applies) | The exploration batch as before, plus `exploration_design` as the space-filling alternative the advice describes. |
| A trusted model (verdict `reliable`) | No `exploration_design` in the analysis. The route still answers, labelled "exploration design, model not used" with `model_not_used.code: model_trusted_not_used`. |

## Honest labels

- Every row: `label: "exploration design, no model yet"`, `model_used: false`.
- The block: `predictions: null` and a note; no key named `pred`, `std`, `mean`,
  `interval(s)`, `ei`, `p_improve` or `quantiles` anywhere (tested recursively).
- `model_not_used`: why the plan is not model-guided, with the engine's thresholds:
  `no_measured_runs`; `n_below_trust_threshold` (the model is not judged below 12
  runs and not called reliable below 20 runs and 8 groups); `model_not_trusted`
  (the verdict and its reasons); `model_trusted_not_used`.
- Started as runs (`POST .../runs`, `mode: "exploration_design"`), design rows record
  `prediction_at_start.available: false` and `pred_source: caller` with no `pred`.

## The generator

- **Methods.** `sobol` (scrambled Sobol, Owen scrambling via scipy) and
  `lhs_maximin` (Latin hypercube: best of 32 random starts by the smallest pairwise
  distance, then column swaps, which keep the Latin property, accepted when they
  raise it). The benchmark also runs `random`, `lhs` (plain) and `lhs_cd`
  (BayOp's default, `optimization="random-cd"`).
- **Determinism.** Everything random comes from `numpy.random.default_rng(seed)`;
  the global numpy and torch generators are not read or advanced (tested), so the
  analysis hook leaves every other analysis field byte-identical.
- **Bounds.** The campaign setup's bounds where declared; otherwise the request's
  `lo`/`hi`; otherwise (seeded campaign) the range already measured, with a warning
  that the plan stays inside it. A model-free plan may go beyond the measured range
  when the setup bounds do (`beyond_observed_range: true`): no surrogate is
  extrapolated.
- **Physical ranges.** A bound outside what is physically possible for the column's
  dimension (pH outside 0-14, a negative concentration, DO above 100%) is refused;
  outside the typical range it is a warning. Unit-aware through
  `kalos.validation.bounds.column_bounds` when it is available (feat/data-checks);
  on an engine without it only unit-free dimensions (pH) are checked, and the
  factor's `physical_check` says what was checked.
- **Linear constraints.** Rejection: every candidate is checked against the
  setup's constraints (relative tolerance 1e-6, the optimizer's own); rejected
  design and pool candidates are counted (`constraints`). A region with no feasible
  candidate is refused (`no_feasible_candidates`) with the counts.
- **Setpoints.** A declared `step` snaps to the absolute grid `k * step` inside the
  bounds (pipetting and weighing increments: with step 0.05 and lo 6.02 the values
  are 6.05, 6.10, ..., never 6.07; rounded to the step's decimals, so 6.35 and never
  6.3500000000000005); declared `levels` snap to the
  nearest level inside the bounds. Kalos never invents a step: an undeclared factor
  is continuous and only rounded to 1/10,000 of its range. Hints from the data
  checks (pH 0.05 or 0.1, temperature 0.5 °C, DO 5%, rpm 5-10) are conventions a
  client may show, never applied. A fully discrete space of at most 16,384 grid
  points is enumerated exactly; when fewer distinct recipes exist than asked, the
  plan is short and says so (`shortfall`) instead of repeating rows.
- **Categoricals and clones.** Levels are balanced counting the runs that already
  exist (jointly over up to 512 level combinations, else per column), and assigned
  round-robin along a farthest-point order of the rows, so each level gets a
  spread-out subset. The clone column follows the setup's `strain_policy`: `fixed`
  holds every row at the named clone, `factor` balances across clones. Covariates
  (passage, generation, ...) are never set.
- **Centre points and replicates.** Optional rows at the centre of the region (the
  feasible candidate nearest it when the centre breaks a constraint) and rows that
  repeat the most spread-out design rows, labelled `role: centre` / `replicate`.
- **Augmenting existing runs.** With any measured or started run, new points fill
  gaps: each is chosen to maximise its minimum distance to every existing point and
  to the points already chosen, from the feasible points of a scrambled-Sobol
  prefix of 4 x (existing + new) points (`method: maximin_fill`). The prefix is
  what keeps maximin out of the corners; see "Gap filling" below.
- **Quality report.** On the continuous factors scaled to [0, 1]: minimum pairwise
  distance, mean nearest-neighbour distance, centered L2 discrepancy, distance to
  the existing runs and discrepancy with them, and `random_reference`: the median
  of 64 random designs of the same size drawn from the same feasible candidates.

## Gap filling (benchmarks/cold_start/augment_study.py, 30 seeds per cell)

Existing points uniform in [0,1]^d; mean over seeds. `to_existing`: smallest
distance from a new point to an existing one (higher is better). CD: centered L2
discrepancy of existing + new (lower is more uniform).

| d | existing+new | method | to_existing | min_all | CD with existing |
|---|---|---|---|---|---|
| 2 | 3+9 | maximin_full | 0.3 | 0.29 | 0.02 |
| 2 | 3+9 | maximin_reflected | 0.229 | 0.222 | 0.01 |
| 2 | 3+9 | maximin_prefix | 0.278 | 0.256 | 0.014 |
| 2 | 3+9 | random | 0.097 | 0.061 | 0.03 |
| 2 | 3+9 | greedy_cd | 0.15 | 0.042 | 0.061 |
| 2 | 10+6 | maximin_full | 0.275 | 0.272 | 0.012 |
| 2 | 10+6 | maximin_reflected | 0.215 | 0.212 | 0.009 |
| 2 | 10+6 | maximin_prefix | 0.241 | 0.236 | 0.01 |
| 2 | 10+6 | random | 0.057 | 0.054 | 0.023 |
| 2 | 10+6 | greedy_cd | 0.11 | 0.043 | 0.027 |
| 4 | 3+9 | maximin_full | 0.743 | 0.732 | 0.162 |
| 4 | 3+9 | maximin_reflected | 0.43 | 0.392 | 0.059 |
| 4 | 3+9 | maximin_prefix | 0.616 | 0.593 | 0.069 |
| 4 | 3+9 | random | 0.331 | 0.256 | 0.084 |
| 4 | 3+9 | greedy_cd | 0.382 | 0.244 | 0.088 |
| 4 | 10+6 | maximin_full | 0.651 | 0.65 | 0.068 |
| 4 | 10+6 | maximin_reflected | 0.408 | 0.4 | 0.045 |
| 4 | 10+6 | maximin_prefix | 0.546 | 0.542 | 0.044 |
| 4 | 10+6 | random | 0.221 | 0.206 | 0.073 |
| 4 | 10+6 | greedy_cd | 0.288 | 0.231 | 0.045 |
| 6 | 3+9 | maximin_full | 1.007 | 0.989 | 0.299 |
| 6 | 3+9 | maximin_reflected | 0.502 | 0.437 | 0.146 |
| 6 | 3+9 | maximin_prefix | 0.857 | 0.838 | 0.168 |
| 6 | 3+9 | random | 0.474 | 0.385 | 0.185 |
| 6 | 3+9 | greedy_cd | 0.552 | 0.413 | 0.143 |
| 6 | 10+6 | maximin_full | 0.915 | 0.913 | 0.14 |
| 6 | 10+6 | maximin_reflected | 0.486 | 0.463 | 0.112 |
| 6 | 10+6 | maximin_prefix | 0.777 | 0.773 | 0.117 |
| 6 | 10+6 | random | 0.414 | 0.385 | 0.131 |
| 6 | 10+6 | greedy_cd | 0.502 | 0.425 | 0.089 |
| 10 | 3+9 | maximin_full | 1.335 | 1.323 | 0.99 |
| 10 | 3+9 | maximin_reflected | 0.675 | 0.52 | 0.432 |
| 10 | 3+9 | maximin_prefix | 1.182 | 1.167 | 0.645 |
| 10 | 3+9 | random | 0.764 | 0.665 | 0.583 |
| 10 | 3+9 | greedy_cd | 0.837 | 0.633 | 0.372 |
| 10 | 10+6 | maximin_full | 1.243 | 1.239 | 0.517 |
| 10 | 10+6 | maximin_reflected | 0.595 | 0.53 | 0.372 |
| 10 | 10+6 | maximin_prefix | 1.095 | 1.093 | 0.434 |
| 10 | 10+6 | random | 0.667 | 0.661 | 0.442 |
| 10 | 10+6 | greedy_cd | 0.754 | 0.677 | 0.302 |

Plain greedy maximin over a 1024-point pool has the largest distances but sends
points into the corners (in 10-D its discrepancy is 1.7x random). Counting the
distance to a face double ("reflected") keeps points inside but in 10-D falls
below random on distance. The shipped Sobol-prefix variant (`maximin_prefix`) keeps
83-93% of the plain maximin distance, is 1.5-4.2x farther from the existing runs
than random points, and has a lower discrepancy than random points in every cell
but one: in 10-D with 3 existing runs it is 0.645 against 0.583 for random, which
the per-request `random_reference` makes visible.

## Which method and how many starting runs (benchmarks/cold_start/)

BENCH_SECTION

## BayOp edge cases (tests/test_bayop_edge_cases.py)

BayOp's fixtures (a fabricated 24-run, 6-setting sheet and its variants) were
regenerated by their recipe. "Before" was checked by running the upload-side tests
on a clean origin/main checkout (4 passed).

| Fixture | BayOp | Kalos before this change | Kalos now |
|---|---|---|---|
| Flat target (all 0.5) | proposed 3 "optimal" points from a flat GP | refused `target_constant` (already handled) | same refusal; the design route gives a plan |
| Single row | proposed 3 points from a 1-point GP | refused `too_few_rows`, a sheet needs 6 runs (already handled) | same refusal; a gap-filling plan around the one run |
| NaN in an input and in the target | the GP raised; routed to FAIL | the blank-target run is excluded and counted; the blank `phosphate_gL` cell is read as absent (0) under the documented concentration-column convention and reported in `blank_cells_as_absent` (already handled) | same; a design skips an existing run with a blank setting and says so |
| Column mismatch (bounds name a column the data lacks) | ValueError; routed to FAIL | setup bounds refused `unknown_feature`; a missing target `target_not_found` (already handled) | same; the design route refuses `unknown_factor` |
| Missing bounds | routed to FAIL | no equivalent (bounds were optional) | the design refuses `factor_without_bounds` (new) |
| Large batch (20) in bounds | checked | no equivalent (5 proposals per batch) | a 96-row design stays in bounds and on its step (new) |

## Limits

- Distances treat every continuous factor's declared range as equally important;
  a narrow range declared too wide dilutes the others.
- Constraints are handled by rejection; a region that is a tiny fraction of its
  box (acceptance below about 1 in 1,000) can run short of candidates and says so.
- Categorical levels are balanced, not optimised jointly with the continuous
  positions.
- The benchmark is simulation and pool replay on synthetic or semi-synthetic
  sheets; no real campaign has been run from a Kalos cold-start plan yet.
