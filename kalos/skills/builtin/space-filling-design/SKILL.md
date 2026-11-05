---
schema: kalos-skill/1
slug: space-filling-design
name: Space-filling starting design
version: 0.1.0
kind: locked
category: design
summary: >-
  Use to plan a first round of runs when there is little or no data: a space-filling set (Sobol or maximin Latin hypercube)
  inside the search box, respecting constraints and existing runs, with a coverage report. Not for later rounds
  where a fitted model exists (use bo-next-batch).
keywords: [DoE, space filling, Latin hypercube, Sobol, initial design, screening]
starting_prompt: Plan 12 starting runs that cover the ranges evenly, avoiding the 4 runs we already have.
inputs:
  - {name: search_box, type: json, required: true, description: "Output of search-box-definition."}
  - {name: existing_runs, type: table, required: false, label: measured, description: "Runs already done; new points keep away from them."}
parameters:
  - {name: n_runs, type: int, default: 12, effect: "Batch size; more runs give better coverage."}
  - {name: method, type: enum, allowed: [sobol, maximin-lhs], default: maximin-lhs, effect: "Sobol gives low-discrepancy sequences; maximin LHS maximises the smallest distance between points."}
  - {name: seed, type: int, default: 0, effect: "Fixes the random stream so the design is reproducible."}
outputs:
  - {name: design, kind: table, label: derived, min_bytes: 8, description: "Proposed recipes, one per row, in engineering units."}
  - {name: coverage_report, kind: json, label: derived, min_bytes: 2, description: "Minimum pairwise distance, per-factor coverage, distance to existing runs."}
tools: []
entrypoint: null
requires_extras: []
implementation:
  status: needs-new-code
  engine:
    - {symbol: "kalos.core.optimize:proposal_separation", ref: "kalos/core/optimize.py:218"}
    - {symbol: "kalos.domains.profile:DesignSpace", ref: "kalos/domains/profile.py:125"}
  gaps:
    - "Only benchmark-private LHS helpers exist (kalos/bench/benchmark.py:24 _lhs, kalos/bench/vs_baselines.py:125 _lhs_design). A production design generator is needed, e.g. on scipy.stats.qmc, with constraint rejection and categorical handling."
scope: read
side_effects: none
refuses_when:
  - n_runs is smaller than the number of free factors plus one.
  - Constraints leave no feasible points in a large random sample.
verification:
  deterministic: [same seed gives identical design, all points inside bounds and constraints, minimum separation reported]
  receipt: true
  review: none
author: Kalos
owner: system
visibility: system
licence: {spdx: Apache-2.0}
source: kalos-engine
status: planned
created: 2026-11-05
updated: 2026-11-05
---
## Purpose
Spend the first experimental budget on learning the whole region evenly, instead of guessing.

## When to use
Round 0, or when a model is refused for lack of data.

## When not to use
Once `trust-report` says the model is usable; then switch to `bo-next-batch`.

## Inputs
A search box; optional existing runs.

## Procedure
1. Generate candidates with the chosen method and seed.
2. Discard infeasible points; repeat until `n_runs` points are found.
3. Push points away from existing runs; report coverage.

## Outputs
`design`, `coverage_report`.

## Stop conditions
Infeasible constraints; too few runs for the factor count.

## Caveats
A space-filling design is about coverage, not about finding the optimum. Classical screening designs (factorial, definitive screening) are a separate, planned skill.

## Example
"Design 16 starting runs across pH 6.6–7.2, 34–37 °C, and three feed strategies."
