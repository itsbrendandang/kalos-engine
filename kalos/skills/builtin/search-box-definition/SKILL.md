---
schema: kalos-skill/1
slug: search-box-definition
name: Input ranges (search box)
version: 0.1.0
kind: locked
category: design
summary: >-
  Use to set the search box for a campaign (the input ranges the engine reports as feature_ranges): which factors may move, their lower and upper limits, categorical levels,
  fixed factors, and linear constraints (for example a total-feed cap), with physical limits inferred from units.
  Not for proposing runs, and not a regulatory operating range.
keywords: [bounds, ranges, factors, constraints, search space, categorical levels]
starting_prompt: Set up the search ranges for pH, temperature and feed rate, keeping total feed under 30% of working volume.
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Checked table with confirmed roles."}
  - {name: overrides, type: json, required: false, description: "Scientist-set bounds, fixed factors or constraints."}
parameters:
  - {name: expand_fraction, type: float, default: 0.0, effect: "How far beyond observed ranges the search may go; 0 keeps proposals inside measured ranges."}
outputs:
  - {name: search_box, type: json, label: derived, description: "Factors with type, bounds, levels, provenance of each bound (observed, physical, scientist), and constraints."}
tools: [engine.build_search_box, engine.validate_constraint]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.domains.profile:build_design_space", ref: "kalos/domains/profile.py:216"}
    - {symbol: "kalos.domains.profile:DesignSpace", ref: "kalos/domains/profile.py:125"}
    - {symbol: "kalos.portal.analysis_parts.ranges:_physical_range", ref: "kalos/portal/analysis_parts/ranges.py:29"}
    - {symbol: "kalos.validation.bounds:infer_dimension", ref: "kalos/validation/bounds.py:236"}
    - {symbol: "kalos.portal.analysis_parts.refusals:_validate_constraint", ref: "kalos/portal/analysis_parts/refusals.py:105"}
  gaps:
    - "expand_fraction is a proposed parameter; current engine bounds come from observed and physical ranges only (needs new code)."
scope: read
side_effects: none
refuses_when:
  - A scientist bound is outside the physical range for that unit.
  - Constraints are infeasible together.
verification:
  deterministic: [every bound records where it came from, constraint feasibility checked on a sample of points]
  receipt: true
  review: none
author: Kalos
owner: system
visibility: system
licence: {spdx: Apache-2.0}
source: kalos-engine
status: active
created: 2026-11-05
updated: 2026-11-05
---
## Purpose
Make the optimiser's allowed region explicit and reviewable before any batch is proposed.

## When to use
Campaign setup, or whenever the scientist widens or narrows the input ranges.

## When not to use
For regulatory operating ranges. This skill only defines a search region.

## Inputs
Checked table; optional overrides.

## Procedure
1. Start from observed ranges per factor.
2. Clip to physical limits implied by units.
3. Apply scientist overrides, fixed factors and constraints; check feasibility.

## Outputs
`search_box` with the source of every bound.

## Stop conditions
Infeasible or unphysical settings.

## Caveats
Searching beyond observed ranges is extrapolation; `distance-to-data` will flag it.

## Example
"Fix the basal medium at lot B, let feed start day vary between day 2 and day 5."
