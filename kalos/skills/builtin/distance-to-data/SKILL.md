---
schema: kalos-skill/1
slug: distance-to-data
name: Distance-to-data check
version: 0.1.0
kind: locked
category: trust
summary: >-
  Use to say, for every proposed or requested recipe, how far it is from the nearest measured run and which factors sit at
  or beyond the edge of the measured range, so extrapolation is visible. Not for scoring how good a recipe is.
keywords: [extrapolation, edge of range, nearest run, novelty, applicability domain]
starting_prompt: How far are these proposed runs from anything we have actually measured?
inputs:
  - {name: recipes, type: table, required: true, label: predicted, description: "Proposed or hand-entered recipes."}
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Measured runs defining the data region."}
  - {name: search_box, type: json, required: true, description: "Input ranges used for range normalisation."}
parameters:
  - {name: edge_fraction, type: float, default: 0.05, effect: "Share of each factor's range treated as 'at the edge'."}
outputs:
  - {name: distance_report, kind: json, label: derived, min_bytes: 2, description: "Per recipe: nearest measured run id, range-normalised distance, factors at or beyond the edge, an in-data / near-edge / extrapolating label."}
tools: []
entrypoint: null
requires_extras: []
implementation:
  status: needs-new-code
  engine:
    - {symbol: "kalos.core.optimize:proposal_separation", ref: "kalos/core/optimize.py:218"}
    - {symbol: "kalos.domains.profile:DesignSpace", ref: "kalos/domains/profile.py:125"}
  gaps:
    - "Reuse the range-normalised Chebyshev metric of proposal_separation, but compute it against measured runs rather than other proposals. Add per-factor edge flags (kalos-web PR #65 has a UI edge-of-range flag that is not on main)."
scope: read
side_effects: none
refuses_when:
  - Recipe columns do not match the search-box factors.
verification:
  deterministic: [distance is 0 for a recipe identical to a measured run, categorical mismatch counts as maximal distance, labels follow fixed thresholds]
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
Make extrapolation impossible to miss. A proposal far from any data should look different from one in well-sampled territory.

## When to use
Alongside every `bo-next-batch` output and every hand-entered "what if" recipe.

## When not to use
To rank recipes.

## Inputs
Recipes, measured runs, search box.

## Procedure
1. Normalise every factor by its design range.
2. Find the nearest measured run per recipe (Chebyshev distance; categorical levels compared exactly).
3. Flag factors within `edge_fraction` of, or beyond, the measured range.
4. Assign a fixed-threshold label.

## Outputs
`distance_report`.

## Stop conditions
Column mismatch.

## Caveats
Distance in factor space ignores which factors matter most. A recipe can be close to the data yet sit in a region where the response changes sharply.

## Example
"Check distance-to-data for the 4 proposals from round 3."
