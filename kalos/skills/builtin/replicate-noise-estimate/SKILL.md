---
schema: kalos-skill/1
slug: replicate-noise-estimate
name: Replicate noise estimate
version: 0.1.0
kind: locked
category: data-quality
summary: >-
  Use to find replicate runs (same recipe repeated), estimate the run-to-run noise floor of the response, test whether
  noise grows with the signal, and state how much of the observed spread is just noise. Not for estimating factor effects.
keywords: [replicates, noise floor, signal to noise, heteroscedasticity, repeatability]
starting_prompt: How noisy is titer between repeated runs of the same recipe, and is that noise bigger than the effects we are chasing?
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Checked table with confirmed factor columns and response."}
  - {name: group_column, type: column, required: false, description: "Explicit replicate or batch key; otherwise identical rounded recipes are grouped."}
parameters:
  - {name: decimals, type: int, default: 6, effect: "Rounding used to decide that two recipes are identical; coarser rounding merges near-identical recipes."}
outputs:
  - {name: noise_report, type: json, label: derived, description: "Rows, distinct recipes, replicated recipes, pooled noise SD, noise-vs-mean trend, signal-to-noise ratio."}
tools: [engine.noise_report, engine.estimate_noise_floor, engine.heteroscedasticity_report]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.core.replicates:noise_report", ref: "kalos/core/replicates.py:354"}
    - {symbol: "kalos.core.replicates:estimate_noise_floor", ref: "kalos/core/replicates.py:175"}
    - {symbol: "kalos.core.replicates:heteroscedasticity_report", ref: "kalos/core/replicates.py:192"}
    - {symbol: "kalos.core.replicates:aggregate_replicates", ref: "kalos/core/replicates.py:125"}
    - {symbol: "kalos.validation.checks:check_replicate_adequacy", ref: "kalos/validation/checks.py:633"}
    - {symbol: "kalos.portal.analysis_parts.noise_blocks:_noise_block", ref: "kalos/portal/analysis_parts/noise_blocks.py:28"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - No recipe is replicated; the noise floor is then reported as "not estimable", never guessed.
verification:
  deterministic: [noise SD recomputed from replicate groups matches the report, groups listed with sizes]
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
Anchor every later claim to the measured repeatability of the process. If the noise is as large as the differences between recipes, say so before proposing anything.

## When to use
After the run-sheet checks; before trusting any model ranking.

## When not to use
To rank factors (`driver-ranking`).

## Inputs
Checked table; optional explicit replicate key.

## Procedure
1. Group replicate runs.
2. Pool within-group variance into a noise floor.
3. Check whether variance rises with the mean (suggests a log scale).
4. Compare the noise floor with the spread across recipes.

## Outputs
`noise_report`.

## Stop conditions
No replicates: return "not estimable" and recommend adding repeat runs in the next round.

## Caveats
Few replicates give a wide uncertainty on the noise floor itself, so report group counts alongside it.

## Example
"Estimate replicate noise using `bioreactor_id` as the grouping key."
