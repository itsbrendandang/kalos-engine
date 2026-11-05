---
schema: kalos-skill/1
slug: target-readiness-check
name: Target readiness check
version: 0.1.0
kind: locked
category: data-quality
summary: >-
  Use to decide whether the chosen response column can support a model at all: enough usable rows, enough
  variation, not constant, not mostly blank. Not for judging model accuracy after fitting.
keywords: [target, response, titer, variation, too few rows, refusal]
starting_prompt: Is titer usable as the optimisation target in this sheet?
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Checked table."}
  - {name: target, type: column, required: true, description: "Proposed response column."}
parameters: []
outputs:
  - {name: target_readiness, type: json, label: derived, description: "Usable-row count, variation class (constant / near-constant / usable), and a refusal reason if any."}
tools: [engine.assess_target_variation]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.validation.target:assess_target_variation", ref: "kalos/validation/target.py:95"}
    - {symbol: "kalos.portal.analysis_parts.refusals:_refuse_unusable_target", ref: "kalos/portal/analysis_parts/refusals.py:60"}
    - {symbol: "kalos.portal.analysis_parts.refusals:_too_few_rows", ref: "kalos/portal/analysis_parts/refusals.py:24"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - Fewer usable rows than the engine minimum.
  - The target is constant or has no numeric values.
verification:
  deterministic: [refusal text comes from the engine's refusal table, never from the model]
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
Refuse early and plainly when the response cannot support optimisation, instead of producing a confident-looking batch from noise.

## When to use
Right after the response column is confirmed.

## When not to use
After a fit, to judge prediction quality (`trust-report`).

## Inputs
Checked table and the response column name.

## Procedure
1. Count usable numeric values (blanks ignored, not zeroed).
2. Classify the variation.
3. Return pass or a named refusal.

## Outputs
`target_readiness`.

## Stop conditions
Any refusal ends the campaign setup until data changes.

## Caveats
Passing this check means "a model can be attempted", not "a model will be good".

## Example
"Check whether `final_titer_g_L` is usable."
