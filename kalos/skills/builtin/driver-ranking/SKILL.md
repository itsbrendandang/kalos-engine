---
schema: kalos-skill/1
slug: driver-ranking
name: Driver ranking
version: 0.1.0
kind: locked
category: analytics
summary: >-
  Use to rank which factors move together with the response (rank correlation with bootstrap intervals and
  false-discovery control), as a starting point for discussion. Not causal and not a substitute for designed experiments.
keywords: [drivers, factor importance, correlation, Spearman, FDR]
starting_prompt: Which process factors track most strongly with titer in this data?
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Checked, normalised table."}
parameters:
  - {name: top_k, type: int, default: 5, effect: "How many factors to list."}
  - {name: q, type: float, default: 0.05, effect: "False-discovery rate for flagging associations."}
outputs:
  - {name: drivers, type: json, label: derived, description: "Factor, rank correlation, bootstrap interval, adjusted p-value, flag."}
tools: [engine.bootstrap_spearman, engine.rank_drivers]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.core.drivers:bootstrap_spearman", ref: "kalos/core/drivers.py:95"}
    - {symbol: "kalos.core.drivers:benjamini_hochberg", ref: "kalos/core/drivers.py:44"}
    - {symbol: "kalos.core.drivers:rank_drivers", ref: "kalos/core/drivers.py:158"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - Fewer rows than needed for a bootstrap interval.
verification:
  deterministic: [seeded bootstrap reproducible, adjusted p-values monotone]
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
Give a quick, honest view of which factors appear to matter, with intervals and correction for testing many factors.

## When to use
Exploring a historical sheet; explaining a model to colleagues.

## When not to use
To claim cause and effect or to set limits.

## Inputs
Checked table.

## Procedure
1. Rank correlation per factor; bootstrap intervals.
2. Adjust p-values for multiple testing.
3. List top factors with their intervals.

## Outputs
`drivers`.

## Stop conditions
Too few rows.

## Caveats
Correlated factors (for example feed rate and total feed) share credit, and confounded historical data can mislead.

## Example
"Rank drivers of lactate at day 7."
