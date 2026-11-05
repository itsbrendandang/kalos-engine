---
schema: kalos-skill/1
slug: scale-up-envelope-check
name: Scale-up envelope check
version: 0.1.0
kind: locked
category: scale-up
summary: >-
  Use to ask whether a recipe measured at small scale is expected to hold at a larger vessel, using the engine's
  scale-up readout: required-column and minimum-scale gates, a leave-one-scale-out backtest ladder, physics proxies (kLa, specific power,
  gas velocity, hydrostatic pressure), and a decision that may be a refusal. Not for choosing a recipe.
keywords: [scale-up, tech transfer, kLa, P/V, backtest, refusal, vessel]
starting_prompt: Will our best 2 L recipe hold at 200 L, and how sure can we be?
inputs:
  - {name: multi_scale_sheet, type: table-file, required: true, label: measured, description: "Runs at two or more scales with scale and agitation/aeration columns."}
  - {name: target_spec, type: json, required: true, description: "Target scale, vessel geometry/physics inputs, recipe parameters."}
parameters:
  - {name: seed, type: int, default: 0, effect: "Seed for every random step (bootstrap, CV shuffles, restarts); recorded in the receipt."}
outputs:
  - {name: scale_readout, kind: json, label: predicted, min_bytes: 2, description: "Gate results, backtest rungs, ratio to baseline, physics features with assumptions, decision (predict / predict with warning / refuse) and provenance."}
tools: [engine.scale_build_readout]
entrypoint: "kalos.scale.readout:build_readout"
requires_extras: [ml]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.scale.readout:build_readout", ref: "kalos/scale/readout.py:883"}
    - {symbol: "kalos.scale.readout:gate", ref: "kalos/scale/readout.py:212"}
    - {symbol: "kalos.scale.readout:ladder", ref: "kalos/scale/readout.py:403"}
    - {symbol: "kalos.scale.readout:decide", ref: "kalos/scale/readout.py:535"}
    - {symbol: "kalos.scale.evaluation:leave_one_scale_out_report", ref: "kalos/scale/evaluation.py:172"}
    - {symbol: "kalos.scale.features:compute_scale_features", ref: "kalos/scale/features.py:333"}
    - {symbol: "kalos.portal.scale_routes:scale_readout", ref: "kalos/portal/scale_routes.py:483"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - A gate fails (missing columns, fewer distinct scales than required).
  - The backtest does not support predicting at the requested scale ratio.
verification:
  deterministic: [refusal names the failed check, physics constants and assumptions listed, every number traceable to a rung]
  receipt: true
  review: optional
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
Answer the scale-up question only as far as past scale transitions support it, and refuse plainly otherwise.

## When to use
Planning tech transfer or a pilot run.

## When not to use
For a single-scale sheet, or as engineering sign-off.

## Inputs
Multi-scale runs and a target specification.

## Procedure
1. Run the gates in order; stop at the first failure.
2. Backtest each smaller-to-larger rung.
3. Compute physics proxies with stated assumptions.
4. Decide: predict, predict with warning, or refuse.

## Outputs
`scale_readout` (predicted values plus diagnostics).

## Stop conditions
Any gate failure or an unsupported backtest.

## Caveats
Physics proxies use generic geometry assumptions unless vessel data is supplied. Engineering review is still required.

## Example
"Readout for moving recipe R12 from 2 L to 50 L with our 50 L vessel geometry."
