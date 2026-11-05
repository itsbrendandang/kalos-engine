---
schema: kalos-skill/1
slug: bo-next-batch
name: Next-batch proposal (Bayesian optimisation)
version: 0.1.0
kind: locked
category: optimisation
summary: >-
  Use to propose the next batch of runs from the campaign's data: fit the Gaussian-process surrogate, score candidates
  with the acquisition function (feasibility-gated when infeasible runs exist), keep proposals distinct from each other and
  from in-flight runs, and return each recipe with predicted value, uncertainty and an explore/exploit reason.
  Not for first rounds without data, and never a replacement for the trust report.
keywords: [Bayesian optimisation, next batch, proposal, GP, acquisition, explore, exploit]
starting_prompt: Propose the next 4 runs to maximise titer, accounting for the 3 runs still in progress.
inputs:
  - {name: campaign, type: campaign-ref, required: true, description: "Campaign with checked data, confirmed roles and a search box."}
parameters:
  - {name: q, type: int, default: 4, effect: "Batch size."}
  - {name: seed, type: int, default: 0, effect: "Seed for reproducible optimisation restarts."}
  - {name: objective, type: enum, allowed: [maximise, minimise], default: maximise, effect: "Direction of the response."}
outputs:
  - {name: proposal_batch, kind: json, label: predicted, min_bytes: 2, description: "Recipes in engineering units with predicted mean, uncertainty, P(feasible) when relevant, rationale, and the minimum separation used."}
tools: [engine.propose_batch, engine.surrogate_fit]
entrypoint: "kalos.core.optimize:propose_batch"
requires_extras: [ml]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.core.optimize:propose_batch", ref: "kalos/core/optimize.py:408"}
    - {symbol: "kalos.core.optimize:ProposalBatch", ref: "kalos/core/optimize.py:318"}
    - {symbol: "kalos.core.surrogate:Surrogate", ref: "kalos/core/surrogate.py:140"}
    - {symbol: "kalos.core.feasibility:FeasibilityClassifier", ref: "kalos/core/feasibility.py:40"}
    - {symbol: "kalos.core.multiobjective:propose_multiobjective", ref: "kalos/core/multiobjective.py:95"}
    - {symbol: "kalos.portal.analysis_parts.proposals:_annotate", ref: "kalos/portal/analysis_parts/proposals.py:16"}
  gaps: []
scope: write
side_effects: writes-campaign
refuses_when:
  - target-readiness-check has refused.
  - The fit fails (FitError); no fallback recipe is invented.
verification:
  deterministic: [proposals inside bounds and constraints, pairwise separation >= min_separation, predicted values labelled predicted and never mixed with measured, run receipt with seed and data hash]
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
Suggest the most informative next experiments, and show exactly why each was chosen and how uncertain it is.

## When to use
Any round after the starting design, once the trust report allows it.

## When not to use
With no or very little data (`space-filling-design`), or to forecast a different scale (`scale-up-envelope-check`).

## Inputs
A campaign reference; the skill reads data, roles, search box and in-flight runs from the engine.

## Procedure
1. Fit the surrogate on checked, unit-normalised data.
2. Optimise the acquisition function; gate by feasibility when needed.
3. Reject near-duplicates of each other and of in-flight runs.
4. Attach prediction, uncertainty and rationale to each recipe.
5. Always present it together with `trust-report` and `distance-to-data`.

## Outputs
`proposal_batch` (predicted values only).

## Stop conditions
Readiness refusal or fit failure.

## Caveats
Predictions are model output, not measurements. The engine's methodology has not yet been tested against wet-lab outcomes.

## Example
"Propose 6 runs; keep temperature fixed at 36.5 °C."
