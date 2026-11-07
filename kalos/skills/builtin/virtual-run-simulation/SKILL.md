---
schema: kalos-skill/1
slug: virtual-run-simulation
name: Virtual run (kinetic simulation)
version: 0.1.0
kind: locked
category: simulation
summary: >-
  Use to simulate a fed-batch or batch culture with a simple kinetic model (Monod-type growth, substrate uptake,
  product formation) for a given recipe, to explore "what if" questions or to build synthetic test data. Every output
  is labelled SIMULATED. Not evidence about the real process, and never mixed with measured data.
keywords: [simulation, kinetic model, Monod, ODE, virtual run, synthetic data]
starting_prompt: Simulate titer over 14 days for this feed schedule with the default kinetic model, clearly labelled as simulated.
inputs:
  - {name: recipe, type: json, required: true, description: "Initial conditions, feed events and process settings."}
  - {name: model, type: enum, required: false, description: "Kinetic model from the library; default monod-basic."}
parameters:
  - {name: t_end_h, type: float, default: 336, effect: "Simulated duration in hours."}
  - {name: noise_cv, type: float, default: 0.0, effect: "Optional multiplicative noise for synthetic data; 0 gives the deterministic trajectory."}
outputs:
  - {name: trajectory, kind: table, label: simulated, min_bytes: 8, description: "Time course of biomass, substrate, product; every row carries data_type=simulated."}
  - {name: endpoints, kind: json, label: simulated, min_bytes: 2, description: "Final and peak values, with the model id and parameters used."}
tools: [engine.ode_model]
entrypoint: "kalos.core.mechanistic:monod_example"
requires_extras: []
implementation:
  status: partial
  engine:
    - {symbol: "kalos.core.mechanistic:ODEModel", ref: "kalos/core/mechanistic.py:60"}
    - {symbol: "kalos.core.mechanistic:monod_example", ref: "kalos/core/mechanistic.py:174"}
    - {symbol: "kalos.core.mechanistic:HybridSurrogate", ref: "kalos/core/mechanistic.py:130"}
  gaps:
    - "The only bound tool, engine.ode_model, runs the engine's batch Monod example (kalos/core/mechanistic.py:174 monod_example; recipe = [mu_max, Ks, S0]) and returns final biomass per recipe, labelled simulated. Feed events, the full trajectory table, noise_cv and a model library are not available yet; the skills.virtual_run tool named in the draft is not bound in v1."
    - "On main, a fed-batch simulator with feed events exists only in experiments/leadgene_pipeline/pipeline/sim.py (simulate :97, endpoint_titer :111), outside the package."
    - "Branch twin-virtual-runs @b5a2585 (not merged) adds the pieces this skill should call: Origin enum measured/simulated/synthetic (kalos/twin/origin.py:45), a seeded generator registry (kalos/twin/generators.py:202 register_generator, :388 generate), fit-path refusal of virtual rows, and POST/GET /api/campaigns/{id}/virtual-runs (kalos/portal/virtual_runs_api.py:85, :114). Re-point implementation refs there once merged; add parameter fitting to measured data as new code."
scope: read
side_effects: none
refuses_when:
  - Integration fails or produces non-finite values.
  - The caller asks to append simulated rows to a measured campaign.
verification:
  deterministic: [mass balance of the ODE closes within tolerance, data_type=simulated on every row, same inputs give same trajectory when noise_cv=0]
  receipt: true
  review: none
author: Kalos
owner: system
visibility: system
licence: {spdx: Apache-2.0}
source: kalos-engine
status: beta
created: 2026-11-05
updated: 2026-11-05
---
## Purpose
A labelled sandbox for reasoning and for testing the optimiser, kept strictly apart from evidence.

## When to use
Teaching, "what if" exploration, generating synthetic benchmark data.

## When not to use
To fill gaps in real data or to support decisions about real batches.

## Inputs
A recipe; optional model choice.

## Procedure
1. Map the recipe to model parameters and initial state.
2. Integrate; apply feed events.
3. Label every output as simulated; report model id and parameters.

## Outputs
`trajectory`, `endpoints` (simulated).

## Stop conditions
Integration failure; any attempt to merge with measured data.

## Caveats
Default parameters are generic. A model that has not been fitted to and checked against the user's process says nothing about it.

## Example
"Virtual run: glucose bolus on days 3, 5, 7; compare with no feed."
