---
schema: kalos-skill/1
slug: campaign-status
name: Campaign status
version: 0.1.0
kind: locked
category: optimisation
summary: >-
  Use to report where a campaign stands (Setup, Plan, Run, Record or Learn), what is in flight, what results are
  awaited, the best measured value so far, and the single next action. Not for changing the campaign.
keywords: [status, stage, round, in flight, next step, best so far]
starting_prompt: Where are we in the titer campaign and what should I do next?
inputs:
  - {name: campaign, type: campaign-ref, required: true, description: "Campaign to summarise."}
parameters: []
outputs:
  - {name: status, type: json, label: derived, description: "Stage, round, runs in flight, awaited results, best measured value (measured only), next action."}
tools: [engine.campaign_summary, engine.derive_stage]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.portal.campaign_stage:derive_stage", ref: "kalos/portal/campaign_stage.py:26"}
    - {symbol: "kalos.portal.campaign_stage:stage_fields", ref: "kalos/portal/campaign_stage.py:111"}
    - {symbol: "kalos.portal.campaign:CampaignStore", ref: "kalos/portal/campaign.py:203"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - Campaign not found for the caller's tenant.
verification:
  deterministic: [best-so-far computed from measured values only, stage matches derive_stage for the stored state]
  receipt: false
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
A one-glance answer to "what now?" that never confuses predicted with measured.

## When to use
Start of any conversation about a campaign.

## When not to use
To log results or start runs. Those are write actions, outside this skill.

## Inputs
A campaign reference.

## Procedure
1. Load the campaign summary.
2. Derive the stage and next action.
3. Report the best measured value and what is outstanding.

## Outputs
`status`.

## Stop conditions
Campaign not found.

## Caveats
Status reflects what has been logged; unlogged bench results are invisible.

## Example
"Status of campaign `mab-x-fb-2026`."
