---
schema: kalos-skill/1
slug: column-role-inference
name: Column-role inference
version: 0.1.0
kind: locked
category: intake
summary: >-
  Use to propose what each run-sheet column is (process factor, categorical factor, response, run or batch identifier,
  grouping key, ignore) with the evidence for each guess, for the scientist to confirm. Not for checking values or fitting models.
keywords: [columns, roles, factors, response, identifier, grouping, confirm columns]
starting_prompt: Which columns in this sheet are factors, which is the response, and which are just identifiers?
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Parsed run sheet."}
parameters:
  - {name: profile, type: enum, allowed: [bioprocess, generic], default: bioprocess, effect: "Selects the domain vocabulary used to recognise headers."}
outputs:
  - {name: column_plan, type: json, label: derived, description: "Per column: proposed role, reason (header match, numeric parse rate, cardinality), and whether confirmation is required."}
tools: [engine.guess_role, engine.offline_plan, engine.column_provenance]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.normalize.synonyms:guess_role", ref: "kalos/normalize/synonyms.py:91"}
    - {symbol: "kalos.normalize.llm:offline_plan", ref: "kalos/normalize/llm.py:175"}
    - {symbol: "kalos.portal.analysis_parts.columns:_resolve_columns", ref: "kalos/portal/analysis_parts/columns.py:54"}
    - {symbol: "kalos.portal.analysis_parts.columns:_auto_group_column", ref: "kalos/portal/analysis_parts/columns.py:147"}
    - {symbol: "kalos.portal.validate:column_provenance", ref: "kalos/portal/validate.py:77"}
    - {symbol: "kalos.domains.profile:ColumnRoles", ref: "kalos/domains/profile.py:55"}
  gaps:
    - "The LLM-assisted planner (kalos/normalize/llm.py:553 propose_plan) is excluded from this locked skill; only the deterministic offline path is used."
scope: read
side_effects: none
refuses_when:
  - No numeric column exists that could serve as a response.
verification:
  deterministic: [every column receives exactly one role, identifier-like headers are never proposed as factors, plan is reproducible for identical input bytes]
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
Turn an arbitrary spreadsheet layout into an explicit column plan the scientist signs off, so the optimiser never treats a batch ID as a process factor.

## When to use
First contact with a new sheet layout, or when headers change.

## When not to use
To fix values or units (use `unit-normalisation`).

## Inputs
A parsed table. Headers can be messy; units in headers are recognised.

## Procedure
1. Classify each header deterministically (header vocabulary, numeric parse rate, distinct-value count).
2. Propose a grouping key when replicate or batch structure is detectable.
3. Mark low-confidence roles as "needs confirmation". Never auto-accept the response column.

## Outputs
`column_plan` with one role per column plus its reason.

## Stop conditions
No candidate response column: stop and ask the scientist.

## Caveats
Header matching is vocabulary-based. Unusual site naming needs confirmation, and confirmation is always the scientist's call.

## Example
"Propose roles for `ambr_screen_07.csv` and flag anything you are unsure of."
