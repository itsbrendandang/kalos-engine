---
schema: kalos-skill/1
slug: identifier-anonymisation
name: Identifier anonymisation
version: 0.1.0
kind: locked
category: governance
summary: >-
  Use to replace run IDs, batch numbers, file and sheet names, and other identifier-like columns with stable salted
  aliases before data leaves the tenant (exports, support tickets, shared reports). Not encryption, and not a guarantee
  that the data cannot be re-identified.
keywords: [anonymise, pseudonymise, identifiers, privacy, aliases, salt]
starting_prompt: Anonymise identifiers in this export before I send it to our CRO.
inputs:
  - {name: table, type: table-file, required: true, label: measured, description: "Table or parsed upload to scrub."}
parameters: []
outputs:
  - {name: anonymised_table, kind: table, label: derived, min_bytes: 8, description: "Same data, identifier columns replaced by aliases; process values untouched."}
  - {name: alias_report, kind: json, label: derived, min_bytes: 2, description: "Which columns were aliased or dropped and why; never the alias key."}
tools: [engine.anonymizer, engine.anonymize_parsed]
entrypoint: "kalos.data.anonymizer:Anonymizer.anonymize_meta"
requires_extras: [portal]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.data.anonymizer:Anonymizer", ref: "kalos/data/anonymizer.py:94"}
    - {symbol: "kalos.portal.analysis_parts.anonymize:anonymize_parsed", ref: "kalos/portal/analysis_parts/anonymize.py:157"}
    - {symbol: "kalos.portal.analysis_parts.identifiers:is_identifier_name", ref: "kalos/portal/analysis_parts/identifiers.py:111"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - No salt is configured in production (aliases would be guessable).
verification:
  deterministic: [same input and salt give the same aliases, no original identifier string appears in output]
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
Share process data without sharing who, which batch, or which file.

## When to use
Before any export or external share.

## When not to use
As a security control for regulated personal data.

## Inputs
A table.

## Procedure
1. Detect identifier-like columns by name and pattern.
2. Replace them with salted, stable aliases, or drop them per policy.
3. Report what changed.

## Outputs
`anonymised_table`, `alias_report`.

## Stop conditions
Missing production salt.

## Caveats
Unusual recipes or dates can still identify a run; review before sharing outside the organisation.

## Example
"Alias `batch_no` and `operator` before export."
