---
schema: kalos-skill/1
slug: dataset-licence-check
name: Dataset licence check
version: 0.1.0
kind: locked
category: governance
summary: >-
  Use before importing an external or public dataset (literature tables, public bioprocess datasets, vendor data) to
  record its licence and terms, and to decide whether the intended use (internal R&D, commercial product, redistribution) is
  allowed, needs an agreement, or is blocked. Not legal advice; unclear cases go to a person.
keywords: [licence, license, dataset, commercial use, terms, attribution, data governance]
starting_prompt: Can we use this public fed-batch dataset to benchmark Kalos for a paying customer?
inputs:
  - {name: dataset_manifest, type: json, required: true, description: "Source URL or citation, stated licence (SPDX id or text), access conditions, intended use."}
parameters:
  - {name: intended_use, type: enum, allowed: [internal-rnd, commercial-service, redistribution], default: internal-rnd, effect: "Use being assessed; stricter uses need more permissive terms."}
outputs:
  - {name: licence_decision, kind: json, label: derived, min_bytes: 2, description: "allowed / needs-agreement / blocked / unknown, the rule that fired, attribution text required, and a reminder that a person must confirm."}
tools: []
entrypoint: null
requires_extras: []
implementation:
  status: needs-new-code
  engine: []
  gaps:
    - "New rule table keyed by SPDX id (e.g. CC-BY-4.0 allowed with attribution; CC-BY-NC-* blocked for commercial-service; controlled access requires an agreement reference; unknown -> unknown). Concept re-implemented from Biomni's per-dataset licence table and commercial-mode switch (repo/license_info.md); no Biomni code needed."
scope: read
side_effects: none
refuses_when:
  - No licence information and no source are supplied.
verification:
  deterministic: [rule table versioned and hashed, every decision cites the rule id, unknown never maps to allowed]
  receipt: true
  review: required
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
Stop non-commercial or restricted data from quietly entering commercial work, and keep attribution obligations visible.

## When to use
Before any external dataset is uploaded to a tenant, used in a benchmark, or bundled in an export.

## When not to use
For the customer's own data.

## Inputs
A dataset manifest.

## Procedure
1. Normalise the licence to an SPDX id where possible.
2. Apply the rule table for the intended use.
3. Return the decision, the rule id and the attribution text; mark it for human confirmation.

## Outputs
`licence_decision`.

## Stop conditions
No licence and no source.

## Caveats
Licence text can be ambiguous, and terms of service can add limits beyond the licence. A person decides.

## Example
"Check a CC BY-NC 4.0 dataset for a commercial benchmark." (Expected result: blocked.)
