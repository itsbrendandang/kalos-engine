---
schema: kalos-skill/1
slug: campaign-provenance-report
name: Campaign export and provenance report
version: 0.1.0
kind: locked
category: provenance
summary: >-
  Use to export a campaign as a self-describing bundle: the data as used (with hashes), every round's proposals,
  measured results, trust reports, engine build, parameters, seeds, skill versions, and the audit trail, so someone else can see
  exactly what was done and reproduce it. Not a regulatory submission document.
keywords: [export, provenance, audit trail, reproducibility, receipt, hash, report]
starting_prompt: Export round 1 to 4 of this campaign with full provenance for our tech-transfer file.
inputs:
  - {name: campaign, type: campaign-ref, required: true, description: "Campaign to export."}
parameters:
  - {name: anonymise, type: bool, default: true, effect: "Replace identifiers and file names with stable aliases in the bundle."}
  - {name: format, type: enum, allowed: [zip-json-csv, pdf-summary], default: zip-json-csv, effect: "Machine bundle or a human summary that references the bundle hash."}
outputs:
  - {name: bundle, type: file, label: derived, description: "Archive with manifest.json (sha256 of every file), data, proposals (predicted), results (measured), trust reports, run records, audit entries."}
  - {name: provenance_summary, type: json, label: derived, description: "Engine build sha, skill ids and content hashes, seeds, timestamps, who did what."}
tools: [engine.run_record_build, engine.audit_entries, engine.campaign_summary]
implementation:
  status: partial
  engine:
    - {symbol: "kalos.portal.run_record:RunRecord", ref: "kalos/portal/run_record.py:74"}
    - {symbol: "kalos.portal.run_record:build", ref: "kalos/portal/run_record.py:152"}
    - {symbol: "kalos.portal.run_record:build_sha", ref: "kalos/portal/run_record.py:67"}
    - {symbol: "kalos.portal.campaign:CampaignStore", ref: "kalos/portal/campaign.py:203"}
    - {symbol: "kalos.portal.campaign_stage:derive_stage", ref: "kalos/portal/campaign_stage.py:26"}
  gaps:
    - "Run records and audit entries exist (CampaignStore.audit_entries, campaign.py:949). The export bundle, manifest and PDF summary need new code. Skill content hashes must be added to run records once the skills framework lands."
    - "Branch gxp-integrity @7ea446c (not merged) adds proposal_provenance (kalos/portal/run_record.py:263 on that branch: engine and library versions, seed, input and config SHA-256, timestamp); the bundle should include it."
scope: read
side_effects: none
refuses_when:
  - The caller's tenant does not own the campaign (reported as not found).
verification:
  deterministic: [manifest hashes verify on re-read, measured and predicted values stored in separate files, bundle hash recorded in the audit log]
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
Make every campaign explainable and reproducible outside Kalos.

## When to use
End of a campaign, before tech transfer, or on request from QA or a partner.

## When not to use
As a signed record. Electronic signatures are not part of Kalos today.

## Inputs
A campaign reference.

## Procedure
1. Collect data as analysed, per-round proposals, results, trust reports and run records.
2. Optionally alias identifiers.
3. Write a manifest of sha256 hashes; package; record the bundle hash in the audit log.

## Outputs
`bundle`, `provenance_summary`.

## Stop conditions
Ownership check fails.

## Caveats
The bundle shows what Kalos did. It does not show that the process or method is fit for regulated use.

## Example
"Export the campaign anonymised, as JSON+CSV."
