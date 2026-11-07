---
schema: kalos-skill/1
slug: run-sheet-checks
name: Run-sheet checks
version: 0.1.0
kind: locked
category: data-quality
summary: >-
  Use before any model sees a run sheet (CSV, TSV or XLSX of past bioprocess runs) to list what is wrong with it:
  unit clashes, impossible values, duplicate runs, gaps, outliers, thin replication, missing controls.
  Not for picking the target column or proposing experiments.
keywords: [run sheet, data quality, upload, findings, QC]
starting_prompt: Check this run sheet and list everything that would make a model's output untrustworthy.
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "One row per run. Header in the first or second row."}
  - {name: target, type: column, required: false, description: "Response column, if already chosen; enables target-aware checks."}
parameters:
  - {name: mode, type: enum, allowed: [warn, strict], default: warn, effect: "strict refuses on any blocking finding; warn reports it and carries on."}
outputs:
  - {name: check_report, kind: json, label: derived, min_bytes: 2, description: "Each finding with its check name, severity, affected rows (capped, with the true count) and any unit conversion applied."}
tools: [engine.parse_upload, engine.validate_frame]
entrypoint: "kalos.validation.runner:validate_frame"
requires_extras: [portal]
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.portal.uploads:parse_upload", ref: "kalos/portal/uploads.py:1066"}
    - {symbol: "kalos.validation.runner:validate_frame", ref: "kalos/validation/runner.py:46"}
    - {symbol: "kalos.validation.report:build_report", ref: "kalos/validation/report.py:146"}
    - {symbol: "kalos.validation.checks:check_duplicate_rows", ref: "kalos/validation/checks.py:328"}
    - {symbol: "kalos.validation.checks:check_physical_bounds", ref: "kalos/validation/checks.py:234"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - The file does not parse into one rectangular table with a recognisable header.
  - mode is strict and at least one blocking finding exists.
verification:
  deterministic: [report parses into the engine ValidationReport shape, every finding names the check that raised it, capped row lists state the uncapped count]
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
Give the scientist a complete, row-referenced list of data problems before anything is fitted, so that a poor proposal can never be blamed on a silent data issue.

## When to use
- A new or edited run sheet arrives.
- Before re-analysis after results were logged.

## When not to use
- To decide which column is the response (use `column-role-inference`, then `target-readiness-check`).
- To judge model quality (use `trust-report`).

## Inputs
A single table, one run per row. Blank cells stay blank; they are never read as zero.

## Procedure
1. Parse the file and pick the header row; reject multi-table or ragged sheets.
2. Run the eleven engine checks (units, physical bounds, duplicates, missingness, outliers, provenance metadata, replicate adequacy, controls, constant columns, informative missingness, constant-within-group).
3. Return every finding; do not edit or drop data in this skill.

## Outputs
`check_report` (JSON). Row lists are capped for size and always carry the full count.

## Stop conditions
Stop and report if the table cannot be parsed. In strict mode, stop on the first blocking finding class and list all of them.

## Caveats
Checks find suspicious data; they cannot prove data is right. Outlier flags are statistical, not judgements that a run failed.

## Example
"Check `fedbatch_2026Q3.xlsx` (sheet `Runs`) in strict mode."
