---
schema: kalos-skill/1
slug: unit-normalisation
name: Unit normalisation
version: 0.1.0
kind: locked
category: data-quality
summary: >-
  Use to read values written with units (for example "5 g/L", "37 °C", "1.2e6 cells/mL") into one base unit per column
  and to flag columns that mix incompatible units. Not for physical-plausibility checks.
keywords: [units, conversion, g/L, mM, temperature, normalise]
starting_prompt: Convert the units in this sheet to one unit per column and tell me where they clash.
inputs:
  - {name: run_sheet, type: table-file, required: true, label: measured, description: "Parsed run sheet with a confirmed column plan."}
parameters: []
outputs:
  - {name: unit_conversions, kind: json, label: derived, min_bytes: 2, description: "Per column: detected unit tokens, chosen base unit, conversion factor, unparsed cells."}
  - {name: normalised_table, kind: table, label: derived, min_bytes: 8, description: "Copy of the table in base units; the original is untouched."}
tools: [engine.parse_value, engine.convert_unit, engine.apply_unit_conversions]
entrypoint: "kalos.validation.runner:apply_unit_conversions"
requires_extras: []
implementation:
  status: implemented
  engine:
    - {symbol: "kalos.normalize.units:parse_value", ref: "kalos/normalize/units.py:134"}
    - {symbol: "kalos.normalize.units:convert", ref: "kalos/normalize/units.py:228"}
    - {symbol: "kalos.validation.runner:apply_unit_conversions", ref: "kalos/validation/runner.py:121"}
    - {symbol: "kalos.validation.checks:check_units_consistency", ref: "kalos/validation/checks.py:87"}
  gaps: []
scope: read
side_effects: none
refuses_when:
  - A column mixes dimensionally incompatible units (for example g/L and °C).
verification:
  deterministic: [round-trip of known unit strings, no value changes without a recorded conversion]
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
Make every number comparable before modelling, and leave a written record of each conversion.

## When to use
After column roles are confirmed and before the run-sheet checks in strict mode.

## When not to use
To decide whether a value is physically possible (`run-sheet-checks` covers bounds).

## Inputs
Confirmed table. Cells may carry unit text.

## Procedure
1. Parse each cell into number plus unit token.
2. Choose the base unit per column; convert; record factors.
3. Report unparsed cells and incompatible mixes. Do not guess.

## Outputs
`unit_conversions`, `normalised_table`.

## Stop conditions
Incompatible units in one column.

## Caveats
Only units in the engine's unit table are understood; anything else is reported, not converted.

## Example
"Normalise units in the uploaded sheet and show me every conversion you applied."
