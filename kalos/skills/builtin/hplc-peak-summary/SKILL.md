---
schema: kalos-skill/1
slug: hplc-peak-summary
name: HPLC peak summary
version: 0.1.0
kind: locked
category: analytics
summary: >-
  Use to summarise chromatograms exported from a chromatography data system (CSV or AIA/ANDI netCDF): baseline,
  peak detection in declared retention windows, areas, concentration from a calibration curve, and system-suitability
  numbers (resolution, tailing, plate count). Not a replacement for the CDS method of record.
keywords: [HPLC, UPLC, chromatogram, peak area, titer, Protein A, calibration, system suitability]
starting_prompt: Summarise these Protein A HPLC runs into titer per sample using the standards in the sequence.
inputs:
  - {name: chromatograms, type: file-set, required: true, label: measured, description: "One trace per injection; time and signal columns or AIA/CDF files."}
  - {name: method, type: json, required: true, description: "Peak windows, baseline region, calibration standards and their concentrations."}
parameters:
  - {name: min_prominence, type: float, default: 0.0, effect: "Ignore peaks below this prominence (signal units); 0 uses an automatic noise-based threshold."}
outputs:
  - {name: peak_table, kind: table, label: derived, min_bytes: 8, description: "Per injection and peak: retention time, area, height, width, concentration, flags."}
  - {name: suitability, kind: json, label: derived, min_bytes: 2, description: "Calibration fit (slope, intercept, R²), resolution, tailing, plate count, and pass/fail against the method's limits."}
tools: []
entrypoint: null
requires_extras: []
implementation:
  status: needs-new-code
  engine: []
  gaps:
    - "No chromatography code in kalos-engine. Build on scipy.signal (find_peaks, peak_widths) plus trapezoid integration; add AIA/CDF reading; calibration by least squares with residual checks."
    - "The bench-ports round plans a labelled synthetic chromatogram generator (kalos/bench/chromatogram.py, not committed at the time of writing); use it for this skill's deterministic tests."
scope: read
side_effects: none
refuses_when:
  - Calibration has fewer than three standard levels or fails its fit limit.
  - A declared peak window contains no peak above threshold (reported per sample, not imputed).
verification:
  deterministic: [synthetic Gaussian peaks recover known areas within tolerance, calibration residuals reported, every concentration traceable to its injection and standards]
  receipt: true
  review: none
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
Fast, transparent titer and purity numbers for development rounds, traceable to raw traces and standards.

## When to use
Screening and development analytics feeding a campaign.

## When not to use
For reportable results. The CDS method of record remains the authority.

## Inputs
Traces and a method description.

## Procedure
1. Baseline-correct using the declared region.
2. Detect and integrate peaks inside each window.
3. Fit the calibration; convert areas to concentrations.
4. Compute suitability metrics; compare with method limits.

## Outputs
`peak_table`, `suitability`.

## Stop conditions
Calibration failure; empty windows are flagged per sample.

## Caveats
Co-eluting peaks and drifting baselines are flagged, not resolved. Results can differ from CDS integration settings.

## Example
"Summarise sequence SEQ-0412 with the Protein A method file."
