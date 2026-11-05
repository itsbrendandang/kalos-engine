---
schema: kalos-skill/1
slug: time-series-endpoint-extraction
name: Endpoint extraction from time series
version: 0.1.0
kind: locked
category: analytics
summary: >-
  Use to turn per-run time courses (viable cell density, viability, glucose, lactate, titer, pH, DO) into one row of
  endpoints per run (final and peak values, time-to-peak, integral of viable cell density, specific growth and productivity,
  phase slopes) ready for a run sheet. Not for sensor fault detection.
keywords: [time series, endpoints, IVCD, growth rate, qP, harvest, feature extraction]
starting_prompt: Turn these daily sampling sheets into one row per run with peak VCD, IVCD, final titer and specific productivity.
inputs:
  - {name: timecourses, type: table-file, required: true, label: measured, description: "Long format: run id, time, variable, value (units in header or column)."}
parameters:
  - {name: endpoints, type: str, default: "final,peak,time_to_peak,ivcd,mu_max,qp", effect: "Comma list of endpoints to compute."}
  - {name: harvest_time_h, type: float, required: false, effect: "Evaluate final values at this time instead of the last sample."}
outputs:
  - {name: endpoint_table, type: table, label: derived, description: "One row per run; each endpoint column names its source variable and method."}
  - {name: extraction_log, type: json, label: derived, description: "Per run: samples used, gaps, interpolation applied, endpoints not computable and why."}
tools: []
implementation:
  status: needs-new-code
  engine: []
  gaps:
    - "A generic trace featuriser exists only as an experiment (experiments/leadgene_pipeline/pipeline/feature_extraction.py:24 extract_features, :112 phase_and_rate_features). Its header says it was ported from another project, so check that project's licence before promoting it. Bioprocess endpoints (IVCD by trapezoid, mu_max, qP from titer vs IVCD) need new code. The bench-ports round plans to promote the leadgene simulator to kalos/bench/cho_sim.py (not committed at the time of writing); its trajectories make good synthetic test inputs."
scope: read
side_effects: none
refuses_when:
  - Fewer than three time points for a run (that run is skipped and listed, never imputed).
verification:
  deterministic: [IVCD on a synthetic constant-density series equals density times duration, endpoints reproducible, skipped runs listed]
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
Bridge sampling data and the run sheet, with every derived number traceable to the samples it used.

## When to use
Before building or extending a run sheet from culture sampling data.

## When not to use
For online sensor QC (planned skill `bioreactor-timeseries-qc`).

## Inputs
Long-format time courses with units.

## Procedure
1. Group by run and variable; sort by time; report gaps.
2. Compute the requested endpoints with stated methods (trapezoid integration, log-linear slope for growth rate, regression of titer on IVCD for qP).
3. Write one row per run, plus a log of anything not computable.

## Outputs
`endpoint_table`, `extraction_log`.

## Stop conditions
A run with too few points is skipped and listed.

## Caveats
Endpoint values depend on sampling frequency; the method and the samples used are recorded with each value.

## Example
"Extract endpoints at a 336 h harvest for all runs in `sampling_export.csv`."
