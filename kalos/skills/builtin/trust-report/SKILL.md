---
schema: kalos-skill/1
slug: trust-report
name: Trust report and refusal reasons
version: 0.1.0
kind: locked
category: trust
summary: >-
  Use to say how far the campaign's model can be trusted this round, and why. It covers held-out ranking skill (grouped
  cross-validation), interval calibration, a baseline comparison, a promotion verdict against fixed gates, GP shape
  diagnostics, and every refusal with its engine reason. Not for producing proposals.
keywords: [reliability, verdict, calibration, cross-validation, refusal, trust, honesty]
starting_prompt: Can I trust this round's proposals? Give me the reasons either way.
inputs:
  - {name: campaign, type: campaign-ref, required: true, description: "Campaign after analysis."}
parameters:
  - {name: seed, type: int, default: 0, effect: "Seed for every random step (bootstrap, CV shuffles, restarts); recorded in the receipt."}
outputs:
  - {name: trust_report, kind: json, label: derived, min_bytes: 2, description: "Out-of-fold Spearman vs floor, calibration (or why unavailable), baseline comparison, promotion pass/fail with failed gates, GP shape notes, list of unmodelled aspects, refusals."}
tools: [engine.grouped_cv_report, engine.interval_calibration, engine.check_gates, engine.gp_shape_report]
entrypoint: "kalos.core.gates:check_gates"
requires_extras: [ml]
implementation:
  status: partial
  engine:
    - {symbol: "kalos.core.evaluation:grouped_cv_report", ref: "kalos/core/evaluation.py:103"}
    - {symbol: "kalos.core.evaluation:interval_calibration", ref: "kalos/core/evaluation.py:296"}
    - {symbol: "kalos.core.evaluation:group_mean_baseline_spearman", ref: "kalos/core/evaluation.py:499"}
    - {symbol: "kalos.core.gates:check_gates", ref: "kalos/core/gates.py:37"}
    - {symbol: "kalos.core.gp_shape:gp_shape_report", ref: "kalos/core/gp_shape.py:238"}
    - {symbol: "kalos.portal.analysis:_analyze_body", ref: "kalos/portal/analysis.py:201"}
  gaps:
    - "The reliability dict (kalos/portal/analysis.py:740) and promotion result (:875) exist inside the analysis result; a standalone, versioned trust-report document and plain-language refusal catalogue need new code."
scope: read
side_effects: none
refuses_when:
  - No analysis exists for the campaign yet.
verification:
  deterministic: [every verdict line cites the metric and threshold it came from, nothing in the report can upgrade an engine verdict, an LLM reviewer may only append concerns]
  receipt: true
  review: optional
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
Be the honesty layer. One page that says "use", "use with care" or "do not use", and gives the numbers behind it.

## When to use
Every time a batch is proposed or results are re-analysed. Show it next to every proposal.

## When not to use
As a release or quality decision. It only judges the model.

## Inputs
A campaign reference.

## Procedure
1. Read grouped cross-validation ranking (Spearman) and compare it with the floor.
2. Report calibration of uncertainty intervals, or why it could not be computed.
3. Compare with the group-mean baseline.
4. Apply promotion gates and list failed gates.
5. Add GP shape notes and anything not modelled.
6. List refusals verbatim from the engine.

## Outputs
`trust_report`.

## Stop conditions
No analysis available.

## Caveats
Cross-validation on small data is itself noisy; the report says how many groups it rests on.

## Example
"Explain why last round's model was marked 'use with care'."
