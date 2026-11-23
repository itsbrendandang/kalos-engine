# Language-model evaluation

Which small local models should Kalos use for its language tasks? This folder
holds a fixed test set, the script that scores models on it through the
engine's own code paths, and the results. Results are in
[`RESULTS.md`](RESULTS.md).

This protocol, the test set and the selection rule below were committed
before the first scored run.

## What is scored

Every task runs through the engine code a user would hit: the cascade
(`kalos/providers/cascade.py`), the validators, and the offline fallbacks. What
is scored is what would be shown.

| Task | Engine path | Items | Correct when |
|---|---|---|---|
| Header mapping | `kalos.normalize.llm` (plan_validator, offline_plan) | 6 sheets, 51 columns | the plan role is an accepted role; the unit token is an accepted spelling |
| Setup suggestions | `kalos.assist.setup_suggest.suggest` | the same 6 sheets (target columns not scored) | the suggested role is an accepted role |
| Assistant answers | `kalos.assist.chat.ask` over the fixture campaign | 17 answerable questions | not a refusal, contains one alternative of each expected group, contains no forbidden string |
| Refusals | the same | 10 questions the record cannot answer | abstained, or the first sentence says the record does not hold it |
| Intent routing | `kalos.assist.tasks.classify_intent` | 16 questions | the intent label matches |
| Plain wording | `kalos.assist.tasks.plain_wording` | 6 engine messages | reported as the share the checks accepted (no gold wording) |

Sheets (`llm_cases.py`): the repo's two example sheets (`examples/synthetic_bioprocess`
with a target-derived `qc_grade` added, `examples/synthetic_scaleup`) and four
generated with a fixed seed: a messy fed-batch sheet (identity columns, units in
cell text, free-text notes), a CHO clone sheet (clone, passage, thaw vial,
viability, a quality tier), a microbial plate sheet (well, plate, strain, media
lot, OD), and a sheet shaped like a public Trichoderma fermentation sheet
(contamination, yield tier). All data is generated; none is measured.

The assistant fixture (`fixtures/assistant_campaign.json`, built by
`build_fixture.py`) is one real engine analysis of the first 60 rows of the
example sheet, with three started runs, two results and one correction. Every
expected answer is a value the engine computed and wrote in those records.
Retrieval is the assistant index's BM25 over those records plus `docs/*.md`.

Also measured, per configuration:

- **Citation validity**: raw (share of the models' sentences whose source
  labels exist) and shown (share of shown sources that exist in the record).
- **Hallucinated numbers**: raw (numbers the models wrote that their cited
  sources do not hold, caught by the checks) and shown (numbers that reached
  the user without being in the record: must be 0).
- **Latency** on this Mac: per item, including escalations, with every model
  loaded before timing (cold-load seconds are reported separately).

## Configurations

| Config | Fast tier | Quality tier |
|---|---|---|
| `offline` | none: the engine's own deterministic paths (the floor, not a candidate) | none |
| `phi4-mini` | phi4-mini (MIT) | phi4-mini |
| `qwen2.5:7b` | qwen2.5:7b (Apache-2.0) | qwen2.5:7b |
| `qwen2.5:14b` | qwen2.5:14b (Apache-2.0) | qwen2.5:14b |
| `cascade` | phi4-mini | qwen2.5:7b |
| `cascade-14b` | phi4-mini | qwen2.5:14b |

A single-model config runs that model once per item (no retry of the same
model); the cascades escalate fast to quality on a failed check. Only MIT or
Apache-2.0 models are candidates.

## Selection rule (fixed before the run)

1. A configuration is eligible only if no hallucinated number was shown and
   every shown citation exists.
2. Its score is the mean of six accuracies: header roles, header units, setup
   roles, answers, refusals and intent.
3. The eligible model configuration with the highest score wins. When two are
   within 2.0 points, the one with the lower total time wins.
4. The winner's fast and quality models become the defaults
   (`KALOS_LLM_FAST_MODEL`, `KALOS_LLM_QUALITY_MODEL`).

`offline` is reported as the floor: where it beats the models, `RESULTS.md`
says so.

## Run it

Needs Ollama with the models pulled (`ollama pull phi4-mini qwen2.5:7b qwen2.5:14b`).

```bash
.venv/bin/python benchmarks/llm/run_eval.py                      # all configs, all tasks
.venv/bin/python benchmarks/llm/run_eval.py --configs cascade --tasks assistant intent
KALOS_TORCH_THREADS=1 .venv/bin/python benchmarks/llm/build_fixture.py   # rebuild the fixture
```

The script writes `results/eval_<date>.json` (every item, with the raw model
replies) and prints the table. Models run at temperature 0 with a fixed seed,
but Ollama does not promise bit-identical replies across versions or machines.
