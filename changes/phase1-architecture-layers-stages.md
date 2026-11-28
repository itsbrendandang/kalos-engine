### Changed - Phase 1 architecture: analysis stages, enforced import layers, unified providers

- **Analysis stage split (`kalos/analysis/stages/{prepare,trust,propose,explain,assemble}.py`, `kalos/portal/analysis.py`)**:
  - Extracted the monolithic `_analyze_body` into five ordered, torch-free-at-import stage modules preserving the exact RNG call order and golden outputs.
- **Zero import-layer violations (`scripts/import_layers.py`, `tests/contract/test_import_layers.py`, `tests/contract/import_layers_baseline.json`)**:
  - Eliminated all 8 import-layer violations and unplaced modules (`kalos._moved`, `kalos/providers/info.py`, `kalos/campaigns/{publish,setup}.py`, `kalos/storage/job_queries.py`, `kalos/portal/config.py`), cleared the baseline to 0 edges, and enabled `ENFORCE = True`.
- **Provider unification, Cascade routing & job hooks (`kalos/providers/{ollama_provider,grounding}.py`, `kalos/normalize/llm.py`, `kalos/assist/{chat,sop,swarm,hooks}.py`, `kalos/portal/{job_runner,app}.py`)**:
  - Consolidated Ollama HTTP health-check/generate calls onto `kalos/providers/ollama_provider.py` and number extraction onto `kalos/providers/grounding.py`.
  - Routed `swarm.review()` and `sop.revise()` through `Cascade.run()`, deferred POSIX `fcntl` in `swarm.py`, and wired an `on_job_done` hook from `app.py` into `JobRunner` -> `assist.hooks.after_job`.
