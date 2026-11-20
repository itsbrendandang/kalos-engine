# Architecture

How the engine is layered, where new code goes, and what is still moving.
This page shows the **target** layout. Today most server code still sits in
`kalos/portal/`; the right-hand column of the table maps each layer to its
modules today. The moves happen one small PR at a time, behind module aliases
(see "Moving code" below).

## Request path

```
kalos-web (Next.js, server-side proxy)
  └─ per-request EdDSA engine token (docs/ENGINE_TOKEN.md) or a static service token
       └─ api      verify token, resolve tenant + scopes (route scopes: docs/route-scopes.json)
            ├─ campaigns  campaign rules, runs, results, gates      ─┐
            ├─ jobs       long fits in a local worker process        ├─ analysis ─ core (GP, CV, conformal)
            └─ assist     grounded assistant (no numbers of its own) ─┘     │
                                                                      ingest (upload parsing)
       storage: <KALOS_STATE_DIR>/portal.db (SQLite, schema v7 on main; migrations: docs/MIGRATIONS.md)
```

## Layers (lowest first)

A layer may import only itself and the layers above it in this list. `api`
is the only exception: it may import every layer below it. Nothing imports
`api` or `bench`.

| Layer | Owns | Target package | Today |
|---|---|---|---|
| util | ids, JSON helpers, small numeric helpers | `kalos/util/` | `portal/tenancy.py`, `portal/serialization.py` |
| data | anonymizer | `kalos/data/` | same |
| config | every `KALOS_*` read, deploy posture | `kalos/config/` | `kalos/envconfig.py`, `portal/config.py` |
| core | **the ML layer**: surrogate, optimizer, CV, conformal, gates, strain statistics | `kalos/core/` (+ `features/`, `versions.py`) | same, **not moved or renamed** |
| libs | domain libraries | `normalize/`, `domains/`, `validation/`, `scale/`, `providers/` | same |
| runtime | heavy-work admission, body-size caps, the state-dir lock | `kalos/runtime/` | `portal/busy.py`, `limits.py`, `engine_lock.py` |
| ingest | upload parsing and its guards, cell typing | `kalos/ingest/` | `portal/uploads.py`, `portal/cells.py` |
| storage | SQLite connection, migrations (one module per version), SQL repositories | `kalos/storage/` | `portal/migrate.py`, the SQL inside `CampaignStore` |
| analysis | one analysis of one sheet, no HTTP and no SQL | `kalos/analysis/` (`run.py`, `stages/{prepare,trust,propose,explain,assemble}.py`, `parts/`) | `portal/analysis.py` (`_analyze_body`, one 1,671-line function), `portal/analysis_parts/`, `pipeline.py`, `setup_settings.py`, ... |
| campaigns | campaign rules: start, record, correct, fold, reanalyze, stages and gates, clones, history | `kalos/campaigns/` (`records/` for projects and lineage) | `portal/campaign.py` (`CampaignStore`) and `portal/campaign_*.py`, `stages.py`, `clone_registry.py` |
| jobs | background job queue, runner, worker | `kalos/jobs/` | `portal/jobs.py`, `job_runner.py`, `job_worker.py`, `reanalyze_job.py` |
| assist | the grounded assistant, SOP drafts, retrieval | `kalos/assist/` | same, minus its two routers |
| api | app factory, middleware, error envelope, routers, auth, response schemas | `kalos/api/` | `portal/app.py`, `portal/*_api.py`, `*_routes.py`, `auth.py`, `jwt_auth.py`, `assist/*routes.py` |
| bench | benchmarks, planted-fault suite | `kalos/bench/` | same |
| kit | torch-free public facade | `kalos/kit/` | same |

`kalos/portal/` ends up as a compatibility package only: module aliases, the
`python -m kalos.portal` entrypoint and the job-worker spawn target. It is
removed one release after the last move.

## Where new code goes

- **A new statistic or model behaviour:** `kalos/core/`. It has no HTTP, no SQL and no campaign concepts. It is a flagged change with a golden diff.
- **A new block in the analysis response:** `kalos/analysis/` (today `portal/analysis_parts/`), as a function that `_analyze_body` calls.
  - Do not grow `_analyze_body` itself.
  - Keep every random-number consumer in its existing call order.
- **New campaign behaviour or a new per-campaign record:** `kalos/campaigns/`.
  - Its SQL goes in `kalos/storage/` (a repository function).
  - A schema change goes in `kalos/storage/migrations/vN.py`, plus `docs/MIGRATIONS.md`.
- **A new route:** a router module (`kalos/api/routers/<resource>.py`; until the API move, `kalos/portal/<resource>_api.py`).
  - The router parses, authorizes (`require_scope`), calls one service function and returns its result. It holds no domain logic.
  - Regenerate the contract files in the same commit: `scripts/export_scope_manifest.py`, `scripts/export_openapi.py`, `scripts/export_routes.py`.
- **A new `KALOS_*` setting:** read it through `kalos/config` (today `kalos/envconfig.py` helpers), never with a bare `os.environ` lookup in domain code. Document it in `docs/HARDENING.md` or the relevant page.
- **Tests:** at the root of `tests/` for now. They move into per-layer folders in one late PR. Contract tests live in `tests/contract/`.

## Guards (tests/contract/)

| Guard | Fails when |
|---|---|
| `test_openapi_drift.py` (`docs/openapi.json`) | a route, parameter or schema changed without regenerating the file |
| `test_route_inventory.py` (`docs/routes.json`) | a route changed, disagrees with `docs/route-scopes.json`, or lives outside a router module |
| `test_import_layers.py` (`scripts/import_layers.py`) | **report-only** for now: prints the edges that break the layer rules above against a baseline. Enforcing after the API move |
| `test_module_aliases.py` (`kalos/_moved.py`) | a moved module's old path is not the very same module object |
| golden oracle (`scripts/golden.py`, `.github/workflows/golden.yml`) | the engine's analysis or HTTP output on a fixed scenario differs byte for byte from the base commit's, computed on the same CI runner, and the PR does not declare the change in `tests/golden/CHANGES.md` (`scripts/golden.py record --reason`) |
| local golden records (`tests/golden/darwin-arm64-…/`, `scripts/golden.py check`) | **not a CI gate**: a fast pre-push check on an arm64 Mac with the `uv.lock` stack (Python 3.11); other machines skip it |

## Moving code

Every move is behaviour-preserving and lands in its own PR:
- `git mv`, with no edits in the moved file beyond import lines;
- an entry in `kalos._moved.MOVED`;
- a two-line shim left at the old path (`from kalos._moved import alias` / `alias(__name__)`). This makes the old path the same module object, so `monkeypatch.setattr(old_module, ...)` keeps working.

Each move PR must leave unchanged:
- the goldens;
- `docs/openapi.json`;
- `docs/route-scopes.json`;
- `docs/routes.json`, apart from the `module` field of the routes it moved;
- the collected test count.

Changes to numbers or wording are never mixed into a move.
