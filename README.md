# Kalos engine


> **Status: under development. Not a public SaaS.**
> Kalos is a scientific optimization platform being built toward a public SaaS.
> There is no hosted service today. User login and organisation management are under development and are **not** on `main`.
> The optimizer is **not scientifically validated** (see [Scientific methodology](#scientific-methodology-and-benchmark-limitations)).
> Nothing here claims enterprise readiness, regulatory compliance, or security certification.

## Overview

Kalos proposes the next batch of experiments for a bioprocess campaign.
You give it a run sheet (CSV, TSV, or Excel) of past runs and a target column, such as titer or yield.
It checks the data, fits a Gaussian-process surrogate (BoTorch), and returns a proposed batch of recipes plus the diagnostics needed to judge how far to trust them.
You then run those experiments, log the measured results, and ask for the next batch.
A separate scale-up readout predicts performance at a larger scale, and refuses when its own backtest does not support a prediction.

This repository is the **engine**: a FastAPI service with no built-in UI.
The workbench UI is the separate [kalos-web](https://github.com/itsbrendandang/kalos-web) repo.

## Target users and use cases

Intended users (the product is not yet available to external users):

- **Bioprocess scientists and process-development teams** with a spreadsheet of runs (media, feeds, pH, temperature) who want a suggested next round and an honest read on what the data can support.
- **Small teams running iterative campaigns** (propose, measure, re-propose) with a few dozen to a couple of thousand runs. The exact GP is capped at 2,000 fit rows (`KALOS_MAX_FIT_ROWS`).
- **Platform or ML engineers** who want a tenant-aware optimization API they can self-host.

Use cases: next-batch proposals for media or process optimisation; tracking an in-flight campaign; a scale-up readout from a multi-scale run sheet.
Not a use case yet: regulated (GxP) work, or anything that depends on the optimizer being validated against real wet-lab outcomes.

## Capabilities

Labels: **IMPLEMENTED** = on `main` with tests; **EXPERIMENTAL** = on `main` or in an open PR but not validated, or opt-in; **PLANNED** = not built.
Open PRs are listed as in-progress, not as implemented.

| Capability | Status | Evidence |
| --- | --- | --- |
| Upload CSV/TSV/Excel, validate, analyse, propose next batch (`POST /api/run`) | IMPLEMENTED | `kalos/portal/app.py`, `docs/UPLOADS.md`, `tests/test_validation_gate.py` |
| GP surrogate (BoTorch), mixed continuous + categorical spaces | IMPLEMENTED | `kalos/core/`, `tests/test_data_and_core.py` |
| Grouped CV, split-conformal bands, fail-closed promotion gates | IMPLEMENTED | `kalos/core/splits.py`, `conformal.py`, `gates.py`; `tests/test_calibration.py`, `tests/test_promotion_and_baselines.py` |
| Data validation gate (units, bounds, duplicates, missingness) | IMPLEMENTED | `kalos/validation/`, `tests/test_validation_gate.py` |
| Propose-measure-re-propose campaign loop with in-flight points | IMPLEMENTED | `kalos/portal/campaign_routes.py`, `docs/CAMPAIGN_LOOP.md`, `tests/test_pending_proposals.py` |
| Scale-up readout that refuses when its backtest fails | IMPLEMENTED (synthetic evidence only) | `kalos/portal/scale_routes.py`, `docs/SCALE_READOUT.md`, `docs/SCALE_EVIDENCE.md` |
| Bearer-token auth with `read`/`write`/`admin` scopes, hashed tokens | IMPLEMENTED | `kalos/portal/auth.py` |
| Tenant-keyed campaign (SQLite) and latest-analysis store | IMPLEMENTED | `kalos/portal/campaign.py`, `docs/HARDENING.md` Phase 1b |
| Upload caps, loopback-only default, remote refusal when open | IMPLEMENTED | `kalos/portal/uploads.py`, `__main__.py`, `app.py` middleware |
| Container deploy pack | IMPLEMENTED (limited verification, see runbook; no CI runs today, use `scripts/check.sh`) | `deploy/` |
| Optimizer beats LHS/random on synthetic surfaces | IMPLEMENTED as a benchmark; not validation | `BENCHMARK.md`, `python -m kalos.bench` |
| Head-to-head vs random, LHS, Optuna TPE, Ax, vanilla GP-EI on synthetic tasks | IMPLEMENTED as a benchmark; not validation | [Benchmark vs other optimizers](#benchmark-vs-other-optimizers), `docs/BENCHMARK.md`, `benchmarks/results/` |
| Mechanistic (Monod) prior mean | EXPERIMENTAL, open PRs [#76](https://github.com/itsbrendandang/kalos-engine/pull/76), [#81](https://github.com/itsbrendandang/kalos-engine/pull/81) | not on `main` |
| Pool-replay benchmark on published media datasets | EXPERIMENTAL, open PR [#77](https://github.com/itsbrendandang/kalos-engine/pull/77) | not on `main` |
| Hard linear constraints on proposals | EXPERIMENTAL, open PR [#78](https://github.com/itsbrendandang/kalos-engine/pull/78) | not on `main` |
| `feature_ranges` in analysis output; GP-default guardrail | in progress, open PRs [#80](https://github.com/itsbrendandang/kalos-engine/pull/80), [#79](https://github.com/itsbrendandang/kalos-engine/pull/79) | not on `main` |
| Repo hygiene (CODEOWNERS, SECURITY.md, PR template) and loopback-by-default compose bind | in progress, open PR [#83](https://github.com/itsbrendandang/kalos-engine/pull/83) (merge conflicts with `main`) and draft PR [#84](https://github.com/itsbrendandang/kalos-engine/pull/84) | not on `main` |
| Verifying short-lived internal JWTs from the web app (EdDSA), engine side | IMPLEMENTED (kalos-web does not issue them yet) | `kalos/portal/jwt_auth.py`, `docs/ENGINE_TOKEN.md`, `tests/test_jwt_auth.py`; PRs [#88](https://github.com/itsbrendandang/kalos-engine/pull/88), [#96](https://github.com/itsbrendandang/kalos-engine/pull/96) |
| Organisations as tenants, with membership | PLANNED | the engine only reads a `tenant` string (from an operator-provisioned token or an engine JWT claim); no organisation model or membership exists in the engine |
| Multiple campaigns per tenant | IMPLEMENTED | `kalos/portal/campaigns_api.py`, `docs/CAMPAIGNS_API.md`, `tests/test_cross_tenant_isolation.py`; PRs [#89](https://github.com/itsbrendandang/kalos-engine/pull/89), [#90](https://github.com/itsbrendandang/kalos-engine/pull/90) |
| Background analysis jobs with a local worker process (`/api/campaigns/{id}/jobs`) | IMPLEMENTED | `kalos/portal/jobs.py`, `job_runner.py`, `job_worker.py`, `docs/JOBS.md`, `tests/test_jobs.py`, `tests/test_jobs_api.py`; PR [#91](https://github.com/itsbrendandang/kalos-engine/pull/91). The synchronous routes still answer 503 busy when the analysis slot is taken |
| Sparse/variational surrogate beyond 2,000 rows | PLANNED | `docs/HARDENING.md` Phase 2 |
| Metrics, structured logging, alert hooks | PLANNED | `docs/HARDENING.md` Phase 3 |
| Scientific validation on independent real wet-lab data | PLANNED | not done; see below |

## Architecture

Solid arrows and boxes exist on `main`. Dashed arrows and boxes are planned and not implemented.

```mermaid
flowchart LR
  B[Browser] --> W["kalos-web: Next.js UI + server-side proxy"]
  W -->|"static bearer token (today)"| E["kalos-engine: FastAPI"]
  E --> S[("SQLite: per-tenant campaigns")]
  E --> M["GP surrogate (BoTorch)"]
  E --> V["EdDSA JWT verification (built; kalos-web does not issue yet)"]
  E --> J["local job workers"]
  B -.->|"planned: login"| A["Better Auth session in web"]
  A -.-> W
  W -.->|"planned: EdDSA JWT, iss kalos-web, aud kalos-engine, 120s max"| E
  E -.-> O["planned: organisations as tenants, membership"]
  classDef planned stroke-dasharray: 5 5,color:#666;
  class A,O planned;
```

Today the browser talks to the Next.js app, whose server-side proxy forwards to this engine using one configured bearer token.
The web app has **no user login**, so anyone who can reach the proxy acts as that token's tenant.
Built on the engine side: verification of short-lived EdDSA-signed internal JWTs (issuer `kalos-web`, audience `kalos-engine`, lifetime of at most 120 seconds) that the engine checks on its own, multiple campaigns per tenant, and background analysis jobs run by local worker processes.
Still planned: the Better Auth session and JWT issuance in the web app, and organisations as tenants with membership.
Current and planned request flows: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Campaign loop details: [`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md).

## Local development

Python 3.11+ (CI uses 3.12). All tools are free and open source.

```bash
python -m pip install -e .               # core, torch-free
python -m pip install -e ".[ml,portal]"  # + BoTorch and the FastAPI service
python -m kalos.portal                   # http://127.0.0.1:8050 (KALOS_HOST / KALOS_PORT to change)
```

The core install is torch-free (`numpy`, `pandas`, `scikit-learn`, `scipy`), enough for `kalos.kit`.
In a second shell, upload the synthetic example (generated by `examples/synthetic_bioprocess/make_synthetic.py`, not measured data):

```bash
curl -F file=@examples/synthetic_bioprocess/synthetic_bioprocess.csv http://127.0.0.1:8050/api/run
curl http://127.0.0.1:8050/api/latest
```

Both commands were run against a fresh install of this branch's code (`/healthz` ok; the upload returned a 10-feature analysis for 160 rows).
State is written to `~/.kalos` (override with `KALOS_STATE_DIR`).
With no tokens configured the API is open on loopback only; see Security.

Checks (same as CI):

```bash
python -m pip install -e ".[ml,portal,bench,dev]"
ruff check kalos/ tests/ examples/
mypy
python -m pytest -q    # 804 tests collected at the time of writing
```

## API

Routes are defined in `kalos/portal/app.py`, `campaign_routes.py`, and `scale_routes.py`.
Scopes come from `kalos/portal/auth.py`.
Every tenant-scoped route reads the tenant from the authenticated token (`Principal.tenant`), never from the request.
In open mode (no tokens configured, loopback only) every caller is an anonymous `default`-tenant principal with `read`+`write`.

| Method | Path | Scope | Tenant scoping | What it does |
| --- | --- | --- | --- | --- |
| POST | `/api/run` | write | publishes result under caller's tenant | Upload CSV/TSV/Excel; returns analysis and proposed batch ([upload guards](docs/UPLOADS.md)) |
| GET | `/api/latest` | read | per tenant | Most recent analysis |
| GET | `/api/campaign` | read | per tenant (SQLite row) | Current campaign state |
| POST | `/api/campaign/start` | write | per tenant | Start a batch of proposed runs |
| POST | `/api/campaign/result` | write | per tenant | Log a measured outcome |
| POST | `/api/campaign/reanalyze` | write | per tenant | Fold results into the dataset and re-propose |
| POST | `/api/scale/readout` | write | none stored; request-scoped | Scale-up readout: prediction with interval, or refusal |
| GET | `/api/single` | read | none (built-in demo surface) | Single-objective demo loop |
| GET | `/api/multi` | read | none (built-in demo surface) | Multi-objective demo loop |
| GET | `/api/providers` | read | none | Credential status of optional provider slots (not the credentials) |
| GET | `/healthz` | none | none | Liveness |
| GET | `/readyz` | none | none | Readiness (store check) |

Contracts: [`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md), [`docs/SCALE_READOUT.md`](docs/SCALE_READOUT.md), [`docs/UPLOADS.md`](docs/UPLOADS.md).
FastAPI also serves its generated OpenAPI docs at `/docs` when the service runs.
On `main`, `/readyz` returns a fixed generic reason and never exception text (`kalos/portal/app.py`).

## Security model

What exists today (`kalos/portal/auth.py`, `config.py`, `uploads.py`, `app.py`):

- Bearer tokens are stored as SHA-256 hashes, compared in constant time, read per request so they can be rotated without a restart. Scopes: `read`, `write`, `admin`.
- Once any token is configured, auth is enforced on every scoped route. There is no half-authenticated state.
- Campaign and latest-analysis state are keyed by tenant; the latest-analysis cache filename is sanitised against path traversal.
- Upload caps: `KALOS_MAX_UPLOAD_MB` (default 25), 100,000 CSV rows, 512 columns, 2,000,000 xlsx cells, 2,000 GP fit rows.
- Loopback guard: `python -m kalos.portal` binds 127.0.0.1 and refuses to serve unauthenticated on another interface. A middleware also refuses non-loopback peers while auth is not enforced, unless `KALOS_ALLOW_OPEN_ACCESS` is set.
- CORS is an explicit allowlist when `KALOS_CORS_ORIGINS` is set; otherwise a localhost-only default.
- Anonymiser for uploads is opt-in (`anonymize=true`).

What is missing or limited:

- **The web app has no user login, and its proxy is unauthenticated.** Do not expose kalos-web or this engine to the public internet.
- Tenants are strings in operator-provisioned tokens. There is no user directory, organisation membership, or self-service token issuance.
- The engine verifies EdDSA engine tokens (`kalos/portal/jwt_auth.py`), but kalos-web does not issue them yet; the web app still sends one static bearer token.
- No rate limiting, audit log, or encryption at rest. TLS is out of scope; use a reverse proxy.
- `/api/single`, `/api/multi`, and `/healthz` are not tenant-scoped (they hold no tenant data).
- The deploy pack has been exercised once end to end; its runbook still lists unverified items (backup drill).
- No independent security review or penetration test has been done. No compliance claims are made (HARDENING.md puts 21 CFR Part 11, GAMP 5, SOC 2, and ISO 27001 out of scope for this repo).

Report vulnerabilities privately; a `SECURITY.md` is proposed in open PR #83.

## Scientific methodology and benchmark limitations

**The optimizer is not scientifically validated.** No claim is made that it improves real wet-lab outcomes.

- **Method.** GP surrogate with expected-improvement-family acquisition (BoTorch), grouped CV, split-conformal intervals, fail-closed promotion gates.
- **Synthetic benchmarks (on `main`).** The `BENCHMARK.md` result below is on synthetic surfaces with known optima, which the optimizer is free to exploit. A noise-free 2-D simulation is not validation of anything about real processes.
- **Hybrid dataset (not shipped).** A Kaggle "Fermentation optimization" dataset (`dataset_0.csv`: 696 of 715 rows flagged synthetic, 19 real, Real flag unverified) is used by `examples/real_data_validation.py` as a smoke test only. The data is kept private and is not in this repository: set `KALOS_DATASET_DIR` to a folder containing it (see `examples/datasets/README.md`); tests skip when it is absent. A pool-replay result on it was weak (Spearman 0.048), so it is not evidence.
- **Amii published-media benchmark (open PR [#77](https://github.com/itsbrendandang/kalos-engine/pull/77)).** Pool replay on published media datasets. After review the author retracted the earlier "2 faster / 2 slower" summary. A paired sign test with Holm correction found no detectable difference from random in a smoke run (8 seeds). Caveats: the BO arm is pool EI rather than qLogNEI, pools come from earlier optimisation campaigns (random is strong), one dataset mixes information sources, objectives are growth/absorbance not titer, the dataset licence is unresolved, and a committed 20-seed results file is still missing.
- **Mechanistic prior benchmark (on `main`, `examples/mechanistic_benchmark.py`).** Monod ODE prior mean in simulation, 10 seeds, 95% of the best *reachable* response (the held-out split is excluded from the target). The oracle and mild priors reached it sooner than the plain GP in 9 of 10 seeds (9-0, one tie; sign test p = 0.004, about 0.016 after correcting over the four levels). Severe and adversarial priors show no reliable benefit. Caveats: noise-free 2-D simulation whose truth shares the ODE's structure, no calibration check. The prior is opt-in and off by default.
- **Scale-up readout.** Evidence is from a synthetic simulator (`docs/SCALE_EVIDENCE.md`); the refusal rule is the safeguard, not a validation of predictions.

Synthetic benchmark from [`BENCHMARK.md`](BENCHMARK.md), simple regret after 18 experiments, 10 seeds (lower is better). Reproduced with `python -m kalos.bench` on the locked stack (torch 2.13.0, BoTorch 0.18.1):

| Surface (synthetic) | Noise | BO regret | LHS regret | Random regret |
| --- | --- | --- | --- | --- |
| Gaussian bump (4d, smooth) | 0% | 0.013 | 2.247 | 3.171 |
| Gaussian bump (4d, smooth) | 15% | 1.873 | 2.247 | 2.592 |
| Ackley (4d, rugged) | 0% | 1.560 | 3.625 | 3.696 |
| Ackley (4d, rugged) | 15% | 3.014 | 3.625 | 3.850 |

BO's margin over LHS shrinks as noise rises (Gaussian 2.23 to 0.37; Ackley 2.07 to 0.61) but stays positive in these four cells, for 10 seeds.
The BO rows depend on the torch build: torch 2.14.1 gives different BO regrets on the same seeds (Gaussian 15% 2.030, Ackley 15% 2.974). LHS and random do not change.
These numbers are not a comparison against other optimizers; see [Benchmark vs other optimizers](#benchmark-vs-other-optimizers).

### Built-in safeguards

- Cross-validation is grouped so a recipe never straddles train and validation: `kalos/core/splits.py`, with the leakage tripwire exercised in `tests/test_data_and_core.py`.
- Promotion gates fail closed, and a failing gate blocks promotion: `kalos/core/gates.py`, `tests/test_promotion_and_baselines.py`.
- Predictions carry distribution-free split-conformal bands: `kalos/core/conformal.py`, `tests/test_calibration.py`.
- A data validation gate checks units, physical bounds, duplicates, missingness, and more before any column is used, and proposals stay physically possible: [`docs/UPLOADS.md`](docs/UPLOADS.md), `tests/test_validation_gate.py`.
- Runs that are started but not yet measured are passed to the acquisition as in-flight points, so a mid-round re-proposal does not repeat them: `tests/test_pending_proposals.py`, `python -m kalos.bench --pending`, [`BENCHMARK.md`](BENCHMARK.md).
- The scale-up readout refuses to predict when its backtest on your own scale ladder does not beat two naive baselines: [`docs/SCALE_READOUT.md`](docs/SCALE_READOUT.md), `tests/test_scale_readout.py`.
- Every analysis response carries `seed` and `engine_version`, and the analyze path is seeded so the same upload gives the same proposals: `tests/test_hardening.py`.

## Benchmark vs other optimizers

Kalos's proposal path against random, Latin hypercube (LHS), Optuna TPE, Ax (default generation strategy) and a plain BoTorch GP-EI, on synthetic tasks with known optima.
The protocol (tasks, seeds, budgets, batch and pending settings, endpoints, statistics) was committed in [`docs/BENCHMARK.md`](docs/BENCHMARK.md) before the run. Raw results are in [`benchmarks/results/`](benchmarks/results/).

**Setup.** 30 seeds (1000-1029) per cell, with bootstrap intervals over seeds.
Three synthetic cells use a 4-D Gaussian bump and a 4-D Ackley, each at 0% and 15% noise: sequential (batch 1), 5 initial points and 18 experiments.
One mixed "bioprocess-like" surface (3 numeric and 2 categorical factors, 15% noise; this is a made-up surface, not process data) uses batches of 4: 8 initial points and 20 experiments, with 2 points of each batch still in flight while the next batch is proposed.
Locked stack: torch 2.13.0, BoTorch 0.18.1, Optuna 5.0.0, Ax 1.3.1 (`uv.lock` for the core pins). Apple M3 Pro, 11 cores, about 16 minutes wall time.

**Result.** Normalised regret of the best queried (true) value after the budget: 0 is the optimum, 1 is the median random point, lower is better. Mean over 30 seeds, 95% bootstrap interval in brackets.
The last part of each baseline cell is the paired call against Kalos: "Kalos better" when the 95% interval of the paired difference excludes zero; otherwise "tie / unresolved".

| Method | Gaussian 0% | Gaussian 15% | Ackley 0% | Ackley 15% | Bioprocess-like 15% (batch 4) |
|---|---|---|---|---|---|
| **Kalos** | **0.002** [0.001, 0.002] | **0.186** [0.120, 0.257] | **0.246** [0.184, 0.309] | **0.521** [0.471, 0.571] | **0.217** [0.171, 0.264] |
| Random | 0.333 [0.271, 0.392]: Kalos better | 0.333 [0.273, 0.393]: Kalos better | 0.605 [0.565, 0.644]: Kalos better | 0.605 [0.565, 0.643]: Kalos better | 0.440 [0.380, 0.498]: Kalos better |
| LHS | 0.355 [0.304, 0.406]: Kalos better | 0.355 [0.301, 0.407]: Kalos better | 0.618 [0.583, 0.653]: Kalos better | 0.618 [0.584, 0.653]: Kalos better | 0.459 [0.404, 0.516]: Kalos better |
| Optuna TPE | 0.133 [0.107, 0.161]: Kalos better | 0.161 [0.126, 0.202]: tie / unresolved | 0.450 [0.413, 0.485]: Kalos better | 0.541 [0.495, 0.581]: tie / unresolved | 0.326 [0.253, 0.401]: Kalos better (borderline) |
| Ax (default) | 0.002 [0.001, 0.002]: tie / unresolved | 0.191 [0.136, 0.253]: tie / unresolved | 0.284 [0.225, 0.346]: tie / unresolved | 0.522 [0.462, 0.582]: tie / unresolved | 0.325 [0.271, 0.379]: Kalos better |
| Vanilla GP-EI (BoTorch) | 0.002 [0.001, 0.003]: tie / unresolved | 0.206 [0.143, 0.273]: tie / unresolved | 0.296 [0.238, 0.357]: tie / unresolved | 0.526 [0.467, 0.583]: tie / unresolved | 0.276 [0.222, 0.332]: Kalos better (borderline) |

What the table says, and what it does not:

- **Random and LHS are the weakest rows in every cell.** Kalos's paired interval against each of them excludes zero in all five cells. Every other model-based method also has a lower mean than both, in every cell.
- **Clean synthetic (Gaussian 0%):** Kalos, Ax and vanilla GP-EI are not separated. All three reach 90% of the optimum in every seed, at about 13 experiments on average. Random, LHS and Optuna TPE reach it in 7-40% of seeds.
- **Noisy synthetic (15%):** Kalos is not separated from Optuna, Ax or GP-EI. Its point estimate is the lowest on Ackley 0% and on Ackley 15% (where all methods are within the same interval). On Gaussian 15% Optuna TPE has a lower point estimate (0.161 against 0.186), and Ax's is essentially level (0.191). Kalos also reaches 90% of the optimum in 53% of seeds on Gaussian 15%, against 30-40% for the baselines.
- **Using only measurements** (regret of the best *noisy* observation, which needs no oracle; in the table file, not the headline), Kalos is not ahead on Gaussian 15% (0.315, against Optuna 0.293 and Ax 0.292) or on Ackley 15% (0.613, against Optuna 0.601).
- **Mixed bioprocess-like:** Kalos has the lowest mean regret (0.217). Against Ax the gap is clear (Holm-adjusted exploratory p about 0.02). Against Optuna and vanilla GP-EI the bootstrap interval only just excludes zero and the Holm-adjusted p is about 0.10, so treat those two as suggestive. Vanilla GP-EI treats categories as ordinal by design, so part of its gap reflects that choice.
- **Cost:** on the mixed task Kalos took a median of about 19 s per proposal (Ax about 8 s, GP-EI about 3 s) and about 118 s per 5-batch run. On the synthetic tasks Kalos took about 0.3-0.5 s per proposal, in line with GP-EI.

**Caveats.**

- All objectives are synthetic. The bioprocess-like surface was written for this benchmark, and its difficulty is not calibrated to any real process. **Nothing here is a real-data claim.** It does not show that Kalos saves experiments in a wet lab.
- Regret uses the best *queried true* value, which is an oracle selection. The noisy-observation view above is the honest practitioner view.
- Every method runs at its library defaults. None is tuned, Kalos included. A tuned Optuna, Ax or BoTorch could do better; this is not an exhaustive search of optimizer configurations.
- Only one budget, one dimension (4-D synthetic), one batch size per task, and one noise level for the mixed task. Results do not generalise beyond these settings.
- **The BO numbers depend on the torch build.** The LHS and random rows do not. The benchmark runs on the torch version in `uv.lock`.
- The first full run failed the reproducibility gate for Ax (its random seeds were not set). The arm was seeded and the whole suite rerun. The first run is kept, unchanged, under [`benchmarks/results/superseded/run1_ax_unseeded/`](benchmarks/results/superseded/run1_ax_unseeded/). Only the Ax rows changed; every other arm reproduced exactly.
- Not included: scikit-optimize, HEBO and other optimizers. They were not evaluated.

**Reproduce** (about 16 minutes on an 11-core laptop; the synthetic sweep above is `python -m kalos.bench`):

```bash
python -m pip install -e ".[ml,bench]"
python -m kalos.bench --vs-baselines           # confirmatory run: writes benchmarks/results/
python -m kalos.bench --vs-baselines --quick   # smoke test, 2 seeds, writes nothing
```

## Current limitations

- No validated real-data win; no independent wet-lab confirmation.
- No user or organisation management. The legacy `/api/campaign*` routes act on one default campaign per tenant; the multi-campaign API is under `/api/campaigns`.
- Exact GP only; capped at 2,000 fit rows. Synchronous analysis runs on the request path (a concurrent sync upload gets 503 busy); background jobs are the alternative (`docs/JOBS.md`).
- Scale-up readout evidence is synthetic.
- State is local files and SQLite; no HA, no managed backups beyond the compose sidecar (its drill was not exercised).

## Roadmap

- **MVP (toward a public beta, not started unless noted):**
  IMPLEMENTED: upload, analysis, campaign loop, scale readout, token auth, tenant-keyed store, multi-campaign API, background jobs, engine-side JWT verification.
  IN PROGRESS (open PRs): #76-#81, #83, #84.
  PLANNED: web login (Better Auth), web-side JWT issuance, organisations as tenants, a rate limit.
- **Later (PLANNED):** sparse/variational surrogate, metrics and alerting, server-side PDF readouts (`TODOS.md`), independent validation on real data, an external security review.
- **EXPERIMENTAL:** mechanistic priors, alternative surrogates (must beat the GP on published datasets first; PR #79).

## Deploy and operate

The deploy pack in `deploy/` has a container image per service, a `compose.yaml`, and a SQLite backup sidecar.
The engine binds to loopback by default and refuses to serve unauthenticated on any other interface.
Auth is bearer tokens configured with `KALOS_AUTH_TOKENS`; copy `deploy/.env.example` and fill it in.
The runbook says what is and is not verified: a first deployment on 2026-09-10 ran the stack end to end, but its own "NOT verified" list still says no image was built and no container was run, and the backup drill was not exercised.
CI builds the engine image and waits for its healthcheck on every push.
TLS is out of scope: put a reverse proxy in front. Full steps: [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md).
Because the web UI has no login, treat any deployment as private-network only. Draft PR #84 would make the web port loopback by default (the compose file on `main` still publishes it on all interfaces).

## Extending to other domains

The optimization engine (`kalos/core/`) and the upload and validate pipeline are domain-neutral.
Biology lives in `kalos/domains/` as data, not code in the engine.

- Declare a `ColumnRoles` schema (target, features, categoricals, groups, ids), or pass the `roles` JSON field to `POST /api/run`, and the analysis runs under the generic profile.
- Mixed continuous and categorical design spaces fit a BoTorch `MixedSingleTaskGP` and propose with `optimize_acqf_mixed`.
- To target a new domain, copy `kalos/domains/generic.py` into a new profile.
  Nothing in `core/` changes.

## Data policy

No client or proprietary data lives in this repo.
The examples and tests use synthetic data, and synthetic data is always labeled as such.
The anonymizer (`anonymize=true` on upload) is opt-in.

## Stack and cost

The whole stack is free and open source: Python, FastAPI, BoTorch/PyTorch, SQLite, Next.js, GitHub Actions. Local development needs no paid service. Optional external-provider slots (`/api/providers`) and the optional LLM header-normalisation tier (`normalize` extra) use paid APIs only if you supply keys; neither is required.

## Repository layout

- `kalos/core/`: the BoTorch surrogate, acquisition, grouped CV, conformal bands, and gates.
- `kalos/kit/`: torch-free facade over the splits, drivers, conformal, gates, and anonymizer.
- `kalos/validation/`: the data-quality checks and their report.
- `kalos/normalize/`: unit registry and header-to-column mapping.
- `kalos/scale/`: scale-up features, transfer, and leave-one-scale-out evaluation.
- `kalos/domains/`: declared column roles and domain profiles.
- `kalos/portal/`: the FastAPI service.
- `kalos/bench/`: the benchmark harness.
- `kalos/data/`, `kalos/features/`, `kalos/providers/`: anonymizer, optional feature embedders, provider credential slots.
- `deploy/`: container files, compose, backup sidecar, runbook.
- `docs/`: API contracts and evidence.
- `examples/`: demos and synthetic data sets.
- `experiments/`: off-path research prototypes, not part of the package or its guarantees.
- `tests/`: pytest suite.

## Contributing

- One focused pull request per change; conventional commits (`feat:`, `fix:`, `docs:`, `chore:`).
- Run the checks above (`ruff`, `mypy`, `pytest`) before opening a PR; update `CHANGELOG.md` with the code.
- Use synthetic data only; never commit client or proprietary data, secrets, or tokens.
- Do not describe a result as validated unless it was measured on independent data and reported with its caveats.
- PR descriptions should state what is implemented, what is incomplete, tests run, security impact, and dependencies.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## About the author

Brendan Dang.
Founder-operator in biotech process optimization, now an ML engineer.
[github.com/itsbrendandang](https://github.com/itsbrendandang)
