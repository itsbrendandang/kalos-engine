# Kalos engine

[![CI](https://github.com/itsbrendandang/kalos-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/itsbrendandang/kalos-engine/actions/workflows/ci.yml?query=branch%3Amain)

Kalos is closed-loop Bayesian optimization for bioprocess development. You upload
a sheet of past runs; it proposes the next batch of recipes; you run them, record
the results, and it proposes again. It works in three steps:

- **Harmonize.** Read a CSV, TSV or Excel run sheet as written: find the header,
  convert known units, classify every cell, and report each row or cell it cannot
  use and why. Nothing is imputed.
- **Predict.** Fit a Gaussian-process model (BoTorch) over continuous and
  categorical inputs, and propose a batch of recipes with a noisy expected
  improvement acquisition that accounts for runs already in flight.
- **Validate.** Grouped cross-validation, a split-conformal 90% band, and a
  reliability verdict that says when the model is not good enough to trust.

This repository is the engine: a Python package and a FastAPI service with no UI.
The scientist-facing app is [kalos-web](https://github.com/itsbrendandang/kalos-web).
It is early software, run self-hosted. It has not been validated in a live lab.

## Status

**Works today (on `main`, with tests):**

- Upload, validation gate, row accounting and analysis (`POST /api/run`).
- Mixed continuous/categorical GP and batch proposals that account for runs
  still in flight. A floor on a second outcome, linear constraints and protein
  sequence features exist in the Python API but are not exposed over HTTP yet.
- Campaigns: several per tenant, record results, re-analyze; long analyses as
  background jobs in local worker processes.
- Scale-up readout that refuses to predict when its backtest fails (synthetic
  evidence only).
- Grounded assistant: next-step guide, cited answers over the tenant's own data,
  SOP drafts.
- EdDSA JWT and static-token authentication, tenant-scoped storage (SQLite).

**In progress** (see the [open pull requests](https://github.com/itsbrendandang/kalos-engine/pulls)):
per-proposal intervals and their calibration at proposed recipes, user-set bounds,
goal direction and linear constraints over HTTP, strain/clone and covariate roles,
prospective scoring of predictions against measured results, and a faster
analysis path.

**Not built:** organization management in the engine (kalos-web owns it), a
sparse GP for more than 2,000 rows, metrics and alerting.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12 (CI uses 3.11, the engine image 3.12).

```bash
uv sync --python 3.11 --extra ml --extra portal
uv run python -m kalos.portal            # http://127.0.0.1:8050, OpenAPI docs at /docs
```

In a second shell, analyze the bundled example and read the result back:

```bash
curl -F file=@examples/synthetic_bioprocess/synthetic_bioprocess.csv \
     -F target=titer_g_L http://127.0.0.1:8050/api/run
curl http://127.0.0.1:8050/api/latest
```

With no credentials configured the API is open, on loopback only, as the
`default` tenant. The core package (`pip install -e .`) has no torch dependency
and is enough for the `kalos.kit` helpers.

**Example datasets** (generated, not measured):

- `examples/synthetic_bioprocess/synthetic_bioprocess.csv`: 160 runs, 10
  continuous inputs and one categorical medium, target `titer_g_L`.
- `examples/synthetic_scaleup/synthetic_scaleup.csv`: a scale ladder for the
  scale-up readout.
- Real published datasets are not redistributed; `examples/datasets/README.md`
  and [`VALIDATION.md`](VALIDATION.md) say where to get them.

**Environment variables that matter:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `KALOS_STATE_DIR` | `~/.kalos` | SQLite database and caches |
| `KALOS_HOST` / `KALOS_PORT` | `127.0.0.1` / `8050` | Bind address |
| `KALOS_ENV` | `development` | `production` removes demo routes and `/docs`, and requires auth and a CORS allowlist |
| `KALOS_JWT_PUBLIC_KEYS_FILE` | unset | Ed25519 JWKS for per-user tokens from kalos-web ([`docs/ENGINE_TOKEN.md`](docs/ENGINE_TOKEN.md)) |
| `KALOS_AUTH_TOKENS` / `KALOS_AUTH_TOKENS_FILE` | unset | Hashed static tokens for service accounts |
| `KALOS_CORS_ORIGINS` | localhost | Browser origins allowed to call the engine |
| `KALOS_ANON_SALT` | dev salt | Salt for anonymized identifiers; required off loopback |
| `KALOS_LLM_PROVIDER` | `ollama` on desktop, else `none` | Model for the assistant and header normalization |
| `KALOS_TORCH_THREADS` | `min(4, cpus)` | Torch threads per analysis (`0` = do not pin) |
| `KALOS_JOBS_ENABLED` | on | `0` turns off background jobs |

Upload limits (`KALOS_MAX_UPLOAD_MB`, `KALOS_MAX_FIT_ROWS`) and the rest are in
[`docs/UPLOADS.md`](docs/UPLOADS.md) and [`docs/HARDENING.md`](docs/HARDENING.md).

## API overview

| Group | Routes | Docs |
| --- | --- | --- |
| Analyze | `POST /api/run`, `GET /api/latest` | [`docs/UPLOADS.md`](docs/UPLOADS.md) |
| Campaigns | `/api/campaigns`, `/api/campaigns/{id}` (`upload`, `analysis`, `runs`, `runs/{run_id}/result`, `reanalyze`) | [`docs/CAMPAIGNS_API.md`](docs/CAMPAIGNS_API.md) |
| Jobs | `/api/campaigns/{id}/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/cancel` | [`docs/JOBS.md`](docs/JOBS.md) |
| Legacy single campaign | `/api/campaign`, `/start`, `/result`, `/reanalyze` | [`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md) |
| Scale-up | `POST /api/scale/readout` | [`docs/SCALE_READOUT.md`](docs/SCALE_READOUT.md) |
| Assistant | `/api/assist/next`, `/ask`, `/index/*`, `/sop/*` | [`docs/ASSISTANT.md`](docs/ASSISTANT.md) |
| Probes | `GET /healthz`, `GET /readyz`, `GET /api/providers` | |

**Auth.** kalos-web signs a short-lived EdDSA (Ed25519) JWT for each request; the
engine verifies it against the configured JWKS (issuer, audience, expiry, `kid`)
and takes the subject, tenant and scopes from the signed claims. Service accounts
use static bearer tokens, stored as SHA-256 hashes. Scopes are `read`, `write`
and `admin`. Every tenant-scoped route reads the tenant from the verified token,
never from the request, and the store is keyed by tenant. Once any credential is
configured, auth is enforced on every scoped route; a broken JWKS fails closed.

## Architecture

```text
browser ── kalos-web (Next.js, sign-in, proxy) ──EdDSA JWT──▶ kalos-engine (FastAPI)
                                                               │
              ┌────────────────────────────────────────────────┤
              ▼                                                ▼
   analysis pipeline (kalos.portal.analysis)         SQLite, per tenant
     harmonize: uploads, validation, normalize        (campaigns, runs, results,
     predict:   kalos.core surrogate + acquisition     jobs, audit log)
     validate:  grouped CV, conformal band, verdict
              │
              └── run in the request, or as a background job in a local worker process
```

- `kalos/core/`: GP surrogate, acquisition, grouped CV, conformal bands, gates.
- `kalos/portal/`: the FastAPI service, campaigns, jobs, auth, storage.
- `kalos/validation/`, `kalos/normalize/`: data checks, units and header mapping.
- `kalos/scale/`, `kalos/assist/`: scale-up readout and the grounded assistant.
- `kalos/kit/`: torch-free helpers (splits, drivers, conformal, gates, anonymizer).
- `kalos/bench/`: benchmark harness. `experiments/` is research code outside the package.

More in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Validation and its limits

- **Intervals.** The global 90% band (`conformal_q`) is calibrated on held-out
  runs (split-conformal on grouped cross-validation); on new runs that resemble
  past runs it covers about 90% or slightly more. Proposals are not like past
  runs: they are chosen where predictions are high. Each proposal's 90% and 95%
  range is therefore sized on the held-out runs each fold model ranked highest,
  to mimic that selection. Measured at proposed recipes in simulation (five
  scenarios, 40 fresh seeds each), the 90% range covered 92-99% and the 95%
  range 96-99%; the ranges are about 2.5-3x wider than the plain band, which
  covered only 60-81% (90%) at proposals. This is simulation evidence, not a
  guarantee. Each proposal interval carries a caveat with the coverage measured
  at proposals and its source; when too few held-out runs are ranked high
  enough, the plain band is shown with its lower measured coverage
  ([`VALIDATION.md`](VALIDATION.md#interval-coverage-at-proposals)).
- **Grouped cross-validation.** Runs that cannot be independent are held out
  together: as-run replicates (settings within 5% of each input's range),
  declared groups, batch, lot, plate, block, round, date and week columns,
  scale columns, and campaign rounds. Batch-like columns are not fitted or
  proposed. Every evaluable grouping is cross-validated and the lowest held-out
  score drives the verdict and the interval calibration (`cv_grouping` reports
  each candidate).
- **Verdict and refusals.** `verdict` is `reliable` (at least 20 runs and 8 CV
  groups, with held-out ordering better than chance), `not_yet` (proposals are
  returned but labeled exploration), or `refused`: fewer than 3 CV groups, or a
  held-out score whose 95% interval does not exceed 0. A refusal withholds the
  proposals and says why and what would help (`refusal`, `proposal_trust`).
  Below 12 runs the held-out score is not judged at all
  (`reliability.verdict: not_judged`, `clears_floor: null`).
- **Optimizer.** On synthetic problems with known optima it beats random search
  and matches a well-configured plain BoTorch qLogNEI. In a retrospective replay
  of six published media datasets it needed about 4x fewer experiments than
  Latin-hypercube sampling on the largest, hardest one, showed no significant
  difference on four, and did worse on one ([`VALIDATION.md`](VALIDATION.md),
  [`BENCHMARK.md`](BENCHMARK.md)). It searches only inside the range of the
  inputs already observed.
- **No live-lab result yet.** Nothing here shows that Kalos improves outcomes in
  a running lab.
- **Runtime.** An analysis of the 160-row example takes about 1-2 minutes today
  (57 s on an idle Apple-silicon machine, around 2 minutes on a busy one), most of
  it in the acquisition step. Web clients should use background jobs. Making it
  faster without changing results is open work.
- **Determinism.** Analyses are seeded; in our checks the same input gave the
  same proposals on the locked stack. Results depend on the torch build and
  platform, so the ML stack is pinned (`uv.lock`, `kalos/versions.py`).

## Deployment

`deploy/compose.yaml` runs the engine image, kalos-web, and a backup sidecar that
snapshots the SQLite database with SQLite's online backup to a host directory
outside the data volume. The engine image uses CPU-only torch. Off loopback the
engine refuses to start unless authentication and `KALOS_ANON_SALT` are set, or
open access is chosen explicitly (`KALOS_ALLOW_OPEN_ACCESS`). TLS is not handled
here: put a reverse proxy in front. Steps, upgrades, restore, and what has and
has not been verified are in [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md).

## Development

```bash
uv sync --python 3.11 --extra ml --extra portal --extra bench --extra dev
uv run ruff check kalos/ tests/ examples/
uv run mypy
uv run pytest -n auto -m "not serial"   # parallel, everything but the serial tests
uv run pytest -m serial                 # determinism tests; never under xdist
```

Markers: `slow` marks the GP-heavy tests (deselect with `-m "not slow"` for a
quick local loop); `serial` marks the determinism tests that compare outputs for
exact equality and must run without xdist. Tests run torch on one thread per
process (`tests/conftest.py`) so parallel workers do not oversubscribe the CPU.

CI (`.github/workflows/ci.yml`) runs on every pull request and every push to
`main`: ruff and mypy, the whole suite except `serial` split across 6 parallel
jobs balanced by the recorded test durations in `.test_durations`, and the
`serial` tests in their own job. It installs the versions in `uv.lock`, with
CPU-only torch. Each run publishes refreshed durations as the `test-durations`
artifact; commit it when the shards drift apart.

Contributions: one focused pull request per change, conventional commit
messages, synthetic data only, and no result described as validated unless it was
measured on independent data.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
