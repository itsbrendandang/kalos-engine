# Kalos engine

Kalos reads a sheet of past runs from your lab and proposes the next recipes to try. It says how far to trust each suggestion, and it refuses when your runs cannot support an answer.

This repository is the engine: the program that checks the sheet, builds the model and proposes recipes. The pages people use are in [kalos-web](https://github.com/itsbrendandang/kalos-web). The Mac app wraps both.

> This README describes branch `desktop-engine`, which is not yet merged. It is stacked on `gxp-integrity` ([PR #120](https://github.com/itsbrendandang/kalos-engine/pull/120)). What `main` has, and what is still on a branch, is in [Branches and pull requests](#branches-and-pull-requests).

## Who it is for

- A process-development scientist with a spreadsheet of runs (media, feeds, pH, temperature, a titer or yield column) who wants a suggested next round and an honest read on whether the data supports it.
- A small team that works in rounds: propose, run, measure, propose again. A few dozen to about two thousand runs.
- One person on one Mac, with the desktop app. Nothing leaves the machine.
- A team that needs sign-in and separate workspaces, with the hosted mode.

Kalos is under development and is not sold or hosted for anyone today.

## One round

| Stage | What you do | What Kalos does |
| --- | --- | --- |
| **Setup** | Upload a run sheet (CSV, TSV or Excel). Pick the target column, and whether higher or lower is better. Confirm which columns are inputs. | Checks the sheet (units, limits, repeats, missing values, columns that look like row numbers or outcomes) and says what it found, with a reason for each column it left out. Builds the model from your runs and tests it on runs it had not seen. |
| **Plan** | Read the proposed recipes and decide whether to run them. Press **Start these runs**. | Proposes a batch of recipes. Each one is marked **Best bet** (the model expects a high value) or **New ground** (the model is unsure, and wants to learn). Each carries a predicted value, a likely range, and a flag when it is far from anything you have tried. |
| **Run** | Run them in the lab. | Keeps the runs in the lab apart from measured ones, and does not propose them again. |
| **Record** | Enter the measured results. | Appends them to the campaign. A recorded result is never changed or deleted. |
| **Learn** | Read what the round taught the model, then plan the next round. | Reports **Reliable**, **Not yet reliable** or **Not enough runs to tell**, with a line on why it matters, and updates the model with the new results. |

Every answer says where it came from: the engine version, the sheet it read (by fingerprint), the settings and the seed. Starting a run records who started it and from which analysis.

One click down from every verdict is the exact number it came from. The method is in [`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md) and [`docs/UPLOADS.md`](docs/UPLOADS.md).

## The desktop app

The Mac app ([kalos-web `desktop/`](https://github.com/itsbrendandang/kalos-web/tree/redesign-notion/desktop), on branch `redesign-notion`, not yet merged) starts this engine as a local process for one person.

**Run it from a checkout** (Python 3.11 or later; everything is free and open source):

```bash
python -m pip install -e ".[ml,portal]"
KALOS_AUTH_TOKENS='[{"token_sha256":"<sha256 of a secret>","subject":"me","tenant":"local","scopes":["read","write"]}]' \
  python -m kalos.portal --desktop
```

**Build the packaged engine** (Apple silicon, needs `uv`):

```bash
scripts/build_desktop_engine.sh out/   # writes out/kalos-engine and out/python/, then starts it once and checks it
```

About 955 MB on disk, 272 MB as a zip, built in about 20 s with a warm cache. The app runs `out/kalos-engine --desktop`.

**Where data lives.** One folder, set by `KALOS_STATE_DIR`. The Mac app uses `~/Library/Application Support/Kalos`; the engine's own default is `~/.kalos`. It holds `portal.db` (campaigns, runs, results, the activity record), the last analysis, and any archived activity rows. To back up, quit and copy the folder.

**Nothing leaves the Mac.** The engine listens on `127.0.0.1` only, on a port the app picks at each launch, and answers only calls that carry the secret the app made for that launch. Checking a sheet, building the model and proposing recipes call no outside service. The only parts of the engine that can talk to a language model are the optional column-name helper (`kalos.normalize.llm`, not used by the upload route) and the assistant (branch `assistant`, not yet merged). Both follow `KALOS_LLM_PROVIDER`: when the app leaves it unset, desktop mode picks `ollama` if a model is already running on this Mac at `127.0.0.1:11434`, else `none`. The current Mac app pins `none`. A hosted provider is never the default. Data goes to a cloud provider only if you set `KALOS_LLM_PROVIDER=anthropic` and give it a key yourself.

Desktop mode also: refuses to start without a secret or on any address other than loopback; runs one analysis at a time; caps the threads a fit may use at half the cores (1 to 4) so the laptop stays usable; turns off the interactive API docs; and, when it is stopped, gives a running analysis 12 s to finish and otherwise drops it so nothing half-written reaches the database. The first fit on a 30-row sheet takes about 5 s, and on a 213-row sheet about 7 s (Apple M3 Pro). Details, exit codes and the `GET /api/health` shape: [`docs/DESKTOP.md`](docs/DESKTOP.md).

## The hosted mode, in short

For a team, the engine runs behind kalos-web on a server you control. People sign in to kalos-web (Google account plus a code from an authenticator app, invite only). kalos-web maps their organization to a workspace (a *tenant*) and signs a token for each request that lives at most 120 s. The engine checks the signature with the public key in `KALOS_JWT_PUBLIC_KEYS_FILE` and reads the tenant from the token, never from the request body. Every campaign, result and activity row is keyed by tenant. Long-lived service tokens (`KALOS_AUTH_TOKENS`) remain for scripts and for the compose stack.

The deploy pack in [`deploy/`](deploy/) (containers, compose, a backup sidecar, a runbook) was exercised once end to end. Its compose file still passes a service token to kalos-web and has not been updated for sign-in; [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md) says what is and is not verified. Put TLS and a reverse proxy in front. The engine refuses to listen on a non-loopback address without tokens and a secret salt, and refuses `KALOS_ENV=production` without tokens.

The contract between the two programs: [`docs/ENGINE_TOKEN.md`](docs/ENGINE_TOKEN.md). The hardening track and what it leaves out: [`docs/HARDENING.md`](docs/HARDENING.md).

## Evidence

**Kalos has not been tested on independent lab data.** No claim is made that it saves experiments in a wet lab. What has been measured:

### Replay on 15 open datasets (branch `bench-open-replay`, [PR #123](https://github.com/itsbrendandang/kalos-engine/pull/123), not yet merged)

Each dataset is a table of measured recipes from a published study. The replay hides the results, lets each strategy pick recipes from the table, reveals the measured value, and counts how many picks it took to reach 90% of the best value in the table. 20 seeds, 4 picks per round, at most 40 picks. **This is a replay of old data, not a test in a lab.** The tables come from earlier optimization campaigns, so picking at random is a strong strategy on them.

Three strategies: **BO** is Kalos's search; **random** picks at random; **LHS** spreads picks evenly. Fewer picks is better. Values are the mean over seeds; a seed that never reached 90% counts at the cap. "Holm (15)" is the sign-test p after correcting for testing 15 datasets; below 0.05 counts as clear.

| dataset | rows | BO | random | LHS | BO − random [95% CI] | Holm (15) |
|---|---:|---:|---:|---:|---|---:|
| DBO_rat_myocyte | 209 | 42.1 (5/20) | 47.4 (4/20) | 48.7 (1/20) | -5.2 [-11.1, -0.2] | 1.000 |
| MOBO_rat_myocyte | 182 | 14.8 (18/20) | 18.4 (20/20) | 15.1 (20/20) | -3.5 [-9.9, +3.6] | 1.000 |
| HeLa_regular | 307 | 42.6 (7/20) | 37.9 (12/20) | 46.5 (2/20) | +4.8 [-2.8, +12.0] | 1.000 |
| HeLa_timesaving | 325 | 35.9 (15/20) | 48.7 (4/20) | 20.4 (20/20) | -12.8 [-19.5, -6.8] | 0.014 |
| Tcell_expansion | 535 | 10.4 (20/20) | 11.2 (20/20) | 11.9 (20/20) | -0.8 [-3.7, +1.8] | 1.000 |
| TF1_expansion | 1562 | 8.8 (20/20) | 11.0 (20/20) | 15.2 (20/20) | -2.1 [-4.9, +0.0] | 1.000 |
| pichia_HSA | 213 | 20.6 (20/20) | 27.1 (15/20) | 28.1 (15/20) | -6.5 [-13.8, +0.8] | 1.000 |
| pichia_HerceptinA | 214 | 16.1 (20/20) | 17.4 (19/20) | 15.3 (20/20) | -1.3 [-7.2, +3.9] | 1.000 |
| pichia_RBDJ | 235 | 16.5 (20/20) | 40.0 (8/20) | 24.1 (18/20) | -23.5 [-29.4, -17.4] | 0.000 |
| trichoderma_ambr | 20 | 5.5 (20/20) | 6.3 (20/20) | 8.1 (20/20) | -0.9 [-2.1, +0.1] | 1.000 |
| trichoderma_ambr_condition | 20 | 5.8 (20/20) | 6.3 (20/20) | 7.6 (20/20) | -0.6 [-1.7, +0.5] | 1.000 |
| candida_rsm_nr1 | 13 | 3.5 (20/20) | 3.5 (20/20) | 4.0 (20/20) | +0.1 [-0.1, +0.2] | 1.000 |
| scfv_refolding_18270843 | 47 | 20.2 (20/20) | 26.6 (20/20) | 21.5 (20/20) | -6.3 [-11.1, -1.6] | 1.000 |
| rudt_ib_refolding_16315733 | 24 | 11.7 (20/20) | 10.5 (20/20) | 11.2 (20/20) | +1.2 [-0.7, +3.0] | 1.000 |
| cho_bioreactor_20829178 | 24 | 8.4 (20/20) | 7.8 (20/20) | 6.8 (20/20) | +0.7 [-1.1, +2.5] | 1.000 |

The honest reading, in the replay's own words:

- **Pichia RBDJ**: BO beats random by about 23 evaluations, and the advantage holds on 17 of 17 non-tied seeds (Holm p < 0.001). This is the one dataset of 15 where BO beats random clearly.
- **HeLa time-saving**: BO beats random (Holm p = 0.014), but LHS is clearly best here (20.4 vs 35.9). The honest reading is "the GP is not the best strategy on this pool".
- **HeLa regular, TF-1, Pichia HSA**: BO is ahead of LHS in some cases, but none of these beats random on the sign test after correction.
- **Everything else**: no detectable difference between BO and random. BO is behind LHS on HeLa regular and on the CHO set. Small pools (13 to 24 recipes) cannot show anything.
- Nothing here supports "BO beats random" as a general claim.

Protocol, raw files and attribution for every dataset: `benchmarks/OPEN_REPLAY.md` on that branch. Use the attribution lines verbatim when the numbers are quoted.

### Made-up surfaces with a known best point ([`BENCHMARK.md`](BENCHMARK.md), [`docs/BENCHMARK.md`](docs/BENCHMARK.md), on `main`)

On two made-up 4-input surfaces, 10 seeds, 18 experiments: with clean measurements Kalos's search ends far closer to the best point than an evenly spread design (regret 0.013 vs 2.247 on the smooth surface; 1.560 vs 3.625 on the rugged one). With 15% measurement noise the gap shrinks to 1.873 vs 2.247 and 3.014 vs 3.625. Against other optimizers (Optuna TPE, Ax, a plain BoTorch model) on the same surfaces plus one made-up mixed surface, 30 seeds, Kalos is clearly ahead of random and of the evenly spread design in every cell; it is not separated from Ax or the plain model on any of the four synthetic cells, nor from Optuna on the noisy ones; on the mixed surface it is ahead of Ax and only borderline ahead of the other two. These are made-up problems, so they say nothing about a real process. The numbers depend on the torch build, which is pinned.

### Scale-up readout ([`docs/SCALE_EVIDENCE.md`](docs/SCALE_EVIDENCE.md), on `main`)

On one made-up dataset of 55 batches from 0.01 L to 2000 L, predicting the five 2000 L batches from the smaller scales gave a lower mean error than two naive guesses (0.53 vs 1.51 and 1.38), with no interval and no test on that gap; the ranking of those five was consistent with chance (p = 0.233 at n = 5). Scale claims are on hold until real batch data exists. The readout's safeguard is that it refuses to predict when its own back-test on your scale ladder does not beat the naive guesses.

### What is measured on every sheet

Rather than a global claim, every analysis carries its own evidence: the model's rank agreement on runs it had not seen, with an interval; how many measured results landed inside their likely range; the checks it passed or failed; and the noise between repeated recipes. An 80-row subset of the shipped example gave different proposals on two identical runs in one process, so proposals are reproducible only in the sense that the inputs, settings and library versions are recorded (`docs/CAMPAIGNS_API.md`, "Determinism").

## What Kalos does not do

- **It does not approve anything.** Starting runs, accepting a result and signing off a procedure are a person's decisions. Kalos records that the decision was made and by which account; it never makes it.
- **It makes no biosafety decisions** and knows nothing about the organism, the containment level or the hazards of a recipe. A proposed recipe is a point in the ranges you uploaded, nothing more.
- **It makes no claims about fitness for regulated work.** No independent review, no certification, no audit by a third party. `docs/HARDENING.md` lists what is out of scope.
- **It does not mix measured and simulated data.** Simulated runs (branch `twin-virtual-runs`, [PR #122](https://github.com/itsbrendandang/kalos-engine/pull/122), not yet merged) are stored in their own table with an origin label and a guard that keeps them out of the model fit. On `main` and this branch there are no simulated runs at all. Example sheets in this repository are generated and labelled as such.
- **It does not sell or host anything today.** No public service exists.
- **It does not ship training data.** Every dataset in the evidence section is private or fetched by the person running the benchmark, under that dataset's own terms.

## Configuration

Every setting is an environment variable. Unset and empty both mean the default. A value outside its range stops the engine at start with one line that names the variable (settings read through `kalos/envconfig.py` and the `KALOS_JOB*` family), or falls back to the default where the comment says so. Sources: `kalos/portal/config.py`, `kalos/envconfig.py`, `kalos/portal/__main__.py`, `desktop.py`, `auth.py`, `jwt_auth.py`, `limits.py`, `uploads.py`, `busy.py`, `jobs.py`, `pipeline.py`, `kalos/normalize/config.py`, `kalos/data/anonymizer.py`, `kalos/bench/`.

`tests/test_readme_env.py` fails when this table and the code disagree: a `KALOS_*` name the code reads must be in the table, and a name in the table must be read by the code, unless the row says `not yet merged` or `tests only`.

### Engine settings

| Variable | Default | Accepted values | What it does |
| --- | --- | --- | --- |
| `KALOS_HOST` | `127.0.0.1` | an address | Where the engine listens. A non-loopback address is refused unless tokens and `KALOS_ANON_SALT` are set (or `KALOS_ALLOW_OPEN_ACCESS`). Desktop mode allows loopback only. |
| `KALOS_PORT` | `8050` | a whole number | The port. |
| `KALOS_STATE_DIR` | `~/.kalos` | a folder | Where `portal.db`, the last analysis and archives live. Created if missing. |
| `KALOS_ENV` | `development` | `development`, `production` | `production` refuses to start without tokens or a public key, whatever `KALOS_ALLOW_OPEN_ACCESS` says. |
| `KALOS_DESKTOP` | unset | `1`, `true`, `yes`, `on` | Desktop mode (same as `--desktop`): loopback only, a token required, one analysis at a time, fewer threads, no API docs. |
| `KALOS_AUTH_TOKENS` | unset | a JSON list of `{token_sha256, subject, tenant, scopes}` | Service tokens, stored as hashes. With none set and no public key, the engine is open on loopback only and every caller is the anonymous `default` tenant. |
| `KALOS_AUTH_TOKENS_FILE` | unset | a path | The same list in a file; wins over the inline value. Read on every request, so tokens rotate without a restart. |
| `KALOS_AUTH_DENY_SUBJECTS` | unset | comma-separated `sub:<subject>` or `sid:<session>` | Refuse these callers with 401. |
| `KALOS_JWT_PUBLIC_KEYS_FILE` | unset | a path to a JWKS file with Ed25519 keys | Accept the short-lived tokens kalos-web signs (hosted mode). |
| `KALOS_JWT_ISSUER` | `kalos-web` | text | The `iss` claim a token must carry. |
| `KALOS_JWT_AUDIENCE` | `kalos-engine` | text | The `aud` claim a token must carry. |
| `KALOS_JWT_LEEWAY_S` | `30` | `0` to `120` | Clock slack when checking token times, in seconds. |
| `KALOS_JWT_MAX_TTL_S` | `120` | `1` to `300` | The longest token life the engine accepts, in seconds. |
| `KALOS_ALLOW_OPEN_ACCESS` | unset | `1`, `true`, `yes` | Serve with no tokens on a non-loopback address, on purpose. Ignored in production; refused in desktop mode. |
| `KALOS_CORS_ORIGINS` | unset | comma-separated origins, e.g. `https://app.example.com` | Browser origins allowed to call the engine. Unset means `http://localhost` and `http://127.0.0.1` on any port. |
| `KALOS_TRUSTED_PROXIES` | unset | comma-separated proxy addresses | Trust `X-Forwarded-For` from these addresses only. Unset means forwarded headers are ignored. |
| `KALOS_ANON_SALT` | unset | a secret string | Salt for pseudonyms made by the opt-in anonymizer. Unset uses a published dev salt with a warning, and blocks a non-loopback bind. |
| `KALOS_VALIDATION_MODE` | `warn` | `warn`, `strict` | `strict` refuses a sheet on findings that `warn` only reports. |
| `KALOS_AUDIT_RETENTION_DAYS` | unset | `1` to `36500` | Copy activity rows older than this many days to `<state dir>/audit-archive/`, at start and daily. Rows are never deleted. Unset archives nothing. Ignored in desktop mode. |
| `KALOS_MAX_UPLOAD_MB` | `25` | a number above 0 | Largest run sheet accepted. Fixed caps on top: 100,000 CSV rows, 512 columns, 2,000,000 Excel cells. |
| `KALOS_MAX_REQUEST_BYTES` | `1048576` (1 MiB) | a whole number above 0 | Largest body on every route except upload. |
| `KALOS_UPLOAD_READ_TIMEOUT_S` | `60` | a number of at least 1 | How long the engine waits to receive an upload's bytes. |
| `KALOS_MAX_FIT_ROWS` | `2000` | a whole number | Most rows the model is built from. A larger sheet is refused with the code `too_many_fit_rows`, never trimmed silently. |
| `KALOS_TORCH_THREADS` | the smaller of 4 and the core count; desktop: half the cores, 1 to 4 | a whole number; `0` means do not pin | Threads one fit may use. |
| `KALOS_ANALYSIS_CACHE` | `4` | a whole number; `0` turns it off; a bad value means `4` | How many recent uploads keep their analysis in memory, so the same sheet is not refit. |
| `KALOS_APPLICABILITY_DISTANCE` | unset | a number above 0 | Distance from your runs beyond which a proposal is flagged `far_from_data`. Unset uses the product default (three times the typical gap between your runs, at least a quarter of the range). A value sent with the upload wins. |
| `KALOS_DEMO_TIMEOUT_S` | `120` | a number of seconds | Wall clock for the two demo routes (`/api/single`, `/api/multi`). |
| `KALOS_JOBS_ENABLED` | on | `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off` | Background analysis jobs. Off makes the job routes answer 503. |
| `KALOS_JOB_WORKERS` | `2`; desktop `1` | a whole number of at least 1 | Analyses that may run at once, foreground and background together. |
| `KALOS_JOB_PER_TENANT` | `1` | a whole number of at least 1 | Analyses one tenant may run at once. |
| `KALOS_JOB_QUEUE_DEPTH` | `8` | a whole number of at least 1 | Queued or running jobs one tenant may hold before 429. |
| `KALOS_JOB_GLOBAL_QUEUE_DEPTH` | `64` | a whole number of at least 1 | Queued or running jobs across all tenants before 429. |
| `KALOS_JOB_TIMEOUT_S` | `1800` | a number of at least 1 | Longest a background job may run before it is stopped. |
| `KALOS_JOB_MAX_UPLOAD_BYTES` | the upload cap (`26214400` at the default 25 MB) | `1` up to the upload cap | Largest sheet a background job accepts. |
| `KALOS_JOB_DEADLINE_S`, `KALOS_JOB_ENGINE_PID`, `KALOS_JOB_WORKER_SESSION` | set by the engine | - | Passed by the engine to each worker process it starts. Do not set them. |
| `KALOS_LLM_PROVIDER` | `anthropic`; desktop: `ollama` when a local Ollama answers, else `none` | `anthropic`, `ollama`, `none` | Which language model the optional column-name helper and the assistant may call. `none` never calls any. `anthropic` also needs `ANTHROPIC_API_KEY`, else it stays offline. |
| `KALOS_OLLAMA_URL` | `http://localhost:11434` | a URL | Where a local Ollama answers. The desktop probe accepts loopback addresses only. |
| `KALOS_OLLAMA_MODEL` | `llama3.1` | a model name Ollama has pulled | The local model. |
| `KALOS_NORMALIZE_MODEL` | `claude-sonnet-5` | a model id | The hosted model for the column-name helper. Unused by `ollama` and `none`. |
| `KALOS_NORMALIZE_MAX_SAMPLE` | `5` | `1` to `20`; a bad value means `5` | Sample values per column shown to the column-name helper (identifiers are screened out first). |
| `KALOS_BUILD_SHA` | unset, recorded as `unknown` | text | Build identifier written into every run record. |
| `KALOS_GIT_SHA` | unset; then `git rev-parse HEAD`, else `unknown` | text | Commit written into a scale-up readout. |
| `KALOS_DATASET_DIR` | unset | a folder | Where the private `dataset_0.csv` lives for `examples/real_data_validation.py`. Tests skip without it. |
| `KALOS_MEDIA_DATA`, `KALOS_AMII_DATA`, `KALOS_INDPENSIM_PATH` | unset | a folder or file | Private benchmark data for `kalos/bench/`. Never in this repository. |
| `KALOS_INTEGRATIONS` | unset | comma-separated names, e.g. `benchling` | Turns on a connector. On branch `benchling-connector`, not yet merged. |
| `KALOS_SKILLS` | unset | `1` | Serves the read-only `/api/skills` routes. On branch `skills-framework` ([PR #124](https://github.com/itsbrendandang/kalos-engine/pull/124)), not yet merged. |
| `KALOS_SKILLS_DIR` | unset | a folder | Team skill folders, one per tenant. On branch `skills-framework` (PR #124), not yet merged. |
| `KALOS_ALLOW_TORCH_DRIFT` | unset | `1` | Lets `tests/test_torch_pin.py` pass on a torch build other than the pinned one. Tests only. |
| `KALOS_DATA` | unset | a folder | A checkout of the private `kalos-data` repository for the scale-up tests. Tests only. |
| `KALOS_BUNDLE_EXTRAS` | `ml portal assist` | extras from `pyproject.toml` | Which extras `scripts/build_desktop_engine.sh` bundles. |
| `KALOS_BUNDLE_PYTHON` | `3.12` | a Python version | Which CPython the build script downloads. |

`ANTHROPIC_API_KEY` is the one setting without the `KALOS_` prefix: the key for the hosted provider. The engine only checks whether it is present; it never logs or stores it.

### Compose stack (`deploy/`)

`KALOS_ENGINE_PORT` (8050), `KALOS_WEB_PORT` (3000), `KALOS_WEB_BIND` (`127.0.0.1`), `KALOS_IMAGE_TAG` (`local`), `KALOS_ENGINE_URL`, `KALOS_ENGINE_TOKEN` and `KALOS_ENGINE_TIMEOUT_MS` are read by `deploy/compose.yaml` and by kalos-web, not by this engine. They are documented in [`deploy/.env.example`](deploy/.env.example) and in the kalos-web README.

## API

Routes are in `kalos/portal/app.py`, `campaign_routes.py`, `campaigns_api.py`, `jobs_api.py` and `scale_routes.py`. Every tenant-scoped route reads the tenant from the token, never from the request. In development (`KALOS_ENV` unset, not desktop) FastAPI serves the generated reference at `/docs`.

| Method | Path | Scope | What it does |
| --- | --- | --- | --- |
| POST | `/api/run` | write | Upload a sheet. Returns the analysis and the proposed recipes ([`docs/UPLOADS.md`](docs/UPLOADS.md)). |
| GET | `/api/latest` | read | The last analysis for this tenant. Never fits. |
| GET, POST | `/api/campaigns`, `/api/campaigns/{id}` (PATCH), `/{id}/upload`, `/{id}/analysis`, `/{id}/runs`, `/{id}/runs/{run_id}/result`, `/{id}/reanalyze` | read / write | Several campaigns per tenant ([`docs/CAMPAIGNS_API.md`](docs/CAMPAIGNS_API.md)). |
| GET, POST | `/api/campaign`, `/api/campaign/start`, `/result`, `/reanalyze` | read / write | The single default campaign per tenant ([`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md)). |
| GET, POST | `/api/campaigns/{id}/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/cancel` | read / write | Background analysis jobs ([`docs/JOBS.md`](docs/JOBS.md)). |
| POST | `/api/scale/readout` | write | Scale-up readout, or a refusal ([`docs/SCALE_READOUT.md`](docs/SCALE_READOUT.md)). |
| GET | `/api/single`, `/api/multi` | read | Demo searches on made-up surfaces. Not tenant data. |
| GET | `/api/providers` | read | Which optional provider slots have a credential (never the credential). |
| GET | `/api/health` | read | Versions, torch state, schema version, analyses in flight. |
| GET | `/healthz`, `/readyz` | none | Liveness and readiness. |

Every change to a campaign, run, result or job writes one row to `audit_log` in `portal.db`: tenant, subject, how they were authenticated, the action and the outcome; never a token, a file or a result value. Since schema v4 the table is append-only at the database level (triggers refuse update and delete), and since v5 recorded results are too. Rows have no hash chain or signature, and someone with write access to the file can still replace it. Only a `campaign.result` row is written in the same transaction as its change; the others are best-effort. (`gxp-integrity`, [PR #120](https://github.com/itsbrendandang/kalos-engine/pull/120), not yet merged. On `main` the table exists, is not append-only, and rows older than 365 days are deleted.)

## Development

```bash
python -m pip install -e ".[ml,portal,bench,dev]"
scripts/check.sh            # ruff + mypy, the same as the pre-push hook
scripts/check.sh --full     # + the whole test suite (about 50 minutes; 2196 passed, 10 skipped, 3 xfailed at the last full run)
python -m pytest -q tests/test_desktop_mode.py tests/test_readme_env.py   # a quick slice
```

There is no CI service; `scripts/pre-push` runs `check.sh` locally. `python -m pytest -q` collects about 1,600 test functions in `tests/`. Tests that need private data skip when its folder is not set. `tests/test_torch_pin.py` fails if the installed torch, BoTorch or GPyTorch differs from the pin in `pyproject.toml` and `uv.lock`.

**Worktrees.** Each branch below lives in its own worktree next to the main checkout (`git worktree list`), so several agents can build and test without stepping on one another: `kalos-engine` (`main`), `-wt-desktop` (this branch), `-wt-gxp`, `-wt-pg`, `-wt-twin`, `-wt-bench`, `-wt-bench-ports`, `-wt-skills`, `-wt-assist`, `-wt-assist-sop`, `-wt-benchling`. Commit with explicit paths from inside the worktree. Never run two engines on the same `KALOS_STATE_DIR`; the engine lock refuses the second.

**Rules.** One focused pull request per change, conventional commits (`feat:`, `fix:`, `docs:`, `chore:`), `CHANGELOG.md` updated in the same change. Synthetic data only; never commit client data, secrets or tokens. Never describe a result as proven unless it was measured on independent data and reported with its caveats. A PR description says what is implemented, what is incomplete, the tests run, the security impact and the dependencies. Vulnerabilities: see [`SECURITY.md`](SECURITY.md).

## Branches and pull requests

`main` (last merged: [#119](https://github.com/itsbrendandang/kalos-engine/pull/119)) has: upload, sheet check, the model and its tests on unseen runs, the campaign loop, several campaigns per tenant, background jobs, service tokens and signed-token verification, the scale-up readout, the synthetic benchmarks, the deploy pack. Everything below is **not yet merged**. Merge order: #120 → #121 → #122 → #125 → #124 → #123, then `desktop-engine` and `assistant`.

| PR / branch | One line |
| --- | --- |
| [#120](https://github.com/itsbrendandang/kalos-engine/pull/120) `gxp-integrity` | Append-only results and activity log, coded fit-cap refusal, provenance on every analysis, pinned torch stack, numerics fixes. Base of every branch below. |
| [#121](https://github.com/itsbrendandang/kalos-engine/pull/121) `postgres-backend` | PostgreSQL behind one storage layer; SQLite stays the default. |
| [#122](https://github.com/itsbrendandang/kalos-engine/pull/122) `twin-virtual-runs` | Simulated runs with origin labels, their own append-only table (schema v6) and a guard that keeps them out of the fit. |
| [#123](https://github.com/itsbrendandang/kalos-engine/pull/123) `bench-open-replay` | The 15-dataset replay above, with protocol and attributions. |
| [#124](https://github.com/itsbrendandang/kalos-engine/pull/124) `skills-framework` | Typed skill files, a registry, a runner with receipts, read-only routes behind `KALOS_SKILLS=1`. |
| [#125](https://github.com/itsbrendandang/kalos-engine/pull/125) `bench-ports` | More benchmark simulators, warm start, other model types as benchmark arms, opt-in exclusion of QC-flagged rows. |
| `desktop-engine` (this branch, no PR yet) | Desktop mode, `/api/health`, the packaged engine, `contract_version` 3, `columns[]` with reasons, `direction` per target, `applicability` flags, the analysis cache, faster fits on small sheets. |
| `assistant` (no PR yet) | The grounded assistant: next step, questions answered only from your own sheets and rounds with sources, procedure drafts. Uses `KALOS_LLM_PROVIDER`. |
| `benchling-connector` (no PR yet) | Read Benchling studies and results; push proposals as drafts only. Off unless `KALOS_INTEGRATIONS=benchling`. |

kalos-web's matching work is [kalos-web #82](https://github.com/itsbrendandang/kalos-web/pull/82) (branch `redesign-notion`, not yet merged): the redesigned pages in plain words, the desktop app and its local mode, Ask Kalos.

## Docs index

| File | What it covers |
| --- | --- |
| [`docs/DESKTOP.md`](docs/DESKTOP.md) | What the Mac app passes, exit codes, `/api/health`, stopping, cold start and fit times, the packaged engine. |
| [`docs/UPLOADS.md`](docs/UPLOADS.md) | Uploading a sheet: formats, caps, the sheet check, column-role safety. |
| [`docs/CAMPAIGN_LOOP.md`](docs/CAMPAIGN_LOOP.md) | The single-campaign loop: start, result, reanalyze, runs in the lab. |
| [`docs/CAMPAIGNS_API.md`](docs/CAMPAIGNS_API.md) | Several campaigns per tenant; the analysis contract (`contract_version` 3), determinism. |
| [`docs/JOBS.md`](docs/JOBS.md) | Background analysis jobs and their limits. |
| [`docs/SCALE_READOUT.md`](docs/SCALE_READOUT.md), [`docs/SCALE_EVIDENCE.md`](docs/SCALE_EVIDENCE.md) | The scale-up readout and the one measurement behind it. |
| [`BENCHMARK.md`](BENCHMARK.md), [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Made-up surface benchmarks and the pre-registered comparison with other optimizers. |
| [`docs/ENGINE_TOKEN.md`](docs/ENGINE_TOKEN.md) | The signed token kalos-web sends in hosted mode. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the browser, kalos-web and the engine talk. |
| [`docs/HARDENING.md`](docs/HARDENING.md) | The hardening track, what exists and what is out of scope. |
| [`docs/MIGRATIONS.md`](docs/MIGRATIONS.md) | `portal.db` schema versions and how to move between them. |
| [`deploy/RUNBOOK.md`](deploy/RUNBOOK.md) | The compose stack: quickstart, checklist, backups, what is verified. |
| [`docs/STE_STYLE_NOTES.md`](docs/STE_STYLE_NOTES.md) | How this README is written. |
| [`CHANGELOG.md`](CHANGELOG.md), [`docs/CHANGELOG_ARCHIVE.md`](docs/CHANGELOG_ARCHIVE.md) | What changed, by branch. |
| [`SECURITY.md`](SECURITY.md) | How to report a vulnerability. |

## Repository layout

`kalos/core/` the model, the search, the tests on unseen runs, the likely ranges and the checks · `kalos/kit/` a torch-free subset · `kalos/validation/` the sheet check · `kalos/normalize/` units and column names · `kalos/scale/` scale-up · `kalos/domains/` column roles per domain · `kalos/portal/` the service · `kalos/bench/` benchmarks · `kalos/data/`, `kalos/features/`, `kalos/providers/` anonymizer, optional embedders, provider slots · `deploy/` containers · `examples/` demos and generated sheets · `experiments/` research prototypes outside the package · `tests/`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## About the author

Brendan Dang. Founder-operator in biotech process optimization, now an ML engineer. [github.com/itsbrendandang](https://github.com/itsbrendandang)
