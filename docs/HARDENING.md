# Kalos production hardening (in-house engineering track)

This is the living plan for taking the Kalos portal from a single-user local tool to a multi-tenant, durable, operable service.
It covers the **behind-the-scenes engineering** only.
Regulatory and certification work (21 CFR Part 11, GAMP 5 validation, SOC 2 / ISO 27001) is handled **offline, out of this repo**, and is intentionally out of scope here.

The readiness assessment that motivates this work rated the science and honesty layer as ready, and the surrounding system as not.
This track closes the surrounding-system gaps in dependency order.

## Principles

- **Backward compatible by default.** Every hardening step ships behind configuration and defaults to the current behavior, so the local dev loop and pilot deployments never break mid-migration.
- **In-house, self-hosted.** No dependency on an external identity provider or SaaS control plane. Auth, storage, and secrets are provisioned and owned by the operator.
- **Fail closed when configured, open when not.** With no auth configured the API stays open (dev/pilot); the moment tokens are provisioned it enforces them. There is no half-authenticated state.
- **Reuse what exists.** The experiments path already has a `SqliteStore` (`kalos/store/`) and a constant-time bearer-token check (`kalos/portal/experiments.py`). New work generalizes those rather than inventing parallel mechanisms.

## Sequence

The phases are ordered by hard dependency: you cannot isolate data per tenant without an identity to isolate by, and you cannot make a service reliable before it has a durable store.

### Phase 1 - Foundation: identity and durable, tenant-scoped state

**1a. Authentication + authorization (in progress).**
A general in-house token layer (`kalos/portal/auth.py`): a bearer token resolves to a `Principal` (subject, tenant, scopes).
Tokens are provisioned by the operator and stored as SHA-256 hashes (never plaintext, never logged); the presented token is hashed and constant-time compared.
With no tokens configured the layer returns an anonymous principal on a `default` tenant with read+write scope, preserving today's behavior and logging a startup warning.
Mutating endpoints (`/api/run`, `/api/campaign/*`) require the `write` scope; the existing runner-token channel is subsumed by the same mechanism over time.

**1b. Tenant-scoped persistence (done).**
Every store read/write is keyed by the caller's `Principal.tenant`.
The campaign is now a SQLite row per tenant in `<state_dir>/portal.db` (`campaigns(tenant, state, updated_at)`), replacing the single `~/.kalos/campaign.json`; the transactional generation token carries over inside the per-tenant `state` blob unchanged.
`/api/latest` is now keyed by tenant too (an in-memory map plus a per-tenant best-effort JSON cache under `<state_dir>/latest/`), replacing the single global `_LATEST`. (Superseded by schema v2: analyses are rows in `portal.db`, see docs/MIGRATIONS.md.)
So two tenants can never see or overwrite each other's campaign or analysis.
Follow-up: move the `latest` cache into the same SQLite store as a row, and add an `owner`/`created_by` column for finer-grained authorization. (Both done by schema v2, see docs/MIGRATIONS.md.)

**1c. Transport + secrets (in progress).**
CORS is now an explicit allowlist when `KALOS_CORS_ORIGINS` is set (the production posture) and the permissive localhost regex otherwise (`kalos/portal/config.py`); the effective security posture (auth + CORS) is logged once at startup, with a warning when the portal is not locked down.
TLS is terminated by a reverse proxy in front of the app (deployment note below); the app speaks plain HTTP on the loopback to the proxy.
Follow-up: a secrets-provider abstraction so `KALOS_*` secrets can come from a file/Vault instead of the environment, and tightening CORS methods/headers from `*` to the minimum the client needs.

**Deployment (transport):** run the app behind a TLS-terminating reverse proxy (nginx/Caddy/an ALB). The proxy holds the certificate and forwards to the app over loopback; set `KALOS_CORS_ORIGINS` to the exact browser origin(s) the proxy serves.

### Phase 2 - Reliability and scale

- Durable store already survives restarts; add backups and a documented restore.
- Move the CPU-bound `_analyze` off the request path onto an async job queue so a long GP fit never holds a worker; surface job status.
- High-availability deployment notes (stateless app tier + shared store).
- Revisit the exact-GP `MAX_FIT_ROWS=2000` cap with a sparse/variational surrogate for large historical datasets.

### Phase 3 - Operability

- Structured, correlatable request logging (never logging secrets or raw client data).
- Health/readiness endpoints, metrics, and alerting hooks.
- A documented deployment (container + config) and runbook.

## Configuration reference

| Variable | Meaning | Default |
| --- | --- | --- |
| `KALOS_AUTH_TOKENS_FILE` | Path to a JSON file of provisioned principals (see below). Takes precedence over the env var. | unset |
| `KALOS_AUTH_TOKENS` | Inline JSON array of provisioned principals. | unset |
| `KALOS_CORS_ORIGINS` | Comma-separated explicit CORS allowlist (e.g. `https://app.acme.com`). When set, replaces the permissive localhost default. | unset (dev localhost) |
| `KALOS_ENV` | Accepted values: `production` or `development` (case-insensitive, surrounding whitespace ignored); unset or empty means `development`. Any other value (e.g. `prod`, `staging`) refuses to start with `ProductionConfigError` naming the accepted values: a typo must not silently drop the production guards. See "Production mode" below. | unset (= `development`) |
| `KALOS_MAX_REQUEST_BYTES` | Body-size cap, in bytes, for every non-upload route (the JSON `/api/campaign/*` bodies). Upload routes (`/api/run`, `/api/scale/readout`) are capped at `KALOS_MAX_UPLOAD_MB` + 1 MiB multipart allowance instead. Must be a positive integer. | `1048576` (1 MiB) |
| `KALOS_JOBS_ENABLED` | `0` turns the background-job queue off: `POST /api/campaigns/{id}/jobs` answers 503 and nothing runs. Job status still reads. (docs/JOBS.md) | `1` |
| `KALOS_JOB_WORKERS` | Analyses running at once in this engine, sync and background together. Must be >= 1. (docs/JOBS.md) | `2` |
| `KALOS_JOB_PER_TENANT` | Analyses one tenant may run at once. With the default one tenant can never hold every slot. Must be >= 1. | `1` |
| `KALOS_JOB_QUEUE_DEPTH` | Queued + running jobs one tenant may hold; more answers 429. Must be >= 1. | `8` |
| `KALOS_JOB_MAX_UPLOAD_BYTES` | Largest upload a job accepts, per tenant. Must be <= the engine's upload cap. | the upload cap |
| `KALOS_BUILD_SHA` | Commit the engine was built from, recorded as `run.build_sha` on every analysis. CI passes `github.sha` at image build (`deploy/engine/Containerfile`). | `unknown` |
| `KALOS_JOB_TIMEOUT_S` | Wall-clock limit for one job; past it the worker is killed and the job is `timed_out`. Must be >= 1. | `1800` |

When neither `KALOS_AUTH_TOKENS_FILE` nor `KALOS_AUTH_TOKENS` is set, the API runs in **open mode** (anonymous `default` tenant, read+write, admin withheld) and logs a warning at startup.

### Production mode (`KALOS_ENV=production`)

Decided once at startup. In production the engine:

- **Refuses to start without enforced authentication.** Importing `kalos.portal.app` (and `python -m kalos.portal`) fails with `ProductionConfigError` unless `KALOS_AUTH_TOKENS[_FILE]` holds at least one token. This applies on every bind, loopback included, and `KALOS_ALLOW_OPEN_ACCESS` does **not** override it. A token list of `[]` counts as no auth.
- **Fails closed at runtime.** Tokens are re-read per request (rotation without restart), so if the token config is emptied after startup every request except `/healthz` and `/readyz` gets a 503, local callers included, instead of silently reopening as anonymous read+write. This holds at two layers: the production middleware refuses the request, and `Authenticator.principal_for` (the `require_principal` dependency) itself raises 503 `authentication is not configured` in production when no tokens load, so a file emptied between the middleware check and the dependency cannot yield the anonymous principal.
- **Disables the API docs and schema:** `/docs`, `/redoc` and `/openapi.json` are 404.
- **Does not serve the demo routes** `/api/single` and `/api/multi` (404). They run CPU-heavy GP loops on a synthetic surface outside the single analysis slot (`kalos/portal/busy.py`), so in a shared deployment any token holder could tie up the CPU with them.

Development (the default) keeps all of the above as before.

### Tenant ids

A provisioned principal's `tenant` must match `^[A-Za-z0-9_-]{1,64}$`. A config containing any entry that does not match is **rejected as a whole** (every authenticated request answers 500 "authentication is misconfigured"; the error is logged server-side without echoing the id). Ids are never rewritten or sanitised, because sanitising maps distinct ids (`acme/bio`, `acme_bio`) onto one storage key. The campaign store and the `/api/latest` helpers take `tenant` as a required keyword (no `default` fallback) and validate it too.

The tenant id `default` is **reserved for open/anonymous mode**. A provisioned token whose `tenant` is `default` is refused the same way (the whole config fails closed), because it would inherit whatever anonymous callers stored there before auth was turned on. Open mode keeps using `default`. Upgrade note: a deployment that provisioned `"tenant": "default"` (the old `deploy/.env.example` suggested it) must rename that tenant before upgrading, or every authenticated request answers 500 and a production start fails; data already stored under `default` stays there and is not reachable by any token.

The per-tenant `/api/latest` cache file is named `t-<hex of the tenant id's UTF-8 bytes>.json`, an injective encoding that also cannot collide on case-insensitive filesystems. Upgrading orphans cache files written under the old sanitised names: a tenant's `/api/latest` reports `has_data: false` until its next upload or reanalyze. The campaign itself (SQLite) is unaffected.

### Request limits

- Body size: a pure-ASGI middleware (`kalos/portal/limits.py`) answers **413** for a `Content-Length` over the route's cap without reading the body (a `Content-Length` of more than 19 significant digits is 413 without being parsed), **400** for a malformed or conflicting `Content-Length`, and counts chunked bodies (no `Content-Length`) as they stream, failing with **413** once they cross the cap. The chunked case uses flag+disconnect (the inner app is told the client disconnected, its answer is discarded, the middleware sends the 413), because raising from `receive` was turned into a generic 400 by FastAPI's body parsing and `BaseHTTPMiddleware` on the real app. Upload paths are matched with the ASGI `root_path` stripped, so a prefix-mounted deployment keeps the upload cap.
- Upload auth precheck: with authentication enforced, a `POST` to an upload route (`/api/run`, `/api/scale/readout`) without an `Authorization: Bearer <token>` header is answered **401** before the multipart body is read. It only checks that a bearer credential is present; the token is still verified by `require_principal`, so a wrong token is the dependency's usual 401 (after the body is read, bounded by the cap above).
- `POST /api/campaign/start`: at most 64 recipes per call; each recipe value is, by shape, a finite JSON number (bools refused), a string of at most 512 characters, or null. The shape check alone does NOT refuse numeric strings (`"7.3"` is a valid string); the store then checks every value against the kind of column it lands in, read from the campaign's dataset (`kalos/portal/recipe_values.py`): a numeric column takes only a finite number (a numeric string is refused, except the exact label of an observed value such as `"200"` for an observed `200`, which is what the engine proposes for a numeric column declared categorical, and is stored as the number); a categorical column takes only one of its observed levels or null (an unknown level is refused, not added); a column with no values takes only null; recipe keys are 1-512 characters and must be columns of the campaign's dataset other than the target; `pred`/`std` are finite numbers or null; `mode`/`reason` are at most 512 characters. Unknown extra fields are ignored and not stored.
- `POST /api/campaign/result`: `id` is 1-512 characters; `value` is a JSON number (a non-finite value is still refused by the store).
- Validation failures on `/api/campaign/*` answer 400 with the usual `{"error": ...}` envelope naming the field (dict keys masked, input values never echoed). Other routes keep FastAPI's 422.

### Probes

`/healthz` is an `async` handler so it answers from the event loop and cannot queue behind worker threads busy with analysis. `/readyz` builds the campaign store and runs a tenant-agnostic read of the schema version (`SELECT MAX(version) FROM schema_migrations`, never a tenant table), both inside one try; on any failure (including a store that cannot be opened) it answers 503 with the fixed reason `campaign store unavailable` and logs the exception server-side (no exception text in the unauthenticated response).

Not covered here: per-tenant concurrency limits (design pending).

### Provisioned-principal shape

```json
[
  {
    "token_sha256": "9f86d0818188...b0f00a08",
    "subject": "svc-lab-ingest",
    "tenant": "acme-bio",
    "scopes": ["read", "write"]
  }
]
```

Generate a token and its hash out of band, store only the hash here, and hand the raw token to the caller once:

```bash
TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
printf '%s' "$TOKEN" | sha256sum          # store this hex under token_sha256
echo "give this to the caller once: $TOKEN"
```

## Next backbone (ranked)

A review of progress against the readiness scorecard surfaced that campaign + latest were tenant-scoped but the **experiments store was still global and ungated** - a live cross-tenant leak. That is fixed (P1 below). Remaining engineering backbone, in dependency order:

- **P1 - Tenant-scope + auth-gate the experiments store: DONE** (`feat/hardening-experiments-tenancy`). `kalos/store/sqlite_store.py` gains a `tenant` column (+ index + one-time backfill migration) and every query filters by it; `kalos/portal/experiments.py` routes require `read`/`write` and pass `Principal.tenant`; `LocalStoreAdapter` is tenant-bound so the runner only touches the caller's experiments. Remote-runner `/result` still operates on the `default` tenant (machine channel, off by default) - multi-tenant remote runners are a follow-up.
- **P2 - Observability**: `/healthz` (liveness) + `/readyz` (`SELECT 1` against the DBs), a request-id + JSON logging middleware that never logs bodies/tokens, optional `/metrics`. Moves the Operational-maturity dimension off BLOCKING; low cost, high review value.
- **P3 - Async job queue for `_analyze`: in review** (`feat/background-jobs`, docs/JOBS.md). A `jobs` table (schema v3) and a submit/poll/cancel API for upload analyses. Reanalyze is not a job yet.
- **P4 - Move `_LATEST` into SQLite: DONE by schema v2** (`feat/campaign-schema-v2`, docs/MIGRATIONS.md): an `analyses` row per campaign, written in the fold's transaction. Original plan: a `latest(tenant, state, updated_at)` row in `portal.db`, retiring the per-tenant file cache; lets the reanalyze latest-write carry a generation stamp in one transaction, closing the documented sub-ms race. Prerequisite for P5 (one DB file = complete tenant state).
- **P5 - Backups + restore/DR**: scheduled SQLite online `.backup()` into `KALOS_BACKUP_DIR` + a documented restore runbook. Do P4 first so a snapshot captures all state.

### One engine per state directory

`python -m kalos.portal` takes an exclusive, non-blocking `flock` on
`<KALOS_STATE_DIR>/engine.lock` before it serves. A second engine on the same
state directory exits with status 2 and a message naming the holder's PID,
instead of sharing `portal.db` and the latest-analysis cache with the first.
The kernel drops the lock when the process dies, so a crash leaves nothing
stale: a SIGKILLed engine leaves `engine.lock` on disk with its dead pid in it,
and the next engine takes the lock anyway (`tests/test_engine_lock.py`).

Exit status 2 has two reasons, and the message says which:

- **held**: another engine process owns the directory (`already holds`).
- **unsupported**: the lock could not be taken for any other reason. A
  filesystem that refuses flock (some network mounts) reports `could not lock
  ... flock`; a platform without POSIX `fcntl` says so. Neither means another
  engine is running. Use a local directory for `KALOS_STATE_DIR`.

Limits, stated plainly:

- **Use `python -m kalos.portal`.** It is the one entry point that takes the
  lock. `uvicorn kalos.portal.app:app --workers N` (N > 1) starts N processes
  that do not take it, so they share `portal.db` without the guard. Run one
  process per state directory, or give each engine its own directory.
- The lock is per host. A state directory on a network filesystem shared by two
  hosts is not protected by flock; do not share one.
- Job worker processes do not take the lock (they are children of the engine).

Implementation: `kalos/portal/engine_lock.py`.

### What `anonymize=true` covers, and what it does not

`anonymize` is a presentation control on the analysis, not an at-rest
protection. Exactly this is covered:

- **Identifier column names** (`is_identifier_name` in
  `kalos/portal/analysis.py`: the profile's `id_hint`, the run-id pattern, and a
  word-level rule, so `Tank ID`, `Patient ID`, `Customer Name` and `run_number`
  all count). Person-name tokens count anywhere in a name, as whole words:
  `operator`, `analyst`, `technician`, `tech`, `email`, and `name` after a subject
  (`Donor_Name_1`, `Lab_Tech`, `Operator`, `email`). Process variables do not
  (`Methanol`, `pH`, `sample_volume_L`, `technology_level`). Identifier names are
  replaced by `col_<hash>` in the response: `provenance`, `target`, `group_col`,
  `features`, `categorical_features`, `proposal_features`, `drivers`, `correlation`
  labels, `gp_shapes` feature names, the response-surface axes and fixed recipe,
  every proposal recipe key, `design_box_exclusions`, `auto_categorical`,
  `data_drops`, validation findings and their prose, and refusal context and
  message. Identifier columns are not model features unless declared in `roles`.
- **Categorical level values of identifier columns** (a declared categorical such
  as `Donor_Name_1`, whose levels are people's names) are replaced by
  `level_<hash>` in the proposal recipes and the response-surface fixed recipe.
  Before, "Jane Smith 0" appeared verbatim there (review H4). Levels of ordinary
  categoricals (`Medium` = `A`/`B`/`C`) are published as they are.
- **Sheet names** are replaced by `sheet_<hash>` in `sheets`, `sheet_used`, the
  refusals (`sheet_not_found`, `no_usable_sheet`) and the analysis-error text.
- **The filename** is not stored: not in `run`, not as the campaign's `dataset`
  label (which becomes `anonymized upload <hash>`), and not in a background
  job's row beyond its extension while the job runs (docs/JOBS.md).
- A campaign **re-analysis** keeps the anonymize setting of the analysis it
  replaces.

Not covered (the response or the store still carries these):

- **`target`** (top-level and `run.target`) and **`run.roles_declared`** keep
  the caller's own column names. A target or declared role that is an
  identifier is shown as sent.
- **Process feature names and their values** (`Methanol`, `pH`, numeric
  settings, categorical levels of ordinary factors) keep their real names. They
  are the process inputs the owner reads. Identifier-like names are aliased as
  above; an identifier name that is not recognised by `is_identifier_name` is
  published as sent, so a naming gap is the thing to fix, not the output.
- **`campaigns.base_rows` at rest.** The campaign stores the full uploaded sheet
  (every column, including identifier columns and the raw values) in
  `portal.db`, unencrypted and unaltered, so that the loop can fold results into
  it. `anonymize` does not change what is stored there. Treat `portal.db` and
  the state directory as holding the raw data. Deleting or pseudonymizing
  `base_rows` is a separate decision (it would break re-analysis of existing
  campaigns).
- **Background job rows while queued or running**: target, roles and sheet are
  stored in the clear until the job reaches a terminal state (docs/JOBS.md).
- **The alias hash** is salted with `KALOS_ANON_SALT`. Without it the dev salt
  applies, and a common name (`Patient ID`) can be recognised by hashing a list
  of candidates. Production must set the salt.

## Known limitations

Each item states what the engine does, the evidence, and why it is not changed here.
These are judgement calls for the science owner; the code is unchanged.

### M5. Feasibility classifier uses `class_weight="balanced"`, which miscalibrates P(feasible)

- **What it does.** `FeasibilityClassifier` (`kalos/core/feasibility.py`) is a
  `StandardScaler` + `LogisticRegression(class_weight="balanced")`. Balanced weighting
  makes the classifier rank well and gives it the minority class equal total weight. It
  does not make its probabilities calibrated to the base rate.
- **Evidence** (synthetic, `/tmp` experiment, 5-fold stratified CV, n=400, the
  feasible class at 90% and 70%): with balanced weights the mean predicted
  P(feasible) is 0.554 against an actual 0.848 (Brier 0.215), and 0.530 against 0.660
  (Brier 0.203). Unweighted gives 0.847 against 0.848 (Brier 0.119), and 0.659 against
  0.660 (Brier 0.186). Balanced weighting therefore understates P(feasible) by
  roughly 30 points on a mostly-feasible sheet.
- **Where it matters.** The gated acquisition multiplies EI by this probability
  (`proposal_gating`), and the per-proposal `p_feasible` is reported to the client.
  Both are biased low when the base rate is high, which makes the optimizer
  over-avoid recipes that are usually fine. The AUC used by the gate is unaffected
  (it is rank-based), so the gate can pass while the probabilities are wrong.
- **Not changed** because the fix (drop balancing, or recalibrate with a held-out
  isotonic/Platt step) changes which recipes are proposed on every zero-inflated
  sheet, and that is a product decision. Tracked as a follow-up.

### M6. The promotion gate reads the POOLED Spearman, not `producer_spearman`

- **What it does.** `_analyze` passes `surrogate_spearman` = the pooled out-of-fold
  Spearman (`rho`, over producers and non-producers together) to `check_gates`. The
  producer-only Spearman (`reliability.producer_spearman`) is reported beside it but
  is not an input to the verdict.
- **Evidence** (synthetic, `/tmp` experiment, n=60, 33 producers, zeros
  for the non-producing half, producer titer pure noise on the features): pooled
  Spearman 0.781, `producer_spearman` 0.021 (`producer_clears_floor: false`), and
  `promotion.passed: true`, with no failures listed. The separation of zeros from
  non-zeros, which is a feasibility signal, carries the verdict. The engine cannot
  rank its producing recipes at all on this sheet.
- **Why it matters.** `promotion.passed` reads as "fit to promote". Under this
  sheet the producer ranking is at chance.
- **Not changed** because switching the input to `producer_spearman` changes the
  verdict on existing uploads, and the choice of which Spearman a promotion must
  clear is a product and validation decision (the same line as `clears_floor`).
  The `promotion.meaning` text does not say this; it should if the gate is kept.

### C3 follow-on: the seeded analysis section is serialized

- `_analyze` holds `_RNG_LOCK` for its whole run, so two admitted analyses (busy.py
  `KALOS_JOB_WORKERS=2`) run their seeded sections one after the other. The result is
  deterministic and identical to sequential, but the parallelism between two
  analyses is lost. A per-call `torch.Generator` through BoTorch would remove the lock;
  that is a larger refactor and is not done.
- Scale readout (`kalos/scale/readout.py`) uses torch outside `_analyze` and does not
  take the lock. Its output is not covered by the determinism guarantee.

### Concentration naming gap (C1 follow-on)

- A blank cell is read as 0 only for a column declared in `roles.sparse`, or whose
  name matches the concentration pattern (`kalos/validation/bounds.py`:
  `titer`, `conc`, `g/L`, `mg/mL`, `glucose`, `methanol`, a salt name, ...).
  `Supplement_g_L` does not match (`g_L` with an underscore is not `g/L`), so its blanks
  drop their runs. This is the safe direction, but a sheet with many such columns will
  lose rows. Declare them in `roles.sparse`, or extend the pattern deliberately.

### M1 decision: identical floor readings are exact copies (open)

- Under the M1 rule, a row identical to another in every column, target included, is a
  copy and is not a replicate. On a zero-floored assay, several runs at the same
  settings that all read exactly `0.0` are therefore copies. The zero-variance floor
  recipes are removed from the noise and ICC statistics, and the variance-mean coupling
  that the heteroscedasticity diagnostic reads (`suggests_transform`) is gone.
- Evidence: `tests/test_noise_scale_and_producers.py::test_analyze_reports_the_scale_diagnostic`
  (strict xfail). On that sheet, 13 repeated `0.0` readings are removed as copies
  (`noise.n_exact_duplicates: 13`). The diagnostic flips from `suggests_transform: true` to `false`.
- The rule is implemented as the spec states. Whether a repeated floor reading (a real
  below-LOD result) should be exempt from the copy rule is a product and assay decision.
  A row-id column would let the engine tell a real repeat from a paste; without one the
  two are indistinguishable from the data alone.

### Replicate-aware fit still counts exact copies (M1 follow-on)

- Exact copies are removed from `noise_report`, the ICC and the heteroscedasticity
  report. The replicate-aware GP fit (`aggregate_replicates`) still sees them, so a
  pasted-twice row still weights its recipe's mean and `n_reps`. The reported noise is
  correct; the fit weighting is not yet.

## Status log

- Phase 1a authentication + authorization: **done** (merged).
- Phase 1b tenant-scoped persistence: **done** (merged) - campaign -> SQLite rows per tenant; `/api/latest` per tenant; every store call keyed by `Principal.tenant`.
- Phase 1c transport/CORS: **done** on `feat/hardening-1c` - CORS allowlist via `KALOS_CORS_ORIGINS`, startup security-posture logging, TLS-via-proxy deployment note. Secrets-provider abstraction deferred.
- P1 experiments store tenant-scope + auth-gate: **done** on `feat/hardening-experiments-tenancy` - closes the cross-tenant leak the review found; the tenancy guarantee now covers all three stores (campaign, latest, experiments).
- P0 engine hardening: **done** on `security/p0-engine-hardening` - tenant-id validation + injective latest-cache names, required `tenant` keyword on the stores, `KALOS_ENV=production`, async `/healthz` and generic `/readyz` reason, body-size cap + bounded campaign bodies. Per-tenant concurrency is out of scope (design pending).
- Background jobs (P3, `feat/background-jobs`, docs/JOBS.md): **in review** - a persistent `jobs` table (schema v3), a worker process per job, a runner that admits jobs through the shared per-tenant and global limiter, status/cancel routes, per-job timeouts, per-tenant queue depth and upload limits, restart recovery, and cross-tenant isolation tests. Per-tenant concurrency, which the P0 entry above left out of scope, is now bounded by `KALOS_JOB_PER_TENANT`.
- Run record (provenance, `feat/run-record`): **in review** - every analysis carries a versioned `run` block (input hash, sheet, target, roles, seed, thread count, build sha, library versions, validation mode, shape). Anonymized runs keep no filename or sheet. docs/CAMPAIGNS_API.md, "Run record". Engine state-directory lock (`engine.lock`) in the same branch.
- P2 observability, P4 latest->SQLite, P5 backups: planned, not started (see "Next backbone" above). P4 is done by schema v2; see the entries above.
- Schema v2 (multi-campaign storage): **in review** on `feat/campaign-schema-v2` - `portal.db` gains `schema_migrations`, multi-campaign `campaigns` (`cmp_` ids, `UNIQUE(id, tenant)`), `analyses` (replacing `_LATEST` and `latest/*.json`), `tenant_defaults` (the legacy routes act on the tenant default) and `audit_log`; WAL, `foreign_keys`, `busy_timeout`; every store write is `BEGIN IMMEDIATE` + generation compare-and-swap; explicit, backed-up, reversible migration (`python -m kalos.portal.migrate up|down|status`). Runbook: docs/MIGRATIONS.md.
