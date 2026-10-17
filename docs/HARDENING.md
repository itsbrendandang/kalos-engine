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
`/api/latest` is now keyed by tenant too (an in-memory map plus a per-tenant best-effort JSON cache under `<state_dir>/latest/`), replacing the single global `_LATEST`.
So two tenants can never see or overwrite each other's campaign or analysis.
Follow-up: move the `latest` cache into the same SQLite store as a row, and add an `owner`/`created_by` column for finer-grained authorization.

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

- Body size: a pure-ASGI middleware (`kalos/portal/limits.py`) answers **413** for a `Content-Length` over the route's cap without reading the body (a `Content-Length` of more than 19 significant digits is 413 without being parsed), **400** for a malformed or conflicting `Content-Length`, and counts chunked bodies (no `Content-Length`) as they stream, failing with **413** once they cross the cap. The chunked case uses flag+disconnect (the inner app is told the client disconnected, its answer is discarded, the middleware sends the 413), because raising from `receive` was turned into a generic 400 by FastAPI's body parsing and `BaseHTTPMiddleware` on the real app. Upload paths are matched with the ASGI `root_path` stripped (on a `/` segment boundary, the same rule as Starlette's routing), so a prefix-mounted deployment keeps the upload cap; the production probe exemption and the `/api/campaign` 400-envelope scoping use the same stripped path.
- Upload auth precheck: with authentication enforced, a `POST` to an upload route (`/api/run`, `/api/scale/readout`) without an `Authorization: Bearer <token>` header is answered **401** before the multipart body is read. It only checks that a bearer credential is present; the token is still verified by `require_principal`, so a wrong token is the dependency's usual 401 (after the body is read, bounded by the cap above).
- `POST /api/campaign/start`: at most 64 recipes per call; each recipe value is, by shape, a finite JSON number (bools refused), a string of at most 512 characters, or null. The shape check alone does NOT refuse numeric strings (`"7.3"` is a valid string); the store then checks every value against the kind of column it lands in, read from the campaign's dataset (`kalos/portal/recipe_values.py`): a numeric column takes only a finite number (a numeric string is refused, except the exact label of an observed value such as `"200"` for an observed `200`, which is what the engine proposes for a numeric column declared categorical, and is stored as the number); a categorical column takes only one of its observed levels or null (an unknown level is refused, not added); a column with no values takes only null; recipe keys are 1-512 characters and must be columns of the campaign's dataset other than the target; `pred`/`std` are finite numbers or null; `mode`/`reason` are at most 512 characters. Unknown extra fields are ignored and not stored.
- `POST /api/campaign/result`: `id` is 1-512 characters; `value` is a JSON number (a non-finite value is still refused by the store).
- Validation failures on `/api/campaign/*` answer 400 with the usual `{"error": ...}` envelope naming the field (dict keys masked, input values never echoed). Other routes keep FastAPI's 422.

### Probes

`/healthz` is an `async` handler so it answers from the event loop and cannot queue behind worker threads busy with analysis. `/readyz` builds the campaign store and runs a tenant-agnostic `SELECT 1 FROM campaigns LIMIT 1`, both inside one try; on any failure (including a store that cannot be opened) it answers 503 with the fixed reason `campaign store unavailable` and logs the exception server-side (no exception text in the unauthenticated response).

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
- **P3 - Async job queue for `_analyze`**: a `jobs(id, tenant, kind, status, result_ref, ...)` table + submit/poll API so `/api/run` and `/reanalyze` enqueue and return a job id, taking the CPU-bound GP fit off the request path and making in-flight work restart-durable.
- **P4 - Move `_LATEST` into SQLite**: a `latest(tenant, state, updated_at)` row in `portal.db`, retiring the per-tenant file cache; lets the reanalyze latest-write carry a generation stamp in one transaction, closing the documented sub-ms race. Prerequisite for P5 (one DB file = complete tenant state).
- **P5 - Backups + restore/DR**: scheduled SQLite online `.backup()` into `KALOS_BACKUP_DIR` + a documented restore runbook. Do P4 first so a snapshot captures all state.

## Status log

- Phase 1a authentication + authorization: **done** (merged).
- Phase 1b tenant-scoped persistence: **done** (merged) - campaign -> SQLite rows per tenant; `/api/latest` per tenant; every store call keyed by `Principal.tenant`.
- Phase 1c transport/CORS: **done** on `feat/hardening-1c` - CORS allowlist via `KALOS_CORS_ORIGINS`, startup security-posture logging, TLS-via-proxy deployment note. Secrets-provider abstraction deferred.
- P1 experiments store tenant-scope + auth-gate: **done** on `feat/hardening-experiments-tenancy` - closes the cross-tenant leak the review found; the tenancy guarantee now covers all three stores (campaign, latest, experiments).
- P0 engine hardening: **done** on `security/p0-engine-hardening` - tenant-id validation + injective latest-cache names, required `tenant` keyword on the stores, `KALOS_ENV=production`, async `/healthz` and generic `/readyz` reason, body-size cap + bounded campaign bodies. Per-tenant concurrency is out of scope (design pending).
- P2 observability, P3 async job queue, P4 latest->SQLite, P5 backups: planned, not started (see "Next backbone" above).
