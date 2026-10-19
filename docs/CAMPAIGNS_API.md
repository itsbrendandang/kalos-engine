# Multi-campaign API (`/api/campaigns*`)

Schema v2 (docs/MIGRATIONS.md) lets a tenant hold any number of named
campaigns. This API exposes them over HTTP. Implementation:
`kalos/portal/campaigns_api.py`. Release-gate tests:
`tests/test_cross_tenant_isolation.py`.

The legacy single-campaign routes (`/api/run`, `/api/latest`,
`/api/campaign`, `/api/campaign/start|result|reanalyze`) are unchanged. They
act on the caller's **default campaign** (`is_default: true` in the list
below), which the first legacy upload creates. A campaign made with
`POST /api/campaigns` does not become the default.

## Rules that apply to every route

| Rule | Behaviour |
|---|---|
| Authentication | Every route needs a principal (`Authorization: Bearer <token>` once tokens are configured; 401 without one). A campaign upload without a bearer token is refused with 401 before its body is read, the same as `/api/run`. |
| Scopes | `read` for GET; `write` for POST and PATCH; `admin` as well for archiving or unarchiving. Missing scope: `403 {"error": "requires '<scope>' scope", "code": "forbidden"}` (docs/HARDENING.md, "Error envelope"). Scope is checked before the campaign id is looked at. |
| Tenant | Always the authenticated principal's tenant. A tenant in the path, query, body or headers is ignored (an unknown body field is a 400). |
| Campaign ids | `cmp_` + 32 lowercase hex characters (a uuid4), validated before the store is read. |
| Not found | A malformed id, a missing id and another tenant's id all get exactly the same response: `404 {"error": "campaign not found"}` (same status, headers and body bytes). The input is never echoed. |
| Archived | An archived campaign can still be read. Every write to it gets `409 {"error": "the campaign is archived; it can be read but not changed"}`. |
| Body errors | `400 {"error": "invalid request body: <field>: <reason>"}`. The same envelope as `/api/campaign/*`, and the submitted value is never echoed. |
| Body size | JSON bodies have the default cap (`KALOS_MAX_REQUEST_BYTES`, 1 MiB). `POST /api/campaigns/{id}/upload` and `POST /api/campaigns/{id}/jobs` have the upload cap (`KALOS_MAX_UPLOAD_MB` + multipart allowance); a job also has its tenant limit (`KALOS_JOB_MAX_UPLOAD_BYTES`). Over the cap: `413`. |
| Heavy work | Upload and reanalyze share the engine's analysis limiter: a global cap and a per-tenant cap (`KALOS_JOB_WORKERS`, `KALOS_JOB_PER_TENANT`, docs/JOBS.md). When it is full for this caller they return `503` with `Retry-After`. Background jobs share the same limiter. |
| Audit | Create, rename, archive, unarchive, upload, start, result and reanalyze each write an `audit_log` row: tenant, subject, auth method (`bearer`, or `anonymous` in open mode), action, campaign id and outcome (`ok` or `http_<status>`). The row never holds a token, a body, a filename or uploaded data. The campaign id is left empty on a 404. Auditing is best-effort and never fails the request. |

## Campaign object

```json
{
  "id": "cmp_3f2a...",
  "name": "Lipase round 2",
  "created_by": "alice@acme",
  "created_at": 1767225600.0,
  "updated_at": 1767225600.0,
  "archived_at": null,
  "archived": false,
  "seeded": false,
  "is_default": false
}
```

`seeded` turns true after the first upload into the campaign.

## Routes

| Method | Path | Scope | Purpose |
|---|---|---|---|
| GET | `/api/campaigns?include_archived=false` | read | List the tenant's campaigns, newest first: `{"campaigns": [<campaign>...]}`. |
| POST | `/api/campaigns` | write | Create a campaign. Body: `{"name": str}`. `201` and the campaign. |
| GET | `/api/campaigns/{id}` | read | `{"campaign": <campaign>, "summary": <loop summary>}`. The summary has the same shape as `GET /api/campaign`, scoped to this campaign, including `analysis_in_sync`. |
| PATCH | `/api/campaigns/{id}` | write (+admin for `archived`) | Body: `{"name"?: str, "archived"?: bool}`, at least one of the two. Returns the campaign. |
| POST | `/api/campaigns/{id}/upload` | write | Upload a run sheet into this campaign (multipart). Same contract as `/api/run`. |
| GET | `/api/campaigns/{id}/analysis` | read | This campaign's published analysis, in the `/api/latest` shape (`{"has_data": false}` before the first upload). |
| POST | `/api/campaigns/{id}/runs` | write | Start runs from a proposed batch. Same body and bounds as `/api/campaign/start`. Returns `{"started": [...]}`. |
| POST | `/api/campaigns/{id}/runs/{run_id}/result` | write | Body: `{"value": number}`. Records the measured outcome and returns the run. |
| POST | `/api/campaigns/{id}/reanalyze` | write | Fold measured runs, re-run the engine and publish the new analysis. Returns `{"analysis": ..., "campaign": <summary>}`. |

### Name validation

Surrounding whitespace is trimmed. The trimmed name must be 1-120 characters
and must not contain control characters (Unicode `Cc`), surrogates, or
line/paragraph separators. It must be a JSON string. Names need not be unique:
campaigns are identified by id only.

### PATCH semantics

The whole PATCH runs in one transaction, in this order: unarchive, rename,
archive.

- `{"archived": false, "name": "x"}` unarchives an archived campaign and
  renames it.
- `{"name": "x", "archived": true}` renames, then archives.
- `{"name": "x"}` on an archived campaign is `409`.
- Archiving the tenant's default campaign clears the default. The next legacy
  `/api/run` upload then starts a fresh default campaign. Unarchiving does not
  restore the default.
- Archive and unarchive are idempotent.

### Upload

The form fields and errors match `POST /api/run`:

- Fields: `file`, `target`, `anonymize`, `roles`, `sheet`. `?sheet=` also
  works; the form field wins if both are set.
- Responses: the same `{"error", "code", ...}` error envelope and the same
  codes (`parse_error`, `fit_failed`, `corrupt_excel`, `sheet_not_found`,
  ...). Excel responses include `sheets` and `sheet_used`.

On success the upload replaces this campaign's state and discards its pending
runs, and publishes the analysis in the same transaction (docs/CAMPAIGN_LOOP.md,
"Seeding"). It never touches the tenant's default campaign or `/api/latest`.

The upload checks for a missing, foreign or archived campaign before any fit
runs. If the campaign is archived while the analysis is running, the upload
returns `409` and nothing is stored.

Uploads through this route run synchronously. To queue the same upload as a
background job (202, polled by id, cancellable, with per-tenant limits), use
`POST /api/campaigns/{id}/jobs` (see "Background jobs" below and docs/JOBS.md).

### Runs and results

- `run_id` is the 32-hex id returned by `POST .../runs`.
- An unknown or malformed run id, or a run id from another campaign or
  tenant, returns `404 {"error": "run not found"}`. That body is distinct from
  the campaign 404 and never echoes the id.
- A non-finite `value` is a `400`. So are bools and numeric strings.

### Reanalyze

Reanalyze has the same transactional contract as `/api/campaign/reanalyze`
(docs/CAMPAIGN_LOOP.md): plan the fold, analyze, then compare-and-swap commit.

- `400` when there is nothing to fold or the data cannot be analyzed.
- `409` when the campaign changed during the analysis, or is archived.
- `503` when the analysis slot is busy.

The new analysis is labelled `campaign round <n>`.

### Background jobs

| Method and path | Scope | Result |
|---|---|---|
| `POST /api/campaigns/{id}/jobs` | `write` | `202` + the job. Multipart fields as `/upload`. Refusals of the sheet are `400` before queueing. `429` when the tenant's queue is full. |
| `GET /api/campaigns/{id}/jobs` | `read` | This campaign's recent jobs, newest first. |
| `GET /api/jobs/{id}` | `read` | One job: `state` (`queued`, `running`, `succeeded`, `failed`, `cancelled`, `timed_out`), `stage`, `progress`, `error` and `campaign_generation`. |
| `POST /api/jobs/{id}/cancel` | `write` | Cancels a queued or running job. Idempotent. |

A job's id is `job_` + 32 hex characters. It is tenant-scoped exactly like a
campaign id: another tenant's job, a missing one and a malformed one all get
the same `404 {"error": "job not found"}`. Lifecycle, error codes, limits and
the restart rules are in docs/JOBS.md.

### Run record (provenance)

Every analysis result carries a `run` block (`kalos/portal/run_record.py`,
schema `run.schema_version` = 1). It says what produced the result:

| Field | Meaning |
|---|---|
| `schema_version` | `1`. Bumped when a field is added, renamed or retyped, from the first release of the record. Schema 1 has not shipped: fields added during review (`kind`, `pending_runs`, the fit and sheet fields below) stayed in version 1. |
| `kind` | `upload` (a run sheet: `/api/run`, `/api/campaigns/{id}/upload`, jobs) or `reanalyze` (`POST /api/campaign/reanalyze`, the campaign's folded base dataset). |
| `input_sha256`, `input_bytes` | SHA-256 and length of the uploaded bytes exactly as received. For `kind=reanalyze`: of the folded dataset serialized as CSV (`fold_bytes`). |
| `filename`, `sheet` | The upload's name and the sheet analyzed (Excel). **Absent** when `anonymize` is true. |
| `target` | The target the analysis used (the resolved one when the request left it blank). |
| `profile` | `bioprocess` (inferred roles) or `generic` (declared roles). |
| `roles_declared` | The `roles` object the caller sent, or `null`. |
| `anonymize`, `seed` | The request's anonymize flag and the fit seed. |
| `torch_threads` | The configured thread count (`KALOS_TORCH_THREADS`). **`null` = unpinned** (the config said 0: torch keeps its own default). |
| `torch_threads_actual` | `torch.get_num_threads()` when the fit ran; `null` if torch was never loaded. |
| `max_fit_rows` | The engine's fit-row cap (`KALOS_MAX_FIT_ROWS`) in force. |
| `sheet_index`, `sheet_hash` | Position of the analysed sheet in the workbook, and a salted hash of its name (the alias hash). Both `null` for a CSV. Kept under anonymize. |
| `build_sha` | `KALOS_BUILD_SHA` at image build (CI sets it to the commit), else `unknown`. |
| `library_versions` | `numpy`, `pandas`, `torch`, `botorch` (`null` if not installed). |
| `validation_mode` | `warn` or `strict` (`KALOS_VALIDATION_MODE`). |
| `pending_runs` | Runs in the incubator the batch was told about (`reanalyze`); `0` for an upload. |
| `n_rows`, `n_features` | Rows parsed; input features the model was fit on (`d` in the result, the kept feature columns; a categorical is one column here, not its one-hot width). Not the top drivers in `proposal_features`. |
| `gates` | Keys in this result where the gate decisions live: `promotion`, `proposal_gating`, `validation`. The decisions are not copied into `run`. |
| `data_drops` | What the fit excluded, as counts only (no column names, so it is safe under `anonymize`): `rows_dropped` (runs excluded for a missing or non-numeric target/feature cell), `censored_rows_excluded` (runs excluded for a censored value such as `<LOD` or `>100`; not imputed), `cells_by_reason` (`missing`, `non_numeric`, `censored`, counted over cells), and `auto_categorical_count`. Per-column detail is in the result's `data_drops.cells_by_column` and in `validation.findings`. |
| `header_row` | The sheet row (0-based) the column names were read from. `1` when row 0 was a merged-header group label row. `null` for a reanalysis. |

Compatibility:

- The top-level `seed` and `engine_version` keys are unchanged.
- The block is stored inside the existing analyses JSON blob. No migration.
  Rows written before it existed have **no `run` key** at all (absent, not
  `null`). Test for the key, not for a null value.
- The sync routes and background jobs attach the same block for the same bytes,
  because both call `kalos.portal.pipeline.compute_upload_analysis`.

Re-analysis (`kind=reanalyze`) keeps the anonymize setting of the analysis it
replaces: the campaign does not store the flag, so a re-analysis of an
anonymized upload is anonymized too (before this, it published the real
identifier names again).

Privacy: with `anonymize=true` the stored `run` has no `filename` and no
`sheet`, only the hash. Workbook sheet names are aliased too: `sheets` and
`sheet_used` in the response, the sheet names in refusals (`sheet_not_found`,
`no_usable_sheet`) and in the analysis-error sentence all read `sheet_<hash>`.
The alias is a salted hash (`KALOS_ANON_SALT`), so it does not let a client
choose a sheet by its alias. A background job keeps no filename under anonymize
either, only its extension, and clears its stored parameters when it reaches a
terminal state (docs/JOBS.md, "Job parameters"). The stored `dataset` label is then
`anonymized upload <first 12 hex of input_sha256>`, not the filename. The
`target` and any declared `roles` keep their real column names, as the
existing `target` field already does; anonymize only pseudonymizes the
identifier-type columns in the analysis body. Full scope, including the raw
sheet stored at rest in `campaigns.base_rows`: docs/HARDENING.md, "What
`anonymize=true` covers".

Determinism: the same bytes give the same proposals on the same library stack
and the same torch thread count (`run.torch_threads`, `run.library_versions`).
That is the only claim. A different torch or botorch version, or a different
thread count, may legitimately change the numbers. Compare `run` before
comparing proposals.

## Not in this slice

- Background jobs for reanalyze (it still runs synchronously; upload has a job route).
- JWT principals (E2). `auth_method` will read the principal's
  `auth_method` once it exists. `tests/test_cross_tenant_isolation_jwt.py` is
  a skipped TODO stub until E2 lands.
- Deleting a campaign, changing the default campaign, and reading the audit
  log over HTTP.
