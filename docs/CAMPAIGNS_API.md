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
| Audit | Create, rename, archive, unarchive, upload, start, result and reanalyze each write an `audit_log` row: tenant, subject, auth method (`bearer`, or `anonymous` in open mode), action, campaign id and outcome (`ok` or `http_<status>`). The row never holds a token, a body, a filename or uploaded data. The campaign id is left empty on a 404. Auditing is best-effort and never fails the request, except for a successful `campaign.result`: that row is written in the same transaction as the change, so a result that cannot be audited is not recorded (`500`). Its `detail` is JSON: `run_id`, `version`, `old_value` (null for the first value), `new_value`, `reason` and `actor`. |

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
| GET | `/api/campaigns?summary=true&limit=50&cursor=...` | read | The same list with per-campaign card fields, paginated (see "Campaigns list summary"). |
| POST | `/api/campaigns` | write | Create a campaign. Body: `{"name": str}`. `201` and the campaign. |
| GET | `/api/campaigns/{id}` | read | `{"campaign": <campaign>, "summary": <loop summary>}`. The summary has the same shape as `GET /api/campaign`, scoped to this campaign, including `analysis_in_sync`, `stage` and `next_action` (see "Stage" below). |
| PATCH | `/api/campaigns/{id}` | write (+admin for `archived`) | Body: `{"name"?: str, "archived"?: bool}`, at least one of the two. Returns the campaign. |
| POST | `/api/campaigns/{id}/upload` | write | Upload a run sheet into this campaign (multipart; `replace=true` confirms replacing the campaign's runs, see below). Same contract as `/api/run`. |
| GET | `/api/campaigns/{id}/analysis` | read | This campaign's published analysis, in the `/api/latest` shape (`{"has_data": false}` before the first upload). |
| POST | `/api/campaigns/{id}/runs` | write | Start runs from a proposed batch. Same body and bounds as `/api/campaign/start`. Returns `{"started": [...]}`. |
| POST | `/api/campaigns/{id}/runs/{run_id}/result` | write | Body: `{"value": number, "reason"?: str}`. Records a measured outcome and returns the run. A correction needs `reason` (see "Recording results"). |
| POST | `/api/campaigns/{id}/reanalyze` | write | Fold measured runs, re-run the engine and publish the new analysis. Returns `{"analysis": ..., "campaign": <summary>}`, where `campaign` is this campaign's summary with `stage` and `next_action`. |

### Stage

`summary` (on `GET /api/campaigns/{id}`, and `campaign` on `POST .../reanalyze`) carries
`stage` and `next_action`, exactly as `GET /api/campaign` does (docs/CAMPAIGN_LOOP.md,
"GET /api/campaign response"). They are derived from this campaign's own pending runs
and its own published analysis, never from another campaign or the tenant's default.
A campaign that was created but never uploaded reads `stage.current: "setup"`.
Each pending run carries `round`, the 0-based round it was started in (`null` for runs
started before the field existed).

### Campaigns list summary

`GET /api/campaigns?summary=true` (scope `read`) returns one page of the tenant's
campaigns, newest-updated first, each a superset of the campaign object:

`has_data` (base rows > 0), `target`, `target_unit` (`{unit, source: user|header|none,
header, note}`, the "Target unit" rule below applied to the raw `target` and the setup's
`target_unit`; a display label, nothing is converted), `n_base`, `round`, `n_awaiting`, `n_measured`,
`goal` `{direction: maximize|minimize, source: setup|default}` (from settings kind
`setup`, `value.goal`; anything else is `maximize` with `source: default`),
`best_measured` `{value, basis: "measured", direction: max|min, reason?}` (best MEASURED
target over the base rows, i.e. uploaded rows plus folded results; measured runs not yet
folded by a reanalyze are not included; `reason` explains a `null` value),
`analysis` (`{present: false}`, or `{present: true, updated, in_sync, engine_version,
cv_spearman, conformal_q, conformal_coverage, reliability: {clears_floor, ci_excludes_zero,
spearman, ci95, spearman_floor, producer_clears_floor}, n_proposals}`, values copied as
stored, `null` when the stored analysis predates a field), `current_stage` (settings kind
`stages`, `value.current_stage`, as stored), `latest_gate_decision` (`{gate, decision,
decided_at}` of the newest gate_decisions row, or `null`), `active_job` (`{id, kind, state,
progress, stage}` of the newest queued/running job, or `null`).

Pagination: `limit` 1..200 (default 50), `cursor` = the previous page's `next_cursor`
(opaque; `null` on the last page). Ordering is `(updated_at desc, id)`; the cursor is a
keyset position, so a campaign updated while a client pages moves to the front and can be
missed or repeated by that walk: restart from the first page to refresh. A malformed
cursor is `400 {"code": "invalid_cursor"}`; a `limit` out of range is `400 invalid_body`;
`limit` or `cursor` without `summary=true` is `400 {"code": "invalid_query"}`. Without
`summary=true` the response is exactly the unpaginated list above.

The page costs a constant number of SQL statements (SQLite JSON functions read the few
fields from the stored blobs); every statement is tenant-scoped.

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

A campaign that has awaiting or measured runs is refused with `409` and
`"code": "campaign_has_runs"`, before the file is read, unless the form carries
`replace=true`. Nothing is changed on refusal. Clients must confirm the replacement with the
user before they send `replace=true`. (The kalos-web client does not do this yet:
its analyze and validate pages post to `/api/run` without the field, so they
receive the 409 as a rejected upload.)

On success the upload replaces this campaign's state and discards its pending
runs, and publishes the analysis in the same transaction (docs/CAMPAIGN_LOOP.md,
"Seeding"). It never touches the tenant's default campaign or `/api/latest`.
A run added while the fit runs is not overwritten: the publish is a
compare-and-swap on the generation the check saw, so the upload answers `409`
(`campaign_changed`) and nothing is written.

The upload checks for a missing, foreign or archived campaign before any fit
runs. If the campaign is archived while the analysis is running, the upload
returns `409` and nothing is stored.

Uploads through this route run synchronously. To queue the same upload as a
background job (202, polled by id, cancellable, with per-tenant limits), use
`POST /api/campaigns/{id}/jobs` (see "Background jobs" below and docs/JOBS.md).

### Recording results

Recorded results are append-only (schema v5, docs/MIGRATIONS.md, "Schema v5").
A result is never overwritten in place.

- Every submission for a run is a new row in `run_results`: the value, the value it
  replaces (`previous_value`, `null` for the first), the reason, the actor (the
  caller's subject, `anonymous` in open mode), the auth method and the server time.
  Rows are numbered by `version`, from 1.
- The first value needs no `reason`. Any later value for the same run is a
  correction and needs a non-blank `reason` (at most 256 characters). Without one the
  response is `400` with `"code": "reason_required"` and nothing is written.
- The campaign's current value for the run is the latest one. The fold and the
  summary use it, as before.
- A successful submission writes a `campaign.result` audit row with the change in its
  `detail` (see the Audit row in the table above). A refused one writes a row with
  `outcome` `http_400` and no detail.
- Each response carries the fields below for this submission. The fields are added
  to the run object; no existing field changed.

| Field | Meaning |
|---|---|
| `result_version` | `1` for the first value, then `2`, `3`, ... |
| `previous_result` | The value this submission replaced, or `null`. |
| `result_reason` | The reason as stored (trimmed), or `null`. |
| `result_actor` | The actor the row records. |
| `result_recorded_at` | The server time, Unix seconds. This is when the engine recorded the value. |

The `reason` field of a run is still the proposal's reason (why the run was
proposed). The correction reason is `result_reason`.

The history is not yet exposed over HTTP (the audit and read API is a separate change).

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

The new analysis is labelled `campaign round <n>`. The response's `campaign` is this campaign, never the tenant's default one.

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

Determinism: **not guaranteed, even for identical runs.** Measured: an 80-row,
6-feature subset of the committed synthetic example, analysed four times in one
process with the same library stack and thread count, gave four different
proposal batches (at 4 threads and at 1; the same happens before the R2 changes,
at 23d11df). The fitted posterior agreed to about 1e-16 between runs, but the
acquisition optimum moved by whole units. Small sheets do repeat
(`tests/test_golden_determinism.py`); the 80x6 case is a recorded xfail in the
same file. The cause is not isolated yet (the usual suspect is acquisition
optimizer RNG or state that `_seeded_analysis` does not reset). A different
torch or botorch version, or a different thread count, can also change the
numbers. Compare `run` and `proposal_provenance` before comparing proposals;
equal provenance does not mean equal proposals.

### Proposal provenance

Every analysis result also carries `proposal_provenance` (additive; built by
`kalos.portal.run_record.proposal_provenance` on the upload, job and reanalyze
paths). It is stored inside the analyses JSON blob with the result, and returned
wherever the analysis is (`/api/run`, `/api/latest`, the campaign upload,
reanalyze and job results). Analyses written before it existed have no such key.

| Field | Meaning |
|---|---|
| `schema_version` | `1`. |
| `engine_version` | `kalos.__version__` (same as the top-level `engine_version`). |
| `build_sha` | As `run.build_sha`. |
| `library_versions` | `torch`, `botorch`, `gpytorch`, `numpy`, `pandas` (`null` if not installed). |
| `seed` | The fit seed (as `run.seed`). |
| `input_sha256` | As `run.input_sha256`. |
| `config_sha256` | SHA-256 of the `run` fields named in `config_fields`, as canonical JSON (sorted keys, no spaces). |
| `config_fields` | `kind`, `target`, `roles_declared`, `profile`, `anonymize`, `seed`, `torch_threads`, `max_fit_rows`, `validation_mode`, `sheet_index`, `header_row`, `pending_runs`, and `proposal_fit_restarts` (from `model_diagnostics.restarts`). |
| `model_artifact_stored` | Always `false`: no fitted model, weights or pickle is stored. A refit needs the input, the config, the seed and the same stack. |
| `provenance_sha256` | SHA-256 over every field above (not `timestamp` / `created_at`). The same input, config, seed, engine and libraries give the same value. It identifies the inputs, config and stack; it is not a guarantee that the proposals repeat (see "Determinism" above), and not a signature. |
| `timestamp`, `created_at` | When the analysis ran: epoch seconds (the result's `timestamp`) and the same instant as UTC ISO 8601. |

Each run added by `start` (`POST /api/campaign/start`, `POST
/api/campaigns/{id}/runs`) carries `analysis_provenance`: `provenance_sha256`,
`input_sha256`, `config_sha256`, `engine_version`, `seed` and `timestamp` of the
campaign's published analysis at start time, or `null` when there is none (or it
predates the block). The recipe is caller input: this records which analysis was
published, not that the recipe came from it. It is stored in the campaign state
and returned in the run, in `GET /api/campaign` and in `GET /api/campaigns/{id}`
`pending`. Runs started earlier have no such key.

### Model diagnostics

Every analysis result carries `model_diagnostics` (additive), the numerical
record of the proposal surrogate's fit (`kalos/core/surrogate.py`):

| Field | Meaning |
|---|---|
| `jitter_rung` | The Cholesky jitter rung the hyperparameter fit succeeded under: `1e-8`, `1e-6` or `1e-4` (standardized output units). |
| `jitter_rung_index` | `0`, `1` or `2`. Above `0` the first rung failed and a larger one was used. |
| `jitter_added_max` | The largest diagonal addition linear_operator actually made during the fit (`0.0` when no Cholesky needed jitter). |
| `jitter_additions` | How many times jitter was added during the fit. |
| `jitter_cap` | `1e-4`. A fit that would need more is refused (`FitError`, HTTP 400). |
| `condition_number` | 2-norm condition number of the fitted training covariance `K + noise*I` (normalized inputs, standardized outputs), from its eigenvalues. |
| `condition_warning` | `true` above `condition_warn_above` (`1e8`). The fit proceeds. |
| `condition_notice` | Plain-language sentence when `condition_warning` is true, else `null`. |
| `condition_warn_above`, `condition_refuse_above` | `1e8` and `1e12`. Above `1e12` the analysis is refused (HTTP 400) with a plain-language reason. These thresholds are defaults, not validated numbers. Because the noise model keeps a floor on the noise, exact replicates with inferred noise stay well conditioned; the warning needs a tight cluster of rows with a fixed small noise, and the refusal is reached only on very large or very clustered designs. |
| `restarts`, `restart_seed` | The proposal fit runs from 3 starts (the default initialization and 2 seeded random ones; `restart_seed` = the analysis seed) up to 1,000 fit rows, and from 1 start above that. The count depends only on the row count, never on time, so the same rows get the same number of starts on any host (the fitted values themselves can still differ by host or build). It is part of `proposal_provenance.config_sha256` (`proposal_fit_restarts`). |
| `restart_mll` | Marginal log likelihood reached from each start (`null` = that start failed numerically). `null` for a one-start fit. |
| `restart_chosen` | Index of the start kept (best likelihood; a random start must beat the default by more than 1e-6). When it is `0` the model is the one-start model, unchanged. |
| `cv_fit_restarts` | `1`. The CV folds behind the trust verdict (`reliability`, `promotion`) fit with one start, so the verdict measures the one-start procedure; the served model may use 3. When `restart_chosen` is `0` the two procedures give the same model. |
| `fit_seconds` | Wall time of the multi-start fit. Measured cost of 3 starts vs 1 (4 threads, loaded M3 Pro): n=80: 1.0 s vs 0.3 s; n=300: 2.4 s vs 0.9 s; n=1000: 9.0 s vs 2.9 s. |
| `noise_fitted` | Fitted noise variance (standardized units); `null` when the noise was fixed. |
| `lengthscales` | Fitted lengthscales (normalized inputs: 1.0 = the design box width). |
| `median_nn_distance` | Median distance from each training row to its nearest neighbour (normalized inputs). |
| `boundary_flags` | Plain sentences, recorded not refused: noise at its lower bound; lengthscales above 10x the box; lengthscales at their lower bound; lengthscales shorter than `median_nn_distance` (the fit may be following noise; seen on a pure-noise target). |

### Promotion gate changes (additive fields)

- `promotion.not_judged`: list of gates that were neither passed nor failed,
  with the reason. Two cases today:
  - `ece: not enough runs to judge calibration (n=<n>, need 20)`: the classifier
    ECE is judged only from 20 pooled out-of-fold runs (`GatesConfig.min_ece_n`).
    Below that it is still reported in `feasibility`-side metrics, but does not
    pass or fail the gate.
  - `group-mean floor: <reason>`: too few recipes with 2+ rows to form the floor.
  `promotion.summary` appends `(not judged: ...)` when the list is not empty.
- Group-mean floor gate: the surrogate must beat the leave-one-row-out recipe
  mean on the same out-of-fold rows by more than a paired bootstrap's
  uncertainty (lower bound of the 95% CI of the Spearman difference above 0).
  A failure reads `surrogate does not beat the group-mean floor: ...`. The raw
  numbers are in `cv_group_mean_baseline.margin`: `difference`, `ci95`,
  `surrogate_spearman`, `group_mean_spearman` (both on the paired rows only, so
  they can differ from the headline CV Spearman), `n_paired`, `n_groups_paired`,
  `passed`, `evaluable`, `reason`. `cv_group_mean_baseline.spearman` is unchanged.
- `reliability.calibration` (interval ECE, reported, not gated) gains
  `judgeable`, `judgement` (`"not enough runs to judge calibration"` when
  10 <= n < 20, else `null`) and `judge_min_n` (20). It is still shown from 10.

### Top-k overlap follows the goal (additive)

- `cv_topk` = |true top-k INTERSECT predicted top-k| / k over the pooled
  out-of-fold predictions, where "top" means the k BEST runs for the campaign
  goal: the k highest for maximize, the k LOWEST for minimize.
  `cv_topk.direction` (`"highest"` | `"lowest"`) says which; analyses stored
  before this field always ranked highest-first.

### Distance to data (additive)

- `proposals[].nearest_training_distance`: distance from the proposed recipe to
  the nearest training row. Each continuous feature is divided by its design-box
  width (1.0 = the full width of that feature); a categorical feature adds 1
  when the level differs; Euclidean over all features. `0` = the recipe is a
  training row. A number only: the engine applies no threshold and no label.
- `training_nn_distance_median`: the same metric between training rows (median
  distance from each row to its nearest other row), for scale.

### Target unit (additive)

Analyses computed after this change carry `target_unit`:
`{"unit": str|null, "source": "user"|"header"|"none", "header": str|null, "note": str}`.
`user` = the unit from the campaign setup, as typed (wins over the header). `header` =
parsed from the target column header for well-defined patterns only (`titer (g/L)`,
`titer [mg/mL]`, `Titer g/L`, `titer_g_L`, `titer_g_per_L`, `yield_pct`); ambiguous bare
suffixes (`_L`, `_C`, `_min`, `od600`) give `unit: null`, `source: "none"`. Display label
only; values are never converted. With `anonymize=True` and a pseudonymized target header,
`header` is the alias and a header-derived unit is dropped (`source: "none"`). Older
stored analyses lack the field.

### Per-proposal intervals, quantiles, EI (additive; analyses computed after this change)

Code: `kalos/portal/analysis_parts/intervals.py`, `kalos/core/conformal.py`
(`normalized_scores`, `normalized_conformal_quantile`), levels in
`analysis_parts/policy.py` `INTERVAL_LEVELS = (0.9, 0.95)`. The legacy
`conformal_q` / `conformal_coverage` (one global 90% half-width) are unchanged.

- `interval_levels`: `[0.9, 0.95]`.
- `interval_method`: how `proposals[].intervals` were built. `method`
  `"conformal_normalized"`, `calibrated_on` `"grouped_cv_oof"` (the same
  repeat-0 out-of-fold set as `oof` and `conformal_q`), `score`
  `abs(y - mu_oof) / sigma_oof`, `quantile` (k-th smallest score,
  `k = ceil((n+1) * coverage)`, an order statistic), `sigma_source`,
  `n_calibration`, `coverage_guarantee` (approximate: OOF scores come from fold
  models), and `levels[]` with per level `coverage`, `available`; when
  available `q_norm`, `rank`, `n_calibration`, `empirical_oof_coverage`,
  `n_inside`, `coverage_by_sigma_half{low_sigma,high_sigma:{n,coverage}}`; when
  refused `code`, `reason`, `what_would_fix_it` (plus `rank`, `min_n` for
  `too_few_calibration_points`). `empirical_oof_coverage` is in-sample (the
  same scores set `q_norm`, so it is >= rank/n by construction); the sigma-half
  split is the informative part.
- Refusal codes (per level): `too_few_calibration_points` (rank > n: 0.90 needs
  9 held-out points, 0.95 needs 19), `degenerate_sigma` (an OOF sd is
  non-finite, <= 0, or < 1e-6 x the median sd; points are never dropped),
  `no_calibration_points`, `nonfinite_oof`. Per proposal also
  `degenerate_proposal_sigma`.
- Which posterior: `sigma_oof` is the fold GP's PREDICTIVE sd (latent +
  fitted observation noise). The proposal's `predictive_std` matches it: the
  surrogate's `observation_noise=True` sd (`sigma_source:
  "surrogate_fitted_noise"`), or, when the replicate-aware surrogate (fixed
  noise per recipe mean) proposed, `sqrt(latent_sd^2 + noise.noise_sd^2)` for one
  new single measurement (`"latent_plus_assay_noise_floor"`). `proposals[].std`
  stays the LATENT sd, unchanged.
- `proposals[].intervals[]`: `{coverage, method: "conformal_normalized",
  calibrated_on: "grouped_cv_oof", n_calibration, available, lo, hi, q_norm,
  empirical_oof_coverage}`; `lo = pred - q_norm * predictive_std`. Refused: `lo`
  / `hi` null plus `code`, `reason`, `what_would_fix_it`.
- `proposals[].gp_intervals[]` (always): `{coverage, method: "gp_posterior",
  posterior: "observation", note: "model-based, not held-out calibrated",
  available, lo, hi}`, Gaussian `pred +/- z * predictive_std`.
- `proposals[].quantiles`: `{basis: "gp_posterior", posterior: "observation",
  note, p05, p25, p50, p75, p95}` of the same Gaussian.
- `proposals[].expected_improvement` / `p_improve`: `{value, incumbent,
  incumbent_basis, maximize, basis: "model", method, posterior: "latent", note,
  description}`. Analytic EI / P(f > incumbent) of the LATENT posterior (`pred`, `std`).
  **`p_improve` (the "chance to beat" the best) is the probability that the
  model's latent MEAN at the recipe beats the incumbent, not the probability
  that one new measurement does**: a single run adds assay noise, which pulls
  the chance toward 0.5 (a mean 1 latent sd above the incumbent gives 0.84; with
  observation noise equal to the latent sd, one run beats it with probability
  0.76). `description` states this on every value.
  `incumbent` is the value the proposal annotation compares against:
  `best_measured_single` (`best`) or `best_reproducible_mean`
  (`noise.best_reproducible`, replicate-aware path). The batch itself is
  chosen by qLogNEI, not by this EI. `maximize` is false when the campaign
  setup goal is minimize (then EI / p_improve are for going BELOW the incumbent).

### Clones and strains (additive; analyses computed after this change)

Full contract: docs/CLONES.md. Code: `kalos/portal/analysis_parts/strain_*.py`,
`kalos/core/strain.py`, `kalos/portal/clones_api.py`, `kalos/portal/clone_registry.py`.

- **Declaring**: setup keys `strain_column`, `strain_policy` (`fixed`, the
  default once a clone column is declared, or `factor`), `strain_level`,
  `covariates` on `PUT /api/campaigns/{id}/setup` (stored only when set, so a
  setup without them keeps its hash). Upload `roles` take `strain`,
  `strain_policy`, `strain_level`, `covariates`. With no roles, a `clone` /
  `strain` / `cell line` column is detected (policy `factor`, finding
  `strain_column_detected`) and passage / generation / PDL / thaw / vial / cell
  bank columns are covariates. New `PUT setup` refusals: `clone_not_in_training`
  (a fixed level with no runs), `covariate_not_proposable` (a bound or
  constraint on a covariate), `unknown_column`. Linear constraints are accepted
  in a clone campaign (`constraints_with_categoricals` still applies otherwise).
- **Analysis fields** (on every analysis; empty or `available: false` without a
  clone): `strain_summary`, `slices_by_level`, `slices_by_level_skipped`,
  `oof_levels`, `covariates`, `drivers_skipped`, `correlation_skipped`,
  `categorical_effects`; per proposal `held` (covariates, never in `recipe`),
  `clone`, `nearest_training_distance_same_clone`, `extrapolated_for_clone`,
  `extrapolated_for_clone_features`; `drivers[].role` and `provenance[].role`.
  `gp_shapes` is now available on mixed (categorical) fits.
- **Proposal refusal**: `proposal_notice.code = "clone_not_in_training"` (no
  proposals) when the fixed clone has no runs.

| Method | Path | Scope | Purpose |
|---|---|---|---|
| GET | `/api/campaigns/{id}/clones` | read | The registry, `seen_in_data` (labels in the latest analysis; empty with a reason when it was anonymized), `unregistered_labels`, `max_clones` (200). |
| PUT | `/api/campaigns/{id}/clones/{label}` | write | Upsert one clone: `status` (`lead`/`backup`/`retired`/`candidate`) plus allowlisted label fields (`host`, `parent_cell_bank`, `construct_id`, `phenotype_label`, `product`, `passage_at_banking`, `note`). Any other key, including `sequence`, `fasta`, `smiles`, `pdb`, `uniprot`, is `400 invalid_body` ("labels only"); the value is never echoed. Audited. |
| POST | `/api/campaigns/{id}/clones/{label}/titer-history` | write | Append one MEASURED row (`basis` harvest_titer / qP / space_time_yield, finite `value`, `unit` from the allowlist matching the basis, optional scale and age). A correction needs `previous_id` and `reason`. Append-only (database triggers). Audited. |
| GET | `/api/campaigns/{id}/clones/{label}/titer-history` | read | Newest first, `before_id` cursor; rows labeled `basis_kind: "measured"`. |
| GET | `/api/campaigns/{id}/clones/summary` | read | `strain_summary` of the latest analysis joined with the registry (`levels[].registry`, `registry_order`, `unregistered_labels`). `409 no_analysis` without an analysis; `200 {available: false, reason}` without a clone; not joined for an anonymized analysis (`registry_join.applied: false`). |

All five: tenant from the principal (another tenant's campaign is the same 404),
scope checked first, default JSON body cap, writes audited without field values.
Schema v7 adds `campaign_clones` and `clone_titer_history` (docs/MIGRATIONS.md);
v8 is reserved for the Crux items.

## Not in this slice

- Background jobs for reanalyze (it still runs synchronously; upload has a job route).
- JWT principals (E2). `auth_method` will read the principal's
  `auth_method` once it exists. `tests/test_cross_tenant_isolation_jwt.py` is
  a skipped TODO stub until E2 lands.
- Deleting a campaign, changing the default campaign, and reading the audit
  log over HTTP.
