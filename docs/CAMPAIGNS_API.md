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
| Audit | Create, rename, archive, unarchive, upload, start, result and reanalyze each write an `audit_log` row: tenant, subject, auth method (`bearer`, or `anonymous` in open mode), action, campaign id and outcome (`ok` or `http_<status>`). The row never holds a token, a body, a filename or uploaded data. The campaign id is left empty on a 404. Auditing is best-effort and never fails the request, except for a successful `campaign.result`: that row is written in the same transaction as the change, so a result that cannot be audited is not recorded (`500`). Its `detail` is JSON: `run_id`, `version`, `old_value` (null for the first value), `new_value`, `reason` and `actor`. A successful upload (`run.upload`, `campaign.upload`) has a `detail` line naming the declared-roles settings it set, changed or cleared, for example `settings: allow_beyond 0.1 -> 0.2; features changed`. Old -> new values are written only for `allow_beyond` (a single number), `direction`, `per_recipe_noise`, `exploration_design` and `applicability_distance`; every other setting is named by its key only, so no column name, term, limit or label reaches the audit log. |

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

- Fields: `file`, `target`, `anonymize`, `roles`, `sheet`, `direction`. `?sheet=` also
  works; the form field wins if both are set. `roles.direction` or the
  `direction` form field (`maximize`, the default, or `minimize`; two different
  values are a `400` `direction_conflict`) sets the optimisation direction
  (docs/UPLOADS.md, "Direction"); any other value is a `400`
  `invalid_direction`. The campaign stores it and reanalyze reuses it.
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
schema `run.schema_version` = 2). It says what produced the result:

| Field | Meaning |
|---|---|
| `schema_version` | `2`. Bumped when a field is added, renamed or retyped, from the first release of the record. Version 2 added `direction`, which is also part of `config_sha256`, so a version-2 `config_sha256` never equals a version-1 one for the same upload. Fields added during review before any release (`kind`, `pending_runs`, the fit and sheet fields below) stayed in version 1. |
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
| `direction` | The direction the analysis optimized: `maximize` or `minimize`. Set from `roles.direction`, the `direction` form field, or (reanalyze) the campaign's stored direction. Part of `config_sha256`, so maximize and minimize on the same bytes hash differently. |

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
| `config_fields` | `kind`, `target`, `roles_declared`, `profile`, `anonymize`, `seed`, `torch_threads`, `max_fit_rows`, `validation_mode`, `sheet_index`, `header_row`, `pending_runs`, `direction`, and `proposal_fit_restarts` (from `model_diagnostics.restarts`). |
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
| `kernel` | `rbf_ard` (continuous inputs), `rbf_ard+categorical` (mixed) or `categorical` (all categorical). |

The kernel is BoTorch's default, not Matern: an ARD RBF kernel with no output
scale, a LogNormal lengthscale prior with loc `sqrt(2) + log(d)/2` and scale
`sqrt(3)` (d = continuous inputs), and a lengthscale floor of 0.025. With
categorical inputs a CategoricalKernel is added (sum and product terms, each
scaled). Inferred noise has a `LogNormal(-4, 1)` prior and a `1e-4` floor
(standardized units). Every fit checks this (`kalos/core/gp_pin.py`) and fails
if a library upgrade changes it. The model was not rebuilt by hand, so
proposals did not change.

### Promotion gate changes (additive fields)

- `promotion.not_judged`: list of gates that were neither passed nor failed,
  with the reason. Three cases today:
  - `ece: not enough runs to judge calibration (n=<n>, need 20)`: the classifier
    ECE is judged only from 20 pooled out-of-fold runs (`GatesConfig.min_ece_n`).
    Below that it is still reported in `feasibility`-side metrics, but does not
    pass or fail the gate.
  - `group-mean floor: <reason>`: too few recipes with 2+ rows to form the floor.
  - `feasibility_auc: not applicable when minimizing` (and the same for `ece`
    and `brier`): the feasibility classifier scores P(y > 0), a maximize
    notion, so under `direction: minimize` its three gates do not pass or fail.
    The surrogate Spearman and group-mean floors still apply.
  `promotion.summary` appends `(not judged: ...)` when the list is not empty.
- `promotion.status`: `passed`, `failed` or `not_measurable`. `failed` means a
  measured gate missed its floor or ceiling. `not_measurable` means every gate
  that blocked could not be measured on this data (for example
  `feasibility_auc` on a sheet with no non-producers). The verdict is the same
  as `passed`; `not_judged` gates do not block and are not counted here.
- `promotion.unmeasured`: the gate keys that blocked only because their metric
  was missing or NaN.
- `promotion.checks`: one entry per check, in this order:
  `surrogate_spearman`, `group_mean_margin`, `feasibility_auc`, `ece`, `brier`.
  Each is `{code, state, checks, why, advice, optional}`. `state` is `passed`,
  `failed`, `not_measurable` (the check could not run on this sheet) or
  `not_judged` (too little data; it does not block). `checks`, `why` and
  `advice` are plain sentences (`advice` is `null` when there is nothing to
  do). The texts live in one table, `kalos/portal/gate_text.py`; codes match
  the assistant's glossary. `feasibility_auc`, `ece` and `brier` are
  `optional: true`: they only matter if some recipes can fail outright, and
  the advice never asks for failed runs on purpose.
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

### Distance to data (additive)

- `proposals[].nearest_training_distance`: distance from the proposed recipe to
  the nearest training row. Each continuous feature is divided by its design-box
  width (1.0 = the full width of that feature); a categorical feature adds 1
  when the level differs; Euclidean over all features. `0` = the recipe is a
  training row. A number only: the engine applies no threshold and no label.
- `training_nn_distance_median`: the same metric between training rows (median
  distance from each row to its nearest other row), for scale.

### Applicability flags (additive)

Two separate flags per proposal. A proposal can be near the data and on the
box edge, or far from the data in the middle of the box.

- `proposals[].applicability`:
  - `distance`: the same number as `nearest_training_distance`.
  - `threshold`, `threshold_source` (`default` or `lab`).
  - `far_from_data`: `distance > threshold`.
  - `at_box_edge`: list of `{feature, side}` for each continuous input within
    1% of the box width of a bound (`side` is `lower` or `upper`). Empty when
    none. The box is `feature_ranges`.
- `applicability` (top level): `threshold`, `source` (`default` or `lab`),
  `set_by` (`default`, `env` or `roles`), `rule` (in words) and
  `edge_fraction` (0.01).

The default threshold is `max(0.25, 3 x training_nn_distance_median)`. It is a
product default, not a tested threshold. A lab sets its own with
`roles.applicability_distance` (a positive number; anything else is a `400`
`invalid_applicability_distance`) or `KALOS_APPLICABILITY_DISTANCE`. Roles
wins. A reanalyze keeps a roles-set threshold.

Proposed values are rounded to 3 places and then clipped into the box, so a
recipe never sits outside `feature_ranges` (replay F9).

### Campaign rows (additive route)

`GET /api/campaigns/{id}/rows` (read scope; 404 outside the caller's tenant,
byte-identical to a missing id) returns every row the campaign's model learns
from, in order: the uploaded sheet, then each measured run folded in by a
reanalyze. Fields: `target`, `columns`, `n_rows`, `n_upload`, `n_from_rounds`,
`offset`, `limit` (query `offset`, `limit` 1-10,000, default 2,000), `rows`,
`kinds`, `kinds_note`. Each row carries its values plus `_origin` (`upload` or
`round`), `_round` (0 for the upload, 1 for runs started before the first
reanalyze, ...; null when the campaign was folded before this was recorded),
`_run_id` (the started run, or null), `_upload_row` (1-based position in the
uploaded sheet, upload rows only), `_kind` (`measured`, or `no_result` for an
upload row whose target cell is blank, which the model does not learn from;
simulated and synthetic rows are not stored in a campaign) and `_index`.
A sheet column named `_origin`, `_round`, `_run_id`, `_kind`, `_index` or
`_upload_row` is shown as `<name> (sheet)` (`(sheet 2)`, ... if that is taken),
so neither it nor the label hides the other; `renamed_columns` lists
`[{column, shown_as}]` and `renamed_note` says so in plain words (null when
nothing was renamed). `columns` uses the shown names.

### Preflight (additive route)

`POST /api/preflight` takes the same multipart fields as `POST /api/run`
(`file`, `target`, `roles`, `sheet`, `anonymize`, `direction`; write scope) and
answers how Kalos reads the sheet without fitting a model: `preflight: true`,
`target`, `targets`, `features`, `categorical_features`, `group_col`,
`fixed_inputs`, `n_rows`, `n_usable`, `n_dropped_incomplete`,
`n_censored_excluded`, `header_row`, `orientation`, `provenance`, `columns`
(see "Column kinds"), `validation`, `contract_version`, `engine_version`, and
for workbooks `sheets` / `sheet_used`. It refuses exactly what an upload
refuses, with the same 400 codes (including `infeasible_linear_constraints`,
checked on the search box). Nothing is stored, no model is fit, torch is
not loaded, and the analysis slot is not taken, so it answers in about a
second. The fit-dependent findings (`search_box`, `cv_ci_unavailable`) only
come with the analysis.

### Column kinds (additive)

`columns`: one entry per uploaded column, in sheet order.

```json
{"name": "CarbonSource", "kind": "category", "role": "excluded", "used_in_fit": false,
 "n_values": 213, "n_levels": 19,
 "levels": [{"label": "src_12", "count": 17}, {"label": "src_10", "count": 14}],
 "finding_code": "category_not_modeled", "reason": "too_many_levels"}
```

- `kind` describes the values: `numeric`, `category` (text labels that repeat),
  `text_id` (about one value per row, or an identifier name), `row_number`,
  `outcome_like` (measured during or after the run), `constant` (one value or
  none on the fitted rows). The target is `numeric`.
- `role` says what the engine did: `target`, `input`, `categorical_input`,
  `group` (a recipe label that groups replicates), `fixed` (a near-constant
  input held at its median), `excluded`. `used_in_fit` is true for `target`,
  `input` and `categorical_input`.
- `levels` (category only): label and row count, most frequent first, at most
  50; `n_levels` is the full count. Null for other kinds.
- A category with at most 12 levels, or one declared in `roles.categoricals`,
  is a categorical input: the model is a mixed categorical + continuous GP and
  `proposals[].recipe` carries the label text. A category with more levels that
  was not declared is excluded, with `finding_code` `category_not_modeled`
  (the same code as its `validation.findings` entry) and `reason`
  `too_many_levels`, `mixed_text_and_numbers` or `too_few_repeats`.
- A client should draw `category` columns on a categorical axis using
  `levels`, never on a numeric one.
- With `anonymize=true`, identifier-like names are aliased and so are the level
  labels of those columns.
- `kind` is a closed list: `numeric`, `category`, `text_id`, `row_number`,
  `outcome_like`, `constant`, `date` (dates or times; never binned as numbers).
- `levels` is present for every `category` column, used or excluded.
- `name` is the same string as `provenance[].name` (a headerless column is
  `column N` in both). An entirely blank column has no `provenance` row but is
  listed in `columns`, in its sheet position: `kind` `constant`, `reason`
  `all_blank`, `used_in_fit` false, `n_values` 0. `provenance`, `features`, `categorical_features` and
  `feature_ranges` are unchanged; `columns` is additive.
- `finding_code` is null or one of `category_not_modeled`,
  `possibly_measured_after_run`, `outcome_like_name`, `after_target_in_sheet`,
  `row_number_column`, `headerless_column` (the code of the finding about this
  column). A column left out by a rule has `used_in_fit` false, its `reason` and
  its `finding_code`, so a client can show it unticked and let the lab tick it
  (declare it in `roles.features`).
- `reason` is null or one of these codes (closed list; the same codes appear in
  `provenance[].reason`):

| code | meaning |
| --- | --- |
| `declared` | the role came from `roles` |
| `requested_target` | the column named as the target |
| `name_matches_target_candidate` | its name reads like an outcome |
| `last_numeric_column` | chosen as target because it is the last number column |
| `numeric_input` | a number input |
| `categorical_text` | a text input fitted as a category |
| `concentration_before_target` | a concentration before the target, kept as an input |
| `headerless` | no header and one number per row: a row index |
| `monotone_row_number` | a counter up or down from any start, at most a few numbers missing: a row index |
| `after_target_in_sheet` | after the target and named like a measurement |
| `outcome_like_name` | named with an outcome word (titer, yield, product, purity, final without a concentration unit, ...) |
| `name_matches_identifier` | its name reads like an id (sample, operator, ...) |
| `constant` | one value |
| `constant_on_fitted_rows` | one value on the rows the model uses |
| `sparse_or_unparsed` | too few usable values |
| `declared_non_numeric` | declared as an input but not numeric |
| `all_blank` | no values |
| `recipe_label` | a label naming each recipe: groups replicates |
| `near_constant` | spread within sensor precision: fixed at its median |
| `not_declared` | roles listed the inputs and this one was not listed |
| `too_many_levels` | a category with more than 12 labels, not declared |
| `mixed_text_and_numbers` | labels and numbers in one column |
| `too_few_repeats` | labels repeat too rarely to learn from |
| `unique_per_row` | text with about one value per row |
| `date_or_time` | dates or times |
| `possibly_measured` | kept as an input, but the name (OD, biomass, lactate, ammonium, ...) reads like something measured during the run; check it |

### Verdict

Every analysis carries `verdict`: one answer to "can I trust the model's
suggestions?". `reliability` and `promotion` are unchanged; `verdict` is
read from them. The campaign summary (`GET /api/campaign`,
`GET /api/campaigns/{id}` `summary`, reanalyze `campaign`) returns the stored
analysis's own `verdict` object, never a recomputed one (`null` for an analysis
written before this field).

Fields: `state` (`reliable`, `not_yet`, `refused`), `code` (`reliable`,
`not-yet`, `insufficient`, `no_ordering`), `headline` (one plain sentence),
`reasons` (`[{code, plain}]`, empty when reliable; codes `too_few_rows`,
`too_few_groups`, `surrogate_spearman`, `cv_ci_unavailable`), `ordering`
(`{state: ok | unsure | weak | absent | unchecked, plain}`), `range_width`
(`{state: ok | too_narrow | too_wide | unchecked, code: bands_*, plain}`,
from `reliability.calibration.z_std`: above 1.15 too narrow, below 0.85 too
wide; reported, not part of the state), `n_runs`, `min_runs` (20),
`n_groups`, `min_groups` (8), `rule`.

"Score" is the held-out rank agreement (`reliability.spearman`), "interval" its
95% interval (`reliability.ci95`), "groups" the CV groups (`cv_n_groups`).
Checked top to bottom:

| state | code | rule | plain sentence |
|---|---|---|---|
| `refused` | `insufficient` | no score, or fewer than 3 groups (no interval) | Kalos cannot suggest recipes from this sheet: there are too few groups of runs to check whether the model is any good. |
| `refused` | `no_ordering` | upper end of the interval at or below 0 | Kalos cannot suggest recipes from this sheet: the model does no better than chance at putting your past runs in order. |
| `reliable` | `reliable` | score at or above 0.20, lower end of the interval above 0, at least 20 runs and 8 groups | Kalos put your past runs in the right order, with enough runs to check the ranking, so its suggestions are worth running. |
| `not_yet` | `not-yet` | everything else | Kalos cannot yet show that it puts your past runs in the right order, so its suggestions are closer to guesses. When only the run or group count holds it back: "The model put your past runs in the right order, but N runs in G groups are too few to be sure, so treat its suggestions as ideas to test." |

Why 20 runs and 8 groups: at 13 runs a score of 0.5 has an interval of about
-0.06 to 0.82, so one lucky fold can carry it; 20 is also where the engine
starts judging the likely range. 8 groups leaves the 5-fold grouped check at
least one group per fold.

When `state` is `refused` the analysis has `proposals: []`, `refusal:
{code, plain, what_would_help}` and `proposal_notice: {code: "model_refused",
reason, requested, returned: 0, message}`. Otherwise `refusal` is `null`. The
campaign stage then stays on `plan` with the plan step "No recipes proposed"
and `next_action.action` "Add runs and upload again". Starting an empty batch
returns `{"started": []}`.
### Analysis identity

Each analysis has an `analysis_id` (for example `an_3f9c0d1e2b4a5c6d`). It is
made when the model is fit and stored with the analysis. `/api/run`,
`/api/latest`, `/api/campaign`, `/api/campaigns/{id}`,
`/api/campaigns/{id}/analysis`, the reanalyze response and the job view show
the same id for the same stored analysis. `/api/run` returns the stored
analysis, so it has the same fields as `/api/latest` (without `has_data`).

The same file with the same target, roles and settings is not fit again while
the engine runs (`analysis_cache`). It gets the same analysis and the same id,
on `/api/run` and on `/api/campaigns/{id}/upload`, when it goes to the same
campaign. The same file sent to another campaign is fit for that campaign and
gets its own id: two campaigns never share an `analysis_id`. Background jobs
run in their own process and fit again.

When a campaign gets a new analysis, `supersedes` names the one it replaced:
`{analysis_id, code, why}`. `code` is `new_upload`, `target_changed`,
`roles_changed`, `direction_changed`, `refit` or `results_added`. `why` is one
plain sentence. `supersedes` is null for a campaign's first analysis. A stored
analysis written before ids existed gets an id derived from its record.

### Affected rows in findings (additive)

A finding about specific rows (`outliers`, `duplicate_rows`,
`physical_bounds`, `units_consistency`, `missingness`,
`informative_missingness`, `dropped_rows`, `dropped_cells`, `censored_rows`,
`censored_values`, `blank_as_absent`) carries:

- `sheet_rows`: EVERY affected row, as the row number the user sees in the
  sheet: the header is row 1 and the first data row is row 2. When the names
  were read from the second row, the header is row 2 and data starts on row 3.
  Blank lines that the CSV reader skipped are counted, so the number matches
  the file. `null` for a sheet read as transposed (runs are columns there).
- `sheet_rows_truncated`: true when more than 5,000 rows were affected; only
  the first 5,000 are listed.
- `values`: `{sheet row: cell}` for a finding about one column (the cell as
  read). Not on findings about whole rows (`duplicate_rows`, row-level
  `missingness`, `dropped_rows`). `null` for an identifier column under
  `anonymize`.
- `detail.n_rows_affected` equals the number of affected rows (the length of
  `sheet_rows` unless truncated).

`rows` is unchanged: at most 20 zero-based row positions in the parsed frame.
A finding not about rows has none of these keys. `counts` counts findings per
severity, so `counts.warning` is the number of warning findings.

### Contract version

Every analysis (`/api/run`, `/api/latest`, reanalyze, job results) carries
`contract_version`, an integer. A response without it is version 1. Fields are
only added; a field never changes meaning or disappears within a version.

- **v2**: adds `contract_version`. The fields added in v2 are listed below this
  line, one bullet per field.
- `provenance[].status` can be `dropped_recipe_label`: with no declared roles, a
  text column that the engine picked as the CV group because each of its levels
  is one recipe of the other inputs (for example `recipe` = R1..R10 next to pH and
  temperature). It groups replicates and CV folds; it is not a model input and
  proposals do not carry it. A declared categorical is never dropped this way.
- `cv_logo.reason` can say that the group column labels exactly the recipes;
  leave-one-group-out is then not run (it would repeat the grouped CV).
- `analysis_cache`: `{"hit": true}` when an upload by the same tenant to the same campaign with the same bytes, target,
  roles, sheet, anonymize flag and engine settings was answered from the
  engine's in-memory copy of an earlier analysis instead of a new fit
  (`KALOS_ANALYSIS_CACHE`, docs/DESKTOP.md); `{"hit": false}` after a fit.
  Upload responses only; reanalyze always fits.
- `cv_ci95_reason`: why `cv_ci95` is null (string), else null. The interval
  needs at least 3 CV groups. A `cv_ci_unavailable` info finding says the same.
- `direction`: `maximize` or `minimize`, what the proposals optimize. `best`
  is the max or the min accordingly. Also on the campaign summary.
- `conformal_caveat`: `{code: "coverage_not_guaranteed", message}`. The
  `conformal_q` band's coverage (`conformal_coverage`) was measured on
  held-out rows near the data. It is not guaranteed for proposals, least of
  all those flagged `far_from_data` or `at_box_edge`. Proposals carry no
  conformal interval of their own, so there is nothing per proposal to label.
- `proposals[].applicability`: per-proposal `far_from_data` and `at_box_edge`
  flags (see "Applicability flags").
- `applicability`: the threshold used, its source and rule.
- `model_diagnostics.kernel`: the GP kernel name (see "Model diagnostics").
- `run.direction`: the effective direction, also in `config_sha256`.
- `proposal_gating.off_for_minimize`: true when the P(feasible) gate was not
  applied because the target is minimized. `proposal_gating.reason` says so.
- `provenance[].role_inferred`: the role the engine gave the column (`target`, `input`, `identifier`, `outcome`, `group`, `fixed_input`, `unused`).
- `provenance[].reason`: a short code for why the column has its status (docs/UPLOADS.md, "Column-role safety").
- `validation.findings[].code`: a machine code for the finding.
- `promotion.status` and `promotion.unmeasured` (see "Promotion gate changes").
- `validation.findings[].code` is never null: it is the check name when no more specific code exists.
- `columns`: the kind and role of every uploaded column (see "Column kinds").
- `validation.findings[].code` can be `category_not_modeled`.
- `fixed_inputs`: `{input: value}` for inputs fixed at their median because their spread is within sensor precision. Proposals carry the value in `recipe`. Provenance status `fixed_near_constant` is new.

- **v3** (branch desktop-engine-2): the fields added in v3 are listed below this
  line, one bullet per field.
- `batch_strategy`: `{kind: "model" | "exploration_design", reason: null |
  "few_recipes" | "cannot_rank_yet", plain, n_distinct_recipes, n_inputs, rule}`.
  An exploration design keeps the best bet as P1 and spreads the other recipes
  over the untested parts of the box (`proposals[].pick` =
  `exploration_design`, `mode` `explore`). It is used when the sheet has fewer
  than 2 x inputs distinct recipes, or the verdict is `not_yet` because the
  ordering is weak or unsure. `roles.exploration_design`: `"auto"` (default) or
  `"off"`; anything else is 400 `invalid_exploration_design`. Measured on
  synthetic surfaces only (examples/exploration_design_benchmark.py, 12 seeds,
  4 inputs, 4 starting runs, 15% noise, 3 rounds of 4): regret 4.20 vs 4.43 on
  a Gaussian bump (paired difference -0.23, 95% interval -1.39 to 0.75) and
  4.02 vs 4.03 on Ackley (-0.01, -0.48 to 0.49): no detectable difference
  either way.
- `model_optimum`: the model's best bet, the recipe with the highest predicted
  value inside the search box: `recipe`, `predicted`, `likely_range` [low, high]
  (the conformal band; `likely_range_source`), `in_batch` (its index in
  `proposals`, 0 = P1, or null when it is already running), `already_running`,
  `near_best_run` (it is the recipe of the best measured run, within the batch
  tolerance), `predicted_vs_best`.
- `verdict.reasons[]` code `possibly_measured_driver`: a flagged input is a top
  driver, so a `reliable` verdict is held at `not_yet`.
- `best_bet_notice`: null, or a plain reason when no recipe qualifies as the best bet
  (none meets the outcome constraint floor or the feasibility gate); `model_optimum`
  is then null. `model_optimum.beyond_tried_range` lists inputs past the tried range.
- `proposals[].pick`: `best_bet` (P1, the model optimum) or `learn`. Every batch
  has one best bet unless it is already running. The best bet has `mode`
  `exploit` even when it is predicted at or slightly below the best run, so a
  batch is never all exploration when a best bet exists.
- `proposals[].predicted_vs_best`: `difference` (predicted minus best measured,
  in the better direction), `likely_half_width`, `relation` (`better`,
  `about_best` = within the likely range, `worse`) and `plain`. A `worse` run
  says it is there to learn, not to win, and what it tests.
- `tried_ranges`: the observed range of each continuous input. `feature_ranges`
  stays the search box actually used; it equals `tried_ranges` unless
  `roles.allow_beyond` widened it.
- `allow_beyond`: `{default, per_input, applied}`. Set it in the column roles:
  `"allow_beyond": 0.10` (10% of the tried range past each end, every input) or
  `{"molasses_g_L": 0.25, "*": 0}`. Fractions 0 to 1; anything else is 400
  `invalid_allow_beyond`. Hard physical limits still apply (pH 0-14, no negative
  amounts). Roles persist with the campaign, so this is a per-campaign setting.
- `proposals[].beyond_tried_range`: `[{feature, side, tried_limit, proposed, by,
  fraction}]` for inputs a proposal sets outside the tried range.
- `validation.findings[]` code `possibly_measured_after_run` (warning): a column
  before the target whose name reads like a run measurement (OD, biomass,
  lactate, ammonium, IVCD, mAb, IgG, `_end`, `day14`, peak, residual, `final` with a
  unit) was kept as an input; check it.
- `validation.findings[]` code `headerless_column`: a text column with no header
  was not used as an input (declare it in roles to use it).
- `validation.findings[]` code `search_box` (info, always present): where
  Kalos searches, in plain words, and how to widen it.
- `verdict`: the one verdict on the model, on every analysis and on the campaign summary (see "Verdict").
- `refusal`: `{code, plain, what_would_help}` when `verdict.state` is `refused` (then `proposals` is empty), else `null`.
- `proposal_notice.code` can be `model_refused` (with `reason`, the refusal code).
- `analysis_id`: the stored analysis's id, the same on every route that reads it
  (see "Analysis identity"). Also on `/api/campaign` and `/api/campaigns/{id}`
  `summary` (the analysis the stage comes from) and on the job view (null until
  the job's analysis is published, or once a newer one replaced it).
- `supersedes`: `{analysis_id, code, why}` for the analysis this one replaced in
  the campaign, or null. `/api/run` now also returns the stored fields `dataset`,
  `updated` and `campaign_generation`.
- `roles_used`: `{source, roles}`. `source` is `declared` when the upload's
  `roles` (kept with the campaign as `settings.roles`, docs/CAMPAIGN_LOOP.md)
  were used, `inferred` when the roles came from the column names; `roles` is
  the declared object, or the target, features, categoricals and direction the
  fit used. A reanalyze uses the declared roles and keeps the campaign's label in
  `dataset` (it was `campaign round <n>`).
- `campaign_findings`: a list of findings about how this analysis differs from
  the one it replaced. Each has `code`, `severity`, `message` (plain), and for
  `inputs_changed` the `added` and `removed` input names. Empty when nothing
  changed or there was no earlier analysis.
- `runs_used`: `{n_upload_rows, n_recorded, recorded}`. `recorded` lists every
  measured run of the campaign that entered the fit, oldest first:
  `{run_id, round, measured_value, predicted_at_start, inside_likely_range}`.
  `inside_likely_range` is true or false when the run's likely range was known
  at start, else null. On an upload `n_recorded` is 0 and `recorded` is empty.
- `delta`: what changed since the analysis this one replaced, or null for the
  first one: `{previous_analysis_id, best: {before, after}, verdict: {before,
  after}, driver_shares: [{input, before, after}], driver_shares_source,
  n_runs: {before, after}, inputs_added, inputs_removed, summary}`. The shares
  are `gp_shapes.features[].relevance` (what the model leans on;
  `driver_shares_source` is `gp_shapes`); a share is null when the input was not
  in that fit. `verdict` is the analysis's `verdict` code when it has one, else
  `reliable` / `not_yet_reliable` from `reliability.clears_floor`. `summary` is
  one plain line, for example "Your 2 results were added. Best is still 113.65."
- `pending[].likely_range_at_start` on `/api/campaign` and the runs routes:
  `{low, high}`, the published analysis's likely range around the run's `pred`
  when the run was started, or null.
- `promotion.checks`: each promotion check in plain words, with its state and whether it is optional (see "Promotion gate changes").
- `validation.findings[].sheet_rows`, `sheet_rows_truncated` and `values`: every affected row as a sheet row number, with the cells (see "Affected rows in findings").
- `noise.floor`: `{sigma, sigma_ci95, var_pooled, var_ci95, df, method}`. The
  assay noise pooled over repeated recipes, each weighted by its repeats minus
  one (`df`), with a 95% interval. Values are null (`df` 0) with no repeats.
  `noise.noise_sd` keeps its meaning (mean of the per-recipe variances).
- `noise.per_recipe`: `{enabled, used, reason, df0, sd_range}`. Set
  `"per_recipe_noise": true` in the column roles (anything but true/false is
  400 `invalid_per_recipe_noise`). It is used only when the noise rises with
  the recipe mean (`noise.scale.homoscedastic` false) and the fit is on recipe
  averages: each recipe's own variance, shrunk toward the pooled one with
  `df0` prior repeats, divided by its repeat count. Off by default; proposals
  are unchanged when it is off.
- `stopping`: advice on whether more rounds can still be told apart from assay
  noise. Every block carries `caveat` `{code: "stopping_advice_only", plain}`:
  advice, not a stop, tried only on one small simulated case. Never removes proposals and never stops anything. `{advisory: true,
  state, plain, p_gain_beyond_noise, max_expected_gain, gain_over_noise,
  noise_sd, threshold, method, n_samples, seed}`. `state` is `keep_going`,
  `gains_within_noise` (the chance of beating the best measured recipe by more
  than `noise_sd` is below `threshold`, 0.10) or `unknown` (no repeated recipes,
  so no noise estimate, or a refused sheet; numbers are null). `noise_sd` is
  `noise.floor.sigma`; `max_expected_gain` is the largest predicted gain over
  the best measured value, in the better direction, in target units. In a
  synthetic simulation (`kalos/bench/stopping_sim.py`: one 2-input bump, noise
  sd 0.5, 10 seeds x 6 rounds) 2 of 30 `gains_within_noise` readings were
  false (the true optimum was still more than one noise sd better), both in
  one seed. One small synthetic case; not a measured rate for real assays.
- `linear_constraints`: `{constraints, tolerance, all_satisfied}`. Set them in the
  column roles as `"linear_constraints": [{"terms": {"A": 1, "B": 1, "C": 1},
  "op": "==", "rhs": 100, "label": "sums to 100"}]` (`op` is `<=`, `>=` or `==`;
  at most 20; continuous inputs only). Each listed constraint adds `text` and its
  `tolerance` (0.001 x the sum of its absolute coefficients, because recipes are
  shown to 3 decimals). The batch, the repair of near-duplicates and the best bet
  are all searched inside them. 400 codes: `invalid_linear_constraint` (malformed,
  or an input that is categorical or not an input), `linear_constraints_with_categorical`
  (the design has a categorical input), `infeasible_linear_constraints` (no recipe in
  the search box meets them all; checked before any fit). An input held fixed
  (`fixed_inputs`) counts at its fixed value. Roles persist with the campaign.
  `POST /api/preflight` refuses infeasible constraints with the same 400.
- `proposals[].constraints_ok` and `model_optimum.constraints_ok`: true when the
  shown recipe meets every linear constraint within its tolerance; null when none
  are declared.
- `joint_feasibility` (every block carries `caveat` `{code:
  "specs_treated_independently", plain}`: each spec is judged on its own, so the
  joint chance can be off when results move together): `{specs, n_samples, method, predicts, seed, available,
  reason}`. Set specs in the column roles: `"specs": [{"column": "Purity", "min":
  95}]` (outcome columns, the target may be one; `min`, `max` or both; at most 5;
  a spec on a declared input or a malformed spec is 400 `invalid_specs`). One model
  per spec column (rows with a value, at least 6). `specs[]` adds
  `n_rows_with_value`, `available`, `reason`. `method` is `independent_mc`: draws
  from each spec's model, sampled independently, so the joint chance is an estimate
  of the product of the per-spec chances (outcome correlation is not modelled).
  `predicts` says the draws are for a new measurement (model plus assay noise).
  `available` is false, with a `reason`, when any spec column is missing or too
  sparse. Advisory: the batch is the same with or without specs.
- `proposals[].p_all_specs` and `model_optimum.p_all_specs`: the chance that every
  spec is met at once (0 to 1), null when `joint_feasibility.available` is false.
- `proposals[].p_specs` and `model_optimum.p_specs`: `{column: chance}` per spec,
  or null.
- 409 `roles_not_applied` on `POST /api/campaign/reanalyze` and
  `POST /api/campaigns/{id}/reanalyze` when the column roles stored with the
  campaign (`settings.roles`) cannot be read. Nothing is committed. Before, the
  reanalyze quietly fitted with roles inferred from the column names. Upload the
  sheet again with valid roles.
- `validation.findings[].why` and `campaign_findings[].why`: one plain sentence
  on what the finding means for the lab's work, keyed by `code`
  (kalos/portal/why_lines.py). Coded 400/409 refusals that go through the upload
  refusal envelope or the campaign routes also carry `why` next to `code`.
  `tests/test_why_lines.py` fails when the engine gains a code without a line.
- 400 `invalid_allow_beyond` also when `roles.allow_beyond` names an input that
  is not a column of the sheet (other than `"*"`); the message names it. Before,
  the name was ignored and that input stayed at its tried range.
- `GET /api/campaigns/{id}/rows`: `_kind` can be `no_result`; `renamed_columns`
  and `renamed_note` (see "Campaign rows").

## Not in this slice

- Background jobs for reanalyze (it still runs synchronously; upload has a job route).
- JWT principals (E2). `auth_method` will read the principal's
  `auth_method` once it exists. `tests/test_cross_tenant_isolation_jwt.py` is
  a skipped TODO stub until E2 lands.
- Deleting a campaign, changing the default campaign, and reading the audit
  log over HTTP.
