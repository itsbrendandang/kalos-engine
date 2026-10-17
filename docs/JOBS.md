# Background analysis jobs (`/api/campaigns/{id}/jobs`, `/api/jobs/{id}`)

A job is one upload analysis, run in the background instead of inside the
request. A GP fit takes tens of seconds to minutes; a job lets the client
submit, get a `202`, and poll, instead of holding a connection and a slot
for the whole fit. Jobs survive a client disconnect and can be cancelled.

The synchronous routes (`/api/run`, `/api/campaigns/{id}/upload`,
`/api/campaign/reanalyze`, `/api/campaigns/{id}/reanalyze`) are unchanged in
contract. Their analysis is the same code a job runs (`kalos/portal/pipeline.py`).

Implementation:

| Module | Role |
|---|---|
| `kalos/portal/jobs.py` | The `jobs` table (schema v3): submit, read, cancel, every state transition. Torch-free. |
| `kalos/portal/jobs_api.py` | The HTTP routes, and the upload size / bearer precheck middlewares. |
| `kalos/portal/job_runner.py` | The scheduler, inside the engine process: admits jobs, supervises workers, enforces time limits and cancellation, recovers after a crash. |
| `kalos/portal/job_worker.py` | Runs ONE job. Also the entry point of the worker process (`python -m kalos.portal.job_worker <job_id>`). |
| `kalos/portal/pipeline.py` | The analysis and the publish, shared with the synchronous routes. |
| `kalos/portal/busy.py` | The admission limiter (global and per-tenant slots) that jobs and sync routes share. |

Release-gate tests: `tests/test_jobs_api.py` (HTTP contract and cross-tenant
isolation), `tests/test_jobs.py` (queue, worker, runner, crash recovery).

## Routes

All routes need a bearer token when auth is enforced (docs/HARDENING.md). The
tenant is always the caller's; no route reads one from the path, query, body
or headers.

| Method and path | Scope | Result |
|---|---|---|
| `POST /api/campaigns/{id}/jobs` | `write` | `202` + `{"job": ...}`, `Location: /api/jobs/<id>`. Multipart: `file`, `target`, `anonymize`, `roles`, `sheet` (same fields as `/api/run`). |
| `GET /api/campaigns/{id}/jobs?limit=N` | `read` | `{"jobs": [...]}`, newest first, `N` 1-200 (default 50). |
| `GET /api/jobs/{id}` | `read` | `{"job": ...}`. |
| `POST /api/jobs/{id}/cancel` | `write` | `200` + `{"job": ...}`. Idempotent; a finished job is returned unchanged. |

### The job object

```json
{
  "id": "job_<32 hex>",
  "campaign_id": "cmp_<32 hex>",
  "kind": "analysis",
  "state": "queued",
  "stage": "queued",
  "progress": 0.0,
  "attempts": 0,
  "created_at": 1760000000.0,
  "started_at": null,
  "finished_at": null,
  "updated_at": 1760000000.0,
  "cancel_requested": false,
  "input_bytes": 48213,
  "error": null,
  "campaign_generation": null
}
```

- `state`: `queued` -> `running` -> one of `succeeded`, `failed`, `cancelled`,
  `timed_out`. A queued job can also go straight to `cancelled`.
- `stage` and `progress` (0-1): `parsing`, `fitting`, `calibrating`,
  `proposing`, `publishing`, then `succeeded`. Progress is reported at stage
  boundaries, not per iteration.
- `error`: `null`, or `{"code", "message"}` for a job that did not succeed.
  The message is fixed engine text; the exception text is never returned
  (it can name customer columns).
- `campaign_generation`: set only when `succeeded`. It is the generation the
  analysis was published under, which is also the `campaign_generation` stamp
  on `GET /api/campaigns/{id}/analysis`. A client can confirm the analysis it
  reads is the one the job made.
- `cancel_requested`: true once a cancel was accepted for a running job.

### Error codes

| `error.code` | Meaning |
|---|---|
| `too_few_rows`, `target_constant`, `empty_file`, `unsupported_file_type`, ... | The engine's own refusal of the sheet, with the same code as `/api/run`. (Refusals are normally caught before queueing, see below.) |
| `parse_error`, `fit_failed` | As `/api/run`. |
| `analysis_failed` | Any other error during the analysis. |
| `campaign_changed` | The campaign changed (a result, a start, another upload) between submission and publish. Nothing was published; the upload must be resubmitted. |
| `campaign_not_found`, `campaign_archived` | The campaign was removed or archived while the job waited. |
| `input_unavailable` | The stored upload is missing or does not match its recorded checksum. |
| `timeout` | Past `KALOS_JOB_TIMEOUT_S`; the worker was stopped. |
| `worker_exited` | The worker process ended without recording an outcome (crash, OOM kill). |
| `worker_unavailable` | The worker process could not be started. |
| `interrupted` | The engine restarted while the job was running. Resubmit. |
| `engine_stopped` | The engine shut down while the job was running. Resubmit. |

### Synchronous refusals (no job is created)

The submit route runs the same pre-fit checks as `/api/run` before queueing
(`kalos/portal/pipeline.py` `preflight_upload`, the same code with the fit
stopped early). A bad sheet is a `400` with the usual `{"error", "code"}`
envelope now, not a failed job later.

| Status | When |
|---|---|
| `400` | The sheet is refused (shape, target, constant target, row caps, ...). |
| `401` | No bearer token, with auth enforced. Refused before the body is read. |
| `403` | Missing scope. |
| `404` | The campaign (or, for `GET`/cancel, the job) is not visible to the caller. See "Isolation". |
| `409` | The campaign is archived. |
| `413` | The upload is over the tenant's job size limit, or over the engine's upload cap. |
| `429` | The tenant already has `KALOS_JOB_QUEUE_DEPTH` queued or running jobs. `Retry-After` is set. |
| `503` | Jobs are disabled (`KALOS_JOBS_ENABLED=0`) or the job settings are malformed. |

## Lifecycle and guarantees

**Submit.** The upload is written under `<KALOS_STATE_DIR>/jobs/<sha256(tenant)[:16]>/<job_id>.bin`
and its SHA-256 and size are recorded. The row is inserted in the same
transaction that checks the queue depth and the campaign, so a refused
submission leaves no row; the file is removed again.

**Run.** The runner admits a queued job when a slot is free (below), starts a
worker process (`python -m kalos.portal.job_worker <id>`, its own session), and
the worker:

1. re-verifies the input against its checksum;
2. runs `compute_upload_analysis` (the sync route's code), reporting each stage;
3. calls `publish_upload_analysis`, which writes the campaign and its analysis
   in ONE transaction, as a compare-and-swap on the generation the job was
   submitted under, with a commit hook that marks the job `succeeded` inside
   that same transaction.

So a job either publishes and reads `succeeded`, or publishes nothing. A
cancel that lands before the commit rolls the publish back. A campaign that
changed while the job waited fails the job (`campaign_changed`) rather than
being overwritten. This is the rule that a stale or cancelled job can never
overwrite newer state.

**Terminal states are written once.** Every transition is a guarded `UPDATE`
(`... AND state IN (...)`). The first terminal writer wins; a worker that
finishes after its job timed out changes nothing.

**Input files** are deleted when the job reaches a terminal state, and any
input file no live job refers to is swept at startup.

## Scheduling and limits

Concurrency, queue depth, upload size and time are all per tenant, and
concurrency is also bounded globally. Defaults are chosen for one engine
process on a modest machine.

| Limit | Env | Default | Scope |
|---|---|---|---|
| Analyses running at once (sync and job together) | `KALOS_JOB_WORKERS` | `2` | whole engine |
| Analyses one tenant may run at once | `KALOS_JOB_PER_TENANT` | `1` | per tenant |
| Queued + running jobs | `KALOS_JOB_QUEUE_DEPTH` | `8` | per tenant |
| Upload size for a job (at most the engine's upload cap) | `KALOS_JOB_MAX_UPLOAD_BYTES` | the engine upload cap | per tenant |
| Wall-clock limit for one job, then its worker is killed | `KALOS_JOB_TIMEOUT_S` | `1800` | per job |
| Turn the queue off (routes answer 503; status still reads) | `KALOS_JOBS_ENABLED` | `1` | whole engine |

- **Fairness.** The runner visits tenants oldest-queued-job first and skips a
  tenant at its per-tenant cap, so one tenant's backlog cannot hold every slot.
- **Sync routes yield to the queue.** A synchronous analysis is admitted only
  if, after it, a slot remains for every other tenant with a queued job. So a
  stream of `/api/run` requests cannot starve the queue. A sync request that
  is refused gets the usual `503` with `Retry-After`.
- **Slots are held by processes.** A slot is held from worker start until the
  worker process has exited, so the count reflects real CPU and memory use.
- Each worker is a separate process: a fit that crashes or runs away cannot
  take the engine's interpreter with it, and a timeout or cancel is a kill.
- Each worker applies the same torch thread budget as the engine
  (`KALOS_TORCH_THREADS`, `kalos/portal/pipeline.py`). Total CPU threads are
  therefore about `KALOS_JOB_WORKERS` times that budget: on a shared machine,
  lower one or the other.

Invalid values for any `KALOS_JOB*` variable (non-numeric, out of range, a
job upload limit above the engine cap) fail the engine's startup, with the
variable named. They are not silently replaced by defaults.

## Cancellation and timeouts

- **Queued:** cancelled at once, no worker is ever started.
- **Running:** `cancel_requested` is recorded, and the runner kills the worker
  within one poll interval (0.5 s). The worker also checks at each progress
  point. Either way nothing is published.
- **Timeout:** the runner kills the worker when `KALOS_JOB_TIMEOUT_S` elapses
  and records `timed_out` (`error.code = timeout`).
- **Engine shutdown:** running workers are killed and their jobs are recorded
  `failed` with `engine_stopped`.

A kill is `SIGKILL` to the worker's process group. A fit cannot be interrupted
inside a single BLAS call, so the kill is the bound, not a checkpoint.

## Restart and crash recovery

On startup the runner, before admitting anything:

1. fails every job left `running` with `interrupted`, killing its worker first
   if the recorded pid is still alive **and** its command line names that job
   (so a reused pid is never signalled);
2. deletes stored inputs that no queued or running job refers to.

Queued jobs survive a restart and are admitted as usual. Running ones are not
resumed: the upload is resubmitted by the client (the analysis is
deterministic, so a resubmit reproduces the result).

## Isolation

The rules are those of the campaign routes (docs/CAMPAIGNS_API.md), applied to jobs:

- Every API-facing query on `jobs` carries `tenant = ?`. A job id of another
  tenant reads exactly like a missing one: `404 {"error": "job not found"}`,
  the same status, headers and bytes. The same holds for cancel, and for
  submitting to or listing another tenant's campaign.
- Malformed job ids (not `job_` + 32 hex) take the same 404 path and are not
  echoed.
- The one cross-tenant read is the scheduler's queue scan, which returns tenant
  ids only. A worker runs its own row only.
- Audit rows (`job.submit`, `job.cancel`) carry the job id; they are read per
  tenant only.
- Input files are stored under a hash of the tenant id, and their paths are
  built by the server, never from caller input.

## Not in this slice

- **Resuming** an interrupted job. It is failed and must be resubmitted.
- **Retention.** Terminal job rows are kept (their input files are not). A
  prune policy is a follow-up.
- **Per-process memory limits** on the worker (`RLIMIT_AS` / cgroups). The
  container deployment is the place to cap memory; the engine does not.
- **Reanalyze as a job.** `/api/campaigns/{id}/reanalyze` still runs
  synchronously under the slot limiter.
- **Multiple engine processes.** The limiter and the runner are per process
  (the engine runs one uvicorn worker, docs/HARDENING.md). Two engine
  processes sharing one `portal.db` would each run a runner; that is not
  supported yet.
