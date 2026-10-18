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
isolation), `tests/test_jobs.py` (queue, worker, runner, crash recovery). The
runner's process handling is tested with real processes: a time-limit kill of a
worker and its child that both ignore SIGTERM, and the exit-path group kill.

## Routes

All routes need a bearer token when auth is enforced (docs/HARDENING.md). The
tenant is always the caller's; no route reads one from the path, query, body
or headers.

| Method and path | Scope | Result |
|---|---|---|
| `POST /api/campaigns/{id}/jobs` | `write` | `202` + `{"job": ...}`, `Location: /api/jobs/<id>`. Multipart: `file`, `target`, `anonymize`, `roles`, `sheet` (same fields as `/api/run`), and `replace` (`true` to confirm that the upload replaces the campaign's runs; see below). |
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
| `campaign_has_runs` | The campaign has awaiting or measured runs and the upload was not confirmed with `replace=true`. Nothing was changed; resubmit with `replace=true` to replace them. |
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
| `400` | Also: `roles` is over 16 KiB (`MAX_ROLES_CHARS`); refused, not truncated. |
| `413` | The upload is over the tenant's job size limit, or over the engine's upload cap. |
| `429` | The tenant already has `KALOS_JOB_QUEUE_DEPTH` queued or running jobs, or the engine already has `KALOS_JOB_GLOBAL_QUEUE_DEPTH` (all tenants). Or the tenant already has `SUBMIT_PER_TENANT` (2) uploads being received. `Retry-After` is set. |
| `409` | `code: upload_pending`: the campaign already has a queued or running upload (one pending upload per campaign). `code: campaign_has_runs`: the campaign has awaiting or measured runs and `replace=true` was not sent. Nothing is queued or written. |
| `503` | Jobs are disabled (`KALOS_JOBS_ENABLED=0`), the job settings are malformed, every preflight slot is in use, or the engine already has `SUBMIT_TOTAL` (4) uploads being received. `Retry-After` is set on the last two. |

## Lifecycle and guarantees

**Submit.** The queue depth is checked first, from a cheap read, so a full
tenant is refused before its upload is read or parsed. The upload is then
checked by the same pre-fit refusals as `/api/run` (parsed on a thread, under
one of `PREFLIGHT_CONCURRENCY` process-wide slots: see "Scheduling and
limits"). Only then is the upload written under `<KALOS_STATE_DIR>/jobs/<sha256(tenant)[:16]>/<job_id>.bin`
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
| Queued + running jobs, all tenants together | `KALOS_JOB_GLOBAL_QUEUE_DEPTH` | `64` | whole engine |
| Uploads being received at once (from the body read until the submit answers) | fixed: `SUBMIT_PER_TENANT` / `SUBMIT_TOTAL` | `2` / `4` | per tenant / whole engine |
| Uploads being checked (parsed) at once, before they are queued | fixed: `PREFLIGHT_CONCURRENCY` | `2` | whole engine |
| Upload size for a job (at most the engine's upload cap) | `KALOS_JOB_MAX_UPLOAD_BYTES` | the engine upload cap | per tenant |
| Wall-clock limit for one job, then its worker is killed | `KALOS_JOB_TIMEOUT_S` | `1800` | per job |
| Turn the queue off (routes answer 503; status still reads) | `KALOS_JOBS_ENABLED` | `1` | whole engine |

- **Replacing runs needs confirmation.** A successful publish replaces the
  campaign's state, so it discards any awaiting or measured runs. An upload
  that would do that is refused at submit (`409`, `campaign_has_runs`) unless
  the form carries `replace=true`. If runs appear after the submit, the job
  is `failed` with `campaign_has_runs` before any analysis, and nothing is
  changed. A confirmed replacement is recorded in the job's `params`.
- **One pending upload per campaign.** A second queued or running upload to a
  campaign is refused at submit (`409`, `upload_pending`). Two of them would
  each run a full fit and the second would fail `campaign_changed` once the
  first published. The check is repeated inside the enqueue transaction.
- **Refused before any write.** A submit over the upload cap, over the queue
  depth, or with an over-long `roles` value is refused before its bytes are
  written to the state directory. Only a race between two submits (both pass
  the cheap check) can write a file that the transaction then refuses, and that
  file is removed again.
- **Engine-wide queue.** `KALOS_JOB_GLOBAL_QUEUE_DEPTH` bounds the jobs held
  by all tenants together, so the number of pending uploads on disk is bounded
  too: at most that many times the upload cap. Total bytes are not capped
  separately; size the state directory for the product of the two.
- **Uploads in memory.** A submission holds its whole upload in memory from the
  read until it is queued or refused. The number of submissions being received
  is therefore capped before the read, per tenant and engine-wide (`429` or
  `503`, `Retry-After`), so N concurrent submits cannot each hold up to the
  upload cap at once. The multipart body itself is spooled to disk by the
  framework above 1 MiB; the cap bounds what the route reads into memory.
- **Fairness.** The runner visits tenants oldest-queued-job first and skips a
  tenant at its per-tenant cap, so one tenant's backlog cannot hold every slot.
- **Sync routes yield to the queue.** A synchronous analysis is admitted only
  if, after it, a slot remains for every other tenant with a queued job. So a
  stream of `/api/run` requests cannot starve the queue. A sync request that
  is refused gets the usual `503` with `Retry-After`.
- **Slots are held by processes.** A slot is held from worker start until the
  worker process has exited, so the count reflects real CPU and memory use.
- **Slots are held by threads too.** A sync analysis holds its slot until its
  thread returns, even if the request was cancelled (for example by
  `asyncio.wait_for` or a client disconnect). Cancelling the request does not
  free the slot early; only a call that never started is released at once.
- Each worker is a separate process: a fit that crashes or runs away cannot
  take the engine's interpreter with it, and a timeout or cancel is a kill.
- Each worker applies the same torch thread budget as the engine
  (`KALOS_TORCH_THREADS`, `kalos/portal/pipeline.py`). Total CPU threads are
  therefore about `KALOS_JOB_WORKERS` times that budget: on a shared machine,
  lower one or the other.

Invalid values for any `KALOS_JOB*` variable (non-numeric, out of range, a
job upload limit above the engine cap, `KALOS_JOB_WORKERS=0`) fail the engine's
startup, with the variable named. `python -m kalos.portal` checks them before
it serves and exits `2` with one line on stderr and no traceback (the same for a
refused runner, below). They are not silently replaced by defaults. The app
lifespan repeats the check, so serving the app directly (`uvicorn`) fails its
boot too, with the traceback that uvicorn prints.

If the engine boots with jobs enabled but its scheduler is not running (its
start failed and was logged), `/readyz` answers `503` with the fixed reason
`background jobs are not running`, so an orchestrator does not route work to
an engine that would accept jobs and never run them.

## Cancellation and timeouts

- **Queued:** cancelled at once, no worker is ever started.
- **Running:** `cancel_requested` is recorded, and the runner kills the worker
  within one poll interval (0.5 s). The worker also checks at each progress
  point. Either way nothing is published.
- **Timeout:** the runner kills the worker when `KALOS_JOB_TIMEOUT_S` elapses
  and records `timed_out` (`error.code = timeout`).
- **Orphaned worker.** A worker whose engine has died (its parent is no longer
  the engine pid the runner passed in, `KALOS_JOB_ENGINE_PID`) does not analyse
  or publish. It checks at start, at each progress point and in the publish
  transaction, and exits (status 3). The publish is refused by the same guard
  that refuses a job no longer `running`, so a publish from an orphan cannot
  overwrite a campaign after a restart has recorded the job as `interrupted`.
- **Engine shutdown:** the scheduler stops starting workers, then running
  workers are killed and their jobs are recorded `failed` with
  `engine_stopped`. A scheduling pass that is under way finishes first (the
  shutdown waits for it, up to 30 s), so a job claimed just before shutdown is
  stopped like any other. If a pass cannot finish in that time, the runner
  lock is kept until the process exits rather than released under a live
  scheduler. The runner lock (below) is released last, after those jobs are
  recorded, so another runner cannot recover them mid-shutdown.

- **Start failure:** if the worker cannot be started, or its pid cannot be
  recorded, the worker (if one was started) is killed and the job is recorded
  `failed` with `worker_unavailable`. Any exception, and any `BaseException`
  such as `KeyboardInterrupt`, takes this path; nothing before the child is
  registered can leak the claimed job or its slot. A slot is released only once the worker
  is gone and its outcome is recorded; if recording fails, the next pass retries.
- **Failed claim:** the claim is one transaction, so a claim that raises leaves
  the job `queued`; the scheduler logs it and tries again on the next pass.

A kill is `SIGKILL` to the worker's process group. A fit cannot be interrupted
inside a single BLAS call, so the kill is the bound, not a checkpoint. Every
way a worker ends (cancel, timeout, a crash, a clean exit) goes through that
group kill, so a child process the worker left behind is stopped too.

## Worker lifetime

Each worker is also given its engine's pid in `KALOS_JOB_ENGINE_PID`. The parent
check compares with that pid, so an engine that dies while its worker is still
importing (a window of about a second) is seen, not missed.

Each worker gets a deadline in `KALOS_JOB_DEADLINE_S`: the seconds left at
spawn, `KALOS_JOB_TIMEOUT_S` plus a 60 s margin. The worker measures it on its
own monotonic clock, so a wall-clock step (NTP, a manual change) neither kills
a live worker nor extends one. The monotonic clock does not advance while the
machine is suspended, so the time limit counts running time, not sleep time.

The runner normally stops a worker at the time limit. But a worker whose engine
has gone (killed, or crashed before it could kill the worker) would otherwise
run on, unsupervised, and so would its children. So a watchdog thread in the
worker process ends it once the deadline has passed, or as soon as its parent
is no longer the engine that started it (checked every second). It then
`SIGKILL`s its own process group, which the worker leads (`ProcessWorker`
starts it as a session leader), so every child it started dies with it, and
exits with status 3. No job state is written: the job is recorded
`interrupted` by the next engine start (see below), and until then a publish is
still all-or-nothing: it commits with the job's `succeeded` state, or not at all.

Only a worker that leads its group sets the group kill (`start_lifetime_watchdog`,
`kill_group`), so the watchdog never signals a group it did not create.

## Restart and crash recovery

On startup the runner, before admitting anything:

1. fails every job left `running` with `interrupted`. For each job whose
   recorded worker pid is still alive, it first checks that the pid's command
   line names that job (the job id is unique, so a reused pid never matches).
   On Linux the command line is read from `/proc`, which needs no helper
   program, so this works in slim images such as `python:3.12-slim`. Where
   `/proc` does not exist (macOS) it is read with `ps`, if installed. A pid
   whose command line cannot be read is **not** signalled. A verified worker's
   process group is then killed (the worker leads it, so the group id is the
   recorded pid; a pid that no longer leads a group is left alone);
2. deletes stored inputs that no queued or running job refers to, and any
   half-written upload (`.tmp`).

The worker's watchdog already ends it, and its children, within about a second
of its engine dying, so step 1 is the backstop for a worker whose watchdog could
not run. A worker that has already exited leaves no command line to check, so
any child it left behind is not signalled by recovery; the normal exit paths
kill the group at once (see "Cancellation and timeouts"), so this is a narrow
residual window.

Queued jobs survive a restart and are admitted as usual. Running ones are not
resumed: the upload is resubmitted by the client (the analysis is
deterministic, so a resubmit reproduces the result).

## Logs

Analysis runs on customer data, so the logs are treated as customer-adjacent.

- **Worker analysis failures** (`analysis_failed`, `parse_error`, `fit_failed`)
  log one line: the code, the exception type and the innermost `file:line in
  function`. They do not log the exception message or the traceback, because
  a parser or `KeyError` message quotes customer column names and cell values
  (`kalos/portal/job_worker.py`, `_log_analysis_failure`). Other codes log the
  code only. A job's `error` (stored in the database, not logged) carries the
  same message the sync route returns for that refusal.
- **Worker startup failures** (the job cannot be loaded, the campaign store
  cannot be opened) log the exception type only, and the worker exits without a
  traceback. A `ValueError` or `OSError` message can quote a customer column or
  a path, so it is never logged.
- **Worker stderr** is inherited from the engine process (`job_runner.py`
  starts workers with no stderr redirect). Anything a worker writes outside
  `run_job` lands in the engine's log stream.
- **Runner logs** (`job_runner.py`) cover scheduling and infrastructure
  failures: spawn, pid, claim, recording an outcome. The runner never parses an
  upload, so customer text should not reach it. Its `log.exception` calls still
  record full tracebacks, so the same expectation applies.
- **The sync route is not stricter.** `/api/run`, the campaign reanalyze and
  the scale readout log full tracebacks in their catch-alls, which can carry
  the same text. Jobs add a second place where it can appear, not a new kind
  of exposure.

Expectation for operators: do not ship engine or worker logs to a third party
unredacted. If logs leave the host, filter them first, or send them only to
systems that are inside the same data-handling boundary as the uploads.

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
- **Multiple engine processes are refused.** The limiter and the runner are per
  process. So the runner takes an exclusive lock on `<KALOS_STATE_DIR>/jobs.runner.lock`
  when it starts (`flock`, released when the process exits, however it exits),
  and a second runner on that state directory refuses with `RunnerLocked`. The
  engine then fails to boot, rather than accepting job submissions no runner
  would ever run. A start with `WEB_CONCURRENCY` above 1 is refused too, with a
  message naming the variable and exit status `2` (`python -m kalos.portal`). Run one engine process per state directory
  (docs/HARDENING.md). The lock is POSIX-only (`fcntl`); on a platform without
  it the engine refuses to start only when jobs are enabled, and
  `KALOS_JOBS_ENABLED=0` runs the sync API there.
  **Keep the state directory on a local filesystem.** `flock` on NFS and SMB
  mounts is emulated or unsupported on many clients: two engines on one share
  can both take the lock, and then both schedule the same jobs. The check
  protects one host with a local state directory, not a shared mount.
