# The engine in the desktop app

The desktop app (kalos-web `desktop/`, a Tauri shell) runs this engine as a local
process for one person on one Mac. The engine keeps its data in SQLite in one
folder on that Mac, listens on 127.0.0.1 only, and answers only callers that
send the token the shell made for this launch.

This page says what the shell must pass, what the engine answers, and how the
packaged engine is built. Code: `kalos/portal/desktop.py`,
`kalos/portal/__main__.py`, `GET /api/health` in `kalos/portal/app.py`,
`scripts/build_desktop_engine.sh`.

## Starting the engine

From a checkout: `python -m kalos.portal --desktop`.
From the packaged engine: `<bundle>/kalos-engine --desktop`.
`KALOS_DESKTOP=1` in the environment does the same as `--desktop`. Started any
other way (for example `KALOS_DESKTOP=1 uvicorn kalos.portal.app:app`), the app
runs the same token, open-access and `KALOS_HOST` refusals when it is imported.

### What the shell passes

| Variable | Required | Value |
| --- | --- | --- |
| `KALOS_DESKTOP` | yes (or `--desktop`) | `1` |
| `KALOS_HOST` | no | `127.0.0.1` (the default). Any other value, `::1` and `localhost` included, is refused. |
| `KALOS_PORT` | yes | A free port the shell picked on 127.0.0.1. |
| `KALOS_STATE_DIR` | yes | The data folder. Holds `portal.db`. Created if missing. Without it the engine uses `~/.kalos`. |
| `KALOS_AUTH_TOKENS` | yes | A JSON list with one record: `[{"token_sha256": "<sha256 hex of the launch secret>", "subject": "desktop-app", "tenant": "local", "scopes": ["read", "write"]}]`. The shell keeps the plain secret and sends it as `Authorization: Bearer <secret>`. |
| `KALOS_CORS_ORIGINS` | yes | `http://127.0.0.1:<web port>`, the app's own origin. Unset (or `*`) is refused, exit 2. |
| `KALOS_TORCH_THREADS` | no | Threads for one fit. Default: half the cores, between 1 and 4. |

Local LLM: in desktop mode an unset `KALOS_LLM_PROVIDER` becomes `ollama` when
something answers on `KALOS_OLLAMA_URL` (default `http://127.0.0.1:11434`, a
0.2 s TCP check at start), else `none`; a hosted provider is never the default.
`KALOS_OLLAMA_MODEL` defaults to `llama3.1`. The local assistant (branch
`assistant`, `kalos/assist/`, not on this branch yet) uses the same settings.
Ollama is not bundled. When it is not running, the chat route
answers 503 `assist_provider_unavailable` and the rest of the engine is
unaffected. The assistant's index lives under `$KALOS_STATE_DIR/assist/`.

The shell must not pass `KALOS_AUTH_TOKENS_FILE` or `KALOS_ALLOW_OPEN_ACCESS`.
The tenant `default` is reserved for open mode and is refused on a token; the
desktop app uses `local`.

### What desktop mode changes

- A token is required. Without one the engine does not start.
- Only 127.0.0.1. `KALOS_HOST` must be `127.0.0.1` (`::1`, `localhost` and
  any other address are refused, exit 2). Requests whose `Host` header is not
  `127.0.0.1` or `localhost` (any port; no `Host` header and other spellings
  such as `LOCALHOST` included) get 400 (a page that points its own name at
  127.0.0.1 is refused before the token check).
- One analysis at a time (`KALOS_JOB_WORKERS=1`); a second upload while one
  runs gets 503 `engine_busy` with `Retry-After`.
- Torch threads capped (see above), so the rest of the laptop stays usable.
- No API docs (`/docs`, `/openapi.json`) and no audit-archive timer
  (`KALOS_AUDIT_RETENTION_DAYS` is ignored; no audit row is deleted).
- Torch and BoTorch are imported in the background once the port is open,
  so the first upload does not wait for them.

### Exit codes and messages

Every refusal is one line on stderr that starts with `kalos engine: `. The
shell shows that line.

| Exit | Meaning | Example |
| --- | --- | --- |
| 0 | Stopped after SIGTERM. Last line `kalos engine: stopped`. | |
| 2 | A setting the engine does not accept. | `kalos engine: desktop mode needs a token: set KALOS_AUTH_TOKENS (the desktop app does this)` |
| 3 | The data folder cannot be used. Nothing was written. | `kalos engine: the database in /path is damaged (SQLite integrity check failed). Your data was not changed. Restore a backup, or move the folder aside to start fresh.` |

| 4 | The server did not start; the data folder was not judged damaged. Do not offer a restore. | `kalos engine: the engine could not start: port 8050 on 127.0.0.1 is already in use (another program, or another Kalos engine)` |

Exit 3 covers: a folder that cannot be written, a `portal.db` that SQLite
cannot read or whose `quick_check` fails, tables that do not have the shape
this engine expects, and a database written by a newer engine. The check runs
before the port opens.

Exit 4 covers what uvicorn itself reports as a failed start (it exits 3 for
all of them, so desktop mode maps them to 4): the port is already in use,
another engine runs the background jobs of this data folder, and a database
upgrade that could not run safely (it changes nothing). The line names the
cause. Note for the desktop shell owner (kalos-web `desktop/src-tauri`,
`main.rs`): exit 4 is new; show the line and offer a retry or another port,
never the restore-a-backup message of exit 3.

## Readiness: `GET /api/health`

Poll it with the bearer token. 401 without it.

```json
{
  "status": "ok",
  "mode": "desktop",
  "engine_version": "0.1.0",
  "contract_version": 2,
  "python_version": "3.12.5",
  "torch_version": "2.13.0",
  "botorch_version": "0.18.1",
  "torch_loaded": true,
  "torch_threads": 4,
  "torch_prewarm": {"seconds": 1.3, "error": null},
  "analysis_in_flight": 0,
  "uptime_s": 4.6,
  "pid": 55551,
  "schema_expected": 5,
  "schema_version": 5,
  "state_dir": "/Users/me/Library/Application Support/Kalos"
}
```

- 200: ready. `torch_loaded` false means the first upload will wait a few
  seconds more for the import.
- 503: `status` `unavailable` with `code` `store_unavailable` or
  `schema_mismatch` and a plain `error`. The exception itself is only logged.
- Versions are read from package metadata; the route never imports torch.
- Outside desktop mode `state_dir`, `pid`, `analysis_in_flight` and the
  library versions (`python_version`, `torch_version`, `botorch_version`) are
  only returned to a token with the `admin` scope.

`/healthz` (no token, `{"status": "ok"}`) is unchanged and stays a bare
liveness ping.

## Stopping

Send SIGTERM, wait up to 15 s, then SIGKILL. On SIGTERM the engine stops taking
requests. An analysis that is running gets 12 s to finish; if it finishes, its
result is saved and returned as usual. If it does not, the engine stops
waiting for it and from then on refuses every database commit in this process,
so its result is not saved: the campaign, its runs and the latest analysis in
`portal.db` stay as they were before it started. Then the process exits 0
(the unfinished computation ends with it).

Measured (Apple M3 Pro, loaded machine): a SIGTERM 1.5 s into a small upload let
it finish and commit (exit 0 after 5 s); a SIGTERM 4 s into a 30 s upload
dropped it with nothing written (exit 0 after 12.2 s, `quick_check` ok).

## Cold start

Measured on an Apple M3 Pro (11 cores) while other jobs used the CPU (load
average 8-25), so treat these as rough.

| | Before | After |
| --- | --- | --- |
| Importing the app | 3.4 s warm, 17.7 s on the first import after install (torch is imported) | 2.5 s warm (no torch) |
| Start to first health answer, checkout | 13.1 s first run, 1.8 s warm | 1.1-1.5 s warm |
| Start to first health answer, packaged engine | | 9.7 s on the very first launch of a fresh bundle (macOS checks the new files), then 1.7-2.2 s |
| Start to torch ready (background import), packaged engine | | 2.4-3.6 s |

## Fit time

The first fit on two sheets, through `POST /api/run`, desktop mode, measured
from a fresh engine start (so the first fit includes loading torch):

| Sheet | Before | After |
| --- | --- | --- |
| 30 rows, 10 recipes x 3 replicates, text `recipe` label, pH, temperature | 67 s (77-88 s reported from the web) | 5.1 s |
| 213 rows (Pichia HSA as published) | 14.4 s | 6.6 s (the row-number and OD columns are no longer inputs, see docs/UPLOADS.md "Column-role safety") |

The 30-row cost had one cause: the text `recipe` column was both the CV group
and a categorical model input. As an input it made the model a mixed GP whose
acquisition solved one continuous problem per label for each of the five
proposals (50 solves, 28 s, with the scipy `ABNORMAL` line-search warnings the
web log showed), and as the group it ran leave-one-group-out over groups that
were exactly the recipes (10 extra fits, 18 s). The label now groups replicates
only (`provenance` status `dropped_recipe_label`), and leave-one-group-out is
skipped when the groups are the recipes. A declared categorical is never
dropped this way.

Real mixed sheets are bounded too: every categorical combination is solved
exactly only while combinations x batch size is at most 64 solves; above that
the alternating optimizer is used (`proposal_optimizer: "mixed_alternating"`).
20 combinations x 5 proposals: 44 s before, 10 s after.

The three-start hyperparameter fit stays inside the budgets stated in
`kalos/core/surrogate.py` (4 threads): 0.47 s at 80 rows, 0.53 s at 300 rows,
4.6 s at 1,000 rows (stated: 1.0, 2.4, 9.0 s).

## Same upload, no refit

The engine keeps the analyses of the last few distinct uploads in memory
(`KALOS_ANALYSIS_CACHE`, default 4, 0 = off, at most 32). The same tenant sending the same
bytes with the same target, roles, sheet, anonymize flag and engine settings return the stored
analysis instead of fitting again, with `analysis_cache: {"hit": true}`.
`GET /api/latest` reads the published analysis and never fits. Background-job
workers are separate processes and do not share the cache.

## Packaged engine

`scripts/build_desktop_engine.sh [OUT_DIR]` (arm64 macOS, needs `uv`) bundles
the extras `ml`, `portal` and, when the checkout defines it, `assist`
(`KALOS_BUNDLE_EXTRAS` overrides the list; `assist-embed` is left out on
purpose). It builds
`OUT_DIR/kalos-engine` (a launcher) and `OUT_DIR/python/` (a relocatable
CPython 3.12.5 from python-build-standalone, `KALOS_BUNDLE_PYTHON` overrides
the exact version, with the dependencies from `uv.lock` installed with
`--require-hashes`, and the engine installed). The launcher runs
`python -I -m kalos.portal "$@"`, so the user's `PYTHON*` variables and user
site-packages are ignored. The script trims headers and the CPython test suite,
byte-compiles everything, then starts the engine once in desktop mode with a
temp folder, checks `/api/health`, and fails the build unless a request
without the token gets 401 and SIGTERM ends the engine with exit 0. It writes
`BUILD_INFO.txt`. It refuses to replace an `OUT_DIR` that is not empty and has
no `BUILD_INFO.txt`.

Measured: 955 MB on disk (torch is 468 MB of it), 272 MB as a zip, built in
about 20 s with a warm uv cache.

The shell runs `<bundle>/kalos-engine --desktop` with the environment above.

## Known gaps

- Data from open mode: an install that ran the engine without a token stored
  its campaigns under tenant `default`; with the desktop token they are under
  `local` and not shown. Moving them needs rewriting rows that the schema
  keeps append-only (`run_results`, `audit_log`), so it needs its own migration
  (offline command, backup first, audit row, only when `local` is empty). Not
  built.
- The bundle has no code signature or notarisation of its own; it relies on the app's
  signing. Intel Macs, Windows and Linux are not built.
- The analysis cache is per process and in memory; a restart refits.
