# Records: projects, factors, lineage, outcomes and units

Records sit above and between campaigns. They were ported from the Voyager
backend (NOTICE) and re-implemented on Kalos's own concepts, so nothing Kalos
already had is stored twice. Schema v9 (docs/MIGRATIONS.md); code in
`kalos/records/` (rules), `kalos/storage/records_repo.py` (SQL) and
`kalos/api/routers/{projects,lineage,units}.py` (HTTP).

## How Voyager maps onto Kalos

| Voyager | Kalos | Notes |
| --- | --- | --- |
| `Project` (name, objective, status, created) | **project** (`projects`) | Grouping above campaigns. `status` is a fixed vocabulary. Optional free-text `sponsor` replaces the customer link. |
| `ProjectStatus` table (Drafting, In Progress, Completed, Terminated) | `kalos.records.projects.STATUSES` + `GET /api/project-statuses` | Code, not a table: `drafting`, `in_progress`, `completed`, `terminated`. |
| `Project.MlConfig` (JSONB) | campaign **setup** (`/api/campaigns/{id}/setup`) | Not ported: model inputs live in one place, the campaign setup. |
| `Experiment` (+ `DataFile` JSONB, `ExperimentStatus`) | **campaign** (its uploaded sheet, analyses, stage track) | Not duplicated. A project groups campaigns (`project_campaigns`). A campaign's state comes from its own loop and its stage/gate registry. |
| `ExperimentIteration` (iteration number) | campaign **round** | Not duplicated. Rounds are recorded with each run's prediction (`run_predictions.round`, schema v8). |
| `ExperimentLink` (parent, child) and its recursive lineage query | **lineage link** (`lineage_links`, append-only) between campaigns or runs, with a relation and the stage of each end | The most valuable part. Bounded traversal, cycle refusal, tenant-checked ends, retraction with a reason instead of edits. |
| `Factor` (name, description, lower/upper int bound, data unit) | **project factor** (`project_factors`) → campaign setup **bounds** | Real-number bounds, `lower < upper`, unit from the registry, physically impossible ranges refused, role `setting` or `covariate`. `GET .../factors/setup-bounds` maps them onto a campaign setup; nothing is applied automatically. |
| `DataUnit` table (public writes) | **units registry** (`kalos/normalize/unit_registry.py`, `GET /api/units`) | Code, read-only, the same for every tenant; consistent with the header-unit parser and the cell converter. |
| `ExperimentDataPoint` (iteration × factor → JSONB value) | a **run's recipe** | Not duplicated. |
| `Outcome`, `OutcomeDataPoint` (JSONB value) | a run's **result** (`run_results`, append-only, every correction kept) | Not duplicated. `GET /api/projects/{id}/outcomes` shows the newest measured version beside the prediction recorded at start, never merged. Results are written only through `POST /api/campaigns/{id}/runs/{run_id}/result`. |
| `Customer` | optional `sponsor` label on a project | Not ported as a table (below). |
| `Employee`, `EmployeeRole`, `AuthorizationRole`, `EmployeeGoogleAuth` | kalos-web organizations and members; engine scopes `read`/`write`/`admin` | Not ported (below). |
| Logging.py correlation id | request id on every log line and on the records' audit rows (`kalos/api/request_context.py`); a caller's `X-Request-ID` (or `X-Correlation-ID`) is used and echoed | Ported. A server-minted id is not echoed (a random header would break byte-identical 404s). The rotating log file is not ported. |
| psycopg2 PostgreSQL | optional PostgreSQL backend (`KALOS_DATABASE_URL`) | Ported as an option; SQLite stays the default (docs/MIGRATIONS.md, "PostgreSQL backend"). |

## Routes

Every route needs a token with the scope shown (docs/route-scopes.json). The
tenant is always the verified token's. Malformed, missing and other-tenant ids
get one byte-identical 404. Errors use the engine envelope `{"error", "code"}`
and never echo what was sent. Every successful write has its audit row in the
same transaction, carrying the request id; a refused write gets a best-effort
audit row with the refusal code only, at most 20 per token per minute (the
rest are counted in one `records.refusals_suppressed` row).

Text fields (project name, objective and sponsor; factor name and description;
link note and retraction reason) are stored and returned exactly as typed after
trimming. No export includes them today; when one does, it must escape cells
that start with `=`, `+`, `-`, `@`, TAB or CR (spreadsheet formulas) with the
engine's export sanitiser (tracked in Linear BRE-151).

| Method | Path | Scope | What |
| --- | --- | --- | --- |
| GET | `/api/project-statuses` | read | the status vocabulary |
| GET | `/api/projects?status=&limit=&offset=` | read | projects, newest update first, with `n_campaigns`; `limit` ≤ 200 |
| POST | `/api/projects` | write | `{name, objective, status?, sponsor?}`; 409 `name_taken`, `too_many_projects` (1000 per tenant) |
| GET | `/api/projects/{project_id}` | read | the project, its campaigns (name, archived, recorded current stage) and factors |
| PATCH | `/api/projects/{project_id}` | write | any of `name`, `objective`, `status`, `sponsor` (`null` clears it) |
| PUT | `/api/projects/{project_id}/campaigns/{campaign_id}` | write | attach one of your campaigns (idempotent); 409 `campaign_in_other_project`; at most 200 per project |
| DELETE | `/api/projects/{project_id}/campaigns/{campaign_id}` | write | detach; the campaign and its links are untouched; works for a campaign that can no longer be read |
| GET | `/api/projects/{project_id}/factors` | read | factor definitions |
| PUT | `/api/projects/{project_id}/factors` | write | create or replace one factor by `name`; 400 `invalid_bounds`, `unknown_unit`, `impossible_bounds`; at most 64 |
| DELETE | `/api/projects/{project_id}/factors?name=` | write | remove a factor definition |
| GET | `/api/projects/{project_id}/factors/setup-bounds?campaign_id=` | read | the `setting` factors as a setup `bounds` object; with a campaign, factors that are not continuous settings of its latest analysis are listed as skipped. `applied` is always false |
| GET | `/api/projects/{project_id}/outcomes?limit=&offset=` | read | per run of the project's campaigns: newest measured value (versions, correction) and the prediction recorded at start; `limit` ≤ 500 |
| GET | `/api/projects/{project_id}/lineage` | read | the project's campaigns and the active links among them (≤ 2000) |
| GET | `/api/lineage/relations` | read | `next_round`, `scale_up`, `scale_down`, `follow_up`, `replicate`, `derived` |
| POST | `/api/lineage/links` | write | `{parent_campaign_id, child_campaign_id, relation, parent_run_id?, child_run_id?, parent_stage?, child_stage?, note?}` |
| GET | `/api/lineage/links/{link_id}` | read | one link with its retraction, if any |
| POST | `/api/lineage/links/{link_id}/retract` | write | `{reason}`; the link and the reason are both kept; 409 `already_retracted` |
| GET | `/api/campaigns/{campaign_id}/lineage?depth=&include_outcomes=` | read | ancestors and descendants over active links, walked one indexed query per hop (depth 1..32, default 6; ≤ 500 campaigns; ≤ 2000 links); `truncated` names any bound that cut the answer; `include_outcomes` adds the measured result and prediction at start of each run named by a link |
| GET | `/api/units` | read | the units registry |

Lineage rules: both ends must be the caller's campaigns (a foreign id is a 404,
so a link can never cross tenants); a run end must be a run the campaign started
or recorded a result for; a link inside one campaign joins two of its runs; no
self-link, no duplicate active link, no cycle (across campaigns the campaign
graph stays acyclic; inside a campaign the run graph does). A stage left out
defaults to the campaign's RECORDED current stage (`/api/campaigns/{id}/stages`),
never a computed one. At most 20,000 links per tenant.

## Not ported, and why

The user can overrule any of these.

- **Google ID-token login and HS256 JWT signing.** kalos-web owns sign-in
  (Better Auth with TOTP) and mints short-lived EdDSA tokens that the engine
  verifies against a public key (docs/HARDENING.md). HS256 with a shared secret
  means every service that verifies a token can also mint one, and a leaked
  secret forges any identity. The engine never signs tokens.
- **`DISABLE_AUTH=1`.** An environment variable that turns authentication off
  is one misconfigured deploy away from an open engine. Kalos's open mode is
  loopback-only and refused in production (`KALOS_ENV=production`).
- **Publicly writable endpoints.** Voyager's factor, iteration, data-point and
  data-unit routes (and the project-status list) needed no token. Every Kalos
  route declares a scope; reads need `read`, writes need `write`.
- **Employees, roles and Google identities.** kalos-web organizations and
  members own users; the engine sees a tenant and scopes in a verified token.
- **Customers as a CRM table.** Kalos is software for a lab's own data, not a
  services business. A sponsor or client name is an optional free-text label on
  a project, stored and shown, never interpreted.
- **Project `MlConfig`.** Model settings live in the campaign setup.
- **Per-environment `.env.<env>` files (`BE_ENV`/`DB_ENV`).** Kalos has one
  profile switch, `KALOS_ENV` (`development`/`production`, unknown values
  refused), and reads secrets from the deployment's environment or files outside
  the source tree. Voyager's CI wrote database passwords into uploaded artifacts;
  Kalos's CI uses a throwaway password for a service container only.
- **The rotating log file.** The engine logs to stderr; the container runtime
  keeps and rotates logs.
- **Voyager's Dockerfile.** Kalos's `deploy/engine/Containerfile` already pins
  the stack, runs as a non-root user and uses CPU-only torch; nothing in the
  Voyager image improves on it. The only CI pattern taken is the PostgreSQL
  service container.
