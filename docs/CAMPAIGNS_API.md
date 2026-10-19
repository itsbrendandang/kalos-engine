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
| Scopes | `read` for GET; `write` for POST and PATCH; `admin` as well for archiving or unarchiving. Missing scope: `403 {"detail": "requires '<scope>' scope"}`. Scope is checked before the campaign id is looked at. |
| Tenant | Always the authenticated principal's tenant. A tenant in the path, query, body or headers is ignored (an unknown body field is a 400). |
| Campaign ids | `cmp_` + 32 lowercase hex characters (a uuid4), validated before the store is read. |
| Not found | A malformed id, a missing id and another tenant's id all get exactly the same response: `404 {"error": "campaign not found"}` (same status, headers and body bytes). The input is never echoed. |
| Archived | An archived campaign can still be read. Every write to it gets `409 {"error": "the campaign is archived; it can be read but not changed"}`. |
| Body errors | `400 {"error": "invalid request body: <field>: <reason>"}`. The same envelope as `/api/campaign/*`, and the submitted value is never echoed. |
| Body size | JSON bodies have the default cap (`KALOS_MAX_REQUEST_BYTES`, 1 MiB). `POST /api/campaigns/{id}/upload` has the upload cap (`KALOS_MAX_UPLOAD_MB` + multipart allowance). Over the cap: `413`. |
| Heavy work | Upload and reanalyze share the engine's single analysis slot. When it is busy they return `503` with `Retry-After`. |
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

Uploads run synchronously for now. Background jobs are a later slice.

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

## Not in this slice

- Background jobs for upload and reanalyze (they run synchronously).
- JWT principals (E2). `auth_method` will read the principal's
  `auth_method` once it exists. `tests/test_cross_tenant_isolation_jwt.py` is
  a skipped TODO stub until E2 lands.
- Deleting a campaign, changing the default campaign, and reading the audit
  log over HTTP.
