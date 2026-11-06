# Benchling connector

Kalos can read runs from Benchling and write two kinds of things back: draft run
plans and recorded results. This page says what moves in each direction, whose name
Benchling puts on each write, and the limits.

The connector is off by default. Turn it on with `KALOS_INTEGRATIONS=benchling`.
While it is off, every path under `/api/integrations/benchling/` answers the same
404 as a path that does not exist.

**Status.** The connector was built from Benchling's public API specs (v2 and v3
`openapi.yaml`) and docs pages. It was tested against a fake Benchling that returns
the shapes in those specs. It has not been run against a real Benchling tenant. See
"Not verified" at the end.

## How to connect

1. In Benchling, an admin creates a Benchling App (Developer Console). The app
   needs read access to the projects and studies you want to import, and write
   access to the studies and result schemas you want Kalos to write to. Note the
   app's client id, client secret and app definition id (`appdef_...`).
2. On the Kalos engine, the operator sets `KALOS_SECRET_KEY` (at least 32 random
   characters, from the environment or a secret manager). Kalos uses it to encrypt
   the client secret at rest. If the key changes, stored secrets cannot be read and
   each workspace must connect again.
3. A Kalos workspace admin calls:

   ```
   POST /api/integrations/benchling/connect
   {"base_url": "https://<tenant>.benchling.com",
    "client_id": "...", "client_secret": "...",
    "app_definition_id": "appdef_..."}
   ```

   Kalos first asks Benchling for a token. If Benchling refuses the credentials,
   nothing is stored. The secret is never returned, written to a log, or put in
   the audit log. Only `https://<tenant>.benchling.com` URLs are accepted.
4. `GET /api/integrations/benchling/status` shows the connection (never the secret),
   whether the stored secret can still be opened, the last imports, the last pushes
   with their Benchling ids, and change notices.

Each Kalos workspace (tenant) has its own connection. No route can read or use
another workspace's connection, imports, pushes or change notices.

## What is pulled

`GET /studies` lists Bioprocess studies (v3) and projects (v2).

`POST /import` turns Benchling runs into a Kalos run sheet, then uploads that sheet
through the normal upload path. The parser, size caps, data checks, the analysis
slot and the error codes are the same as for a file upload.

- **From a Bioprocess study (v3):** `{"study_id": "stdy_...", "target_field": "<result field id>"}`
  (add `"result_schema_id"` to skip the search for the schema). Kalos reads the
  study's procedure runs, their conditions and replicates, the planned and measured
  parameter values, and the results in the chosen result schema with that schema's
  field definitions.
  - One row per condition replicate (one physical run).
  - Inputs are the planned parameter values (the setpoints). A parameter that has
    only measured values is added too, and the import report says so, so you can
    check it is not a stand-in for the outcome.
  - The target is the chosen numeric result field. Its unit goes into the column
    name, for example `Titer [g/L]`.
- **From assay runs (v2):** `{"assay_run_schema_id": "...", "result_schema_id": "...",
  "target_field": "<field system name>"}`. One row per assay run. Number and
  dropdown run fields are inputs. Text, link and date fields are left out, and the
  report counts them.
- Add `"campaign_id"` to import into an existing campaign. Otherwise Kalos makes a
  new one. A campaign that already has runs is refused unless you send
  `"replace": true`, the same as a file upload.

Every import returns and stores:

- a **column provenance table**: for each column, the Benchling id it came from (a
  parameter version, a result field or a run field), the display name, the unit and
  how many rows have a value;
- an **import report**: how many runs were seen, how many became rows, and each
  skipped run counted by reason (no result, more than one result, a result that is
  not a number, two different values for one parameter, no parameter values), with
  up to five example ids per reason;
- the **sheet filename**, which records the source id and the fetch time in UTC,
  for example `benchling-stdy_9CYiCggY-20261106T142501Z.csv`, plus the sheet's
  SHA-256.

Kalos never guesses a value. If a run is unclear, it is skipped and counted.

## What is written (drafts only)

**Proposals → a draft run plan.** `POST /push-proposals {"campaign_id", "study_id"}`
writes the campaign's current proposals into the study it was imported from as one
procedure run plan. Each proposal becomes a condition (one per method) with one
replicate. Kalos never promotes the plan. Promoting starts a run in the lab, and a
person does that in Benchling. The client refuses any `:promote` call before it is
sent. Columns that cannot be planned (for example, measured-only parameters) are
left out and listed in the response.

**Recorded results → Benchling results.** `POST /push-results {"campaign_id"}` writes
the results recorded in Kalos that are not in Benchling yet. Each one becomes a
result in the result schema and field the campaign was imported with. All are
written in one result transaction: if any part fails, the transaction is aborted
and nothing is written. A result already pushed is not pushed again. A correction
in Kalos (a new version) is written as a new Benchling result. The earlier one
stays as it is.

**Analysis report → a file on an entry.** Add `"report_entry_id"` and `"report_field"`
(a blob-link field on that entry) to `/push-results` to upload the campaign's
analysis as a JSON file and link it from the entry.

## Who the audit trail names

- **Results and the report file** are written only with the logged-in user's own
  Benchling token, sent in the `X-Benchling-Delegated-Token` header. Without it,
  Kalos answers 400 and makes no Benchling call. The app's own credentials are
  never used for these writes. So in Benchling's audit log and `createdBy` field,
  the person is named, not the app. Kalos uses the token for that one request and
  does not store or log it. Getting the token (Benchling's sign-in and consent
  screen) is the web app's job.
- **Draft run plans** use the user's token when the header is present. Without it
  they use the app's token, and Benchling names the app as creator.
- **In Kalos**, every push writes one row to the connector's push audit table. The
  row holds: who in Kalos, which Benchling identity wrote (`app` or `delegated`),
  the creator id Benchling returned, the campaign, the study or schema, the outcome,
  and every Benchling id returned (plan and condition ids, transaction and result
  ids, file id). If a push fails part-way, the ids that were already created are
  recorded too. The table rejects updates and deletes. A matching row
  (`benchling.import`, `benchling.push_proposals`, `benchling.push_results`,
  `benchling.connect`) goes into the main audit log.

## Change detection

- **Webhooks:** `POST /api/integrations/benchling/webhook/{tenant}` takes
  `v2.assayRun.created` and `v3.run.created`. Kalos checks each delivery the way
  Benchling's "Webhook Security" page describes:
  - The timestamp must be within 5 minutes.
  - An ECDSA P-256 signature over `{webhook-id}.{webhook-timestamp}.{body}` must
    match a key from the app's published key set. The key set is cached for at most
    6 hours and fetched again once when no key matches.
  - The payload must name this workspace's app and Benchling URL.

  Any failed check gets the same 401. A repeated delivery is a no-op. Other event
  types are acknowledged and ignored.
- **Polling (Bioprocess):** Benchling has no events for procedure runs or results.
  `POST /poll {"study_id", "result_schema_id"?}` lists what changed since the last
  poll and records each change once.

Change notices only tell a person that new data exists. Nothing is imported
automatically.

## Limits

- **Rate.** At most 60 requests in any 30 seconds per Benchling tenant, per Kalos
  process. 60 per 30 s is Benchling's smallest documented budget. On a 429, Kalos
  waits for `Retry-After` (v3) or `x-rate-limit-reset` (v2), or backs off
  exponentially, then tries again (up to 5 times). Several Kalos processes do not
  share the budget.
- **Size of a study.** About 2 calls per procedure run, plus 4 per 100 replicates,
  plus 1 per 100 results. A 1,000-row study takes roughly 100 calls, so allow about
  a minute.
- **Tokens.** App tokens last 15 minutes. Kalos asks for a new one a minute before
  expiry and once after a 401.
- **Retries.** Reads are retried after 5xx errors. Writes are not, because a write
  that failed with a 5xx may still have been applied.
- **Run plans.** At most 100 conditions per method and 500 replicates per plan.
  Conditions are sent 25 per call.
- **Results.** Sent 100 per call inside one transaction.
- **Report file.** Up to 10 MB.
- **Beta endpoints.** The Bioprocess endpoints are beta in Benchling's v3 API and
  can change without notice. Kalos sends `EARLY-ACCESS: true` only on the paths the
  spec marks beta.

## Not verified (no tenant was available)

- How a v3 result points at a run. Kalos assumes one of the result's link fields
  points at the condition replicate, at its method execution instance, or at the
  procedure run when that run has one replicate. Other results are skipped and
  counted.
- The shape of a planned or measured `value` on read. Kalos assumes the same
  TypedValue shape the plan input uses, or a bare number or text.
- Whether a v3 result schema id is accepted as a v2 `schemaId` when writing results.
  The v2 field key used is the field's `systemName`.
- Whether the procedure run's flowchart config version id is the
  `workflowFlowchartConfigVersionId` a new run plan expects. It can be overridden in
  the request.
- The v2 unit of the target field is matched by display name.
- Which app permissions each call needs, and how Benchling counts requests made
  with a user token against the app's budget.
