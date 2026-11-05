# portal.db schema migrations

`<KALOS_STATE_DIR>/portal.db` (default `~/.kalos/portal.db`) holds the campaign
loop: campaigns, their published analyses, the tenant's default campaign, and
an audit log. Its schema is versioned in `schema_migrations`. The code is
`kalos/portal/migrate.py`; the tests are `tests/test_campaign_schema_v2.py`.

| Version | Shape | Code |
| --- | --- | --- |
| v1 | `campaigns(tenant PRIMARY KEY, state, updated_at)`, one campaign per tenant; analyses in memory plus `latest/*.json` | before `feat/campaign-schema-v2` |
| v2 | `schema_migrations`, `campaigns`, `analyses`, `tenant_defaults`, `audit_log` (below) | `feat/campaign-schema-v2` |
| v3 | adds the `jobs` table (background jobs, `docs/JOBS.md`) | `feat/background-jobs` |
| v4 | adds two triggers on `audit_log` that reject UPDATE and DELETE (append-only; below) | current (`SCHEMA_VERSION = 4`) |

A v1 database has no `schema_migrations` table.

## Schema v2

- `schema_migrations(version PK, name, applied_at)`.
- `campaigns(id PK, tenant, name, state, generation, created_by, created_at, updated_at, archived_at, UNIQUE(id, tenant))`.
  `id` is `cmp_` plus 32 lowercase hex characters (a uuid4). `state` is the v1
  JSON blob, unchanged, including its `generation` token. The `generation`
  column mirrors that token so the store's compare-and-swap is plain SQL. A
  NULL `state` is a campaign that exists but has never been seeded.
- `analyses(campaign_id PK, tenant, state, dataset, campaign_generation, updated_at)`,
  `FOREIGN KEY (campaign_id, tenant) REFERENCES campaigns(id, tenant)`. This is
  what `/api/latest` serves. It replaces the in-memory map and the
  `latest/*.json` files.
- `tenant_defaults(tenant PK, campaign_id)`, with the same composite FK. The
  legacy single-campaign routes (`/api/run`, `/api/latest`, `/api/campaign`,
  `/api/campaign/start|result|reanalyze`) act on this campaign. The first
  upload creates it.
- `audit_log(id, ts, tenant, subject, auth_method, action, campaign_id, job_id, outcome, request_id, detail)`.
  It holds identifiers and short codes, never tokens, uploaded data or file
  names.

Every connection sets `PRAGMA foreign_keys = ON` and `busy_timeout = 5000`.
The file is in WAL mode. The composite foreign keys mean an analysis or a
default can only point at a campaign of the same tenant.

## Schema v3: background jobs

Schema v3 adds one table, `jobs`, for the background-job queue (docs/JOBS.md).
The change is additive: no v2 table is altered, and no data moves.

- `python -m kalos.portal.migrate up` brings a v2 database to v3. It takes an
  online backup first, named `portal.db.pre-v3-<UTC timestamp>` (a v1 database
  gets `pre-v2-...` and goes through v2 to v3 in the same transaction), then
  creates the table and its two indexes in one `BEGIN IMMEDIATE` transaction.
- A v3 database is a no-op for `up`. The engine runs the same check at startup.
- `python -m kalos.portal.migrate down` steps back ONE version (v3 -> v2). It
  renames `jobs` to `jobs_v3_rolledback` rather than dropping it, so the queue
  history is kept. `down --to 1` steps all the way back.
- Job input files are NOT in the database: they live under
  `<KALOS_STATE_DIR>/jobs/`. A rollback leaves them in place; the next engine
  start with jobs enabled sweeps any that no live job refers to.
- `down` is for running OLDER code, which does not know the `jobs` table. An
  engine started on a v2 database migrates it back to v3 (with a fresh
  `pre-v3` backup), so do not start the new engine after a rollback you did not
  intend to undo.

Restoring the `pre-v3` backup gives the exact pre-upgrade state, as for v2.

## Refusals that come before any change

The version row is not the only evidence of a schema. Three shapes are refused
with `SchemaError`, and the refusal happens before a backup is taken or any table
is renamed. `up`, `up --dry-run`, `down` and the engine's startup all refuse them:

- **v2 tables but no version row.** `analyses`, `tenant_defaults` or `jobs` exist,
  and `schema_migrations` is missing or empty. Reading it as v1 would rename
  `campaigns` and break the foreign keys that name it, so the migration does not
  guess the version.
- **A recorded version whose tables do not match it** (for example, version 3 with a
  v1-shaped `campaigns`, or version 4 without the audit triggers). `dry-run` checks
  the shape too, so it no longer reports "nothing to do" for a database the engine
  refuses to start on.
- **A version newer than this code.**

The rename-based repair of an incompatible `campaigns` table (`campaigns_backup_N`)
runs only on a database read as v1 or v0. It is not run on any v2+ database.
`status` still reports, without refusing. Repair of a refused database is a manual
step on a copy, with the engine stopped.

## Schema v4: append-only audit log

Schema v4 adds no table and changes no row. It adds two triggers on
`audit_log`:

- `audit_log_no_update` (BEFORE UPDATE) and `audit_log_no_delete` (BEFORE DELETE)
  both `RAISE(ABORT, ...)`. The engine sees `sqlite3.IntegrityError` with the
  message `audit_log is append-only: UPDATE is rejected` (or `DELETE`). The row is
  unchanged.
- A statement that matches no row (for example, a DELETE with a WHERE clause that
  selects nothing) runs without error, as in any SQLite trigger.
- The engine has no delete path for audit rows. The retention setting archives
  rows and never deletes them (docs/HARDENING.md, "Audit retention").
- Do not use SQLite's `REPLACE` conflict resolution on `audit_log`. With
  `recursive_triggers` off, a REPLACE removes a row without firing the DELETE
  trigger. The engine does not use it.

`up` from v3 takes a verified backup named `portal.db.pre-v4-<UTC timestamp>` and
creates the triggers in one `BEGIN IMMEDIATE` transaction. A v3 database that is
already at v4 is a no-op.

`down` (v4 -> v3) drops the two triggers and forgets version 4. No row is touched.
Rolling back past v4 removes the append-only guard, so it is a decision for the
operator, with the `pre-down-<ts>` backup in hand. `down --to 3` stops at v3.

The v3 -> v4 migration is tested on a copy of a v3 database (up, down, up again,
audit rows unchanged, the source file never opened for writing):
`tests/test_audit_append_only.py`.

## What the v1 -> v2 migration does

It runs on its own when the engine first opens a v1 database
(`CampaignStore.__init__` calls `ensure_schema`). An operator can also run it
ahead of time with the CLI. Either way it does the following:

1. Reads the version, without writing. A v2 database is a no-op, which makes
   the migration idempotent. A version newer than the code makes it refuse to
   start (`SchemaError`).
2. Takes an online SQLite backup (the backup API, safe while other connections
   are open) to `portal.db.pre-v2-<UTC timestamp>` next to the database. The
   copy is written under a `.tmp` name, checked with `PRAGMA quick_check` and
   a table-set comparison, given the database's file mode, and only then
   renamed. **If the backup fails, the migration aborts and the database is
   left exactly as it was** (`MigrationError`, and the engine does not start).
3. Runs one `BEGIN IMMEDIATE` transaction:
   - It re-checks the version, in case another process migrated in the
     meantime.
   - A `campaigns` table of the wrong shape (for example, the composite-key
     table from an unmerged multi-campaign branch) is renamed to
     `campaigns_backup_N`. Its newest row per tenant is carried over, the same
     repair the v1 store did.
   - `campaigns` is renamed to `campaigns_v1`, or `campaigns_v1_2` and so on
     if that name is taken. **It is kept, not dropped.**
   - The v2 tables are created.
   - Each v1 row becomes one campaign named `Imported campaign`, with
     `created_by = 'migration:v2'` and `created_at = updated_at = the v1
     updated_at`. The state blob is copied byte for byte, corrupt JSON
     included, and the campaign becomes that tenant's default.
   - A tenant id that fails `^[A-Za-z0-9_-]{1,64}$` is imported **unchanged**,
     with a warning. It is never renamed, merged or overwritten. Auth refuses
     such ids, so the campaign can't be reached through the API until an
     operator deals with it, but no data is lost.
   - `latest/*.json` files are imported into `analyses`, but only when the
     filename maps to exactly one tenant (see the table below).
   - It records `schema_migrations(2, ...)`.

   Any error rolls the whole transaction back.
4. Switches the file to WAL, after the commit. The pragma rewrites the file
   header, so it waits until the migration has succeeded.

Every warning (repairs, corrupt rows, odd tenant ids, skipped files) is logged
at WARNING on the `kalos.portal` logger.

### `latest/*.json` import rules

The files are read, never moved, renamed or deleted.

| File | Result |
| --- | --- |
| `t-<hex>.json` (the PR #86 name: hex of the tenant's UTF-8 bytes) | Imported for the decoded tenant. |
| `t-<hex>.json` whose stem is also a v1 tenant id | Skipped as ambiguous. |
| `<name>.json` from before PR #86, `<name>` without `_` | Imported for tenant `<name>`. The old sanitiser only ever wrote `_`, so such a name was never rewritten. |
| `<name>.json` from before PR #86, `<name>` containing `_` | Skipped as ambiguous: `acme_bio.json` may belong to `acme_bio` or to `acme/bio`. |
| Both a `t-<hex>` file and an old-style file for one tenant | The `t-<hex>` file is imported (it is the one the code was serving). The other is skipped. |
| Corrupt JSON, or not a JSON object | Skipped. |
| A tenant with an analysis file but no v1 campaign | An unseeded default campaign is created to own the analysis. `/api/latest` keeps serving it, and `/api/campaign` still reports `has_campaign: false`, as it did on v1. |

## Runbook

The examples assume the compose layout (`KALOS_STATE_DIR=/home/kalos/.kalos`).
Add `--state-dir DIR` or `--db PATH` to target another location.

### Before upgrading

1. Make sure the regular backup sidecar has a recent `portal.db` backup
   (deploy/RUNBOOK.md).
2. Preview the migration. This writes nothing and takes no backup:

   ```sh
   python -m kalos.portal.migrate status
   python -m kalos.portal.migrate up --dry-run
   ```

   Check the tenant count and any `warn:` or `skip:` lines.
3. Make sure the state directory has room for one more copy of `portal.db`.

### Upgrade

Either deploy the new engine and let it migrate on first start, or migrate
explicitly with the engine stopped:

```sh
python -m kalos.portal.migrate up
python -m kalos.portal.migrate status    # expect: schema version: v4; journal_mode=wal
```

Keep the `portal.db.pre-v2-<ts>`, `pre-v3-<ts>` and `pre-v4-<ts>` copies (whichever apply) until the release is accepted.

### Verify

- `/readyz` returns `{"status": "ok"}`.
- Each tenant's `GET /api/campaign` shows the same `round`, `n_base` and
  pending runs as before.
- `GET /api/latest` shows the same dataset, except for files the migration
  reported as skipped.

### Roll back, option A: `down` (keeps work done on v2)

Stop the engine first. A running v2 process would see its tables vanish.

```sh
python -m kalos.portal.migrate down --dry-run   # the plan
python -m kalos.portal.migrate down
```

`down` does the following:

- Takes a `portal.db.pre-down-<ts>` backup.
- Rebuilds the v1 `campaigns(tenant PK, state, updated_at)` from each tenant's
  default campaign. A tenant whose default is unseeded gets its newest seeded,
  non-archived campaign instead, and a tenant with none gets no row.
- Renames every v2 table to `<name>_v2_rolledback` (nothing is dropped, so
  non-default campaigns stay recoverable) and drops only the v2 indexes.
- Writes each tenant's default analysis to `latest/t-<hex>.json`, moving any
  existing file aside to `<name>.pre-down-<ts>`.

The database stays in WAL mode, which the v1 code handles transparently. Then
deploy the previous engine.

Starting a v2 engine again later re-runs the migration. It takes a fresh
backup, and the second v1 copy becomes `campaigns_v1_2`.

#### Caveat: a rollback past v3 leaves `jobs_v3_rolledback` pointing at a renamed table

This applies when `down` goes from v3 all the way to v1 (`down --to 1`, or two
`down` steps). The v3 -> v2 step renames `jobs` to `jobs_v3_rolledback`. Its
foreign key still names the live parent, `campaigns (id, tenant)`, at that
point. The v2 -> v1 step then renames that parent to `campaigns_v2_rolledback`.
SQLite rewrites foreign-key references inside other tables on a rename, so the
rolled-back copy now reads `REFERENCES "campaigns_v2_rolledback" (id, tenant)`.

- The data is intact. Every job row is still in `jobs_v3_rolledback`, and
  `PRAGMA foreign_key_check` passes, because the parent it names exists.
- The copy is not self-consistent. Its foreign key names a table whose name
  says it was rolled back, not the live `campaigns`. Nothing in the v1 engine
  reads that table, so this has no effect on the running service. It matters
  only if someone restores the copy or inspects it by hand.
- The fix is optional. To remove the dangling reference, rebuild the copy
  without its foreign-key constraints (create a table with the same columns and
  no `FOREIGN KEY` clauses, copy the rows, drop the old copy, rename the new
  one), with the engine stopped and a backup taken first. Or leave the copy as
  it is. Do not bring the copy back by renaming it to `jobs`: the foreign key
  would still name the wrong table. A later `up` creates an empty `jobs` table
  and does not move the history; if the history must be live again, restore
  the `pre-v3` backup.

### Roll back, option B: restore the pre-v2 backup (exact pre-upgrade state)

Use this if `down` is not an option. It loses every write made since the
upgrade.

```sh
# engine stopped
mv portal.db portal.db.failed-v2
# keep the WAL with its database (SQLite pairs them by name)
[ -f portal.db-wal ] && mv portal.db-wal portal.db.failed-v2-wal
rm -f portal.db-shm
cp portal.db.pre-v2-<ts> portal.db
chown 10001:10001 portal.db             # compose layout
```

Then deploy the previous engine. The `latest/` files were never touched, so the
v1 code serves them as before.

### Failure modes

| Symptom | Meaning | Action |
| --- | --- | --- |
| `MigrationError: backup ... failed` at startup | The backup could not be written or verified (disk full, permissions). Nothing was changed. | Fix the disk or permissions and restart. |
| `SchemaError: ... newer than this code` | An older engine was started on a newer database. | Deploy the newer engine, or run `down` with the newer code. |
| `SchemaError: ... does not have the v2 shape` | Tables were changed by hand after the migration. | Restore the pre-v2 backup, or repair by hand. |
| `database is locked` during `up`/`down` | Another process holds the write lock for longer than `busy_timeout`. | Stop the engine and retry. |

## Operational notes

- **WAL and the backup sidecar.** `deploy/backup/backup.sh` uses
  `sqlite3 .backup`, which is correct for WAL too. Its comments still describe
  rollback-journal mode. The sidecar mounts the state volume read-only, and
  SQLite can read a WAL database from a read-only mount only while the
  `-wal`/`-shm` files exist. This has **not been verified** on the compose
  stack yet (see the PR).
- Backups and `.pre-*` copies contain tenant data. They get the database's
  file mode, so keep the state directory `0700`.
- `audit_log` grows without bound. The engine never deletes it (schema v4). The optional archive copies old rows to `<state dir>/audit-archive/`; it does not free space in `portal.db`. Physical removal is a separate decision and needs its own reviewed migration.
