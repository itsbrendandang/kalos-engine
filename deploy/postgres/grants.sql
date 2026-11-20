-- Kalos on PostgreSQL: the grants for the RUNTIME role (docs/MIGRATIONS.md,
-- "PostgreSQL backend", roles).
--
-- Two roles, on purpose:
--   * the OWNER role owns the schema and its tables. Only
--     `python -m kalos.storage.migrations.pg_runner up` uses it, through
--     KALOS_DATABASE_MIGRATE_URL, to create and upgrade tables and triggers.
--   * the RUNTIME role is what the engine connects as (KALOS_DATABASE_URL).
--     It can read and write rows, but it does not own the tables, so it cannot
--     drop or disable the append-only triggers, ALTER or TRUNCATE tables, or
--     create new ones.
--
-- Run as the owner (or a superuser) after `pg_runner up`, e.g.:
--   psql "$KALOS_DATABASE_MIGRATE_URL" -v schema=public -v runtime_role=kalos_rt -f grants.sql
-- Re-run it after every `up` that adds tables.

GRANT USAGE ON SCHEMA :"schema" TO :"runtime_role";

-- Read-only: the schema version (the engine checks it at startup).
GRANT SELECT ON :"schema".schema_migrations TO :"runtime_role";

-- Append-only: rows may be added and read, never changed or removed. (The
-- triggers reject UPDATE/DELETE/TRUNCATE even for the owner; withholding the
-- privileges makes the runtime role fail before it reaches them.)
GRANT SELECT, INSERT ON :"schema".audit_log, :"schema".lineage_links,
    :"schema".lineage_retractions TO :"runtime_role";

-- Mutable records.
GRANT SELECT, INSERT, UPDATE, DELETE ON :"schema".projects, :"schema".project_campaigns,
    :"schema".project_factors TO :"runtime_role";

-- Identity columns (audit_log, lineage_links, lineage_retractions).
GRANT USAGE ON ALL SEQUENCES IN SCHEMA :"schema" TO :"runtime_role";
