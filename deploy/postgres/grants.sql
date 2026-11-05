-- Example grant model for the kalos PostgreSQL backend. No secrets here:
-- set the password separately (\password kalos_engine in psql, or your
-- secret manager). Run as a superuser or the database owner, AFTER the
-- schema exists (run `python -m kalos.portal.migrate up` as the owner role
-- first, with KALOS_DATABASE_URL pointing at the owner role).
--
-- Two roles:
--   kalos_owner   owns the schema and tables; runs migrations; not used by the engine.
--   kalos_engine  what the engine connects as; NOT an owner, NOT a superuser.
--
-- Why: the append-only triggers on audit_log and run_results stop UPDATE,
-- DELETE and TRUNCATE for every role, but a table OWNER (or a superuser) can
-- ALTER TABLE ... DISABLE TRIGGER. The REVOKEs below are a second, independent
-- layer: kalos_engine has no UPDATE, DELETE or TRUNCATE privilege on those
-- tables in the first place. Neither layer stops a database superuser.
--
-- Replace `public` if you use a dedicated schema (and set search_path in
-- KALOS_DATABASE_URL: ?options=-csearch_path%3Dkalos).

CREATE ROLE kalos_engine LOGIN;  -- set the password out of band

GRANT CONNECT ON DATABASE kalos TO kalos_engine;
GRANT USAGE ON SCHEMA public TO kalos_engine;

-- Mutable tables: the engine reads and writes them.
GRANT SELECT, INSERT, UPDATE, DELETE ON campaigns, analyses, tenant_defaults, jobs TO kalos_engine;

-- Append-only tables: read and insert only.
GRANT SELECT, INSERT ON audit_log, run_results TO kalos_engine;
REVOKE UPDATE, DELETE, TRUNCATE ON audit_log, run_results FROM kalos_engine;

-- Schema bookkeeping: read only (the engine never migrates a server database).
GRANT SELECT ON schema_migrations TO kalos_engine;

-- Nothing else: no CREATE on the schema, no TRUNCATE anywhere, no ownership.
REVOKE CREATE ON SCHEMA public FROM kalos_engine;
