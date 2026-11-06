# Architecture

Status: today's system plus a clearly marked planned design. Nothing marked "planned" is implemented.

## Today

kalos-web has two modes (its `main` has sign-in; its `redesign-notion` branch, not yet merged, adds the desktop app's local mode).

**Hosted mode.** The person signs in to kalos-web (Google account plus an authenticator code; invite only). kalos-web maps the organization to a tenant and signs an EdDSA token for each request (`iss kalos-web`, `aud kalos-engine`, at most 120 s). The engine verifies it with `KALOS_JWT_PUBLIC_KEYS_FILE` and reads the tenant from the token. Service-account tokens (`KALOS_AUTH_TOKENS`) remain for scripts and the compose stack.

**Desktop (local) mode.** The Mac app starts the engine on 127.0.0.1 with a secret made for that launch: the engine gets its SHA-256 in `KALOS_AUTH_TOKENS` (tenant `local`), the web server gets the plain value as `KALOS_ENGINE_TOKEN` and sends it as a bearer on every call. There is no sign-in because there is one person on one machine; the web server itself only answers its own window (a second per-launch secret). See `docs/DESKTOP.md`.

```mermaid
sequenceDiagram
  participant B as Browser
  participant W as kalos-web (Next.js + server-side proxy)
  participant E as Engine (FastAPI)
  participant S as SQLite (per-tenant campaign row)
  B->>W: upload / plan / record (hosted: signed-in session; desktop: the launch secret)
  W->>E: same request + EdDSA token (hosted) or the per-launch bearer secret (desktop)
  E->>E: verify the token (or hash + constant-time compare), resolve tenant + scopes
  E->>S: read/write campaign keyed by tenant
  E-->>W: analysis, proposals, campaign state
  W-->>B: rendered result
```

- Identity: operator-provisioned bearer tokens (`kalos/portal/auth.py`), plus engine-side verification of EdDSA engine JWTs (`kalos/portal/jwt_auth.py`, `docs/ENGINE_TOKEN.md`). Tenant = a string on the token or the JWT `tenant` claim.
- Persistence: `<KALOS_STATE_DIR>/portal.db` (default `~/.kalos`, schema v5 on `gxp-integrity` and above, v3 on `main`), several campaigns per tenant (`kalos/portal/campaign.py`, `docs/CAMPAIGNS_API.md`); the legacy routes act on one default campaign per tenant. Latest analysis in memory plus a per-tenant JSON cache.
- Compute: the GP fit runs in the request, or as a background job in a local worker process (`kalos/portal/jobs.py`, `docs/JOBS.md`). A second concurrent synchronous upload gets a 503 busy answer (`kalos/portal/busy.py`).

## Hosted request flow (both sides built; organisation membership is kalos-web's)

```mermaid
sequenceDiagram
  participant B as Browser
  participant W as kalos-web
  participant E as Engine
  B->>W: sign in (Better Auth session)
  W->>W: resolve user + active organisation
  W->>E: request + EdDSA JWT (iss kalos-web, aud kalos-engine, lifetime <= 120 s)
  E->>E: verify signature, iss, aud, exp independently; tenant = the token's claim (mapped from the organisation by kalos-web)
  E-->>W: result (long fits can be submitted as background jobs, built: docs/JOBS.md)
```

The engine has no organisation model of its own: it reads a tenant string from the token. kalos-web keeps organisations, membership and the organisation-to-tenant map (`KALOS_TENANT_MAP_FILE`). The compose stack in `deploy/` still passes a service token to kalos-web and has not been updated for sign-in.

Open design questions: key distribution and rotation for the EdDSA key.
