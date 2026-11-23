# kalos deployment runbook

Covers the stack in `deploy/compose.yaml`: the kalos engine
(FastAPI/uvicorn, `python -m kalos.portal`), kalos-web (Next.js), and a
SQLite backup sidecar.
Read `deploy/compose.yaml`'s header comment first if you have not. It
explains the one front door (kalos-web), how the web proxy authenticates to
the engine with per-request EdDSA engine tokens, and why the engine needs that
auth configured to start at all.

TLS termination is out of scope: put a host-level reverse proxy in front of
the one published port. Nothing here does TLS itself.


## Verified quickstart (first real deployment, 2026-09-10)

The whole stack, from a clean machine with Docker running:

```
cd kalos/deploy
cp .env.example .env        # at that time: KALOS_AUTH_TOKENS, KALOS_ENGINE_TOKEN, KALOS_ANON_SALT
docker compose up -d --build
```

That run used the static-token setup this stack had then. The current stack
authenticates the web to the engine with EdDSA engine tokens instead (see
"Keys" below); follow "Deploy" for a new installation.

Then open http://localhost:3000 - the ONLY published port (bound to loopback by default; see "Who can reach the web app" below). Verified numbers
from the reference run (M-series Mac, Docker VM 11 CPU / 8 GB): engine image
1.82 GB, web 314 MB; first analyze on the 160-run demo sheet completed through
the authed proxy in ~19s; a concurrent second upload got the honest 503 busy
answer in 0.1s.

Env knobs that exist because this deployment found the need for them:
- `KALOS_ENGINE_TIMEOUT_MS` (web) - proxy timeout for engine calls, default
  120000. The original 30s guess 502'd mid-fit on a cold container.
- `KALOS_TORCH_THREADS` (engine) - the compose passes 0 (= do not pin) by
  default: the bare-metal 4-thread pin measured a >12x pathological slowdown
  (47s -> 600s+) on this image's linux-aarch64 torch/OpenBLAS build.
- `KALOS_CV_WORKERS` (engine) - worker processes for cross-validation fold
  fits, default one per CPU; `1` fits them in the engine process. The numbers
  do not depend on it.
- One analysis at a time is enforced engine-side (HTTP 503 + Retry-After when
  busy; kalos/portal/busy.py): a client that gives up cannot stack orphaned
  fits against its own retry.

## Prerequisites

- Docker with Compose v2 (`docker compose version`, not the standalone
  `docker-compose` v1 binary).
- This repo (`kalos-engine`) and `kalos-web` checked out as SIBLING directories,
  e.g. both under `~/GitHub/`. `deploy/compose.yaml`'s `web.build`
  section resolves `kalos-web`'s path as `../../kalos-web` relative to this
  file - if your checkout layout differs, that one path is the thing to
  edit, nothing else in this pack assumes a particular layout.
  The web image is built from kalos-web's own `Dockerfile`, at the root of
  that checkout.

## Container files

One directory per image, each with a `Containerfile` (read by both Docker
and Podman), plus one compose file:

| Path | Image | Build context |
|---|---|---|
| `deploy/engine/Containerfile` | `kalos-engine` (FastAPI/BoTorch portal) | repo root |
| `kalos-web/Dockerfile` (in the kalos-web repo) | `kalos-web` (Next.js) | the sibling `kalos-web` checkout |
| `deploy/backup/Containerfile` | `kalos-backup` (SQLite backup sidecar) | `deploy/backup/` |
| `deploy/compose.yaml` | the whole stack | - |

The repo-root `.containerignore` (with `.dockerignore` as a symlink to it)
is an allowlist: the engine build context contains only `pyproject.toml`,
`README.md` and `kalos/`. A local `.env`, `deploy/backups/`, `.venv` and
`.git` never reach the builder. Add a path there only if the engine
Containerfile starts to COPY it.

### Podman

The same files work under Podman:

```bash
podman build --format docker -f deploy/engine/Containerfile -t kalos-engine:local .
podman compose -f deploy/compose.yaml up -d --build
```

Docker looks for a file named `Dockerfile` by default, so outside compose
pass the file explicitly: `docker build -f deploy/engine/Containerfile .`
(Podman finds `Containerfile` on its own).

Use `--format docker`: Podman's default OCI image format drops the
`HEALTHCHECK` instruction, and compose's `depends_on: service_healthy`
gate and the CI boot test rely on it.

## Deploy

```bash
cd kalos/deploy

# 1. Real environment, never committed.
cp .env.example .env
# Edit .env: KALOS_ENGINE_JWT_KID, KALOS_ANON_SALT, BETTER_AUTH_SECRET,
# BETTER_AUTH_URL, GOOGLE_CLIENT_ID/SECRET and AUTH_TRUSTED_PROXIES at minimum.

# 2. The engine-token key pair, the engine's JWKS and an empty tenants map
#    (see "Keys" for what each file is).
mkdir -p secrets engine-config web-config
# OpenSSL 1.1.1+ (`openssl genpkey -algorithm ed25519`) also works; macOS's
# LibreSSL openssl does not support Ed25519, so this uses Node (needed anyway).
node -e "process.stdout.write(require('crypto').generateKeyPairSync('ed25519') \
  .privateKey.export({type: 'pkcs8', format: 'pem'}))" > secrets/engine-jwt-ed25519.pem
KALOS_ENGINE_JWT_PRIVATE_KEY_FILE=secrets/engine-jwt-ed25519.pem \
KALOS_ENGINE_JWT_KID="$(grep '^KALOS_ENGINE_JWT_KID=' .env | cut -d= -f2)" \
  node ../../kalos-web/scripts/engine-jwks.mjs > engine-config/engine-jwks.json
echo '{}' > web-config/tenants.json
# Both containers run as uid 10001. The private key stays readable by the web
# container only; the engine gets the public JWKS only.
chmod 600 secrets/engine-jwt-ed25519.pem
chown -R 10001:10001 secrets engine-config web-config   # or with sudo

# 3. The backup sidecar writes to a host directory outside the named volumes,
#    which Docker would otherwise create root-owned.
mkdir -p backups
chown -R 10001:10001 backups   # or: sudo chown -R 10001:10001 backups

# 4. Check the configuration, build, create the web's auth database, start.
docker compose config --quiet
docker compose build
docker compose run --rm --no-deps -e AUTH_MIGRATE=only web   # migrate, then exit
docker compose up -d

# 5. Confirm the engine enforces auth (it exits non-zero instead when the JWKS
#    or KALOS_ANON_SALT is missing or malformed): an unauthenticated call must
#    get 401, and the log must not say "auth is OPEN".
docker compose exec engine python -c "import urllib.request as u, urllib.error as e
try: u.urlopen('http://127.0.0.1:8050/api/campaigns'); print('OPEN: fix the JWKS')
except e.HTTPError as x: print(x.code)"          # expect 401
docker compose logs engine | grep -i "auth is OPEN"   # expect no output

# 6. Health.
docker compose ps                    # engine and web should show "healthy"
curl -f http://localhost:${KALOS_WEB_PORT:-3000}/
```

Then onboard the first organization ("Onboarding" below).

## Keys

- `secrets/engine-jwt-ed25519.pem`: the Ed25519 private key (PKCS#8 PEM)
  kalos-web signs a 60 s engine token with for every proxied request. Only the
  web container mounts it (`KALOS_ENGINE_JWT_PRIVATE_KEY_FILE`). One key pair
  per environment.
- `engine-config/engine-jwks.json`: the matching PUBLIC key as a JWKS,
  printed by kalos-web's `scripts/engine-jwks.mjs` (it never prints private
  material, and the engine refuses a key that carries `d`). The engine reads it
  from `KALOS_JWT_PUBLIC_KEYS_FILE` and verifies issuer, audience, expiry,
  `kid`, tenant and scopes (`docs/ENGINE_TOKEN.md`). The `kid` must equal the
  web's `KALOS_ENGINE_JWT_KID`.
- The engine re-reads the JWKS when the file changes. Replace it atomically
  (write a new file in `engine-config/`, then `mv` it over the old one); the
  directory is mounted, so the rename is seen.
- **Rotation**: generate a new key with a new kid; add its public key to the
  JWKS next to the old one; switch the web to the new key and kid
  (`docker compose up -d web`); wait at least 150 s (the engine's maximum
  token lifetime plus leeway); remove the old key from the JWKS. On a
  suspected compromise, remove the old public key first (engine calls fail
  until the web is switched). Details: `docs/ENGINE_TOKEN.md`, "Key rotation".
- **Service accounts** (scripts, CI) may use static bearer tokens instead:
  hashed records in `KALOS_AUTH_TOKENS` (`.env.example` has the shape).
  kalos-web never sends one, and the stack does not need any.

## Onboarding

kalos-web is invite-only. Each organization maps to one engine tenant through
`web-config/tenants.json` (`{"<organization id>": "<tenant>"}`), which
kalos-web's `scripts/create-org.mjs` writes and the web re-reads on change.
An organization missing from the map gets 403. The tenant in every engine
token comes from this map, and the engine keys all data by it.

```bash
cd kalos/deploy
# web-config is mounted read-write at a second path for this one-shot job
# (the server's /config mount stays read-only).
docker compose run --rm --no-deps \
  -v "$(pwd)/web-config:/onboard" \
  web node scripts/create-org.mjs --tenant-map /onboard/tenants.json \
  --name "Acme Bio" --slug acme --owner-email owner@acme.example --tenant acme
```

It prints an invitation link; send it to the owner, who signs in with Google
and enrols TOTP. kalos-web's `docs/AUTH.md` covers invitations and roles.

## Upgrade

```bash
cd kalos/deploy
git -C .. pull                       # or however the engine repo updates
git -C ../../kalos-web pull          # kalos-web separately

docker compose build
docker compose run --rm --no-deps -e AUTH_MIGRATE=only web   # web auth schema, if it changed
docker compose up -d                 # recreates only the services whose image changed
docker compose logs -f engine web    # watch both healthchecks go green
```

The `kalos-state` volume is untouched by a rebuild - it is a named volume,
not baked into the image, so an upgrade never loses `experiments.db`,
`portal.db`, or `runner.lock`.

## Rollback

This pack does not push to a registry (out of scope - see "What was
verified"), so rollback here means rebuilding the previous commit's image
under a distinct tag, not pulling one back down:

```bash
cd kalos/deploy
git -C .. checkout <previous-good-sha>
git -C ../../kalos-web checkout <previous-good-sha>   # if it changed too

KALOS_IMAGE_TAG=rollback-$(date +%Y%m%d) docker compose build
KALOS_IMAGE_TAG=rollback-$(date +%Y%m%d) docker compose up -d
```

Data is not rolled back with the code - `kalos-state` is shared across
versions. If the previous version's schema is genuinely incompatible with
what the bad version wrote, restore from a backup (below) instead of just
rolling back the image.

## Backup / restore drill

The backup sidecar (`deploy/backup/`) runs `sqlite3 <db> ".backup <dest>"`
against `experiments.db` and `portal.db` on `BACKUP_INTERVAL_SECONDS`
(default daily), keeping `BACKUP_KEEP_DAYS` (default 7) of history under
`deploy/backups/{experiments,portal}/`. See `deploy/backup/backup.sh`'s
header comment for exactly why `.backup` and not `cp`/`tar` of the live
files: `experiments.db` runs in rollback-journal mode and `portal.db` (schema
v2, docs/MIGRATIONS.md) in WAL mode, and in both a plain file copy can land
mid-write or miss commits still in `-wal`, producing a backup that opens fine
but is silently missing data. `.backup` uses SQLite's Online Backup API, which
is safe against a concurrently writing engine process. The sidecar's mount is
read-only; with the engine stopped (no `-wal`) it falls back to an
`immutable=1` snapshot, and every snapshot must pass `PRAGMA quick_check`.

**Verify a backup is actually happening:**

```bash
docker compose logs backup --tail 20
ls -la deploy/backups/experiments/ deploy/backups/portal/
```

**Restore drill** (run this periodically against a scratch stack, not only
when something is actually on fire - a backup nobody has ever restored from
is a hope, not a backup):

```bash
cd kalos/deploy

# 1. Stop the engine so nothing writes to the live db while restoring.
docker compose stop engine

# 2. Pick the backup to restore and copy it into the live volume, under the
#    live filenames. `docker compose exec` won't work with the engine
#    stopped, so use `docker run` against the same named volume directly.
LATEST_EXP=$(ls -t deploy/backups/experiments/*.db | head -1)
LATEST_CAMPAIGN=$(ls -t deploy/backups/portal/*.db | head -1)
docker run --rm \
  -v kalos-state:/state \
  -v "$(pwd)/backups:/backups:ro" \
  alpine:3.20 sh -c "
    cp /backups/experiments/$(basename "$LATEST_EXP") /state/experiments.db &&
    cp /backups/portal/$(basename "$LATEST_CAMPAIGN") /state/portal.db &&
    chown 10001:10001 /state/experiments.db /state/portal.db
  "

# 3. Bring the engine back up and confirm the restored data is visible.
docker compose start engine
docker compose logs -f engine
docker compose exec engine python -c "import sqlite3; \
  print(sqlite3.connect('/home/kalos/.kalos/portal.db').execute('select count(*) from campaigns').fetchone())"
```

Then sign in to the web app and check that the campaigns are there.

The web's own database (users, sessions, organizations; volume
`kalos-web-data`) is not covered by this sidecar. Back it up with kalos-web's
`scripts/auth-backup.sh` (kalos-web `docs/AUTH.md`).

The `latest/*.json` per-tenant analysis cache is intentionally NOT part of
this drill - it is explicitly best-effort persistence in the source itself
(`kalos/portal/app.py`'s `_save_latest`) and regenerates the moment a tenant
re-runs an analysis, so it does not need restore-grade rigor. The backup
script still copies it opportunistically under
`deploy/backups/latest-cache/` if you want it anyway.

## Log locations

- `docker compose logs engine` - uvicorn errors (it runs at log level
  `warning`) plus kalos's own warnings (`kalos.portal`): startup refusals,
  "auth is OPEN" if no auth source is configured, and the per-request "refused
  a remote request" warning if it ever fires. Per-token refusals
  (`kalos auth: rejected jwt reason=...`) are logged at INFO, so they do not
  appear at this level.
- `docker compose logs web` - Next.js server output.
- `docker compose logs backup` - one line per backup attempt (success with
  the file written and its size, or an explicit `FAILED` line that leaves
  prior backups untouched rather than clobbering them with a partial file).
- Nothing is written to a host log directory by default; add a `logging:`
  driver block per service in `compose.yaml` if centralized log
  shipping is needed later - out of scope for v0.

## Health checks

Both `engine` and `web` carry an image-level `HEALTHCHECK` (see
`deploy/engine/Containerfile` / `kalos-web/Dockerfile` for exactly what each checks and
why). The engine's hits `GET /healthz` (wave-2 fix; an earlier pass of this
deploy pack used `GET /` as a stand-in because the route did not exist yet
- see git history for that reasoning). `/healthz` is unauthenticated by
design, returns a fixed `{"status": "ok"}`, and never touches the database,
so it stays a pure liveness check. `docker compose ps` shows `healthy` /
`unhealthy` / `starting` per service; `web`'s startup is ordered on
`engine` reaching `healthy` via `depends_on: condition: service_healthy`.

## Security checklist

- [ ] `engine-config/engine-jwks.json` holds only the public key(s) of the
      web's signing key, and its kid equals `KALOS_ENGINE_JWT_KID`.
- [ ] `secrets/engine-jwt-ed25519.pem` is mode 600, owned by uid 10001, and
      mounted only into `web`. It is never committed (`deploy/secrets/` is
      gitignored) and never placed in `.env`.
- [ ] `KALOS_AUTH_TOKENS` is empty unless a service account needs it; each
      record has `"service_account": true`, one tenant, and no `admin`.
- [ ] `KALOS_ANON_SALT` is a real random secret, not the placeholder.
- [ ] An unauthenticated engine call gets 401 and `docker compose logs engine`
      has no "auth is OPEN" warning ("Deploy", step 5).
- [ ] kalos-web's boot check passed (`docker compose logs web`): it exits 1 on
      a missing or placeholder `BETTER_AUTH_SECRET`, `BETTER_AUTH_URL`, Google
      client, engine key or kid, or an invalid tenants map.
- [ ] TLS is terminated by a host-level reverse proxy (nginx, Caddy, a cloud
      load balancer) in front of the one published port, and that proxy's
      address is in `AUTH_TRUSTED_PROXIES`. This stack serves plain HTTP.
- [ ] Only ONE port is published: `KALOS_WEB_PORT` (default 3000, on
      `KALOS_WEB_BIND`, default 127.0.0.1). The engine is compose-internal.
      Confirm with `docker compose ps`.
- [ ] Exactly one `engine` replica. The engine lock and the store's
      single-writer assumption are not safe under
      `docker compose up --scale engine=N` for N>1.

## Changing the engine URL

The web reaches the engine at `KALOS_ENGINE_URL` (`http://engine:8050` in
this stack), read at request time: change it in `compose.yaml` and run
`docker compose up -d web`.

Dev only: kalos-web can also be built with `NEXT_PUBLIC_API_URL` so the
browser calls an engine directly. That bypasses sign-in and engine tokens, so
it only works against an open local engine; never use it in this stack.

## What was verified vs. not

**Engine-token setup (this revision):** `docker compose config` resolves the
stack with a filled-in `.env`; a JWKS printed by kalos-web's
`scripts/engine-jwks.mjs` from a fresh Ed25519 key was accepted by the
engine, which then served a token signed with that key and refused an
unsigned request (run outside Docker). No image was built and no container
was run for this revision.

**Earlier revisions:**

**Verified:**
- `deploy/compose.yaml` parses and resolves correctly:
  `docker compose config` (Compose v2, installed in this environment)
  succeeds against a filled-in `.env`, with all three build contexts
  resolving to the expected paths (this repo's root for `engine`, the
  sibling `kalos-web` checkout for `web`, `deploy/backup` for `backup`) and
  all `${VAR}` substitutions resolving as intended.
- Every fact this pack's comments assert about the kalos source was
  confirmed by reading the actual files, not assumed, as of this task:
  `kalos/portal/__main__.py`'s uvicorn entrypoint and `assert_safe_exposure`
  call; `kalos/portal/config.py`'s bind/CORS/salt logic; `kalos/portal/
  auth.py`'s token model; `kalos/portal/app.py`'s registered routes (no
  `/healthz` at the time) and the `_refuse_open_remote_access` middleware;
  `kalos/store/sqlite_store.py` and `kalos/portal/campaign.py`'s
  `KALOS_STATE_DIR` handling (and the `SqliteStore`/`runner.lock` gap where
  it was then ignored). **Both gaps were closed in a later wave**:
  `kalos/portal/app.py` now has `GET /healthz`, and `SqliteStore`/
  `SingletonLock` now read `KALOS_STATE_DIR` too - see the "Health checks"
  section above and this Containerfile's `KALOS_STATE_DIR` comment for the
  current state. `kalos/runner/singleton.py`'s single-writer lock;
  `kalos-web/package.json` and `next.config.ts` (no `output: "standalone"`
  today); `kalos-web/
  lib/api.ts` and its callers (all `"use client"`, no `Authorization`
  header sent anywhere).
- `deploy/backup/backup.sh` runs cleanly under `sh -n` (syntax check) and
  was reasoned through by hand for the failure modes it claims to handle
  (missing db file yet, a failed `.backup` call, pruning). `shellcheck` was
  not available in this environment (neither the binary nor a working
  Docker daemon to run its image) to run a real lint pass - worth doing
  before this script sees production.

**NOT verified (stated plainly, not glossed over):**
- No image was actually built (`docker build`/`docker compose build`) -
  this task's brief explicitly excludes that as heavy; a real build could
  still surface a missed system package, a pip resolution conflict, or (for
  `web`) the `output: "standalone"` gap actually failing the build as
  documented.
- No container was actually run; the healthchecks, the auth-refusal
  startup path, and the backup/restore drill are reasoned from reading the
  source, not exercised end-to-end.
- `deploy/backup/backup.sh` was not executed against a real SQLite file.
