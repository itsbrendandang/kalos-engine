#!/usr/bin/env bash
set -euo pipefail

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo 'Docker Compose v2 or newer is required.' >&2; exit 1; }
docker info >/dev/null
export KALOS_WEB_DIR=${KALOS_WEB_DIR:-"$(dirname "$ROOT")/kalos-web"}
if [[ ! -d "$KALOS_WEB_DIR" ]]; then
    git clone https://github.com/itsbrendandang/kalos-web "$KALOS_WEB_DIR"
fi
KALOS_WEB_DIR=$(CDPATH='' cd -- "$KALOS_WEB_DIR" && pwd)
export KALOS_WEB_DIR
compose=(docker compose --env-file "$ROOT/deploy/.env" -f "$ROOT/deploy/compose.yaml")
engine_only=false
if [[ ${1:-} == --engine-only ]]; then engine_only=true; shift; fi
if [[ $# -gt 0 && ${1:-} != up ]]; then
    exec "${compose[@]}" "$@"
fi
if [[ ${1:-} == up ]]; then shift; fi
if [[ ! -e deploy/.env ]]; then
    mkdir -p deploy/secrets
    docker run --rm -i -v "$ROOT/deploy:/deploy" node:22-alpine node <<'NODE'
const fs = require('fs');
const crypto = require('crypto');
const secret = () => crypto.randomBytes(32).toString('hex');
const token = secret();
const kid = `local-${crypto.randomBytes(8).toString('hex')}`;
const { privateKey, publicKey } = crypto.generateKeyPairSync('ed25519');
fs.writeFileSync('/deploy/secrets/engine-private.pem', privateKey.export({ type: 'pkcs8', format: 'pem' }), { mode: 0o400 });
fs.chownSync('/deploy/secrets/engine-private.pem', 10001, 10001);
fs.writeFileSync('/deploy/secrets/engine-public.jwks', JSON.stringify({ keys: [{ ...publicKey.export({ format: 'jwk' }), kid, use: 'sig', alg: 'EdDSA' }] }));
fs.writeFileSync('/deploy/secrets/tenants.json', '{}', { mode: 0o600 });
fs.chownSync('/deploy/secrets/tenants.json', 10001, 10001);
const env = {
  KALOS_ENGINE_TOKEN: token,
  KALOS_AUTH_TOKENS: JSON.stringify([{ token_sha256: crypto.createHash('sha256').update(token).digest('hex'), subject: 'local-ops', tenant: 'local', scopes: ['read', 'write'], service_account: true }]),
  KALOS_ANON_SALT: secret(), BETTER_AUTH_SECRET: secret(),
  BETTER_AUTH_URL: 'http://localhost:3000', KALOS_PUBLIC_URL: 'http://localhost:3000',
  KALOS_CORS_ORIGINS: 'http://localhost:3000', KALOS_ENGINE_JWT_KID: kid,
  GOOGLE_CLIENT_ID: '', GOOGLE_CLIENT_SECRET: '',
  AUTH_TRUSTED_PROXIES: 'none', AUTH_ACCEPT_NO_PROXY: '1',
  KALOS_TORCH_THREADS: '0', KALOS_ENGINE_CPUS: '0'
};
fs.writeFileSync('/deploy/.env', Object.entries(env).map(([k,v]) => `${k}='${v}'`).join('\n') + '\n', { flag: 'wx', mode: 0o600 });
NODE
fi
# Also repairs ownership if an interrupted first run created the file as root.
docker run --rm -v "$ROOT/deploy:/deploy" node:22-alpine chown "$(id -u):$(id -g)" /deploy/.env
mkdir -p deploy/backups
docker run --rm -v "$ROOT/deploy/backups:/backups" node:22-alpine sh -c 'chown 10001:10001 /backups && chmod 700 /backups'
if ! "$engine_only"; then
    if ! "${compose[@]}" config --format json | docker run --rm -i node:22-alpine node -e '
let s=""; process.stdin.on("data", d => s+=d).on("end", () => {
const e=JSON.parse(s).services.web.environment;
process.exit(e.GOOGLE_CLIENT_ID?.trim() && e.GOOGLE_CLIENT_SECRET?.trim() ? 0 : 1);
});'; then
        echo 'Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in deploy/.env, then rerun.' >&2
        echo 'Register Google redirect URI: http://localhost:3000/api/auth/callback/google' >&2
        echo 'Or run ./scripts/kalos-up.sh --engine-only for engine + backups.' >&2
        exit 1
    fi
fi
services=()
if "$engine_only"; then services=(engine backup); fi
"${compose[@]}" up --build -d --wait --wait-timeout 180 "$@" ${services[@]+"${services[@]}"}
if "$engine_only"; then
    echo 'Engine healthy (Compose network only).'
else
    echo 'Open http://localhost:3000 (Google OAuth credentials required for sign-in).'
    echo 'Known limitation: web data calls fail engine auth; see deploy/RUNBOOK.md.'
fi
