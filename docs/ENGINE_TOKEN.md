# Engine token: the kalos-web -> kalos-engine request contract

kalos-web authenticates users through its session provider. For each call it makes to the engine, it mints a short-lived **engine token**: a JWT signed with Ed25519 (EdDSA) that names the user, their session, their tenant and their scopes.
The engine checks that token against a set of public keys it trusts. It does not know or care which session provider the web uses.

Verifier: `kalos/portal/jwt_auth.py`, wired into `kalos/portal/auth.py`.
Tests: `tests/test_jwt_auth.py`, including the shared vectors in `tests/vectors/engine-token.json` (owned by kalos-web, `__tests__/vectors/engine-token.json`; regenerate there and copy here).

## Transport

```
Authorization: Bearer <compact JWS>
```

A bearer token with **exactly two dots** takes the JWT branch. Any other bearer token is treated as a static token (a `secrets.token_urlsafe` value contains no dots) and matched against the hashed records in `KALOS_AUTH_TOKENS[_FILE]`.
A two-dot token is never tried as a static token, so a static token must not contain exactly two dots.

## Header

| Field | Requirement |
| --- | --- |
| `alg` | Must be exactly `EdDSA`. `none`, `HS*`, `RS*`, `ES*`, `PS*` and anything else are refused before any signature check. |
| `kid` | Required, non-empty string. It picks exactly one key from the JWKS file. If the `kid` is missing or unknown, the token is refused. The engine never tries all of its keys. |
| `typ` | Optional. If present, it must be `JWT`. |
| `crit` | Must be absent. |

`jku`, `x5u`, `jwk` and other key-locating headers are ignored. Keys come only from the engine's own JWKS file.

## Claims

All of these are required. If any one is missing (or `null`), the token is refused.

| Claim | Type | Rule |
| --- | --- | --- |
| `iss` | string | Must equal `KALOS_JWT_ISSUER` (default `kalos-web`). |
| `aud` | string | Must equal `KALOS_JWT_AUDIENCE` (default `kalos-engine`). It must be a single string. An array is refused, even if it contains the right audience. |
| `sub` | string | The user id. 1-256 characters. Becomes `Principal.subject`. |
| `sid` | string | The web session id. 1-256 characters. Becomes `Principal.session_id`. |
| `jti` | string | A unique token id. 1-256 characters. Becomes `Principal.token_id`, and it is the only token-derived identifier that is logged. |
| `tenant` | string | Must fully match `^[A-Za-z0-9_-]{1,64}$` (`kalos/portal/tenancy.py`). A bad value is **refused, never sanitised**. Case is preserved (`ACME` and `acme` are different tenants). Becomes `Principal.tenant`, which keys every store read and write. |
| `scopes` | array of strings | Every entry must be one of the scopes below. One unknown entry refuses the whole token. The engine does not filter it out and continue. An empty array is accepted, but that principal gets 403 on every gated route. |
| `iat` | NumericDate | Must not be in the future by more than the leeway. |
| `nbf` | NumericDate | Must not be in the future by more than the leeway. |
| `exp` | NumericDate | Must be after `iat`. The token is expired when `exp <= now - leeway`. `exp - iat` must be no more than `KALOS_JWT_MAX_TTL_S`, even when the signature is valid. |

Other claims are ignored.

## Scope mapping

| Token scope | Engine scope | Grants |
| --- | --- | --- |
| `campaigns:read` | `read` | GET routes (`/api/latest`, `/api/campaign`, ...) |
| `campaigns:write` | `write` | Mutating routes (`/api/run`, `/api/campaign/*`, ...) |
| `org:admin` | `admin` | Reserved. No engine route requires it today. |

Scopes **do not imply each other**. The engine's `require_scope` is a plain membership check, and static tokens have always listed `read` and `write` separately. So `campaigns:write` alone does not grant `read`, and `org:admin` alone grants neither. The web mints `["campaigns:read", "campaigns:write"]` for members, and all three scopes for admins.

## Configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `KALOS_JWT_PUBLIC_KEYS_FILE` | Path to the JWKS file of trusted public keys. Setting it turns the JWT branch on and turns open mode off. | unset (JWT disabled) |
| `KALOS_JWT_ISSUER` | Required `iss`. | `kalos-web` |
| `KALOS_JWT_AUDIENCE` | Required `aud`. | `kalos-engine` |
| `KALOS_JWT_LEEWAY_S` | Clock-skew allowance for `exp`/`nbf`/`iat`, in seconds, from 0 to 120. | `30` |
| `KALOS_JWT_MAX_TTL_S` | Longest allowed `exp - iat`, in seconds, from 1 to **300** (hard ceiling). | `120` |
| `KALOS_AUTH_DENY_SUBJECTS` | Comma-separated subjects and/or session ids to refuse (401). Applies to static-token subjects too. Read on every request. | unset |

If a knob is out of range, or the JWKS file is unreadable or malformed, auth **fails closed**:

- Every authenticated request gets `503 {"detail": "authentication unavailable"}`.
- In `KALOS_ENV=production`, the engine refuses to start: importing `kalos.portal.app` raises `ProductionConfigError`, and `python -m kalos.portal` exits 2.
- Previously loaded keys are not kept after the file goes bad.

### JWKS file

```json
{
  "keys": [
    {"kty": "OKP", "crv": "Ed25519", "kid": "kalos-web-2026-10", "x": "<base64url 32-byte public key>", "alg": "EdDSA", "use": "sig"}
  ]
}
```

Rules for every key:

- `kty` must be `OKP`, and `crv` must be `Ed25519`.
- `kid` is required and must be unique within the file.
- `x` must decode to 32 bytes.
- `alg` and `use` are optional. If present, they must be `EdDSA` and `sig`.
- A key that contains `d` (private key material) makes the engine reject the whole file. Only public keys ever go to the engine.

On every request the engine `stat`s the file. It re-parses only when the mtime, size or inode changes. To update the file, write a new copy and `rename` it over the old one. The rename is atomic and gives the file a new inode, so the change is always noticed.

## Production rules (`KALOS_ENV=production`)

- **At startup, at least one usable auth source is required:** a JWKS with at least one key, or at least one service-account static token. The same check runs on every request. If both sources become empty at runtime, every request except the probes gets 503.
- **Static tokens are for service accounts only.** Every static record must carry `"service_account": true`, must not hold `admin`, and carries exactly one `tenant` (a `tenants` list is refused in every environment). If any record breaks these rules, the whole token config is refused at load.
- Users reach the engine only with engine tokens.

## Errors and logging

- Every refused token gets the same response: `401 {"detail": "invalid or expired token"}` with `WWW-Authenticate: Bearer`. A missing header gets `401 "authentication required"`. A token that is valid but lacks the scope gets `403`.
- The server logs one line per refusal: `kalos auth: rejected jwt reason=<code> kid=<kid> jti=<jti>`. `kid` and `jti` are printed only if they are short, plain identifiers. Otherwise they are logged as `<unprintable>`, so a token cannot forge log lines. `jti` is logged only after the signature has verified. The token itself is never logged.
- Reason codes: `too_large`, `malformed`, `bad_alg`, `missing_kid`, `bad_typ`, `crit_header`, `unknown_kid`, `bad_signature`, `missing_claim`, `bad_aud`, `bad_iss`, `invalid`, `bad_time_claim`, `ttl_too_long`, `expired`, `not_yet_valid`, `issued_in_future`, `bad_identity_claim`, `bad_tenant`, `bad_scopes`, `denied`, `jwt_not_configured`.

## Revocation

The engine has no revocation database. Instead:

- **Short lifetimes are the main control.** Tokens live for at most 120 s by default (300 s at most), plus the leeway.
- **Deny-list.** Put a user's `sub` or a session's `sid` in `KALOS_AUTH_DENY_SUBJECTS` to refuse them right away. The engine reads the variable on every request, but a changed environment variable usually needs a process restart (or a config reload by the supervisor). Revoke the session at the web as well, so that no new tokens are minted.

## Key rotation

Planned rotation, with no downtime:

1. Generate a new Ed25519 key pair in the web's secret store, with a new `kid`.
2. **Add** the new public key to the engine's JWKS file, next to the old one. Write it atomically (write, then rename). The engine picks it up on the next request.
3. **Switch the signer** at the web, so it mints with the new private key and `kid`.
4. **Wait at least `KALOS_JWT_MAX_TTL_S` + `KALOS_JWT_LEEWAY_S`** (150 s with the defaults). After that, no token signed with the old key can still be valid.
5. **Remove** the old public key from the JWKS file. Destroy the old private key.

Compromised signing key:

1. **Remove its public key from the JWKS file immediately.** Every token signed with it stops working on the next request. There is no grace period.
2. Add a new public key and switch the signer (steps 1-3 above). Engine calls fail with 401 until the web signs with the new key.
3. If needed, deny-list the affected subjects or sessions, and investigate the web side.

## Not covered

- `jti` replay detection: there is no replay cache. A captured token can be replayed until it expires, which is at most TTL + leeway. TLS and short TTLs are the controls.
- Binding a token to a specific request (method or path claims): not implemented.
