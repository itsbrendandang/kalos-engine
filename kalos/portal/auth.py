"""Kalos portal - in-house authentication + authorization.

The portal is moving from a single-user local tool to a multi-tenant service
(see docs/HARDENING.md, Phase 1). This module is the identity foundation: a
bearer token resolves to a `Principal` (subject, tenant, scopes), and FastAPI
dependencies gate endpoints on scope.

Design (deliberately self-hosted, no external identity provider):

- Tokens are provisioned by the operator and stored as SHA-256 HASHES, never
  plaintext, never logged. At request time the presented token is hashed and
  constant-time compared (`hmac.compare_digest`) against the configured hashes,
  so neither the hash set nor a wrong guess leaks timing about the real token.
- Configuration is read AT REQUEST TIME from `KALOS_AUTH_TOKENS_FILE` (a JSON
  file, preferred) or `KALOS_AUTH_TOKENS` (inline JSON), so tokens can be
  rotated without a process restart and tests can override per-case.
- **Open mode**: when neither is configured the API stays open - every request
  gets an anonymous `Principal` on the `default` tenant with read+write scope
  (today's behavior) - and a warning is logged once at startup. `admin` is never
  granted in open mode. The moment tokens are provisioned, enforcement turns on;
  there is no half-authenticated state.

Authentication is by **static bearer token**: a `secrets.token_urlsafe` value is
hashed and matched against the provisioned records. In `KALOS_ENV=production`
static records are for service accounts only (`service_account: true`, exactly
one tenant, never `admin`).

`KALOS_AUTH_DENY_SUBJECTS` (comma-separated subjects) is the revocation lever: a
match is a 401. There is no revocation DB.

Every rejection answers the same generic 401 body; the token itself is never
logged.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, Header, HTTPException

from kalos.portal.auth_errors import AuthConfigError
from kalos.portal.config import is_production
from kalos.portal.tenancy import (  # noqa: F401 - TENANT_ID_PATTERN/validate_tenant_id re-exported
    RESERVED_ANON_TENANT,
    TENANT_ID_PATTERN,
    is_valid_tenant_id,
    validate_tenant_id,
)

log = logging.getLogger("kalos.portal")

# Subject/session-id validity for static token records. (Previously shared with
# the EdDSA JWT path; that path was removed, so the one helper the static path
# needs lives here now.)
_MAX_ID_CHARS = 256
_SUBJECT_CHARS = re.compile(r"[\x21-\x2b\x2d-\x7e]+")  # printable ASCII, no space or comma


def is_valid_subject(value: object) -> bool:
    """Whether `value` is an acceptable subject / session id: 1-256 printable
    ASCII characters without space or comma."""
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_ID_CHARS
        and _SUBJECT_CHARS.fullmatch(value) is not None
    )


READ = "read"
WRITE = "write"
ADMIN = "admin"
_VALID_SCOPES = frozenset({READ, WRITE, ADMIN})

# The tenant/subject an unauthenticated request is attributed to in open mode.
_ANON_TENANT = RESERVED_ANON_TENANT
_ANON_SUBJECT = "anonymous"
_OPEN_MODE_SCOPES = frozenset({READ, WRITE})  # never ADMIN without a real token

# Auth methods recorded on a Principal.
AUTH_ANONYMOUS = "anonymous"
AUTH_STATIC = "static"


DENY_VAR = "KALOS_AUTH_DENY_SUBJECTS"

# The one body every bad/expired/revoked token gets: no hint of which check failed.
_INVALID_TOKEN_DETAIL = "invalid or expired token"  # noqa: S105 - response detail text, not a credential


@dataclass(frozen=True)
class Principal:
    """The authenticated caller: who they are, which tenant's data they may
    touch, and what they may do. In open mode this is the anonymous principal."""

    subject: str
    tenant: str
    scopes: frozenset[str]
    anonymous: bool = False
    auth_method: str = AUTH_STATIC  # 'anonymous' | 'static'
    session_id: str | None = None
    token_id: str | None = None

    def has(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass
class _TokenRecord:
    token_sha256: str
    subject: str
    tenant: str
    scopes: frozenset[str]
    service_account: bool = False


@dataclass(frozen=True)
class _DenyList:
    subjects: frozenset[str]
    sessions: frozenset[str]

    def denies(self, subject: str, session_id: str | None = None) -> bool:
        return subject in self.subjects or (
            session_id is not None and session_id in self.sessions
        )


def _deny_list() -> _DenyList:
    """Subjects and session ids refused outright (`KALOS_AUTH_DENY_SUBJECTS`).

    Comma-separated entries: `sub:<subject>` denies a subject, `sid:<session>`
    denies one session, and a bare `<value>` (legacy, ambiguous) denies it as
    either. Read per request, but it is a process environment variable: in a
    deployment a change needs a restart (or supervisor reload) to reach the
    process."""
    subjects: set[str] = set()
    sessions: set[str] = set()
    for part in os.environ.get(DENY_VAR, "").split(","):
        entry = part.strip()
        if not entry:
            continue
        if entry.startswith("sub:"):
            subjects.add(entry[4:])
        elif entry.startswith("sid:"):
            sessions.add(entry[4:])
        else:
            subjects.add(entry)
            sessions.add(entry)
    return _DenyList(frozenset(subjects), frozenset(sessions))


@dataclass
class Authenticator:
    """Resolves bearer tokens to `Principal`s from the current configuration.

    Config is re-read on every `principal_for` call so token rotation needs no
    restart; parsing a small JSON token list per request is negligible next to a
    GP fit, and it keeps the class stateless and test-friendly.
    """

    _warned_open: bool = field(default=False, init=False, repr=False)

    def _load_records(self) -> list[_TokenRecord]:
        raw = self._load_raw_config()
        if raw is None:
            return []
        production = is_production()
        records: list[_TokenRecord] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise AuthConfigError(f"token entry {i} is not an object")
            token_hash = entry.get("token_sha256")
            subject = entry.get("subject")
            tenant = entry.get("tenant")
            scopes_in = entry.get("scopes", [READ])
            if not isinstance(token_hash, str) or not token_hash:
                raise AuthConfigError(f"token entry {i} is missing 'token_sha256'")
            if not isinstance(subject, str) or not subject:
                raise AuthConfigError(f"token entry {i} is missing 'subject'")
            if not is_valid_subject(subject):
                # Same alphabet as engine-token sub/sid, so every subject can be
                # named in KALOS_AUTH_DENY_SUBJECTS (comma-separated).
                raise AuthConfigError(
                    f"token entry {i} has an invalid 'subject'; use 1-256 printable ASCII "
                    "characters without spaces or commas"
                )
            if not isinstance(tenant, str) or not tenant:
                raise AuthConfigError(f"token entry {i} is missing 'tenant'")
            if not is_valid_tenant_id(tenant):
                # Fail closed: the whole config is rejected (every request 500s
                # with "authentication is misconfigured") rather than this entry
                # being skipped or its id rewritten. The id itself is not echoed.
                raise AuthConfigError(
                    f"token entry {i} has an invalid 'tenant' id; tenant ids must match "
                    "^[A-Za-z0-9_-]{1,64}$ (letters, digits, '_' and '-', 1-64 chars)"
                )
            if tenant == _ANON_TENANT:
                # Reserved for open mode: a token on 'default' would inherit
                # whatever anonymous callers stored there before auth was on.
                raise AuthConfigError(
                    f"token entry {i} uses the reserved tenant id '{_ANON_TENANT}' "
                    "(open/anonymous mode only); provision a named tenant"
                )
            if not isinstance(scopes_in, list) or not all(s in _VALID_SCOPES for s in scopes_in):
                raise AuthConfigError(
                    f"token entry {i} has invalid scopes; allowed: {sorted(_VALID_SCOPES)}"
                )
            if "tenants" in entry:
                raise AuthConfigError(
                    f"token entry {i} has 'tenants'; a static token carries exactly one 'tenant'"
                )
            service_account = entry.get("service_account", False)
            if not isinstance(service_account, bool):
                raise AuthConfigError(f"token entry {i} has a non-boolean 'service_account'")
            if production:
                # Production: static tokens are long-lived, so they are confined
                # to service accounts with a single tenant and no admin. Refuse
                # the whole config otherwise.
                if not service_account:
                    raise AuthConfigError(
                        f"token entry {i}: KALOS_ENV=production only accepts static tokens "
                        "for service accounts ('service_account': true)"
                    )
                if ADMIN in scopes_in:
                    raise AuthConfigError(
                        f"token entry {i}: KALOS_ENV=production refuses the 'admin' scope "
                        "on static tokens"
                    )
            records.append(
                _TokenRecord(
                    token_sha256=token_hash.lower(),
                    subject=subject,
                    tenant=tenant,
                    scopes=frozenset(scopes_in),
                    service_account=service_account,
                )
            )
        return records

    @staticmethod
    def _load_raw_config() -> list[Any] | None:
        """The provisioned-principal list from file (preferred) or inline env,
        or None when auth is not configured (open mode). Read at request time."""
        path = os.environ.get("KALOS_AUTH_TOKENS_FILE", "").strip()
        if path:
            try:
                text = Path(path).expanduser().read_text()
            except OSError as exc:
                raise AuthConfigError(f"cannot read KALOS_AUTH_TOKENS_FILE: {exc}") from exc
        else:
            text = os.environ.get("KALOS_AUTH_TOKENS", "").strip()
            if not text:
                return None
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise AuthConfigError(f"auth token config is not valid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise AuthConfigError("auth token config must be a JSON array")
        return data

    def has_auth_source(self) -> bool:
        """Whether at least one credential could actually authenticate: a static
        token record. This is what production requires at startup and at runtime
        (see `kalos.portal.app`)."""
        return bool(self._load_records())

    def is_configured(self) -> bool:
        """Whether an auth config is PRESENT. Not the same as whether it enforces.

        See `enforces` - a config that parses to an empty token list is present but
        grants everyone the anonymous principal. Prefer `enforces` for any security
        decision.
        """
        return self._load_raw_config() is not None

    def enforces(self) -> bool:
        """Whether authentication is ACTUALLY enforced: at least one usable token.

        `is_configured` is not a safe proxy for this, and the difference is a real
        misconfiguration rather than a hypothetical. With `KALOS_AUTH_TOKENS=[]` -
        which a config template rendering an empty array, or a tokens file
        containing `[]`, produces easily - `is_configured` returns True while
        `principal_for` falls back to the anonymous read+write principal, because
        it keys off whether any RECORD parsed. The deployment looks locked down and
        is wide open.

        Every security decision (the bind refusal, the remote-request guard, the
        startup posture line) keys off this method so the answer matches what
        `principal_for` will actually do.
        """
        return bool(self._load_records())

    def principal_for(self, authorization: str | None) -> Principal:
        """The `Principal` for a request's `Authorization` header.

        Open mode (no static tokens): the anonymous principal. Configured: a
        `Bearer <token>` mapping to a provisioned record, else 401 with a generic
        body.
        """
        records = self._load_records()
        if not records:
            if is_production():
                # Fail closed: the middleware checked the tokens a moment ago, but
                # they are re-read here, so a file emptied in between must not
                # yield the anonymous read+write principal in production.
                log.error("kalos auth: KALOS_ENV=production but no tokens loaded; refusing")
                raise HTTPException(status_code=503, detail="authentication is not configured")
            if not self._warned_open:
                log.warning(
                    "kalos auth: no tokens configured - API is OPEN (anonymous '%s' tenant, "
                    "read+write). Set KALOS_AUTH_TOKENS[_FILE] to enforce authentication.",
                    _ANON_TENANT,
                )
                self._warned_open = True
            return Principal(
                subject=_ANON_SUBJECT,
                tenant=_ANON_TENANT,
                scopes=_OPEN_MODE_SCOPES,
                anonymous=True,
                auth_method=AUTH_ANONYMOUS,
            )

        token = _bearer(authorization)
        if token is None:
            raise HTTPException(status_code=401, detail="authentication required")
        return self._principal_from_static(token, records)

    @staticmethod
    def _principal_from_static(token: str, records: list[_TokenRecord]) -> Principal:
        presented = hashlib.sha256(token.encode("utf-8")).hexdigest()
        match: _TokenRecord | None = None
        for rec in records:
            # constant-time compare so a wrong token can't be recovered byte-by-byte;
            # no early exit, so the loop length does not reveal the match position
            if hmac.compare_digest(presented, rec.token_sha256) and match is None:
                match = rec
        if match is None:
            log.info("kalos auth: rejected static token reason=no_match")
            raise _invalid_token()
        if _deny_list().denies(match.subject):
            log.info("kalos auth: rejected static token reason=denied")
            raise _invalid_token()
        return Principal(
            subject=match.subject, tenant=match.tenant, scopes=match.scopes, auth_method=AUTH_STATIC
        )


def _invalid_token() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=_INVALID_TOKEN_DETAIL,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _bearer(authorization: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header, or None."""
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    token = authorization[len(prefix):].strip()
    return token or None


# --- module singleton + FastAPI dependencies -------------------------------- #

_AUTH: Authenticator | None = None


def get_authenticator() -> Authenticator:
    """The module-level `Authenticator`. Tests override via
    `app.dependency_overrides[get_authenticator]`."""
    global _AUTH
    if _AUTH is None:
        _AUTH = Authenticator()
    return _AUTH


def require_principal(
    authorization: str | None = Header(default=None),
    auth: Authenticator = Depends(get_authenticator),
) -> Principal:
    """FastAPI dependency: the authenticated `Principal` for this request
    (anonymous in open mode, else resolved from the bearer token or 401)."""
    try:
        return auth.principal_for(authorization)
    except AuthConfigError:
        log.exception("kalos auth: token configuration is invalid")
        raise HTTPException(status_code=500, detail="authentication is misconfigured") from None


def require_scope(scope: str):
    """Build a dependency that requires the caller's principal to hold `scope`
    (403 otherwise). Use as `Depends(require_scope(WRITE))` on an endpoint."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"unknown scope {scope!r}")

    def _dep(principal: Principal = Depends(require_principal)) -> Principal:
        if not principal.has(scope):
            raise HTTPException(status_code=403, detail=f"requires '{scope}' scope")
        return principal

    return _dep
