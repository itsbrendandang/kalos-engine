"""Verification of the internal engine token (EdDSA JWT) minted by kalos-web.

Contract: docs/ENGINE_TOKEN.md. kalos-web authenticates the human user with its
session provider and, per engine call, mints a short-lived Ed25519-signed JWT
carrying the user's `sub`, `sid`, `tenant` and `scopes`. The engine is
independent of the session provider: all it knows is the set of public keys it
trusts (a JWKS file) and the claim contract below.

Hard rules, each one a deliberate refusal rather than a best effort:

- **EdDSA only.** The header `alg` must be exactly `EdDSA` and the verifying key
  is always an Ed25519 public key, so `none`, HS256-with-the-public-key-as-secret
  (alg confusion) and RS*/ES* are refused before any cryptography runs, and
  PyJWT is additionally called with `algorithms=["EdDSA"]`.
- **Key by `kid`, never by trial.** The header `kid` selects exactly one key from
  the JWKS file. An absent or unknown `kid` is a rejection; the verifier never
  tries every key it holds.
- **Short-lived.** `exp - iat` may not exceed `KALOS_JWT_MAX_TTL_S` (default
  120 s, configurable up to 300 s), whatever the signature says.
- **Strict claims.** `iss`, `aud` (a single string), `sub`, `sid`, `jti`,
  `tenant`, `scopes`, `iat`, `nbf`, `exp` are all required. `tenant` must match
  the tenant-id pattern (rejected, never sanitised); every entry of `scopes`
  must be a known scope (an unknown scope rejects the token, it is not dropped).
- **Fail closed on configuration.** An unreadable or malformed key file raises
  `JwtConfigError` (503 / startup refusal); stale keys are never served after
  the file goes bad.

This module never logs or returns token material; callers log only the reason
code, `kid` and `jti` (both sanitised by `safe_log_value`).
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from kalos.portal.auth_errors import JwtConfigError
from kalos.portal.config import is_production
from kalos.portal.tenancy import is_valid_tenant_id

log = logging.getLogger("kalos.portal")

# --- configuration ------------------------------------------------------------ #

KEYS_FILE_VAR = "KALOS_JWT_PUBLIC_KEYS_FILE"
ISSUER_VAR = "KALOS_JWT_ISSUER"
AUDIENCE_VAR = "KALOS_JWT_AUDIENCE"
LEEWAY_VAR = "KALOS_JWT_LEEWAY_S"
MAX_TTL_VAR = "KALOS_JWT_MAX_TTL_S"

DEFAULT_ISSUER = "kalos-web"
DEFAULT_AUDIENCE = "kalos-engine"
DEFAULT_LEEWAY_S = 30
MAX_LEEWAY_S = 120
DEFAULT_MAX_TTL_S = 120
MAX_MAX_TTL_S = 300  # hard ceiling on the configurable TTL

ALGORITHM = "EdDSA"
REQUIRED_CLAIMS = ("exp", "iat", "nbf", "iss", "aud", "sub", "tenant", "scopes", "jti", "sid")

# Token scopes (the web's vocabulary) -> engine scopes (kalos.portal.auth).
# No implication: `campaigns:write` maps to `write` only, matching the engine's
# existing semantics where `require_scope` is plain set membership and `write`
# does not imply `read`. The web mints `campaigns:read` alongside
# `campaigns:write` for members.
SCOPE_MAP: Mapping[str, str] = {
    "campaigns:read": "read",
    "campaigns:write": "write",
    "org:admin": "admin",
}

# A compact JWS of this contract is a few hundred bytes; refuse to parse anything
# absurd before base64/JSON decoding it.
MAX_TOKEN_CHARS = 8192
_MAX_ID_CHARS = 256  # sub / sid / jti / kid
_SAFE_LOG_VALUE = re.compile(r"[A-Za-z0-9._:@-]{1,128}")
_B64URL = re.compile(r"[A-Za-z0-9_-]+")
# sub / sid: printable ASCII without space or comma, so every value can be named
# in the comma-separated KALOS_AUTH_DENY_SUBJECTS list and logged verbatim.
_SUBJECT_CHARS = re.compile(r"[\x21-\x2b\x2d-\x7e]+")
# NumericDate bounds: positive and before 2^34 s (year ~2514). Anything else is
# garbage, and huge ints would overflow float conversion.
_MAX_NUMERIC_DATE = 2**34

# The kalos-web shared test vectors (tests/vectors/engine-token.json) publish
# these PUBLIC keys and kids. Their private halves are in the web repo's
# generator output, valid nowhere: production refuses to trust them.
TEST_VECTOR_KID_PREFIX = "kalos-test-vectors-"
TEST_VECTOR_PUBLIC_X = frozenset({
    "cWkcwK8XARmOnPgLAatvFnSuSS7yT2ITxICY16VmG50",
    "jcxO2NevUQ-pQlh4IiQsVWz_mUfmIhwxhXEvyEj9YYA",
})


@dataclass(frozen=True)
class JwtSettings:
    keys_file: Path
    issuer: str
    audience: str
    leeway_s: int
    max_ttl_s: int


def _int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise JwtConfigError(f"{name} must be an integer") from None
    if not lo <= value <= hi:
        raise JwtConfigError(f"{name} must be between {lo} and {hi}")
    return value


def jwt_settings_from_env() -> JwtSettings | None:
    """The JWT verification settings, or None when `KALOS_JWT_PUBLIC_KEYS_FILE`
    is unset (JWT auth disabled). Read at request time, like the static tokens."""
    path = os.environ.get(KEYS_FILE_VAR, "").strip()
    if not path:
        return None
    issuer = os.environ.get(ISSUER_VAR, "").strip() or DEFAULT_ISSUER
    audience = os.environ.get(AUDIENCE_VAR, "").strip() or DEFAULT_AUDIENCE
    return JwtSettings(
        keys_file=Path(path).expanduser(),
        issuer=issuer,
        audience=audience,
        leeway_s=_int_env(LEEWAY_VAR, DEFAULT_LEEWAY_S, lo=0, hi=MAX_LEEWAY_S),
        max_ttl_s=_int_env(MAX_TTL_VAR, DEFAULT_MAX_TTL_S, lo=1, hi=MAX_MAX_TTL_S),
    )


# --- the public-key file (JWKS) ---------------------------------------------- #

def _b64url_decode(value: str) -> bytes:
    if not _B64URL.fullmatch(value):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# --- config-error logging, once per change -------------------------------------- #

_log_lock = threading.Lock()
_last_logged_error: tuple[object, str] | None = None


def log_config_error_once(exc: JwtConfigError) -> None:
    """Log a JWT config error at ERROR, but only when it differs from the last
    one logged (message, plus the key file's stat signature for file errors).
    A broken key file otherwise logs on every request."""
    global _last_logged_error
    key = (getattr(exc, "change_key", None), str(exc))
    with _log_lock:
        if key == _last_logged_error:
            return
        _last_logged_error = key
    log.error("kalos auth: engine-token (JWT) configuration is invalid: %s", exc)


def _clear_logged_error() -> None:
    global _last_logged_error
    with _log_lock:
        _last_logged_error = None


# --- Ed25519 public-key point validation ---------------------------------------- #
# OpenSSL (via `cryptography`) loads ANY 32 bytes as an Ed25519 public key and
# does not reject small-order points at verify time. With the identity point as
# the "public key", the signature R=identity, S=0 verifies for EVERY message, so
# a JWKS entry with such an `x` would let anyone mint tokens. Each key is
# therefore decoded here (RFC 8032 5.1.3) and refused unless it is a canonical
# encoding of a point of large order. Public data only, so plain Python ints
# (not constant time) are fine.

_P = 2**255 - 19
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _ed25519_decode(raw: bytes) -> tuple[int, int]:
    """Decode a 32-byte Ed25519 point to affine (x, y); `ValueError` if it is
    not on the curve or not canonically encoded."""
    if len(raw) != 32:
        raise ValueError("wrong length")
    n = int.from_bytes(raw, "little")
    sign = n >> 255
    y = n & ((1 << 255) - 1)
    if y >= _P:
        raise ValueError("non-canonical y")
    u = (y * y - 1) % _P
    v = (_D * y * y + 1) % _P
    x2 = u * pow(v, _P - 2, _P) % _P
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        raise ValueError("not on the curve")
    if x == 0 and sign == 1:
        raise ValueError("non-canonical x sign")
    if (x & 1) != sign:
        x = _P - x
    return x, y


def _ed25519_add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Affine twisted-Edwards addition (a=-1; complete since d is a non-square)."""
    (x1, y1), (x2, y2) = a, b
    t = _D * x1 * x2 * y1 * y2 % _P
    x3 = (x1 * y2 + y1 * x2) * pow(1 + t, _P - 2, _P) % _P
    y3 = (y1 * y2 + x1 * x2) * pow(1 - t, _P - 2, _P) % _P
    return x3, y3


def _ed25519_has_small_order(point: tuple[int, int]) -> bool:
    """Whether [8]point is the identity (the 8 torsion points, identity included)."""
    q = point
    for _ in range(3):
        q = _ed25519_add(q, q)
    return q == (0, 1)


def validate_ed25519_public_key(raw: bytes) -> Ed25519PublicKey:
    """`raw` as an `Ed25519PublicKey`, or `ValueError` if it is off-curve,
    non-canonical, or of small order (identity, all-zero, and the rest of the
    8-torsion, in any encoding)."""
    point = _ed25519_decode(raw)
    if _ed25519_has_small_order(point):
        raise ValueError("small-order point")
    return Ed25519PublicKey.from_public_bytes(raw)


def parse_jwks(text: str, *, production: bool = False) -> dict[str, Ed25519PublicKey]:
    """Parse a JWKS document into `{kid: Ed25519PublicKey}`.

    Strict: every key must be `kty=OKP`, `crv=Ed25519`, carry a unique `kid` and
    a 32-byte `x` that is a canonical, large-order curve point (see
    `validate_ed25519_public_key`); `alg`/`use`, when present, must be
    `EdDSA`/`sig`. A key with private material (`d`) rejects the whole file - a
    private key in the engine's config is an incident, not something to quietly
    ignore. With `production`, the published test-vector keys and kids are
    refused too. Error messages name the entry index, never key material.
    """
    try:
        data = json.loads(text)
    except ValueError:
        raise JwtConfigError(f"{KEYS_FILE_VAR} is not valid JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("keys"), list):
        raise JwtConfigError(f"{KEYS_FILE_VAR} must be a JWKS object: {{\"keys\": [...]}}")
    keys: dict[str, Ed25519PublicKey] = {}
    for i, jwk in enumerate(data["keys"]):
        if not isinstance(jwk, dict):
            raise JwtConfigError(f"JWKS key {i} is not an object")
        if "d" in jwk:
            raise JwtConfigError(f"JWKS key {i} contains private key material ('d'); refusing")
        if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
            raise JwtConfigError(f"JWKS key {i} must be kty=OKP, crv=Ed25519")
        if "alg" in jwk and jwk["alg"] != ALGORITHM:
            raise JwtConfigError(f"JWKS key {i} has alg other than {ALGORITHM}")
        if "use" in jwk and jwk["use"] != "sig":
            raise JwtConfigError(f"JWKS key {i} has use other than 'sig'")
        kid = jwk.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > _MAX_ID_CHARS:
            raise JwtConfigError(f"JWKS key {i} is missing a valid 'kid'")
        if kid in keys:
            raise JwtConfigError(f"JWKS key {i} duplicates an earlier kid")
        x = jwk.get("x")
        try:
            if not isinstance(x, str):
                raise ValueError("missing x")
            raw = _b64url_decode(x)
            if len(raw) != 32:
                raise ValueError("wrong length")
            if x != _b64url_encode(raw):
                raise ValueError("non-canonical base64url")
            key = validate_ed25519_public_key(raw)
        except (ValueError, binascii.Error):
            raise JwtConfigError(
                f"JWKS key {i} has an invalid Ed25519 'x' (must be a canonical, "
                "large-order curve point)"
            ) from None
        if production and (kid.startswith(TEST_VECTOR_KID_PREFIX) or x in TEST_VECTOR_PUBLIC_X):
            raise JwtConfigError(
                f"JWKS key {i} is a published TEST-VECTOR key; KALOS_ENV=production refuses it"
            )
        keys[kid] = key
    return keys


_Sig = tuple[int, int, int, int, bool] | None


class JwksKeyStore:
    """The trusted public keys, re-read whenever the file changes.

    The file's (mtime_ns, ctime_ns, size, inode) is checked on every lookup (one
    `stat`); the JSON is re-parsed only when that changes (or `KALOS_ENV`
    flips). Rotate by writing the new file and renaming it over the old one
    (atomic, new inode). A file that becomes unreadable or malformed raises
    `JwtConfigError` on every lookup until fixed: the previously loaded keys are
    dropped, never served stale. The error is logged once per change, not once
    per request.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cached: tuple[_Sig, dict[str, Ed25519PublicKey]] | None = None
        self._failed: tuple[_Sig, JwtConfigError] | None = None

    def _fail(self, sig: _Sig, exc: JwtConfigError) -> JwtConfigError:
        exc.change_key = sig  # type: ignore[attr-defined]
        self._cached = None
        self._failed = (sig, exc)
        log_config_error_once(exc)
        return exc

    def keys(self) -> Mapping[str, Ed25519PublicKey]:
        production = is_production()
        with self._lock:
            try:
                st = os.stat(self.path)
            except OSError:
                sig: _Sig = None
            else:
                sig = (st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino, production)
            if sig is not None and self._cached is not None and self._cached[0] == sig:
                return self._cached[1]
            if self._failed is not None and self._failed[0] == sig:
                raise self._failed[1]
            if sig is None:
                raise self._fail(sig, JwtConfigError(f"cannot read {KEYS_FILE_VAR}"))
            try:
                text = self.path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raise self._fail(sig, JwtConfigError(f"cannot read {KEYS_FILE_VAR}")) from None
            try:
                keys = parse_jwks(text, production=production)
            except JwtConfigError as exc:
                raise self._fail(sig, exc) from None
            self._cached = (sig, keys)
            if self._failed is not None:
                self._failed = None
                _clear_logged_error()
            return keys


# --- token verification -------------------------------------------------------- #

class TokenRejected(Exception):
    """A presented engine token is not acceptable. `reason` is a short code for
    server-side logs; the HTTP response never carries it."""

    def __init__(self, reason: str, *, kid: str | None = None, token_id: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.kid = kid
        self.token_id = token_id


@dataclass(frozen=True)
class VerifiedToken:
    subject: str
    tenant: str
    scopes: frozenset[str]  # ENGINE scopes (read/write/admin), already mapped
    session_id: str
    token_id: str
    kid: str


def safe_log_value(value: object) -> str:
    """`value` if it is a short, plain identifier, else a placeholder: `kid`
    and `jti` come from the token, so they must not be able to forge log lines."""
    if isinstance(value, str) and _SAFE_LOG_VALUE.fullmatch(value):
        return value
    return "<unprintable>"


def _numeric_date(claims: Mapping[str, Any], name: str) -> float:
    value = claims.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name)
    try:
        as_float = float(value)
    except (OverflowError, TypeError):
        raise ValueError(name) from None
    if not math.isfinite(as_float) or not 0 < as_float < _MAX_NUMERIC_DATE:
        raise ValueError(name)
    return as_float


def _identifier(claims: Mapping[str, Any], name: str, *, subject_chars: bool = False) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value or len(value) > _MAX_ID_CHARS:
        raise ValueError(name)
    if subject_chars and not _SUBJECT_CHARS.fullmatch(value):
        raise ValueError(name)
    return value


def verify_engine_token(
    token: str,
    *,
    settings: JwtSettings,
    keys: Mapping[str, Ed25519PublicKey],
    now: float,
) -> VerifiedToken:
    """Verify `token` against `keys` at unix time `now`; raise `TokenRejected`.

    Time checks are done here (not by PyJWT) so `now` is injectable: the shared
    test vectors pin a `verify_at` per token. Semantics match PyJWT's: expired
    when `exp <= now - leeway`; not yet valid when `nbf > now + leeway`; also
    refused when `iat > now + leeway`.
    """
    if len(token) > MAX_TOKEN_CHARS:
        raise TokenRejected("too_large")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise TokenRejected("malformed") from None
    kid = header.get("kid")
    if header.get("alg") != ALGORITHM:
        raise TokenRejected("bad_alg", kid=kid if isinstance(kid, str) else None)
    if not isinstance(kid, str) or not kid:
        raise TokenRejected("missing_kid")
    if "typ" in header and header["typ"] != "JWT":
        raise TokenRejected("bad_typ", kid=kid)
    if "crit" in header:
        raise TokenRejected("crit_header", kid=kid)
    key = keys.get(kid)
    if key is None:
        raise TokenRejected("unknown_kid", kid=kid)

    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            key=key,
            algorithms=[ALGORITHM],
            audience=settings.audience,
            options={
                "verify_signature": True,
                "verify_aud": True,
                # iss is compared EXACTLY below, not by PyJWT (2.10.0 matched
                # substrings of a string iss: CVE-2024-53861).
                "verify_iss": False,
                "strict_aud": True,  # aud must be exactly the configured string
                "verify_exp": False,  # time checks below, against the injected clock
                "verify_nbf": False,
                "verify_iat": False,
                "require": list(REQUIRED_CLAIMS),
            },
        )
    except jwt.InvalidSignatureError:
        raise TokenRejected("bad_signature", kid=kid) from None
    except jwt.MissingRequiredClaimError:
        raise TokenRejected("missing_claim", kid=kid) from None
    except jwt.InvalidAudienceError:
        raise TokenRejected("bad_aud", kid=kid) from None
    except jwt.PyJWTError:
        raise TokenRejected("invalid", kid=kid) from None
    except (TypeError, ValueError):
        # malformed claim shapes PyJWT does not wrap (e.g. a list where it
        # expects a string)
        raise TokenRejected("invalid", kid=kid) from None

    iss = claims.get("iss")
    if not isinstance(iss, str) or iss != settings.issuer:
        raise TokenRejected("bad_iss", kid=kid)

    # Signature verified from here on, so jti is the issuer's own value.
    raw_jti = claims.get("jti")
    jti = raw_jti if isinstance(raw_jti, str) else None
    try:
        exp = _numeric_date(claims, "exp")
        iat = _numeric_date(claims, "iat")
        nbf = _numeric_date(claims, "nbf")
    except ValueError:
        raise TokenRejected("bad_time_claim", kid=kid, token_id=jti) from None
    if exp <= iat or nbf > exp:
        raise TokenRejected("bad_time_claim", kid=kid, token_id=jti)
    if exp - iat > settings.max_ttl_s:
        raise TokenRejected("ttl_too_long", kid=kid, token_id=jti)
    leeway = settings.leeway_s
    if exp <= now - leeway:
        raise TokenRejected("expired", kid=kid, token_id=jti)
    if nbf > now + leeway:
        raise TokenRejected("not_yet_valid", kid=kid, token_id=jti)
    if iat > now + leeway:
        raise TokenRejected("issued_in_future", kid=kid, token_id=jti)

    try:
        subject = _identifier(claims, "sub", subject_chars=True)
        session_id = _identifier(claims, "sid", subject_chars=True)
        token_id = _identifier(claims, "jti")
    except ValueError:
        raise TokenRejected("bad_identity_claim", kid=kid, token_id=jti) from None

    tenant = claims.get("tenant")
    if not isinstance(tenant, str) or not is_valid_tenant_id(tenant):
        raise TokenRejected("bad_tenant", kid=kid, token_id=token_id)

    raw_scopes = claims.get("scopes")
    if not isinstance(raw_scopes, list) or not all(
        isinstance(s, str) and s in SCOPE_MAP for s in raw_scopes
    ):
        raise TokenRejected("bad_scopes", kid=kid, token_id=token_id)

    return VerifiedToken(
        subject=subject,
        tenant=tenant,
        scopes=frozenset(SCOPE_MAP[s] for s in raw_scopes),
        session_id=session_id,
        token_id=token_id,
        kid=kid,
    )
