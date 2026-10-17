"""Engine-token (EdDSA JWT) verification: kalos/portal/jwt_auth.py and the JWT
branch of kalos/portal/auth.py. Contract: docs/ENGINE_TOKEN.md.

Every key here is generated at test time with `cryptography`; nothing secret is
committed. The shared vectors in tests/vectors/engine-token.json carry PUBLIC
keys and pre-signed tokens only (copied from kalos-web, which owns them).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from kalos.portal.auth import (  # noqa: E402
    ADMIN,
    READ,
    WRITE,
    AuthConfigError,
    Authenticator,
    require_principal,
)
from kalos.portal.auth_errors import JwtConfigError  # noqa: E402
from kalos.portal.jwt_auth import (  # noqa: E402
    REQUIRED_CLAIMS,
    SCOPE_MAP,
    parse_jwks,
)
from kalos.portal.tenancy import TENANT_ID_PATTERN  # noqa: E402

NOW = 1_800_000_000  # fixed verification clock for in-process tests
_ENV_VARS = (
    "KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_JWT_PUBLIC_KEYS_FILE",
    "KALOS_JWT_ISSUER", "KALOS_JWT_AUDIENCE", "KALOS_JWT_LEEWAY_S", "KALOS_JWT_MAX_TTL_S",
    "KALOS_AUTH_DENY_SUBJECTS", "KALOS_ENV", "KALOS_ALLOW_OPEN_ACCESS",
)
_DROP = object()


# --- helpers ------------------------------------------------------------------ #

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwk(priv: Ed25519PrivateKey, kid: str) -> dict:
    raw = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return {"kty": "OKP", "crv": "Ed25519", "x": _b64url(raw), "kid": kid,
            "alg": "EdDSA", "use": "sig"}


def _write_jwks(path: Path, keys: dict[str, Ed25519PrivateKey]) -> None:
    path.write_text(json.dumps({"keys": [_jwk(p, kid) for kid, p in keys.items()]}))


def _claims(**overrides) -> dict:
    claims = {
        "iss": "kalos-web", "aud": "kalos-engine", "sub": "user_1", "tenant": "acme-bio",
        "scopes": ["campaigns:read", "campaigns:write"], "jti": "jti-1", "sid": "sess-1",
        "iat": NOW, "nbf": NOW, "exp": NOW + 60,
    }
    for k, v in overrides.items():
        if v is _DROP:
            claims.pop(k, None)
        else:
            claims[k] = v
    return claims


def _sign(priv, claims: dict | None = None, *, kid: str | None = "k1",
          alg: str = "EdDSA", headers: dict | None = None) -> str:
    hdrs = dict(headers or {})
    if kid is not None:
        hdrs["kid"] = kid
    return jwt.encode(claims if claims is not None else _claims(), priv, algorithm=alg,
                      headers=hdrs)


def _sign_raw(priv: Ed25519PrivateKey, claims: dict, kid: str = "k1") -> str:
    """Sign without PyJWT's encode-side claim checks (it refuses a non-string iss)."""
    header = {"alg": "EdDSA", "typ": "JWT", "kid": kid}
    signing_input = (_b64url(json.dumps(header).encode()) + "."
                     + _b64url(json.dumps(claims).encode()))
    return signing_input + "." + _b64url(priv.sign(signing_input.encode()))


def _unsigned(header: dict, claims: dict, signature: bytes = b"") -> str:
    signing_input = (_b64url(json.dumps(header).encode()) + "."
                     + _b64url(json.dumps(claims).encode()))
    return signing_input + "." + _b64url(signature)


def _status(auth: Authenticator, token: str | None) -> int:
    try:
        auth.principal_for(None if token is None else f"Bearer {token}")
    except HTTPException as exc:
        return exc.status_code
    return 200


def _reason(auth: Authenticator, token: str, caplog) -> str:
    """Assert `token` is a generic 401 and return the logged reason code, so a
    test fails if a token is refused for the wrong reason."""
    caplog.clear()
    caplog.set_level(logging.INFO, logger="kalos.portal")
    with pytest.raises(HTTPException) as exc:
        auth.principal_for(f"Bearer {token}")
    assert (exc.value.status_code, exc.value.detail) == (401, "invalid or expired token")
    reasons = [r.getMessage().split("reason=")[1].split()[0]
               for r in caplog.records if "reason=" in r.getMessage()]
    assert len(reasons) == 1, reasons
    return reasons[0]


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@pytest.fixture
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def keys(tmp_path, clean_env):
    """A JWKS file with one key `k1`, wired into the env. Returns a namespace."""
    k1 = Ed25519PrivateKey.generate()
    path = tmp_path / "jwks.json"
    _write_jwks(path, {"k1": k1})
    clean_env.setenv("KALOS_JWT_PUBLIC_KEYS_FILE", str(path))

    class NS:
        pass

    ns = NS()
    ns.k1 = k1
    ns.path = path
    ns.auth = Authenticator(clock=lambda: NOW)
    return ns


# --- the happy path ------------------------------------------------------------ #

def test_valid_token_resolves_to_a_jwt_principal(keys):
    p = keys.auth.principal_for(f"Bearer {_sign(keys.k1)}")
    assert p.auth_method == "jwt"
    assert p.anonymous is False
    assert p.subject == "user_1"
    assert p.tenant == "acme-bio"
    assert p.session_id == "sess-1"
    assert p.token_id == "jti-1"
    assert p.scopes == frozenset({READ, WRITE})


@pytest.mark.parametrize("scopes,expected", [
    (["campaigns:read"], {READ}),
    (["campaigns:write"], {WRITE}),  # no implication: write does not grant read
    (["org:admin"], {ADMIN}),
    (["campaigns:read", "campaigns:write", "org:admin"], {READ, WRITE, ADMIN}),
    ([], set()),
])
def test_scope_mapping(keys, scopes, expected):
    p = keys.auth.principal_for(f"Bearer {_sign(keys.k1, _claims(scopes=scopes))}")
    assert p.scopes == frozenset(expected)


# --- signature / key / algorithm attacks --------------------------------------- #

def test_wrong_key_is_rejected(keys, caplog):
    assert _reason(keys.auth, _sign(Ed25519PrivateKey.generate()), caplog) == "bad_signature"


def test_unknown_kid_is_rejected(keys, caplog):
    assert _reason(keys.auth, _sign(keys.k1, kid="nope"), caplog) == "unknown_kid"


def test_missing_kid_is_rejected(keys, caplog):
    assert _reason(keys.auth, _sign(keys.k1, kid=None), caplog) == "missing_kid"


def test_key_is_selected_by_kid_never_by_trial(keys):
    """k2 is trusted, but a k2 signature under kid=k1 must not verify."""
    k2 = Ed25519PrivateKey.generate()
    _write_jwks(keys.path, {"k1": keys.k1, "k2": k2})
    assert _status(keys.auth, _sign(k2, kid="k2")) == 200
    assert _status(keys.auth, _sign(k2, kid="k1")) == 401


@pytest.mark.parametrize("alg", ["none", "None", "NONE"])
def test_alg_none_is_rejected(keys, alg, caplog):
    token = _unsigned({"alg": alg, "typ": "JWT", "kid": "k1"}, _claims())
    assert token.count(".") == 2
    assert _reason(keys.auth, token, caplog) == "bad_alg"


@pytest.mark.parametrize("secret_kind", ["raw", "pem", "jwk_x"])
def test_hs256_alg_confusion_with_public_key_as_secret_is_rejected(keys, secret_kind, caplog):
    pub = keys.k1.public_key()
    secret = {
        "raw": pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
        "pem": pub.public_bytes(serialization.Encoding.PEM,
                                serialization.PublicFormat.SubjectPublicKeyInfo),
        "jwk_x": _jwk(keys.k1, "k1")["x"].encode(),
    }[secret_kind]
    header = {"alg": "HS256", "typ": "JWT", "kid": "k1"}
    signing_input = (_b64url(json.dumps(header).encode()) + "."
                     + _b64url(json.dumps(_claims()).encode()))
    sig = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    assert _reason(keys.auth, signing_input + "." + _b64url(sig), caplog) == "bad_alg"


def test_rs256_is_rejected(keys, caplog):
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert _reason(keys.auth, _sign(rsa_key, alg="RS256"), caplog) == "bad_alg"


def test_es256_is_rejected(keys, caplog):
    ec_key = ec.generate_private_key(ec.SECP256R1())
    assert _reason(keys.auth, _sign(ec_key, alg="ES256"), caplog) == "bad_alg"


def test_tampered_payload_is_rejected(keys):
    header, _payload, sig = _sign(keys.k1).split(".")
    forged = _b64url(json.dumps(_claims(tenant="globex")).encode())
    assert _status(keys.auth, f"{header}.{forged}.{sig}") == 401


@pytest.mark.parametrize("token", ["a.b.c", "..", "x..", "eyJhbGciOiJFZERTQSJ9.@@@.###"])
def test_garbage_two_dot_tokens_are_401_not_500(keys, token):
    assert _status(keys.auth, token) == 401


def test_oversized_token_is_rejected(keys):
    token = _sign(keys.k1, _claims(pad="x" * 9000))
    assert _status(keys.auth, token) == 401


def test_crit_header_is_rejected(keys):
    assert _status(keys.auth, _sign(keys.k1, headers={"crit": ["exp"]})) == 401


def test_non_jwt_typ_is_rejected(keys):
    assert _status(keys.auth, _sign(keys.k1, headers={"typ": "at+jwt"})) == 401


# --- time ----------------------------------------------------------------------- #

def test_expired_is_rejected_beyond_leeway(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(iat=NOW - 91, nbf=NOW - 91,
                                                     exp=NOW - 31))) == 401


def test_expired_within_leeway_is_accepted(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(iat=NOW - 89, nbf=NOW - 89,
                                                     exp=NOW - 29))) == 200


def test_nbf_in_the_future_is_rejected(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(nbf=NOW + 31))) == 401
    assert _status(keys.auth, _sign(keys.k1, _claims(nbf=NOW + 29))) == 200


def test_iat_in_the_future_is_rejected(keys):
    claims = _claims(iat=NOW + 31, nbf=NOW, exp=NOW + 90)
    assert _status(keys.auth, _sign(keys.k1, claims)) == 401


def test_leeway_is_configurable(keys, clean_env):
    clean_env.setenv("KALOS_JWT_LEEWAY_S", "0")
    assert _status(keys.auth, _sign(keys.k1, _claims(iat=NOW - 61, nbf=NOW - 61,
                                                     exp=NOW - 1))) == 401


@pytest.mark.parametrize("bad", ["soon", True, None, float("inf")])
def test_non_numeric_time_claims_are_rejected(keys, bad):
    # json.dumps writes inf as Infinity; PyJWT's json parser rejects or keeps it - either way 401
    assert _status(keys.auth, _sign(keys.k1, _claims(exp=bad))) == 401


def test_ttl_over_max_is_rejected_even_when_signed(keys, caplog):
    assert _reason(keys.auth, _sign(keys.k1, _claims(exp=NOW + 121)), caplog) == "ttl_too_long"
    assert _status(keys.auth, _sign(keys.k1, _claims(exp=NOW + 120))) == 200


def test_exp_not_after_iat_is_rejected(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(exp=NOW))) == 401


def test_nbf_after_exp_is_rejected(keys, caplog):
    token = _sign(keys.k1, _claims(nbf=NOW + 61, exp=NOW + 60))
    assert _reason(keys.auth, token, caplog) == "bad_time_claim"


@pytest.mark.parametrize("claim", ["exp", "iat", "nbf"])
@pytest.mark.parametrize("bad", [10**400, -5, 0, 2**34, 1e300])
def test_out_of_range_numeric_dates_are_rejected(keys, caplog, claim, bad):
    assert _reason(keys.auth, _sign(keys.k1, _claims(**{claim: bad})), caplog) == "bad_time_claim"


def test_max_ttl_is_configurable_up_to_300(keys, clean_env):
    clean_env.setenv("KALOS_JWT_MAX_TTL_S", "300")
    assert _status(keys.auth, _sign(keys.k1, _claims(exp=NOW + 300))) == 200
    assert _status(keys.auth, _sign(keys.k1, _claims(exp=NOW + 301))) == 401


@pytest.mark.parametrize("var,value", [
    ("KALOS_JWT_MAX_TTL_S", "301"), ("KALOS_JWT_MAX_TTL_S", "0"), ("KALOS_JWT_MAX_TTL_S", "2m"),
    ("KALOS_JWT_LEEWAY_S", "-1"), ("KALOS_JWT_LEEWAY_S", "121"),
])
def test_out_of_range_knobs_fail_closed(keys, clean_env, var, value):
    clean_env.setenv(var, value)
    with pytest.raises(JwtConfigError):
        keys.auth.principal_for(f"Bearer {_sign(keys.k1)}")


# --- issuer / audience / claims ------------------------------------------------- #

def test_wrong_audience_is_rejected(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(aud="someone-else"))) == 401


def test_audience_must_be_a_single_string(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(aud=["kalos-engine"]))) == 401


@pytest.mark.parametrize("iss", [
    "evil", "kalos", "kalos-we", "alos-web", "kalos-web ", "KALOS-WEB", "",  # substrings etc.
    ["kalos-web"], ["k", "a"], {"kalos-web": 1}, 1,
])
def test_wrong_issuer_is_rejected_exactly(keys, caplog, iss):
    assert _reason(keys.auth, _sign_raw(keys.k1, _claims(iss=iss)), caplog) in {"bad_iss", "invalid"}


def test_string_issuers_are_compared_exactly(keys, caplog):
    # CVE-2024-53861 (PyJWT 2.10.0): "kalos" in "kalos-web" passed. Ours is exact.
    assert _reason(keys.auth, _sign(keys.k1, _claims(iss="kalos")), caplog) == "bad_iss"


def test_issuer_and_audience_are_configurable(keys, clean_env):
    clean_env.setenv("KALOS_JWT_ISSUER", "web-2")
    clean_env.setenv("KALOS_JWT_AUDIENCE", "engine-2")
    assert _status(keys.auth, _sign(keys.k1)) == 401
    assert _status(keys.auth, _sign(keys.k1, _claims(iss="web-2", aud="engine-2"))) == 200


@pytest.mark.parametrize("claim", REQUIRED_CLAIMS)
def test_missing_required_claim_is_rejected(keys, claim, caplog):
    assert _reason(keys.auth, _sign(keys.k1, _claims(**{claim: _DROP})), caplog) == "missing_claim"


@pytest.mark.parametrize("claim", ["sub", "sid", "jti"])
@pytest.mark.parametrize("bad", ["", 123, ["x"], "x" * 257])
def test_identity_claims_must_be_bounded_strings(keys, claim, bad):
    assert _status(keys.auth, _sign(keys.k1, _claims(**{claim: bad}))) == 401


@pytest.mark.parametrize("claim", ["sub", "sid"])
@pytest.mark.parametrize("bad", ["user 1", "a,b", "ü", "tab\there", "nl\n", "\x7f"])
def test_sub_and_sid_are_printable_ascii_without_comma(keys, caplog, claim, bad):
    token = _sign(keys.k1, _claims(**{claim: bad}))
    assert _reason(keys.auth, token, caplog) == "bad_identity_claim"


def test_sub_may_use_the_rest_of_printable_ascii(keys):
    p = keys.auth.principal_for(f"Bearer {_sign(keys.k1, _claims(sub='u|ser@x.com!~'))}")
    assert p.subject == "u|ser@x.com!~"


@pytest.mark.parametrize("tenant", [
    "", "a" * 65, "../x", "a/b", "acme\n", "ácme", "acme bio", "acme.bio", 123, ["acme"],
])
def test_bad_tenants_are_rejected_not_sanitised(keys, tenant, caplog):
    assert _reason(keys.auth, _sign(keys.k1, _claims(tenant=tenant)), caplog) == "bad_tenant"


@pytest.mark.parametrize("tenant", ["ACME", "Acme_Bio-2", "a" * 64, "x"])
def test_good_tenants_pass_through_unchanged(keys, tenant):
    p = keys.auth.principal_for(f"Bearer {_sign(keys.k1, _claims(tenant=tenant))}")
    assert p.tenant == tenant


@pytest.mark.parametrize("scopes", [
    ["campaigns:read", "campaigns:delete"],  # unknown scope rejects, it is not filtered
    ["read"],  # engine scope names are not token scopes
    ["admin"],
    "campaigns:read",  # not a list
    [["campaigns:read"]],
    [1],
    {"campaigns:read": True},
])
def test_bad_scopes_reject_the_token(keys, scopes, caplog):
    assert _reason(keys.auth, _sign(keys.k1, _claims(scopes=scopes)), caplog) == "bad_scopes"


# --- deny-list ----------------------------------------------------------------- #

def test_deny_list_by_subject_and_by_session(keys, clean_env):
    token = _sign(keys.k1)
    clean_env.setenv("KALOS_AUTH_DENY_SUBJECTS", "someone-else, sess-9")
    assert _status(keys.auth, token) == 200
    clean_env.setenv("KALOS_AUTH_DENY_SUBJECTS", "someone-else, user_1")
    assert _status(keys.auth, token) == 401
    clean_env.setenv("KALOS_AUTH_DENY_SUBJECTS", "sess-1")
    assert _status(keys.auth, token) == 401


@pytest.mark.parametrize("entries,denied", [
    ("sub:user_1", True),
    ("sid:sess-1", True),
    ("sid:user_1", False),  # prefixed entries only match their own kind
    ("sub:sess-1", False),
    ("user_1", True),  # bare entries (legacy) match either
    ("sess-1", True),
    (" sub:other , sid:other ", False),
])
def test_deny_list_prefixes(keys, clean_env, entries, denied):
    clean_env.setenv("KALOS_AUTH_DENY_SUBJECTS", entries)
    assert _status(keys.auth, _sign(keys.k1)) == (401 if denied else 200)


def test_deny_list_applies_to_static_tokens(clean_env):
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("svc-tok"), "subject": "svc", "tenant": "acme", "scopes": [READ]},
    ]))
    auth = Authenticator()
    assert _status(auth, "svc-tok") == 200
    clean_env.setenv("KALOS_AUTH_DENY_SUBJECTS", "svc")
    assert _status(auth, "svc-tok") == 401


# --- key rotation and the JWKS file ---------------------------------------------- #

def _bump_mtime(path: Path) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_key_rotation_with_two_keys(keys):
    k2 = Ed25519PrivateKey.generate()
    # 1. publish the new key alongside the old one: both verify
    _write_jwks(keys.path, {"k1": keys.k1, "k2": k2})
    _bump_mtime(keys.path)
    assert _status(keys.auth, _sign(keys.k1, kid="k1")) == 200
    assert _status(keys.auth, _sign(k2, kid="k2")) == 200
    # 2. retire the old key: its tokens stop verifying, the new one keeps working
    _write_jwks(keys.path, {"k2": k2})
    _bump_mtime(keys.path)
    assert _status(keys.auth, _sign(keys.k1, kid="k1")) == 401
    assert _status(keys.auth, _sign(k2, kid="k2")) == 200


def test_jwks_file_is_reloaded_when_mtime_changes(keys):
    token = _sign(keys.k1)
    assert _status(keys.auth, token) == 200
    k3 = Ed25519PrivateKey.generate()
    keys.path.write_text(json.dumps({"keys": [_jwk(k3, "k1")]}))  # same kid, new key
    _bump_mtime(keys.path)
    assert _status(keys.auth, token) == 401
    assert _status(keys.auth, _sign(k3)) == 200


def test_jwks_file_is_not_reparsed_when_unchanged(keys, monkeypatch):
    assert _status(keys.auth, _sign(keys.k1)) == 200
    calls = []
    import kalos.portal.jwt_auth as jwt_auth

    real = jwt_auth.parse_jwks
    monkeypatch.setattr(jwt_auth, "parse_jwks", lambda text: calls.append(1) or real(text))
    for _ in range(3):
        assert _status(keys.auth, _sign(keys.k1)) == 200
    assert calls == []


_K = _jwk(Ed25519PrivateKey.generate(), "kx")


@pytest.mark.parametrize("bad", [
    "{not json",
    json.dumps([_K]),  # a bare list, not a JWKS object
    json.dumps({"keys": "nope"}),
    json.dumps({"keys": ["nope"]}),
    json.dumps({"keys": [{**_K, "kid": ""}]}),
    json.dumps({"keys": [{k: v for k, v in _K.items() if k != "kid"}]}),
    json.dumps({"keys": [{**_K, "kty": "RSA"}]}),
    json.dumps({"keys": [{**_K, "crv": "Ed448"}]}),
    json.dumps({"keys": [{**_K, "alg": "RS256"}]}),
    json.dumps({"keys": [{**_K, "use": "enc"}]}),
    json.dumps({"keys": [{**_K, "x": _b64url(b"short")}]}),
    json.dumps({"keys": [{**_K, "x": "not base64!"}]}),
    json.dumps({"keys": [{**_K, "d": "private-material"}]}),
    json.dumps({"keys": [_K, _K]}),  # duplicate kid
])
def test_malformed_jwks_fails_closed(keys, bad):
    keys.path.write_text(bad)
    _bump_mtime(keys.path)
    with pytest.raises(JwtConfigError):
        keys.auth.principal_for(f"Bearer {_sign(keys.k1)}")
    with pytest.raises(JwtConfigError):
        keys.auth.enforces()


def test_jwks_going_bad_after_a_good_load_does_not_serve_stale_keys(keys):
    assert _status(keys.auth, _sign(keys.k1)) == 200
    keys.path.write_text("{broken")
    _bump_mtime(keys.path)
    with pytest.raises(JwtConfigError):
        keys.auth.principal_for(f"Bearer {_sign(keys.k1)}")


def test_missing_jwks_file_fails_closed(keys):
    keys.path.unlink()
    with pytest.raises(JwtConfigError):
        keys.auth.principal_for(f"Bearer {_sign(keys.k1)}")


def test_jwt_config_error_maps_to_503_without_detail(keys):
    keys.path.write_text("{broken")
    _bump_mtime(keys.path)
    with pytest.raises(HTTPException) as exc:
        require_principal(authorization=f"Bearer {_sign(keys.k1)}", auth=keys.auth)
    assert exc.value.status_code == 503
    assert exc.value.detail == "authentication unavailable"


# --- B1: small-order / non-canonical Ed25519 public keys ---------------------------- #

_L = 2**252 + 27742317777372353535851937790883648493  # prime subgroup order


def _encode_point(x: int, y: int) -> bytes:
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _scalar_mult(k: int, pt: tuple[int, int]) -> tuple[int, int]:
    from kalos.portal.jwt_auth import _ed25519_add

    acc, base = (0, 1), pt
    while k:
        if k & 1:
            acc = _ed25519_add(acc, base)
        base = _ed25519_add(base, base)
        k >>= 1
    return acc


def _try_decode(raw: bytes) -> tuple[int, int] | None:
    from kalos.portal.jwt_auth import _ed25519_decode

    try:
        return _ed25519_decode(raw)
    except ValueError:
        return None


def _torsion_points() -> list[tuple[int, int]]:
    """The 8 points of the Ed25519 8-torsion subgroup, computed (not hardcoded)."""
    for i in range(1000):
        pt = _try_decode(hashlib.sha256(b"torsion-%d" % i).digest())
        if pt is None:
            continue
        t = _scalar_mult(_L, pt)  # kill the prime-order part, keep the torsion part
        if _scalar_mult(4, t) != (0, 1):  # order exactly 8: generates the subgroup
            pts = [_scalar_mult(k, t) for k in range(8)]
            assert len(set(pts)) == 8
            return pts
    raise AssertionError("no order-8 point found")


def _small_order_encodings() -> list[bytes]:
    from kalos.portal.jwt_auth import _P

    torsion = _torsion_points()
    out = {_encode_point(x, y) for x, y in torsion}
    assert len(out) == 8
    for x, y in torsion:
        if y + _P < 2**255:  # non-canonical y (y=0 -> p, y=1 -> p+1), both sign bits
            out.add((y + _P).to_bytes(32, "little"))
            out.add(((y + _P) | (1 << 255)).to_bytes(32, "little"))
        if x == 0:  # non-canonical sign bit on x=0 (identity and (0,-1))
            out.add((y | (1 << 255)).to_bytes(32, "little"))
    return sorted(out)


_SMALL_ORDER = _small_order_encodings()


def test_small_order_set_includes_zero_and_identity():
    assert bytes(32) in _SMALL_ORDER
    assert b"\x01" + bytes(31) in _SMALL_ORDER
    assert len(_SMALL_ORDER) >= 8 + 4


@pytest.mark.parametrize("raw", _SMALL_ORDER, ids=lambda b: b.hex()[:8] + ".." + b.hex()[-4:])
def test_parse_jwks_rejects_small_order_and_non_canonical_points(raw):
    jwk = {"kty": "OKP", "crv": "Ed25519", "kid": "evil", "x": _b64url(raw)}
    with pytest.raises(JwtConfigError, match="large-order"):
        parse_jwks(json.dumps({"keys": [jwk]}))


def test_parse_jwks_rejects_off_curve_points():
    off = next(raw for raw in (hashlib.sha256(b"off-%d" % i).digest() for i in range(1000))
               if _try_decode(raw) is None)
    with pytest.raises(JwtConfigError):
        parse_jwks(json.dumps({"keys": [
            {"kty": "OKP", "crv": "Ed25519", "kid": "k", "x": _b64url(off)}]}))


def test_parse_jwks_rejects_non_canonical_base64url():
    good = _jwk(Ed25519PrivateKey.generate(), "k")
    raw = base64.urlsafe_b64decode(good["x"] + "=")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    twin = good["x"][:-1] + alphabet[alphabet.index(good["x"][-1]) ^ 1]  # flip a pad bit
    assert base64.urlsafe_b64decode(twin + "=") == raw and twin != good["x"]
    with pytest.raises(JwtConfigError):
        parse_jwks(json.dumps({"keys": [{**good, "x": twin}]}))


def _forge_with_small_order_key(raw_key: bytes, claims: dict) -> str:
    """Forge an EdDSA JWT for a small-order public key A with S=0, by picking
    R in <A> so that R = -[h]A (h = SHA-512(R || A || M) mod L)."""
    a = _try_decode(raw_key)
    assert a is not None
    order = next(k for k in range(1, 9) if _scalar_mult(k, a) == (0, 1))
    multiples = [_scalar_mult(k, a) for k in range(order)]
    header = _b64url(json.dumps({"alg": "EdDSA", "typ": "JWT", "kid": "k1"}).encode())
    for attempt in range(200):
        payload = _b64url(json.dumps({**claims, "jti": f"forged-{attempt}"}).encode())
        msg = f"{header}.{payload}".encode()
        for k in range(order):
            r_enc = _encode_point(*multiples[k])
            h = int.from_bytes(hashlib.sha512(r_enc + raw_key + msg).digest(), "little") % _L
            if (-h) % order == k:
                return f"{header}.{payload}.{_b64url(r_enc + bytes(32))}"
    raise AssertionError("could not forge")


@pytest.mark.parametrize("raw_key", [bytes(32), b"\x01" + bytes(31)], ids=["zero", "identity"])
def test_forged_token_with_a_small_order_key_is_never_accepted(keys, raw_key):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from kalos.portal.jwt_auth import JwtSettings, verify_engine_token

    forged = _forge_with_small_order_key(raw_key, _claims())
    # Why the JWKS guard exists: the crypto backend loads such a key, and current
    # OpenSSL verifies the forged S=0 signature.
    settings = JwtSettings(keys.path, "kalos-web", "kalos-engine", 30, 120)
    weak = {"k1": Ed25519PublicKey.from_public_bytes(raw_key)}
    try:
        verify_engine_token(forged, settings=settings, keys=weak, now=NOW)
        backend_accepts = True
    except Exception:  # noqa: BLE001
        backend_accepts = False
    # The engine refuses the key outright (fail closed), so the forgery never verifies.
    keys.path.write_text(json.dumps({"keys": [
        {"kty": "OKP", "crv": "Ed25519", "kid": "k1", "x": _b64url(raw_key)}]}))
    _bump_mtime(keys.path)
    with pytest.raises(JwtConfigError):
        keys.auth.principal_for(f"Bearer {forged}")
    with pytest.raises(HTTPException) as exc:
        require_principal(authorization=f"Bearer {forged}", auth=keys.auth)
    assert exc.value.status_code == 503
    if not backend_accepts:
        pytest.skip("this crypto backend rejects the forgery itself; the JWKS guard still holds")


# --- production refuses the published test-vector keys ------------------------------ #

def test_production_refuses_test_vector_keys(keys, clean_env):
    vec = json.loads((Path(__file__).parent / "vectors" / "engine-token.json").read_text())
    primary = vec["keys"]["primary"]["public_jwk"]
    other = vec["keys"]["other"]["public_jwk"]
    clean_env.setenv("KALOS_ENV", "production")
    for jwk in [primary, {**primary, "kid": "prod-1"}, {**other, "kid": "prod-2"},
                _jwk(Ed25519PrivateKey.generate(), "kalos-test-vectors-new")]:
        keys.path.write_text(json.dumps({"keys": [jwk]}))
        _bump_mtime(keys.path)
        with pytest.raises(JwtConfigError, match="TEST-VECTOR"):
            keys.auth.jwt_key_count()
    clean_env.delenv("KALOS_ENV")
    keys.path.write_text(json.dumps({"keys": [primary]}))
    _bump_mtime(keys.path)
    assert keys.auth.jwt_key_count() == 1  # development accepts them (the vector tests)


# --- M1: the key file is validated even when static tokens exist -------------------- #

def test_broken_jwks_fails_closed_even_with_static_records(keys, clean_env):
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("svc-tok"), "subject": "svc", "tenant": "acme",
         "scopes": [READ], "service_account": True},
    ]))
    keys.path.write_text("{broken")
    _bump_mtime(keys.path)
    with pytest.raises(JwtConfigError):
        keys.auth.has_auth_source()
    with pytest.raises(JwtConfigError):
        keys.auth.enforces()


# --- key-file cache and logging -------------------------------------------------- #

def test_jwks_reload_notices_same_size_same_mtime_rewrites(keys):
    """ctime is part of the cache key: an in-place rewrite that restores the old
    mtime and keeps the size is still picked up."""
    token = _sign(keys.k1)
    assert _status(keys.auth, token) == 200
    st = keys.path.stat()
    time.sleep(0.02)
    k3 = Ed25519PrivateKey.generate()
    _write_jwks(keys.path, {"k1": k3})  # same kid -> same length
    os.utime(keys.path, ns=(st.st_atime_ns, st.st_mtime_ns))
    st2 = keys.path.stat()
    assert (st2.st_size, st2.st_mtime_ns, st2.st_ino) == (st.st_size, st.st_mtime_ns, st.st_ino)
    assert st2.st_ctime_ns != st.st_ctime_ns
    assert _status(keys.auth, token) == 401
    assert _status(keys.auth, _sign(k3)) == 200


def _errors(caplog) -> int:
    return len([r for r in caplog.records if r.levelno >= logging.ERROR])


def test_key_file_errors_are_logged_once_per_change(keys, caplog):
    caplog.set_level(logging.ERROR, logger="kalos.portal")
    keys.path.write_text("{broken")
    _bump_mtime(keys.path)
    token = f"Bearer {_sign(keys.k1)}"
    for _ in range(5):
        with pytest.raises(HTTPException):
            require_principal(authorization=token, auth=keys.auth)
    assert _errors(caplog) == 1
    keys.path.write_text("{still broken}")  # a change: logged again
    _bump_mtime(keys.path)
    with pytest.raises(HTTPException):
        require_principal(authorization=token, auth=keys.auth)
    assert _errors(caplog) == 2
    _write_jwks(keys.path, {"k1": keys.k1})  # fixed, then broken again: logged again
    _bump_mtime(keys.path)
    assert _status(keys.auth, _sign(keys.k1)) == 200
    keys.path.write_text("{broken")
    _bump_mtime(keys.path)
    with pytest.raises(HTTPException):
        require_principal(authorization=token, auth=keys.auth)
    assert _errors(caplog) == 3


def test_parse_jwks_accepts_the_vector_public_key():
    vectors = json.loads((Path(__file__).parent / "vectors" / "engine-token.json").read_text())
    jwk = vectors["keys"]["primary"]["public_jwk"]
    assert list(parse_jwks(json.dumps({"keys": [jwk]}))) == [jwk["kid"]]


def test_empty_jwks_enforces_and_rejects_everything(keys):
    keys.path.write_text(json.dumps({"keys": []}))
    _bump_mtime(keys.path)
    assert keys.auth.enforces() is True  # never anonymous
    assert keys.auth.has_auth_source() is False  # but production would refuse it
    assert _status(keys.auth, _sign(keys.k1)) == 401
    assert _status(keys.auth, None) == 401


# --- static tokens and open mode --------------------------------------------------- #

def test_static_token_still_works_alongside_jwt(keys, clean_env):
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("svc-tok"), "subject": "svc", "tenant": "acme",
         "scopes": [READ, WRITE], "service_account": True},
    ]))
    p = keys.auth.principal_for("Bearer svc-tok")
    assert p.auth_method == "static" and p.subject == "svc" and p.tenant == "acme"
    assert p.session_id is None and p.token_id is None
    assert keys.auth.principal_for(f"Bearer {_sign(keys.k1)}").auth_method == "jwt"
    assert _status(keys.auth, "wrong-static") == 401


def test_two_dot_static_token_falls_back_when_jwt_is_off(clean_env):
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("a.b.c"), "subject": "s", "tenant": "acme", "scopes": [READ]},
    ]))
    p = Authenticator().principal_for("Bearer a.b.c")
    assert p.auth_method == "static" and p.subject == "s"
    assert _status(Authenticator(), "x.y.z") == 401


def test_unparseable_two_dot_token_falls_back_to_static(keys, clean_env):
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("a.b.c"), "subject": "s", "tenant": "acme", "scopes": [READ]},
    ]))
    assert keys.auth.principal_for("Bearer a.b.c").auth_method == "static"
    assert _status(keys.auth, "x.y.z") == 401


def test_a_parseable_jwt_that_fails_verification_never_falls_back(keys, clean_env, caplog):
    forged = _sign(Ed25519PrivateKey.generate())  # well-formed JWS, wrong key
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha(forged), "subject": "s", "tenant": "acme", "scopes": [READ]},
    ]))
    assert _reason(keys.auth, forged, caplog) == "bad_signature"


def test_open_mode_unaffected_in_dev(clean_env):
    auth = Authenticator()
    assert auth.enforces() is False
    p = auth.principal_for(None)
    assert p.anonymous is True and p.auth_method == "anonymous"
    assert p.tenant == "default" and p.scopes == frozenset({READ, WRITE})


def test_rejections_are_generic_and_never_log_the_token(keys, caplog):
    token = _sign(keys.k1, _claims(tenant="../x"))
    caplog.set_level(logging.DEBUG, logger="kalos.portal")
    with pytest.raises(HTTPException) as exc:
        keys.auth.principal_for(f"Bearer {token}")
    assert exc.value.status_code == 401
    assert exc.value.detail == "invalid or expired token"
    assert token not in caplog.text
    assert token.split(".")[1] not in caplog.text and token.split(".")[2] not in caplog.text
    assert "reason=bad_tenant" in caplog.text and "kid=k1" in caplog.text
    assert "jti=jti-1" in caplog.text


@pytest.mark.parametrize("token_fn", [
    lambda k: _sign(Ed25519PrivateKey.generate()),
    lambda k: _sign(k, kid="nope"),
    lambda k: _sign(k, _claims(exp=NOW - 100, iat=NOW - 160, nbf=NOW - 160)),
    lambda k: _sign(k, _claims(scopes=["evil"])),
])
def test_every_rejection_has_the_same_body(keys, token_fn):
    with pytest.raises(HTTPException) as exc:
        keys.auth.principal_for(f"Bearer {token_fn(keys.k1)}")
    assert (exc.value.status_code, exc.value.detail) == (401, "invalid or expired token")


def test_unprintable_kid_is_not_logged_verbatim(keys, caplog):
    caplog.set_level(logging.DEBUG, logger="kalos.portal")
    _status(keys.auth, _sign(keys.k1, kid="evil\nkalos auth: forged line"))
    assert "forged line" not in caplog.text
    assert "kid=<unprintable>" in caplog.text


# --- production static-token rules (in-process: _load_records reads KALOS_ENV) ---- #

def _static(clean_env, **entry) -> None:
    rec = {"token_sha256": _sha("svc-tok"), "subject": "svc", "tenant": "acme",
           "scopes": [READ, WRITE], **entry}
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([rec]))


def test_production_refuses_non_service_account_static_tokens(clean_env):
    clean_env.setenv("KALOS_ENV", "production")
    _static(clean_env)
    with pytest.raises(AuthConfigError, match="service_account"):
        Authenticator().principal_for("Bearer svc-tok")
    _static(clean_env, service_account=False)
    with pytest.raises(AuthConfigError):
        Authenticator().principal_for("Bearer svc-tok")


def test_production_refuses_admin_on_static_tokens(clean_env):
    clean_env.setenv("KALOS_ENV", "production")
    _static(clean_env, service_account=True, scopes=[READ, ADMIN])
    with pytest.raises(AuthConfigError, match="admin"):
        Authenticator().principal_for("Bearer svc-tok")


def test_static_tokens_carry_exactly_one_tenant(clean_env):
    _static(clean_env, service_account=True, tenants=["acme", "globex"])
    with pytest.raises(AuthConfigError, match="exactly one"):
        Authenticator().principal_for("Bearer svc-tok")


def test_service_account_must_be_boolean(clean_env):
    _static(clean_env, service_account="yes")
    with pytest.raises(AuthConfigError):
        Authenticator().principal_for("Bearer svc-tok")


def test_production_accepts_a_service_account_static_token(clean_env):
    clean_env.setenv("KALOS_ENV", "production")
    _static(clean_env, service_account=True)
    p = Authenticator().principal_for("Bearer svc-tok")
    assert p.auth_method == "static" and p.tenant == "acme"


def test_development_keeps_legacy_static_records(clean_env):
    _static(clean_env, scopes=[READ, WRITE, ADMIN])  # no service_account, admin: fine in dev
    assert Authenticator().principal_for("Bearer svc-tok").has(ADMIN)


# --- production startup (subprocess: posture is decided at import) ---------------- #

def _run(script: str, tmp_path, **env: str) -> subprocess.CompletedProcess:
    full_env = {k: v for k, v in os.environ.items() if k not in _ENV_VARS}
    full_env.update({"KALOS_STATE_DIR": str(tmp_path)}, **env)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], env=full_env,
                          capture_output=True, text=True, timeout=180)


def test_production_refuses_to_start_with_an_empty_jwks(tmp_path):
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps({"keys": []}))
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production",
             KALOS_JWT_PUBLIC_KEYS_FILE=str(path))
    assert r.returncode != 0
    assert "requires authentication" in r.stderr


def test_production_refuses_to_start_with_a_malformed_jwks(tmp_path):
    path = tmp_path / "jwks.json"
    path.write_text("{broken")
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production",
             KALOS_JWT_PUBLIC_KEYS_FILE=str(path))
    assert r.returncode != 0
    assert "ProductionConfigError" in r.stderr


def test_production_refuses_to_start_with_an_out_of_range_ttl(tmp_path):
    path = tmp_path / "jwks.json"
    _write_jwks(path, {"k1": Ed25519PrivateKey.generate()})
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production",
             KALOS_JWT_PUBLIC_KEYS_FILE=str(path), KALOS_JWT_MAX_TTL_S="600")
    assert r.returncode != 0
    assert "ProductionConfigError" in r.stderr


def test_production_refuses_to_start_with_a_non_service_account_static_token(tmp_path):
    tokens = json.dumps([{"token_sha256": _sha("t"), "subject": "s", "tenant": "acme",
                          "scopes": [READ]}])
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production",
             KALOS_AUTH_TOKENS=tokens)
    assert r.returncode != 0
    assert "ProductionConfigError" in r.stderr


def test_production_refuses_to_start_with_a_malformed_jwks_and_valid_static_tokens(tmp_path):
    path = tmp_path / "jwks.json"
    path.write_text("{broken")
    tokens = json.dumps([{"token_sha256": _sha("t"), "subject": "s", "tenant": "acme",
                          "scopes": [READ], "service_account": True}])
    r = _run("import kalos.portal.app", tmp_path, KALOS_ENV="production",
             KALOS_JWT_PUBLIC_KEYS_FILE=str(path), KALOS_AUTH_TOKENS=tokens)
    assert r.returncode != 0
    assert "ProductionConfigError" in r.stderr


def test_production_middleware_503s_on_a_jwks_broken_at_runtime_despite_static_tokens(tmp_path):
    path = tmp_path / "jwks.json"
    _write_jwks(path, {"k1": Ed25519PrivateKey.generate()})
    tokens = json.dumps([{"token_sha256": _sha("svc-tok"), "subject": "s", "tenant": "acme",
                          "scopes": [READ], "service_account": True}])
    script = f"""
        import json, os, pathlib
        from fastapi.testclient import TestClient
        from kalos.portal.app import app
        c = TestClient(app)
        auth = {{"Authorization": "Bearer svc-tok"}}
        before = c.get("/api/latest", headers=auth).status_code
        p = pathlib.Path({str(path)!r})
        p.write_text("{{broken")
        st = p.stat(); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        after = c.get("/api/latest", headers=auth)
        print(json.dumps([before, after.status_code, after.text]))
    """
    r = _run(script, tmp_path, KALOS_ENV="production", KALOS_JWT_PUBLIC_KEYS_FILE=str(path),
             KALOS_AUTH_TOKENS=tokens)
    assert r.returncode == 0, r.stderr
    before, after, body = json.loads(r.stdout.strip().splitlines()[-1])
    assert (before, after) == (200, 503)
    assert "broken" not in body


def test_main_entrypoint_exits_2_on_a_malformed_jwks(tmp_path):
    path = tmp_path / "jwks.json"
    path.write_text("{broken")
    r = _run("import kalos.portal.__main__ as m; m.main()", tmp_path, KALOS_ENV="production",
             KALOS_JWT_PUBLIC_KEYS_FILE=str(path))
    assert r.returncode == 2
    assert "authentication configuration is invalid" in r.stderr
    assert "Traceback" not in r.stderr


def test_main_entrypoint_exits_2_on_an_empty_jwks_in_production(tmp_path):
    path = tmp_path / "jwks.json"
    path.write_text(json.dumps({"keys": []}))
    r = _run("import kalos.portal.__main__ as m; m.main()", tmp_path, KALOS_ENV="production",
             KALOS_JWT_PUBLIC_KEYS_FILE=str(path))
    assert r.returncode == 2
    assert "requires authentication" in r.stderr


def test_production_starts_with_jwt_only_and_fails_closed_at_runtime(tmp_path):
    """JWT keys alone satisfy production; a JWKS broken at runtime is a 503."""
    k1 = Ed25519PrivateKey.generate()
    path = tmp_path / "jwks.json"
    _write_jwks(path, {"k1": k1})
    now = int(time.time())
    token = _sign(k1, _claims(iat=now, nbf=now, exp=now + 60))
    script = f"""
        import json, os, pathlib
        from fastapi.testclient import TestClient
        from kalos.portal.app import app
        c = TestClient(app)
        auth = {{"Authorization": "Bearer " + os.environ["TEST_ENGINE_TOKEN"]}}
        out = [c.get("/api/latest", headers=auth).status_code,
               c.get("/api/latest").status_code,
               c.get("/api/latest", headers={{"Authorization": "Bearer x.y.z"}}).status_code]
        p = pathlib.Path({str(path)!r})
        p.write_text("{{broken")
        st = p.stat(); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        r = c.get("/api/latest", headers=auth)
        out += [r.status_code, r.text, c.get("/healthz").status_code]
        print(json.dumps(out))
    """
    r = _run(script, tmp_path, KALOS_ENV="production", KALOS_JWT_PUBLIC_KEYS_FILE=str(path),
             TEST_ENGINE_TOKEN=token)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out[:4] == [200, 401, 401, 503]
    assert "broken" not in out[4] and "jwks" not in out[4].lower()
    assert out[5] == 200


# --- end to end: the store sees the tenant from the JWT ---------------------------- #

def test_tenant_from_the_jwt_is_what_the_store_sees(keys, tmp_path):
    import numpy as np
    import pandas as pd
    from fastapi.testclient import TestClient

    from kalos.portal.app import app
    from kalos.portal.auth import get_authenticator
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Methanol": rng.uniform(0, 4, 8).round(3),
                       "pH": rng.uniform(5, 7, 8).round(2),
                       "lipase_titer": rng.uniform(0, 5, 8).round(3)})
    store = CampaignStore(tmp_path / "state")
    store.seed(df, "lipase_titer", ["Methanol", "pH"], tenant="acme-bio")

    seen: list[str] = []
    real_summary = store.summary

    def spy_summary(*, tenant, **kwargs):
        seen.append(tenant)
        return real_summary(tenant=tenant, **kwargs)

    store.summary = spy_summary  # type: ignore[method-assign]
    app.dependency_overrides[get_campaign_store] = lambda: store
    app.dependency_overrides[get_authenticator] = lambda: keys.auth
    try:
        tc = TestClient(app)

        def hdr(**claims):
            return {"Authorization": f"Bearer {_sign(keys.k1, _claims(**claims))}"}

        acme = tc.get("/api/campaign", headers=hdr())
        assert acme.status_code == 200 and acme.json()["has_campaign"] is True
        globex = tc.get("/api/campaign", headers=hdr(tenant="globex", sub="user_2"))
        assert globex.status_code == 200 and globex.json()["has_campaign"] is False
        assert seen == ["acme-bio", "globex"]

        body = {"recipes": [{"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
                             "mode": "explore", "reason": "x"}]}
        # a read-only token may not write; a write token writes to ITS tenant
        assert tc.post("/api/campaign/start", json=body,
                       headers=hdr(scopes=["campaigns:read"])).status_code == 403
        r = tc.post("/api/campaign/start", json=body, headers=hdr())
        assert r.status_code == 200, r.text
        assert real_summary(tenant="acme-bio")["n_awaiting"] == 1
        assert real_summary(tenant="globex")["has_campaign"] is False

        bad = tc.get("/api/campaign", headers=hdr(tenant="../acme-bio"))
        assert bad.status_code == 401 and "invalid or expired token" in bad.text
        assert "tenant" not in bad.text
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)
        app.dependency_overrides.pop(get_authenticator, None)


# --- shared vectors from kalos-web (tests/vectors/engine-token.json) -------------- #

_VECTORS = json.loads((Path(__file__).parent / "vectors" / "engine-token.json").read_text())


def _vector_auth(tmp_path, clean_env, verify_at: int) -> Authenticator:
    spec = _VECTORS["spec"]
    path = tmp_path / "vector-jwks.json"
    path.write_text(json.dumps({"keys": [_VECTORS["keys"]["primary"]["public_jwk"]]}))
    clean_env.setenv("KALOS_JWT_PUBLIC_KEYS_FILE", str(path))
    clean_env.setenv("KALOS_JWT_ISSUER", spec["iss"])
    clean_env.setenv("KALOS_JWT_AUDIENCE", spec["aud"])
    clean_env.setenv("KALOS_JWT_MAX_TTL_S", str(spec["max_ttl_seconds"]))
    clean_env.setenv("KALOS_JWT_LEEWAY_S", str(spec["recommended_clock_leeway_seconds"]))
    return Authenticator(clock=lambda: float(verify_at))


def test_vector_spec_matches_the_engine_contract():
    spec = _VECTORS["spec"]
    assert spec["alg"] == "EdDSA" and spec["crv"] == "Ed25519"
    assert spec["header_kid_required"] is True
    assert set(spec["required_claims"]) == set(REQUIRED_CLAIMS)
    assert set(spec["scopes"]) == set(SCOPE_MAP)
    assert spec["tenant_pattern"] == f"^{TENANT_ID_PATTERN.pattern}$"
    assert spec["max_ttl_seconds"] == 120 and spec["recommended_clock_leeway_seconds"] == 30
    assert "d" not in _VECTORS["keys"]["primary"]["public_jwk"]  # public keys only


@pytest.mark.parametrize("vec", _VECTORS["valid"], ids=lambda v: v["name"])
def test_engine_accepts_every_valid_vector(tmp_path, clean_env, vec):
    auth = _vector_auth(tmp_path, clean_env, vec["verify_at"])
    p = auth.principal_for(f"Bearer {vec['token']}")
    claims = vec["claims"]
    assert p.auth_method == "jwt"
    assert p.subject == claims["sub"]
    assert p.tenant == claims["tenant"]
    assert p.session_id == claims["sid"]
    assert p.token_id == claims["jti"]
    assert p.scopes == frozenset(SCOPE_MAP[s] for s in claims["scopes"])


# Why each known vector must fail. A vector added upstream later still has to be
# rejected (401) even before it gets an entry here.
_EXPECTED_REASON = {
    "expired": "expired", "not_yet_valid": "not_yet_valid", "wrong_audience": "bad_aud",
    "wrong_issuer": "bad_iss", "ttl_over_120": "ttl_too_long", "missing_kid": "missing_kid",
    "unknown_kid": "unknown_kid", "signed_by_other_key_same_kid": "bad_signature",
    "tampered_payload": "bad_signature", "alg_none": "bad_alg",
    "invalid_tenant_format": "bad_tenant", "missing_tenant": "missing_claim",
    "unknown_scope": "bad_scopes", "scopes_not_array": "bad_scopes",
    "missing_jti": "missing_claim", "missing_sub": "missing_claim",
}


@pytest.mark.parametrize("vec", _VECTORS["invalid"], ids=lambda v: v["name"])
def test_engine_rejects_every_invalid_vector(tmp_path, clean_env, vec, caplog):
    auth = _vector_auth(tmp_path, clean_env, vec["verify_at"])
    reason = _reason(auth, vec["token"], caplog)
    if vec["name"] in _EXPECTED_REASON:
        assert reason == _EXPECTED_REASON[vec["name"]]
