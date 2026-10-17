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


def test_wrong_issuer_is_rejected(keys):
    assert _status(keys.auth, _sign(keys.k1, _claims(iss="evil"))) == 401


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


def test_two_dot_token_without_jwt_config_is_rejected(clean_env):
    clean_env.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("a.b.c"), "subject": "s", "tenant": "acme", "scopes": [READ]},
    ]))
    # A two-dot bearer always takes the JWT branch, even if a static hash matches.
    assert _status(Authenticator(), "a.b.c") == 401


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
