"""Engine-side tenant isolation for JWT (engine-token) callers, end to end.

Contract: docs/ENGINE_TOKEN.md. The verifier itself is unit-tested in
tests/test_jwt_auth.py; this module drives the REAL app over HTTP:

- the main case runs `uvicorn kalos.portal.app:app` in a SUBPROCESS with
  `KALOS_ENV=production` (the production posture is decided at import), a real
  Ed25519 JWKS file, a service-account static record and a deny-list env;
- rotation runs a second production server whose JWKS file is replaced
  atomically while it is up;
- the leeway boundaries need a fixed clock, so they run in a production
  subprocess with the authenticator's clock injected (a live clock cannot pin
  "exp + 29 s" versus "exp + 31 s");
- the shared web vectors (tests/vectors/engine-token.json) run through the HTTP
  app in-process.

Nothing is skipped or stubbed. Every key is generated here at test time; no
secret is committed. Server logs are captured in full and checked for the
reason codes and for the absence of any token material.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS = json.loads((Path(__file__).parent / "vectors" / "engine-token.json").read_text())

ISS = "kalos-web"
AUD = "kalos-engine"
GENERIC_BODY = {"error": "invalid or expired token", "code": "unauthenticated"}
DROP = object()  # claim-override sentinel: remove the claim entirely

# Every token this module sends to a live server, so the log check can prove
# none of them (or any of their segments) reached the log.
_SENT: list[str] = []

_KALOS_ENV_PREFIX = "KALOS_"


# --- token minting ------------------------------------------------------------ #

def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _json_b64(obj: Any) -> str:
    return _b64(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def _hs256(secret: bytes, claims: dict, kid: str) -> str:
    """An HS256 token MAC'd with `secret`, built by hand: PyJWT refuses to use a
    PEM as an HMAC key, but an attacker has no such check."""
    import hmac as _hmac

    return _compact({"alg": "HS256", "typ": "JWT", "kid": kid}, _json_b64(claims),
                    lambda msg: _hmac.new(secret, msg, hashlib.sha256).digest())


def _pub_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _jwk(key: Ed25519PrivateKey, kid: str) -> dict:
    return {"kty": "OKP", "crv": "Ed25519", "x": _b64(_pub_raw(key)), "kid": kid,
            "alg": "EdDSA", "use": "sig"}


def _write_jwks(path: Path, keys: dict[str, Ed25519PrivateKey]) -> None:
    """Write the key file atomically (temp file, then rename): the rotation
    procedure in docs/ENGINE_TOKEN.md, and never a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"keys": [_jwk(k, kid) for kid, k in keys.items()]}))
    os.replace(tmp, path)


def _claims(now: int | None = None, **over: Any) -> dict:
    now = int(time.time()) if now is None else now
    claims: dict[str, Any] = {
        "iss": ISS, "aud": AUD, "sub": "user-acme", "sid": "sess-acme", "tenant": "acme",
        "scopes": ["campaigns:read", "campaigns:write"], "jti": str(uuid.uuid4()),
        "iat": now, "nbf": now, "exp": now + 60,
    }
    for k, v in over.items():
        if v is DROP:
            claims.pop(k, None)
        else:
            claims[k] = v
    return claims


def _compact(header: dict, payload: str, sign) -> str:
    signing_input = f"{_json_b64(header)}.{payload}"
    return f"{signing_input}.{_b64(sign(signing_input.encode('ascii')))}"


def mint(key: Ed25519PrivateKey, kid: str, *, header: dict | None = None,
         claims: dict | None = None, raw_payload: str | None = None) -> str:
    """A real EdDSA token: header `{alg, typ, kid}` (overridable), claims signed
    by `key`. `raw_payload` skips the JSON encoding (for hostile payloads)."""
    hdr = {"alg": "EdDSA", "typ": "JWT", "kid": kid}
    if header is not None:
        hdr = {k: v for k, v in {**hdr, **header}.items() if v is not DROP}
    payload = raw_payload if raw_payload is not None else _json_b64(_claims() if claims is None else claims)
    token = _compact(hdr, payload, key.sign)
    _SENT.append(token)
    return token


@dataclass(frozen=True)
class Who:
    """A caller identity the test server knows: its key, kid, tenant, sub, sid."""

    key: Ed25519PrivateKey
    kid: str
    tenant: str
    sub: str
    sid: str
    scopes: tuple[str, ...] = ("campaigns:read", "campaigns:write")

    def token(self, **over: Any) -> str:
        base = {"tenant": self.tenant, "sub": self.sub, "sid": self.sid, "scopes": list(self.scopes)}
        base.update(over)
        return mint(self.key, self.kid, claims=_claims(**base))

    def header(self, **over: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(**over)}"}


# --- the live production server ------------------------------------------------ #

_LAUNCH = """
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")
import uvicorn
uvicorn.run("kalos.portal.app:app", host="127.0.0.1", port=int(sys.argv[1]), log_level="info")
"""

_REASON_RE = re.compile(r"kalos auth: rejected (?:jwt|static token) reason=(\S+)")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _prod_env(state: Path, jwks: Path, extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_KALOS_ENV_PREFIX)}
    env.update({
        "KALOS_ENV": "production",
        "KALOS_STATE_DIR": str(state),
        "KALOS_JWT_PUBLIC_KEYS_FILE": str(jwks),
        "KALOS_TORCH_THREADS": "0",
    })
    env.update(extra)
    return env


class LiveServer:
    def __init__(self, proc: subprocess.Popen, base: str, log_path: Path, jwks: Path) -> None:
        self.proc = proc
        self.base = base
        self.log_path = log_path
        self.jwks = jwks

    def request(self, method: str, path: str, *, auth: Who | str | None = None,
                headers: dict[str, str] | None = None, **kw: Any) -> httpx.Response:
        hdrs = dict(headers or {})
        if isinstance(auth, Who):
            hdrs["Authorization"] = f"Bearer {auth.token()}"
        elif isinstance(auth, str):
            hdrs["Authorization"] = auth if auth.startswith(("Bearer ", "bearer ")) else f"Bearer {auth}"
            _SENT.append(auth)
        with httpx.Client(base_url=self.base, timeout=300) as c:
            return c.request(method, path, headers=hdrs, **kw)

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace")

    def log_mark(self) -> int:
        return self.log_path.stat().st_size if self.log_path.exists() else 0

    def reasons_since(self, mark: int) -> list[str]:
        return _REASON_RE.findall(self.log_text()[mark:])


def _start_server(root: Path, jwks: Path, extra: dict[str, str]) -> Iterator[LiveServer]:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    log_path = root / "server.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-c", _LAUNCH, str(port)],
            env=_prod_env(state, jwks, extra), cwd=REPO_ROOT,
            stdout=logf, stderr=subprocess.STDOUT,
        )
    srv = LiveServer(proc, f"http://127.0.0.1:{port}", log_path, jwks)
    deadline = time.time() + 300
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited early:\n{srv.log_text()[-4000:]}")
            try:
                if httpx.get(srv.base + "/healthz", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"server did not start:\n{srv.log_text()[-4000:]}")
            time.sleep(0.3)
        yield srv
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


KEY_A = Ed25519PrivateKey.generate()  # the web's signing key (kid k-main)
KEY_ATTACKER = Ed25519PrivateKey.generate()  # not in any JWKS
KEY_ROTATE_OLD = Ed25519PrivateKey.generate()
KEY_ROTATE_NEW = Ed25519PrivateKey.generate()
KID = "k-main"

ACME = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-acme")
GLOBEX = Who(KEY_A, KID, tenant="globex", sub="user-globex", sid="sess-globex")
READ_ONLY = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-acme", scopes=("campaigns:read",))
WRITE_ONLY = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-acme", scopes=("campaigns:write",))
ADMIN_ONLY = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-acme", scopes=("org:admin",))
CAROL = Who(KEY_A, KID, tenant="carol", sub="user-carol", sid="sess-carol")  # upload test
DAN = Who(KEY_A, KID, tenant="dan", sub="user-dan", sid="sess-dan")  # reanalyze test
BANNED_SUB = Who(KEY_A, KID, tenant="acme", sub="user-banned", sid="sess-other")
REVOKED_SID = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-revoked")

SVC_TOKEN = "svc-" + secrets.token_urlsafe(24)


@pytest.fixture(scope="module")
def main(tmp_path_factory) -> Iterator[LiveServer]:
    """Production server: one JWKS key, a service-account static token, and a
    deny-list naming one subject and one session."""
    root = tmp_path_factory.mktemp("main")
    jwks = root / "jwks.json"
    _write_jwks(jwks, {KID: KEY_A})
    static = root / "static-tokens.json"
    static.write_text(json.dumps([{
        "token_sha256": hashlib.sha256(SVC_TOKEN.encode()).hexdigest(),
        "subject": "svc-ingest", "tenant": "svc-ops",
        "scopes": ["read", "write"], "service_account": True,
    }]))
    yield from _start_server(root, jwks, {
        "KALOS_AUTH_TOKENS_FILE": str(static),
        "KALOS_AUTH_DENY_SUBJECTS": "sub:user-banned,sid:sess-revoked",
    })


# --- seeding two real tenants through the real API ------------------------------ #

def _sheet(seed: int, n: int = 10) -> str:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    methanol = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    titer = 1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)
    df = pd.DataFrame({"Methanol": methanol.round(3), "pH": ph.round(2),
                       "lipase_titer": titer.round(3)})
    return df.to_csv(index=False)


_RECIPE = {"recipes": [{"recipe": {"Methanol": 1.5, "pH": 6.0}, "pred": 1.0,
                        "std": 0.1, "mode": "explore", "reason": "seed"}]}


@dataclass
class World:
    acme_runs: list[str]
    globex_runs: list[str]


@pytest.fixture(scope="module")
def world(main: LiveServer) -> World:
    """Each tenant uploads its own run sheet, starts a batch, and measures one run.
    The two tenants use different dataset names and different values."""
    for who, name, seed in ((ACME, "acme-runs.csv", 1), (GLOBEX, "globex-runs.csv", 2)):
        r = main.request("POST", "/api/run", auth=who,
                         files={"file": (name, _sheet(seed), "text/csv")})
        assert r.status_code == 200, r.text
    ids: dict[str, list[str]] = {}
    for who in (ACME, GLOBEX):
        r = main.request("POST", "/api/campaign/start", auth=who, json=_RECIPE)
        assert r.status_code == 200, r.text
        ids[who.tenant] = [run["id"] for run in r.json()["started"]]
    r = main.request("POST", "/api/campaign/result", auth=ACME,
                     json={"id": ids["acme"][0], "value": 1.25})
    assert r.status_code == 200, r.text
    return World(acme_runs=ids["acme"], globex_runs=ids["globex"])


# --- valid tokens see only their own tenant ------------------------------------- #

def test_each_tenant_reads_only_its_own_latest_and_campaign(main, world):
    a_latest = main.request("GET", "/api/latest", auth=ACME).json()
    b_latest = main.request("GET", "/api/latest", auth=GLOBEX).json()
    assert a_latest["has_data"] is True and a_latest["dataset"] == "acme-runs.csv"
    assert b_latest["has_data"] is True and b_latest["dataset"] == "globex-runs.csv"
    assert "globex" not in json.dumps(a_latest) and "acme" not in json.dumps(b_latest)

    a_camp = main.request("GET", "/api/campaign", auth=ACME)
    b_camp = main.request("GET", "/api/campaign", auth=GLOBEX)
    assert a_camp.status_code == 200 and b_camp.status_code == 200
    a_text, b_text = a_camp.text, b_camp.text
    assert all(i in a_text for i in world.acme_runs)
    assert not any(i in a_text for i in world.globex_runs)
    assert all(i in b_text for i in world.globex_runs)
    assert not any(i in b_text for i in world.acme_runs)
    assert a_camp.json()["n_measured"] >= 1
    assert b_camp.json()["n_measured"] == 0


def test_providers_and_probes_need_no_tenant_and_leak_none(main, world):
    assert main.request("GET", "/healthz").json() == {"status": "ok"}
    assert main.request("GET", "/readyz").status_code == 200
    assert main.request("GET", "/api/providers", auth=ACME).status_code == 200


# --- tenant A can never reach tenant B ------------------------------------------ #

def test_b_run_id_in_result_body_is_indistinguishable_from_a_missing_id(main, world):
    """A's token names B's real run id: the answer must be byte-for-byte the
    same as for an id that exists nowhere (only the echoed id differs)."""
    b_id = world.globex_runs[0]
    missing = uuid.uuid4().hex
    r_b = main.request("POST", "/api/campaign/result", auth=ACME, json={"id": b_id, "value": 9.9})
    r_m = main.request("POST", "/api/campaign/result", auth=ACME, json={"id": missing, "value": 9.9})
    assert r_b.status_code == r_m.status_code
    assert r_b.text.replace(b_id, "<ID>") == r_m.text.replace(missing, "<ID>")
    # ... and the write did not land: B's run is still awaiting its result.
    b_camp = main.request("GET", "/api/campaign", auth=GLOBEX).json()
    b_run = next(run for run in b_camp["pending"] if run["id"] == b_id)
    assert b_run["result"] is None and b_camp["n_measured"] == 0


@pytest.mark.parametrize("surface", ["query", "header", "path"])
def test_b_run_id_on_any_surface_is_identical_to_a_missing_id(main, world, surface):
    """B's run id, carried in the query, a header or the path of A's result
    call, gets exactly the answer a made-up id gets (the id is only echoed)."""
    b_id = world.globex_runs[0]
    missing = uuid.uuid4().hex

    def call(run_id: str):
        body = {"id": run_id, "value": 3.0}
        if surface == "query":
            return main.request("POST", f"/api/campaign/result?id={run_id}&campaign_id={run_id}",
                                auth=ACME, json=body)
        if surface == "header":
            return main.request("POST", "/api/campaign/result", auth=ACME, json={"value": 3.0},
                                headers={"X-Campaign-Id": run_id, "X-Run-Id": run_id})
        return main.request("POST", f"/api/campaign/{run_id}/result", auth=ACME, json=body)

    r_b, r_m = call(b_id), call(missing)
    assert r_b.status_code == r_m.status_code, (surface, r_b.status_code, r_m.status_code)
    assert r_b.text.replace(b_id, "<ID>") == r_m.text.replace(missing, "<ID>")
    b_camp = main.request("GET", "/api/campaign", auth=GLOBEX).json()
    assert next(run for run in b_camp["pending"] if run["id"] == b_id)["result"] is None


def test_b_run_id_cannot_be_read_by_a(main, world):
    """No route exposes one run by id; a read of B's run through A's view is
    just A's campaign, which contains none of B's ids."""
    b_id = world.globex_runs[0]
    assert b_id not in main.request("GET", "/api/campaign", auth=ACME).text
    assert main.request("GET", f"/api/campaign/{b_id}", auth=ACME).status_code == 404
    assert main.request("GET", f"/api/campaign/{uuid.uuid4().hex}", auth=ACME).status_code == 404


@pytest.mark.parametrize("where", ["query", "header", "body", "path"])
def test_a_tenant_selector_naming_b_is_ignored_on_every_surface(main, world, where):
    """`tenant` is taken from the token only: never from the query, a header,
    the body or the path. A's requests with B named everywhere still act on A."""
    b = "globex"
    before_b = main.request("GET", "/api/latest", auth=GLOBEX).json()
    before_b_camp = main.request("GET", "/api/campaign", auth=GLOBEX).json()
    kw: dict[str, Any] = {}
    if where == "query":
        url_tail = f"?tenant={b}&tenant_id={b}&campaign_tenant={b}"
    elif where == "header":
        url_tail = ""
        kw["headers"] = {"X-Tenant": b, "X-Kalos-Tenant": b, "X-Tenant-Id": b, "X-Tenant-Scope": b}
    elif where == "body":
        url_tail = ""
    else:
        url_tail = ""
    a_before = main.request("GET", "/api/latest", auth=ACME).json()
    latest = main.request("GET", f"/api/latest{url_tail}", auth=ACME, **kw).json()
    assert latest == a_before and "globex" not in json.dumps(latest)

    camp_url = "/api/campaign/start" + (url_tail if where == "query" else "")
    body = dict(_RECIPE)
    if where == "body":
        body = {**_RECIPE, "tenant": b, "campaign": {"tenant": b}}
    if where == "path":
        assert main.request("GET", f"/api/tenants/{b}/latest", auth=ACME).status_code == 404
        assert main.request("GET", f"/api/{b}/latest", auth=ACME).status_code == 404
    started = main.request("POST", camp_url, auth=ACME, json=body, **kw)
    assert started.status_code == 200, started.text
    new_ids = [run["id"] for run in started.json()["started"]]
    assert new_ids and not set(new_ids) & set(world.globex_runs)
    # B is unchanged by A's write: same pending set, same counts.
    after_b_camp = main.request("GET", "/api/campaign", auth=GLOBEX).json()
    assert after_b_camp == before_b_camp
    assert main.request("GET", "/api/latest", auth=GLOBEX).json() == before_b
    # A's write landed in A, not B.
    assert set(new_ids) <= {run["id"] for run in main.request("GET", "/api/campaign", auth=ACME).json()["pending"]}


def test_an_upload_naming_b_lands_in_the_uploader_only(main, world):
    """carol uploads while naming globex (query and header). Her analysis is
    hers; globex's and acme's are unchanged."""
    first = main.request("POST", "/api/run", auth=CAROL,
                         files={"file": ("carol-first.csv", _sheet(5), "text/csv")})
    assert first.status_code == 200, first.text
    before_b = main.request("GET", "/api/latest", auth=GLOBEX).json()
    before_a = main.request("GET", "/api/latest", auth=ACME).json()
    r = main.request("POST", "/api/run?tenant=globex", auth=CAROL,
                     files={"file": ("carol-second.csv", _sheet(7), "text/csv")},
                     headers={"X-Tenant": "globex"})
    assert r.status_code == 200, r.text
    assert main.request("GET", "/api/latest", auth=CAROL).json()["dataset"] == "carol-second.csv"
    assert main.request("GET", "/api/latest", auth=GLOBEX).json() == before_b
    assert main.request("GET", "/api/latest", auth=ACME).json() == before_a


def test_a_reanalyze_never_touches_b(main, world):
    """dan re-analyzes its own campaign (its body names globex). globex's and
    acme's analyses and campaigns are unchanged."""
    seeded = main.request("POST", "/api/run", auth=DAN,
                          files={"file": ("dan-runs.csv", _sheet(9), "text/csv")})
    assert seeded.status_code == 200, seeded.text
    started = main.request("POST", "/api/campaign/start", auth=DAN, json=_RECIPE).json()
    run_id = started["started"][0]["id"]
    assert main.request("POST", "/api/campaign/result", auth=DAN,
                        json={"id": run_id, "value": 1.5}).status_code == 200
    before_b = main.request("GET", "/api/latest", auth=GLOBEX).json()
    before_b_camp = main.request("GET", "/api/campaign", auth=GLOBEX).json()
    before_a = main.request("GET", "/api/latest", auth=ACME).json()
    r = main.request("POST", "/api/campaign/reanalyze", auth=DAN, json={"tenant": "globex"})
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["dataset"].startswith("campaign round")
    assert main.request("GET", "/api/latest", auth=GLOBEX).json() == before_b
    assert main.request("GET", "/api/campaign", auth=GLOBEX).json() == before_b_camp
    assert main.request("GET", "/api/latest", auth=ACME).json() == before_a


def test_b_reanalyze_with_no_campaign_of_its_own_is_its_own_error(main):
    """A tenant with no campaign gets its OWN 'no data' answer; it does not see
    another tenant's campaign to fold."""
    nobody = Who(KEY_A, KID, tenant="nobody", sub="user-nobody", sid="sess-nobody")
    r = main.request("POST", "/api/campaign/reanalyze", auth=nobody)
    assert r.status_code == 400 and "no active campaign" in r.json()["error"]
    assert main.request("GET", "/api/campaign", auth=nobody).json() == {"has_campaign": False}


# --- rejection matrix: every refused token is the same generic 401 -------------- #

def _refused(main: LiveServer, token: str, reason: str | set[str], path: str = "/api/latest",
             method: str = "GET", json_body: Any = None) -> None:
    mark = main.log_mark()
    r = main.request(method, path, auth=token, json=json_body)
    assert r.status_code == 401, (r.status_code, r.text)
    assert r.json() == GENERIC_BODY
    assert r.headers.get("www-authenticate") == "Bearer"
    got = set(main.reasons_since(mark))
    expected = {reason} if isinstance(reason, str) else reason
    assert got == expected, (got, expected)


def _swap_payload(token: str, **claims: Any) -> str:
    """The same token with a different payload and the ORIGINAL signature."""
    head, mid, sig = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(mid + "=" * (-len(mid) % 4)))
    payload.update(claims)
    return f"{head}.{_json_b64(payload)}.{sig}"


def _valid_token() -> str:
    return ACME.token()


def _flip_sig(token: str) -> str:
    head, mid, sig = token.split(".")
    flipped = sig[:-4] + ("AAAA" if not sig.endswith("AAAA") else "BBBB")
    return f"{head}.{mid}.{flipped}"


_BIG_JTI = "j" * 9000
_LONG_TENANT = "t" * 65

_RSA = rsa.generate_private_key(public_exponent=65537, key_size=2048)

MATRIX: list[tuple[str, Any, str]] = [
    # --- signature and key selection
    ("tampered_payload", lambda: _swap_payload(_valid_token(), tenant="globex"), "bad_signature"),
    ("tampered_signature", lambda: _flip_sig(_valid_token()), "bad_signature"),
    ("signed_by_attacker_with_real_kid", lambda: mint(KEY_ATTACKER, KID, claims=_claims(tenant="globex")), "bad_signature"),
    ("unknown_kid", lambda: mint(KEY_A, "k-rogue", claims=_claims()), "unknown_kid"),
    ("missing_kid", lambda: mint(KEY_A, "", header={"kid": DROP}, claims=_claims()), "missing_kid"),
    ("empty_kid", lambda: mint(KEY_A, "", claims=_claims()), "missing_kid"),
    ("non_string_kid", lambda: mint(KEY_A, "", header={"kid": ["k-main"]}, claims=_claims()), "missing_kid"),
    # --- algorithm confusion
    ("alg_none", lambda: _compact({"alg": "none", "typ": "JWT", "kid": KID}, _json_b64(_claims()), lambda _m: b""), "bad_alg"),
    ("alg_None_case", lambda: _compact({"alg": "None", "typ": "JWT", "kid": KID}, _json_b64(_claims()), lambda _m: b""), "bad_alg"),
    ("hs256_public_raw_as_secret", lambda: jwt.encode(_claims(), _pub_raw(KEY_A), algorithm="HS256", headers={"kid": KID}), "bad_alg"),
    ("hs256_public_pem_as_secret", lambda: _hs256(KEY_A.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo), _claims(), KID), "bad_alg"),
    ("rs256", lambda: jwt.encode(_claims(), _RSA, algorithm="RS256", headers={"kid": KID}), "bad_alg"),
    # --- structure
    ("two_dots_not_a_jws", lambda: "a.b.c", "no_match"),
    ("two_dots_empty_parts", lambda: "..", "no_match"),
    ("crit_header", lambda: mint(KEY_A, KID, header={"crit": ["exp"]}, claims=_claims()), "crit_header"),
    ("typ_not_jwt", lambda: mint(KEY_A, KID, header={"typ": "JWS"}, claims=_claims()), "bad_typ"),
    ("oversized_token", lambda: mint(KEY_A, KID, claims=_claims(jti=_BIG_JTI)), "too_large"),
    # --- time
    ("expired", lambda: mint(KEY_A, KID, claims=_claims(iat=int(time.time()) - 300, nbf=int(time.time()) - 300, exp=int(time.time()) - 240)), "expired"),
    ("expired_long_ago", lambda: mint(KEY_A, KID, claims=_claims(iat=1_000_000_000, nbf=1_000_000_000, exp=1_000_000_060)), "expired"),
    ("not_yet_valid", lambda: mint(KEY_A, KID, claims=_claims(iat=int(time.time()) + 300, nbf=int(time.time()) + 300, exp=int(time.time()) + 360)), "not_yet_valid"),
    ("issued_in_future", lambda: mint(KEY_A, KID, claims=_claims(iat=int(time.time()) + 300, exp=int(time.time()) + 360)), "issued_in_future"),
    ("ttl_121", lambda: mint(KEY_A, KID, claims=_claims(exp=int(time.time()) + 121)), "ttl_too_long"),
    ("exp_before_iat", lambda: mint(KEY_A, KID, claims=_claims(exp=int(time.time()) - 5)), "bad_time_claim"),
    ("exp_as_string", lambda: mint(KEY_A, KID, claims=_claims(exp=str(int(time.time()) + 60))), "bad_time_claim"),
    ("exp_as_bool", lambda: mint(KEY_A, KID, claims=_claims(exp=True)), "bad_time_claim"),
    ("exp_huge", lambda: mint(KEY_A, KID, claims=_claims(exp=10**400)), "bad_time_claim"),
    ("exp_nan", lambda: mint(KEY_A, KID, raw_payload=_b64(json.dumps({**_claims(), "exp": float("nan")}).encode())), "bad_time_claim"),
    ("missing_exp", lambda: mint(KEY_A, KID, claims=_claims(exp=DROP)), "missing_claim"),
    ("missing_iat", lambda: mint(KEY_A, KID, claims=_claims(iat=DROP)), "missing_claim"),
    ("missing_nbf", lambda: mint(KEY_A, KID, claims=_claims(nbf=DROP)), "missing_claim"),
    # --- audience and issuer
    ("wrong_aud", lambda: mint(KEY_A, KID, claims=_claims(aud="kalos-admin")), "bad_aud"),
    ("aud_array_containing_engine", lambda: mint(KEY_A, KID, claims=_claims(aud=[AUD])), "bad_aud"),
    ("aud_array_with_extra", lambda: mint(KEY_A, KID, claims=_claims(aud=[AUD, "other"])), "bad_aud"),
    ("missing_aud", lambda: mint(KEY_A, KID, claims=_claims(aud=DROP)), "missing_claim"),
    ("wrong_iss", lambda: mint(KEY_A, KID, claims=_claims(iss="kalos-admin")), "bad_iss"),
    ("iss_superstring", lambda: mint(KEY_A, KID, claims=_claims(iss="kalos-web.evil.example")), "bad_iss"),
    ("iss_substring", lambda: mint(KEY_A, KID, claims=_claims(iss="evil-kalos-web")), "bad_iss"),
    ("iss_array", lambda: mint(KEY_A, KID, claims=_claims(iss=[ISS])), "bad_iss"),
    ("missing_iss", lambda: mint(KEY_A, KID, claims=_claims(iss=DROP)), "missing_claim"),
    # --- identity claims
    ("sub_space", lambda: mint(KEY_A, KID, claims=_claims(sub="user acme")), "bad_identity_claim"),
    ("sub_comma", lambda: mint(KEY_A, KID, claims=_claims(sub="a,b")), "bad_identity_claim"),
    ("sub_non_ascii", lambda: mint(KEY_A, KID, claims=_claims(sub="usér")), "bad_identity_claim"),
    ("sub_control", lambda: mint(KEY_A, KID, claims=_claims(sub="user\nacme")), "bad_identity_claim"),
    ("sub_empty", lambda: mint(KEY_A, KID, claims=_claims(sub="")), "bad_identity_claim"),
    ("sub_257", lambda: mint(KEY_A, KID, claims=_claims(sub="u" * 257)), "bad_identity_claim"),
    ("sub_number", lambda: mint(KEY_A, KID, claims=_claims(sub=123)), "bad_identity_claim"),
    ("jti_number", lambda: mint(KEY_A, KID, claims=_claims(jti=123)), "bad_identity_claim"),
    ("sid_space", lambda: mint(KEY_A, KID, claims=_claims(sid="sess acme")), "bad_identity_claim"),
    ("sid_del", lambda: mint(KEY_A, KID, claims=_claims(sid="sess\x7f")), "bad_identity_claim"),
    ("sid_257", lambda: mint(KEY_A, KID, claims=_claims(sid="s" * 257)), "bad_identity_claim"),
    ("missing_sub", lambda: mint(KEY_A, KID, claims=_claims(sub=DROP)), "missing_claim"),
    ("missing_sid", lambda: mint(KEY_A, KID, claims=_claims(sid=DROP)), "missing_claim"),
    ("missing_jti", lambda: mint(KEY_A, KID, claims=_claims(jti=DROP)), "missing_claim"),
    # --- tenant
    ("tenant_default_reserved", lambda: mint(KEY_A, KID, claims=_claims(tenant="default")), "bad_tenant"),
    ("tenant_traversal", lambda: mint(KEY_A, KID, claims=_claims(tenant="../globex")), "bad_tenant"),
    ("tenant_slash", lambda: mint(KEY_A, KID, claims=_claims(tenant="acme/globex")), "bad_tenant"),
    ("tenant_space", lambda: mint(KEY_A, KID, claims=_claims(tenant="acme globex")), "bad_tenant"),
    ("tenant_too_long", lambda: mint(KEY_A, KID, claims=_claims(tenant=_LONG_TENANT)), "bad_tenant"),
    ("tenant_empty", lambda: mint(KEY_A, KID, claims=_claims(tenant="")), "bad_tenant"),
    ("tenant_number", lambda: mint(KEY_A, KID, claims=_claims(tenant=7)), "bad_tenant"),
    ("tenant_list", lambda: mint(KEY_A, KID, claims=_claims(tenant=["acme", "globex"])), "bad_tenant"),
    ("tenant_missing", lambda: mint(KEY_A, KID, claims=_claims(tenant=DROP)), "missing_claim"),
    # --- scopes (refused whole, never filtered)
    ("scope_unknown", lambda: mint(KEY_A, KID, claims=_claims(scopes=["campaigns:admin"])), "bad_scopes"),
    ("scope_engine_name_not_web_name", lambda: mint(KEY_A, KID, claims=_claims(scopes=["write"])), "bad_scopes"),
    ("scope_mixed_good_and_bad", lambda: mint(KEY_A, KID, claims=_claims(scopes=["campaigns:read", "superuser"])), "bad_scopes"),
    ("scope_empty", lambda: mint(KEY_A, KID, claims=_claims(scopes=[])), "bad_scopes"),
    ("scope_string", lambda: mint(KEY_A, KID, claims=_claims(scopes="campaigns:write")), "bad_scopes"),
    ("scope_missing", lambda: mint(KEY_A, KID, claims=_claims(scopes=DROP)), "missing_claim"),
]

@pytest.mark.parametrize("name,make,reason", MATRIX, ids=[m[0] for m in MATRIX])
def test_refused_token_is_the_same_generic_401(main, world, name, make, reason):
    token = make()
    _SENT.append(token)
    _refused(main, token, reason)


def test_no_authorization_header_is_authentication_required(main, world):
    r = main.request("GET", "/api/latest")
    assert r.status_code == 401 and r.json() == {"error": "authentication required", "code": "unauthenticated"}


def test_lowercase_bearer_scheme_is_not_accepted(main, world):
    r = main.request("GET", "/api/latest", headers={"Authorization": f"bearer {ACME.token()}"})
    assert r.status_code == 401 and r.json() == {"error": "authentication required", "code": "unauthenticated"}


@pytest.mark.parametrize("name,make,reason", [
    ("tampered_payload", lambda: _swap_payload(_valid_token(), tenant="globex"), "bad_signature"),
    ("expired", lambda: mint(KEY_A, KID, claims=_claims(iat=int(time.time()) - 300, nbf=int(time.time()) - 300, exp=int(time.time()) - 240)), "expired"),
    ("bad_tenant", lambda: mint(KEY_A, KID, claims=_claims(tenant="default")), "bad_tenant"),
    ("unknown_kid", lambda: mint(KEY_A, "k-rogue", claims=_claims()), "unknown_kid"),
])
def test_write_route_refuses_bad_tokens_before_touching_data(main, world, name, make, reason):
    before = main.request("GET", "/api/campaign", auth=ACME).json()
    token = make()
    _SENT.append(token)
    _refused(main, token, reason, path="/api/campaign/start", method="POST", json_body=_RECIPE)
    assert main.request("GET", "/api/campaign", auth=ACME).json() == before


def test_every_refusal_body_is_identical_across_reasons(main, world):
    """No reason, claim name, tenant or key id reaches the caller: the body is
    the same bytes for every refused token."""
    bodies = set()
    for _name, make, _reason in MATRIX[:: max(1, len(MATRIX) // 12)]:
        token = make()
        _SENT.append(token)
        r = main.request("GET", "/api/latest", auth=token)
        assert r.status_code == 401
        bodies.add(r.text)
    assert bodies == {json.dumps(GENERIC_BODY, separators=(",", ":"))}


# --- scope escalation -------------------------------------------------------------- #

def test_read_only_token_cannot_write(main, world):
    r = main.request("POST", "/api/campaign/start", auth=READ_ONLY, json=_RECIPE)
    assert r.status_code == 403 and r.json() == {"error": "requires 'write' scope", "code": "forbidden"}
    r = main.request("POST", "/api/run", auth=READ_ONLY,
                     files={"file": ("x.csv", _sheet(3), "text/csv")})
    assert r.status_code == 403 and r.json() == {"error": "requires 'write' scope", "code": "forbidden"}
    r = main.request("POST", "/api/campaign/reanalyze", auth=READ_ONLY)
    assert r.status_code == 403


def test_write_only_token_cannot_read(main, world):
    """`campaigns:write` does not imply read (docs/ENGINE_TOKEN.md)."""
    r = main.request("GET", "/api/latest", auth=WRITE_ONLY)
    assert r.status_code == 403 and r.json() == {"error": "requires 'read' scope", "code": "forbidden"}
    assert main.request("GET", "/api/campaign", auth=WRITE_ONLY).status_code == 403


def test_admin_only_token_grants_neither_read_nor_write(main, world):
    """`org:admin` maps to `admin`, which no route requires today: on its own it
    is no access at all, and it is tenant-scoped like everything else."""
    assert main.request("GET", "/api/latest", auth=ADMIN_ONLY).status_code == 403
    assert main.request("POST", "/api/campaign/start", auth=ADMIN_ONLY, json=_RECIPE).status_code == 403


def test_admin_scope_is_tenant_scoped(main, world):
    everything = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-acme",
                     scopes=("campaigns:read", "campaigns:write", "org:admin"))
    a_latest = main.request("GET", "/api/latest", auth=ACME).json()
    assert main.request("GET", "/api/latest", auth=everything).json() == a_latest
    assert main.request("GET", "/api/latest", auth=GLOBEX).json()["dataset"] == "globex-runs.csv"


def test_scope_refusal_detail_names_only_the_scope(main, world):
    r = main.request("POST", "/api/campaign/start", auth=READ_ONLY, json=_RECIPE)
    assert r.text == '{"error":"requires \'write\' scope","code":"forbidden"}'
    assert "acme" not in r.text and "exp" not in r.text


# --- deny-list (KALOS_AUTH_DENY_SUBJECTS on the live server) ------------------- #

def test_denied_subject_and_denied_session_are_refused(main, world):
    _refused(main, BANNED_SUB.token(), "denied")
    _refused(main, REVOKED_SID.token(), "denied")


def test_denial_is_per_subject_or_session_and_nothing_else(main, world):
    assert main.request("GET", "/api/latest", auth=ACME).status_code == 200
    # same user, a different session: not denied (sid:sess-revoked only)
    other = Who(KEY_A, KID, tenant="acme", sub="user-acme", sid="sess-fresh")
    assert main.request("GET", "/api/latest", auth=other).status_code == 200


def test_denied_subject_on_write_route_is_refused_too(main, world):
    _refused(main, BANNED_SUB.token(), "denied", path="/api/campaign/start",
             method="POST", json_body=_RECIPE)


# --- static tokens -------------------------------------------------------------- #

def test_service_account_static_token_reaches_only_its_own_tenant(main, world):
    r = main.request("GET", "/api/latest", auth=SVC_TOKEN)
    assert r.status_code == 200 and r.json() == {"has_data": False}
    assert main.request("POST", "/api/campaign/start", auth=SVC_TOKEN, json=_RECIPE).status_code == 400


def test_static_token_ignores_tenant_named_in_headers_and_query(main, world):
    """A static token resolves to ITS record's tenant (svc-ops). Naming acme or
    globex in headers or query changes nothing."""
    r = main.request("GET", "/api/latest?tenant=acme", auth=SVC_TOKEN,
                     headers={"X-Tenant": "acme", "X-Kalos-Tenant": "globex"})
    assert r.status_code == 200 and r.json() == {"has_data": False}


def test_unprovisioned_static_token_is_refused_when_jwt_is_on(main, world):
    """JWT mode is on: a bearer that is neither an engine token nor a provisioned
    service-account token gets the generic 401 (no open or fallback access)."""
    _refused(main, "not-a-provisioned-token", "no_match")
    _refused(main, "x" * 40, "no_match")


def test_two_dot_static_lookalike_falls_back_and_is_refused(main, world):
    _refused(main, "abc.def.ghi", "no_match")


def test_duplicate_authorization_headers_never_mix_tenants(main, world):
    """Two Authorization headers: only ONE credential may act. The answer must be
    exactly the answer of one of the two credentials alone (observed: the first
    header wins), never a blend of both tenants' data."""
    svc_only = main.request("GET", "/api/latest", auth=SVC_TOKEN).json()
    acme_only = main.request("GET", "/api/latest", auth=ACME).json()
    assert svc_only == {"has_data": False} and acme_only["has_data"] is True
    for pair in ([SVC_TOKEN, ACME.token()], [ACME.token(), SVC_TOKEN]):
        with httpx.Client(base_url=main.base, timeout=60) as c:
            r = c.get("/api/latest", headers=[("Authorization", f"Bearer {t}") for t in pair])
        _SENT.extend(pair)
        assert r.status_code == 200 and r.json() in (svc_only, acme_only), r.text


# --- production surface ----------------------------------------------------- #

@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_production_serves_no_interactive_docs(main, path):
    assert main.request("GET", path).status_code == 404


@pytest.mark.parametrize("path", ["/api/single", "/api/multi"])
def test_production_does_not_register_demo_routes_even_with_a_valid_token(main, path):
    r = main.request("GET", path, auth=ACME, params={"rounds": 1, "q": 1})
    assert r.status_code == 404


# --- logs never contain token material ----------------------------------------- #

def test_engine_logs_never_contain_a_token_or_its_segments(main, world):
    """Every token sent to this server, and each of its three segments, must be
    absent from the complete log; the log must still show the refusals."""
    for _name, make, _reason in MATRIX[:: max(1, len(MATRIX) // 15)]:
        token = make()
        _SENT.append(token)
        main.request("GET", "/api/latest", auth=token)
    main.request("GET", "/api/latest", auth=ACME)
    for token in (mint(KEY_ATTACKER, KID, claims=_claims()), "a.b.c", "not-a-provisioned-token"):
        _SENT.append(token)
        main.request("GET", "/api/latest", auth=token)
    log = main.log_text()
    assert "kalos auth: rejected jwt reason=bad_signature" in log
    assert "kalos auth: rejected static token reason=no_match" in log
    for token in set(_SENT):
        parts = token.split(".")
        if len(parts) != 3 or len(token) < 40:
            continue  # short garbage can match by accident; real tokens cannot
        assert token not in log
        for seg in parts:
            if len(seg) >= 16:  # an empty or tiny segment (alg none) matches anything
                assert seg not in log
    assert SVC_TOKEN not in log
    assert "Bearer" not in log


# --- key rotation on a live server (two kids, atomic file replace) -------------- #

@pytest.fixture(scope="module")
def rotation(tmp_path_factory) -> Iterator[LiveServer]:
    root = tmp_path_factory.mktemp("rotation")
    jwks = root / "jwks.json"
    _write_jwks(jwks, {"k-old": KEY_ROTATE_OLD})
    yield from _start_server(root, jwks, {})


def test_jwks_rotation_with_two_kids_then_fail_closed_on_a_live_server(rotation):
    """One ordered scenario on one server: the file is replaced atomically, the
    engine picks up each change on the next request, and a broken file gives a
    503 (once logged) until it is fixed."""
    old = Who(KEY_ROTATE_OLD, "k-old", tenant="acme", sub="user-acme", sid="sess-1")
    new = Who(KEY_ROTATE_NEW, "k-new", tenant="globex", sub="user-globex", sid="sess-2")
    assert rotation.request("GET", "/api/latest", auth=old).status_code == 200

    # Add the new key next to the old one: both verify, each by its own kid.
    _write_jwks(rotation.jwks, {"k-old": KEY_ROTATE_OLD, "k-new": KEY_ROTATE_NEW})
    assert rotation.request("GET", "/api/latest", auth=new).status_code == 200
    assert rotation.request("GET", "/api/latest", auth=old).status_code == 200

    # A token naming the NEW kid but signed by the OLD key is refused: the
    # verifier picks one key by kid and never tries the others.
    swapped = mint(KEY_ROTATE_OLD, "k-new", claims=_claims(tenant="globex"))
    _refused(rotation, swapped, "bad_signature")

    # Retire the old key: its tokens stop on the very next request.
    _write_jwks(rotation.jwks, {"k-new": KEY_ROTATE_NEW})
    _refused(rotation, old.token(), "unknown_kid")
    assert rotation.request("GET", "/api/latest", auth=new).status_code == 200

    # A broken file: every request is a 503, logged once, and nothing is served
    # from the previously loaded keys.
    rotation.jwks.write_text("{not json")
    mark = rotation.log_mark()
    for _ in range(2):
        r = rotation.request("GET", "/api/latest", auth=new)
        assert r.status_code == 503 and r.text.count("authentication unavailable") == 1
    assert rotation.log_text()[mark:].count("engine-token (JWT) configuration is invalid") == 1
    # Fixed: the same key works again.
    _write_jwks(rotation.jwks, {"k-new": KEY_ROTATE_NEW})
    assert rotation.request("GET", "/api/latest", auth=new).status_code == 200


# --- leeway and time boundaries (production app, injected clock) ----------------- #

_T = 1_900_000_000

_BOUNDARY_SCRIPT = """
import json, logging, os, re, sys
from fastapi.testclient import TestClient
from kalos.portal.app import app
from kalos.portal.auth import Authenticator, get_authenticator

class _Cap(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []
    def emit(self, record):
        self.msgs.append(record.getMessage())

def _fixed(auth):
    # no parameters: FastAPI must not read a default as a query parameter
    return lambda: auth


cap = _Cap()
logging.getLogger("kalos.portal").addHandler(cap)
logging.getLogger("kalos.portal").setLevel(logging.INFO)
client = TestClient(app)
out = []
for case in json.loads(os.environ["BOUNDARY_CASES"]):
    auth = Authenticator(clock=lambda t=case["clock"]: float(t))
    app.dependency_overrides[get_authenticator] = _fixed(auth)
    cap.msgs.clear()
    r = client.get("/api/latest", headers={"Authorization": "Bearer " + case["token"]})
    reasons = [re.search(r"reason=(\\S+)", m).group(1) for m in cap.msgs if "reason=" in m]
    out.append({"name": case["name"], "status": r.status_code, "reasons": reasons,
                "body": r.text})
print("RESULT " + json.dumps(out))
"""


def _at(now: int, **claims: Any) -> dict:
    base = {"iat": _T, "nbf": _T, "exp": _T + 60}
    base.update(claims)
    return _claims(now=now, **base)


def test_leeway_boundaries_in_production_with_an_injected_clock(tmp_path):
    key = Ed25519PrivateKey.generate()
    jwks = tmp_path / "jwks.json"
    _write_jwks(jwks, {KID: key})
    m = lambda claims: mint(key, KID, claims=claims)  # noqa: E731
    cases = [
        # exp: expired when exp <= now - 30 (leeway 30)
        ("exp+29 accepted", m(_at(_T, iat=_T - 60, nbf=_T - 60, exp=_T)), _T + 29, 200, []),
        ("exp+30 refused", m(_at(_T, iat=_T - 60, nbf=_T - 60, exp=_T)), _T + 30, 401, ["expired"]),
        ("exp+31 refused", m(_at(_T, iat=_T - 60, nbf=_T - 60, exp=_T)), _T + 31, 401, ["expired"]),
        ("exp exactly now accepted", m(_at(_T, iat=_T - 60, nbf=_T - 60, exp=_T)), _T, 200, []),
        # nbf: not yet valid when nbf > now + 30
        ("nbf-30 accepted", m(_at(_T, iat=_T + 60, nbf=_T + 60, exp=_T + 120)), _T + 30, 200, []),
        ("nbf-31 refused", m(_at(_T, iat=_T + 60, nbf=_T + 60, exp=_T + 120)), _T + 29, 401, ["not_yet_valid"]),
        # iat: issued in the future when iat > now + 30 (nbf already passed)
        ("iat-30 accepted", m(_at(_T, iat=_T + 60, nbf=_T, exp=_T + 120)), _T + 30, 200, []),
        ("iat-31 refused", m(_at(_T, iat=_T + 60, nbf=_T, exp=_T + 120)), _T + 29, 401, ["issued_in_future"]),
        # ttl: exp - iat <= 120 whatever the signature says
        ("ttl 120 accepted", m(_at(_T, iat=_T, nbf=_T, exp=_T + 120)), _T, 200, []),
        ("ttl 121 refused", m(_at(_T, iat=_T, nbf=_T, exp=_T + 121)), _T, 401, ["ttl_too_long"]),
    ]
    payload = [{"name": n, "token": t, "clock": c} for n, t, c, _s, _r in cases]
    env = _prod_env(tmp_path / "state", jwks, {"BOUNDARY_CASES": json.dumps(payload)})
    (tmp_path / "state").mkdir(exist_ok=True)
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_BOUNDARY_SCRIPT)],
                          env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    result = json.loads(next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))[7:])
    got = {r["name"]: r for r in result}
    for name, _t, _c, status, reasons in cases:
        assert got[name]["status"] == status, (name, got[name])
        assert got[name]["reasons"] == reasons, (name, got[name])
        if status == 401:
            assert got[name]["body"] == json.dumps(GENERIC_BODY, separators=(",", ":"))


# --- the shared web vectors, through the HTTP app -------------------------------- #

@pytest.mark.parametrize("vec", VECTORS["valid"], ids=lambda v: v["name"])
def test_web_vector_valid_token_is_accepted_over_http(tmp_path, monkeypatch, vec):
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal.app import app
    from kalos.portal.auth import Authenticator, get_authenticator

    spec = VECTORS["spec"]
    jwks = tmp_path / "vector-jwks.json"
    jwks.write_text(json.dumps({"keys": [VECTORS["keys"]["primary"]["public_jwk"]]}))
    for var in ("KALOS_ENV", "KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_AUTH_DENY_SUBJECTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KALOS_JWT_PUBLIC_KEYS_FILE", str(jwks))
    monkeypatch.setenv("KALOS_JWT_ISSUER", spec["iss"])
    monkeypatch.setenv("KALOS_JWT_AUDIENCE", spec["aud"])
    monkeypatch.setenv("KALOS_JWT_LEEWAY_S", str(spec["recommended_clock_leeway_seconds"]))
    monkeypatch.setenv("KALOS_JWT_MAX_TTL_S", str(spec["max_ttl_seconds"]))
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    auth = Authenticator(clock=lambda: float(vec["verify_at"]))
    app.dependency_overrides[get_authenticator] = lambda: auth
    try:
        r = TestClient(app).get("/api/latest", headers={"Authorization": f"Bearer {vec['token']}"})
        assert r.status_code == 200, (vec["name"], r.text)
        assert r.json() == {"has_data": False}  # the claimed tenant has no data here
    finally:
        app.dependency_overrides.pop(get_authenticator, None)


@pytest.mark.parametrize("vec", VECTORS["invalid"], ids=lambda v: v["name"])
def test_web_vector_invalid_token_is_refused_over_http(tmp_path, monkeypatch, vec):
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal.app import app
    from kalos.portal.auth import Authenticator, get_authenticator

    spec = VECTORS["spec"]
    jwks = tmp_path / "vector-jwks.json"
    jwks.write_text(json.dumps({"keys": [VECTORS["keys"]["primary"]["public_jwk"]]}))
    for var in ("KALOS_ENV", "KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_AUTH_DENY_SUBJECTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KALOS_JWT_PUBLIC_KEYS_FILE", str(jwks))
    monkeypatch.setenv("KALOS_JWT_ISSUER", spec["iss"])
    monkeypatch.setenv("KALOS_JWT_AUDIENCE", spec["aud"])
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    auth = Authenticator(clock=lambda: float(vec["verify_at"]))
    app.dependency_overrides[get_authenticator] = lambda: auth
    try:
        r = TestClient(app).get("/api/latest", headers={"Authorization": f"Bearer {vec['token']}"})
        assert r.status_code == 401, (vec["name"], r.status_code)
        assert r.json() == GENERIC_BODY
    finally:
        app.dependency_overrides.pop(get_authenticator, None)


# --- startup refusals that only a fresh process can show ------------------------ #

def test_production_refuses_to_start_with_the_published_vector_keys(tmp_path):
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [VECTORS["keys"]["primary"]["public_jwk"]]}))
    env = _prod_env(tmp_path, jwks, {})
    proc = subprocess.run([sys.executable, "-c", "import kalos.portal.app"], env=env,
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0
    assert "ProductionConfigError" in proc.stderr
