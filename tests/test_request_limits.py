"""Request limits (P0 hardening): a body-size cap enforced by middleware before
any route parses the body, and Pydantic bounds on the /api/campaign/start and
/api/campaign/result bodies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app, get_campaign_store  # noqa: E402
from kalos.portal.campaign import CampaignStore  # noqa: E402
from kalos.portal.limits import (  # noqa: E402
    DEFAULT_MAX_REQUEST_BYTES,
    RequestSizeLimitMiddleware,
    max_request_bytes,
)
from kalos.portal.uploads import MAX_UPLOAD_BYTES  # noqa: E402


@pytest.fixture(autouse=True)
def _open_mode(monkeypatch):
    for var in ("KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE", "KALOS_ALLOW_OPEN_ACCESS", "KALOS_ENV"):
        monkeypatch.delenv(var, raising=False)


def _df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({"Methanol": m.round(3), "pH": ph.round(2),
                         "lipase_titer": (1.5 * m + rng.normal(0, 0.1, n)).round(3)})


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = CampaignStore(tmp_path)
    monkeypatch.setattr(portal, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal, "_LATEST", {})
    monkeypatch.setattr(campaign_module, "_STORE", s)
    app.dependency_overrides[get_campaign_store] = lambda: s
    try:
        yield s
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


@pytest.fixture
def client(store):
    return TestClient(app)


def _proposal(**over):
    p = {"recipe": {"Methanol": 2.0, "pH": 6.0}, "pred": 1.0, "std": 0.1,
         "mode": "explore", "reason": "x"}
    p.update(over)
    return p


# --- body size cap: Content-Length checked before parsing ------------------- #

def test_oversized_json_body_is_413_before_parsing(client, store, monkeypatch):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    calls: list = []
    monkeypatch.setattr(CampaignStore, "start", lambda *a, **k: calls.append(1))
    body = b'{"recipes": [' + b" " * (DEFAULT_MAX_REQUEST_BYTES + 10) + b"]}"
    r = client.post("/api/campaign/start", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json() == {"error": "The request body is too large."}
    assert calls == []  # the route never ran


def test_a_normal_json_body_is_unaffected(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": [_proposal()]})
    assert r.status_code == 200


def test_upload_route_gets_the_upload_cap_not_the_json_cap(client):
    """A body just over MAX_UPLOAD_BYTES still reaches the upload guard (which
    answers its own 400 message); one far over the file cap + multipart
    allowance is refused at the door with 413."""
    just_over = b"a,b,c\n" + b"1,2,3\n" * (MAX_UPLOAD_BYTES // 6 + 10)
    r = client.post("/api/run", files={"file": ("runs.csv", just_over, "text/csv")})
    assert r.status_code == 400
    assert r.json()["error"] == "The uploaded file is too large."

    way_over = b"0" * (MAX_UPLOAD_BYTES + 3 * 1024 * 1024)
    r = client.post("/api/run", files={"file": ("runs.csv", way_over, "text/csv")})
    assert r.status_code == 413


@pytest.mark.parametrize("value", [b"abc", b"-1", b"1, 2"])
def test_malformed_content_length_is_400(value):
    seen: list = []

    async def inner(scope, receive, send):  # pragma: no cover - must not run
        seen.append(scope)

    mw = RequestSizeLimitMiddleware(inner, default_cap=10, upload_cap=10, upload_paths=())
    sent: list = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    import asyncio

    asyncio.run(mw({"type": "http", "path": "/x", "headers": [(b"content-length", value)]}, receive, send))
    assert sent[0]["status"] == 400
    assert seen == []


def _mini_app(cap: int) -> TestClient:
    mini = FastAPI()

    @mini.post("/echo")
    async def echo(request: Request) -> dict:
        return {"n": len(await request.body())}

    mini.add_middleware(RequestSizeLimitMiddleware, default_cap=cap, upload_cap=cap, upload_paths=())
    return TestClient(mini)


def test_chunked_body_without_content_length_is_counted_and_refused():
    tc = _mini_app(100)

    def chunks():  # a generator body is sent chunked, with no Content-Length
        for _ in range(10):
            yield b"x" * 50

    r = tc.post("/echo", content=chunks())
    assert r.status_code == 413
    ok = tc.post("/echo", content=iter([b"x" * 40, b"y" * 40]))
    assert ok.status_code == 200 and ok.json() == {"n": 80}


def test_max_request_bytes_env(monkeypatch):
    monkeypatch.setenv("KALOS_MAX_REQUEST_BYTES", "")
    assert max_request_bytes() == DEFAULT_MAX_REQUEST_BYTES
    monkeypatch.setenv("KALOS_MAX_REQUEST_BYTES", "2048")
    assert max_request_bytes() == 2048
    for bad in ("0", "-5", "lots"):
        monkeypatch.setenv("KALOS_MAX_REQUEST_BYTES", bad)
        with pytest.raises(ValueError):
            max_request_bytes()


# --- /api/campaign/start body bounds ---------------------------------------- #

def test_at_most_64_recipes_per_start(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    assert client.post("/api/campaign/start", json={"recipes": [_proposal()] * 64}).status_code == 200
    r = client.post("/api/campaign/start", json={"recipes": [_proposal()] * 65})
    assert r.status_code == 400
    assert "recipes" in r.json()["error"]
    assert store.summary(tenant="default")["n_awaiting"] == 64  # the 65 were not appended


@pytest.mark.parametrize("recipe", [
    {"Methanol": True},                 # bool is not a measurement
    {"Methanol": "2" * 513},            # over-long string value
    {"M" * 513: 1.0},                   # over-long key
    {"": 1.0},                          # empty key
    {"Methanol": [1, 2]},               # not a scalar
    {"Methanol": {"nested": 1}},
])
def test_recipe_values_and_keys_are_bounded(client, store, recipe):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": [_proposal(recipe=recipe)]})
    assert r.status_code == 400
    assert "error" in r.json()
    assert store.summary(tenant="default")["pending"] == []


@pytest.mark.parametrize("raw", ['NaN', 'Infinity', '-Infinity'])
def test_non_finite_recipe_numbers_are_refused(client, store, raw):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    body = '{"recipes": [{"recipe": {"Methanol": %s}}]}' % raw
    r = client.post("/api/campaign/start", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize("field, value", [
    ("mode", "m" * 513), ("reason", "r" * 513), ("pred", "high"), ("std", True),
])
def test_proposal_metadata_is_bounded(client, store, field, value):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": [_proposal(**{field: value})]})
    assert r.status_code == 400


def test_strings_up_to_512_chars_are_accepted(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": [_proposal(reason="r" * 512)]})
    assert r.status_code == 200


def test_recipe_keys_must_be_campaign_feature_columns(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    for recipe in ({"Methanol": 1.0, "secret_col": 2.0}, {"lipase_titer": 9.9}):
        r = client.post("/api/campaign/start", json={"recipes": [_proposal(recipe=recipe)]})
        assert r.status_code == 400
        assert "secret_col" not in r.text  # caller input is not echoed
    assert store.summary(tenant="default")["pending"] == []


def test_validation_errors_mask_dict_keys(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": [_proposal(recipe={"evil<script>": [1]})]})
    assert r.status_code == 400
    assert "evil" not in r.text


def test_extra_proposal_fields_are_ignored_not_stored(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": [_proposal(vals=[1, 2], p_feasible=0.9)]})
    assert r.status_code == 200
    assert "vals" not in r.json()["started"][0]


def test_real_engine_proposals_with_a_categorical_still_start(client, store):
    """The engine's own proposals (including a categorical dimension decoded
    to its label, and extra annotation fields) must pass the new bounds."""
    rng = np.random.default_rng(0)
    n = 30
    m = rng.uniform(0, 4, n)
    df = pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": m.round(3),
        "pH": rng.uniform(5, 7, n).round(2),
        "lipase_titer": (1.5 * m + rng.normal(0, 0.3, n)).round(3),
    })
    result = portal._analyze(df, "lipase_titer")
    store.seed(df, result["target"], result["proposal_features"], tenant="default")
    r = client.post("/api/campaign/start", json={"recipes": result["proposals"]})
    assert r.status_code == 200, r.text
    assert len(r.json()["started"]) == len(result["proposals"])


# --- /api/campaign/result body bounds --------------------------------------- #

@pytest.mark.parametrize("body", [
    {"id": "x" * 513, "value": 1.0},
    {"id": "", "value": 1.0},
    {"id": "abc", "value": True},
    {"id": "abc", "value": "7.3"},
    {"id": "abc"},
])
def test_result_body_is_bounded(client, store, body):
    r = client.post("/api/campaign/result", json=body)
    assert r.status_code == 400
    assert "error" in r.json()


def test_validation_400_is_scoped_to_campaign_routes(client):
    # other routes keep FastAPI's default 422 envelope
    r = client.get("/api/single", params={"rounds": 0})
    assert r.status_code == 422
    assert "detail" in r.json()


# --- review fixes: chunked 413 on the real app, huge Content-Length, root_path #

def _chunks(n: int, size: int):
    for _ in range(n):
        yield b"x" * size


def test_chunked_over_cap_json_body_is_413_on_the_real_app(client, store, monkeypatch):
    """The real app has a BaseHTTPMiddleware (the open-access guard) and
    FastAPI body parsing between this middleware and the route; neither may
    turn the over-cap stream into a generic 400."""
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    calls: list = []
    monkeypatch.setattr(CampaignStore, "start", lambda *a, **k: calls.append(1))
    r = client.post("/api/campaign/start", content=_chunks(30, 100_000),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert r.json() == {"error": "The request body is too large."}
    assert calls == []


def test_chunked_over_cap_upload_is_413_on_the_real_app(client):
    head = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="a.csv"\r\n'
            b"Content-Type: text/csv\r\n\r\n")

    def body():
        yield head
        yield from _chunks(60, 500_000)  # ~30 MB > upload cap
        yield b"\r\n--b--\r\n"

    r = client.post("/api/run", content=body(),
                    headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert r.json() == {"error": "The request body is too large."}


def test_chunked_under_cap_json_body_still_works_on_the_real_app(client, store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    import json as _json

    raw = _json.dumps({"recipes": [_proposal()]}).encode()
    r = client.post("/api/campaign/start", content=iter([raw[:10], raw[10:]]),
                    headers={"content-type": "application/json"})
    assert r.status_code == 200


@pytest.mark.parametrize("value", [b"9" * 20, b"9" * 5000, b"0" * 30 + b"1" * 20])
def test_content_length_over_19_digits_is_413_not_a_crash(value):
    seen: list = []

    async def inner(scope, receive, send):  # pragma: no cover - must not run
        seen.append(scope)

    mw = RequestSizeLimitMiddleware(inner, default_cap=10, upload_cap=10, upload_paths=())
    sent: list = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    import asyncio

    asyncio.run(mw({"type": "http", "path": "/x", "headers": [(b"content-length", value)]}, receive, send))
    assert sent[0]["status"] == 413
    assert seen == []


def test_leading_zero_content_length_under_cap_is_allowed():
    tc = _mini_app(100)
    r = tc.post("/echo", content=b"x" * 5, headers={"content-length": "0000000000000000000000005"})
    assert r.status_code == 200 and r.json() == {"n": 5}


def test_upload_paths_match_with_root_path_stripped():
    mw = RequestSizeLimitMiddleware(lambda *a: None, default_cap=1, upload_cap=99,
                                    upload_paths={"/api/run"})
    from kalos.portal.limits import route_path

    assert route_path({"path": "/engine/api/run", "root_path": "/engine"}) == "/api/run"
    assert route_path({"path": "/api/run/", "root_path": ""}) == "/api/run"
    assert route_path({"path": "/engine", "root_path": "/engine"}) == "/"
    assert mw.cap_for(route_path({"path": "/engine/api/run", "root_path": "/engine"})) == 99


@pytest.mark.parametrize("path,root,expected", [
    ("/engineX/api/run", "/engine", "/engineX/api/run"),  # not a segment boundary
    ("/engine/api/run", "/engine/", "/api/run"),
    ("/api/run", "/engine", "/api/run"),  # server that does not prefix path
    ("/engine/", "/engine", "/"),
])
def test_route_path_strips_root_path_only_on_a_segment_boundary(path, root, expected):
    from starlette._utils import get_route_path

    from kalos.portal.limits import route_path

    assert route_path({"path": path, "root_path": root}) == expected
    # agrees with Starlette's own routing rule (modulo the trailing slash)
    assert route_path({"path": path, "root_path": root}) == (
        get_route_path({"path": path, "root_path": root.rstrip("/")}).rstrip("/") or "/"
    )


def test_campaign_validation_envelope_under_a_root_path(store):
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="default")
    tc = TestClient(app, root_path="/engine")
    r = tc.post("/engine/api/campaign/start", json={"recipes": "nope"})
    assert r.status_code == 400 and "error" in r.json()


def test_upload_under_a_root_path_gets_the_upload_cap():
    """Mounted under a prefix (uvicorn --root-path), the upload route must
    still get the upload cap, not the 1 MiB JSON cap."""
    tc = TestClient(app, root_path="/engine")
    payload = b"a,b,c\n" + b"1,2,3\n" * 400_000  # ~2.4 MB: over 1 MiB, under the upload cap
    r = tc.post("/engine/api/run", files={"file": ("runs.csv", payload, "text/csv")})
    assert r.status_code != 413


# --- review fixes: an upload with no bearer token is refused before its body - #

def _enforce(monkeypatch):
    import hashlib
    import json as _json

    monkeypatch.setenv("KALOS_AUTH_TOKENS", _json.dumps([{
        "token_sha256": hashlib.sha256(b"tok").hexdigest(),
        "subject": "s", "tenant": "acme", "scopes": ["read", "write"],
    }]))


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer "}])
def test_upload_without_a_bearer_is_401_before_the_body_is_read(monkeypatch, headers):
    _enforce(monkeypatch)
    read: list = []

    def body():
        read.append(1)
        yield b"--b\r\n"

    r = TestClient(app).post("/api/run", content=body(),
                             headers={"content-type": "multipart/form-data; boundary=b", **headers})
    assert r.status_code == 401
    assert r.json() == {"detail": "authentication required"}
    assert read == []  # the body generator was never pulled


def test_upload_precheck_lets_a_bearer_through_to_real_verification(monkeypatch):
    _enforce(monkeypatch)
    r = TestClient(app).post("/api/run", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")},
                             headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert r.json() == {"detail": "invalid or expired token"}  # the dependency's answer, not the precheck's


def test_upload_precheck_is_inert_in_open_mode(client):
    r = client.post("/api/run", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code != 401
