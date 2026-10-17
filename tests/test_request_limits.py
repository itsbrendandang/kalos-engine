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
