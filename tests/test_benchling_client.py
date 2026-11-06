"""Benchling client: auth modes, 15-minute token refresh, the 60/30 s token bucket,
429 Retry-After handling, 5xx policy, EARLY-ACCESS only on beta paths,
pagination, typed errors, and secrets kept out of messages and logs."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
from benchling_fake import BASE, STUDY_ID, FakeBenchling, bioprocess_study

from kalos.integrations.benchling.client import (
    AppCredentials,
    BenchlingAuthError,
    BenchlingClient,
    BenchlingConfigError,
    BenchlingForbidden,
    BenchlingInvalidRequest,
    BenchlingNotFound,
    BenchlingRateLimited,
    BenchlingServerError,
    ClientCredentialsToken,
    DelegatedToken,
    TokenBucket,
    is_beta_path,
    normalize_base_url,
)

FIXTURES = Path(__file__).parent / "fixtures" / "benchling"


class Clock:
    """A fake monotonic clock whose sleep advances time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


def _client(fake: FakeBenchling, clock: Clock | None = None, **kw) -> BenchlingClient:
    clock = clock or Clock()
    creds = AppCredentials(fake.client_id, fake.client_secret)
    bucket = kw.pop("bucket", None) or TokenBucket(60, 30.0, clock=clock, sleep=clock.sleep)
    return BenchlingClient(BASE, ClientCredentialsToken(creds, clock=clock),
                           transport=fake.transport(), bucket=bucket, sleep=clock.sleep, **kw)


# --- auth ------------------------------------------------------------------------ #


def test_client_credentials_token_is_fetched_once_and_sent_as_bearer():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=1, replicates=1)
    with _client(fake) as c:
        c.get("/api/v3/study/items")
        c.get("/api/v3/study/items")
    token_calls = [x for x in fake.calls if x.path == "/oauth/token"]
    assert len(token_calls) == 1
    assert token_calls[0].body["grant_type"] == "client_credentials"
    api = [x for x in fake.calls if x.path != "/oauth/token"]
    assert all(x.headers["authorization"] == "Bearer app-token-1" for x in api)


def test_token_is_refreshed_before_the_15_minute_expiry():
    fake = FakeBenchling()
    clock = Clock()
    with _client(fake, clock) as c:
        c.request("GET", "/api/v3/study/items")
        clock.now += 13 * 60  # within the 60 s margin? no: 13 min < 14 min
        c.request("GET", "/api/v3/study/items")
        assert fake.tokens_issued == ["app-token-1"]
        clock.now += 60 + 1  # now 14 min+ after issue: inside the refresh margin
        c.request("GET", "/api/v3/study/items")
        assert fake.tokens_issued == ["app-token-1", "app-token-2"]
    assert fake.calls[-1].headers["authorization"] == "Bearer app-token-2"


def test_a_401_gets_one_new_token_then_succeeds():
    fake = FakeBenchling()
    with _client(fake) as c:
        c.get("/api/v3/study/items")
        fake.revoked.add("app-token-1")
        c.get("/api/v3/study/items")
    assert fake.tokens_issued == ["app-token-1", "app-token-2"]


def test_wrong_app_secret_is_an_auth_error_without_the_secret_in_it(caplog):
    fake = FakeBenchling()
    creds = AppCredentials(fake.client_id, "wrong-secret-XYZ")
    caplog.set_level(logging.DEBUG)
    with BenchlingClient(BASE, ClientCredentialsToken(creds), transport=fake.transport(),
                         bucket=TokenBucket(60, 30.0)) as c, pytest.raises(BenchlingAuthError) as err:
        c.get("/api/v3/study/items")
    assert "wrong-secret-XYZ" not in str(err.value)
    assert "wrong-secret-XYZ" not in caplog.text
    assert "wrong-secret-XYZ" not in repr(creds) and "wrong-secret-XYZ" not in repr(c.auth)


def test_delegated_token_is_sent_as_is_and_never_hits_oauth():
    fake = FakeBenchling()
    fake.delegated_tokens["user-token-abc"] = "ent_alice"
    with BenchlingClient(BASE, DelegatedToken("user-token-abc"), transport=fake.transport(),
                         bucket=TokenBucket(60, 30.0)) as c:
        c.get("/api/v3/study/items")
        assert c.mode == "delegated"
    assert [x.path for x in fake.calls] == ["/api/v3/study/items"]
    assert fake.calls[0].headers["authorization"] == "Bearer user-token-abc"
    assert "user-token-abc" not in repr(DelegatedToken("user-token-abc"))


def test_expired_delegated_token_without_refresh_is_an_auth_error():
    fake = FakeBenchling()
    with BenchlingClient(BASE, DelegatedToken("stale-user-token"), transport=fake.transport(),
                         bucket=TokenBucket(60, 30.0)) as c, pytest.raises(BenchlingAuthError):
        c.get("/api/v3/study/items")
    assert fake.paths() == ["/api/v3/study/items"]  # no retry, no oauth call


def test_delegated_token_refresh_keeps_the_new_single_use_refresh_token():
    fake = FakeBenchling()
    seen: list[str] = []
    clock = Clock()
    tok = DelegatedToken("stale", refresh_token="refresh-0",
                         credentials=AppCredentials(fake.client_id, fake.client_secret),
                         on_refresh=seen.append, clock=clock)
    with BenchlingClient(BASE, tok, transport=fake.transport(), bucket=TokenBucket(60, 30.0)) as c:
        c.get("/api/v3/study/items")  # 401 -> refresh -> retry
    grants = [x.body["grant_type"] for x in fake.calls if x.path == "/oauth/token"]
    assert grants == ["refresh_token"]
    assert seen == ["refresh-1"]


# --- rate limit -------------------------------------------------------------------- #


def test_bucket_allows_60_requests_then_waits_for_the_first_token_to_return():
    clock = Clock()
    b = TokenBucket(60, 30.0, clock=clock, sleep=clock.sleep)
    for _ in range(60):
        assert b.acquire() == 0.0
    assert b.available == 0
    waited = b.acquire()
    assert waited == pytest.approx(30.0)
    # the other 59 tokens came back at the same instant
    for _ in range(59):
        assert b.acquire() == 0.0


def test_client_never_exceeds_60_requests_in_any_30_second_window():
    fake = FakeBenchling()
    clock = Clock()
    stamps: list[float] = []
    orig = fake.handle

    def stamped(req):
        stamps.append(clock.now)
        return orig(req)

    fake.handle = stamped  # type: ignore[method-assign]
    with _client(fake, clock) as c:
        for _ in range(150):
            c.get("/api/v3/study/items")
    for i, t in enumerate(stamps):
        window = [s for s in stamps[i:] if s < t + 30.0 - 1e-9]
        assert len(window) <= 60


def test_429_with_retry_after_pauses_the_bucket_then_retries():
    fake = FakeBenchling()
    fake.script["/api/v3/study/items"] = [(429, {"Retry-After": "4"}, {"title": "Too Many Requests"})]
    clock = Clock()
    with _client(fake, clock) as c:
        body = c.get("/api/v3/study/items")
        assert body["items"] == []
        assert c.rate_limited == 1
    assert 4.0 in clock.slept
    assert fake.paths().count("/api/v3/study/items") == 2


def test_429_v2_uses_x_rate_limit_reset():
    fake = FakeBenchling()
    fake.script["/api/v2/projects"] = [(429, {"x-rate-limit-reset": "7"}, {"error": {"message": "Rate limit exceeded."}})]
    clock = Clock()
    with _client(fake, clock) as c:
        c.get("/api/v2/projects")
    assert 7.0 in clock.slept


def test_429_without_header_backs_off_exponentially_and_gives_up_typed():
    fake = FakeBenchling()
    fake.script["/api/v3/study/items"] = [(429, {}, {"title": "busy"})] * 4
    clock = Clock()
    with _client(fake, clock, max_retries=3) as c, pytest.raises(BenchlingRateLimited) as err:
        c.get("/api/v3/study/items")
    assert err.value.status == 429
    waits = [s for s in clock.slept if s >= 1.0]
    assert len(waits) == 3 and waits == sorted(waits)  # 2^1, 2^2, 2^3 (+ jitter)


def test_5xx_retries_a_get_but_never_a_write():
    fake = FakeBenchling()
    fake.script["/api/v3/study/items"] = [(503, {}, {"title": "down"})]
    with _client(fake) as c:
        c.get("/api/v3/study/items")
    assert fake.paths().count("/api/v3/study/items") == 2
    fake.script["/api/v3/procedure-run-plan"] = [(500, {}, {"title": "boom"})]
    with _client(fake) as c, pytest.raises(BenchlingServerError):
        c.post("/api/v3/procedure-run-plan", {"studyId": STUDY_ID, "description": "x"})
    assert fake.paths("POST").count("/api/v3/procedure-run-plan") == 1


# --- headers, pagination, errors ----------------------------------------------------- #


def test_early_access_header_matches_the_spec_beta_list():
    spec = json.loads((FIXTURES / "v3_beta_ops.json").read_text())
    for op in spec["operations"]:
        concrete = op["path"].replace("{", "").replace("}", "")
        assert is_beta_path(concrete) is op["beta"], op
    for v2 in ("/api/v2/assay-runs", "/api/v2/result-transactions", "/api/v2/blobs"):
        assert not is_beta_path(v2)


def test_early_access_header_is_sent_only_on_beta_paths():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=1, replicates=1)
    with _client(fake) as c:
        c.get("/api/v3/study/items")
        c.get("/api/v3/procedure-run/items", {"study.eq": STUDY_ID})
        c.get("/api/v3/result/items", {"schema.eq": "x"})
        c.get("/api/v2/projects")
    seen = {x.path: x.headers.get("early-access") for x in fake.calls if x.path != "/oauth/token"}
    assert seen == {"/api/v3/study/items": None, "/api/v3/procedure-run/items": "true",
                    "/api/v3/result/items": None, "/api/v2/projects": None}


def test_paginate_follows_next_token_v3_and_v2():
    fake = bioprocess_study(FakeBenchling(), n_runs=7, conditions_per_run=1, replicates=1)
    for i in range(130):
        fake.projects.append({"id": f"src_{i:08d}", "name": f"p{i}"})
    with _client(fake) as c:
        runs = list(c.paginate("/api/v3/procedure-run/items", {"study.eq": STUDY_ID}, page_size=3))
        projects = list(c.paginate("/api/v2/projects", items_key="projects"))
    assert len(runs) == 7
    assert len(projects) == 131
    pages = [x for x in fake.calls if x.path == "/api/v3/procedure-run/items"]
    assert len(pages) == 3 and pages[1].params["nextToken"] == "3"
    assert all(x.params["pageSize"] == "100" for x in fake.calls if x.path == "/api/v2/projects")


def test_errors_are_typed_and_carry_benchling_title_not_the_request():
    fake = FakeBenchling()
    fake.script["/api/v3/procedure-run-plan"] = [
        (400, {}, {"type": "x", "title": "description too long", "detail": None, "status": 400, "instance": "/"}),
    ]
    fake.script["/api/v3/study/items"] = [(403, {}, {"title": "Forbidden"})]
    with _client(fake) as c:
        with pytest.raises(BenchlingInvalidRequest) as bad:
            c.post("/api/v3/procedure-run-plan", {"studyId": "s", "description": "SECRET-RECIPE"})
        assert bad.value.title == "description too long"
        assert "SECRET-RECIPE" not in str(bad.value)
        with pytest.raises(BenchlingForbidden):
            c.get("/api/v3/study/items")
        with pytest.raises(BenchlingNotFound):
            c.get("/api/v3/nothing-here")


def test_transport_errors_on_get_are_retried_then_typed():
    calls = {"n": 0}

    def flaky(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/oauth/token":
            return httpx.Response(200, json={"access_token": "t", "token_type": "Bearer", "expires_in": 900})
        calls["n"] += 1
        raise httpx.ConnectError("boom")

    clock = Clock()
    c = BenchlingClient(BASE, ClientCredentialsToken(AppCredentials("a", "b"), clock=clock),
                        transport=httpx.MockTransport(flaky), bucket=TokenBucket(60, 30.0, clock=clock,
                                                                                  sleep=clock.sleep),
                        sleep=clock.sleep, max_retries=2)
    from kalos.integrations.benchling.client import BenchlingTransportError

    with pytest.raises(BenchlingTransportError):
        c.get("/api/v3/study/items")
    assert calls["n"] == 3


@pytest.mark.parametrize("url", [
    "http://acme.benchling.com", "https://acme.benchling.com.evil.example", "https://evil.example",
    "https://acme.benchling.com:8443", "https://user:pw@acme.benchling.com", "https://acme.benchling.com/api",
    "https://a.b.benchling.com", "https://benchling.com",
])
def test_base_url_must_be_a_benchling_tenant(url):
    with pytest.raises(BenchlingConfigError):
        normalize_base_url(url)


def test_base_url_normalized():
    assert normalize_base_url(" https://ACME.benchling.com/ ") == "https://acme.benchling.com"


def test_paths_must_be_api_paths():
    fake = FakeBenchling()
    with _client(fake) as c, pytest.raises(BenchlingConfigError):
        c.get("https://evil.example/api/v3/x")


# --- review fixes: limiter per (tenant, host), bounded waits ------------------------------------ #


def test_buckets_are_per_kalos_tenant_and_host():
    from kalos.integrations.benchling.client import bucket_for, reset_buckets

    reset_buckets()
    a = bucket_for(BASE, "acme")
    assert bucket_for(BASE, "acme") is a
    assert bucket_for(BASE, "globex") is not a
    assert bucket_for("https://other.benchling.com", "acme") is not a
    for _ in range(60):
        a.acquire()
    assert bucket_for(BASE, "globex").available == 60  # acme used up only its own budget
    reset_buckets()


def test_acquire_raises_instead_of_waiting_past_the_deadline():
    from kalos.integrations.benchling.client import RateLimitWaitExceeded

    clock = Clock()
    b = TokenBucket(2, 30.0, clock=clock, sleep=clock.sleep)
    b.acquire()
    b.acquire()
    with pytest.raises(RateLimitWaitExceeded) as err:
        b.acquire(deadline=clock.now + 5.0)
    assert err.value.retry_after == pytest.approx(30.0)
    assert clock.slept == []  # it did not sleep
    b.pause(100.0)
    with pytest.raises(RateLimitWaitExceeded):
        b.acquire(deadline=clock.now + 50.0)


def test_client_gives_up_when_the_limiter_wait_exceeds_max_wait():
    from kalos.integrations.benchling.client import RateLimitWaitExceeded

    fake = FakeBenchling()
    clock = Clock()
    bucket = TokenBucket(3, 30.0, clock=clock, sleep=clock.sleep)
    with _client(fake, clock, bucket=bucket, max_wait_s=10.0) as c:
        c.get("/api/v3/study/items")  # token + 1 call
        c.get("/api/v3/study/items")
        with pytest.raises(RateLimitWaitExceeded):
            c.get("/api/v3/study/items")
    assert fake.paths().count("/api/v3/study/items") == 2
