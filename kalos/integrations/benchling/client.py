"""HTTP client for one Benchling tenant (REST v2 and v3).

Built from the public specs (`https://benchling.com/api/v2/openapi.yaml`,
`/api/v3/openapi.yaml`) and the docs pages "Authentication", "V2 Limits" and
"V3 Rate Limits". No live tenant was available while writing it; see
docs/BENCHLING.md for what that leaves unchecked.

What it does:

- **Auth, two modes.** App credentials: `POST /oauth/token` with
  `grant_type=client_credentials`. The token lasts `expires_in` seconds (900 in
  the docs) and there is no refresh token, so a new one is requested shortly
  before expiry and once after a 401. Delegated: a user's access token (from the
  authorization-code flow, done by the web app) is sent as is. If a refresh token
  and the app credentials are given it is refreshed with `grant_type=refresh_token`
  and the new single-use refresh token is handed to `on_refresh`.
- **Rate limit.** A token bucket per Benchling host, 60 tokens, each returned
  30 s after use, so at most 60 requests in any 30 s window by
  default (the documented per-tenant budget for user keys, the smallest one), shared
  by every client in this process. A 429 pauses the bucket for `Retry-After`
  seconds (v3), else `x-rate-limit-reset` (v2), else an exponential backoff with
  jitter, then retries up to `max_retries` times.
- **5xx.** A GET is retried with backoff (the docs: "significantly back off or
  abort"). A write is never retried after a 5xx, because it may have been applied.
- **EARLY-ACCESS.** The `EARLY-ACCESS: true` header is sent only on the v3 paths
  the spec marks beta (`/api/v3/procedure*`, the result-schema
  field-definitions list, the study authors list). Nowhere else.
- **Pagination.** `paginate()` follows `nextToken` (v3 `items`, v2 named lists).
- **Drafts only.** Writes are allowed by list (run plan, plan conditions, result
  transaction, blob, entry field); anything else, `:promote` in any spelling
  included, is refused before it is sent.
- **Errors.** Every failure is a `BenchlingError` subclass carrying the status and
  Benchling's short title. Tokens and secrets never appear in a message, a repr or a
  log line.
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("kalos.integrations.benchling")

DEFAULT_RATE_LIMIT = 60          # requests ...
DEFAULT_RATE_PERIOD_S = 30.0     # ... per this many seconds, per Benchling tenant
TOKEN_REFRESH_MARGIN_S = 60.0    # ask for a new token this long before expiry
DEFAULT_TOKEN_LIFETIME_S = 900.0  # "Access tokens expire after 15 minutes"
MAX_BACKOFF_S = 30.0              # longest exponential backoff step
MAX_RETRY_AFTER_S = 30.0          # longest Retry-After / x-rate-limit-reset honoured; longer -> give up
DEFAULT_DEADLINE_S = 180.0        # whole-operation budget of one client (one Kalos request)
DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_ITEMS = 20_000
MAX_PAGE_SIZE = 100              # v2 and v3 both cap pageSize at 100
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_WAIT_S = 60.0         # longest the limiter may block a single call

# v3 operations the spec marks beta (they declare the EARLY-ACCESS header).
# Every `/procedure*` operation is beta; of the result-schema operations only the
# field-definitions list is, and of the study operations only the authors list.
# tests/fixtures/benchling/v3_beta_ops.json holds the list taken from the spec and
# the client is tested against it.
_BETA_PATHS = (
    re.compile(r"^/api/v3/procedure[a-z-]*(?:[/:]|$)"),
    re.compile(r"^/api/v3/result-schema/[^/]+/field-definitions/items$"),
    re.compile(r"^/api/v3/study/[^/]+/authors/items$"),
)


# Kalos writes drafts only. Writes are allowed BY LIST: the five write paths the
# connector uses, matched in full with plain ids. Anything else is refused before
# it is sent, whatever its spelling (`:Promote`, `%3Apromote`, `:promote/`), and
# so is any other method. Promoting a run plan starts a run in the lab, which is a
# person's decision in Benchling.
_ID = r"[A-Za-z0-9_-]{1,128}"
_ALLOWED_WRITES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("POST", re.compile(r"/api/v3/procedure-run-plan")),
    ("POST", re.compile(r"/api/v3/procedure-run-plan-condition:batch-create")),
    ("POST", re.compile(rf"/api/v2/result-transactions(?:/{_ID}/results|/{_ID}:commit|/{_ID}:abort)?")),
    ("POST", re.compile(r"/api/v2/blobs")),
    ("PATCH", re.compile(rf"/api/v2/entries/{_ID}")),
)
_READ_PATH = re.compile(r"/api/v[23](?:/[A-Za-z0-9_-]+)+")


def write_allowed(method: str, path: str) -> bool:
    """Whether (method, path) is one of the connector's own writes."""
    return any(m == method and p.fullmatch(path) for m, p in _ALLOWED_WRITES)


def is_beta_path(path: str) -> bool:
    """Whether `path` (`/api/v3/...`, no query string) is a beta v3 operation."""
    return any(p.match(path) for p in _BETA_PATHS)


# --- errors ------------------------------------------------------------------- #


class BenchlingError(Exception):
    """A Benchling call failed. `status` is the HTTP status (0 for a transport
    failure); `title` is Benchling's short error title or message, clipped."""

    def __init__(self, message: str, *, status: int = 0, title: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.title = title


class BenchlingConfigError(BenchlingError):
    """The connection settings are unusable (bad base URL, missing credentials)."""


class BenchlingAuthError(BenchlingError):
    """Benchling refused the credentials or the token (401, or /oauth/token failed)."""


class BenchlingForbidden(BenchlingError):
    """403: the app or user lacks access to the object."""


class BenchlingNotFound(BenchlingError):
    """404."""


class BenchlingConflict(BenchlingError):
    """409."""


class BenchlingInvalidRequest(BenchlingError):
    """400 / 413 / 422: Benchling rejected the request body or parameters."""


class BenchlingRateLimited(BenchlingError):
    """429 after every retry. `retry_after` is the last wait Benchling asked for."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw: Any) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class RateLimitWaitExceeded(BenchlingRateLimited):
    """Kalos's own limiter would have to wait longer than the caller allows.
    Nothing was sent."""


class BenchlingDeadlineExceeded(BenchlingError):
    """The operation's deadline passed before Benchling finished answering."""


class BenchlingTooMuchData(BenchlingError):
    """A list returned more items or pages than the caller allows."""


class BenchlingServerError(BenchlingError):
    """5xx."""


class BenchlingTransportError(BenchlingError):
    """The request did not complete (DNS, TLS, timeout, connection reset)."""


_STATUS_ERRORS: dict[int, type[BenchlingError]] = {
    400: BenchlingInvalidRequest, 401: BenchlingAuthError, 403: BenchlingForbidden,
    404: BenchlingNotFound, 409: BenchlingConflict, 413: BenchlingInvalidRequest,
    422: BenchlingInvalidRequest, 429: BenchlingRateLimited,
}

_MAX_TITLE = 200


def _error_title(response: httpx.Response) -> str | None:
    """Benchling's own short error text: v3 problem+json `title`/`detail`, v2
    `error.message`. Clipped; never the request."""
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    title: Any = None
    if isinstance(body.get("error"), dict):
        title = body["error"].get("message") or body["error"].get("userMessage")
    if title is None:
        detail = body.get("detail")
        title = body.get("title") or (detail if isinstance(detail, str) else None)
    if not isinstance(title, str):
        return None
    return title[:_MAX_TITLE]


def _error_for(response: httpx.Response, method: str, path: str) -> BenchlingError:
    status = response.status_code
    title = _error_title(response)
    cls = _STATUS_ERRORS.get(status)
    if cls is None:
        cls = BenchlingServerError if status >= 500 else BenchlingError
    msg = f"Benchling {method} {path} returned {status}"
    if title:
        msg += f": {title}"
    return cls(msg, status=status, title=title)


# --- rate limit ---------------------------------------------------------------- #


class TokenBucket:
    """`capacity` tokens; each request takes one, and each token comes back
    exactly `period_s` after it was taken. So no window of `period_s` seconds
    ever holds more than `capacity` requests (a plain refill-rate bucket would let
    up to twice that through: a full burst plus the refill). `acquire()` blocks
    until a token is free; `pause(s)` blocks every caller for `s` seconds (after a
    429, so the whole budget backs off together, as the V3 rate-limit page asks)."""

    def __init__(
        self, capacity: int = DEFAULT_RATE_LIMIT, period_s: float = DEFAULT_RATE_PERIOD_S, *,
        clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity < 1 or period_s <= 0:
            raise ValueError("rate limit must be at least 1 request per positive period")
        self.capacity = capacity
        self.period_s = period_s
        self._clock = clock
        self._sleep = sleep
        self._taken: deque[float] = deque()  # when each token in use was taken
        self._paused_until = 0.0
        self._lock = threading.Lock()

    def _return_tokens(self, now: float) -> None:
        while self._taken and self._taken[0] + self.period_s <= now:
            self._taken.popleft()

    def now(self) -> float:
        return self._clock()

    def acquire(self, deadline: float | None = None) -> float:
        """Take one token, waiting as needed. Returns the seconds waited. With a
        `deadline` (a value of this bucket's clock), raises RateLimitWaitExceeded
        instead of waiting past it."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._return_tokens(now)
                if now < self._paused_until:
                    wait = self._paused_until - now
                elif len(self._taken) < self.capacity:
                    self._taken.append(now)
                    return waited
                else:
                    wait = self._taken[0] + self.period_s - now
                if deadline is not None and now + wait > deadline:
                    raise RateLimitWaitExceeded(
                        "the Benchling request budget is used up for longer than this request may wait",
                        retry_after=wait,
                    )
            self._sleep(wait)
            waited += wait

    def pause(self, seconds: float) -> None:
        with self._lock:
            now = self._clock()
            self._paused_until = max(self._paused_until, now + max(0.0, seconds))

    @property
    def available(self) -> int:
        with self._lock:
            self._return_tokens(self._clock())
            return self.capacity - len(self._taken)


_BUCKETS: dict[tuple[str, str], TokenBucket] = {}
_BUCKETS_LOCK = threading.Lock()


def bucket_for(base_url: str, tenant: str | None = None) -> TokenBucket:
    """The process-wide bucket for one (Kalos tenant, Benchling host), so one
    workspace cannot use up another's budget. Several processes do not share it
    (docs/BENCHLING.md, "Limits")."""
    key = (tenant or "", urlsplit(base_url).netloc.lower())
    with _BUCKETS_LOCK:
        bucket = _BUCKETS.get(key)
        if bucket is None:
            bucket = _BUCKETS[key] = TokenBucket()
        return bucket


def reset_buckets() -> None:
    """Forget every bucket (tests)."""
    with _BUCKETS_LOCK:
        _BUCKETS.clear()


# --- base URL ------------------------------------------------------------------- #

_TENANT_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.benchling\.com")


def normalize_base_url(base_url: str) -> str:
    """`https://<tenant>.benchling.com`, nothing else: https, a single tenant
    label under benchling.com, no port, path, query, user info or fragment. This
    is the only host the stored credentials are ever sent to."""
    if not isinstance(base_url, str):
        raise BenchlingConfigError("base_url must be a string")
    parts = urlsplit(base_url.strip())
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https" or parts.port is not None or parts.username or parts.password
        or parts.query or parts.fragment or parts.path not in ("", "/")
        or _TENANT_HOST.fullmatch(host) is None
    ):
        raise BenchlingConfigError("base_url must look like https://<tenant>.benchling.com")
    return f"https://{host}"


# --- tokens ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class AppCredentials:
    """A Benchling App's client id and secret. The secret is kept out of repr."""

    client_id: str
    client_secret: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.client_id or not self.client_secret:
            raise BenchlingConfigError("client id and client secret are both required")


class TokenSource(Protocol):
    kind: str  # "app" | "delegated"

    def bearer(
        self, http: httpx.Client, base_url: str, bucket: TokenBucket, deadline: float | None = None,
    ) -> str: ...

    def invalidate(self) -> bool:
        """Drop the cached token after a 401. True if a fresh one can be obtained."""
        ...


def _token_request(
    http: httpx.Client, base_url: str, bucket: TokenBucket, form: Mapping[str, str],
    deadline: float | None = None,
) -> dict[str, Any]:
    bucket.acquire(deadline)
    try:
        resp = http.post(
            f"{base_url}/oauth/token", data=dict(form),
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise BenchlingTransportError(
            f"could not reach Benchling for a token ({type(exc).__name__})"
        ) from None
    if resp.status_code == 429:
        raise BenchlingRateLimited("Benchling rate-limited the token request", status=429)
    if resp.status_code != 200:
        # The body of a failed token call is not echoed: only the status.
        raise BenchlingAuthError(
            f"Benchling refused the token request ({resp.status_code})", status=resp.status_code,
        )
    try:
        body = resp.json()
    except ValueError:
        raise BenchlingAuthError("Benchling returned an unreadable token response") from None
    if not isinstance(body, dict) or not isinstance(body.get("access_token"), str):
        raise BenchlingAuthError("Benchling returned a token response without an access token")
    return body


def _lifetime(body: Mapping[str, Any]) -> float:
    value = body.get("expires_in")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return DEFAULT_TOKEN_LIFETIME_S


class ClientCredentialsToken:
    """App mode: `client_credentials` grant, cached until `TOKEN_REFRESH_MARGIN_S`
    before it expires."""

    kind = "app"

    def __init__(self, credentials: AppCredentials, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._creds = credentials
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()
        self.fetches = 0  # how many tokens were requested (status and tests)

    def __repr__(self) -> str:
        return f"ClientCredentialsToken(client_id={self._creds.client_id!r})"

    def bearer(
        self, http: httpx.Client, base_url: str, bucket: TokenBucket, deadline: float | None = None,
    ) -> str:
        with self._lock:
            now = self._clock()
            if self._token is None or now >= self._expires_at - TOKEN_REFRESH_MARGIN_S:
                body = _token_request(http, base_url, bucket, {
                    "grant_type": "client_credentials",
                    "client_id": self._creds.client_id,
                    "client_secret": self._creds.client_secret,
                }, deadline)
                self._token = body["access_token"]
                self._expires_at = now + _lifetime(body)
                self.fetches += 1
            return self._token

    def invalidate(self) -> bool:
        with self._lock:
            self._token = None
            self._expires_at = 0.0
        return True


class DelegatedToken:
    """Delegated mode: a logged-in user's access token. Actions are attributed to
    that user in Benchling's audit log and `createdBy` fields.

    With `refresh_token` and `credentials` it refreshes itself (single-use refresh
    tokens: the replacement goes to `on_refresh`, and the caller must keep it).
    Without them, an expired token is a `BenchlingAuthError` and the user signs in
    again. `expires_in`, when known, triggers the refresh before expiry."""

    kind = "delegated"

    def __init__(
        self, access_token: str, *, refresh_token: str | None = None,
        credentials: AppCredentials | None = None, expires_in: float | None = None,
        on_refresh: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(access_token, str) or not access_token.strip():
            raise BenchlingConfigError("a delegated access token is required")
        self._token: str | None = access_token.strip()
        self._refresh = refresh_token
        self._creds = credentials
        self._clock = clock
        self._expires_at = clock() + expires_in if expires_in else None
        self._on_refresh = on_refresh
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return "DelegatedToken(<redacted>)"

    def _can_refresh(self) -> bool:
        return self._refresh is not None and self._creds is not None

    def bearer(
        self, http: httpx.Client, base_url: str, bucket: TokenBucket, deadline: float | None = None,
    ) -> str:
        with self._lock:
            now = self._clock()
            stale = self._token is None or (
                self._expires_at is not None and now >= self._expires_at - TOKEN_REFRESH_MARGIN_S
            )
            if stale:
                if not self._can_refresh():
                    if self._token is None:
                        raise BenchlingAuthError(
                            "the delegated token was refused and cannot be refreshed; sign in to Benchling again",
                            status=401,
                        )
                    return self._token
                assert self._creds is not None and self._refresh is not None
                body = _token_request(http, base_url, bucket, {
                    "grant_type": "refresh_token",
                    "client_id": self._creds.client_id,
                    "client_secret": self._creds.client_secret,
                    "refresh_token": self._refresh,
                }, deadline)
                self._token = body["access_token"]
                self._expires_at = now + _lifetime(body)
                new_refresh = body.get("refresh_token")
                if isinstance(new_refresh, str) and new_refresh:
                    self._refresh = new_refresh
                    if self._on_refresh is not None:
                        self._on_refresh(new_refresh)
            assert self._token is not None
            return self._token

    def invalidate(self) -> bool:
        with self._lock:
            self._token = None
            return self._can_refresh()


# --- the client -------------------------------------------------------------------- #


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds to wait from `Retry-After` (v3) or `x-rate-limit-reset` (v2)."""
    for name in ("retry-after", "x-rate-limit-reset"):
        raw = response.headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip())
        except ValueError:
            continue
        if value >= 0:
            return value
    return None


_IDEMPOTENT = frozenset({"GET", "HEAD"})


class BenchlingClient:
    """One Benchling tenant, one identity (an app or a delegated user).

    `transport` lets tests (and a future proxy) replace the network; everything
    else about a request is the same. Use as a context manager or call `close()`.
    """

    def __init__(
        self, base_url: str, auth: TokenSource, *,
        transport: httpx.BaseTransport | None = None,
        bucket: TokenBucket | None = None,
        tenant: str | None = None,
        max_wait_s: float = DEFAULT_MAX_WAIT_S,
        deadline_s: float | None = DEFAULT_DEADLINE_S,
        max_retries: int = 5,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self.auth = auth
        self.bucket = bucket if bucket is not None else bucket_for(self.base_url, tenant)
        self.max_wait_s = max_wait_s
        self.timeout_s = timeout_s
        # One client serves one Kalos request: everything it sends must finish
        # by this point on the bucket's clock.
        self.deadline = None if deadline_s is None else self.bucket.now() + deadline_s
        self.max_retries = max(0, int(max_retries))
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - backoff jitter, not security
        self._http = httpx.Client(
            transport=transport, timeout=timeout_s, follow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": "kalos-benchling/1"},
        )
        self.requests_sent = 0
        self.rate_limited = 0

    @property
    def mode(self) -> str:
        return self.auth.kind

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> BenchlingClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"BenchlingClient({self.base_url!r}, mode={self.mode!r})"

    def authenticate(self) -> None:
        """Obtain a token now (app mode: one POST /oauth/token). Used to check
        credentials before they are stored."""
        self.auth.bearer(self._http, self.base_url, self.bucket, self._wait_deadline())

    def _wait_deadline(self) -> float:
        """How long the limiter may block one call: max_wait_s, never past the
        operation deadline."""
        wait_until = self.bucket.now() + self.max_wait_s
        return wait_until if self.deadline is None else min(wait_until, self.deadline)

    def _remaining(self, method: str, path: str) -> float | None:
        if self.deadline is None:
            return None
        left = self.deadline - self.bucket.now()
        if left <= 0:
            raise BenchlingDeadlineExceeded(f"Benchling {method} {path}: the request deadline passed")
        return left

    def _pause_or_give_up(self, seconds: float, method: str, path: str) -> None:
        left = self._remaining(method, path)
        if left is not None and seconds >= left:
            raise BenchlingDeadlineExceeded(f"Benchling {method} {path}: retrying would pass the request deadline")

    def _backoff(self, attempt: int) -> float:
        return min(MAX_BACKOFF_S, 2.0 ** attempt + self._rng.random())

    def request(
        self, method: str, path: str, *, params: Mapping[str, Any] | None = None,
        json: Any = None, expect: tuple[int, ...] = (200, 201, 202, 204),
    ) -> Any:
        """Send one call and return the decoded JSON body (None for an empty body).
        `path` starts with `/api/v2/` or `/api/v3/`."""
        method = method.upper()
        if "promote" in path.lower():
            raise BenchlingConfigError("Kalos never promotes a run plan; a person does that in Benchling")
        if method in _IDEMPOTENT:
            if _READ_PATH.fullmatch(path) is None:
                raise BenchlingConfigError("path must be a plain /api/v2 or /api/v3 path")
        elif not write_allowed(method, path):
            raise BenchlingConfigError("Kalos only makes the writes it documents (drafts and delegated results)")
        headers: dict[str, str] = {}
        if is_beta_path(path):
            headers["EARLY-ACCESS"] = "true"
        query = {k: _param(v) for k, v in (params or {}).items() if v is not None}
        url = self.base_url + path
        reauthed = False
        attempt = 0
        while True:
            self._remaining(method, path)
            token = self.auth.bearer(self._http, self.base_url, self.bucket, self._wait_deadline())
            self.bucket.acquire(self._wait_deadline())
            left = self._remaining(method, path)
            timeout = self.timeout_s if left is None else max(0.1, min(self.timeout_s, left))
            self.requests_sent += 1
            try:
                resp = self._http.request(
                    method, url, params=query or None, json=json,
                    headers={**headers, "Authorization": f"Bearer {token}"}, timeout=timeout,
                )
            except httpx.HTTPError as exc:
                if method in _IDEMPOTENT and attempt < self.max_retries:
                    attempt += 1
                    delay = self._backoff(attempt)
                    self._pause_or_give_up(delay, method, path)
                    self._sleep(delay)
                    continue
                raise BenchlingTransportError(
                    f"Benchling {method} {path} did not complete ({type(exc).__name__})"
                ) from None
            status = resp.status_code
            log.debug("benchling %s %s -> %s", method, path, status)
            if status in expect:
                if not resp.content:
                    return None
                try:
                    return resp.json()
                except ValueError:
                    raise BenchlingError(
                        f"Benchling {method} {path} returned a body that is not JSON", status=status,
                    ) from None
            if status == 401 and not reauthed and self.auth.invalidate():
                reauthed = True
                continue
            if status == 429:
                self.rate_limited += 1
                wait = _retry_after(resp)
                # A wait longer than MAX_RETRY_AFTER_S is not sat out: the caller
                # gets a 503 with that Retry-After instead of a hung request.
                if attempt >= self.max_retries or (wait is not None and wait > MAX_RETRY_AFTER_S):
                    err = _error_for(resp, method, path)
                    raise BenchlingRateLimited(
                        str(err), retry_after=wait, status=429, title=err.title,
                    )
                attempt += 1
                delay = wait if wait is not None else self._backoff(attempt)
                self._pause_or_give_up(delay, method, path)
                self.bucket.pause(delay)
                continue
            if status >= 500 and method in _IDEMPOTENT and attempt < self.max_retries:
                attempt += 1
                delay = self._backoff(attempt)
                self._pause_or_give_up(delay, method, path)
                self._sleep(delay)
                continue
            raise _error_for(resp, method, path)

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, json: Any = None, params: Mapping[str, Any] | None = None) -> Any:
        return self.request("POST", path, json=json, params=params)

    def patch(self, path: str, json: Any = None) -> Any:
        return self.request("PATCH", path, json=json)

    def paginate(
        self, path: str, params: Mapping[str, Any] | None = None, *, items_key: str = "items",
        page_size: int = MAX_PAGE_SIZE, max_pages: int = DEFAULT_MAX_PAGES,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> Iterator[dict[str, Any]]:
        """Every item of a list endpoint, following `nextToken`. `items_key` is
        `items` for v3 and the resource name for v2 (`assayRuns`, `projects`, ...).
        Raises BenchlingTooMuchData past `max_items` items or `max_pages` pages."""
        query = dict(params or {})
        query["pageSize"] = max(1, min(int(page_size), MAX_PAGE_SIZE))
        seen = 0
        for _ in range(max_pages):
            body = self.get(path, query)
            if not isinstance(body, dict):
                raise BenchlingError(f"Benchling GET {path} returned an unexpected page")
            items = body.get(items_key) or []
            if not isinstance(items, list):
                raise BenchlingError(f"Benchling GET {path} returned an unexpected page")
            for item in items:
                if isinstance(item, dict):
                    seen += 1
                    if seen > max_items:
                        raise BenchlingTooMuchData(f"Benchling GET {path} returned more than {max_items} items")
                    yield item
            token = body.get("nextToken")
            if not token:
                return
            query["nextToken"] = token
        raise BenchlingTooMuchData(f"Benchling GET {path} returned more than {max_pages} pages")


def _param(value: Any) -> str:
    """A query value: lists become the comma lists the spec uses (`explode: false`),
    booleans become `true`/`false`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ",".join(_param(v) for v in value)
    return str(value)


_API_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def api_id(value: object) -> str:
    """A Benchling object id that is safe to put in a path segment (letters,
    digits, `_`, `-`). Raises BenchlingConfigError otherwise; never echoes it."""
    if not isinstance(value, str) or _API_ID.fullmatch(value) is None:
        raise BenchlingConfigError("not a Benchling id")
    return value


def chunks(ids: list[str], size: int = 100) -> Iterator[list[str]]:
    """`*.anyOf` filters accept at most 100 ids (spec `maxItems: 100`)."""
    for i in range(0, len(ids), size):
        yield ids[i:i + size]
