"""Kalos portal - request body size limits (P0 hardening).

A pure-ASGI middleware that bounds every request body BEFORE any route parses
it, so a single oversized request cannot make this single-process engine
buffer it into memory (Pydantic/JSON parsing reads the whole body first; the
multipart parser spools it).

Two caps, chosen by path:

- Upload routes (`/api/run`, `/api/scale/readout`) get `MAX_UPLOAD_BYTES` (the
  file cap enforced by `kalos.portal.uploads`) plus a fixed allowance for the
  multipart framing and small form fields. The file cap itself is still
  enforced by the route; this outer cap only stops a body that cannot possibly
  be a valid upload from being read at all.
- Every other route gets `KALOS_MAX_REQUEST_BYTES` (default 1 MiB). The JSON
  bodies on `/api/campaign/*` are a few KiB in practice.

Enforcement:

1. `Content-Length` over the cap -> 413 immediately, the body is never read.
   A malformed `Content-Length` -> 400.
   A `Content-Length` of more than 19 digits is over any cap -> 413 without
   calling `int()` on it.
2. No `Content-Length` (chunked transfer) -> the body stream is counted as it
   is read and the request fails with 413 the moment it crosses the cap, so a
   client cannot dodge (1) by omitting the header. This uses a flag+disconnect
   pattern rather than raising: once over the cap the inner app is handed an
   `http.disconnect` (so it stops reading), everything it tries to send is
   swallowed, and this middleware answers the 413 itself. Raising from
   `receive` does not work on the real app: FastAPI's body parsing and
   `BaseHTTPMiddleware` turn the exception into a generic 400.

Paths are matched with `scope["root_path"]` stripped, so a deployment mounted
under a prefix still gives the upload routes the upload cap.

`UploadAuthPrecheckMiddleware` (separate, below) refuses an upload with no
bearer token before the multipart body is read, when authentication is
enforced. It only checks that a `Bearer` credential is PRESENT; verifying it
stays with the `require_principal` dependency.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any

log = logging.getLogger("kalos.portal")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

DEFAULT_MAX_REQUEST_BYTES = 1024 * 1024
# Multipart boundary/headers plus the small form fields (`target`, `roles`,
# `anonymize`) that ride alongside the uploaded file.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

_ERR_TOO_LARGE = "The request body is too large."
_ERR_BAD_LENGTH = "Invalid Content-Length header."
_ERR_AUTH_REQUIRED = "authentication required"


def max_request_bytes() -> int:
    """The non-upload body cap from `KALOS_MAX_REQUEST_BYTES` (bytes).

    Empty/unset means the default, matching the other sizing knobs. A value
    that is not a positive integer is an operator error and raises, rather
    than silently running with no limit.
    """
    raw = os.environ.get("KALOS_MAX_REQUEST_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BYTES
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"KALOS_MAX_REQUEST_BYTES must be a positive integer; got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"KALOS_MAX_REQUEST_BYTES must be a positive integer; got {raw!r}")
    return value


# A Content-Length with more digits than this is over any cap we could set
# (19 digits covers every int64), so it is refused without parsing it: `int()`
# on a multi-thousand-digit string raises ValueError (CPython's int digit limit).
_MAX_CONTENT_LENGTH_DIGITS = 19


def route_path(scope: Scope) -> str:
    """The request path with the ASGI `root_path` prefix removed, trailing
    slashes dropped, so route matching is independent of where the app is
    mounted."""
    path = scope.get("path", "") or ""
    root = scope.get("root_path", "") or ""
    if root and path.startswith(root):
        path = path[len(root):]
    return path.rstrip("/") or "/"


class RequestSizeLimitMiddleware:
    """Reject request bodies larger than the configured cap (see module doc)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        default_cap: int,
        upload_cap: int,
        upload_paths: Iterable[str],
    ) -> None:
        self.app = app
        self.default_cap = default_cap
        self.upload_cap = upload_cap
        self.upload_paths = frozenset(p.rstrip("/") for p in upload_paths)

    def cap_for(self, path: str) -> int:
        return self.upload_cap if path.rstrip("/") in self.upload_paths else self.default_cap

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = route_path(scope)
        cap = self.cap_for(path)
        lengths = [v for k, v in scope.get("headers", []) if k.lower() == b"content-length"]
        if lengths:
            # Conflicting duplicates are a request-smuggling smell; refuse them.
            if len(set(lengths)) != 1 or not lengths[0].strip().isdigit():
                await _send_error(send, 400, _ERR_BAD_LENGTH)
                return
            digits = lengths[0].strip().lstrip(b"0")
            if len(digits) > _MAX_CONTENT_LENGTH_DIGITS or int(digits or b"0") > cap:
                log.warning("refused a request to %s with Content-Length over the %d-byte cap", path, cap)
                await _send_error(send, 413, _ERR_TOO_LARGE)
                return

        received = 0
        over_cap = False

        async def _counting_receive() -> Message:
            nonlocal received, over_cap
            if over_cap:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    log.warning("refused a streamed request to %s over the %d-byte cap", path, cap)
                    # Flag, and tell the inner app the client went away so it
                    # stops reading. Whatever it answers is swallowed below and
                    # replaced with the 413.
                    over_cap = True
                    return {"type": "http.disconnect"}
            return message

        response_started = False

        async def _guarded_send(message: Message) -> None:
            nonlocal response_started
            if over_cap:
                return  # the inner app's answer to a "disconnect" is not sent
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, _counting_receive, _guarded_send)
        except Exception:
            if not over_cap:
                raise
            # The inner app raised on the synthetic disconnect; the 413 below
            # is the real answer.
        if over_cap and not response_started:
            await _send_error(send, 413, _ERR_TOO_LARGE)


def _has_bearer(scope: Scope) -> bool:
    """Whether the request carries an `Authorization: Bearer <non-empty>` header
    (presence only - the token is verified by `require_principal`)."""
    values = [v for k, v in scope.get("headers", []) if k.lower() == b"authorization"]
    if len(values) != 1:
        return False
    value = values[0]
    return value.startswith(b"Bearer ") and bool(value[len(b"Bearer "):].strip())


class UploadAuthPrecheckMiddleware:
    """Refuse an upload with no bearer token BEFORE its multipart body is read.

    The upload routes take up to `MAX_UPLOAD_BYTES` + multipart overhead, and
    FastAPI parses (spools) the whole form before the auth dependency runs, so
    an anonymous caller could make the engine read ~25 MB per request just to
    be told 401. When authentication is enforced, a request to an upload path
    without a `Bearer` credential is answered 401 here, body unread.

    Deliberately minimal: it does NOT verify the token (the dependency does
    that, constant-time, with scopes), and it passes through untouched when
    auth is not enforced, when the auth config is broken (the dependency or
    the production guard answers 500/503), and for every non-upload path.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        upload_paths: Iterable[str],
        auth_enforced: Callable[[], bool],
    ) -> None:
        self.app = app
        self.upload_paths = frozenset(p.rstrip("/") for p in upload_paths)
        self.auth_enforced = auth_enforced

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and route_path(scope) in self.upload_paths
            and not _has_bearer(scope)
        ):
            try:
                enforced = self.auth_enforced()
            except Exception:  # noqa: BLE001 - a broken config is the dependency's 500
                enforced = False
            if enforced:
                # Same body shape as the dependency's 401 (`{"detail": ...}`),
                # so a client sees one answer whichever layer refused it.
                await _send_error(
                    send, 401, _ERR_AUTH_REQUIRED, key="detail",
                    extra_headers=[(b"www-authenticate", b"Bearer")],
                )
                return
        await self.app(scope, receive, send)


async def _send_error(
    send: Send,
    status: int,
    message: str,
    *,
    key: str = "error",
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    body = json.dumps({key: message}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"connection", b"close"),
            *(extra_headers or []),
        ],
    })
    await send({"type": "http.response.body", "body": body})
