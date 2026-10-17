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
2. No `Content-Length` (chunked transfer) -> the body stream is counted as it
   is read and the request fails with 413 the moment it crosses the cap, so a
   client cannot dodge (1) by omitting the header.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any

from starlette.exceptions import HTTPException

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
        self.upload_paths = frozenset(upload_paths)

    def cap_for(self, path: str) -> int:
        return self.upload_cap if path.rstrip("/") in self.upload_paths else self.default_cap

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        cap = self.cap_for(path)
        lengths = [v for k, v in scope.get("headers", []) if k.lower() == b"content-length"]
        if lengths:
            # Conflicting duplicates are a request-smuggling smell; refuse them.
            if len(set(lengths)) != 1 or not lengths[0].strip().isdigit():
                await _send_error(send, 400, _ERR_BAD_LENGTH)
                return
            declared = int(lengths[0])
            if declared > cap:
                log.warning("refused a %d-byte request to %s (cap %d)", declared, path, cap)
                await _send_error(send, 413, _ERR_TOO_LARGE)
                return

        received = 0

        async def _counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    log.warning("refused a streamed request to %s over the %d-byte cap", path, cap)
                    # Starlette's HTTPException is re-raised as-is by FastAPI's
                    # body parsing, so the route answers 413 rather than a
                    # generic "error parsing the body" 400.
                    raise HTTPException(status_code=413, detail=_ERR_TOO_LARGE)
            return message

        response_started = False

        async def _tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, _counting_receive, _tracking_send)
        except HTTPException as exc:
            # The cap tripped somewhere the framework did not convert it (e.g. a
            # handler streaming the body itself). Answer 413 if we still can.
            if exc.status_code != 413 or response_started:
                raise
            await _send_error(send, 413, _ERR_TOO_LARGE)


async def _send_error(send: Send, status: int, message: str) -> None:
    body = json.dumps({"error": message}).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"connection", b"close"),
        ],
    })
    await send({"type": "http.response.body", "body": body})
