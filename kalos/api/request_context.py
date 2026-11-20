"""Request ids and structured logs, adopted from Voyager's py-common `Logging.py`
and `Middleware.py` (NOTICE).

What is taken: one id per request, held in a `contextvars.ContextVar` so every
log line written while serving the request carries it (`RequestIdFilter`), and
echoed on the response so a caller (kalos-web's proxy, a job callback) can quote
it. Kalos already had an `audit_log.request_id` column with nothing filling it;
the records routes now record this id there.

What is changed for Kalos:
- the header is `X-Request-ID`; Voyager's `X-Correlation-ID` is accepted too. An
  incoming id is used only if it is 1-128 characters of ``[A-Za-z0-9._:-]``;
  anything else is replaced by a fresh ``req_<32 hex>`` (a caller cannot inject
  newlines or markup into the server log);
- the response carries `X-Request-ID` only when the CALLER sent a usable id (it
  is echoed unchanged). A server-minted id goes to the logs and audit rows but
  not into the response, so two requests that must answer byte-identically
  (a foreign campaign id and a missing one) still do: a random header would
  make them differ. kalos-web's proxy sends its own id to correlate;
- a pure ASGI middleware (no Starlette `BaseHTTPMiddleware`, which buffers
  streaming responses);
- `KALOS_LOG_FORMAT=json` writes one JSON object per line (ts, level, logger,
  request_id, message) for a log shipper; `text` (the default) keeps the plain
  format with the id added. Unknown values are refused (kalos.envconfig).

What is not taken: the daily rotating log FILE and its 7-file cleanup. The
engine logs to stderr and the container runtime (deploy/compose.yaml) keeps and
rotates the logs; a second rotation policy inside the app would fight it.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from kalos.envconfig import env_enum

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"
LOG_FORMAT_ENV = "KALOS_LOG_FORMAT"
LOG_FORMATS = ("text", "json")
_VALID_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("kalos_request_id", default=None)

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def current_request_id() -> str | None:
    """The id of the request being served, or None outside a request."""
    return _request_id.get()


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex


def accept_request_id(value: str | None) -> str:
    """The caller's id when it is safe to log, else a fresh one."""
    if value is not None and _VALID_ID.fullmatch(value):
        return value
    return new_request_id()


class RequestIdMiddleware:
    """Sets the request id for the request's context; echoes a caller's usable
    `X-Request-ID` on the response (HTTP only; other ASGI scopes pass through)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        incoming: str | None = None
        for name, value in scope.get("headers") or ():
            key = name.decode("latin-1").lower()
            if key in ("x-request-id", "x-correlation-id"):
                incoming = value.decode("latin-1")
                if key == "x-request-id":
                    break
        rid = accept_request_id(incoming)
        echo = rid == incoming
        token = _request_id.set(rid)

        async def _send(message: Message) -> None:
            if echo and message.get("type") == "http.response.start":
                headers = [
                    (k, v) for k, v in message.get("headers") or ()
                    if k.decode("latin-1").lower() != "x-request-id"
                ]
                headers.append((b"x-request-id", rid.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Adds `record.request_id` ('-' outside a request) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. The message is the formatted message only;
    callers already log identifiers and codes, never payloads or credentials."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            out["exc_type"] = record.exc_info[0].__name__
            out["traceback"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


TEXT_FORMAT = "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s %(message)s"
_CONFIGURED_ATTR = "_kalos_request_id_handler"


def log_format_setting() -> str:
    return env_enum(LOG_FORMAT_ENV, LOG_FORMATS, "text")


def configure_logging(*, fmt: str | None = None, level: int = logging.WARNING) -> logging.Handler:
    """Install one stderr handler on the `kalos` logger (idempotent): the chosen
    format, the request-id filter. `level` stays WARNING by default, as before,
    so this changes the shape of log lines, not how many there are."""
    logger = logging.getLogger("kalos")
    for h in logger.handlers:
        if getattr(h, _CONFIGURED_ATTR, False):
            return h
    chosen = fmt or log_format_setting()
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if chosen == "json" else logging.Formatter(TEXT_FORMAT))
    setattr(handler, _CONFIGURED_ATTR, True)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)
    return handler


__all__ = [
    "CORRELATION_ID_HEADER", "LOG_FORMAT_ENV", "REQUEST_ID_HEADER", "JsonFormatter",
    "RequestIdFilter", "RequestIdMiddleware", "accept_request_id", "configure_logging",
    "current_request_id", "log_format_setting", "new_request_id",
]
