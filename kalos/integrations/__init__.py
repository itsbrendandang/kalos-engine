"""Connectors to outside systems. Each one is off unless `KALOS_INTEGRATIONS`
names it (see `kalos.integrations.enabled`)."""
from __future__ import annotations

import os
from typing import Any

INTEGRATIONS_VAR = "KALOS_INTEGRATIONS"
BENCHLING = "benchling"
BENCHLING_PREFIX = "/api/integrations/benchling"


def enabled(name: str) -> bool:
    """Whether the integration `name` is switched on. Read per call, so a test
    or an operator can flip it without a restart. The value is a comma list,
    e.g. `KALOS_INTEGRATIONS=benchling`."""
    raw = os.environ.get(INTEGRATIONS_VAR, "")
    return name in {part.strip().lower() for part in raw.split(",") if part.strip()}


_NOT_FOUND = b'{"error":"Not Found","code":"not_found"}'


class BenchlingGate:
    """ASGI gate: with the Benchling connector off, any request under its prefix
    gets the app's unknown-path 404 before its body is read. Imports nothing from
    the connector (no httpx). The app adds it innermost, so refusals that come
    before routing (open-mode remote 503, body-size 413) look the same for these
    paths as for unknown ones."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and not enabled(BENCHLING):
            path = scope.get("path", "")
            if path == BENCHLING_PREFIX or path.startswith(BENCHLING_PREFIX + "/"):
                await send({"type": "http.response.start", "status": 404, "headers": [
                    (b"content-length", str(len(_NOT_FOUND)).encode()), (b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": _NOT_FOUND})
                return
        await self.app(scope, receive, send)
