"""The HTTP contract number, sent on every response as `X-Kalos-Contract`.

One integer for the API a client can rely on (docs/openapi.json). Today it is
the analysis contract number (`ANALYSIS_CONTRACT_VERSION`); `/api/health` and
every analysis body already carry it as `contract_version`, and the header lets
a client read it from any response, refusals included, without parsing a body.

Pure ASGI on purpose: it never reads or buffers the body, so streaming and
upload paths are untouched. Installed by `kalos.portal.contract_install`.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from kalos.portal.analysis_parts.policy import ANALYSIS_CONTRACT_VERSION

CONTRACT_VERSION: int = ANALYSIS_CONTRACT_VERSION
CONTRACT_HEADER = "X-Kalos-Contract"

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_HEADER_NAME = CONTRACT_HEADER.lower().encode("latin-1")
_HEADER_VALUE = str(CONTRACT_VERSION).encode("latin-1")


class ContractHeaderMiddleware:
    """Adds `X-Kalos-Contract: <CONTRACT_VERSION>` to every HTTP response that
    passes through it (a header a route already set is left alone)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_header(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers") or [])
                if not any(name.lower() == _HEADER_NAME for name, _ in headers):
                    headers.append((_HEADER_NAME, _HEADER_VALUE))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_header)


__all__ = ["CONTRACT_HEADER", "CONTRACT_VERSION", "ContractHeaderMiddleware"]
