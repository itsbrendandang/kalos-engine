"""Slot for a local Ollama server, the default model for the assistant.

No credential: Ollama is a local HTTP server. `available()` says whether it is
SELECTED (`KALOS_LLM_PROVIDER=ollama`, or desktop mode with no provider set),
not whether the server answers - that is checked when a request is made, so a
status call never opens a socket.

`local()` is True only when the configured URL points at this machine
(localhost, 127.0.0.1, ::1). A remote Ollama URL means the assistant's
prompts leave the machine, and the status says so.

The HTTP client is stdlib `urllib` (no extra dependency), same as
`kalos.normalize.llm`'s Ollama path, and shares its env vars:
`KALOS_OLLAMA_URL` (default http://localhost:11434) and `KALOS_OLLAMA_MODEL`
(default llama3.1).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

from kalos.providers.base import ProviderStatus

_URL_ENV = "KALOS_OLLAMA_URL"
_MODEL_ENV = "KALOS_OLLAMA_MODEL"
_TIMEOUT_ENV = "KALOS_OLLAMA_TIMEOUT_S"
DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}

_CAPABILITY = (
    "A model running on this machine through Ollama answers assistant questions "
    "and revises SOP draft prose; nothing is sent off the machine when the URL is local."
)
_FALLBACK = (
    "Without a model the assistant still gives the next step for each campaign and "
    "drafts SOPs from the record; asking questions is turned off."
)


class OllamaUnavailable(RuntimeError):
    """The Ollama server could not be reached or gave no usable answer. The
    message is fixed and carries no prompt text."""


def ollama_url() -> str:
    return (os.environ.get(_URL_ENV) or DEFAULT_URL).strip().rstrip("/")


def ollama_model() -> str:
    return (os.environ.get(_MODEL_ENV) or DEFAULT_MODEL).strip()


def _timeout() -> float:
    raw = (os.environ.get(_TIMEOUT_ENV) or "").strip()
    try:
        value = float(raw) if raw else 120.0
    except ValueError:
        value = 120.0
    return max(1.0, min(value, 900.0))


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK


def selected() -> bool:
    """Whether the assistant's provider resolves to Ollama right now."""
    raw = (os.environ.get("KALOS_LLM_PROVIDER") or "").strip().lower()
    if raw:
        return raw == "ollama"
    desktop = (os.environ.get("KALOS_DESKTOP") or "").strip().lower()
    return desktop in {"1", "true", "yes", "on"}


class OllamaProvider:
    """Status slot plus a minimal chat call for a local Ollama server."""

    name = "ollama"

    def available(self) -> bool:
        return selected()

    def local(self) -> bool:
        return is_loopback_url(ollama_url())

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.available(),
            required_env=(),
            missing_env=(),
            capability=_CAPABILITY,
            fallback=_FALLBACK,
        )

    def chat(self, system: str, prompt: str, *, model: str | None = None) -> str:
        """One non-streamed chat turn at temperature 0. Raises `OllamaUnavailable`."""
        url = ollama_url()
        if urlparse(url).scheme not in {"http", "https"}:
            raise OllamaUnavailable("the Ollama URL must be http or https")
        body = json.dumps({
            "model": model or ollama_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked just above
            f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_timeout()) as response:  # noqa: S310 - configured local URL
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise OllamaUnavailable("the Ollama server did not answer") from exc
        content = (data.get("message") or {}).get("content") if isinstance(data, dict) else None
        if not isinstance(content, str):
            raise OllamaUnavailable("the Ollama server gave no answer text")
        return content


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "OllamaProvider",
    "OllamaUnavailable",
    "is_loopback_url",
    "ollama_model",
    "ollama_url",
    "selected",
]
