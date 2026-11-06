"""Slot for a local or self-hosted OpenAI-compatible chat server.

This is the seam for an OpenAI-style `/v1/chat/completions` endpoint - most
local runtimes expose one (llama.cpp's server, LM Studio, vLLM, and Hermes'
llama-server all do). It sits alongside `ollama_provider`: same local-first
posture, same fixed error messages that never carry prompt text, same lazy
env reads, but it speaks the OpenAI chat API instead of Ollama's native one.

No SDK: the HTTP client is stdlib `urllib` (no new dependency), mirroring the
Ollama path. An API key is optional - a bare local server needs none, a keyed
one reads `KALOS_OPENAI_API_KEY`. `available()` reports whether this provider
is SELECTED (`KALOS_LLM_PROVIDER=openai`), not whether the server answers;
that is checked only when a request is made, so a status call opens no socket.

`local()` is True only when the configured URL points at this machine
(localhost, 127.0.0.1, ::1). A remote URL means prompts leave the machine,
and the status says so.

Env vars:
  `KALOS_OPENAI_BASE_URL` (default http://localhost:8080/v1)
  `KALOS_OPENAI_MODEL`    (no default; must be set to select a model)
  `KALOS_OPENAI_API_KEY`  (optional; sent as `Authorization: Bearer` when set)
  `KALOS_OPENAI_TIMEOUT_S` (default 120, clamped to [1, 900])
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

from kalos.providers.base import ProviderStatus

_URL_ENV = "KALOS_OPENAI_BASE_URL"
_MODEL_ENV = "KALOS_OPENAI_MODEL"
_KEY_ENV = "KALOS_OPENAI_API_KEY"
_TIMEOUT_ENV = "KALOS_OPENAI_TIMEOUT_S"
DEFAULT_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = ""
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}

_CAPABILITY = (
    "A model behind an OpenAI-compatible server answers assistant questions and "
    "revises SOP draft prose; nothing is sent off the machine when the URL is local."
)
_FALLBACK = (
    "Without a model the assistant still gives the next step for each campaign and "
    "drafts SOPs from the record; asking questions is turned off."
)


class OpenAICompatUnavailable(RuntimeError):
    """The OpenAI-compatible server could not be reached or gave no usable
    answer. The message is fixed and carries no prompt text."""


def openai_url() -> str:
    return (os.environ.get(_URL_ENV) or DEFAULT_URL).strip().rstrip("/")


def openai_model() -> str:
    return (os.environ.get(_MODEL_ENV) or DEFAULT_MODEL).strip()


def _api_key() -> str:
    return (os.environ.get(_KEY_ENV) or "").strip()


def _timeout() -> float:
    raw = (os.environ.get(_TIMEOUT_ENV) or "").strip()
    try:
        value = float(raw) if raw else 120.0
    except ValueError:
        value = 120.0
    return max(1.0, min(value, 900.0))


def _opener() -> urllib.request.OpenerDirector:
    """An opener that ignores HTTP(S)_PROXY: a prompt for a local model must
    never be routed through a proxy (which would send it off the machine)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(request: urllib.request.Request, timeout: float) -> bytes:
    with _opener().open(request, timeout=timeout) as response:
        return bytes(response.read())


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK


def selected() -> bool:
    """Whether the assistant's provider resolves to this one right now."""
    raw = (os.environ.get("KALOS_LLM_PROVIDER") or "").strip().lower()
    return raw == "openai"


class OpenAICompatibleProvider:
    """Status slot plus a minimal chat call for an OpenAI-compatible server."""

    name = "openai"

    def available(self) -> bool:
        return selected()

    def local(self) -> bool:
        return is_loopback_url(openai_url())

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
        """One non-streamed chat turn at temperature 0. Raises
        `OpenAICompatUnavailable` on any transport or shape failure."""
        url = openai_url()
        if urlparse(url).scheme not in {"http", "https"}:
            raise OpenAICompatUnavailable("the OpenAI base URL must be http or https")
        chosen = model or openai_model()
        if not chosen:
            raise OpenAICompatUnavailable("no model is set (KALOS_OPENAI_MODEL)")
        body = json.dumps({
            "model": chosen,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        key = _api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(  # noqa: S310 - scheme checked just above
            f"{url}/chat/completions", data=body, headers=headers, method="POST",
        )
        try:
            data = json.loads(_open(request, _timeout()).decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise OpenAICompatUnavailable("the OpenAI-compatible server did not answer") from exc
        content = None
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    # Some reasoning servers put the final text in
                    # `content` and the chain-of-thought in
                    # `reasoning_content`; we only ever read `content`.
        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatUnavailable("the OpenAI-compatible server gave no answer text")
        return content


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_URL",
    "OpenAICompatUnavailable",
    "OpenAICompatibleProvider",
    "is_loopback_url",
    "openai_model",
    "openai_url",
    "selected",
]
