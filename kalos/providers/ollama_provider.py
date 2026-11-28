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
`KALOS_OLLAMA_URL` (default http://localhost:11434). Which model answers is
chosen by the cascade (`kalos.providers.cascade`): a fast model
(`KALOS_LLM_FAST_MODEL`) and a quality model (`KALOS_LLM_QUALITY_MODEL`), both
MIT or Apache-2.0 licensed by default. `KALOS_OLLAMA_MODEL`, when set, is the
quality model (the one model older engines used for everything).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from kalos.providers.base import ProviderStatus
from kalos.providers.text import strip_thinking

_STDLIB_URLOPEN = urllib.request.urlopen

_URL_ENV = "KALOS_OLLAMA_URL"
_TIMEOUT_ENV = "KALOS_OLLAMA_TIMEOUT_S"
DEFAULT_URL = "http://localhost:11434"
# The quality model is the default for a single-model call (benchmarks/llm:
# the measured winner). Apache-2.0. See kalos.providers.cascade for both tiers.
DEFAULT_MODEL = "qwen2.5:7b-instruct"
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
    """The quality-tier model (the single model a non-cascaded call uses)."""
    from kalos.providers.cascade import quality_model

    return quality_model()


def _tagged_model(model: str) -> str:
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


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


def _open_response(
    url_or_request: str | urllib.request.Request, *, timeout: float
) -> Any:
    if urllib.request.urlopen is not _STDLIB_URLOPEN:
        return urllib.request.urlopen(url_or_request, timeout=timeout)  # noqa: S310
    request = (
        urllib.request.Request(url_or_request, method="GET")  # noqa: S310
        if isinstance(url_or_request, str)
        else url_or_request
    )
    return _opener().open(request, timeout=timeout)


def _open(request: urllib.request.Request, timeout: float) -> bytes:
    with _opener().open(request, timeout=timeout) as response:
        return bytes(response.read())


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK


def health_check(base_url: str, *, timeout: float = 5.0) -> bool:
    """Return True iff `base_url/api/tags` answers HTTP 200 without using proxies."""
    url = base_url.strip().rstrip("/") + "/api/tags"
    if urlparse(url).scheme not in {"http", "https"}:
        return False
    try:
        with _open_response(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except Exception:  # noqa: BLE001 - any failure means unreachable
        return False


def generate(
    base_url: str,
    model: str,
    prompt: str,
    *,
    timeout: float = 60.0,
    temperature: float = 0.0,
    json_mode: bool = True,
) -> str | None:
    """POST `prompt` to `base_url/api/generate` without proxies; None on any failure."""
    url = base_url.strip().rstrip("/") + "/api/generate"
    if urlparse(url).scheme not in {"http", "https"}:
        return None
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with _open_response(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            raw = response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - any failure means no response
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    text = data.get("response") if isinstance(data, dict) else None
    return text if isinstance(text, str) else None


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

    def chat(
        self, system: str, prompt: str, *, model: str | None = None, json_mode: bool = False,
    ) -> str:
        """One non-streamed chat turn at temperature 0. `json_mode` asks the
        server to constrain the reply to JSON (still validated by the caller).
        Raises `OllamaUnavailable`."""
        url = ollama_url()
        if urlparse(url).scheme not in {"http", "https"}:
            raise OllamaUnavailable("the Ollama URL must be http or https")
        payload: dict[str, object] = {
            "model": model or ollama_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "seed": 0},
        }
        if json_mode:
            payload["format"] = "json"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked just above
            f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            data = json.loads(_open(request, _timeout()).decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise OllamaUnavailable("the Ollama server did not answer") from exc
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaUnavailable("the Ollama server gave no answer text")
        content = strip_thinking(content)
        if not content:
            raise OllamaUnavailable("the Ollama server gave no answer text")
        return content

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Embedding vectors for `texts` from `/api/embed`, one per text, in
        order. Raises `OllamaUnavailable` (unreachable server, model not
        pulled, or a malformed reply)."""
        url = ollama_url()
        if urlparse(url).scheme not in {"http", "https"}:
            raise OllamaUnavailable("the Ollama URL must be http or https")
        body = json.dumps({"model": model, "input": texts, "truncate": True}).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme checked just above
            f"{url}/api/embed", data=body, headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            data = json.loads(_open(request, _timeout()).decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise OllamaUnavailable("the Ollama server did not answer") from exc
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if (
            not isinstance(vectors, list) or len(vectors) != len(texts)
            or not all(isinstance(v, list) and v for v in vectors)
        ):
            raise OllamaUnavailable("the Ollama server gave no embeddings")
        return [[float(x) for x in v] for v in vectors]

    def probe(self) -> dict[str, bool]:
        """Check the server and selected model, only when explicitly called."""
        url = ollama_url()
        if urlparse(url).scheme not in {"http", "https"}:
            return {"reachable": False, "model_present": False}
        request = urllib.request.Request(f"{url}/api/tags")  # noqa: S310
        try:
            data = json.loads(_open(request, 2.0))
            models = data.get("models", []) if isinstance(data, dict) else []
            present = any(
                isinstance(item, dict) and isinstance(item.get("name"), str)
                and _tagged_model(item["name"]) == _tagged_model(ollama_model())
                for item in models
            )
            return {"reachable": True, "model_present": present}
        except (OSError, ValueError, TypeError):
            return {"reachable": False, "model_present": False}


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
