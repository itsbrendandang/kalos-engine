"""Which model the assistant talks to, and the one call it makes.

`KALOS_LLM_PROVIDER` (shared with `kalos.normalize`):
  - `ollama`: a local Ollama server (`kalos.providers.ollama_provider`).
  - `anthropic`: the Anthropic API (needs `ANTHROPIC_API_KEY` and the
    `normalize` extra). Retrieved chunks are sent off the machine.
  - `none`: no model. `/api/assist/ask` and SOP revise answer 404; the guide,
    the index and SOP drafts keep working.
Unset means `ollama` in desktop mode (`KALOS_DESKTOP=1`) and `none`
otherwise: the assistant sends data off the machine only when someone set
`anthropic` on purpose. (`kalos.normalize` keeps its own `anthropic` default.)

A model here only ever receives retrieved chunks and the guide step, and its
reply is checked before anyone sees it (`kalos.assist.chat`, `.sop`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Protocol

from kalos.envconfig import env_enum
from kalos.providers.ollama_provider import (
    OllamaProvider,
    OllamaUnavailable,
    is_loopback_url,
    ollama_model,
    ollama_url,
)
from kalos.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
    OpenAICompatUnavailable,
    openai_model,
    openai_url,
)
from kalos.providers.openai_compatible_provider import (
    is_loopback_url as openai_is_loopback_url,
)

AssistProvider = Literal["ollama", "openai", "anthropic", "none"]
_PROVIDERS: tuple[AssistProvider, ...] = ("ollama", "openai", "anthropic", "none")
_ANTHROPIC_MODEL_ENV = "KALOS_ASSIST_ANTHROPIC_MODEL"
_DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


class ModelUnavailable(RuntimeError):
    """The selected model could not be reached or gave no usable reply."""


class ChatModel(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def local(self) -> bool:
        """Whether the model runs on this machine (nothing leaves it)."""
        ...

    def complete(self, system: str, prompt: str) -> str:
        """One reply to `prompt` under `system`. Raises `ModelUnavailable`."""
        ...


def assist_provider() -> AssistProvider:
    """The provider the assistant uses right now. A bad value raises ConfigError."""
    desktop = (os.environ.get("KALOS_DESKTOP") or "").strip().lower() in {"1", "true", "yes", "on"}
    default: AssistProvider = "ollama" if desktop else "none"
    return env_enum("KALOS_LLM_PROVIDER", _PROVIDERS, default)  # type: ignore[return-value]


@dataclass
class OllamaChat:
    model: str
    url: str
    provider: str = "ollama"

    @property
    def local(self) -> bool:
        return is_loopback_url(self.url)

    def complete(self, system: str, prompt: str) -> str:
        try:
            return OllamaProvider().chat(system, prompt, model=self.model)
        except OllamaUnavailable as exc:
            raise ModelUnavailable(str(exc)) from exc


@dataclass
class OpenAICompatChat:
    model: str
    url: str
    provider: str = "openai"

    @property
    def local(self) -> bool:
        return openai_is_loopback_url(self.url)

    def complete(self, system: str, prompt: str) -> str:
        try:
            return OpenAICompatibleProvider().chat(system, prompt, model=self.model)
        except OpenAICompatUnavailable as exc:
            raise ModelUnavailable(str(exc)) from exc


@dataclass
class AnthropicChat:
    model: str
    provider: str = "anthropic"
    local: bool = False

    def complete(self, system: str, prompt: str) -> str:
        from kalos.normalize import credentials_available

        if not credentials_available():
            raise ModelUnavailable("no Anthropic API key is set")
        try:
            import anthropic  # optional: the `normalize` extra
        except ImportError as exc:
            raise ModelUnavailable("the anthropic package is not installed") from exc
        try:
            client = anthropic.Anthropic(timeout=60.0, max_retries=1)
            reply = client.messages.create(
                model=self.model, max_tokens=1200, system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure is "unavailable"
            raise ModelUnavailable("the Anthropic API did not answer") from exc
        parts = [getattr(block, "text", "") for block in getattr(reply, "content", [])]
        text = "".join(p for p in parts if isinstance(p, str))
        if not text:
            raise ModelUnavailable("the Anthropic API gave no answer text")
        return text


def model_info() -> dict[str, object]:
    """`{provider, model, local}` for the status route; never a credential."""
    provider = assist_provider()
    if provider == "ollama":
        return {"provider": "ollama", "model": ollama_model(), "local": is_loopback_url(ollama_url())}
    if provider == "openai":
        return {"provider": "openai", "model": openai_model(), "local": openai_is_loopback_url(openai_url())}
    if provider == "anthropic":
        model = os.environ.get(_ANTHROPIC_MODEL_ENV) or _DEFAULT_ANTHROPIC_MODEL
        return {"provider": "anthropic", "model": model, "local": False}
    return {"provider": "none", "model": None, "local": True}


def get_chat_model() -> ChatModel | None:
    """The configured chat model, or None when the provider is `none`. A FastAPI
    dependency, so tests can override it with a fake."""
    provider = assist_provider()
    if provider == "ollama":
        return OllamaChat(model=ollama_model(), url=ollama_url())
    if provider == "openai":
        return OpenAICompatChat(model=openai_model(), url=openai_url())
    if provider == "anthropic":
        return AnthropicChat(model=os.environ.get(_ANTHROPIC_MODEL_ENV) or _DEFAULT_ANTHROPIC_MODEL)
    return None


__all__ = [
    "AnthropicChat",
    "AssistProvider",
    "ChatModel",
    "ModelUnavailable",
    "OllamaChat",
    "OpenAICompatChat",
    "assist_provider",
    "get_chat_model",
    "model_info",
]
