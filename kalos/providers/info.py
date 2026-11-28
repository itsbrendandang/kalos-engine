"""Provider selection and status metadata (libs layer, no assist dependency)."""
from __future__ import annotations

import os
from typing import Literal

from kalos.envconfig import env_enum
from kalos.paid import no_paid_calls
from kalos.providers.ollama_provider import (
    OllamaProvider,
    is_loopback_url,
    ollama_model,
    ollama_url,
)
from kalos.providers.openai_compatible_provider import (
    OpenAICompatibleProvider,
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


def assist_provider() -> AssistProvider:
    """The provider the assistant uses right now. A bad value raises ConfigError."""
    desktop = (os.environ.get("KALOS_DESKTOP") or "").strip().lower() in {"1", "true", "yes", "on"}
    default: AssistProvider = "ollama" if desktop else "none"
    return env_enum("KALOS_LLM_PROVIDER", _PROVIDERS, default)  # type: ignore[return-value]


def model_info(*, probe: bool = False) -> dict[str, object]:
    """`{provider, model, local}` for the status route; never a credential."""
    provider = assist_provider()
    if provider == "ollama":
        from kalos.providers.cascade import models_info

        return {
            "provider": "ollama",
            "model": ollama_model(),
            "local": is_loopback_url(ollama_url()),
            **models_info(),
            **(OllamaProvider().probe() if probe else {}),
        }
    if provider == "openai":
        return {
            "provider": "openai",
            "model": openai_model(),
            "local": openai_is_loopback_url(openai_url()),
            **(OpenAICompatibleProvider().probe() if probe else {}),
        }
    if provider == "anthropic":
        from kalos.normalize import credentials_available

        model = os.environ.get(_ANTHROPIC_MODEL_ENV) or _DEFAULT_ANTHROPIC_MODEL
        return {
            "provider": "anthropic",
            "model": model,
            "local": False,
            "available": credentials_available() and not no_paid_calls(),
        }
    return {"provider": "none", "model": None, "local": True}


__all__ = [
    "_ANTHROPIC_MODEL_ENV",
    "_DEFAULT_ANTHROPIC_MODEL",
    "AssistProvider",
    "assist_provider",
    "model_info",
]
