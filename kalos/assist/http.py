"""Shared HTTP responses for the assistant routes (one error envelope)."""
from __future__ import annotations

from fastapi.responses import JSONResponse

from kalos.assist.llm import assist_provider


def assist_disabled() -> bool:
    """True when `KALOS_LLM_PROVIDER=none`: the model-backed routes answer 404."""
    return assist_provider() == "none"


def assist_disabled_response() -> JSONResponse:
    return JSONResponse(
        {"error": "the assistant is turned off on this engine", "code": "assist_disabled"},
        status_code=404,
    )


def provider_unavailable_response() -> JSONResponse:
    return JSONResponse(
        {"error": "the assistant's model could not be reached", "code": "assist_provider_unavailable"},
        status_code=503,
    )


__all__ = [
    "assist_disabled",
    "assist_disabled_response",
    "provider_unavailable_response",
]
