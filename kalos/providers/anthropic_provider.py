"""Slot for the Anthropic API key that already gates `kalos.normalize`'s LLM tier.

This provider does not add a new capability - it reports the status of a
credential the codebase already reads (`kalos/normalize/config.py`,
`credentials_available()`). `kalos.normalize` is imported lazily, inside the
methods that need it, and the `anthropic` SDK itself is never imported at
module scope here - it is an optional extra (`.[normalize]`), and this
module must stay importable without it installed.
"""
from __future__ import annotations

from kalos.paid import no_paid_calls
from kalos.providers.base import ProviderStatus

_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

_CAPABILITY = (
    "LLM-assisted column and unit normalization - mapping a messy client run "
    "sheet onto the canonical schema, tolerant of unusual or ambiguous headers."
)
_FALLBACK = (
    "The deterministic offline synonym+units mapper in kalos.normalize "
    "(offline_plan) still runs, so normalization works, just less tolerant "
    "of unusual headers."
)


class AnthropicProvider:
    """Credential check for `ANTHROPIC_API_KEY`, delegating to
    `kalos.normalize.credentials_available` (the existing check) rather than
    re-reading the env var independently."""

    name = "anthropic"

    def available(self) -> bool:
        from kalos.normalize import credentials_available

        return credentials_available() and not no_paid_calls()

    def status(self) -> ProviderStatus:
        from kalos.normalize import credentials_available

        ok = self.available()
        return ProviderStatus(
            name=self.name,
            available=ok,
            required_env=(_API_KEY_ENV_VAR,),
            missing_env=() if credentials_available() else (_API_KEY_ENV_VAR,),
            disabled_by=("KALOS_NO_PAID_CALLS",) if no_paid_calls() else (),
            capability=_CAPABILITY,
            fallback=_FALLBACK,
        )


__all__ = ["AnthropicProvider"]
