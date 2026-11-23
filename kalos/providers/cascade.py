"""A two-tier model cascade: a fast model first, a quality model when it fails.

The rule this module exists to enforce: a language model never produces a
number, an interval, a verdict or a proposal that reaches a user unchecked.
Every call goes through `Cascade.run` with a `validate` function that checks
the reply against engine data (column names that exist, run ids and sources
that exist, numbers equal to the engine's). A reply that fails is never shown:

1. tasks that start on the FAST tier (header mapping, short labels, intent
   routing, plain wording of engine warnings and refusals, setup suggestions)
   escalate to the QUALITY tier when the fast reply fails validation or the
   fast model does not answer;
2. tasks that start on the QUALITY tier (assistant answers with citations, SOP
   drafts, literature summaries) go straight there;
3. when every tier fails, `fallback()` runs: the caller's existing offline,
   deterministic path. The result says so in `provider_path` (for example
   `fast>quality>offline`), so a response can show which path produced it.

Models (Ollama, local by default; licences checked, see docs/ASSISTANT.md):

| tier    | env                       | default      | licence    |
|---------|---------------------------|--------------|------------|
| fast    | `KALOS_LLM_FAST_MODEL`    | `phi4-mini`  | MIT        |
| quality | `KALOS_LLM_QUALITY_MODEL` | `qwen2.5:7b` | Apache-2.0 |

`KALOS_OLLAMA_MODEL` (the one model older engines used) still works: set on
its own, it is the only model (both tiers, no cascade); set together with
`KALOS_LLM_FAST_MODEL`, it is the quality model.
`KALOS_LLM_CASCADE=0` turns escalation off (a fast task then uses the fast
model only, and falls back offline when it fails).

Which provider runs is still `KALOS_LLM_PROVIDER` (unchanged): this module
only decides which MODEL of that provider answers. A remote provider (the
Anthropic API) stays opt-in; with it both tiers are the same remote model.

Env vars are read at call time, never at import time.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

log = logging.getLogger("kalos.providers.cascade")

Tier = Literal["fast", "quality"]

# Measured defaults (benchmarks/llm/RESULTS.md). Both are MIT / Apache-2.0.
DEFAULT_FAST_MODEL = "phi4-mini"
DEFAULT_QUALITY_MODEL = "qwen2.5:7b"
FAST_ENV = "KALOS_LLM_FAST_MODEL"
QUALITY_ENV = "KALOS_LLM_QUALITY_MODEL"
LEGACY_ENV = "KALOS_OLLAMA_MODEL"
CASCADE_ENV = "KALOS_LLM_CASCADE"

# Which tier each task starts on. Escalation always goes fast -> quality.
TASK_TIER: dict[str, Tier] = {
    "header_mapping": "fast",
    "label": "fast",
    "intent": "fast",
    "wording": "fast",
    "setup_suggestions": "fast",
    "answer": "quality",
    "sop": "quality",
    "summary": "quality",
}

# Why a reply was not shown. Stable codes: a response and the eval report them.
VALIDATION_REASONS = (
    "invalid_json",
    "schema",
    "unknown_column",
    "nonexistent_citation",
    "mismatched_number",
    "unsupported_claim",
    "empty",
)


class ValidationFailed(Exception):
    """The reply failed a check against engine data. `reason` is one of
    `VALIDATION_REASONS`; `detail` is a short engine-written note (never the
    model's text, which may hold anything)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class TierUnavailable(RuntimeError):
    """The tier's model could not be reached or gave no reply text."""


class TierModel(Protocol):
    """One model behind one tier. `complete` raises any exception on failure;
    the cascade treats every exception as "this tier did not answer"."""

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def local(self) -> bool: ...

    def complete(self, system: str, prompt: str) -> str: ...


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _legacy_single() -> str:
    """`KALOS_OLLAMA_MODEL` when it is the only model setting: older engines
    used that one model for everything, and still do (no cascade)."""
    legacy = _env(LEGACY_ENV)
    return legacy if legacy and not _env(FAST_ENV) and not _env(QUALITY_ENV) else ""


def fast_model() -> str:
    return _env(FAST_ENV) or _legacy_single() or DEFAULT_FAST_MODEL


def quality_model() -> str:
    return _env(QUALITY_ENV) or _env(LEGACY_ENV) or DEFAULT_QUALITY_MODEL


def escalation_enabled() -> bool:
    return _env(CASCADE_ENV).lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Attempt:
    """One tier's try: which model, what happened, how long it took."""

    tier: str
    provider: str
    model: str
    outcome: Literal["ok", "invalid", "unavailable"]
    reason: str | None = None
    detail: str | None = None
    seconds: float = 0.0


T = TypeVar("T")


@dataclass
class CascadeResult(Generic[T]):
    value: T
    provider_path: str
    attempts: list[Attempt] = field(default_factory=list)
    offline: bool = False

    @property
    def model(self) -> str | None:
        ok = [a for a in self.attempts if a.outcome == "ok"]
        return ok[-1].model if ok else None

    def report(self) -> dict[str, Any]:
        """JSON-safe block for a response: the path, each attempt, and whether
        the offline fallback produced the value."""
        return {
            "provider_path": self.provider_path,
            "offline_fallback": self.offline,
            "model": self.model,
            "attempts": [
                {k: v for k, v in asdict(a).items() if k != "seconds"} | {"seconds": round(a.seconds, 3)}
                for a in self.attempts
            ],
        }


def _complete(model: TierModel, system: str, prompt: str, json_mode: bool) -> str:
    if json_mode:
        complete_json = getattr(model, "complete_json", None)
        if callable(complete_json):
            return str(complete_json(system, prompt))
    return str(model.complete(system, prompt))


@dataclass
class Cascade:
    """Fast and quality models of one provider. Either may be None (no model
    for that tier). When both are the same model object or model name, a fast
    task runs once on it and does not retry the same model."""

    fast: TierModel | None
    quality: TierModel | None
    escalate: bool = True

    def tiers_for(self, task: str, start: Tier | None = None) -> list[tuple[str, TierModel]]:
        first = start or TASK_TIER.get(task, "quality")
        out: list[tuple[str, TierModel]] = []
        if first == "fast" and self.fast is not None:
            out.append(("fast", self.fast))
        if self.quality is not None and (first == "quality" or self.escalate or self.fast is None):
            same = bool(out) and (
                out[0][1] is self.quality
                or (out[0][1].provider, out[0][1].model) == (self.quality.provider, self.quality.model)
            )
            if not same:
                out.append(("quality", self.quality))
        return out

    def run(
        self,
        task: str,
        system: str,
        prompt: str,
        validate: Callable[[str], T],
        fallback: Callable[[], T],
        *,
        json_mode: bool = False,
        start: Tier | None = None,
    ) -> CascadeResult[T]:
        """Try each tier in order; return the first reply `validate` accepts.
        `validate` returns the checked value or raises `ValidationFailed`.
        When every tier fails, return `fallback()` with `offline=True`."""
        attempts: list[Attempt] = []
        path: list[str] = []
        for tier, model in self.tiers_for(task, start):
            path.append(tier)
            t0 = time.perf_counter()
            try:
                reply = _complete(model, system, prompt, json_mode)
            except Exception as exc:  # noqa: BLE001 - any failure: this tier did not answer
                attempts.append(Attempt(tier, model.provider, model.model, "unavailable",
                                        "unavailable", type(exc).__name__, time.perf_counter() - t0))
                log.info("cascade %s: %s tier did not answer", task, tier)
                continue
            try:
                value = validate(reply)
            except ValidationFailed as exc:
                attempts.append(Attempt(tier, model.provider, model.model, "invalid",
                                        exc.reason, exc.detail[:200] or None, time.perf_counter() - t0))
                log.info("cascade %s: %s tier failed validation (%s)", task, tier, exc.reason)
                continue
            attempts.append(Attempt(tier, model.provider, model.model, "ok",
                                    seconds=time.perf_counter() - t0))
            return CascadeResult(value, ">".join(path), attempts, offline=False)
        path.append("offline")
        return CascadeResult(fallback(), ">".join(path), attempts, offline=True)

    def all_unavailable(self, result: CascadeResult[Any]) -> bool:
        """True when the result is offline only because no tier answered at all
        (as opposed to a reply that failed the checks)."""
        return bool(result.attempts) and all(a.outcome == "unavailable" for a in result.attempts)


@dataclass
class OllamaTier:
    """One Ollama model as a cascade tier."""

    model: str
    url: str
    provider: str = "ollama"

    @property
    def local(self) -> bool:
        from kalos.providers.ollama_provider import is_loopback_url

        return is_loopback_url(self.url)

    def complete(self, system: str, prompt: str) -> str:
        from kalos.providers.ollama_provider import OllamaProvider

        return OllamaProvider().chat(system, prompt, model=self.model)

    def complete_json(self, system: str, prompt: str) -> str:
        from kalos.providers.ollama_provider import OllamaProvider

        return OllamaProvider().chat(system, prompt, model=self.model, json_mode=True)


def ollama_cascade(*, fast: str | None = None, quality: str | None = None,
                   escalate: bool | None = None) -> Cascade:
    """The Ollama cascade from the environment (or explicit model names)."""
    from kalos.providers.ollama_provider import ollama_url

    url = ollama_url()
    return Cascade(
        fast=OllamaTier(fast or fast_model(), url),
        quality=OllamaTier(quality or quality_model(), url),
        escalate=escalation_enabled() if escalate is None else escalate,
    )


def single_model_cascade(model: TierModel | None) -> Cascade | None:
    """A cascade with one model on both tiers (a remote provider, or a test
    fake): no escalation to the same model, offline fallback after it."""
    if model is None:
        return None
    return Cascade(fast=model, quality=model, escalate=False)


def models_info() -> dict[str, Any]:
    """The configured tiers, for status routes. Names only."""
    return {"fast_model": fast_model(), "quality_model": quality_model(),
            "escalation": escalation_enabled()}


__all__ = [
    "CASCADE_ENV",
    "DEFAULT_FAST_MODEL",
    "DEFAULT_QUALITY_MODEL",
    "FAST_ENV",
    "QUALITY_ENV",
    "TASK_TIER",
    "VALIDATION_REASONS",
    "Attempt",
    "Cascade",
    "CascadeResult",
    "OllamaTier",
    "TierModel",
    "TierUnavailable",
    "ValidationFailed",
    "escalation_enabled",
    "fast_model",
    "models_info",
    "ollama_cascade",
    "quality_model",
    "single_model_cascade",
]
