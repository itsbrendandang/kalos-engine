"""Small fast-tier tasks: intent routing, short labels, plain wording.

Each task runs through the cascade (`kalos.providers.cascade`): the fast model
first, the quality model when the fast reply fails its check, and a fixed
deterministic answer when both fail. None of them produces a number, an
interval, a verdict or a proposal:

- `classify_intent`: which part of the assistant a question is for. The reply
  must be one of `INTENTS`; the offline fallback is a keyword rule.
- `short_label`: a two-to-six word label for a column or a group of runs. It
  may only use words from the engine text it was given (plus stop words), no
  digits at all; the fallback is the column name itself.
- `plain_wording`: an engine warning or refusal reworded for a bench
  scientist. It must keep every number of the engine's text exactly and add
  none, name no column the engine did not name, and use no banned or claim
  word the engine text does not use; the fallback is the engine's own text.
"""
from __future__ import annotations

import re
from typing import Any

from kalos.assist.index import tokens
from kalos.assist.words import banned_in, unsupported_claims
from kalos.providers.cascade import Cascade, CascadeResult, ValidationFailed
from kalos.providers.grounding import identifiers_in, number_keys, parse_json_object

INTENTS = ("campaign_data", "next_step", "sop", "docs", "setup", "out_of_scope")

_INTENT_SYSTEM = """You route a question to one part of a lab assistant for a bioprocess
optimization tool. Reply with ONE JSON object: {"intent": "<one of: campaign_data, next_step,
sop, docs, setup, out_of_scope>"}.
- campaign_data: about this campaign's runs, results, inputs, model checks or proposed recipes.
- next_step: what to do next in the campaign.
- sop: writing or editing a procedure (SOP) draft.
- docs: how the tool works, its settings or its routes.
- setup: which column is the target, inputs, units, column roles.
- out_of_scope: anything else (other labs, literature, general chemistry, the weather).
No other text."""

_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sop", ("sop", "procedure", "protocol draft")),
    ("next_step", ("next step", "what should i do", "what do i do next", "what next")),
    ("setup", ("column role", "which column", "unit", "target column", "setup")),
    ("docs", ("how does kalos", "setting", "env", "route", "api", "documentation")),
    ("campaign_data", ("run", "titer", "result", "recipe", "round", "measured", "predicted",
                       "input", "model", "best", "campaign")),
)


def _intent_offline(question: str) -> str:
    q = question.lower()
    for intent, words in _INTENT_KEYWORDS:
        if any(w in q for w in words):
            return intent
    return "out_of_scope"


def classify_intent(question: str, cascade: Cascade | None) -> CascadeResult[str]:
    """The intent of `question`, one of `INTENTS`."""
    def validate(reply: str) -> str:
        obj = parse_json_object(reply)
        intent = obj.get("intent")
        if not isinstance(intent, str) or intent.strip().lower() not in INTENTS:
            raise ValidationFailed("schema", "the intent is not one of the allowed intents")
        return intent.strip().lower()

    fallback = lambda: _intent_offline(question)  # noqa: E731
    if cascade is None:
        return CascadeResult(fallback(), "offline", [], offline=True)
    return cascade.run("intent", _INTENT_SYSTEM, f"Question: {question[:500]}", validate, fallback,
                       json_mode=True)


_LABEL_SYSTEM = """Write a short plain label (2 to 6 words) for a column of a lab run sheet.
Use only words from the description you are given. No digits, no units you were not given.
Reply with ONE JSON object: {"label": "..."}"""

_LABEL_STOP = frozenset({"of", "the", "a", "an", "and", "in", "per", "for", "at", "to", "by"})


def short_label(column: str, description: str, cascade: Cascade | None) -> CascadeResult[str]:
    """A short label for `column`, built only from words in `column` and
    `description` (engine text)."""
    allowed = set(tokens(column + " " + description)) | _LABEL_STOP

    def validate(reply: str) -> str:
        obj = parse_json_object(reply)
        label = obj.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValidationFailed("empty", "no label")
        label = " ".join(label.split())
        if re.search(r"\d", label):
            raise ValidationFailed("mismatched_number", "a label holds no digits")
        words = re.findall(r"[a-z]+", label.lower())
        if not 1 <= len(words) <= 6:
            raise ValidationFailed("schema", "a label is one to six words")
        if any(w not in allowed and not set(tokens(w)) <= allowed for w in words):
            raise ValidationFailed("unsupported_claim", "the label uses words the engine did not give")
        return label

    fallback = lambda: column  # noqa: E731
    if cascade is None:
        return CascadeResult(fallback(), "offline", [], offline=True)
    prompt = f"Column: {column}\nDescription: {description[:400]}"
    return cascade.run("label", _LABEL_SYSTEM, prompt, validate, fallback, json_mode=True)


_WORDING_SYSTEM = """Reword a message from a lab data tool for a bench scientist, in at most
two short plain sentences. Keep every number exactly as written and add no new number. Keep
column names exactly as written. Do not add advice, causes, promises or approvals.
Reply with ONE JSON object: {"text": "..."}"""

MAX_WORDING_CHARS = 400


def plain_wording(message: str, cascade: Cascade | None, *, columns: list[str] | None = None) -> CascadeResult[str]:
    """`message` (engine-written) in plain words. Every number of the reply is
    one of the message's numbers, and every number of the message is kept."""
    source_numbers = number_keys(message)
    known = identifiers_in(message) | set(columns or ())

    def validate(reply: str) -> str:
        obj = parse_json_object(reply)
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationFailed("empty", "no text")
        text = " ".join(text.split())
        if len(text) > MAX_WORDING_CHARS:
            raise ValidationFailed("schema", "too long")
        if number_keys(text) != source_numbers:
            raise ValidationFailed("mismatched_number", "the wording changed, dropped or added a number")
        if identifiers_in(text) - known:
            raise ValidationFailed("unknown_column", "the wording names a column the message did not")
        if banned_in(text) or unsupported_claims(text, [message]):
            raise ValidationFailed("unsupported_claim", "the wording adds a claim the message does not make")
        return text

    fallback = lambda: message  # noqa: E731
    if cascade is None:
        return CascadeResult(fallback(), "offline", [], offline=True)
    return cascade.run("wording", _WORDING_SYSTEM, f"Message: {message[:1000]}", validate, fallback,
                       json_mode=True)


def report(result: CascadeResult[Any]) -> dict[str, Any]:
    return {"value": result.value, **result.report()}


__all__ = ["INTENTS", "classify_intent", "plain_wording", "report", "short_label"]
