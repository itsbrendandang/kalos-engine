"""Checks a model reply must pass against engine data before anyone sees it.

Each check raises `ValidationFailed` with a stable reason code (see
`kalos.providers.cascade.VALIDATION_REASONS`). They are deliberately strict and
deliberately simple: they never try to repair a reply, only to refuse it.

- `parse_json_object`: the reply is one JSON object (fences and prose around it
  are tolerated, nothing else).
- `check_columns`: every column a reply names is a column of the sheet.
- `check_numbers`: every number a reply states is one the engine gave it.
- `identifiers_in`: column-like tokens (`titer_g_L`, `DO_pct`) in free text, so
  a sentence that names a column the sheet does not have can be refused.
- `run_ids_in`: run ids (32 hex characters) in free text, so a sentence citing a
  run that does not exist can be refused.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from kalos.providers.cascade import ValidationFailed

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_IDENT = re.compile(r"(?<![\w./-])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![\w/-])")
_RUN_ID = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])")
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """The single JSON object in `text`. Raises `ValidationFailed('invalid_json')`."""
    stripped = _FENCE.sub("", text or "").strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValidationFailed("invalid_json", "no JSON object in the reply")
    try:
        obj = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValidationFailed("invalid_json", "the reply is not valid JSON") from exc
    if not isinstance(obj, dict):
        raise ValidationFailed("invalid_json", "the reply is not a JSON object")
    return obj


def number_keys(text: str) -> set[str]:
    """Every number in `text`, normalised so 2e9 equals 2000000000 and a
    decimal comma reads as a point."""
    out: set[str] = set()
    for raw in _NUMBER.findall(text or ""):
        s = raw.replace(",", ".")
        try:
            out.add(repr(float(s)))
        except ValueError:
            out.add(s)
    return out


def numbers_from(values: Iterable[Any]) -> set[str]:
    """Number keys for engine values (numbers and strings) given to a model."""
    out: set[str] = set()
    for v in values:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out.add(repr(float(v)))
            out |= number_keys(f"{v}")
        elif isinstance(v, str):
            out |= number_keys(v)
    return out


def check_numbers(text: str, allowed: set[str], *, what: str = "the reply") -> None:
    """Raise `mismatched_number` when `text` states a number not in `allowed`."""
    extra = number_keys(text) - allowed
    if extra:
        raise ValidationFailed("mismatched_number", f"{what} states {len(extra)} number(s) the engine did not give")


def check_columns(names: Iterable[str], known: Iterable[str]) -> None:
    """Raise `unknown_column` when any of `names` is not a known column."""
    known_set = {str(k) for k in known}
    unknown = [n for n in names if n not in known_set]
    if unknown:
        raise ValidationFailed("unknown_column", f"{len(unknown)} column name(s) not in the sheet")


def identifiers_in(text: str) -> set[str]:
    """Column-like tokens: words joined by underscores (`titer_g_L`)."""
    return set(_IDENT.findall(text or ""))


def run_ids_in(text: str) -> set[str]:
    return set(_RUN_ID.findall(text or ""))


__all__ = [
    "check_columns",
    "check_numbers",
    "identifiers_in",
    "number_keys",
    "numbers_from",
    "parse_json_object",
    "run_ids_in",
]
