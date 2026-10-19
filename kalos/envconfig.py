"""Fail-closed reading of enumerated and bounded `KALOS_*` environment settings.

One rule for every enum-valued setting: an unknown value RAISES `ConfigError`
instead of falling back to a default. A typo such as `KALOS_VALIDATION_MODE=stict`
or `KALOS_ENV=prod` would otherwise silently select the permissive side of the
setting, which is exactly the failure a startup check exists to stop.

Unset and empty mean "use the default", matching the sizing knobs elsewhere
(`kalos.portal.uploads`): compose and k8s templates pass `NAME=${NAME:-}`.

This module is stdlib-only and import-light on purpose: `kalos.normalize` and
`kalos.portal` both read settings through it.
"""
from __future__ import annotations

import os
from collections.abc import Iterable


class ConfigError(RuntimeError):
    """A `KALOS_*` setting holds a value the engine does not accept."""


def env_enum(
    name: str,
    accepted: Iterable[str],
    default: str,
    *,
    error: type[ConfigError] = ConfigError,
) -> str:
    """The lower-cased, stripped value of `name` when it is one of `accepted`,
    `default` when it is unset or empty, and otherwise raises `error`.

    `error` lets a caller raise its own subclass (e.g. `ProductionConfigError`)
    so existing `except` clauses keep matching. The message is one line and
    truncates the rejected value, so it is safe to print at startup.
    """
    options = tuple(accepted)
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in options:
        return raw
    listed = ", ".join(repr(o) for o in options)
    raise error(
        f"{name}={raw[:40]!r} is not recognised. Accepted values: {listed} "
        f"(case-insensitive), or unset/empty for {default!r}. Unknown values are "
        "refused rather than defaulted, so a typo cannot silently change the setting "
        "(docs/HARDENING.md)."
    )


def env_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    error: type[ConfigError] = ConfigError,
) -> int:
    """The integer value of `name` within `[minimum, maximum]`, `default` when
    unset or empty. A non-integer or out-of-range value raises `error`."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = None
    if value is None or not minimum <= value <= maximum:
        raise error(
            f"{name}={raw[:40]!r} must be a whole number from {minimum} to {maximum} "
            f"(or unset for {default}) (docs/HARDENING.md)."
        )
    return value
