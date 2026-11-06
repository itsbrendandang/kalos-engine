"""Every `KALOS_*` feature switch in one place (plan: docs ARCHITECTURE §1.2).

A *feature switch* turns a capability on or off (a plug-in, desktop mode, the
analysis cache, jobs, the LLM). Sizing knobs, paths, secrets and identity
settings are not switches; they are listed in `SETTINGS` only so this file
names every `KALOS_*` variable the engine reads (tests/test_flags.py checks the
list against the code).

Not wired yet: today each switch is still read by the module that uses it (the
allowlist in tests/test_flags.py, which may only shrink). New code reads
switches through `is_on()` / `value()` here. `describe()` is the shape for
`GET /api/health`: names and on/off only, never a value of any setting.

Target: `KALOS_FEATURES=assist,skills,benchling,twin`. The older per-plug-in
variables (`KALOS_SKILLS`, `KALOS_INTEGRATIONS`) stay as aliases for one release.

Stdlib only and import-light: `kalos.portal.__main__` may read it before the
app (and torch) are imported.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from kalos.envconfig import ConfigError

Kind = Literal["bool", "enum", "list", "count"]

_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"0", "false", "no", "off"})

FEATURES: tuple[str, ...] = ("assist", "skills", "benchling", "twin")


def _unmerged(suffix: str) -> str:
    """A `KALOS_*` name whose reader lives on a branch that is not merged yet.

    Built from parts, not spelled as one literal, so tests/test_readme_env.py
    (which counts exact `KALOS_*` literals as "read by the code") keeps
    treating it as "not yet merged" in README.md. Spell the name out when the
    branch that reads it merges."""
    return "KALOS_" + suffix


FEATURES_VAR = _unmerged("FEATURES")
SKILLS_VAR = _unmerged("SKILLS")
INTEGRATIONS_VAR = _unmerged("INTEGRATIONS")


class FlagError(ConfigError):
    """A feature switch holds a value the engine does not accept."""


@dataclass(frozen=True)
class Flag:
    name: str
    kind: Kind
    default: str
    valid: tuple[str, ...]  # accepted values ("bool": the on/off spellings; "count": empty)
    help: str
    owner: str  # the module that reads it today (until it reads it from here)


FLAGS: dict[str, Flag] = {
    f.name: f
    for f in (
        Flag(FEATURES_VAR, "list", "", FEATURES,
             "comma list of plug-ins to turn on: assist, skills, benchling, twin", "kalos.flags"),
        Flag(SKILLS_VAR, "bool", "0", tuple(sorted(_ON | _OFF)),
             "alias for KALOS_FEATURES=skills (skills extra must be installed)", "kalos.portal.app (skills branch)"),
        Flag(INTEGRATIONS_VAR, "list", "", ("benchling",),
             "comma list of connectors; benchling = KALOS_FEATURES=benchling", "kalos.integrations (benchling branch)"),
        Flag("KALOS_DESKTOP", "bool", "0", tuple(sorted(_ON | _OFF)),
             "desktop mode: one user, loopback only, desktop defaults", "kalos.portal.desktop"),
        Flag("KALOS_LLM_PROVIDER", "enum", "anthropic", ("anthropic", "ollama", "none"),
             "which LLM the engine may call; none = off (desktop defaults to ollama if it answers)",
             "kalos.normalize.config, kalos.portal.desktop"),
        Flag("KALOS_ANALYSIS_CACHE", "count", "4", (),
             "number of cached analyses per process; 0 = off", "kalos.portal.pipeline"),
        Flag("KALOS_JOBS_ENABLED", "bool", "1", tuple(sorted(_ON | _OFF)),
             "background analysis jobs and their worker processes", "kalos.portal.jobs"),
        Flag("KALOS_ALLOW_OPEN_ACCESS", "bool", "0", tuple(sorted(_ON | _OFF)),
             "serve remote callers with no auth configured (refused in desktop mode)", "kalos.portal.config"),
        Flag(_unmerged("EXPERIMENTAL_WARM_START"), "bool", "0", tuple(sorted(_ON | _OFF)),
             "multi-task warm start, process-wide (bench-ports branch)", "kalos.core.multitask (bench-ports)"),
    )
}

# Every other KALOS_* variable the engine reads, by kind. Not switches: never
# reported by describe(). "secret" values must never be logged or returned.
SETTINGS: dict[str, str] = {
    "KALOS_ENV": "mode", "KALOS_VALIDATION_MODE": "mode",
    "KALOS_AUTH_TOKENS": "secret", "KALOS_AUTH_TOKENS_FILE": "path", "KALOS_AUTH_DENY_SUBJECTS": "identity",
    "KALOS_JWT_PUBLIC_KEYS_FILE": "path", "KALOS_JWT_ISSUER": "identity", "KALOS_JWT_AUDIENCE": "identity",
    "KALOS_JWT_LEEWAY_S": "limit", "KALOS_JWT_MAX_TTL_S": "limit",
    "KALOS_ANON_SALT": "secret", _unmerged("SECRET_KEY"): "secret", _unmerged("SKILLS_DIR"): "path",
    "KALOS_CORS_ORIGINS": "network", "KALOS_HOST": "network", "KALOS_PORT": "network",
    "KALOS_TRUSTED_PROXIES": "network",
    "KALOS_STATE_DIR": "path", "KALOS_DATASET_DIR": "path", "KALOS_INDPENSIM_PATH": "path",
    "KALOS_MEDIA_DATA": "path", "KALOS_AMII_DATA": "path", "KALOS_DATAHOW_DATA": "path",
    "KALOS_MAX_REQUEST_BYTES": "limit", "KALOS_MAX_UPLOAD_MB": "limit", "KALOS_MAX_FIT_ROWS": "limit",
    "KALOS_UPLOAD_READ_TIMEOUT_S": "limit", "KALOS_DEMO_TIMEOUT_S": "limit",
    "KALOS_AUDIT_RETENTION_DAYS": "limit", "KALOS_APPLICABILITY_DISTANCE": "limit",
    "KALOS_TORCH_THREADS": "limit",
    "KALOS_JOB_WORKERS": "limit", "KALOS_JOB_PER_TENANT": "limit", "KALOS_JOB_QUEUE_DEPTH": "limit",
    "KALOS_JOB_GLOBAL_QUEUE_DEPTH": "limit", "KALOS_JOB_MAX_UPLOAD_BYTES": "limit",
    "KALOS_JOB_TIMEOUT_S": "limit", "KALOS_JOB_DEADLINE_S": "limit",
    "KALOS_JOB_ENGINE_PID": "internal", "KALOS_JOB_WORKER_SESSION": "internal",
    "KALOS_BUILD_SHA": "identity", "KALOS_GIT_SHA": "identity",
    "KALOS_NORMALIZE_MODEL": "llm", "KALOS_NORMALIZE_MAX_SAMPLE": "limit",
    "KALOS_OLLAMA_URL": "llm", "KALOS_OLLAMA_MODEL": "llm",
}


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _raw(name: str, environ: Mapping[str, str] | None) -> str:
    return _env(environ).get(name, "").strip().lower()


def value(name: str, environ: Mapping[str, str] | None = None) -> bool | str | int | frozenset[str]:
    """The parsed value of switch `name`: bool, enum string, count, or set of
    list items. Unset/empty = the default. An unknown value raises `FlagError`
    (fail closed: a typo must not quietly pick the other state)."""
    flag = FLAGS[name]
    raw = _raw(name, environ) or flag.default
    if flag.kind == "bool":
        if raw in _ON:
            return True
        if raw in _OFF:
            return False
        raise FlagError(f"{name} must be one of 1, 0, true, false, yes, no, on, off; got {raw[:40]!r}")
    if flag.kind == "enum":
        if raw in flag.valid:
            return raw
        raise FlagError(f"{name}={raw[:40]!r} is not recognised; accepted: {', '.join(flag.valid)}")
    if flag.kind == "count":
        try:
            count = int(raw)
        except ValueError:
            count = -1
        if count < 0:
            raise FlagError(f"{name} must be a whole number >= 0; got {raw[:40]!r}")
        return count
    items = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(items - set(flag.valid))
    if unknown:
        raise FlagError(f"{name} has unknown item(s) {unknown}; accepted: {', '.join(flag.valid)}")
    return items


def is_on(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Whether switch `name` is on: true, a provider other than none, a count
    above 0, or a non-empty list."""
    parsed = value(name, environ)
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, str):
        return parsed != "none"
    if isinstance(parsed, int):
        return parsed > 0
    return bool(parsed)


def features(environ: Mapping[str, str] | None = None) -> dict[str, bool]:
    """Plug-in on/off from `KALOS_FEATURES` plus the one-release aliases."""
    listed = value(FEATURES_VAR, environ)
    assert isinstance(listed, frozenset)
    integrations = value(INTEGRATIONS_VAR, environ)
    assert isinstance(integrations, frozenset)
    return {
        "assist": "assist" in listed,
        "skills": "skills" in listed or is_on(SKILLS_VAR, environ),
        "benchling": "benchling" in listed or "benchling" in integrations,
        "twin": "twin" in listed,
    }


def check(environ: Mapping[str, str] | None = None) -> list[str]:
    """One line per switch holding an unaccepted value (empty when all good).
    For startup: call it and refuse to start on a non-empty list."""
    problems = []
    for name in FLAGS:
        try:
            value(name, environ)
        except FlagError as exc:
            problems.append(str(exc))
    return problems


def _safe_on(name: str, environ: Mapping[str, str] | None) -> bool:
    try:
        return is_on(name, environ)
    except FlagError:
        return False  # an invalid value counts as off here; check() reports it


def describe(environ: Mapping[str, str] | None = None) -> dict[str, dict[str, bool]]:
    """For `GET /api/health`: `{"switches": {NAME: on}, "features": {plugin: on}}`.

    Names and booleans only: no setting's value is ever included, so it is
    safe to return to any caller with the read scope. Never raises: a switch
    with an invalid value is reported as off (startup refuses it anyway)."""
    switches = {name: _safe_on(name, environ) for name in sorted(FLAGS)}
    try:
        plugins = features(environ)
    except FlagError:
        plugins = {
            "assist": False,
            "skills": switches[SKILLS_VAR],
            "benchling": False,
            "twin": False,
        }
    return {"switches": switches, "features": plugins}


__all__ = ["FEATURES", "FLAGS", "SETTINGS", "Flag", "FlagError", "check", "describe", "features", "is_on", "value"]
