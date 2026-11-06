"""kalos/flags.py: one registry for every KALOS_* feature switch.

1. Every `KALOS_*` name the engine code uses is registered (FLAGS or SETTINGS).
2. No module outside flags.py / portal/config.py / envconfig.py names a feature
   switch directly. Today's offenders are in `ALLOWED_DIRECT_READS`, which may
   only SHRINK: adding a pair fails review, and a pair that no longer reads the
   switch must be removed (the test fails until it is).
3. `describe()` returns names and booleans only.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from kalos import flags

ROOT = Path(__file__).resolve().parents[1]
KALOS = ROOT / "kalos"
_NAME = re.compile(r"^KALOS_[A-Z0-9_]+$")

# Modules allowed to read switches: the registry and the runtime config readers.
OWNERS = {"kalos/flags.py", "kalos/portal/config.py", "kalos/envconfig.py"}

# (module, switch) pairs that read a switch directly today. May only shrink.
ALLOWED_DIRECT_READS = {
    ("kalos/normalize/config.py", "KALOS_LLM_PROVIDER"),
    ("kalos/portal/desktop.py", "KALOS_ALLOW_OPEN_ACCESS"),
    ("kalos/portal/desktop.py", "KALOS_DESKTOP"),
    ("kalos/portal/desktop.py", "KALOS_LLM_PROVIDER"),
    ("kalos/portal/jobs.py", "KALOS_JOBS_ENABLED"),
    ("kalos/portal/pipeline.py", "KALOS_ANALYSIS_CACHE"),
}


def _name_literals() -> dict[str, set[str]]:
    """module path -> the exact `KALOS_*` string constants in it (docstrings
    and messages that merely mention a name are not exact constants)."""
    found: dict[str, set[str]] = {}
    for path in sorted(KALOS.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=rel)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _NAME.match(node.value):
                found.setdefault(rel, set()).add(node.value)
    return found


def test_every_kalos_variable_is_registered():
    used = set().union(*_name_literals().values())
    unregistered = sorted(used - set(flags.FLAGS) - set(flags.SETTINGS))
    assert not unregistered, f"register these in kalos/flags.py (FLAGS if a switch, else SETTINGS): {unregistered}"
    assert not set(flags.FLAGS) & set(flags.SETTINGS)


def test_no_new_direct_reads_of_feature_switches():
    direct = {
        (rel, name)
        for rel, names in _name_literals().items()
        if rel not in OWNERS
        for name in names
        if name in flags.FLAGS
    }
    new = sorted(direct - ALLOWED_DIRECT_READS)
    assert not new, f"read these through kalos.flags instead: {new}"
    gone = sorted(ALLOWED_DIRECT_READS - direct)
    assert not gone, f"fixed: remove from ALLOWED_DIRECT_READS in tests/test_flags.py: {gone}"


def test_defaults():
    env: dict[str, str] = {}
    assert flags.check(env) == []
    assert flags.is_on("KALOS_DESKTOP", env) is False
    assert flags.is_on("KALOS_JOBS_ENABLED", env) is True
    assert flags.value("KALOS_ANALYSIS_CACHE", env) == 4
    assert flags.value("KALOS_LLM_PROVIDER", env) == "anthropic"
    assert flags.features(env) == {"assist": False, "skills": False, "benchling": False, "twin": False}


def test_values_and_aliases():
    env = {"KALOS_DESKTOP": "Yes", "KALOS_ANALYSIS_CACHE": "0", "KALOS_LLM_PROVIDER": "none",
           "KALOS_SKILLS": "1", "KALOS_INTEGRATIONS": "benchling", "KALOS_FEATURES": "twin, assist"}
    assert flags.is_on("KALOS_DESKTOP", env)
    assert not flags.is_on("KALOS_ANALYSIS_CACHE", env)
    assert not flags.is_on("KALOS_LLM_PROVIDER", env)
    assert flags.features(env) == {"assist": True, "skills": True, "benchling": True, "twin": True}


@pytest.mark.parametrize(("name", "raw"), [
    ("KALOS_DESKTOP", "ture"), ("KALOS_LLM_PROVIDER", "openai"),
    ("KALOS_ANALYSIS_CACHE", "-1"), ("KALOS_FEATURES", "assist,mcp"),
])
def test_unknown_values_fail_closed(name, raw):
    with pytest.raises(flags.FlagError):
        flags.value(name, {name: raw})
    assert flags.check({name: raw})


def test_describe_is_names_and_booleans_only():
    secret = "s3cr3t-value-never-returned"
    env = {name: secret for name in flags.SETTINGS} | {"KALOS_DESKTOP": "1", "KALOS_FEATURES": "nope"}
    described = flags.describe(env)
    assert set(described) == {"switches", "features"}
    assert set(described["switches"]) == set(flags.FLAGS)
    assert set(described["features"]) == set(flags.FEATURES)
    for group in described.values():
        assert all(isinstance(v, bool) for v in group.values())
    assert described["switches"]["KALOS_DESKTOP"] is True
    assert described["switches"]["KALOS_FEATURES"] is False  # invalid counts as off, never raises
    assert secret not in repr(described)
