"""Fail-closed `KALOS_*` enum settings (review H3).

An unknown value must RAISE, never fall back to the permissive default:
`KALOS_VALIDATION_MODE=stict` used to silently become `warn`. The startup check
(`python -m kalos.portal`) turns the same error into a one-line refusal, exit 2.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from kalos.envconfig import ConfigError, env_enum, env_int
from kalos.normalize.config import load_config
from kalos.portal.analysis import validation_mode
from kalos.portal.config import (
    ProductionConfigError,
    check_env_settings,
    deployment_env,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("KALOS_ENV", "KALOS_VALIDATION_MODE", "KALOS_LLM_PROVIDER",
                 "KALOS_AUDIT_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)


def test_env_enum_unknown_value_raises_one_line_naming_the_variable(monkeypatch):
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "stict")
    with pytest.raises(ConfigError) as exc:
        env_enum("KALOS_VALIDATION_MODE", ("warn", "strict"), "warn")
    message = str(exc.value)
    assert "KALOS_VALIDATION_MODE='stict'" in message
    assert "\n" not in message


def test_env_enum_truncates_a_hostile_value(monkeypatch):
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "x" * 500)
    with pytest.raises(ConfigError) as exc:
        env_enum("KALOS_VALIDATION_MODE", ("warn", "strict"), "warn")
    assert len(str(exc.value)) < 400


@pytest.mark.parametrize("raw, expected", [("", "warn"), ("  ", "warn"), ("Strict", "strict"), (" warn ", "warn")])
def test_validation_mode_accepts_known_values_and_unset(monkeypatch, raw, expected):
    monkeypatch.setenv("KALOS_VALIDATION_MODE", raw)
    assert validation_mode() == expected


def test_validation_mode_typo_raises_instead_of_failing_open(monkeypatch):
    # Old behaviour: 'stict' -> 'warn' (the permissive mode) without a word.
    monkeypatch.setenv("KALOS_VALIDATION_MODE", "stict")
    with pytest.raises(ConfigError, match="KALOS_VALIDATION_MODE"):
        validation_mode()


def test_llm_provider_typo_raises(monkeypatch):
    monkeypatch.setenv("KALOS_LLM_PROVIDER", "nonee")  # a typo of "none" must not reach anthropic
    with pytest.raises(ConfigError, match="KALOS_LLM_PROVIDER"):
        load_config()


def test_env_is_still_refused_with_the_production_error_class(monkeypatch):
    monkeypatch.setenv("KALOS_ENV", "prod")
    with pytest.raises(ProductionConfigError):
        deployment_env()
    assert issubclass(ProductionConfigError, ConfigError)


def test_env_int_bounds_and_default(monkeypatch):
    assert env_int("KALOS_X", 7, minimum=1, maximum=10) == 7
    monkeypatch.setenv("KALOS_X", "10")
    assert env_int("KALOS_X", 7, minimum=1, maximum=10) == 10
    for bad in ("0", "11", "abc", "1.5"):
        monkeypatch.setenv("KALOS_X", bad)
        with pytest.raises(ConfigError):
            env_int("KALOS_X", 7, minimum=1, maximum=10)


def test_check_env_settings_passes_on_defaults():
    check_env_settings()


@pytest.mark.parametrize("name, value", [
    ("KALOS_ENV", "prod"),
    ("KALOS_VALIDATION_MODE", "stict"),
    ("KALOS_LLM_PROVIDER", "olama"),
    ("KALOS_AUDIT_RETENTION_DAYS", "0"),
])
def test_check_env_settings_refuses_each_bad_enum_or_bound(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        check_env_settings()


def test_entrypoint_exits_2_with_one_line_on_a_typo(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("KALOS_")}
    env.update({"KALOS_STATE_DIR": str(tmp_path), "KALOS_VALIDATION_MODE": "stict"})
    out = subprocess.run(
        [sys.executable, "-m", "kalos.portal"], env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 2
    lines = [line for line in out.stderr.splitlines() if line.strip()]
    assert len(lines) == 1, out.stderr
    assert "KALOS_VALIDATION_MODE" in lines[0]
    assert "Traceback" not in out.stderr
