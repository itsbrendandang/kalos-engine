"""deploy/engine/Containerfile must install the torch version uv.lock resolves.

Before this pin the image resolved whatever the CPU index had newest
(2.14.1+cpu) while the lock, CI and the tests were on 2.13.0. These tests read
both files, so a bump in one without the other fails here.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTAINERFILE = ROOT / "deploy" / "engine" / "Containerfile"


def _locked_version(package: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    matches = [p["version"] for p in lock["package"] if p["name"] == package]
    assert len(matches) == 1, f"{package} should resolve to exactly one version in uv.lock"
    return matches[0]


def test_containerfile_pins_torch_to_the_locked_version():
    text = CONTAINERFILE.read_text()
    m = re.search(r"^ARG TORCH_VERSION=(\S+)$", text, re.M)
    assert m, "Containerfile must declare ARG TORCH_VERSION"
    assert m.group(1) == _locked_version("torch")
    assert 'torch==${TORCH_VERSION}' in text
    assert "--index-url https://download.pytorch.org/whl/cpu" in text


def test_image_healthcheck_polls_readyz_with_a_start_period():
    """The image HEALTHCHECK must be readiness (store writable, disk free), not
    liveness, and must allow a cold torch import before a failed probe counts."""
    text = CONTAINERFILE.read_text()
    m = re.search(r"^HEALTHCHECK (.*?)\n\s+CMD (.*)$", text, re.M | re.S)
    assert m, "Containerfile must declare a HEALTHCHECK"
    flags, cmd = m.group(1), m.group(2)
    assert "http://127.0.0.1:8050/readyz" in cmd
    assert "/healthz" not in cmd
    start = re.search(r"--start-period=(\d+)s", flags)
    assert start and int(start.group(1)) >= 60


def test_compose_engine_healthcheck_polls_readyz_with_a_start_period():
    compose = (ROOT / "deploy" / "compose.yaml").read_text()
    block = compose.split("  engine:", 1)[1].split("\n  web:", 1)[0]
    assert "healthcheck:" in block
    assert "http://127.0.0.1:8050/readyz" in block
    assert "/healthz" not in block
    assert re.search(r"start_period: \d+s", block)
