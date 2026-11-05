"""The numerical stack is pinned (compmodel PLAN E4).

pyproject.toml's `ml` extra, uv.lock, kalos/versions.py PINNED_ML and the engine
image's TORCH_VERSION must name the same torch / botorch / gpytorch; and the
runtime must match the pin unless KALOS_ALLOW_TORCH_DRIFT=1 (then the test
skips and says what differs).
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest

from kalos.versions import PINNED_ML, library_versions, public_version, stack_record

ROOT = Path(__file__).resolve().parents[1]


def _lock_versions() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in lock["package"] if p["name"] in PINNED_ML}


def test_pyproject_pins_match_uv_lock_and_versions_module():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ml = meta["project"]["optional-dependencies"]["ml"]
    pins = dict(re.fullmatch(r"([a-z-]+)==([0-9.]+)", spec).groups() for spec in ml)
    assert pins == PINNED_ML == _lock_versions()


def test_engine_image_torch_matches_the_pin():
    containerfile = (ROOT / "deploy" / "engine" / "Containerfile").read_text(encoding="utf-8")
    m = re.search(r"^ARG TORCH_VERSION=(\S+)$", containerfile, re.M)
    assert m and m.group(1) == PINNED_ML["torch"]


def test_runtime_matches_the_pin():
    installed = {k: public_version(v) for k, v in library_versions(tuple(PINNED_ML)).items()}
    drift = {k: (installed[k], v) for k, v in PINNED_ML.items() if installed[k] != v}
    if drift and os.environ.get("KALOS_ALLOW_TORCH_DRIFT") == "1":
        pytest.skip(f"KALOS_ALLOW_TORCH_DRIFT=1; installed vs pinned: {drift}")
    assert not drift, (
        f"installed vs pinned {drift}; results depend on the build. "
        "Install the pinned versions (uv sync) or set KALOS_ALLOW_TORCH_DRIFT=1."
    )


def test_stack_record_shape():
    rec = stack_record()
    assert set(rec) == {"library_versions", "pinned", "pin_drift"}
    assert {"torch", "botorch", "gpytorch"} <= set(rec["library_versions"])
    assert public_version("2.13.0+cpu") == "2.13.0"
