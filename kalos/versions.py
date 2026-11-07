"""Pinned numerical-stack versions and what is installed (torch-free).

The GP numbers depend on the torch / botorch / gpytorch build (BENCHMARK.md
records a torch upgrade that changed BO regrets on the same seeds). The `ml`
extra in pyproject.toml pins these exact versions, and they match uv.lock;
tests/test_torch_pin.py checks all three agree and that the runtime matches,
unless KALOS_ALLOW_TORCH_DRIFT=1.

`library_versions()` reads `importlib.metadata`, so importing this module never
loads torch. Every analysis (`proposal_provenance.library_versions`) and every
bench artifact records it.
"""
from __future__ import annotations

from importlib import metadata

# Keep in step with pyproject.toml [project.optional-dependencies] ml and uv.lock.
PINNED_ML: dict[str, str] = {"torch": "2.13.0", "botorch": "0.18.1", "gpytorch": "1.15.2"}

RECORDED_LIBRARIES: tuple[str, ...] = (
    "torch", "botorch", "gpytorch", "linear-operator", "numpy", "scipy", "pandas",
)


def library_versions(names: tuple[str, ...] = RECORDED_LIBRARIES) -> dict[str, str | None]:
    """Installed version per distribution name; None when not installed."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def public_version(version: str | None) -> str | None:
    """`2.13.0+cpu` -> `2.13.0` (the CPU wheel's local label is not a different
    release; `torch==2.13.0` accepts it, PEP 440)."""
    return None if version is None else version.split("+", 1)[0]


def stack_record() -> dict[str, object]:
    """The block a bench artifact embeds: installed versions, the pins, and
    whether the installed torch/botorch/gpytorch match them."""
    installed = library_versions()
    drift = {
        k: installed.get(k) for k, v in PINNED_ML.items() if public_version(installed.get(k)) != v
    }
    return {"library_versions": installed, "pinned": dict(PINNED_ML), "pin_drift": drift}
