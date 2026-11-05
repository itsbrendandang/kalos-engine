"""The generator registry: every source of virtual (non-measured) rows, by id.

A generator is a seeded function that returns a table of runs (design columns
plus outcome columns) for a validated parameter set. Each one is registered with
an id, a version, the origin of its output, its default parameters, and the
source files whose content makes up its code hash. Nothing produces virtual rows
outside this registry (docs/VIRTUAL_RUNS.md).

Registering a generator (for example a new module in `kalos/bench/`)::

    from kalos.twin.generators import register_generator

    @register_generator(
        "my_surface", version="1", origin="synthetic",
        description="...", defaults={"noise": 0.1},
    )
    def _generate(params, rng, n):
        ...  # return a DataFrame with n rows, using only `rng` for randomness

`discover()` imports `kalos.bench` modules whose source mentions
`register_generator` (a text check, so modules that do not register are never
imported and their torch imports are not paid), plus `kalos.twin.builtin` for the
generators that wrap existing code. A module can therefore register itself
without any edit to this file.

Determinism: `seed_sequence(...)` builds `np.random.SeedSequence` from
`[tenant, campaign, twin_version, job]` (strings are mapped to integers by
SHA-256), and `generate(...)` hands the generator one `PCG64` stream from it.
The stored record keeps the entropy, the initial PCG64 state, the solver settings
and the library versions, so the same inputs on the same build give the same rows
(tests/test_twin_generators.py, golden replay).

Torch-free.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import pkgutil
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kalos.twin.origin import (
    ORIGIN_ATTR,
    ORIGIN_COLUMN,
    VIRTUAL_ORIGINS,
    Origin,
    OriginArray,
    parse_origin,
)

GeneratorFn = Callable[[dict[str, Any], np.random.Generator, int], pd.DataFrame]

MAX_ROWS_PER_BATCH = 500
GENERATOR_ID_CHARS = 64
_LIBRARIES = ("numpy", "scipy", "pandas")


class GeneratorError(ValueError):
    """Unknown generator, bad parameters, or a generator that broke its contract.
    The message is authored here and safe to return to an API caller."""


@dataclass(frozen=True)
class GeneratorSpec:
    """One registered generator. `params` given by a caller are checked against
    `defaults`: unknown keys are refused, and each value must have the type of its
    default (ints may stand in for floats; `null` only for keys in `nullable`) and
    pass `validate`, if given."""

    id: str
    version: str
    origin: Origin
    description: str
    fn: GeneratorFn
    defaults: Mapping[str, Any] = field(default_factory=dict)
    source_files: tuple[Path, ...] = ()
    solver: Mapping[str, Any] = field(default_factory=dict)
    design_columns: tuple[str, ...] = ()
    outcome_columns: tuple[str, ...] = ()
    validate: Callable[[dict[str, Any]], None] | None = None
    available: Callable[[], bool] = lambda: True
    nullable: tuple[str, ...] = ()

    def resolve_params(self, params: Mapping[str, Any] | None) -> dict[str, Any]:
        given = dict(params or {})
        unknown = sorted(set(given) - set(self.defaults))
        if unknown:
            # The names are the caller's input and are not echoed back.
            raise GeneratorError(
                f"generator {self.id!r} does not take {len(unknown)} of the given parameter(s); "
                "see its defaults for the accepted names"
            )
        out = dict(self.defaults)
        for key, value in given.items():
            if value is None and key in self.nullable:
                out[key] = None
                continue
            out[key] = _coerce(self.id, key, value, self.defaults[key])
        if self.validate is not None:
            self.validate(out)
        return out

    def code_sha256(self) -> str:
        """SHA-256 over the generator's source files (path name + bytes), so an
        edit to the code changes the recorded version even if `version` was not
        bumped. A missing file hashes as its name only."""
        h = hashlib.sha256()
        for p in self.source_files:
            h.update(p.name.encode())
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(b"<missing>")
        return h.hexdigest()

    def full_version(self) -> str:
        """`<version>+<first 12 hex of code_sha256>`, e.g. `1+3f2a9c0d1e2b`."""
        return f"{self.version}+{self.code_sha256()[:12]}"

    def info(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.full_version(),
            "origin": str(self.origin),
            "description": self.description,
            "defaults": dict(self.defaults),
            "design_columns": list(self.design_columns),
            "outcome_columns": list(self.outcome_columns),
            "solver": dict(self.solver),
            "available": bool(self.available()),
        }


def _coerce(gen_id: str, key: str, value: Any, default: Any) -> Any:
    bad = GeneratorError(f"generator {gen_id!r}: parameter {key!r} has the wrong type")
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise bad
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise bad
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad
        if not math.isfinite(float(value)):
            raise GeneratorError(f"generator {gen_id!r}: parameter {key!r} must be finite")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise bad
        return value
    if default is None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise bad
        if value is not None and not math.isfinite(float(value)):
            raise GeneratorError(f"generator {gen_id!r}: parameter {key!r} must be finite")
        return None if value is None else float(value)
    raise bad


# --- the registry ------------------------------------------------------------ #

_REGISTRY: dict[str, GeneratorSpec] = {}
_DISCOVERED = False


def register(spec: GeneratorSpec) -> GeneratorSpec:
    """Add `spec`. Re-registering the same id with the same function is a no-op
    (a module imported twice); a different function under a taken id raises."""
    if not spec.id or len(spec.id) > GENERATOR_ID_CHARS or not all(
        c.isalnum() or c in "_.-" for c in spec.id
    ):
        raise GeneratorError("generator id must be 1-64 characters of [A-Za-z0-9_.-]")
    if spec.origin not in VIRTUAL_ORIGINS:
        raise GeneratorError("a generator's origin must be 'simulated' or 'synthetic'")
    existing = _REGISTRY.get(spec.id)
    if existing is not None:
        if existing.fn is spec.fn or (
            getattr(existing.fn, "__qualname__", None) == getattr(spec.fn, "__qualname__", None)
            and getattr(existing.fn, "__module__", None) == getattr(spec.fn, "__module__", None)
        ):
            _REGISTRY[spec.id] = spec
            return spec
        raise GeneratorError(f"generator id {spec.id!r} is already registered")
    _REGISTRY[spec.id] = spec
    return spec


def register_generator(
    gen_id: str,
    *,
    version: str,
    origin: Origin | str = Origin.SYNTHETIC,
    description: str,
    defaults: Mapping[str, Any] | None = None,
    source_files: tuple[Path | str, ...] | None = None,
    solver: Mapping[str, Any] | None = None,
    design_columns: tuple[str, ...] = (),
    outcome_columns: tuple[str, ...] = (),
    validate: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[GeneratorFn], GeneratorFn]:
    """Decorator form of `register`. `source_files` defaults to the file that
    defines the function."""

    def deco(fn: GeneratorFn) -> GeneratorFn:
        files = source_files
        if files is None:
            code = getattr(fn, "__code__", None)
            files = (code.co_filename,) if code is not None else ()
        register(GeneratorSpec(
            id=gen_id, version=version, origin=parse_origin(origin), description=description,
            fn=fn, defaults=dict(defaults or {}), source_files=tuple(Path(f) for f in files),
            solver=dict(solver or {}), design_columns=tuple(design_columns),
            outcome_columns=tuple(outcome_columns), validate=validate,
        ))
        return fn

    return deco


def discover() -> None:
    """Import the built-in wrappers and every `kalos.bench` module that registers
    a generator. Idempotent and cheap after the first call."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    importlib.import_module("kalos.twin.builtin")
    # find_spec does not execute kalos/bench/__init__.py (which imports torch);
    # only a module that registers a generator is imported.
    spec = importlib.util.find_spec("kalos.bench")
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    for loc in locations:
        for mod in pkgutil.iter_modules([loc]):
            path = Path(loc) / f"{mod.name}.py"
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "register_generator" in text:
                importlib.import_module(f"kalos.bench.{mod.name}")
    _DISCOVERED = True


def get_generator(gen_id: str) -> GeneratorSpec:
    discover()
    spec = _REGISTRY.get(gen_id) if isinstance(gen_id, str) else None
    if spec is None:
        raise GeneratorError("unknown generator id")
    return spec


def list_generators() -> list[GeneratorSpec]:
    discover()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


# --- seeding ----------------------------------------------------------------- #


def _to_int(part: str | int | None) -> int:
    """A SeedSequence entropy word for one seed component. Ints pass through
    (non-negative), None is 0, strings are the first 8 bytes of their SHA-256."""
    if part is None:
        return 0
    if isinstance(part, bool):
        raise GeneratorError("seed components must be strings, integers or None")
    if isinstance(part, int):
        if part < 0:
            raise GeneratorError("seed components must be non-negative")
        return part
    if isinstance(part, str):
        return int.from_bytes(hashlib.sha256(part.encode("utf-8")).digest()[:8], "big")
    raise GeneratorError("seed components must be strings, integers or None")


def seed_entropy(
    *, tenant: str, campaign_id: str, twin_version: str | None, job_id: str
) -> list[int]:
    """`[tenant, campaign, twin_version, job]` as SeedSequence entropy words
    (the plan's `SeedSequence([tenant, campaign, twin_version, job_id, k])`; the
    row index `k` is recorded per row, see `VirtualBatch`)."""
    return [_to_int(tenant), _to_int(campaign_id), _to_int(twin_version), _to_int(job_id)]


def seed_sequence(entropy: list[int]) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy)


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for dist in ("kalos", *_LIBRARIES):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def canonical_json(obj: Any) -> str:
    """The one serialization used for hashing: sorted keys, no spaces, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --- generation -------------------------------------------------------------- #


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


@dataclass(frozen=True)
class VirtualRow:
    """One generated run, ready to store in `virtual_runs`."""

    k: int
    design: dict[str, Any]
    outcome: dict[str, Any]

    @property
    def design_sha256(self) -> str:
        return sha256_json(self.design)


@dataclass(frozen=True)
class VirtualBatch:
    """The output of one `generate` call plus everything needed to replay it."""

    generator_id: str
    generator_version: str
    origin: Origin
    params: dict[str, Any]
    params_sha256: str
    seed_entropy: list[int]
    pcg64_state: dict[str, Any]
    solver: dict[str, Any]
    library_versions: dict[str, str | None]
    twin_version_id: str | None
    rows: list[VirtualRow]
    design_columns: list[str]
    outcome_columns: list[str]

    @property
    def frame(self) -> pd.DataFrame:
        """The rows as a DataFrame with a non-removable-by-accident `origin`
        column and `attrs["kalos_origin"]`, so `check_origin` refuses it."""
        records = [{**r.design, **r.outcome} for r in self.rows]
        df = pd.DataFrame(records, columns=[*self.design_columns, *self.outcome_columns])
        df.insert(0, ORIGIN_COLUMN, str(self.origin))
        df.attrs[ORIGIN_ATTR] = str(self.origin)
        return df

    @property
    def X(self) -> OriginArray:  # noqa: N802 - the engine's design-matrix name
        return OriginArray(
            [[r.design[c] for c in self.design_columns] for r in self.rows], self.origin
        )

    @property
    def y(self) -> OriginArray:
        first = self.outcome_columns[0]
        return OriginArray([r.outcome[first] for r in self.rows], self.origin)


def generate(
    gen_id: str,
    params: Mapping[str, Any] | None,
    n: int,
    *,
    tenant: str,
    campaign_id: str,
    job_id: str,
    twin_version_id: str | None = None,
) -> VirtualBatch:
    """Run generator `gen_id` for `n` rows with a seed derived from
    `(tenant, campaign_id, twin_version_id, job_id)`. Pure: writes nothing."""
    spec = get_generator(gen_id)
    if not spec.available():
        raise GeneratorError(f"generator {spec.id!r} is not available in this install")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_ROWS_PER_BATCH:
        raise GeneratorError(f"n must be an integer from 1 to {MAX_ROWS_PER_BATCH}")
    if spec.origin is Origin.SIMULATED and not twin_version_id:
        raise GeneratorError("simulated rows need a process model (twin) version")
    if spec.origin is Origin.SYNTHETIC and twin_version_id is not None:
        raise GeneratorError("synthetic rows are not tied to a process model (twin) version")
    resolved = spec.resolve_params(params)
    entropy = seed_entropy(
        tenant=tenant, campaign_id=campaign_id, twin_version=twin_version_id, job_id=job_id
    )
    bitgen = np.random.PCG64(seed_sequence(entropy))
    state = bitgen.state
    rng = np.random.Generator(bitgen)
    df = spec.fn(dict(resolved), rng, n)
    if not isinstance(df, pd.DataFrame) or len(df) != n:
        raise GeneratorError(f"generator {spec.id!r} did not return {n} rows")
    design_cols = [c for c in spec.design_columns if c in df.columns] or [
        c for c in df.columns if c not in spec.outcome_columns
    ]
    outcome_cols = [c for c in spec.outcome_columns if c in df.columns]
    if not outcome_cols:
        raise GeneratorError(f"generator {spec.id!r} returned no outcome column")
    if any(str(c).strip().lower() == ORIGIN_COLUMN for c in df.columns):
        raise GeneratorError(f"generator {spec.id!r} must not emit its own origin column")
    rows = [
        VirtualRow(
            k=int(i),
            design={str(c): _jsonable(df[c].iloc[i]) for c in design_cols},
            outcome={str(c): _jsonable(df[c].iloc[i]) for c in outcome_cols},
        )
        for i in range(n)
    ]
    return VirtualBatch(
        generator_id=spec.id,
        generator_version=spec.full_version(),
        origin=spec.origin,
        params=resolved,
        params_sha256=sha256_json(resolved),
        seed_entropy=entropy,
        pcg64_state=_jsonable_state(state),
        solver=dict(spec.solver),
        library_versions=library_versions(),
        twin_version_id=twin_version_id,
        rows=rows,
        design_columns=[str(c) for c in design_cols],
        outcome_columns=[str(c) for c in outcome_cols],
    )


def _jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    """`PCG64.state` as JSON (its 128-bit ints as decimal strings)."""
    inner = state.get("state", {})
    return {
        "bit_generator": state.get("bit_generator"),
        "state": {k: str(v) for k, v in inner.items()},
        "has_uint32": int(state.get("has_uint32", 0)),
        "uinteger": int(state.get("uinteger", 0)),
    }


__all__ = [
    "MAX_ROWS_PER_BATCH",
    "GeneratorError",
    "GeneratorSpec",
    "VirtualBatch",
    "VirtualRow",
    "canonical_json",
    "discover",
    "generate",
    "get_generator",
    "library_versions",
    "list_generators",
    "register",
    "register_generator",
    "seed_entropy",
    "seed_sequence",
    "sha256_json",
]
