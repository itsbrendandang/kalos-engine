"""The arm adapter interface (PROTOCOL.md section 6).

An ARM is anything that, given the runs measured so far, proposes the next
batch, puts an interval on each proposal, and says whether its model can be
trusted. The harness calls `Arm.propose(ctx)` once per round and never edits
the engine. Two ways to build a candidate:

1. Engine-patch arm (required to SHIP). Subclass
   `kalos.bench.ml_v2.arms.kalos_ref.KalosEngineArm` and override
   `engine_patch()` to return a context manager that swaps engine internals
   in-process for the duration of one analysis (e.g. replace
   `kalos.core.surrogate.Surrogate`, `kalos.core.optimize.propose_batch`, or
   `kalos.portal.analysis.calibrate_selection_aware`). The production
   `_analyze` path runs unchanged around the patch, so the trust layer, the
   refusals and the planted-fault suite all see the candidate. Because the
   change is expressible as a patch, the engine change it implies is concrete.
2. Standalone arm (baselines such as BayBE, Ax or a plain BoTorch loop).
   Implement `Arm` directly. Standalone arms are scored on every metric they
   provide but cannot ship (`shippable = False`): they have no engine path for
   the planted-fault suite.

Load an arm from a spec string `"package.module:factory"`; the factory takes
one `dict` of configuration and returns the arm. The spec, configuration and
the git commit are recorded with every result. A factory must be importable in
a fresh (spawned) worker process and must be deterministic given
`ctx.seed` and its configuration.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .space import Space


@dataclass(frozen=True)
class RoundContext:
    """Everything an arm may use to propose a round. Read-only by convention.

    `sheet` holds the measured runs: every input in `space.inputs`, every
    covariate, an optional `batch` column (null_batch_column scenario) and the
    target column `target`. `round_index[i]` is the campaign round of row i (0 =
    initial design), the same thing the engine reads from
    `df.attrs["kalos_round_index"]`; it is None when the sheet has no round
    structure the user would know about. `roles` are the declared column roles in
    `ColumnRoles` vocabulary (`categoricals`, `strain`, `strain_policy`,
    `strain_level`, `covariates`, `groups`). `candidates` is the not-yet-run pool
    in pool replay (inputs only, no outcomes; arms may use it or ignore it; the
    harness snaps every proposal to the nearest candidate anyway).
    """

    suite: str
    task: str
    seed: int
    round: int
    sheet: pd.DataFrame
    target: str
    space: Space
    q: int
    levels: tuple[float, ...]
    roles: Mapping[str, Any] = field(default_factory=dict)
    round_index: tuple[int, ...] | None = None
    goal: str = "maximize"
    candidates: pd.DataFrame | None = None

    @property
    def arm_seed(self) -> int:
        """A seed for the arm's own randomness, distinct per task, seed and round."""
        h = hashlib.sha256(f"{self.task}|{self.seed}|{self.round}".encode()).hexdigest()
        return int(h[:8], 16)


@dataclass
class Band:
    lo: float | None
    hi: float | None

    @property
    def available(self) -> bool:
        return self.lo is not None and self.hi is not None and np.isfinite(self.lo) and np.isfinite(self.hi)


@dataclass
class Proposal:
    recipe: dict[str, Any]
    pred: float | None = None
    bands: dict[float, Band] = field(default_factory=dict)  # level -> band
    label: str = ""  # free text, e.g. "trusted" / "exploration_untrusted"


@dataclass
class Verdict:
    """`trusted`: the arm tells the user its model is reliable enough to act on
    (the engine: `proposal_trust.status == "trusted"`). `withheld`: no batch."""

    trusted: bool
    state: str = ""
    withheld: bool = False
    cv_spearman: float | None = None
    clears_floor: bool | None = None


# band_at(df_of_recipes, level) -> (lo, hi) arrays, or None when unavailable.
BandFn = Callable[[pd.DataFrame, float], "tuple[np.ndarray, np.ndarray] | None"]
PredictFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class ArmOutput:
    proposals: list[Proposal]
    verdict: Verdict
    band_at: BandFn | None = None  # interval at arbitrary recipes (held-out runs)
    predict: PredictFn | None = None  # model mean at arbitrary recipes (prospective ranking)
    info: dict[str, Any] = field(default_factory=dict)  # JSON-serializable extras


@runtime_checkable
class Arm(Protocol):
    name: str
    shippable: bool

    def propose(self, ctx: RoundContext) -> ArmOutput: ...

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        """In-process patch that turns the engine into this arm (planted-fault
        suite). Standalone arms return `contextlib.nullcontext()` and set
        `shippable = False`."""
        ...


def load_arm(spec: str, config: Mapping[str, Any] | None = None) -> Arm:
    if ":" not in spec:
        raise ValueError(f"arm spec must be 'module:factory', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(mod_name), fn_name)
    arm = factory(dict(config or {}))
    if not isinstance(arm, Arm):
        raise TypeError(f"{spec} did not return an Arm (needs name, shippable, propose, engine_patch)")
    return arm


def config_hash(spec: str, config: Mapping[str, Any] | None) -> str:
    blob = json.dumps({"spec": spec, "config": dict(config or {})}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@contextlib.contextmanager
def patched(obj: Any, attr: str, value: Any) -> Iterator[None]:
    """Helper for `engine_patch`: set `obj.attr = value`, restore on exit."""
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)
