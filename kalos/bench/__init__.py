"""Closed-loop benchmark: does the kalos BO loop actually beat space-filling?

The one honest question a bioprocess-optimization product must answer before it
is sold: given a fixed experiment budget, does the Bayesian-optimization loop
reach a good recipe in FEWER experiments than just running a space-filling
design (Latin Hypercube) or random sampling?

This package answers it on synthetic surfaces with KNOWN optima (so true simple
regret is measurable and the whole thing is reproducible from committed code),
while sweeping observation noise - the same measurement noise that governs
whether the model helps at all on real, noisy assay data.
"""
from typing import TYPE_CHECKING, Any

from .objectives import (  # noqa: F401
    MixedObjective,
    Objective,
    ackley,
    gaussian_bump,
    mixed_bump,
)

if TYPE_CHECKING:
    from .benchmark import run_benchmark, run_mixed_one, run_one, summarize  # noqa: F401

# `benchmark` imports torch (kalos.core.optimize). It is loaded on first use, so
# importing a torch-free subpackage such as `kalos.bench.faults` (whose fast tier
# must stay torch-free) does not pay for torch. `from kalos.bench import
# run_benchmark` works exactly as before.
_LAZY = frozenset({"run_benchmark", "run_mixed_one", "run_one", "summarize"})


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MixedObjective",
    "Objective",
    "ackley",
    "gaussian_bump",
    "mixed_bump",
    "run_benchmark",
    "run_mixed_one",
    "run_one",
    "summarize",
]
