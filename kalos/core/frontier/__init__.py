"""Torch-free frontier surrogate primitives built from first principles."""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .acquisition import (
        ActionBatch,
        ActionSlot,
        AdaptiveTuRBO,
        JointQLogNEI,
        TrustRegionState,
        VOIAllocator,
    )
    from .hierarchical_gp import HierarchicalBatchGP, MultiScaleTransferGP
    from .shift_conformal import AdaptiveConformalState, ShiftWeightedConformal
    from .warped_sparse_gp import WarpedSparseGP

_LAZY: dict[str, tuple[str, str]] = {
    "WarpedSparseGP": (".warped_sparse_gp", "WarpedSparseGP"),
    "HierarchicalBatchGP": (".hierarchical_gp", "HierarchicalBatchGP"),
    "MultiScaleTransferGP": (".hierarchical_gp", "MultiScaleTransferGP"),
    "JointQLogNEI": (".acquisition", "JointQLogNEI"),
    "TrustRegionState": (".acquisition", "TrustRegionState"),
    "AdaptiveTuRBO": (".acquisition", "AdaptiveTuRBO"),
    "ActionSlot": (".acquisition", "ActionSlot"),
    "ActionBatch": (".acquisition", "ActionBatch"),
    "VOIAllocator": (".acquisition", "VOIAllocator"),
    "ShiftWeightedConformal": (".shift_conformal", "ShiftWeightedConformal"),
    "AdaptiveConformalState": (".shift_conformal", "AdaptiveConformalState"),
}

__all__ = [
    "ActionBatch",
    "ActionSlot",
    "AdaptiveConformalState",
    "AdaptiveTuRBO",
    "HierarchicalBatchGP",
    "JointQLogNEI",
    "MultiScaleTransferGP",
    "ShiftWeightedConformal",
    "TrustRegionState",
    "VOIAllocator",
    "WarpedSparseGP",
]


def __getattr__(name: str) -> Any:
    try:
        module_path, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_path, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
