"""Kalos platform: BoTorch Bayesian optimization for bioprocess development, with
NVIDIA BioNeMo / ESM-2 protein features and a FastAPI portal.
"""
from .core.conformal import split_conformal
from .core.drivers import bootstrap_spearman, rank_drivers
from .core.gates import GatesConfig, PromotionResult, check_gates
from .core.splits import assert_no_group_leakage, make_splits, row_hash_groups
from .data.anonymizer import Anonymizer

__version__ = "0.1.0"

# These transitively import torch/botorch/gpytorch (~220 MB of RSS). Loaded
# lazily via `__getattr__` (PEP 562) so `import kalos` - and everything that
# imports kalos, including the idle `--watch` poller and portal boot - does
# not pay the torch tax until one of these is actually touched.
_LAZY = {
    "Surrogate": (".core.surrogate", "Surrogate"),
    "propose": (".core.optimize", "propose"),
    "MultiObjectiveSurrogate": (".core.multiobjective", "MultiObjectiveSurrogate"),
    "propose_multiobjective": (".core.multiobjective", "propose_multiobjective"),
    "grouped_cv_spearman": (".core.evaluation", "grouped_cv_spearman"),
}

__all__ = [
    "Anonymizer",
    "GatesConfig",
    "MultiObjectiveSurrogate",
    "PromotionResult",
    "Surrogate",
    "assert_no_group_leakage",
    "bootstrap_spearman",
    "check_gates",
    "grouped_cv_spearman",
    "make_splits",
    "propose",
    "propose_multiobjective",
    "rank_drivers",
    "row_hash_groups",
    "split_conformal",
]


def __getattr__(name: str):
    try:
        module_path, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_path, __name__), attr)
    globals()[name] = value  # cache: repeat access hits the module dict, not __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
