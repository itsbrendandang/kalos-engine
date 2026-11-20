"""`optimize_acqf_mixed` with the per-level L-BFGS solves in the analysis worker pool.

botorch's `optimize_acqf_mixed` (0.18.1, botorch/optim/optimize.py:1148-1393) picks
a q-batch greedily: for each of the q slots it runs one full multi-start L-BFGS
solve per categorical assignment (`fixed_features_list`), keeps the best, adds it
to `X_pending` and moves on. On the 160-row example that is 5 slots x 3 levels =
15 solves, 83% of the analysis. The solves of one slot are independent.

This module computes the same batch with those solves in parallel:

1. For each slot, the initial conditions of every level are generated HERE, in
   level order, by the same `gen_batch_initial_conditions` call botorch makes, so
   they consume the global RNG exactly as the serial loop does (the L-BFGS solve
   between two levels draws nothing).
2. Only the RNG-free solves go to the pool, each with its initial conditions, run
   by `optimize_acqf` exactly as botorch's batch path runs them.
3. The best candidate is chosen with botorch's own rule (max over restarts, then
   the first level holding the overall max).

If any solve of a slot hit botorch's OptimizationWarning retry (which draws new
initial conditions from the global RNG), raised, or moved the RNG, that slot is
recomputed by botorch's own serial `optimize_acqf_mixed(q=1)` from the RNG state
saved before step 1, so the result is again what serial gives.

Used only on the plain path the engine takes (botorch 0.18.1, `raw_samples` set,
no equality/nonlinear constraints, no level that fixes every dimension, a
picklable acquisition, the pool in use, 2+ levels); anything else calls botorch.
"""
from __future__ import annotations

import pickle
import warnings
from typing import TYPE_CHECKING, Any

from . import parallel

if TYPE_CHECKING:
    import torch

# torch/botorch are imported inside the functions: this module stays torch-free at
# import time (tests/test_core_torch_contract.py); its only importer,
# kalos/core/optimize.py, loads them anyway.

# The botorch release whose `optimize_acqf_mixed` this module reproduces. Any other
# version runs botorch's own function (a pin bump must re-run the identity test,
# tests/test_parallel_identity.py, and update this).
BOTORCH_VERSION = "0.18.1"


def _botorch_version() -> str:
    import botorch

    return str(botorch.__version__)


def _solve_level(acq_blob: bytes, bounds: torch.Tensor, ics: torch.Tensor,
                 fixed_features: dict, num_restarts: int, ineq: list | None) -> tuple:
    """Worker side: botorch's batch solve of ONE level from given initial
    conditions. Returns (candidates, acq values, clean); `clean` is False when the
    solve raised, emitted an OptimizationWarning (serial would have retried from
    new random initial conditions) or touched the global RNG."""
    from botorch.exceptions.warnings import OptimizationWarning
    from botorch.optim import optimize_acqf

    acq = pickle.loads(acq_blob)  # noqa: S301 - pickled by the parent for its own pool
    start = parallel.capture_rng()
    warned = False

    def _gen(*args: Any, **kwargs: Any) -> Any:
        nonlocal warned
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", OptimizationWarning)
            out = _gen_candidates_scipy()(*args, **kwargs)
        for w in caught:
            if issubclass(w.category, OptimizationWarning):
                warned = True
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return out

    try:
        cands, vals = optimize_acqf(
            acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts,
            raw_samples=None, options={}, inequality_constraints=ineq,
            fixed_features=fixed_features, batch_initial_conditions=ics,
            return_best_only=False, gen_candidates=_gen, return_acq_values=True,
        )
    except Exception:  # noqa: BLE001 - any failure: the parent redoes the slot serially
        parallel.restore_rng(start)
        return None, None, False
    if not parallel.capture_rng().same_as(start):
        parallel.restore_rng(start)
        return None, None, False
    return cands, vals, not warned


def _gen_candidates_scipy() -> Any:
    from botorch.generation.gen import gen_candidates_scipy

    return gen_candidates_scipy


def _eligible(acq_function: Any, bounds: torch.Tensor, fixed_features_list: list,
              raw_samples: int | None, kwargs: dict) -> bool:
    from botorch.acquisition.acquisition import OneShotAcquisitionFunction

    if any(v is not None for v in kwargs.values()):
        return False
    if raw_samples is None or len(fixed_features_list) < 2:
        return False
    if isinstance(acq_function, OneShotAcquisitionFunction):
        return False
    d = bounds.shape[-1]
    if any(len({int(i) % d for i in ff}) == d for ff in fixed_features_list):
        return False  # botorch's all-features-fixed shortcut: nothing to solve
    if parallel.active_workers() <= 1:
        return False
    return _botorch_version() == BOTORCH_VERSION


def _one_slot(acq_function: Any, bounds: torch.Tensor, num_restarts: int, raw_samples: int,
              fixed_features_list: list, ineq: list | None) -> tuple[torch.Tensor, torch.Tensor]:
    """botorch's `optimize_acqf_mixed(q=1, return_best_only=True)`, solves in parallel."""
    import torch
    from botorch.optim import optimize_acqf_mixed as _botorch_optimize_acqf_mixed
    from botorch.optim.initializers import gen_batch_initial_conditions

    state = parallel.capture_rng()
    d = bounds.shape[-1]
    ffs = [{int(i) % d: v for i, v in ff.items()} for ff in fixed_features_list]
    try:
        blob = pickle.dumps(acq_function, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - not picklable: botorch's serial loop
        blob = None
    if blob is not None:
        ics = [
            gen_batch_initial_conditions(
                acq_function=acq_function, bounds=bounds, q=1, num_restarts=num_restarts,
                raw_samples=raw_samples, fixed_features=ff, options={},
                inequality_constraints=ineq, equality_constraints=None,
            )
            for ff in ffs
        ]
        solved = parallel.run_ordered(
            _solve_level, [(blob, bounds, ic, ff, num_restarts, ineq) for ic, ff in zip(ics, ffs)],
        )
        if all(clean for _c, _v, clean in solved):
            cand_list = [c for c, _v, _ok in solved]
            val_list = [v for _c, v, _ok in solved]
            max_res = torch.max(torch.stack(val_list), dim=-1)
            best_batch_idx = torch.argmax(max_res.values)
            best_idx = max_res.indices[best_batch_idx]
            return cand_list[best_batch_idx][best_idx], val_list[best_batch_idx][best_idx]
        parallel.restore_rng(state)
    cand, val = _botorch_optimize_acqf_mixed(
        acq_function=acq_function, bounds=bounds, q=1, num_restarts=num_restarts,
        raw_samples=raw_samples, fixed_features_list=fixed_features_list, options={},
        inequality_constraints=ineq, return_best_only=True,
    )
    return cand, val


def optimize_acqf_mixed(
    acq_function: Any,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    fixed_features_list: list[dict[int, float]],
    raw_samples: int | None = None,
    inequality_constraints: list | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop-in for botorch's `optimize_acqf_mixed` (the keywords the engine passes),
    returning the identical result; see the module docstring."""
    import torch
    from botorch.optim import optimize_acqf_mixed as _botorch_optimize_acqf_mixed

    if not _eligible(acq_function, bounds, fixed_features_list, raw_samples, kwargs):
        return _botorch_optimize_acqf_mixed(
            acq_function=acq_function, bounds=bounds, q=q, num_restarts=num_restarts,
            fixed_features_list=fixed_features_list, raw_samples=raw_samples,
            inequality_constraints=inequality_constraints, **kwargs,
        )
    assert raw_samples is not None
    if q == 1:
        return _one_slot(acq_function, bounds, num_restarts, raw_samples,
                         fixed_features_list, inequality_constraints)
    # botorch's greedy loop for q > 1 (optimize.py:1350-1393), slot by slot.
    base_X_pending = acq_function.X_pending
    candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
    for _ in range(q):
        candidate, _acq_value = _one_slot(acq_function, bounds, num_restarts, raw_samples,
                                          fixed_features_list, inequality_constraints)
        candidates = torch.cat([candidates, candidate], dim=-2)
        acq_function.set_X_pending(
            torch.cat([base_X_pending, candidates], dim=-2)
            if base_X_pending is not None
            else candidates
        )
    acq_function.set_X_pending(base_X_pending)
    return candidates, acq_function(candidates)


__all__ = ["BOTORCH_VERSION", "optimize_acqf_mixed"]
