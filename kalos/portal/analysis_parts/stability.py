"""Opt-in proposal stability check: rerun the proposal step with extra seeds.

Off by default. `KALOS_PROPOSAL_STABILITY_SEEDS=K` (K >= 2, at most
`MAX_SEEDS`) reruns the proposal step K-1 more times after the served batch
and reports how often each served proposal comes back (`kalos.core.stability`).
`0` or `1` (or unset) is off: the analysis does not rerun anything and its
response carries no `proposal_stability` key and no `proposals[].stability`, so
an analysis with the feature off is byte-identical to one from before it
existed.

What a rerun varies: the acquisition optimizer's seed (raw samples, multi-start
initial conditions, the MC sampler) always, and the GP hyperparameter restart
seed when the served fit used more than one start (`proposal_restarts`). The
data, bounds, setup, pending runs, gates and constraints are the served ones.
The served batch is always the seed-0 output; reruns never replace it.

Purity: the reruns run inside `torch.random.fork_rng` with numpy's global state
restored, so every other number in the response is the same with the feature on
or off (asserted by tests/test_robustness.py).

Cost: each rerun is one full proposal step. Reruns go to the analysis worker
pool (`kalos.core.parallel.run_ordered`, perf/parallel-analysis) when it exists
and is enabled, else run serially. docs/CAMPAIGNS_API.md "Proposal stability"
has the measured seconds.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from kalos.core.stability import (
    STABILITY_TOL,
    UNSTABLE_BELOW,
    batch_agreement,
    seed_schedule,
)
from kalos.envconfig import env_int

log = logging.getLogger("kalos.portal")

SEEDS_VAR = "KALOS_PROPOSAL_STABILITY_SEEDS"
MAX_SEEDS = 10
RECOMMENDED_SEEDS = 5


def configured_seeds() -> int:
    """K from `KALOS_PROPOSAL_STABILITY_SEEDS`; 0 means off. Out of range raises
    `ConfigError` (fail-closed, like every enumerated/bounded setting)."""
    k = env_int(SEEDS_VAR, 0, minimum=0, maximum=MAX_SEEDS)
    return 0 if k < 2 else k


def resolve_seeds(requested: int | None) -> int:
    """An explicit per-call K wins over the environment; <2 means off."""
    if requested is None:
        return configured_seeds()
    k = int(requested)
    if not 0 <= k <= MAX_SEEDS:
        raise ValueError(f"stability seeds must be 0..{MAX_SEEDS}")
    return 0 if k < 2 else k


def _rerun_proposal(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    fit_kwargs: dict[str, Any] | None,
    surrogate: Any,
    propose_kwargs: dict[str, Any],
    seed: int,
) -> np.ndarray | None:
    """One rerun of the proposal step at `seed`; None if it failed.

    Module-level so the worker pool can pickle it. `fit_kwargs=None` reuses the
    served `surrogate` (its fit was single-start, so a refit would be the same
    fit); otherwise the surrogate is refit with `restart_seed=seed`.
    """
    from kalos.core.optimize import propose_batch
    from kalos.core.surrogate import Surrogate

    try:
        if fit_kwargs is not None:
            surrogate = Surrogate().fit(X_fit, y_fit, restart_seed=int(seed), **fit_kwargs)
        batch = propose_batch(surrogate=surrogate, seed=int(seed), **propose_kwargs).X
    except Exception as exc:  # a failed rerun is reported, never fatal
        log.info("stability rerun seed=%s failed: %s", seed, type(exc).__name__)
        return None
    return np.asarray(batch, float)


def _run_tasks(fn: Callable[..., Any], tasks: list[tuple]) -> tuple[list[Any], int]:
    """Reruns through the analysis worker pool when available, else serially."""
    try:
        from kalos.core import parallel  # perf/parallel-analysis
    except ImportError:
        return [fn(*t) for t in tasks], 1
    workers = int(parallel.active_workers()) if hasattr(parallel, "active_workers") else 1
    return list(parallel.run_ordered(fn, tasks)), max(1, workers)


def proposal_stability_block(
    *,
    n_seeds: int,
    base_seed: int,
    served: np.ndarray,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    surrogate: Any,
    fit_kwargs: dict[str, Any],
    propose_kwargs: dict[str, Any],
    cat_dims: list[int] | None,
    post_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """The `proposal_stability` response block (see the module docstring)."""
    import torch

    seeds = seed_schedule(base_seed, n_seeds)
    restarts = int(fit_kwargs.get("restarts", 1))
    varied = ["acquisition_optimizer_seed"]
    if restarts > 1:
        varied.append("gp_hyperparameter_restart_seed")
    block: dict[str, Any] = {
        "enabled": True,
        "available": False,
        "reason": skip_reason,
        "n_seeds": int(n_seeds),
        "seeds": seeds,
        "served_seed": seeds[0],
        "varied": varied,
        "held_fixed": ["data", "bounds_and_setup", "pending_runs", "gates_and_constraints"],
        "tolerance": STABILITY_TOL,
        "distance": "range_normalized_chebyshev",
        "unstable_below": UNSTABLE_BELOW,
        "what_it_measures": (
            "how often each served proposal comes back when only the optimizer seed "
            "changes; seed sensitivity of the search on this data, not proposal quality "
            "and not data uncertainty"
        ),
        "batch_stability": None,
        "n_unstable": 0,
        "n_reruns_compared": 0,
        "n_reruns_failed": 0,
        "per_proposal": [],
        "seconds": 0.0,
        "workers": 1,
    }
    if skip_reason is not None:
        return block
    S = np.asarray(served, float)
    if S.ndim != 2 or S.shape[0] == 0:
        block["reason"] = "no proposals were served, so there is nothing to compare"
        return block
    lower = np.asarray(propose_kwargs["bounds"], float)[0]
    upper = np.asarray(propose_kwargs["bounds"], float)[1]
    refit = dict(fit_kwargs) if restarts > 1 else None
    if refit is not None:
        refit.pop("restart_seed", None)
        refit["check_conditioning"] = False
    tasks = [
        (np.asarray(X_fit, float), np.asarray(y_fit, float), refit,
         None if refit is not None else surrogate, dict(propose_kwargs), int(s))
        for s in seeds[1:]
    ]
    t0 = time.perf_counter()
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            reruns, workers = _run_tasks(_rerun_proposal, tasks)
    finally:
        np.random.set_state(np_state)
    if post_filter is not None:
        # The served batch went through the same final check (setup bounds and
        # linear constraints); a rerun is compared on the same footing.
        reruns = [None if r is None else (r[post_filter(r)] if len(r) else r) for r in reruns]
    agreement = batch_agreement(S, reruns, lower, upper, cat_dims)
    block.update(agreement)
    block["available"] = agreement["n_reruns_compared"] > 0
    if not block["available"]:
        block["reason"] = "every rerun failed, so stability could not be measured"
    block["seconds"] = round(time.perf_counter() - t0, 2)
    block["workers"] = workers
    return block


def annotate_proposals(proposals: list[dict[str, Any]], block: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy each served proposal's agreement onto it (`proposals[i].stability`)."""
    per = {int(p["index"]): p for p in block.get("per_proposal", [])}
    out = []
    for i, prop in enumerate(proposals):
        p = per.get(i)
        if p is None:
            out.append(prop)
            continue
        out.append({**prop, "stability": {
            k: p[k] for k in ("agreement", "seeds_matched", "seeds_compared",
                             "median_nearest_distance", "unstable", "label")
        }})
    return out


__all__ = [
    "MAX_SEEDS", "RECOMMENDED_SEEDS", "SEEDS_VAR", "annotate_proposals", "configured_seeds",
    "proposal_stability_block", "resolve_seeds",
]
