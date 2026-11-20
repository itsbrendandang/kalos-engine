"""Analysis-side glue for the campaign setup (`kalos.portal.setup_settings`):
call `propose_batch` with the setup's box / constraints / goal, and build the
empty batch returned when the setup cannot be applied. Torch is only touched
inside the functions (via `kalos.core.optimize`), never at import time.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from kalos.portal.setup_settings import SetupPlan, withheld_notice

log = logging.getLogger("kalos.portal")


def withheld_batch(plan: SetupPlan, *, q: int, d: int) -> Any:
    """An empty `ProposalBatch` carrying the `setup_not_applied` notice."""
    from kalos.core.optimize import NEAR_DUPLICATE_TOL, ProposalBatch

    reason = (plan.refusal or {}).get("code", "setup_not_applied")
    return ProposalBatch(
        np.empty((0, d)), q, 0, float(NEAR_DUPLICATE_TOL), withheld_notice(q, str(reason))
    )


def _propose_with_setup(
    propose_batch: Callable[..., Any], plan: SetupPlan | None, kwargs: dict[str, Any], *, d: int,
) -> Any:
    """`propose_batch(**kwargs)`, with the plan's proposal box, linear
    constraints and goal when a setup is given (`plan=None` is the historical
    call, unchanged). If the optimizer cannot satisfy the constraints (it retries,
    then raises), the plan is refused and an empty batch is returned: proposals
    are withheld rather than produced without a constraint."""
    if plan is None:
        return propose_batch(**kwargs)
    call = {**kwargs, "bounds": plan.bounds, **plan.propose_kwargs()}
    if not plan.linear_constraints:
        return propose_batch(**call)
    try:
        return propose_batch(**call)
    except (RuntimeError, ValueError) as exc:
        log.warning("setup constraints could not be satisfied: type=%s", type(exc).__qualname__)
        plan._refuse(
            "constraints_unsatisfiable",
            "the optimizer could not find recipes that satisfy every linear constraint "
            "inside the search range; proposals are withheld rather than dropping a constraint",
            "loosen the linear constraints or widen the bounds",
        )
        plan.linear_constraints = []
        return withheld_batch(plan, q=int(kwargs.get("q", 5)), d=d)


__all__ = ["_propose_with_setup", "withheld_batch"]
