"""Random and Latin-hypercube baseline arms (standalone, never shippable).

Both make no model claim: no intervals, never trusted. In pool replay the random
arm picks uniformly among the not-yet-run rows (as VALIDATION.md); the LHS arm
proposes the next q points of one Latin hypercube of the whole budget, drawn once
per (task, seed), which the harness snaps to the nearest unrun row.
"""
from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..adapter import ArmOutput, Proposal, RoundContext, Verdict
from ..protocol import ROUNDS


def _seed(ctx: RoundContext, salt: str, with_round: bool) -> int:
    key = f"{salt}|{ctx.task}|{ctx.seed}" + (f"|{ctx.round}" if with_round else "")
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class RandomArm:
    name = "random"
    shippable = False

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def propose(self, ctx: RoundContext) -> ArmOutput:
        rng = np.random.default_rng(_seed(ctx, "random", True))
        if ctx.candidates is not None and len(ctx.candidates):
            idx = rng.choice(len(ctx.candidates), min(ctx.q, len(ctx.candidates)), replace=False)
            recs = ctx.candidates.iloc[np.sort(idx)].to_dict("records")
        else:
            recs = ctx.space.sample(ctx.q, rng).to_dict("records")
        return ArmOutput([Proposal({str(k): v for k, v in r.items()}) for r in recs], Verdict(False, "baseline"))


class LHSArm(RandomArm):
    name = "lhs"

    def propose(self, ctx: RoundContext) -> ArmOutput:
        rng = np.random.default_rng(_seed(ctx, "lhs", False))
        design = ctx.space.sample(ROUNDS * ctx.q, rng, lhs=True)
        k = max(0, ctx.round - 1) % ROUNDS
        recs = design.iloc[k * ctx.q:(k + 1) * ctx.q].to_dict("records")
        return ArmOutput([Proposal({str(c): v for c, v in r.items()}) for r in recs], Verdict(False, "baseline"))


def make_random(config: Mapping[str, Any] | None = None) -> RandomArm:
    return RandomArm(config)


def make_lhs(config: Mapping[str, Any] | None = None) -> LHSArm:
    return LHSArm(config)
