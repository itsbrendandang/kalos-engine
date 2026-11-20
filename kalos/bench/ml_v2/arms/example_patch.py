"""Example engine-patch candidate (a template; not a proposal to ship).

It is the reference arm with ONE proposal-surrogate hyperparameter start
instead of three (`kalos.portal.analysis.PROPOSAL_FIT_RESTARTS`), applied as an
in-process patch around each analysis:

    --arm kalos.bench.ml_v2.arms.example_patch:make_arm --arm-config '{"restarts": 1}'

Copy this file to start a candidate. Patch module attributes that the engine
looks up AT CALL TIME (module globals, names imported inside `_analyze_body`
such as `kalos.core.surrogate.Surrogate` or `kalos.core.optimize.propose_batch`,
or module-level imports of `kalos.portal.analysis` such as
`calibrate_selection_aware`). Patching a name another module bound with
`from x import y` at import time has no effect there: patch the binding the
caller uses (e.g. `kalos.portal.analysis.calibrate_selection_aware`, not
`kalos.portal.analysis_parts.intervals.calibrate_selection_aware`).
"""
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from ..adapter import patched
from .kalos_ref import KalosEngineArm


class FewerRestartsArm(KalosEngineArm):
    name = "example_fewer_restarts"

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        import kalos.portal.analysis as an

        return patched(an, "PROPOSAL_FIT_RESTARTS", int(self.config.get("restarts", 1)))


def make_arm(config: Mapping[str, Any] | None = None) -> FewerRestartsArm:
    return FewerRestartsArm(config)
