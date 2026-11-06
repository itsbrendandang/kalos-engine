"""Promotion gates: a model is only fit to serve if it clears the floors.

Carried over from the lean engine (and the multi-agent review that hardened it):
fail-CLOSED. A missing or NaN required metric BLOCKS promotion rather than
silently passing — the gate must not vanish on exactly the data it exists to
guard (e.g. an unmeasurable feasibility AUC on single-class grouped folds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TypeGuard


@dataclass
class GatesConfig:
    min_spearman: float = 0.20
    min_feasibility_auc: float = 0.65
    max_ece: float = 0.15
    max_brier: float = 0.25
    # Below this many scored runs the ECE is not judged (neither pass nor fail):
    # at small n a calibrated model's ECE exceeds `max_ece` too often for the
    # number to mean anything (compmodel PLAN E5). Applies only when the caller
    # passes `ece_n`; without it the ECE is judged as before.
    min_ece_n: int = 20
    enabled: bool = True


NOT_ENOUGH_RUNS_FOR_CALIBRATION = "not enough runs to judge calibration"
NOT_APPLICABLE_WHEN_MINIMIZING = "not applicable when minimizing"


@dataclass
class PromotionResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    # Gates that were neither passed nor failed, with the reason (e.g. ECE on
    # fewer than `min_ece_n` runs).
    not_judged: list[str] = field(default_factory=list)
    # Gates that blocked only because their metric could not be measured
    # (missing/NaN), e.g. `feasibility_auc` on a sheet with no non-producers.
    unmeasured: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """`passed`, `failed` (a measured gate missed its floor or ceiling), or
        `not_measurable` (every blocking gate was unmeasured). Same verdict as
        `passed`; it only says why a `False` is false."""
        if self.passed:
            return "passed"
        return "not_measurable" if len(self.unmeasured) == len(self.failures) else "failed"


def _finite(x: Any) -> TypeGuard[float]:
    # bool is an int subclass; a stray True/False is not a real metric value.
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_gates(stats: dict[str, Any], config: GatesConfig | None = None) -> PromotionResult:
    cfg = config or GatesConfig()
    if not cfg.enabled:
        return PromotionResult(True, [], {}, "promotion gates disabled")

    f: list[str] = []
    m: dict[str, float] = {}
    unmeasured: list[str] = []

    def floor(key: str, lo: float) -> None:
        v = stats.get(key)
        if _finite(v):
            m[key] = float(v)
            if v < lo:
                f.append(f"{key}={v:.3f} < min {lo}")
        else:
            f.append(f"{key} missing/NaN (unmeasured)")  # fail closed
            unmeasured.append(key)

    def ceiling(key: str, hi: float) -> None:
        v = stats.get(key)
        if _finite(v):
            m[key] = float(v)
            if v > hi:
                f.append(f"{key}={v:.3f} > max {hi}")
        else:
            f.append(f"{key} missing/NaN (unmeasured)")  # fail closed
            unmeasured.append(key)

    not_judged: list[str] = []
    floor("surrogate_spearman", cfg.min_spearman)
    # The feasibility classifier scores P(y > 0), a maximize notion. When the
    # caller says it does not apply (minimize), its three gates are reported as
    # not judged with that reason instead of passing or failing.
    feas_na = stats.get("feasibility_not_applicable")
    if isinstance(feas_na, str) and feas_na:
        not_judged.extend(f"{key}: {feas_na}" for key in ("feasibility_auc", "ece", "brier"))
    else:
        floor("feasibility_auc", cfg.min_feasibility_auc)
        ece_n = stats.get("ece_n")
        if _finite(ece_n) and ece_n < cfg.min_ece_n:
            not_judged.append(
                f"ece: {NOT_ENOUGH_RUNS_FOR_CALIBRATION} (n={int(ece_n)}, need {cfg.min_ece_n})"
            )
        else:
            ceiling("ece", cfg.max_ece)
        ceiling("brier", cfg.max_brier)
    # Group-mean floor (compmodel PLAN E13): only when the caller passes the
    # paired comparison (`group_mean_margin` in kalos/core/evaluation.py).
    gm = stats.get("group_mean_margin")
    if isinstance(gm, dict):
        if gm.get("evaluable"):
            lo = (gm.get("ci95") or [float("nan")])[0]
            m["group_mean_margin_ci_lo"] = float(lo)
            if not gm.get("passed"):
                f.append(
                    f"surrogate does not beat the group-mean floor: paired difference "
                    f"{gm.get('difference')} with 95% CI lower bound {lo} <= 0"
                )
        else:
            not_judged.append(f"group-mean floor: {gm.get('reason') or 'not evaluable'}")

    passed = not f
    summary = "ALL GATES PASSED" if passed else "GATES FAILED: " + "; ".join(f)
    if not_judged:
        summary += " (not judged: " + "; ".join(not_judged) + ")"
    return PromotionResult(passed, f, m, summary, not_judged, unmeasured)


__all__ = ["NOT_APPLICABLE_WHEN_MINIMIZING", "NOT_ENOUGH_RUNS_FOR_CALIBRATION", "GatesConfig", "PromotionResult", "check_gates"]
