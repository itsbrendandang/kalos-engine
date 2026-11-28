"""Regression tests for confirmed bugs found during the Phase 2 ML stack audit."""
from __future__ import annotations

import json

import numpy as np
import pytest

from kalos.core.evaluation import group_mean_margin, grouped_cv_report
from kalos.core.optimize import propose_batch
from kalos.core.surrogate import Surrogate
from kalos.portal.analysis_parts.noise_blocks import _alternative_scale_block
from kalos.portal.analysis_parts.strain_block import choose_fixed_level
from kalos.portal.analysis_parts.strain_roles import StrainPlan


def test_propose_batch_does_not_clip_pending_to_fixed_features(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    X = np.column_stack([rng.uniform(0.0, 1.0, 10), np.array([0.0] * 5 + [1.0] * 5)])
    y = X[:, 0] + 0.5 * X[:, 1]
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])
    s = Surrogate().fit(X, y, bounds=bounds, cat_dims=[1])

    seen_pending: list[np.ndarray | None] = []
    import kalos.core.optimize as opt_mod

    orig = opt_mod.qLogNoisyExpectedImprovement

    class _SpyAcq(orig):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            xp = kwargs.get("X_pending")
            seen_pending.append(None if xp is None else xp.detach().cpu().numpy().copy())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(opt_mod, "qLogNoisyExpectedImprovement", _SpyAcq)
    pending = np.array([[0.42, 1.0]])  # in-flight on clone 1 while proposing for fixed clone 0
    propose_batch(
        s,
        bounds,
        q=1,
        cat_dims=[1],
        cat_cardinalities=[2],
        fixed_features={1: 0.0},
        pending=pending,
        seed=0,
    )
    assert seen_pending and seen_pending[0] is not None
    assert seen_pending[0][0, 1] == pytest.approx(1.0)


def test_alternative_scale_block_handles_finite_rho_with_nan_ci95() -> None:
    def _fake_cv(*_a, **_kw):
        return {
            "spearman": 0.65,
            "ci95": (float("nan"), float("nan")),
            "oof_actual": [1.0, 2.0, 3.0, 4.0],
            "oof_pred": [1.1, 1.9, 3.2, 3.8],
            "oof_std": [0.3, 0.3, 0.3, 0.3],
        }

    def _fake_cal(*_a, **_kw):
        return {"available": False, "reason": "too_few"}

    out = _alternative_scale_block(
        {"suggests_transform": True, "log_offset": 0.1, "icc_log": 0.4, "reason": "test"},
        np.array([[0.0], [0.3], [0.7], [1.0]]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        groups=["g1", "g1", "g2", "g2"],
        bounds=np.array([[0.0], [1.0]]),
        cat_dims=None,
        grouped_cv_report=_fake_cv,
        interval_calibration=_fake_cal,
    )
    assert out["cv_spearman"] == 0.65
    assert out["cv_ci95"] is None
    json.dumps(out, allow_nan=False)


def test_choose_fixed_level_rejects_level_with_zero_fitted_runs() -> None:
    plan = StrainPlan(column="clone", policy="fixed", requested_level="CloneB", requested_level_source="setup")
    lvl, _src, refusal = choose_fixed_level(plan, ("CloneA", "CloneB"), {"CloneA": 8, "CloneB": 0})
    assert lvl is None
    assert refusal is not None and refusal["code"] == "clone_not_in_training"


def test_evaluation_handles_mixed_group_types_and_negative_oof_index() -> None:
    X = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    y = np.linspace(1.0, 5.0, 12)
    groups = ["A", "A", "B", "B", 1, 1, None, None, "C", "C", "D", "D"]
    rep = grouped_cv_report(X, y, groups=groups, n_splits=4, bounds=np.array([[0.0], [1.0]]))
    assert "spearman" in rep

    gm = group_mean_margin(y, groups, [-1, 0, 1], [1.0, 2.0, 3.0])
    assert gm["evaluable"] is False
