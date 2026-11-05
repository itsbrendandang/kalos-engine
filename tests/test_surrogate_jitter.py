"""The Cholesky jitter ladder (kalos/core/surrogate.py, compmodel PLAN E1).

The rung the fit succeeded under and the largest diagonal addition actually made
are recorded on the Surrogate and in the analysis result (`model_diagnostics`).
The rung is applied to float64 (it used to set only the float32 value, so it was
a no-op for the float64 GP), each Cholesky gets one try, and an addition above
`JITTER_CAP` fails loudly with `FitError`.
"""
from __future__ import annotations

import warnings

import linear_operator
import numpy as np
import pytest
import torch
from linear_operator.utils.warnings import NumericalWarning

from kalos.core import surrogate as surrogate_module
from kalos.core.surrogate import JITTER_CAP, FitError, Surrogate

BOUNDS = np.array([[0.0, 0.0], [1.0, 1.0]])


def _design(n: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 2))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.05, n)
    return X, y


def test_well_posed_fit_records_the_first_rung():
    s = Surrogate().fit(*_design(), bounds=BOUNDS)
    d = s.fit_diagnostics
    assert d["jitter_rung"] == surrogate_module._FIT_JITTERS[0]
    assert d["jitter_rung_index"] == 0
    assert 0.0 <= d["jitter_added_max"] <= JITTER_CAP
    assert d["jitter_cap"] == JITTER_CAP == 1e-4


def test_rung_applies_to_float64_with_one_try(monkeypatch):
    seen: list[tuple[float, float, int]] = []
    real = surrogate_module.fit_gpytorch_mll

    def spy(mll, *a, **kw):
        seen.append((
            linear_operator.settings.cholesky_jitter.value(torch.double),
            linear_operator.settings.cholesky_jitter.value(torch.float),
            linear_operator.settings.cholesky_max_tries.value(),
        ))
        return real(mll, *a, **kw)

    monkeypatch.setattr(surrogate_module, "fit_gpytorch_mll", spy)
    Surrogate().fit(*_design(), bounds=BOUNDS)
    assert seen == [(1e-8, 1e-8, 1)]


def test_a_failed_first_rung_is_recorded_and_barely_moves_the_fit(monkeypatch):
    X, y = _design()
    grid = np.random.default_rng(1).uniform(0, 1, (50, 2))
    torch.manual_seed(0)
    base_mean, _ = Surrogate().fit(X, y, bounds=BOUNDS).posterior(grid)

    real = surrogate_module.fit_gpytorch_mll
    calls = {"n": 0}

    def fail_once(mll, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise linear_operator.utils.errors.NotPSDError("forced")
        return real(mll, *a, **kw)

    monkeypatch.setattr(surrogate_module, "fit_gpytorch_mll", fail_once)
    torch.manual_seed(0)
    s = Surrogate().fit(X, y, bounds=BOUNDS)
    assert s.fit_diagnostics["jitter_rung_index"] == 1
    assert s.fit_diagnostics["jitter_rung"] == 1e-6
    mean, _ = s.posterior(grid)
    np.testing.assert_allclose(mean, base_mean, atol=1e-3)


def test_every_rung_failing_raises_fit_error_naming_the_cap(monkeypatch):
    def always(mll, *a, **kw):
        raise linear_operator.utils.errors.NotPSDError("forced")

    monkeypatch.setattr(surrogate_module, "fit_gpytorch_mll", always)
    with pytest.raises(FitError, match="cap"):
        Surrogate().fit(*_design(), bounds=BOUNDS)


def test_a_diagonal_addition_above_the_cap_fails_loudly(monkeypatch):
    """If linear_operator ever adds more than the cap (e.g. a library change that
    ignores max_tries), the fit is refused, not accepted quietly."""
    real = surrogate_module.fit_gpytorch_mll

    def too_much(mll, *a, **kw):
        real(mll, *a, **kw)
        warnings.warn("A not p.d., added jitter of 1.0e-02 to the diagonal", NumericalWarning, stacklevel=2)

    monkeypatch.setattr(surrogate_module, "fit_gpytorch_mll", too_much)
    with pytest.raises(FitError, match="above the cap"):
        Surrogate().fit(*_design(), bounds=BOUNDS)


def test_recorded_additions_are_counted(monkeypatch):
    real = surrogate_module.fit_gpytorch_mll

    def noted(mll, *a, **kw):
        real(mll, *a, **kw)
        warnings.warn("A not p.d., added jitter of 1.0e-08 to the diagonal", NumericalWarning, stacklevel=2)
        warnings.warn("A not p.d., added jitter of 1.0e-08 to the diagonal", NumericalWarning, stacklevel=2)

    monkeypatch.setattr(surrogate_module, "fit_gpytorch_mll", noted)
    d = Surrogate().fit(*_design(), bounds=BOUNDS).fit_diagnostics
    assert d["jitter_additions"] == 2 and d["jitter_added_max"] == 1e-8


def test_analysis_result_exposes_the_jitter_record():
    import pandas as pd

    from kalos.portal.pipeline import compute_upload_analysis

    X, y = _design(24)
    raw = pd.DataFrame({"a": X[:, 0].round(4), "b": X[:, 1].round(4), "t": y.round(4)})
    _, result = compute_upload_analysis(raw.to_csv(index=False).encode(), "t", False, "x.csv")
    md = result["model_diagnostics"]
    assert md["jitter_rung"] in surrogate_module._FIT_JITTERS
    assert md["jitter_cap"] == JITTER_CAP
    assert md["jitter_added_max"] <= JITTER_CAP
