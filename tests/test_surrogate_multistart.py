"""Multi-start hyperparameter fit and boundary checks (compmodel PLAN E7).

The proposal surrogate fits from the default start plus 2 seeded random starts
and keeps the best marginal likelihood; CV folds keep one start. Every fit
records its final noise, lengthscales and boundary flags.
"""
from __future__ import annotations

import numpy as np
import torch

from kalos.core.surrogate import LONG_LENGTHSCALE, Surrogate

B = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def _design(n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 3))
    return X, np.sin(4 * X[:, 0]) + 0.3 * X[:, 1] + rng.normal(0, 0.05, n)


def test_restart_record_and_best_mll_is_kept():
    X, y = _design()
    d = Surrogate().fit(X, y, bounds=B, restarts=3, restart_seed=7).fit_diagnostics
    assert d["restarts"] == 3 and d["restart_seed"] == 7
    assert len(d["restart_mll"]) == 3
    vals = [v for v in d["restart_mll"] if v is not None]
    assert d["restart_mll"][d["restart_chosen"]] == max(vals) or (
        d["restart_chosen"] == 0 and max(vals) - d["restart_mll"][0] <= 1e-6 + 1e-9
    )
    assert d["fit_seconds"] >= 0


def test_restarts_are_seeded():
    X, y = _design()
    a = Surrogate().fit(X, y, bounds=B, restarts=3, restart_seed=7).fit_diagnostics
    b = Surrogate().fit(X, y, bounds=B, restarts=3, restart_seed=7).fit_diagnostics
    assert a["restart_mll"] == b["restart_mll"] and a["restart_chosen"] == b["restart_chosen"]


def test_default_start_winning_leaves_the_one_start_model_and_rngs_unchanged():
    X, y = _design()
    grid = np.random.default_rng(1).uniform(0, 1, (25, 3))
    torch.manual_seed(3)
    np.random.seed(3)
    one = Surrogate().fit(X, y, bounds=B)
    after_one = (torch.rand(1).item(), np.random.rand())
    torch.manual_seed(3)
    np.random.seed(3)
    multi = Surrogate().fit(X, y, bounds=B, restarts=3, restart_seed=7)
    after_multi = (torch.rand(1).item(), np.random.rand())
    assert after_one == after_multi  # global RNG state matches a one-start fit
    if multi.fit_diagnostics["restart_chosen"] == 0:
        np.testing.assert_array_equal(one.posterior(grid)[0], multi.posterior(grid)[0])


def test_one_start_is_the_default():
    d = Surrogate().fit(*_design(), bounds=B).fit_diagnostics
    assert "restarts" not in d
    assert set(d) >= {"noise_fitted", "lengthscales", "boundary_flags"}


def test_pure_noise_target_sets_a_boundary_flag():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 3))
    y = rng.normal(0, 1, 30)
    d = Surrogate().fit(X, y, bounds=B, restarts=3).fit_diagnostics
    assert d["boundary_flags"], d
    # Measured: on pure noise the fit does not hit a bound; it shortens the
    # lengthscales below the row spacing and follows the noise. That is the flag.
    assert any("nearest-neighbour" in f for f in d["boundary_flags"])


def test_well_posed_noisy_target_has_no_flags():
    d = Surrogate().fit(*_design(), bounds=B, restarts=3).fit_diagnostics
    assert not any("nearest-neighbour" in f for f in d["boundary_flags"])
    assert all(v <= LONG_LENGTHSCALE for v in d["lengthscales"][:2])


def test_noise_free_smooth_target_flags_noise_at_its_floor():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    d = Surrogate().fit(X, y, bounds=B).fit_diagnostics
    assert any("noise at its lower bound" in f for f in d["boundary_flags"])
    assert d["noise_fitted"] is not None


def test_fixed_noise_reports_no_fitted_noise():
    X, y = _design()
    d = Surrogate().fit(X, y, bounds=B, noise=0.01).fit_diagnostics
    assert d["noise_fitted"] is None
    assert not any("noise at" in f for f in d["boundary_flags"])


def test_analysis_uses_three_starts_and_reports_them():
    import pandas as pd

    from kalos.portal.pipeline import compute_upload_analysis

    X, y = _design(24)
    raw = pd.DataFrame({"a": X[:, 0].round(4), "b": X[:, 1].round(4), "c": X[:, 2].round(4),
                        "t": y.round(4)}).to_csv(index=False).encode()
    _, r = compute_upload_analysis(raw, "t", False, "x.csv")
    md = r["model_diagnostics"]
    assert md["restarts"] == 3 and len(md["restart_mll"]) == 3
    assert isinstance(md["boundary_flags"], list) and isinstance(md["lengthscales"], list)


def test_extra_starts_stop_at_the_time_budget(monkeypatch):
    from kalos.core import surrogate as surrogate_module

    monkeypatch.setattr(surrogate_module, "MULTI_START_BUDGET_S", 0.0)
    d = Surrogate().fit(*_design(), bounds=B, restarts=3).fit_diagnostics
    assert d["restarts_skipped"] == 2 and d["restart_mll"][1:] == [None, None]
    assert d["restart_chosen"] == 0
