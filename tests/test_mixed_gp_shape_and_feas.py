"""Tests for mixed-GP ARD lengthscale recovery and one-hot categorical feasibility."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("botorch")

import torch

from kalos.core.feasibility import (
    FeasibilityClassifier,
    feasibility_cv_report,
    feasible_labels,
)
from kalos.core.gp_shape import ard_lengthscales, gp_shape_report
from kalos.core.optimize import _FeasibilityGate, propose
from kalos.core.surrogate import Surrogate
from kalos.envconfig import ConfigError
from kalos.portal.analysis import _analyze
from kalos.portal.config import check_env_settings, feas_onehot_cat_setting


def test_ard_lengthscales_mixed_gp_categorical_first_and_equal_counts():
    rng = np.random.default_rng(1)
    n = 30
    # Categorical col 0 comes BEFORE continuous cols 1, 2.
    cat = np.tile([0, 1, 2], n // 3)
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = 2.0 * x1 - 1.5 * (x2 - 0.5) ** 2 + 0.4 * (cat == 2) + rng.normal(0, 0.05, n)
    X = np.column_stack([cat, x1, x2])
    bounds = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])

    s = Surrogate().fit(X, y, bounds=bounds, cat_dims=[0])
    ls = ard_lengthscales(s, 2, sweep_columns=[1, 2])
    assert ls is not None and ls.shape == (2,)
    assert np.all(np.isfinite(ls)) and np.all(ls > 0)

    # Mismatched sweep_columns pointing to the categorical column returns None
    # (never returns CategoricalKernel's lengthscale!).
    assert ard_lengthscales(s, 1, sweep_columns=[0]) is None

    rep = gp_shape_report(s, X, y, feature_names=["x1", "x2"], sweep_columns=[1, 2])
    assert rep.available is True, rep.reason
    assert len(rep.features) == 2

    # Equal continuous and categorical dim counts: d_cont=1, n_cat=1
    X_eq = np.column_stack([x1, cat])
    s_eq = Surrogate().fit(X_eq, y, bounds=np.array([[0.0, 0.0], [1.0, 2.0]]), cat_dims=[1])
    ls_eq = ard_lengthscales(s_eq, 1, sweep_columns=[0])
    assert ls_eq is not None and ls_eq.shape == (1,)
    assert ard_lengthscales(s_eq, 1, sweep_columns=[1]) is None


def test_ard_lengthscales_returns_none_for_isotropic_and_bad_values():
    from gpytorch.kernels import RBFKernel, ScaleKernel

    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (20, 3))
    y = X[:, 0] + rng.normal(0, 0.05, 20)
    s = Surrogate().fit(X, y, bounds=np.array([[0, 0, 0], [1, 1, 1]]))
    # Replace with an isotropic kernel (no ard_num_dims -> scalar lengthscale)
    s.model.covar_module = ScaleKernel(RBFKernel()).to(dtype=torch.double)
    assert ard_lengthscales(s, 3) is None

    # Non-positive / non-finite lengthscale returns None
    s2 = Surrogate().fit(X, y, bounds=np.array([[0, 0, 0], [1, 1, 1]]))
    with torch.no_grad():
        s2.model.covar_module.raw_lengthscale.fill_(float("nan"))
    assert ard_lengthscales(s2, 3) is None


def test_onehot_feasibility_classifier_parity_and_gradients():
    rng = np.random.default_rng(5)
    n = 80
    x0 = rng.uniform(0.0, 10.0, n)
    cat = rng.integers(0, 3, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    # Non-ordinal categorical effect + continuous slope
    logit = 0.6 * (x0 - 5.0) - 2.5 * (cat == 1) + 1.5 * (cat == 2) + 0.3 * x2
    y_bin = (logit + rng.normal(0, 0.3, n) > 0).astype(int)
    X = np.column_stack([x0, cat, x2])

    clf = FeasibilityClassifier(cat_dims=[1], cat_cardinalities=[3]).fit(X, y_bin)
    assert clf.fitted

    spec = clf.torch_gate_spec()
    assert spec["cont_dims"] == [0, 2]
    assert spec["cat_dims"] == [1]
    assert spec["cat_cardinalities"] == [3]
    assert len(spec["cat_weights"]) == 1 and spec["cat_weights"][0].shape == (3,)

    gate = _FeasibilityGate.from_spec(spec)
    Xt = torch.tensor(X, dtype=torch.double, requires_grad=True)
    log_p = gate.log_p_feasible(Xt)
    p_torch = torch.exp(log_p).detach().numpy()
    p_sk = clf.predict_proba(X)
    assert np.allclose(p_torch, p_sk, atol=1e-12)

    log_p.sum().backward()
    assert Xt.grad is not None
    assert torch.isfinite(Xt.grad).all()
    assert torch.any(Xt.grad[:, 0] != 0.0)
    assert torch.any(Xt.grad[:, 2] != 0.0)
    assert torch.all(Xt.grad[:, 1] == 0.0)


def test_onehot_feasibility_separates_non_ordinal_middle_category():
    # Category 1 is infeasible while categories 0 and 2 are feasible:
    # an ordinal linear classifier on code 0/1/2 cannot separate {1} from {0, 2}.
    rng = np.random.default_rng(7)
    n = 90
    x_cont = rng.uniform(0.0, 1.0, n)
    cat = np.tile([0, 1, 2], n // 3)
    y = np.where(cat == 1, 0.0, 5.0 + 0.5 * x_cont + rng.normal(0, 0.1, n))
    X = np.column_stack([x_cont, cat])

    rep_ord = feasibility_cv_report(X, y, n_splits=5, seed=0)
    rep_oh = feasibility_cv_report(
        X, y, n_splits=5, seed=0, cat_dims=[1], cat_cardinalities=[3]
    )
    assert rep_ord["auc"] < 0.65
    assert rep_oh["auc"] > 0.95

    clf_oh = FeasibilityClassifier(cat_dims=[1], cat_cardinalities=[3]).fit(X, feasible_labels(y))
    p_levels = clf_oh.predict_proba(np.array([[0.5, 0.0], [0.5, 1.0], [0.5, 2.0]]))
    assert p_levels[0] > 0.8 and p_levels[2] > 0.8 and p_levels[1] < 0.2

    # Continuous-only propose() still works with legacy torch_gate_params()
    X_cont = rng.uniform(0, 1, (30, 2))
    y_cont = np.where(X_cont[:, 0] > 0.7, 0.0, 3.0 + X_cont[:, 1])
    s = Surrogate().fit(X_cont, y_cont, bounds=np.array([[0, 0], [1, 1]]))
    clf_cont = FeasibilityClassifier().fit(X_cont, feasible_labels(y_cont))
    assert clf_cont.torch_gate_params()[0].shape == (2,)
    pts = propose(s, np.array([[0, 0], [1, 1]]), q=2, seed=0, feasibility_classifier=clf_cont)
    assert pts.shape == (2, 2)


def test_feas_onehot_cat_flag_on_and_off_in_analyze(monkeypatch):
    monkeypatch.delenv("KALOS_FEAS_ONEHOT_CAT", raising=False)
    assert feas_onehot_cat_setting() is False
    check_env_settings()

    monkeypatch.setenv("KALOS_FEAS_ONEHOT_CAT", "bad")
    with pytest.raises(ConfigError, match="KALOS_FEAS_ONEHOT_CAT"):
        check_env_settings()

    # Sheet where middle alphabetical category ("MedB", code 1) is infeasible
    # while "MedA" (code 0) and "MedC" (code 2) are strong producers.
    rng = np.random.default_rng(12)
    n = 45
    medium = np.tile(["MedA", "MedB", "MedC"], n // 3)
    ph = rng.uniform(6.0, 8.0, n)
    feed = rng.uniform(1.0, 5.0, n)
    titer = np.where(
        medium == "MedB",
        0.0,
        4.0 + 0.5 * feed - 0.3 * (ph - 7.0) ** 2 + rng.normal(0, 0.1, n),
    )
    df = pd.DataFrame({
        "medium": medium,
        "pH": ph.round(3),
        "feed": feed.round(3),
        "titer": titer.round(3),
    })

    monkeypatch.setenv("KALOS_FEAS_ONEHOT_CAT", "0")
    out_off = _analyze(df, "titer")
    assert out_off["proposal_gating"]["gated"] is False

    monkeypatch.setenv("KALOS_FEAS_ONEHOT_CAT", "1")
    out_on = _analyze(df, "titer")
    assert out_on["proposal_gating"]["gated"] is True
    assert out_on["proposal_gating"]["feasibility_auc"] > 0.9
    for p in out_on["proposals"]:
        assert p["recipe"]["medium"] in {"MedA", "MedC"}
