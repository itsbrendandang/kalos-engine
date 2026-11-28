"""End-to-end integration tests for the opt-in KALOS_FRONTIER_ENGINE=1 pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core.frontier import WarpedSparseGP
from kalos.core.gp_shape import ard_lengthscales, gp_shape_report
from kalos.envconfig import ConfigError
from kalos.portal.analysis import _analyze
from kalos.portal.config import frontier_engine_setting


def _make_bioprocess_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_unique = 16
    glucose = rng.uniform(2.0, 20.0, size=n_unique)
    ph = rng.uniform(6.2, 7.6, size=n_unique)
    temp = rng.uniform(30.0, 38.0, size=n_unique)
    # Replicate first 4 recipes so replicate_aware path also gets tested when needed
    idx = list(range(n_unique)) + [0, 1, 2, 3]
    g_col = glucose[idx]
    p_col = ph[idx]
    t_col = temp[idx]
    y_latent = (
        1.5
        + 8.0 * np.exp(-0.5 * ((p_col - 6.9) / 0.35) ** 2)
        * (g_col / (4.0 + g_col))
        - 0.08 * (t_col - 34.0) ** 2
    )
    titer = np.maximum(y_latent + rng.normal(0.0, 0.2, size=len(idx)), 0.1)
    batches = [f"B{(i % 4) + 1}" for i in range(len(idx))]
    return pd.DataFrame({
        "batch": batches,
        "glucose_g_l": np.round(g_col, 4),
        "ph": np.round(p_col, 4),
        "temp_c": np.round(t_col, 4),
        "titer_g_l": np.round(titer, 4),
    })


def test_frontier_gp_shape_and_ard_lengthscales() -> None:
    """ard_lengthscales and gp_shape_report inspect WarpedSparseGP cleanly."""
    rng = np.random.default_rng(11)
    d = 4
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    X = rng.uniform(0.0, 1.0, size=(30, d))
    # Dim 0 sharp interior optimum, dim 1 monotonic up, dims 2-3 inert
    y = 6.0 * np.exp(-8.0 * (X[:, 0] - 0.5) ** 2) + 2.5 * X[:, 1] + rng.normal(0.0, 0.1, size=30)

    gp = WarpedSparseGP(sparsity_strength=0.1, robust=False).fit(X, y, bounds, seed=11)
    ls = ard_lengthscales(gp, d)
    assert ls is not None and ls.shape == (d,)
    assert np.all(np.isfinite(ls)) and np.all(ls > 0.0)

    report = gp_shape_report(
        gp,
        X,
        y,
        feature_names=["x0", "x1", "x2", "x3"],
        cv_spearman=0.90,
    ).to_dict()
    assert report["available"] is True
    by_name = {f["name"]: f for f in report["features"]}
    assert by_name["x0"]["shape"] == "interior_optimum"
    assert by_name["x1"]["shape"] == "monotonic_up"


def test_frontier_engine_config_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """frontier_engine_setting defaults to False and fails closed on invalid strings."""
    monkeypatch.delenv("KALOS_FRONTIER_ENGINE", raising=False)
    assert frontier_engine_setting() is False

    monkeypatch.setenv("KALOS_FRONTIER_ENGINE", "1")
    assert frontier_engine_setting() is True

    monkeypatch.setenv("KALOS_FRONTIER_ENGINE", "invalid_value")
    with pytest.raises(ConfigError):
        frontier_engine_setting()


def test_analyze_with_frontier_engine_opt_in_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """KALOS_FRONTIER_ENGINE=1 swaps in WarpedSparseGP + JointQLogNEI; default stays Surrogate."""
    df = _make_bioprocess_df()

    # 1. Opt-in frontier engine with setup constraints & minimize goal
    monkeypatch.setenv("KALOS_FRONTIER_ENGINE", "1")
    setup = {
        "goal": "minimize",
        "bounds": {
            "glucose_g_l": {"lo": 3.0, "hi": 18.0},
            "ph": {"lo": 6.3, "hi": 7.5},
            "temp_c": {"lo": 31.0, "hi": 37.0},
        },
        "linear_constraints": [{"coefficients": {"glucose_g_l": 1.0, "temp_c": 0.5}, "op": "<=", "rhs": 35.0}],
    }
    res_f = _analyze(df, "titer_g_l", setup=setup)
    assert res_f["model_diagnostics"]["surrogate_kind"] == "warped_sparse_gp"
    assert res_f["gp_shapes"]["available"] is True
    props = res_f["proposals"]
    assert 1 <= len(props) <= 5
    for row in props:
        rec = row["recipe"]
        assert 3.0 - 1e-6 <= float(rec["glucose_g_l"]) <= 18.0 + 1e-6
        assert 6.3 - 1e-6 <= float(rec["ph"]) <= 7.5 + 1e-6
        assert 31.0 - 1e-6 <= float(rec["temp_c"]) <= 37.0 + 1e-6
        assert float(rec["glucose_g_l"]) + 0.5 * float(rec["temp_c"]) <= 35.0 + 1e-5

    # 2. Default (unset) keeps standard BoTorch Surrogate
    monkeypatch.delenv("KALOS_FRONTIER_ENGINE", raising=False)
    res_def = _analyze(df, "titer_g_l")
    assert res_def["model_diagnostics"].get("surrogate_kind") != "warped_sparse_gp"
    assert 1 <= len(res_def["proposals"]) <= 5
