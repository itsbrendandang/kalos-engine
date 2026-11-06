"""Noise floor with a 95% interval and optional per-recipe noise (QUEUE round 3
item 5): kalos/core/replicates.py `pooled_noise_floor`,
`shrunken_recipe_variances`, `noise.floor` and `noise.per_recipe`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.core.replicates import (
    PER_RECIPE_DF0,
    noise_report,
    pooled_noise_floor,
    shrunken_recipe_variances,
)
from kalos.domains import ColumnRoles, InvalidPerRecipeNoise


def test_pooled_variance_is_df_weighted():
    out = pooled_noise_floor(np.array([1.0, 4.0, 0.0]), np.array([2, 4, 1]))
    # (1*1 + 3*4) / (1 + 3); the singleton contributes nothing.
    assert out["df"] == 4
    assert out["var_pooled"] == pytest.approx(13.0 / 4.0)
    lo, hi = out["var_ci95"]
    assert lo < out["var_pooled"] < hi
    assert out["sd_ci95"] == pytest.approx([np.sqrt(lo), np.sqrt(hi)])
    none = pooled_noise_floor(np.array([0.0, 0.0]), np.array([1, 1]))
    assert none["df"] == 0 and np.isnan(none["var_pooled"])


def test_noise_report_keeps_mean_of_variances_and_adds_pooled():
    X = np.array([[0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [2.0]])
    y = np.array([1.0, 2.0, 5.0, 7.0, 9.0, 11.0, 3.0])
    nr = noise_report(X, y)
    # mean of variances: (0.5 + 6.6667) / 2 (unchanged meaning)
    assert nr["noise_var"] == pytest.approx((0.5 + 20.0 / 3.0) / 2.0)
    assert nr["noise_var_pooled"] == pytest.approx((1 * 0.5 + 3 * 20.0 / 3.0) / 4.0)
    assert nr["noise_df"] == 4


def test_chi_square_interval_covers_about_95_percent():
    sigma = 0.4
    hits_var = hits_sd = 0
    n_seeds = 400
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        reps = rng.integers(2, 5, size=8)
        X = np.repeat(np.arange(8.0), reps).reshape(-1, 1)
        y = rng.uniform(0, 10, 8)[X[:, 0].astype(int)] + rng.normal(0, sigma, X.shape[0])
        nr = noise_report(X, y)
        lo, hi = nr["noise_var_ci95"]
        hits_var += lo <= sigma**2 <= hi
        slo, shi = nr["noise_sd_ci95"]
        hits_sd += slo <= sigma <= shi
    assert 0.92 <= hits_var / n_seeds <= 0.98
    assert hits_sd == hits_var


def test_shrinkage_pulls_toward_pooled():
    v = shrunken_recipe_variances(np.array([4.0, 0.0, 1.0]), np.array([5, 1, 2]), 1.0)
    assert v[0] == pytest.approx((4 * 4.0 + PER_RECIPE_DF0 * 1.0) / (4 + PER_RECIPE_DF0))
    assert v[1] == pytest.approx(1.0)  # singleton: the pooled value
    assert v[2] == pytest.approx(1.0)


def test_parse_per_recipe_noise():
    assert ColumnRoles.from_dict({"target": "y"}).per_recipe_noise is False
    assert ColumnRoles.from_dict({"target": "y", "per_recipe_noise": True}).per_recipe_noise is True
    for bad in ("yes", 1, 0, [True]):
        with pytest.raises(InvalidPerRecipeNoise):
            ColumnRoles.from_dict({"target": "y", "per_recipe_noise": bad})


def test_bad_per_recipe_noise_is_a_coded_400():
    from kalos.portal.pipeline import _roles_and_profile
    from kalos.portal.uploads import UploadRejected

    with pytest.raises(UploadRejected) as exc:
        _roles_and_profile(json.dumps({"target": "y", "per_recipe_noise": "yes"}))
    assert exc.value.code == "invalid_per_recipe_noise"


def _hetero_sheet(seed: int = 3, n_rec: int = 10, reps: int = 3) -> pd.DataFrame:
    """Replicated sheet whose noise sd grows with the recipe mean."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 10, n_rec)
    b = rng.uniform(1, 5, n_rec)
    mu = 2 + 0.8 * a - 0.3 * (b - 3) ** 2
    rows = []
    for i in range(n_rec):
        sd = 0.05 + 0.08 * mu[i]
        for _ in range(reps):
            rows.append({"a_g_L": round(a[i], 3), "b_g_L": round(b[i], 3),
                         "titer_g_L": round(mu[i] + rng.normal(0, sd), 4)})
    return pd.DataFrame(rows)


def _capture_fit_noise(monkeypatch) -> list:
    from kalos.core import surrogate

    seen: list = []
    real_fit = surrogate.Surrogate.fit

    def spy(self, X, y, bounds, *, noise=None, **kw):
        seen.append(None if noise is None else np.asarray(noise, float).copy())
        return real_fit(self, X, y, bounds, noise=noise, **kw)

    monkeypatch.setattr(surrogate.Surrogate, "fit", spy)
    return seen


def _fit_noises(seen: list, n_recipes: int) -> list:
    return [v for v in seen if v is not None and v.shape == (n_recipes,)]


def test_per_recipe_noise_on_uses_recipe_variances(monkeypatch):
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    seen = _capture_fit_noise(monkeypatch)
    roles = ColumnRoles.from_dict({"target": "titer_g_L", "per_recipe_noise": True})
    out = _analyze(_hetero_sheet(), roles=roles)
    json.dumps(out["noise"], allow_nan=False)
    pr = out["noise"]["per_recipe"]
    assert out["noise"]["scale"]["homoscedastic"] is False
    assert pr["enabled"] is True and pr["used"] is True and pr["df0"] == PER_RECIPE_DF0
    assert pr["sd_range"][0] < pr["sd_range"][1]
    (noise_vec,) = _fit_noises(seen, 10)
    assert np.unique(np.round(noise_vec, 12)).size > 1  # differs per recipe (all n = 3)
    floor = out["noise"]["floor"]
    assert floor["df"] == 20 and floor["sigma_ci95"][0] < floor["sigma"] < floor["sigma_ci95"][1]


def test_per_recipe_noise_off_by_default_keeps_the_pooled_fit(monkeypatch):
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    seen = _capture_fit_noise(monkeypatch)
    sheet = _hetero_sheet()
    out = _analyze(sheet, "titer_g_L")
    pr = out["noise"]["per_recipe"]
    assert pr["enabled"] is False and pr["used"] is False and pr["sd_range"] is None
    (noise_vec,) = _fit_noises(seen, 10)
    # The pre-existing proposal fit: mean-of-variances sigma^2 / n_reps for every recipe.
    nr = noise_report(sheet[["a_g_L", "b_g_L"]].to_numpy(float), sheet["titer_g_L"].to_numpy(float))
    np.testing.assert_array_equal(noise_vec, np.full(10, nr["noise_var"] / 3.0))
    explicit_off = _analyze(sheet, "titer_g_L", per_recipe_noise=False)
    assert explicit_off["proposals"] == out["proposals"]
