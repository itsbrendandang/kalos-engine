"""Tests for the real-data validation module and the mechanistic bridge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.bench.real_data import (
    DATASET_0, RealDataMapping, compare, frame_to_pool, indpensim_path, load_indpensim_pool,
    load_real_frame, plotly_html, run_pool_bo, write_report,
)
from kalos.core.mechanistic import HybridSurrogate, MechanisticModel, ODEModel, monod_example

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/datasets/dataset_0.csv"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="dataset_0.csv not present")


# ---- real-data module ----------------------------------------------------

@needs_data
def test_load_excludes_contaminated_by_default():
    all_rows = load_real_frame(DATA, exclude_contaminated=False)
    clean = load_real_frame(DATA)
    assert len(all_rows) == 715
    assert 0 < len(clean) < len(all_rows)
    assert (clean["contamination"] == "N").all()


@needs_data
def test_column_roles_mapping():
    roles = DATASET_0.to_column_roles()
    assert roles.target == "max_yield_fpu_ml"
    assert "duration_days" in roles.features and "ph" in roles.features
    assert roles.target not in roles.features


def test_load_rejects_bad_path_and_missing_columns(tmp_path):
    with pytest.raises(ValueError):
        load_real_frame(tmp_path / "nope.csv")
    txt = tmp_path / "x.txt"
    txt.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_real_frame(txt)
    bad = tmp_path / "x.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_real_frame(bad, RealDataMapping(titer="t", timepoint="a", drivers=("b",)))


def test_run_pool_bo_shapes_and_monotone():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, 3))
    y = -((X - 0.5) ** 2).sum(1) + 1.0
    res = run_pool_bo(X, y, rounds=3, q=2, n_init=5, seed=0)
    assert len(res["bo"]) == len(res["random"]) == len(res["lhs"]) == 4
    assert all(b >= a for a, b in zip(res["lhs"], res["lhs"][1:]))
    assert all(b >= a for a, b in zip(res["bo"], res["bo"][1:]))
    with pytest.raises(ValueError, match="too small"):
        run_pool_bo(X[:8], y[:8], rounds=3, q=2, n_init=5)


def test_report_written_and_html_escaped(tmp_path):
    curve = {"n_rows": 1, "bo_mean": [0.1, 0.2], "bo_std": [0, 0], "random_mean": [0.1, 0.1]}
    summary = {"real": curve, "synthetic": curve}
    jp, hp = write_report(summary, tmp_path / "out")
    assert json.loads(jp.read_text())["real"]["n_rows"] == 1
    assert "Plotly.newPlot" in hp.read_text()
    assert "</script><script>" not in plotly_html(summary, title="</script><script>alert(1)")


# ---- mechanistic bridge --------------------------------------------------

def test_monod_example_is_mechanistic_model_and_monotone_in_substrate():
    m = monod_example()
    assert isinstance(m, MechanisticModel)
    y = m.evaluate(np.array([[0.3, 0.5, 2.0], [0.3, 0.5, 10.0]]))
    assert np.isfinite(y).all() and y[1] > y[0]


def test_ode_failure_yields_nan_not_exception():
    bad = ODEModel(lambda t, s, p: [np.inf], lambda x: x, lambda x: [1.0], lambda t, y: y[0, -1], 1.0)
    assert np.isnan(bad.evaluate(np.array([[1.0]]))).all()
    with pytest.raises(ValueError):
        ODEModel(lambda t, s, p: [0], lambda x: x, lambda x: [1.0], lambda t, y: 0.0, 0.0)


def test_hybrid_surrogate_fit_predict():
    rng = np.random.default_rng(1)
    lo, hi = np.array([0.1, 0.1, 1.0]), np.array([0.6, 1.0, 10.0])
    X = lo + rng.uniform(size=(20, 3)) * (hi - lo)
    mech = monod_example()
    y = mech.evaluate(X) * 1.1 + rng.normal(0, 0.01, 20)  # mechanistic model is 10% off
    h = HybridSurrogate(mech)
    with pytest.raises(RuntimeError):
        h.predict(X)
    h.fit(X, y, bounds=np.vstack([lo, hi]))
    mu, sd = h.predict(X)
    assert mu.shape == sd.shape == (20,)
    assert np.sqrt(np.mean((mu - y) ** 2)) < 0.5 * np.sqrt(np.mean((mech.evaluate(X) - y) ** 2))
    with pytest.raises(TypeError):
        HybridSurrogate(object())


def test_hybrid_accepts_plain_callable_and_none():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(15, 2))
    y = X.sum(1)
    b = np.array([[0, 0], [1, 1]], float)
    mu_c, _ = HybridSurrogate(lambda Z: np.atleast_2d(Z).sum(1)).fit(X, y, b).predict(X)
    mu_n, _ = HybridSurrogate(None).fit(X, y, b).predict(X)
    assert np.allclose(mu_c, y, atol=1e-2) and np.allclose(mu_n, y, atol=0.3)


def test_campaign_sql_identifier_guard():
    from kalos.portal.campaign import _checked_identifier

    assert _checked_identifier("campaigns_backup_3") == "campaigns_backup_3"
    for bad in ("x; DROP TABLE campaigns", "a b", "1abc", "", 'a"b', "campaigns_backup_1\n"):
        with pytest.raises(ValueError):
            _checked_identifier(bad)


def test_compare_requires_five_seeds():
    X = np.random.default_rng(0).uniform(size=(40, 2))
    with pytest.raises(ValueError, match="at least 5 seeds"):
        compare((X, X.sum(1)), (X, X.sum(1)), seeds=[0, 1, 2])


def test_indpensim_loader_aggregates_per_batch(tmp_path, monkeypatch):
    rows = ["Batch ID,Penicillin concentration(P:g/L),a,b"]
    for b in (1, 2, 3):
        for t in range(4):
            rows.append(f"{b},{b * 10 + t},{b + t},{2 * b}")
    f = tmp_path / "100_Batches_IndPenSim_V3.csv"
    f.write_text("\n".join(rows))
    X, y, names = load_indpensim_pool(f, recipe_cols=("a", "b"))
    assert X.shape == (3, 2) and y.tolist() == [13.0, 23.0, 33.0] and names == ["a", "b"]
    monkeypatch.setenv("KALOS_MEDIA_DATA", str(tmp_path))
    monkeypatch.delenv("KALOS_INDPENSIM_PATH", raising=False)
    assert indpensim_path() == f.resolve()
    with pytest.raises(ValueError, match="missing columns"):
        load_indpensim_pool(f)  # default IndPenSim column names are absent
    monkeypatch.delenv("KALOS_MEDIA_DATA")
    with pytest.raises(FileNotFoundError):
        load_indpensim_pool(None)
