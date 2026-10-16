"""Tests for the real-data validation module and the mechanistic bridge."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kalos.bench.real_data import (
    DATASET_0, RealDataMapping, compare, indpensim_path, load_indpensim_pool,
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
    summary = {"arms": {"real_only": curve, "synthetic_fixture": curve}}
    jp, hp = write_report(summary, tmp_path / "out")
    assert json.loads(jp.read_text())["arms"]["real_only"]["n_rows"] == 1
    assert "Plotly.newPlot" in hp.read_text()
    html = plotly_html(summary, title="</script><script>alert(1)")
    assert "</script><script>" not in html
    body = html.split("<script>const f=", 1)[1].rsplit(";Plotly", 1)[0]
    assert "<" not in body and json.loads(body)["data"]  # still valid JSON


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


def test_compare_requires_five_seeds():
    X = np.random.default_rng(0).uniform(size=(40, 2))
    with pytest.raises(ValueError, match="at least 5 seeds"):
        compare({"a": (X, X.sum(1))}, seeds=[0, 1, 2])


def test_compare_shrinks_small_pool_and_skips_tiny_one():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(12, 2))
    out = compare({"small": (X, X.sum(1)), "tiny": (X[:4], X[:4].sum(1))}, rounds=10, q=2, n_init=6, seeds=list(range(5)))
    small = out["arms"]["small"]
    assert small["rounds"] < 10 and small["requested_rounds"] == 10 and small["n_init"] + small["rounds"] * 2 <= 12
    assert "skipped" in out["arms"]["tiny"]


def test_report_has_no_nan_curves_for_skipped_arm(tmp_path):
    out = {"arms": {"x": {"skipped": "tiny", "n_rows": 3}}}
    jp, hp = write_report(out, tmp_path)
    assert "Plotly.newPlot" in hp.read_text()


@needs_data
def test_real_only_filters_to_measured_rows_and_excludes_min_do():
    allr = load_real_frame(DATA, exclude_contaminated=False)
    real = load_real_frame(DATA, exclude_contaminated=False, real_only=True)
    assert 0 < len(real) < 25 < len(allr)
    assert (real["data_type"] == "Real").all()
    assert "min_do_pct" not in DATASET_0.features  # measured minimum, not a setpoint


def test_real_only_requires_data_type_column(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("t,a\n1,2\n")
    with pytest.raises(ValueError, match="data_type"):
        load_real_frame(f, RealDataMapping(titer="t", timepoint="a", drivers=()), real_only=True)


def test_non_finite_features_and_targets_are_dropped(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("t,a,b\n1,2,3\ninf,2,3\n1,inf,3\n1,-inf,3\n1,2,nan\n2,3,4\n")
    df = load_real_frame(f, RealDataMapping(titer="t", timepoint="a", drivers=("b",)))
    assert len(df) == 2 and np.isfinite(df[["t", "a", "b"]].to_numpy()).all()


def test_bo_and_random_share_initial_design_and_are_repeatable():
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(50, 3))
    y = -((X - 0.4) ** 2).sum(1)
    a = run_pool_bo(X, y, rounds=2, q=2, n_init=5, seed=7)
    b = run_pool_bo(X, y, rounds=2, q=2, n_init=5, seed=7)
    assert a == b  # same seed, identical outcome
    assert a["bo"][0] == a["random"][0]  # same starting best => same initial design
    assert run_pool_bo(X, y, rounds=2, q=2, n_init=5, seed=8) != a


def test_gp_is_only_fit_on_already_measured_rows(monkeypatch):
    from kalos.core.surrogate import Surrogate
    from kalos.bench.real_data import _normalize

    calls = []
    real_fit = Surrogate.fit

    def spy(self, X, y, *args, **kw):
        calls.append((np.array(X), np.array(y)))
        return real_fit(self, X, y, *args, **kw)

    monkeypatch.setattr(Surrogate, "fit", spy)
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(40, 2))
    y = X.sum(1)
    res = run_pool_bo(X, y, rounds=3, q=2, n_init=5, seed=1)
    ev, Xn = res["evaluated"], _normalize(X)
    assert len(calls) == 3
    for r, (Xf, yf) in enumerate(calls):
        seen = ev[: 5 + r * 2]  # rows revealed BEFORE round r
        assert np.allclose(Xf, Xn[seen]) and np.allclose(yf, y[seen])
        assert len(set(ev[5 + r * 2:])) == len(ev[5 + r * 2:])  # later picks never seen by the fit


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


# ---- mechanistic hardening -----------------------------------------------

def _ode(rhs=None, params=None, y0=None, readout=None):
    return ODEModel(
        rhs or (lambda t, s, p: [-s[0]]),
        params or (lambda x: x),
        y0 or (lambda x: [1.0]),
        readout or (lambda t, y: y[0, -1]),
        1.0,
    )


@pytest.mark.parametrize("which", ["rhs", "params", "y0", "readout"])
@pytest.mark.parametrize("exc", [RuntimeError, TypeError, IndexError, KeyError])
def test_ode_callbacks_that_raise_yield_nan(which, exc):
    def boom(*a, **k):
        raise exc("x")

    kw = {"rhs": dict(rhs=boom), "params": dict(params=boom), "y0": dict(y0=boom), "readout": dict(readout=boom)}[which]
    assert np.isnan(_ode(**kw).evaluate(np.array([[1.0], [2.0]]))).all()


def test_ode_nan_readout_is_nan_and_good_rows_survive():
    ok = _ode(y0=lambda x: [x[0]], readout=lambda t, y: np.nan if y[0, 0] > 1.5 else y[0, -1])
    out = ok.evaluate(np.array([[1.0], [2.0]]))
    assert np.isfinite(out[0]) and np.isnan(out[1])


@pytest.mark.parametrize("bad", [lambda X: 1.0, lambda X: np.zeros(3), lambda X: np.zeros((len(X), 1))])
def test_hybrid_rejects_scalar_and_wrong_shape_output(bad):
    X = np.random.default_rng(0).uniform(size=(8, 2))
    with pytest.raises(ValueError, match="one value per row"):
        HybridSurrogate(bad).fit(X, X.sum(1), np.array([[0, 0], [1, 1]], float))


def test_script_exits_cleanly_when_dataset_missing(tmp_path):
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(ROOT / "examples/real_data_validation.py"), "--data", str(tmp_path / "nope.csv")],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 2 and "dataset not found" in r.stderr and "Traceback" not in r.stderr
