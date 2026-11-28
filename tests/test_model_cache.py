"""Tests for kalos.core.model_cache (SurrogateCache)."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kalos.core.model_cache import SurrogateCache


def test_model_cache_import_is_torch_free() -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kalos.core.model_cache; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "False"


def _sample_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    X_cont = rng.uniform(0.0, 5.0, size=(12, 2))
    cat = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float).reshape(-1, 1)
    X_mixed = np.hstack([X_cont, cat])
    y = 1.5 * X_cont[:, 0] - 0.4 * X_cont[:, 1] + 0.3 * cat[:, 0] + rng.normal(0.0, 0.05, size=12)
    b_cont = np.array([[0.0, 0.0], [5.0, 5.0]])
    b_mixed = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 2.0]])
    return X_cont, X_mixed, y, (b_cont, b_mixed)


@pytest.mark.parametrize("kind", ["continuous", "mixed", "fixed_noise_cont", "fixed_noise_mixed"])
def test_surrogate_cache_roundtrip_posterior_identity(tmp_path: Path, kind: str) -> None:
    pytest.importorskip("botorch")
    from kalos.core.surrogate import Surrogate

    X_cont, X_mixed, y, bounds_pair = _sample_data()
    if kind in ("continuous", "fixed_noise_cont"):
        X = X_cont
        bounds = bounds_pair[0]
        cat_dims = None
        X_test = np.array([[1.2, 3.4], [4.1, 0.8]])
    else:
        X = X_mixed
        bounds = bounds_pair[1]
        cat_dims = [2]
        X_test = np.array([[1.2, 3.4, 0.0], [4.1, 0.8, 2.0]])

    noise = np.linspace(0.01, 0.05, len(y)) if "fixed_noise" in kind else None
    surrogate = Surrogate().fit(
        X, y, bounds=bounds, cat_dims=cat_dims, noise=noise,
        restarts=2, restart_seed=17, check_conditioning=True,
    )

    cache = SurrogateCache(tmp_path / "models")
    assert cache.get(
        X, y, bounds, cat_dims=cat_dims, noise=noise,
        restarts=2, restart_seed=17, check_conditioning=True,
    ) is None

    key = cache.put(
        surrogate, X, y, bounds, cat_dims=cat_dims, noise=noise,
        restarts=2, restart_seed=17, check_conditioning=True,
    )
    assert key is not None
    assert not list((tmp_path / "models").glob("*.tmp"))

    loaded = cache.get(
        X, y, bounds, cat_dims=cat_dims, noise=noise,
        restarts=2, restart_seed=17, check_conditioning=True,
    )
    assert loaded is not None
    assert loaded.fit_diagnostics == surrogate.fit_diagnostics
    assert loaded._cat_dims == surrogate._cat_dims
    assert np.allclose(loaded._y, surrogate._y)

    for obs in (False, True):
        m1, sd1 = surrogate.posterior(X_test, observation_noise=obs)
        m2, sd2 = loaded.posterior(X_test, observation_noise=obs)
        assert np.allclose(m1, m2, atol=1e-12, rtol=1e-12)
        assert np.allclose(sd1, sd2, atol=1e-12, rtol=1e-12)


def test_surrogate_cache_invalidates_on_any_input_change(tmp_path: Path) -> None:
    pytest.importorskip("botorch")
    from kalos.core.surrogate import Surrogate

    X_cont, _X_mixed, y, bounds_pair = _sample_data()
    bounds = bounds_pair[0]
    s = Surrogate().fit(X_cont, y, bounds=bounds, noise=0.02, restarts=1, restart_seed=0)
    cache = SurrogateCache(tmp_path)
    cache.put(
        s, X_cont, y, bounds, cat_dims=None, noise=0.02,
        restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0",
    )

    X_alt = X_cont.copy()
    X_alt[0, 0] += 1e-4
    y_alt = y.copy()
    y_alt[0] += 1e-4
    b_alt = bounds.copy()
    b_alt[1, 0] += 0.1

    variants = [
        dict(X_fit=X_alt, y_fit=y, bounds=bounds, cat_dims=None, noise=0.02, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y_alt, bounds=bounds, cat_dims=None, noise=0.02, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=b_alt, cat_dims=None, noise=0.02, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=[1], noise=0.02, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=None, noise=0.03, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=None, noise=None, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=None, noise=0.02, restarts=2, restart_seed=0, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=None, noise=0.02, restarts=1, restart_seed=1, check_conditioning=False, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=None, noise=0.02, restarts=1, restart_seed=0, check_conditioning=True, engine_version="0.1.0"),
        dict(X_fit=X_cont, y_fit=y, bounds=bounds, cat_dims=None, noise=0.02, restarts=1, restart_seed=0, check_conditioning=False, engine_version="0.2.0"),
    ]
    for kwargs in variants:
        assert cache.get(**kwargs) is None


def test_surrogate_cache_corrupt_entries_fail_soft(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("botorch")
    from kalos.core.surrogate import Surrogate

    X_cont, _X_mixed, y, bounds_pair = _sample_data()
    bounds = bounds_pair[0]
    s = Surrogate().fit(X_cont, y, bounds=bounds)
    cache = SurrogateCache(tmp_path)
    key = cache.put(s, X_cont, y, bounds)
    assert key is not None

    weights_path = tmp_path / f"{key}.pt"
    meta_path = tmp_path / f"{key}.json"
    orig_meta = meta_path.read_text(encoding="utf-8")

    # 1. Corrupt weights file
    weights_path.write_bytes(b"not a valid torch state dict")
    with caplog.at_level(logging.WARNING):
        assert cache.get(X_cont, y, bounds) is None
    assert any("SurrogateCache" in rec.message for rec in caplog.records)

    # 2. Corrupt JSON metadata
    caplog.clear()
    meta_path.write_text("{bad json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert cache.get(X_cont, y, bounds) is None
    assert any("SurrogateCache" in rec.message for rec in caplog.records)

    # 3. Tampered shape metadata
    caplog.clear()
    bad_meta = json.loads(orig_meta)
    bad_meta["shape"] = [999, 2]
    meta_path.write_text(json.dumps(bad_meta), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert cache.get(X_cont, y, bounds) is None
    assert any("SurrogateCache" in rec.message for rec in caplog.records)


def test_surrogate_cache_lru_eviction(tmp_path: Path) -> None:
    pytest.importorskip("botorch")
    from kalos.core.surrogate import Surrogate

    X_cont, _X_mixed, y, bounds_pair = _sample_data()
    bounds = bounds_pair[0]
    cache = SurrogateCache(tmp_path, max_entries=3)

    keys: list[str] = []
    ys: list[np.ndarray] = []
    for i in range(3):
        yi = y + i * 0.5
        s = Surrogate().fit(X_cont, yi, bounds=bounds)
        k = cache.put(s, X_cont, yi, bounds)
        assert k is not None
        keys.append(k)
        ys.append(yi)

    # Touch entry 0 via get() so entry 1 becomes the least-recently-used.
    assert cache.get(X_cont, ys[0], bounds) is not None

    y3 = y + 3 * 0.5
    s3 = Surrogate().fit(X_cont, y3, bounds=bounds)
    k3 = cache.put(s3, X_cont, y3, bounds)
    assert k3 is not None

    remaining = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert len(remaining) == 3
    assert keys[1] not in remaining
    assert set(remaining) == {keys[0], keys[2], k3}
    assert not (tmp_path / f"{keys[1]}.pt").exists()
