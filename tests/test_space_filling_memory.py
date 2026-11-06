"""Security review S4: the maximin batch never builds a candidates x rows x
inputs array. At 2000 rows x 40 inputs the old code peaked near 2.8 GB."""
from __future__ import annotations

import tracemalloc

import numpy as np

from kalos.core.space_filling import _nearest, maximin_batch


def test_maximin_peak_memory_at_2000_rows_by_40_inputs():
    d = 40
    X = np.random.default_rng(0).random((2000, d))
    b = np.vstack([np.zeros(d), np.ones(d)])
    tracemalloc.start()
    try:
        out = maximin_batch(X, b, 4)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert out.shape == (4, d)
    assert peak < 200e6, f"peak {peak / 1e6:.0f} MB"


def test_nearest_matches_the_direct_formula_with_categories():
    rng = np.random.default_rng(1)
    C = rng.random((50, 3))
    P = rng.random((7, 3))
    C[:, 2] = rng.integers(0, 3, 50)
    P[:, 2] = rng.integers(0, 3, 7)
    d2 = ((C[:, None, :2] - P[None, :, :2]) ** 2).sum(-1) + (C[:, None, 2] != P[None, :, 2])
    assert np.allclose(_nearest(C, P, [2]), np.sqrt(d2).min(axis=1))
    assert np.isinf(_nearest(C, np.empty((0, 3)), [2])).all()
