"""Tests for the labelled synthetic chromatogram fixture."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from kalos.bench.chromatogram import (
    BASELINE_KINDS,
    generate_chromatogram,
    make_baseline,
    make_peak,
)


def test_deterministic_by_seed():
    a = generate_chromatogram(7)
    b = generate_chromatogram(7)
    np.testing.assert_array_equal(a.signal, b.signal)
    assert a.peak_table().equals(b.peak_table())
    assert a.baseline_kind == b.baseline_kind


def test_different_seeds_differ():
    a = generate_chromatogram(1)
    b = generate_chromatogram(2)
    assert not np.allclose(a.signal, b.signal)


@pytest.mark.parametrize("seed", range(20))
def test_peak_count_and_centres_in_range(seed):
    c = generate_chromatogram(seed, n_peaks=(2, 5), run_time=20.0)
    assert 2 <= len(c.peaks) <= 5
    for p in c.peaks:
        assert 0.0 <= p.start <= p.center <= p.end <= 20.0
    centres = np.array([p.center for p in c.peaks])
    assert np.all(np.diff(centres) >= 1.5 - 1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_labelled_area_matches_numeric_integral(seed):
    c = generate_chromatogram(seed, n_points=20000)
    for p in c.peaks:
        numeric = trapezoid(p.evaluate(c.time), c.time)
        assert numeric == pytest.approx(p.area, rel=1e-4, abs=1e-6)
    total = trapezoid(c.clean - c.baseline, c.time)
    assert total == pytest.approx(sum(p.area for p in c.peaks), rel=1e-4)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian", "skewed"])
def test_apex_and_edges_are_exact(shape):
    p = make_peak(shape, 10.0, 0.4, 1.2, skew=4.0, run_window=(0.0, 20.0),
                  edge_fraction=0.05)
    t = np.linspace(0, 20, 200001)
    y = p.evaluate(t)
    assert t[np.argmax(y)] == pytest.approx(10.0, abs=1e-3)
    assert y.max() == pytest.approx(1.2, rel=1e-6)
    assert p.evaluate(np.array([p.start, p.end])) == pytest.approx([0.06, 0.06], rel=1e-5)


@pytest.mark.parametrize("seed", range(20))
def test_find_peaks_recovers_centres_on_noiseless_flat_trace(seed):
    c = generate_chromatogram(
        seed, n_peaks=(2, 4), width_range=(0.1, 0.4), min_separation=4.0,
        baseline="flat", noise_level=0.0,
    )
    idx, _ = find_peaks(c.signal)
    found = c.time[idx]
    for p in c.peaks:
        assert np.min(np.abs(found - p.center)) < 0.05


def test_noise_level_respected():
    c = generate_chromatogram(3, noise_level=0.01)
    resid = c.signal - c.clean
    assert 0.007 < resid.std() < 0.013


def test_baselines_differ_by_kind():
    t = np.linspace(0, 20, 500)
    curves = {k: make_baseline(t, k, 0.5) for k in BASELINE_KINDS}
    assert np.ptp(curves["flat"]) == 0
    assert np.all(np.diff(curves["drift"]) > 0)
    assert not np.allclose(curves["wavy"], curves["flat"])
    assert not np.allclose(curves["wavy"], curves["drift"])
    for k in BASELINE_KINDS:
        assert generate_chromatogram(0, baseline=k).baseline_kind == k


def test_frames_shape():
    c = generate_chromatogram(5)
    df = c.to_frame()
    assert list(df.columns) == ["time", "signal", "clean", "baseline"]
    assert len(df) == len(c.time)
    pt = c.peak_table()
    assert len(pt) == len(c.peaks)
    assert {"center", "start", "end", "height", "area"} <= set(pt.columns)


def test_impossible_separation_raises():
    with pytest.raises(ValueError):
        generate_chromatogram(0, n_peaks=5, min_separation=10.0)
