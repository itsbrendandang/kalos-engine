"""Labelled synthetic chromatograms: a ground-truth fixture for peak ingestion.

What this IS: a seeded generator of 1-D chromatogram-like traces (time vs
detector signal) built from 2-5 analytic peaks (Gaussian, Lorentzian, or skewed
Gaussian) on a flat, drifting, or wavy baseline plus Gaussian noise, returned
together with the exact labels a peak-integration step should recover: apex
time, start/end, height, and area. Because every peak is an analytic function,
the labels are computed from the noiseless model, not estimated from the trace:
areas are closed-form integrals over the run window (Gaussian via the normal
CDF, Lorentzian via arctan, skewed via the skew-normal CDF), and start/end are
the times where the noiseless peak falls below `edge_fraction * height`.

What this is NOT: a model of a real column, detector, or separation. Peak
shapes, widths, baselines, and noise are generic placeholders chosen so a parser
or integrator has something with known answers to be tested against. It carries
NO link to process factors or yield - the idea this was re-implemented from
attached a "yield" computed from random, unrelated factors, and that link was
deliberately not reproduced. This is an ingestion fixture, not an optimization
(BO) surface, and it is not registered as a benchmark objective.

Deterministic: all randomness comes from one `numpy.random.Generator` seeded by
the caller; no stdlib `random`, no global numpy state.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

PeakShape = Literal["gaussian", "lorentzian", "skewed"]
BaselineKind = Literal["flat", "drift", "wavy"]

PEAK_SHAPES: tuple[PeakShape, ...] = ("gaussian", "lorentzian", "skewed")
BASELINE_KINDS: tuple[BaselineKind, ...] = ("flat", "drift", "wavy")


def _skew_mode(alpha: float) -> float:
    """Mode of the standard skew-normal with shape `alpha` (numerical)."""
    if alpha == 0.0:
        return 0.0
    res = optimize.minimize_scalar(
        lambda z: -stats.skewnorm.pdf(z, alpha), bounds=(-3.0, 3.0), method="bounded",
        options={"xatol": 1e-10},
    )
    return float(res.x)


@dataclass(frozen=True)
class Peak:
    """One labelled peak.

    `center` is the APEX time (for the skewed shape too: the skew-normal is
    shifted so its mode sits exactly at `center`). `width` is the shape's scale:
    the standard deviation for a Gaussian, the half-width at half-maximum for a
    Lorentzian, and the skew-normal scale for a skewed peak. `area` is the exact
    integral of the noiseless peak over the run window; `start`/`end` are where
    the noiseless peak drops below `edge_fraction * height`, clipped to the run.
    """

    shape: PeakShape
    center: float
    width: float
    height: float
    skew: float
    area: float
    start: float
    end: float

    def _skew_params(self) -> tuple[float, float]:
        """(loc, pdf value at the mode) for the skewed shape."""
        mode = _skew_mode(self.skew)
        loc = self.center - mode * self.width
        peak_pdf = float(stats.skewnorm.pdf(mode, self.skew)) / self.width
        return loc, peak_pdf

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """The noiseless peak contribution at times `t`."""
        t = np.asarray(t, dtype=float)
        if self.shape == "gaussian":
            return self.height * np.exp(-0.5 * ((t - self.center) / self.width) ** 2)
        if self.shape == "lorentzian":
            return self.height / (1.0 + ((t - self.center) / self.width) ** 2)
        loc, peak_pdf = self._skew_params()
        pdf = stats.skewnorm.pdf(t, self.skew, loc=loc, scale=self.width)
        return self.height * pdf / peak_pdf


def _area(shape: PeakShape, center: float, width: float, height: float, skew: float,
          t0: float, t1: float) -> float:
    """Exact integral of the noiseless peak over [t0, t1]."""
    if shape == "gaussian":
        a, b = (t0 - center) / width, (t1 - center) / width
        return float(height * width * np.sqrt(2 * np.pi) * (special.ndtr(b) - special.ndtr(a)))
    if shape == "lorentzian":
        a, b = (t0 - center) / width, (t1 - center) / width
        return float(height * width * (np.arctan(b) - np.arctan(a)))
    mode = _skew_mode(skew)
    loc = center - mode * width
    peak_pdf = float(stats.skewnorm.pdf(mode, skew)) / width
    mass = stats.skewnorm.cdf(t1, skew, loc=loc, scale=width) - stats.skewnorm.cdf(
        t0, skew, loc=loc, scale=width
    )
    return float(height * mass / peak_pdf)


def _edges(shape: PeakShape, center: float, width: float, skew: float, frac: float,
           t0: float, t1: float) -> tuple[float, float]:
    """Times either side of the apex where the noiseless peak equals `frac * height`."""
    if shape == "gaussian":
        half = width * np.sqrt(2.0 * np.log(1.0 / frac))
        lo, hi = center - half, center + half
    elif shape == "lorentzian":
        half = width * np.sqrt(1.0 / frac - 1.0)
        lo, hi = center - half, center + half
    else:
        mode = _skew_mode(skew)
        target = frac * float(stats.skewnorm.pdf(mode, skew))

        def g(z: float) -> float:
            return float(stats.skewnorm.pdf(z, skew)) - target

        span = 12.0
        z_lo = optimize.brentq(g, mode - span, mode)
        z_hi = optimize.brentq(g, mode, mode + span)
        lo = center + (z_lo - mode) * width
        hi = center + (z_hi - mode) * width
    return float(max(lo, t0)), float(min(hi, t1))


def make_peak(shape: PeakShape, center: float, width: float, height: float, *,
              skew: float = 0.0, run_window: tuple[float, float] = (0.0, 20.0),
              edge_fraction: float = 0.01) -> Peak:
    """Build a `Peak` with its exact labels (area, start, end) for a run window."""
    if shape not in PEAK_SHAPES:
        raise ValueError(f"unknown peak shape {shape!r}; expected one of {PEAK_SHAPES}")
    if width <= 0 or height <= 0:
        raise ValueError("peak width and height must be positive")
    if not 0.0 < edge_fraction < 1.0:
        raise ValueError("edge_fraction must be in (0, 1)")
    skew = float(skew) if shape == "skewed" else 0.0
    t0, t1 = run_window
    start, end = _edges(shape, center, width, skew, edge_fraction, t0, t1)
    return Peak(
        shape=shape, center=float(center), width=float(width), height=float(height),
        skew=skew, area=_area(shape, center, width, height, skew, t0, t1),
        start=start, end=end,
    )


def make_baseline(t: np.ndarray, kind: BaselineKind, strength: float) -> np.ndarray:
    """Baseline under the peaks: a constant offset, a linear ramp from 0 to
    `strength` across the run, or a slow sinusoid of amplitude `strength`
    riding on an offset of `strength` (so the trace stays positive-ish)."""
    t = np.asarray(t, dtype=float)
    span = float(t[-1] - t[0]) or 1.0
    if kind == "flat":
        return np.full_like(t, strength)
    if kind == "drift":
        return strength * (t - t[0]) / span
    if kind == "wavy":
        return strength + strength * np.sin(2.0 * np.pi * 1.5 * (t - t[0]) / span)
    raise ValueError(f"unknown baseline kind {kind!r}; expected one of {BASELINE_KINDS}")


@dataclass
class Chromatogram:
    """A generated trace and its ground truth.

    `signal` is what a detector would report (peaks + baseline + noise).
    `clean` is the same trace without noise and `baseline` the baseline alone,
    so `clean - baseline` is exactly the sum of `Peak.evaluate` over `peaks`.
    """

    time: np.ndarray
    signal: np.ndarray
    clean: np.ndarray
    baseline: np.ndarray
    peaks: tuple[Peak, ...]
    baseline_kind: BaselineKind
    noise_level: float
    seed: int | None = None
    meta: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """The trace as a long table: time, signal, clean, baseline."""
        return pd.DataFrame(
            {"time": self.time, "signal": self.signal, "clean": self.clean,
             "baseline": self.baseline}
        )

    def peak_table(self) -> pd.DataFrame:
        """One row per labelled peak, sorted by apex time."""
        cols = ["peak_id", "shape", "center", "start", "end", "height", "area", "width", "skew"]
        rows = [
            {"peak_id": i, "shape": p.shape, "center": p.center, "start": p.start,
             "end": p.end, "height": p.height, "area": p.area, "width": p.width,
             "skew": p.skew}
            for i, p in enumerate(self.peaks)
        ]
        return pd.DataFrame(rows, columns=cols)


def _sample_centers(rng: np.random.Generator, n: int, lo: float, hi: float,
                    min_separation: float, max_tries: int = 1000) -> np.ndarray:
    """Draw `n` apex times in [lo, hi] at least `min_separation` apart."""
    if n > 1 and (n - 1) * min_separation > hi - lo:
        raise ValueError(
            f"cannot place {n} peaks {min_separation} apart inside [{lo}, {hi}]"
        )
    for _ in range(max_tries):
        c = np.sort(rng.uniform(lo, hi, size=n))
        if n < 2 or np.min(np.diff(c)) >= min_separation:
            return c
    raise ValueError(
        f"could not place {n} peaks {min_separation} apart after {max_tries} draws; "
        "lower min_separation or n_peaks"
    )


def generate_chromatogram(
    seed: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    run_time: float = 20.0,
    n_points: int = 2000,
    n_peaks: int | tuple[int, int] = (2, 5),
    shapes: Sequence[PeakShape] = PEAK_SHAPES,
    width_range: tuple[float, float] = (0.1, 0.5),
    height_range: tuple[float, float] = (0.2, 1.5),
    skew_range: tuple[float, float] = (-5.0, 5.0),
    baseline: BaselineKind | None = None,
    baseline_strength: float = 0.5,
    noise_level: float = 0.01,
    min_separation: float = 1.5,
    edge_margin: float = 2.0,
    edge_fraction: float = 0.01,
) -> Chromatogram:
    """Generate one labelled chromatogram.

    `n_peaks` is a fixed count or an inclusive `(low, high)` range. Apex times
    are drawn uniformly inside `[edge_margin, run_time - edge_margin]` and kept
    at least `min_separation` apart so labelled peaks stay resolvable (raise
    `min_separation` relative to `width_range` for baseline-separated peaks;
    heavy Lorentzian tails still overlap). `baseline=None` draws the kind at
    random. Pass either `seed` or a ready `rng`; the same seed gives the same
    trace and labels.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if run_time <= 2 * edge_margin:
        raise ValueError("run_time must exceed 2 * edge_margin")
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    if isinstance(n_peaks, int):
        k = n_peaks
    else:
        k = int(rng.integers(n_peaks[0], n_peaks[1] + 1))
    if k < 1:
        raise ValueError("need at least one peak")
    kind: BaselineKind = baseline if baseline is not None else BASELINE_KINDS[
        int(rng.integers(len(BASELINE_KINDS)))
    ]
    if kind not in BASELINE_KINDS:
        raise ValueError(f"unknown baseline kind {kind!r}; expected one of {BASELINE_KINDS}")

    t = np.linspace(0.0, run_time, n_points)
    centers = _sample_centers(rng, k, edge_margin, run_time - edge_margin, min_separation)
    peaks: list[Peak] = []
    for c in centers:
        shape = shapes[int(rng.integers(len(shapes)))]
        width = float(rng.uniform(*width_range))
        height = float(rng.uniform(*height_range))
        skew = float(rng.uniform(*skew_range)) if shape == "skewed" else 0.0
        peaks.append(make_peak(shape, float(c), width, height, skew=skew,
                               run_window=(0.0, run_time), edge_fraction=edge_fraction))

    base = make_baseline(t, kind, baseline_strength)
    clean = base + np.sum([p.evaluate(t) for p in peaks], axis=0)
    noise = rng.normal(0.0, noise_level, size=n_points) if noise_level > 0 else 0.0
    return Chromatogram(
        time=t, signal=clean + noise, clean=clean, baseline=base, peaks=tuple(peaks),
        baseline_kind=kind, noise_level=float(noise_level), seed=seed,
    )


__all__ = [
    "BASELINE_KINDS",
    "PEAK_SHAPES",
    "Chromatogram",
    "Peak",
    "generate_chromatogram",
    "make_baseline",
    "make_peak",
]
