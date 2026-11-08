"""Proposals must not depend on what else is running (review C3).

torch's default generator and numpy's global RNG are process-wide. Two analyses
on two threads used to reseed each other mid-fit (busy.py admits two), so the
batch changed with thread timing. `_analyze` now runs inside `_seeded_analysis`
(lock + fork), and `propose_batch` gets `seed=ANALYZE_SEED` explicitly.
"""
from __future__ import annotations

import json
import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kalos.portal import analysis
from kalos.portal.analysis import ANALYZE_SEED, _analyze


def _sheet() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 24
    ph = rng.uniform(6.0, 7.5, n).round(3)
    glu = rng.uniform(5.0, 20.0, n).round(2)
    titer = (8 - 2 * (ph - 6.8) ** 2 + 0.05 * glu + rng.normal(0, 0.1, n)).round(4)
    return pd.DataFrame({"pH": ph, "Glucose_g_L": glu, "titer": titer})


def _stable(result: dict) -> str:
    """The result minus wall-clock fields, as a comparable string.

    Drops `timestamp` (seconds since epoch) and `model_diagnostics.fit_seconds`
    (how long the GP fit took): both are measured durations, not analysis output,
    so they differ by a millisecond between two runs of the same seeded sheet and
    would make a determinism comparison test the clock, not the math.
    """
    trimmed = {k: v for k, v in result.items() if k != "timestamp"}
    diag = trimmed.get("model_diagnostics")
    if isinstance(diag, dict) and "fit_seconds" in diag:
        trimmed["model_diagnostics"] = {**diag, "fit_seconds": None}
    return json.dumps(trimmed, sort_keys=True, default=str)


def test_two_concurrent_analyses_match_sequential_output():
    df = _sheet()
    sequential = [_stable(_analyze(df.copy(), target="titer")) for _ in range(2)]
    assert sequential[0] == sequential[1]

    outputs: list[str | None] = [None, None]
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def run(i: int) -> None:
        try:
            barrier.wait()
            outputs[i] = _stable(_analyze(df.copy(), target="titer"))
        except BaseException as exc:  # surfaced below, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert outputs[0] == sequential[0]
    assert outputs[1] == sequential[0]


def test_the_analysis_does_not_change_the_callers_global_rng():
    np.random.seed(7)
    expected = np.random.rand()
    np.random.seed(7)
    _analyze(_sheet(), target="titer")
    assert np.random.rand() == expected


def test_run_record_seed_is_the_seed_the_acquisition_used(monkeypatch):
    """The `seed` the result (and so the run record) reports must be the seed that
    `propose_batch` actually received, not a value that only describes it."""
    import kalos.core.optimize as optimize

    seen: list[object] = []
    real = optimize.propose_batch

    def spy(*args, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(optimize, "propose_batch", spy)
    out = _analyze(_sheet(), target="titer")
    assert seen == [ANALYZE_SEED]
    assert out["seed"] == ANALYZE_SEED
    assert analysis.ANALYZE_SEED == 1234
