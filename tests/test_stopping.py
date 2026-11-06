"""Advisory stopping signal (QUEUE round 3 item 7): kalos/core/stopping.py and
the analysis `stopping` block. Advice only; never changes proposals."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")

from kalos.core.stopping import STOP_P_THRESHOLD  # noqa: E402
from kalos.portal.analysis import _analyze  # noqa: E402


def _sheet(kind: str, *, reps: int = 3, seed: int = 0, n_rec: int = 12, flip: bool = False) -> pd.DataFrame:
    """`flat`: every recipe has the same mean (noise sd 0.4). `steep`: titer rises
    fast toward a corner of the box that no run has tried (noise sd 0.1)."""
    rng = np.random.default_rng(seed)
    if kind == "steep":
        a = np.r_[rng.uniform(0, 4, n_rec - 2), 0.0, 10.0]
        b = np.r_[rng.uniform(0, 4, n_rec - 2), 10.0, 0.0]
        mu, sd = 3 * a + 3 * b, 0.1
    else:
        a, b = rng.uniform(0, 10, n_rec), rng.uniform(0, 10, n_rec)
        mu, sd = np.full(n_rec, 5.0), 0.4
    rows = []
    for i in range(n_rec):
        for _ in range(reps):
            v = mu[i] + rng.normal(0, sd)
            rows.append({"a_g_L": round(a[i], 3), "b_g_L": round(b[i], 3),
                         "titer_g_L": round(-v if flip else v, 4)})
    return pd.DataFrame(rows)


def _check_shape(st: dict) -> None:
    json.dumps(st, allow_nan=False)
    assert st["advisory"] is True
    assert st["plain"].endswith("This is advice, not a stop.")
    for word in ("posterior", "regret", "chi-square"):
        assert word not in st["plain"]


def test_flat_surface_with_replicates_says_gains_within_noise():
    st = _analyze(_sheet("flat"), "titer_g_L")["stopping"]
    _check_shape(st)
    assert st["state"] == "gains_within_noise"
    assert st["p_gain_beyond_noise"] < STOP_P_THRESHOLD
    assert st["noise_sd"] == pytest.approx(0.4, abs=0.15)
    assert st["n_samples"] > 0 and "run-to-run spread" in st["plain"]


def test_steep_surface_far_from_optimum_says_keep_going():
    st = _analyze(_sheet("steep"), "titer_g_L")["stopping"]
    _check_shape(st)
    assert st["state"] == "keep_going"
    assert st["p_gain_beyond_noise"] >= STOP_P_THRESHOLD
    assert st["max_expected_gain"] > st["noise_sd"]


def test_minimize_reads_gains_in_the_better_direction():
    out = _analyze(_sheet("steep", flip=True), "titer_g_L", direction="minimize")
    st = out["stopping"]
    assert out["direction"] == "minimize"
    assert st["state"] == "keep_going" and st["max_expected_gain"] > 0


def test_no_replicates_is_unknown_and_says_why():
    out = _analyze(_sheet("steep", reps=1, n_rec=16), "titer_g_L")
    st = out["stopping"]
    _check_shape(st)
    assert st["state"] == "unknown"
    assert st["p_gain_beyond_noise"] is None and st["noise_sd"] is None
    assert "repeated runs" in st["plain"]


def test_stopping_block_does_not_change_proposals(monkeypatch):
    import kalos.portal.analysis as analysis

    sheet = _sheet("steep")
    with_block = _analyze(sheet, "titer_g_L")
    monkeypatch.setattr(analysis, "stopping_advice", lambda *a, **k: {"state": "stub"})
    without = _analyze(sheet, "titer_g_L")
    assert without["stopping"] == {"state": "stub"}
    assert with_block["proposals"] == without["proposals"]
    assert with_block["model_optimum"] == without["model_optimum"]


def test_bench_simulation_smoke():
    from kalos.bench.stopping_sim import ROUNDS, run

    r = run(seeds=1)
    assert r["n_snapshots"] == ROUNDS
    assert {row["state"] for row in r["rows"]} <= {"keep_going", "gains_within_noise", "unknown"}
    json.dumps(r, allow_nan=False)
