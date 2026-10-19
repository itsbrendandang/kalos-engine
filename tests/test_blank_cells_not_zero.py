"""A blank cell is UNKNOWN, not 0.0 (review C1, critical).

The old code zero-filled every blank feature cell and then treated 0.0 as a
valid value, so the design box widened to include 0 and proposals landed at
pH 0.0. Repro: 24 runs, pH uniform 6.0-7.5, three blank pH cells gave
proposals with pH [0.295, 0.723, 0.534, 1.771, 0.0].

Now only sparse-declared and concentration-type columns read a blank as 0
(absent). Any other blank drops its run through the row accounting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kalos.domains import ColumnRoles  # noqa: E402
from kalos.portal.analysis import (  # noqa: E402
    _absent_is_zero_columns,
    _analyze,
    _physical_range,
)

N = 24


def _ph_sheet() -> pd.DataFrame:
    ph = np.linspace(6.0, 7.5, N)
    temp = np.tile([30.0, 34.0, 37.0], N // 3)
    rng = np.random.default_rng(0)
    titer = 10 - 3 * (ph - 6.8) ** 2 + 0.1 * temp + rng.normal(0, 0.05, N)
    df = pd.DataFrame({"pH": ph, "Temperature": temp,
                       "Glucose_g_L": np.linspace(5, 20, N), "titer": titer})
    df.loc[[2, 9, 17], "pH"] = np.nan
    return df


def test_blank_ph_never_yields_a_proposal_outside_the_observed_range():
    df = _ph_sheet()
    observed = df["pH"].dropna()
    out = _analyze(df, target="titer")
    proposed = [p["recipe"]["pH"] for p in out["proposals"]]
    assert proposed, "expected a proposed batch"
    assert all(observed.min() <= v <= observed.max() for v in proposed), proposed
    assert all(v > 0.5 for v in proposed), proposed  # the old defect proposed 0.0


def test_blank_ph_run_is_dropped_and_reported_as_missing():
    out = _analyze(_ph_sheet(), target="titer")
    assert out["n"] == N - 3
    assert out["n_dropped_incomplete"] == 3
    assert out["data_drops"]["cells_by_column"]["pH"] == {"missing": 3}
    assert out["data_drops"]["blank_cells_as_absent"] == {}


def test_only_sparse_and_concentration_columns_read_blank_as_zero():
    cols = ["pH", "Temperature", "Glucose_g_L", "Methanol"]
    assert _absent_is_zero_columns(cols, None) == {"Glucose_g_L", "Methanol"}
    roles = ColumnRoles.from_dict({"target": "titer", "sparse": ["Temperature"]})
    assert _absent_is_zero_columns(cols, roles) == {"Temperature", "Glucose_g_L", "Methanol"}


def test_a_declared_sparse_column_keeps_the_absent_zero_convention():
    df = _ph_sheet()
    df["Additive"] = np.linspace(0.1, 2.0, N)
    df.loc[5, "Additive"] = None
    roles = ColumnRoles.from_dict({"target": "titer", "sparse": ["Additive"]})
    out = _analyze(df, target="titer", roles=roles)
    assert out["n"] == N - 3  # only the three blank pH runs are dropped
    assert out["data_drops"]["blank_cells_as_absent"] == {"Additive": 1}


def test_box_from_observed_cells_only_for_non_sparse_columns():
    observed = np.array([6.0, 6.5, 7.5])
    assert _physical_range("pH", observed) == (6.0, 7.5, 0)


def test_absent_cells_extend_a_sparse_box_to_include_zero():
    observed = np.array([2.0, 3.0])
    lo, hi, n = _physical_range("Glucose_g_L", observed, absent=2)
    assert (lo, hi, n) == (0.0, 3.0, 0)


def test_an_in_flight_recipe_with_a_blank_ph_is_dropped_and_reported():
    """Review H2: an awaiting run with no pH is not zero-filled into the design.
    It is dropped and counted in `n_pending_dropped`."""
    out = _analyze(
        _ph_sheet(),
        target="titer",
        pending=[
            {"pH": 6.5, "Glucose_g_L": 10.0, "Temperature": 34.0},
            {"pH": None, "Glucose_g_L": 10.0, "Temperature": 34.0},
        ],
    )
    assert out["n_pending_considered"] == 1
    assert out["n_pending_dropped"] == 1
