"""A malformed constraint is a caller error, refused cleanly (review L1).

A NaN floor used to reach botorch and raise `RuntimeError: probability tensor
contains either inf, nan or element < 0` from inside the acquisition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kalos.portal.analysis import _analyze  # noqa: E402


def _sheet() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 24
    return pd.DataFrame({
        "a": np.linspace(0, 1, n),
        "b": rng.uniform(0, 1, n),
        "purity": rng.uniform(90, 99, n),
        "titer": rng.uniform(1, 2, n) + np.linspace(0, 1, n),
    })


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), float("-inf"), "95", True, None])
def test_a_non_finite_or_non_numeric_floor_is_a_value_error(floor):
    with pytest.raises(ValueError, match="constraint.floor"):
        _analyze(_sheet(), target="titer", constraint={"column": "purity", "floor": floor})


@pytest.mark.parametrize("column", ["", None, 3])
def test_a_bad_constraint_column_is_a_value_error(column):
    with pytest.raises(ValueError, match="constraint.column"):
        _analyze(_sheet(), target="titer", constraint={"column": column, "floor": 95.0})


def test_a_finite_floor_still_runs():
    out = _analyze(_sheet(), target="titer", constraint={"column": "purity", "floor": 95.0})
    assert out["constraint"]["column"] == "purity"
