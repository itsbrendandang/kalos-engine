"""BRE-117: physically impossible values, with bounds from a unit-aware table.

The header's unit decides what a column measures (`DO_pct` is a percentage,
`antifoam_mgL` is mg/L, `glutamine_mM` a concentration); the name is the
fallback. `\\bdo\\b` never matched `DO_pct`, so DO = 140 % passed silently.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kalos.validation.bounds import column_bounds, infer_dimension
from kalos.validation.checks import check_physical_bounds


def _findings(col: str, values: list) -> list:
    return check_physical_bounds(pd.DataFrame({col: values}))


def _severities(col: str, values: list) -> list[str]:
    return [f.severity for f in _findings(col, values)]


def test_do_pct_above_100_or_below_0_is_impossible() -> None:
    assert infer_dimension("DO_pct") == "percent"
    (f,) = _findings("DO_pct", [40.0, 55.0, 140.0, 60.0])
    assert f.severity == "error" and f.rows == (2,)
    assert "140" in f.message and "row 2" in f.message
    assert "oxygen-enriched" in f.message  # says when > 100 % could be real
    assert _severities("DO_pct", [40.0, -3.0]) == ["error"]
    assert _findings("DO_pct", [20.0, 80.0, 100.0, 0.0]) == []


@pytest.mark.parametrize("col", ["pH", "ph_setpoint", "culture pH (units)"])
def test_ph_outside_0_14_is_impossible(col: str) -> None:
    assert _severities(col, [7.0, 15.2]) == ["error"]
    assert _severities(col, [7.0, -0.5]) == ["error"]
    assert _findings(col, [3.5, 7.2]) == []  # acidic fungal / yeast processes are usual


@pytest.mark.parametrize("col", ["titer_g_L", "glutamine_mM", "antifoam_mgL", "VCD (10^6 cells/mL)",
                                 "lactate (mmol/L)", "feed_rate_mLh", "agitation_rpm", "inoculum_vol_ml"])
def test_negative_amounts_concentrations_and_flows_are_impossible(col: str) -> None:
    assert _severities(col, [1.0, 2.0, -0.4]) == ["error"]


def test_bounds_follow_the_header_unit() -> None:
    # 600 mg/L antifoam is ordinary; the g/L typical ceiling (500) must not apply.
    assert _findings("antifoam_mgL", [20.0, 600.0]) == []
    assert _severities("titer_g_L", [1.0, 600.0]) == ["warning"]
    # Gas flow in L/min at plant scale is not a feed pump rate.
    assert _findings("airflow_L_per_min", [0.05, 250.0]) == []
    cb = column_bounds("antifoam_mgL")
    assert cb is not None and cb.unit == "mg/L" and "header" in cb.source


def test_relative_percentages_are_not_bounded_at_100() -> None:
    for col in ("recovery (%)", "yield_pct", "titer_change_pct", "inhibition (%)"):
        assert _findings(col, [50.0, 120.0, -5.0]) == [], col


def test_shares_of_a_whole_are_bounded_at_100() -> None:
    for col in ("viability (%)", "purity_pct", "aggregate (%)"):
        assert _severities(col, [95.0, 101.0]) == ["error"], col


def test_other_percentages_warn_above_100_and_refuse_negatives() -> None:
    assert _severities("feed_pct", [5.0, 150.0]) == ["warning"]
    assert _severities("feed_pct", [5.0, -1.0]) == ["error"]


def test_cells_with_their_own_unit_are_converted_first() -> None:
    # 98.6 F is 37 C: fine. A bare 98.6 in a Celsius column is not.
    assert _findings("Temp (C)", ["37.0", "98.6 F"]) == []
    assert _severities("Temp (C)", [37.0, 98.6]) == ["warning"]


def test_unknown_columns_get_no_bounds() -> None:
    assert column_bounds("scale_L") is None
    assert column_bounds("medium_base") is None
    assert column_bounds("phosphate_g_L") is not None
    assert column_bounds("phosphate_g_L").dimension == "concentration"  # type: ignore[union-attr]
