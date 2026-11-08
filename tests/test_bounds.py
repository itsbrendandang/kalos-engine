"""Tests for `kalos.validation.bounds`: the physical-range registry and the
header -> dimension inference used by `checks.check_physical_bounds`.
"""
from __future__ import annotations

import math

import pytest

from kalos.validation.bounds import DIMENSION_BOUNDS, DimensionBounds, infer_dimension

# --- registry invariants ---------------------------------------------------- #


def test_registry_keys_match_dimension_field() -> None:
    for key, bounds in DIMENSION_BOUNDS.items():
        assert isinstance(bounds, DimensionBounds)
        assert bounds.dimension == key


def test_typical_range_nested_inside_hard_range() -> None:
    for bounds in DIMENSION_BOUNDS.values():
        assert bounds.hard_lo <= bounds.typical_lo <= bounds.typical_hi <= bounds.hard_hi


def test_ph_and_percent_are_bounded_physical_scales() -> None:
    ph = DIMENSION_BOUNDS["ph"]
    assert (ph.hard_lo, ph.hard_hi) == (0.0, 14.0)
    assert (ph.typical_lo, ph.typical_hi) == (4.0, 9.0)
    assert ph.unit == ""

    pct = DIMENSION_BOUNDS["percent"]
    assert (pct.hard_lo, pct.hard_hi) == (0.0, 100.0)
    # percent deliberately has no inner band: it only ever produces ERRORs.
    assert (pct.typical_lo, pct.typical_hi) == (pct.hard_lo, pct.hard_hi)


def test_unbounded_above_dimensions_have_nonneg_floor_and_base_units() -> None:
    for dim, unit in (("concentration", "_g_l"), ("flow_rate", "_ml_h"), ("time_h", "_h")):
        bounds = DIMENSION_BOUNDS[dim]
        assert bounds.hard_lo == 0.0
        assert math.isinf(bounds.hard_hi)
        assert bounds.unit == unit
    assert DIMENSION_BOUNDS["concentration"].typical_hi == 500.0
    assert DIMENSION_BOUNDS["flow_rate"].typical_hi == 10_000.0
    assert DIMENSION_BOUNDS["time_h"].typical_hi == 720.0


def test_dimension_bounds_is_frozen() -> None:
    with pytest.raises(AttributeError):
        DIMENSION_BOUNDS["ph"].hard_hi = 15.0  # type: ignore[misc]


# --- header -> dimension inference ----------------------------------------- #


@pytest.mark.parametrize(
    "header, expected",
    [
        ("pH", "ph"),
        ("culture pH (units)", "ph"),
        ("p.H", "ph"),
        ("Temp (C)", "temperature_c"),
        ("OD600", "od"),
        ("DO%", "percent"),
        ("Viability", "percent"),
        ("glucose_g_L", "concentration"),
        ("feed_rate_mL_h", "flow_rate"),
        ("Time (hours)", "time_h"),
        ("VCD", "cell_density"),
        ("Osmolality (mOsm/kg)", "osmolality"),
        ("pCO2 (mmHg)", "partial_pressure"),
        ("culture_day", "nonneg_count"),
    ],
)
def test_infer_dimension_matches_real_world_headers(header: str, expected: str) -> None:
    assert infer_dimension(header) == expected
    assert expected in DIMENSION_BOUNDS


def test_ph_token_must_stand_alone() -> None:
    # Regression guard: "phosphate"/"morphology" used to be swallowed by a
    # bare "ph" substring match and handed pH's 0-14 hard range.
    assert infer_dimension("phosphate_g_L") == "concentration"
    assert infer_dimension("morphology") is None


def test_concentration_wins_over_flow_rate_on_ambiguous_header() -> None:
    # Documented ordering: analyte vocabulary is checked before "feed|flow|rate".
    assert infer_dimension("sodium hydroxide feed rate") == "concentration"


def test_unknown_or_empty_header_yields_none() -> None:
    assert infer_dimension("") is None
    assert infer_dimension("well_id") is None
