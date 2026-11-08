"""Tests for `kalos.normalize.units` helpers that are not covered
by `test_normalize_core.py`: `base_unit_label` and `is_known_unit`.

Both are pure, deterministic functions with clear contracts:
- `base_unit_label`: maps a canonical suffix to a human-readable name,
  degrading to the raw suffix for unknowns.
- `is_known_unit`: tells a caller whether a unit token is in the registry
  (and therefore safe to auto-convert), excluding bare numbers (None).
"""
from __future__ import annotations

import pytest

from kalos.normalize.units import base_unit_label, is_known_unit

# --- is_known_unit ---------------------------------------------------------- #


class TestIsKnownUnit:
    """Verify `is_known_unit` recognises registered tokens and rejects
    unknowns, bare numbers (None), and common trap tokens like "L"."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            # Temperature
            ("C", True),
            ("c", True),
            ("F", True),
            ("K", True),
            ("celsius", True),
            ("kelvin", True),
            # Percent
            ("%", True),
            ("pct", True),
            # Concentration
            ("g/L", True),
            ("g/l", True),
            ("mg/mL", True),
            ("mg/ml", True),
            # Flow rate
            ("mL/h", True),
            ("ml/h", True),
            ("uL/h", True),
            ("µL/h", True),
            ("L/h", True),
            ("ml/min", True),
            ("mL/min", True),
            ("l/min", True),
            # Time
            ("h", True),
            ("hr", True),
            ("min", True),
            # Agitation / Pressure / Mass
            ("rpm", True),
            ("bar", True),
            ("g", True),
            ("kg", True),
            # Dimensionless (known)
            ("pH", True),
            ("ph", True),
            ("od", True),
            ("OD", True),
        ],
    )
    def test_known_units(self, token: str, expected: bool) -> None:
        assert is_known_unit(token) is expected

    @pytest.mark.parametrize(
        "token, expected",
        [
            # None = bare number, no unit to convert
            (None, False),
            # Volume "L" and "mL" deliberately NOT registered (vessel-size labels)
            ("L", False),
            ("mL", False),
            ("l", False),
            # Completely bogus
            ("furlongs", False),
            ("foobar", False),
            ("m", False),
            ("mm", False),
            ("s", False),
            ("W", False),
        ],
    )
    def test_unknown_or_excluded_units(self, token: str, expected: bool) -> None:
        assert is_known_unit(token) is expected


# --- base_unit_label -------------------------------------------------------- #


class TestBaseUnitLabel:
    """Verify `base_unit_label` maps canonical suffixes to human-readable
    names and degrades gracefully for unknown suffixes."""

    @pytest.mark.parametrize(
        "suffix, expected",
        [
            ("_c", "Celsius"),
            ("_pct", "percent"),
            ("_g_l", "g/L"),
            ("_ml_h", "mL/h"),
            ("_h", "hours"),
            ("_rpm", "rpm"),
            ("_bar", "bar"),
            ("_g", "grams"),
            ("", "dimensionless"),
        ],
    )
    def test_known_suffixes(self, suffix: str, expected: str) -> None:
        assert base_unit_label(suffix) == expected

    def test_unknown_suffix_degrades_to_raw(self) -> None:
        # Unknown suffixes return the raw string, not a crash
        assert base_unit_label("_unknown_foo") == "_unknown_foo"
        assert base_unit_label("foo") == "foo"
