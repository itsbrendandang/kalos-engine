"""The search box versus the tried range (first-time-user study, Finding 11):
kalos/portal/analysis_parts/search_box.py, ranges.widen_box, roles.allow_beyond."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.domains import ColumnRoles, InvalidAllowBeyond
from kalos.portal.analysis_parts.ranges import widen_box
from kalos.portal.analysis_parts.search_box import beyond_tried


def _rising_sheet(n: int = 13) -> pd.DataFrame:
    """Biomass rises with molasses all the way to the top of the tried range."""
    rng = np.random.default_rng(1)
    mol = np.tile([80.0, 100.0, 120.0], n)[:n]
    amm = np.tile([5.0, 12.5, 20.0, 12.5], n)[:n]
    bio = (0.15 * mol - 0.02 * (amm - 12.5) ** 2 + rng.normal(0, 0.2, n)).round(3)
    return pd.DataFrame({"molasses_g_L": mol, "nitrogen_g_L": amm, "biomass_g_L": bio})


def test_parse_allow_beyond():
    assert ColumnRoles.from_dict({"target": "y"}).allow_beyond == ()
    assert ColumnRoles.from_dict({"target": "y", "allow_beyond": 0.1}).allow_beyond == (("*", 0.1),)
    r = ColumnRoles.from_dict({"target": "y", "allow_beyond": {"pH": 0, "*": 0.2}})
    assert dict(r.allow_beyond) == {"pH": 0.0, "*": 0.2}
    for bad in (-0.1, 1.5, "10%", True, [0.1], {"pH": "x"}):
        with pytest.raises(InvalidAllowBeyond):
            ColumnRoles.from_dict({"target": "y", "allow_beyond": bad})


def test_widen_box_respects_physical_bounds():
    assert widen_box("molasses_g_L", 80, 120, 0.1) == pytest.approx((76, 124))
    assert widen_box("pH", 6.0, 13.8, 0.1) == pytest.approx((5.22, 14.0))  # pH <= 14
    assert widen_box("feed_x", 0.0, 10.0, 0.2) == pytest.approx((0.0, 12.0))  # never below 0
    assert widen_box("coded", -1.0, 1.0, 0.1, coded=True) == pytest.approx((-1.2, 1.2))
    assert widen_box("same", 5.0, 5.0, 0.5) == (5.0, 5.0)
    assert widen_box("molasses_g_L", 80, 120, 0.0) == (80, 120)


def test_beyond_tried_flags():
    tried = {"a": [0.0, 10.0], "b": [1.0, 2.0]}
    flags = beyond_tried({"a": 11.0, "b": 1.5, "cat": "x"}, tried)
    assert flags == [{"feature": "a", "side": "upper", "tried_limit": 10.0, "proposed": 11.0,
                      "by": 1.0, "fraction": 0.1}]
    assert beyond_tried({"a": 5.0, "b": 2.0}, tried) == []


def test_default_box_is_the_tried_range_and_says_so():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_rising_sheet(), "biomass_g_L")
    assert out["allow_beyond"] == {"default": 0.0, "per_input": {}, "applied": {}}
    assert out["tried_ranges"]["molasses_g_L"] == [80.0, 120.0]
    assert out["feature_ranges"]["molasses_g_L"] == [80.0, 120.0]
    assert all(p["beyond_tried_range"] == [] for p in out["proposals"])
    box = [f for f in out["validation"]["findings"] if f["code"] == "search_box"]
    assert len(box) == 1 and "only searches inside the range you have tried" in box[0]["message"]
    assert "allow_beyond" in box[0]["message"]


def test_allow_beyond_widens_and_flags_proposals():
    pytest.importorskip("botorch")
    from kalos.portal.analysis import _analyze

    roles = ColumnRoles.from_dict({"target": "biomass_g_L", "allow_beyond": {"molasses_g_L": 0.25}})
    out = _analyze(_rising_sheet(), roles=roles)
    json.dumps(out["allow_beyond"], allow_nan=False)
    assert out["allow_beyond"]["applied"] == {"molasses_g_L": 0.25}
    assert out["feature_ranges"]["molasses_g_L"] == [70.0, 130.0]
    assert out["tried_ranges"]["molasses_g_L"] == [80.0, 120.0]
    assert out["feature_ranges"]["nitrogen_g_L"] == out["tried_ranges"]["nitrogen_g_L"]
    # Biomass rises with molasses to the top of the tried range: the best bet goes past it.
    best = out["proposals"][0]
    assert best["pick"] == "best_bet" and best["recipe"]["molasses_g_L"] > 120
    assert best["beyond_tried_range"][0]["feature"] == "molasses_g_L"
    assert best["beyond_tried_range"][0]["side"] == "upper"
    box = next(f for f in out["validation"]["findings"] if f["code"] == "search_box")
    assert "25% past each end" in box["message"] and "flagged" in box["message"]


def test_rounding_at_the_edge_is_not_beyond():
    """Planted (security review nit): a shown value 0.0004 past the tried edge
    is 3-decimal rounding, not a recipe beyond the tried range."""
    tried = {"a": [0.0, 1.0]}
    assert beyond_tried({"a": 1.0004}, tried) == []
    assert beyond_tried({"a": -0.0004}, tried) == []
    assert beyond_tried({"a": 1.002}, tried)[0]["side"] == "upper"
