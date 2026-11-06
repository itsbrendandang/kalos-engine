"""Near-constant inputs (CHO 20829178 shape): an inline pH of 6.98-7.03 and a
temperature of 36.98-37.01 C are one setpoint read with sensor noise. They are
fixed at their median, not searched; proposals carry the fixed value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis_parts.near_constant import near_constant_inputs


def _cho_like(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    ph = np.round(rng.uniform(6.98, 7.03, n), 3)
    ph[0], ph[1] = 6.98, 7.03
    temp = np.round(rng.uniform(36.98, 37.01, n), 2)
    agitation = rng.choice([100.0, 150.0, 200.0], n)
    o2 = rng.uniform(21.0, 40.0, n)
    ivcd = 1500 + 3 * agitation + 20 * o2 + rng.normal(0, 50, n)
    return pd.DataFrame({
        "pH_inline": ph, "temperature_C": temp, "agitation_rpm": agitation,
        "O2_inlet": o2, "IVCD": ivcd,
    })


def test_rule_fixes_sensor_noise_spread_only():
    df = _cho_like()
    fixed = near_constant_inputs(df, ["pH_inline", "temperature_C", "agitation_rpm", "O2_inlet"])
    assert set(fixed) == {"pH_inline", "temperature_C"}
    assert fixed["pH_inline"]["value"] == pytest.approx(float(np.median(df["pH_inline"])))


def test_rule_is_conservative():
    df = pd.DataFrame({
        "pH": [6.9, 7.0, 7.1, 7.0],          # 0.2 units: a real pH range
        "temp": [36.8, 37.0, 37.2, 37.0],    # 0.4 C: searched
        "glucose": [10.0, 10.001, 10.002, 10.0],  # unknown precision: 0.02% -> fixed
        "feed": [1.0, 1.005, 1.0, 1.002],    # 0.5% -> searched (fallback is 0.1%)
        "zero": [0.0, 0.0, 0.0, 0.001],      # median 0: never fixed
    })
    assert set(near_constant_inputs(df, list(df.columns))) == {"glucose"}


def test_cho_like_analysis_fixes_near_constant_inputs():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_cho_like(), target="IVCD")
    assert "pH_inline" not in out["features"] and "temperature_C" not in out["features"]
    assert set(out["fixed_inputs"]) == {"pH_inline", "temperature_C"}
    assert "pH_inline" not in out["feature_ranges"]
    for p in out["proposals"]:
        assert p["recipe"]["pH_inline"] == out["fixed_inputs"]["pH_inline"]
        assert p["recipe"]["temperature_C"] == out["fixed_inputs"]["temperature_C"]
    prov = {r["name"]: r for r in out["provenance"]}
    assert prov["pH_inline"]["status"] == "fixed_near_constant"
    assert prov["pH_inline"]["reason"] == "near_constant"
    assert prov["pH_inline"]["role_inferred"] == "fixed_input"
    found = [f for f in out["validation"]["findings"] if f["code"] == "near_constant_input"]
    assert {f["column"] for f in found} == {"pH_inline", "temperature_C"}
