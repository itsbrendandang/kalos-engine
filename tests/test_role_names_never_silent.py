"""Name heuristics never drop an input silently (real-backtest bug).

`pH_production`, a real 11-level pH setpoint of the production phase, was dropped
as an "output" because its name contains "product". Now: "production" is a phase,
not a product measurement; a pH / temperature / setpoint name is never read as an
output; and every column the outcome-name rule does exclude is reported with the
reason (provenance `reason`, finding `derived_output` with reason `outcome_name`).
Synthetic sheet only; no partner data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.domains import BIOPROCESS_PROFILE
from kalos.portal.analysis_parts.columns import is_outcome_name, name_inferred_outputs


@pytest.mark.parametrize("name,expected", [
    ("pH_production", False),
    ("temp_production_C", False),
    ("temperature_production_phase", False),
    ("production_feed_rate_mL_h", False),
    ("titer_setpoint", False),
    ("titer_g_L", True),
    ("product_conc_g_L", True),
    ("productivity_qP", True),
    ("Purity (%)", True),
])
def test_outcome_name_rule(name: str, expected: bool) -> None:
    assert is_outcome_name(name, BIOPROCESS_PROFILE) is expected


def _sheet(n: int = 44) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    ph = np.resize(np.round(np.linspace(6.4, 7.4, 11), 2), n)  # 11 levels, 4 runs each
    rng.shuffle(ph)
    temp = rng.choice([30.0, 32.0, 34.0], n)
    feed = rng.uniform(0.5, 2.0, n).round(3)
    titer = (2.0 - 3.0 * (ph - 6.9) ** 2 - 0.05 * (temp - 32) ** 2 + 0.3 * feed
             + rng.normal(0, 0.05, n)).round(3)
    return pd.DataFrame({
        "run_id": [f"R{i:03d}" for i in range(n)],
        "pH_production": ph,
        "temp_production_C": temp,
        "feed_mL_h": feed,
        # a real second output, and an outcome-named column with setpoint-like levels
        "product_conc_g_L": (titer * 0.8 + rng.normal(0, 0.1, n)).round(3),
        "score_level": rng.choice([1.0, 2.0, 3.0], n),
        "titer_g_L": titer,
    })


def test_name_inferred_outputs_are_listed_with_a_reason() -> None:
    df = _sheet()
    out = name_inferred_outputs(df, "titer_g_L", BIOPROCESS_PROFILE, None, already=set())
    assert set(out) == {"product_conc_g_L", "score_level"}
    assert out["product_conc_g_L"] == {"reason": "outcome_name", "looks_like_setting": False}
    assert out["score_level"]["looks_like_setting"] is True


def test_production_phase_setpoint_is_kept_and_every_exclusion_is_explained() -> None:
    from kalos.portal.analysis import _analyze

    out = _analyze(_sheet(), "titer_g_L")
    assert "pH_production" in out["features"] and "temp_production_C" in out["features"]
    prov = {p["name"]: p for p in out["provenance"]}
    assert prov["pH_production"]["status"] == "kept_feature"
    # every column that is not used says why, in plain words
    for row in out["provenance"]:
        if row["status"] != "kept_feature":
            assert row["reason"], row
    assert prov["product_conc_g_L"]["status"] == "dropped_output"
    assert "measured output" in prov["product_conc_g_L"]["reason"]
    assert "roles.features" in prov["product_conc_g_L"]["reason"]
    assert prov["run_id"]["status"] == "dropped_id" and prov["run_id"]["reason"]
    findings = {f["column"]: f for f in out["validation"]["findings"] if f["check"] == "derived_output"}
    assert findings["product_conc_g_L"]["severity"] == "info"
    assert findings["product_conc_g_L"]["detail"]["reason"] == "outcome_name"
    assert findings["score_level"]["severity"] == "warning"  # few repeated levels: check the role
    assert "check this role" in findings["score_level"]["message"]
    for p in out.get("proposals") or []:
        assert "pH_production" in p["recipe"]
