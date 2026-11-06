"""A coded / pseudonymised factor (Rudt ambr250 refolding: feed coded -1/0/+1)
is not a flow rate. The unit-based physical-bounds check skips it with an info
finding (`coded_column`), validation does not fail, and its design box is the
observed range, not [0, 1]. A column with a unit is still checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.validation import validate_frame
from kalos.validation.bounds import is_coded_column


def _rudt_like(feed_name: str = "feed_code") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 24
    feed = np.array([-1.0, 0.0, 1.0] * 8)
    feed[5] = -0.03
    ph = rng.choice([6.2, 7.0, 7.5, 8.0], n)
    temp = rng.choice([26.5, 30.0, 35.0], n)
    y = 0.001 + 0.0004 * feed - 0.0001 * (ph - 7) ** 2 + rng.normal(0, 5e-5, n)
    return pd.DataFrame({"pH": ph, "temperature_C": temp, feed_name: feed,
                         "refolded_concentration": y})


@pytest.mark.parametrize("name", ["feed_code", "feed_rate_coded", "feed"])
def test_coded_feed_passes_with_info_finding(name):
    report = validate_frame(_rudt_like(name), target="refolded_concentration")
    assert report.status != "fail"
    infos = [f for f in report.findings if f.column == name]
    assert [f.code for f in infos] == ["coded_column"]
    assert infos[0].severity == "info"


def test_feed_with_a_unit_is_still_checked():
    df = _rudt_like("feed")
    df = df.rename(columns={"feed": "feed_mL_h"})
    report = validate_frame(df, target="refolded_concentration")
    assert report.status == "fail"
    assert any(f.column == "feed_mL_h" and f.severity == "error" for f in report.findings)


def test_is_coded_column_rules():
    assert is_coded_column("feed_code", [0.0, 1.0])
    assert is_coded_column("pH level", [-1.0, 1.0])
    assert is_coded_column("feed", [-1.0, 1.0])  # weak hint, no unit, negative
    assert not is_coded_column("feed", [0.1, 0.5])  # no negatives: a normal feed
    assert not is_coded_column("feed_mL_h", [-1.0, 1.0])  # unit in name
    assert not is_coded_column("feed", [-1.0, 1.0], cell_units=True)
    assert not is_coded_column("temperature", [-999.0, 37.0])  # strong name
    assert not is_coded_column("Methanol", [-1.0, 1.0])  # concentration: strong


def test_coded_feed_keeps_observed_box_in_analysis():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_rudt_like(), target="refolded_concentration")
    assert out["validation"]["status"] != "fail"
    assert out["feature_ranges"]["feed_code"] == [-1.0, 1.0]
    assert not any(e["column"] == "feed_code" for e in out["design_box_exclusions"])
    codes = {f["code"] for f in out["validation"]["findings"] if f["column"] == "feed_code"}
    assert "coded_column" in codes
