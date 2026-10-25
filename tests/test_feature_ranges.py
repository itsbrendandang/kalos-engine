"""The analysis reports each continuous input's SEARCH BOX (`design.dims`) so clients
can flag proposals sitting on the edge of what has been tried."""
import numpy as np
import pandas as pd

from kalos.domains import ColumnRoles
from kalos.portal import app as portal


def _frame(n=24):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "pH": np.linspace(6.0, 7.5, n).round(3),
        "temp_C": rng.uniform(30, 38, n).round(1),
        "lipase_titer": rng.uniform(1, 5, n).round(3),
    })


def test_feature_ranges_match_the_observed_values_and_cover_every_proposal():
    df = _frame()
    out = portal._analyze(df, target="lipase_titer")
    fr = out["feature_ranges"]
    assert set(fr) == set(out["features"])
    # exact bounds for a column with known min/max (no absent-as-zero widening for pH)
    assert fr["pH"][0] <= 6.0 + 1e-6 and fr["pH"][1] >= 7.5 - 1e-6
    assert fr["pH"][0] > 0
    assert fr["temp_C"][0] <= df["temp_C"].min() + 1e-6 and fr["temp_C"][1] >= df["temp_C"].max() - 1e-6
    for p in out["proposals"]:
        for name, v in zip(out["proposal_features"], p["vals"]):
            lo, hi = fr[name]
            assert lo - 1e-6 <= v <= hi + 1e-6


def test_categorical_inputs_have_no_range():
    df = _frame()
    df["strain"] = ["A", "B", "C"] * 8
    out = portal._analyze(df, target="lipase_titer")
    assert "strain" not in out["feature_ranges"]


def test_feature_range_keys_are_aliased_when_anonymized():
    df = _frame()
    df = df.rename(columns={"temp_C": "sample_id"})
    roles = ColumnRoles(target="lipase_titer", features=("pH", "sample_id"))
    out = portal._analyze(df, roles=roles, anonymize=True)
    assert "sample_id" not in out["feature_ranges"]
    assert set(out["feature_ranges"]) == set(out["features"])
