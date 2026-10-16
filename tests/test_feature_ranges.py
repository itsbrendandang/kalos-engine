"""The analysis reports each continuous input's search range so clients can flag
proposals sitting on the edge of what has been tried."""
import numpy as np
import pandas as pd

from kalos.portal import app as portal


def test_analysis_reports_feature_ranges_covering_every_proposal():
    rng = np.random.default_rng(0)
    n = 24
    df = pd.DataFrame({
        "pH": rng.uniform(6.0, 7.5, n).round(2),
        "temp_C": rng.uniform(30, 38, n).round(1),
        "lipase_titer": rng.uniform(1, 5, n).round(3),
    })
    out = portal._analyze(df)
    fr = out["feature_ranges"]
    assert set(fr) <= set(out["features"]) and "pH" in fr
    assert fr["pH"][0] >= 6.0 - 1e-6 and fr["pH"][1] <= 7.5 + 1e-6 and fr["pH"][0] < fr["pH"][1]
    for p in out["proposals"]:
        for name, v in zip(out["proposal_features"], p["vals"]):
            if name in fr:
                assert fr[name][0] - 1e-6 <= v <= fr[name][1] + 1e-6
