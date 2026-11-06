"""The DataHow in-silico runs as an endpoint table (kalos/bench/insilico_hybrid.py).
The data is not in the repository; the real-file test runs only when
KALOS_DATAHOW_DATA points at it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.bench.insilico_hybrid import (
    INPUTS,
    TARGET,
    data_dir,
    endpoint_table,
    load_runs,
    varying_inputs,
)


def _long(n_runs: int = 3, steps: int = 4) -> pd.DataFrame:
    rows = []
    for r in range(n_runs):
        for t in range(steps):
            rows.append({
                "run": r, "timesteps": t, "W:temp": 36.0 if t < 2 else 33.0 + r,
                "W:pH": 7.0 if t < 2 else 6.9, "Z:Stir": 200.0, "Z:DO": 40.0 + r,
                "X:Glc": 5.0 + r if t == 0 else 3.0, "X:Gln": 4.0, "X:Titer": 100.0 * t + r,
                "BolusFeed_Glc:Glc": 0.0 if t < 2 else 5.0 + r, "BolusFeed_Gln:Gln": 1.0,
            })
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0)  # order must not matter


def test_endpoint_table_one_row_per_run():
    out = endpoint_table(_long())
    assert list(out["run"]) == [0, 1, 2]
    r1 = out.iloc[1]
    assert r1["temp_start"] == 36.0 and r1["temp_end"] == 34.0
    assert r1["Glc_start"] == 6.0 and r1["feed_Glc_bolus"] == 6.0
    assert r1[TARGET] == 301.0  # the last step
    assert set(INPUTS) <= set(out.columns)
    assert "stir" not in varying_inputs(out) and "DO" in varying_inputs(out)


def test_missing_data_says_where_to_put_it(tmp_path, monkeypatch):
    monkeypatch.delenv("KALOS_DATAHOW_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="KALOS_DATAHOW_DATA"):
        load_runs()
    with pytest.raises(FileNotFoundError, match=r"INSILICO_HYBRID\.md"):
        load_runs(tmp_path)


@pytest.mark.skipif(data_dir() is None, reason="KALOS_DATAHOW_DATA not set")
def test_real_files_give_120_runs():
    df = load_runs()
    assert len(df) == 120 and (df["split"] == "train").sum() == 20
    assert np.isfinite(df[TARGET]).all()
