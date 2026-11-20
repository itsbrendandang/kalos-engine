"""`leave_one_scale_out_report` never returns a report over zero rows
(VOYAGER_PORT WS3 gap (b)). Before the fix, a sheet without agitation/airflow
under the default physics feature set came back with `n_rows_used: 0`,
`n_scales: 0` and no error. Synthetic data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.scale.evaluation import leave_one_scale_out_report
from kalos.scale.transfer import ScaleFeatureConfig


def _method_ladder() -> pd.DataFrame:
    """5 scales, one run each (invented values)."""
    return pd.DataFrame({"scale_L": [1.5, 4.0, 20.0, 2500.0, 30000.0], "titer_g_L": [0.72, 0.70, 1.84, 0.69, 1.52]})




def test_loso_physics_config_without_agitation_or_airflow_raises_naming_them():
    df = _method_ladder()
    with pytest.raises(ValueError, match=r"missing column\(s\) \['agitation_rpm', 'airflow_L_per_min'\]"):
        leave_one_scale_out_report(df, "titer_g_L", [])


def test_loso_scale_only_config_uses_the_rows():
    df = _method_ladder()
    report = leave_one_scale_out_report(df, "titer_g_L", [], config=ScaleFeatureConfig(feature_set="scale_only"))
    assert report["n_rows_used"] == 5 and report["n_rows_dropped"] == 0 and report["n_scales"] == 5


def test_loso_missing_target_column_raises():
    with pytest.raises(ValueError, match="titer"):
        leave_one_scale_out_report(
            _method_ladder()[["scale_L"]], "titer", [], config=ScaleFeatureConfig(feature_set="scale_only")
        )


def test_loso_all_rows_non_finite_raises_naming_the_column():
    df = _method_ladder().assign(titer_g_L=np.nan)
    with pytest.raises(ValueError, match=r"all 5 rows.*\['titer_g_L'\]"):
        leave_one_scale_out_report(df, "titer_g_L", [], config=ScaleFeatureConfig(feature_set="scale_only"))
