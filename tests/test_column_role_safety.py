"""Column-role safety (kalos/portal/analysis_parts/roles.py).

A raw published sheet (Pichia HSA shape: a blank-header row index first, the
target, then OD readings taken during the run) must not train on the row index
or the OD columns by default. A concentration input before the given target is
kept. Inputs that follow the target with ordinary names stay inputs.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from kalos.domains import BIOPROCESS_PROFILE, ColumnRoles
from kalos.portal.analysis_parts.columns import _resolve_columns
from kalos.portal.analysis_parts.roles import column_role_safety
from kalos.portal.uploads import _parse_upload, numeric_columns


def _pichia_like_csv(n: int = 40) -> bytes:
    rng = np.random.default_rng(7)
    source = rng.integers(0, 5, n).astype(float)
    conc = rng.uniform(0, 50, n)
    gly = rng.uniform(0, 10, n)
    met = rng.uniform(0, 7, n)
    sp = rng.uniform(1, 6, n)
    titer = 10 + 0.4 * conc + 2 * met - 0.5 * gly + rng.normal(0, 2, n)
    od_og = 0.05 * titer + rng.normal(0, 0.1, n)
    od_prod = 0.3 * titer + rng.normal(0, 0.2, n)
    df = pd.DataFrame({
        "": np.arange(n), "CarbonSource": source, "Carbon_conc": conc, "Gly": gly,
        "Met": met, "SP": sp, "Titer": titer, "OD_OG": od_og, "OD_Prod": od_prod,
    })
    return df.to_csv(index=False).encode()


def _safety(df: pd.DataFrame, target: str | None, roles: ColumnRoles | None = None):
    num = numeric_columns(df)
    tgt, cont, cats, _g, declared = _resolve_columns(df, num, target, roles, BIOPROCESS_PROFILE)
    header = df.attrs.get("kalos_header") or {}
    return column_role_safety(
        df, num, tgt, cont, cats, declared=declared, roles=roles, profile=BIOPROCESS_PROFILE,
        headerless=list(header.get("named_by_position") or []),
        target_given=(roles is not None or target is not None),
    )


def test_pichia_like_default_roles_exclude_row_index_and_post_target_od():
    df = _parse_upload(_pichia_like_csv())
    out = _safety(df, "Titer")
    assert "column 1" not in out.cont_feats
    assert out.overrides["column 1"] == ("dropped_id", "headerless")
    assert out.overrides["OD_OG"] == ("dropped_output", "after_target_in_sheet")
    assert out.overrides["OD_Prod"] == ("dropped_output", "after_target_in_sheet")
    # Carbon_conc sits before the given target: kept as an input.
    assert "Carbon_conc" in out.cont_feats
    assert out.reasons["Carbon_conc"] == "concentration_before_target"
    assert {f.code for f in out.findings} >= {
        "row_number_column", "after_target_in_sheet", "concentration_before_target",
    }


def test_carbon_conc_still_dropped_when_target_is_not_given():
    df = _parse_upload(_pichia_like_csv())
    out = _safety(df, None)
    assert "Carbon_conc" not in out.cont_feats


def test_named_consecutive_row_number_is_excluded_but_unsorted_integer_input_is_not():
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame({
        "No": np.arange(1, n + 1), "dose": rng.permutation(np.arange(1, n + 1)),
        "pH": rng.uniform(6, 7.5, n), "titer": rng.uniform(1, 5, n),
    })
    out = _safety(df, "titer")
    assert out.overrides["No"] == ("dropped_id", "monotone_row_number")
    assert "dose" in out.cont_feats and "pH" in out.cont_feats


def test_inputs_after_the_target_with_plain_names_stay_inputs():
    rng = np.random.default_rng(1)
    n = 12
    df = pd.DataFrame({
        "titer": rng.uniform(1, 5, n), "pH": rng.uniform(6, 7.5, n),
        "temperature": rng.uniform(30, 37, n),
    })
    out = _safety(df, "titer")
    assert out.cont_feats == ["pH", "temperature"]
    assert not out.overrides


def test_declared_features_are_never_excluded_but_are_flagged():
    df = _parse_upload(_pichia_like_csv())
    roles = ColumnRoles(target="Titer", features=["column 1", "Gly", "OD_Prod"])
    out = _safety(df, None, roles)
    assert out.cont_feats == ["column 1", "Gly", "OD_Prod"]
    flagged = {f.column for f in out.findings if f.code == "feature_looks_outcome_like"}
    assert "OD_Prod" in flagged


def test_pichia_like_analysis_does_not_use_leaky_columns():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    df = _parse_upload(_pichia_like_csv())
    out = _analyze(df, target="Titer")
    assert "column 1" not in out["features"]
    assert "OD_OG" not in out["features"] and "OD_Prod" not in out["features"]
    assert "Carbon_conc" in out["features"]
    for p in out["proposals"]:
        assert not ({"column 1", "OD_OG", "OD_Prod"} & set(p))
    prov = {r["name"]: r for r in out["provenance"]}
    assert len(prov) == 9  # every column of the sheet is listed
    assert prov["column 1"]["status"] == "dropped_id"
    assert prov["column 1"]["reason"] == "headerless"
    assert prov["column 1"]["role_inferred"] == "identifier"
    assert prov["OD_Prod"]["status"] == "dropped_output"
    assert prov["OD_Prod"]["reason"] == "after_target_in_sheet"
    assert prov["Carbon_conc"]["status"] == "kept_feature"
    assert prov["Titer"]["reason"] == "requested_target"
    assert all(r["reason"] for r in prov.values())
    codes = {f["code"] for f in out["validation"]["findings"]}
    assert "row_number_column" in codes and "after_target_in_sheet" in codes


def test_csv_round_trip_helper_is_a_real_csv():
    # guard: the fixture really has a blank first header
    assert io.BytesIO(_pichia_like_csv()).readline().startswith(b",CarbonSource")


@pytest.mark.parametrize("name, values", [
    ("Excel row", np.arange(2, 14)),                                  # 2..n+1
    ("Line", np.arange(101, 113)),                                    # any start
    ("Pos", np.arange(12, 0, -1)),                                    # descending
    ("Entry", np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                         18, 19, 20, 21])),                           # one deleted row
    ("No", np.arange(0, 12)),                                         # 0..n-1 (as before)
])
def test_row_number_any_start_descending_or_small_gaps_is_excluded(name, values):
    rng = np.random.default_rng(3)
    n = len(values)
    df = pd.DataFrame({name: values, "pH": rng.uniform(6, 7.5, n), "titer": rng.uniform(1, 5, n)})
    out = _safety(df, "titer")
    assert out.overrides[name] == ("dropped_id", "monotone_row_number")
    assert name not in out.cont_feats and "pH" in out.cont_feats


@pytest.mark.parametrize("values", [
    np.arange(10, 130, 10),                       # a dose ladder: sorted, unique, sparse
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]),  # big gaps
    np.array([3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # not monotone
])
def test_sorted_sparse_or_unsorted_integer_inputs_stay_inputs(values):
    rng = np.random.default_rng(4)
    n = len(values)
    df = pd.DataFrame({"dose": values, "pH": rng.uniform(6, 7.5, n), "titer": rng.uniform(1, 5, n)})
    out = _safety(df, "titer")
    assert "dose" in out.cont_feats and "dose" not in out.overrides


def test_a_declared_row_number_like_feature_is_kept():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"Line": np.arange(2, 14), "pH": rng.uniform(6, 7.5, 12),
                       "titer": rng.uniform(1, 5, 12)})
    out = _safety(df, None, ColumnRoles(target="titer", features=["Line", "pH"]))
    assert out.cont_feats == ["Line", "pH"] and "Line" not in out.overrides
