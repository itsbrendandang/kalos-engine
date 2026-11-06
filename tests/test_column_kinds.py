"""`columns` in the analysis response: every uploaded column with its kind and
what the engine did with it (kalos/portal/analysis_parts/column_kinds.py)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")

from kalos.portal.analysis import _analyze
from kalos.portal.analysis_parts.column_kinds import column_kinds


def _sheet(n: int = 40, n_sources: int = 19, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gly = rng.uniform(0, 10, n).round(2)
    met = rng.uniform(0, 5, n).round(2)
    titer = (10 + gly - 0.5 * (met - 2.5) ** 2 + rng.normal(0, 0.3, n)).round(3)
    return pd.DataFrame({
        "": np.arange(n),                                    # headerless row index
        "CarbonSource": [f"src_{i % n_sources}" for i in range(n)],
        "Sample": [f"S-{1000 + i}" for i in range(n)],      # one value per row
        "Gly": gly, "Met": met,
        "Vessel": ["V1"] * n,                                # constant
        "Titer": titer,
        "OD_Prod": (titer * 0.4 + rng.normal(0, 0.1, n)).round(3),  # measured after
    })


def _by_name(out: dict) -> dict:
    return {c["name"]: c for c in out["columns"]}


def test_every_column_gets_a_kind_and_a_text_input_is_never_numeric():
    df = _sheet()
    out = _analyze(df, "Titer")
    json.dumps(out["columns"], allow_nan=False)
    assert len(out["columns"]) == df.shape[1]
    cols = _by_name(out)
    src = cols["CarbonSource"]
    assert src["kind"] == "category" and src["role"] == "excluded" and not src["used_in_fit"]
    assert src["n_levels"] == 19 and src["finding_code"] == "category_not_modeled"
    assert src["reason"] == "too_many_levels"
    assert sum(lv["count"] for lv in src["levels"]) == 40
    assert src["levels"][0]["count"] >= src["levels"][-1]["count"]
    assert "CarbonSource" not in out["features"]
    assert all("CarbonSource" not in p["recipe"] for p in out["proposals"])
    codes = [f for f in out["validation"]["findings"] if f["code"] == "category_not_modeled"]
    assert len(codes) == 1 and codes[0]["column"] == "CarbonSource"
    assert "declare it categorical" in codes[0]["message"]
    assert cols["Gly"]["kind"] == "numeric" and cols["Gly"]["role"] == "input"
    assert cols["Gly"]["used_in_fit"] is True and cols["Gly"]["levels"] is None
    assert cols["Titer"]["role"] == "target"
    assert cols["Sample"]["kind"] == "text_id"
    assert cols["Vessel"]["kind"] == "constant"
    assert cols["OD_Prod"]["kind"] == "outcome_like" and not cols["OD_Prod"]["used_in_fit"]
    row_index = next(c for c in out["columns"] if c["kind"] == "row_number")
    assert row_index["used_in_fit"] is False


def test_a_small_category_is_a_categorical_input_and_proposals_carry_labels():
    df = _sheet(n=36, n_sources=3).drop(columns=["", "Sample", "Vessel", "OD_Prod"])
    out = _analyze(df, "Titer")
    src = _by_name(out)["CarbonSource"]
    assert src["kind"] == "category" and src["role"] == "categorical_input"
    assert src["used_in_fit"] is True and src["n_levels"] == 3
    assert {lv["label"] for lv in src["levels"]} == {"src_0", "src_1", "src_2"}
    assert all(p["recipe"]["CarbonSource"] in {"src_0", "src_1", "src_2"} for p in out["proposals"])


def test_mixed_text_and_numbers_and_rare_labels_say_why():
    df = pd.DataFrame({
        "mixed": ["a", "b", "1", "a", "b", "2"],
        "rare": ["x1", "x2", "x3", "x1", "x4", "x5"],
        "Titer": [1, 2, 3, 4, 5, 6],
    })
    prov = [{"name": "mixed", "status": "dropped_sparse"}, {"name": "rare", "status": "dropped_sparse"},
            {"name": "Titer", "status": "target"}]
    cols, findings = column_kinds(df, prov, kept_cont=[], kept_cats=[], max_auto_levels=12)
    by = {c["name"]: c for c in cols}
    assert by["mixed"]["reason"] == "mixed_text_and_numbers"
    assert by["rare"]["kind"] == "category" and by["rare"]["reason"] == "too_few_repeats"
    assert {f.code for f in findings} == {"category_not_modeled"}


def test_anonymize_aliases_identifier_columns_and_their_levels():
    df = _sheet(n=36, n_sources=3).drop(columns=[""])
    df["Operator"] = ["Jane Smith", "Ali Khan", "Jane Smith", "Ali Khan"] * 9
    out = _analyze(df, "Titer", anonymize=True)
    text = json.dumps(out["columns"])
    assert "Jane Smith" not in text and "Operator" not in text
    assert "CarbonSource" in text  # process variables keep their names
