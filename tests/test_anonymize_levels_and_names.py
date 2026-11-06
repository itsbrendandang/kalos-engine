"""`anonymize=True` must not leak identifier names or their values (review H4).

Before: categorical level values were published verbatim (a recipe carried
"Jane Smith 0" and "alice_secret_op"), and name-like columns (Donor_Name_*,
Operator, Analyst, Technician, Lab_Tech, email) were not identifiers at all.

The check is a scan of the whole serialized response: no identifier column name
and no identifier level value may appear in it. Process variables keep their names.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from kalos.domains import ColumnRoles
from kalos.portal.analysis import _analyze, is_identifier_name

_PEOPLE_COLUMNS = ["Donor_Name_1", "Operator", "Analyst", "Technician", "Lab_Tech", "email"]
_NAMES = ["Jane Smith 0", "alice_secret_op", "Bob Jones 1", "carol_secret_op"]


@pytest.mark.parametrize("name", _PEOPLE_COLUMNS)
def test_person_columns_are_identifiers(name):
    assert is_identifier_name(name)


@pytest.mark.parametrize("name", ["Methanol", "pH", "sample_volume_L", "technology_level"])
def test_process_variables_are_not_identifiers(name):
    assert not is_identifier_name(name)


def _people_sheet() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 24
    ph = rng.uniform(6.0, 7.5, n).round(3)
    methanol = rng.uniform(0.5, 3.5, n).round(3)
    people = [_NAMES[i % len(_NAMES)] for i in range(n)]
    titer = (3 + 1.2 * methanol - 1.5 * (ph - 6.8) ** 2 + rng.normal(0, 0.1, n)).round(4)
    return pd.DataFrame({
        "Donor_Name_1": people,
        "Operator": people,
        "email": [f"{p.split()[0].lower()}@example.org" for p in people],
        "Methanol": methanol,
        "pH": ph,
        "lipase_titer": titer,
    })


def test_no_identifier_name_or_level_value_appears_in_an_anonymized_response():
    roles = ColumnRoles.from_dict({
        "target": "lipase_titer",
        "features": ["Methanol", "pH"],
        "categoricals": ["Donor_Name_1"],
    })
    out = _analyze(_people_sheet(), target="lipase_titer", roles=roles, anonymize=True)
    blob = json.dumps(out, default=str)
    for name in _NAMES:
        assert name not in blob, f"level value {name!r} leaked"
    assert "Donor_Name_1" not in blob and "Operator" not in blob
    assert "email" not in blob
    # Process variables keep their real names.
    assert "Methanol" in blob and "pH" in blob
    assert "Methanol" in out["features"]


def test_the_recipes_carry_pseudonyms_not_the_level_values():
    roles = ColumnRoles.from_dict({
        "target": "lipase_titer",
        "features": ["Methanol", "pH"],
        "categoricals": ["Donor_Name_1"],
    })
    out = _analyze(_people_sheet(), target="lipase_titer", roles=roles, anonymize=True)
    for row in out["proposals"]:
        for _key, value in row["recipe"].items():
            if isinstance(value, str):
                assert value not in _NAMES
                assert value.startswith("level_")


def test_fixed_inputs_keys_are_aliased_under_anonymize():
    """A declared identifier-named input that is fixed (near-constant) must not
    leak its real name through `fixed_inputs`."""
    from kalos.portal.analysis_parts.anonymize import _anonymize_result

    out = _anonymize_result({"fixed_inputs": {"Operator": 3.0, "pH": 7.0}, "provenance": []})
    assert "Operator" not in out["fixed_inputs"] and "pH" in out["fixed_inputs"]
    assert sorted(out["fixed_inputs"].values()) == [3.0, 7.0]
    assert next(iter(k for k in out["fixed_inputs"] if k != "pH")).startswith("col_")
