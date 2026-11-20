"""`target_unit` (VIZ_SPEC R6): the display unit of the analysis target, from the
user's setup or a well-defined header pattern, never invented."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.analysis_parts.anonymize import _anonymize_result
from kalos.portal.analysis_parts.units import (
    MAX_USER_UNIT_CHARS,
    parse_header_unit,
    target_unit_block,
)


@pytest.mark.parametrize(("header", "unit"), [
    ("titer_g_L", "g/L"),
    ("titer_g_l", "g/L"),
    ("titer (g/L)", "g/L"),
    ("titer (g/l)", "g/L"),
    ("titer [mg/mL]", "mg/mL"),
    ("Titer g/L", "g/L"),
    ("titer_gL", "g/L"),
    ("titer_g_per_L", "g/L"),
    ("yield_pct", "%"),
    ("Yield (%)", "%"),
    ("Yield %", "%"),
    ("feed_rate_mL_h", "mL/h"),
    ("rate_mg_L_h", "mg/L/h"),
    ("glucose_mM", "mM"),
    ("Temp (°C)", "°C"),
    ("osmolality_mOsm_kg", "mOsm/kg"),
    ("VCD (10^6 cells/mL)", "10^6 cells/mL"),
    ("activity [U/mL]", "U/mL"),
    ("VCD (\u00d710^6 cells/mL)", "\u00d710^6 cells/mL"),
    # no unit: unrecognized or ambiguous suffixes are not guessed
    ("od600", None),
    ("OD600", None),
    ("pH", None),
    ("lipase_titer", None),
    ("temp_C", None),           # one-letter suffix alone: could be a label
    ("sample_volume_L", None),  # "L" alone collides with vessel labels ("5L")
    ("pH_min", None),           # min = minimum, not minutes
    ("temp min", None),
    ("viable_cells", None),
    ("titer A", None),
    ("titer (replicate 2)", None),
    ("x [AU]", None),
    ("Stage_1_L", None),
    ("", None),
])
def test_parse_header_unit(header, unit):
    assert parse_header_unit(header) == unit


def test_block_from_header():
    assert target_unit_block("titer_g_L", None) == {
        "unit": "g/L", "source": "header", "header": "titer_g_L",
        "note": "parsed from the target column header",
    }


def test_block_none_when_header_states_no_unit():
    b = target_unit_block("od600", None)
    assert b["unit"] is None and b["source"] == "none" and b["header"] == "od600"
    assert b["note"]
    b = target_unit_block(None, None)
    assert b == {"unit": None, "source": "none", "header": None, "note": "no target column"}


def test_user_unit_wins_and_is_kept_as_typed():
    b = target_unit_block("titer_g_L", "mg/mL")
    assert b["unit"] == "mg/mL" and b["source"] == "user" and b["header"] == "titer_g_L"
    assert "g/L" in b["note"]  # the disagreeing header is surfaced, not hidden
    b = target_unit_block("od600", " AU ")
    assert b["unit"] == "AU" and b["source"] == "user"
    assert "suggests" not in b["note"]


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_USER_UNIT_CHARS + 1), "g/\x00L", "m\ng", 5, None])
def test_bad_user_unit_is_ignored(bad):
    b = target_unit_block("titer_g_L", bad)
    assert b["source"] == "header" and b["unit"] == "g/L"


# --- anonymize ----------------------------------------------------------------------- #


def _anon(target: str, user: str | None) -> dict:
    result = {"target": target, "target_unit": target_unit_block(target, user)}
    return _anonymize_result(result)


def test_anonymize_aliased_header_drops_header_unit():
    out = _anon("Analyst (g/L)", None)
    tu = out["target_unit"]
    assert tu["header"] == out["target"] != "Analyst (g/L)"
    assert tu["unit"] is None and tu["source"] == "none"
    assert "Analyst" not in json.dumps(out) and "g/L" not in json.dumps(tu)


def test_anonymize_aliased_header_keeps_user_unit_without_header_hint():
    out = _anon("Analyst (g/L)", "mg/mL")
    tu = out["target_unit"]
    assert tu["unit"] == "mg/mL" and tu["source"] == "user"
    assert tu["header"] == out["target"]
    assert "g/L" not in tu["note"] and "Analyst" not in json.dumps(out)


def test_anonymize_unaliased_header_is_unchanged():
    out = _anon("titer_g_L", None)
    assert out["target"] == "titer_g_L"
    assert out["target_unit"]["unit"] == "g/L" and out["target_unit"]["header"] == "titer_g_L"


# --- in the analysis --------------------------------------------------------------- #


def _sheet(target: str) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 16
    m = rng.uniform(0.5, 3.5, n).round(3)
    ph = rng.uniform(6.0, 7.5, n).round(3)
    return pd.DataFrame({"Methanol": m, "pH": ph,
                         target: (2 + m - (ph - 6.8) ** 2 + rng.normal(0, 0.1, n)).round(4)})


def test_analysis_carries_target_unit():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    out = _analyze(_sheet("titer (g/L)"), target="titer (g/L)")
    assert out["target_unit"] == target_unit_block("titer (g/L)", None)
    out = _analyze(_sheet("titer_g_L"), target="titer_g_L", user_target_unit="mg/mL",
                   anonymize=True)
    assert out["target_unit"]["unit"] == "mg/mL" and out["target_unit"]["source"] == "user"
    assert out["target_unit"]["header"] == out["target"] == "titer_g_L"
