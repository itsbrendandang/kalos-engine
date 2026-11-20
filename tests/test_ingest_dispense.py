"""Dispense-volume columns: detection and the C1V1 conversion (kalos.ingest.dispense).

The fixture tests/fixtures/lab_ingest/dispense_volumes.csv is synthetic: it
copies the header styles of liquid-handler media sheets ("<name>.uL.to.add",
"<name> uL to add", "<name>_uL", "(mL)") with made-up components and values.
Every expected concentration below is worked out by hand in the comment next
to it: c = c_stock x V / V_final, summed over the columns that deliver it.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from kalos.ingest import dispense
from kalos.ingest.dispense import (
    FinalVolumeColumn,
    Quantity,
    concentration_unit,
    convert,
    detect_dispense_columns,
    parse_dispense_header,
    parse_quantity,
)
from kalos.validation.report import Finding

FIX = Path(__file__).parent / "fixtures" / "lab_ingest" / "dispense_volumes.csv"

VOLUME_COLUMNS = ["Glycerol.uL.to.add", "Citric.acid..uL.to.add", "Trace mix uL to add",
                  "Base medium_uL", "Methanol (mL)", "Water uL to add"]
STOCKS = {"Glycerol": "500 g/L", "Citric acid": (100, "g/L"), "Methanol": Quantity(100, "% v/v")}
MIXES = {
    "Trace mix": {"Citric acid": "5 g/L", "Zinc sulfate": "1 g/L", "Biotin": "0.02 g/L"},
    "Base medium": {"Magnesium sulfate": "5 g/L", "Glycerol": "10 g/L"},
    "Water": {},
}
OUT = ["Glycerol (g/L)", "Citric acid (g/L)", "Zinc sulfate (g/L)", "Biotin (g/L)",
       "Magnesium sulfate (g/L)", "Methanol (% v/v)"]


def sheet() -> pd.DataFrame:
    # read as text so "n/a" and blanks stay as written
    return pd.read_csv(FIX, dtype=str, keep_default_na=False)


def checks(result) -> dict[str, Finding]:
    return {f.check: f for f in result.findings}


def checks_by_column(result, check: str) -> dict[str | None, Finding]:
    return {f.column: f for f in result.findings if f.check == check}


# -- detection ---------------------------------------------------------------


@pytest.mark.parametrize(("header", "component", "unit"), [
    ("Glycerol.uL.to.add", "glycerol", "uL"),
    ("Citric.acid..uL.to.add", "citric acid", "uL"),
    ("GLYCEROL UL TO ADD", "glycerol", "uL"),
    ("glycerol-ul-to-add", "glycerol", "uL"),
    ("Trace mix uL to add", "trace mix", "uL"),
    ("Base medium_uL", "base medium", "uL"),
    ("Methanol.uL", "methanol", "uL"),
    ("Yeast extract (\u00b5L)", "yeast extract", "uL"),
    ("Yeast extract (\u03bcl)", "yeast extract", "uL"),
    ("Methanol (mL)", "methanol", "mL"),
    ("Feed (L)", "feed", "L"),
    ("Glycerol volume (uL)", "glycerol", "uL"),
    ("Volume (mL)", None, "mL"),
    ("Glycerol, added as a liquid", "glycerol", None),
    ("Base medium added as a liquid (mL)", "base medium", "mL"),
    ("Antifoam dispensed (uL)", "antifoam", "uL"),
    ("Sample uL to add", "sample", "uL"),  # "to add" outranks the vessel words
])
def test_volume_headers_are_flagged(header, component, unit):
    dc = parse_dispense_header(header)
    assert dc is not None, header
    assert (dc.component, dc.unit) == (component, unit)


@pytest.mark.parametrize("header", [
    # concentrations in every common spelling
    "Glycerol (g/L)", "Glycerol_g.L", "Glycerol_g_per_L", "Citric acid (mg/mL)", "MgSO4 (mM)",
    "Glycerol (% w/v)", "Methanol (% v/v)", "Trace mix (mL/L)", "Trace_mix_mL.L", "Trace mix uL/mL",
    "Product_ng.ul", "Product (\u00b5g/mL)", "Product (mg/L)",
    # rates, cell counts
    "Feed rate (mL/h)", "Air (L/min)", "Feed_mL.h", "Cells per mL", "VCD_1e6_per_mL",
    # vessel and process volumes
    "Culture_Volume.mL", "Working volume (mL)", "Final volume (uL)", "Fill volume (L)",
    "Inoculum volume (mL)", "Scale (L)", "Shake flask 250 mL", "Sample volume (uL)",
    # labels and other columns
    "Clone L", "5L", "Sample", "Well.Label", "OD600", "pH", "Time (h)", "Glycerol to add",
    "Volume", "antifoam.204", "Glucose (g)",
])
def test_concentration_and_other_headers_are_not_flagged(header):
    assert parse_dispense_header(header) is None, header


def test_detect_on_fixture_is_one_error_per_volume_column():
    found = detect_dispense_columns(sheet())
    assert [f.column for f in found] == VOLUME_COLUMNS
    for f in found:
        assert f.check == "volumes_not_concentrations" and f.severity == "error"
        assert "kept" in f.message and "stock concentration" in f.message
    assert found[4].detail == {"component": "methanol", "volume_unit": "mL", "pattern": "unit_suffix"}


# -- units ---------------------------------------------------------------------


@pytest.mark.parametrize(("unit", "want"), [
    ("g/L", "g/L"), ("g/l", "g/L"), ("G/L", "g/L"), ("mg/mL", "mg/mL"), ("\u00b5g/mL", "ug/mL"),
    ("% w/v", "% w/v"), ("%(w/v)", "% w/v"), ("% v/v", "% v/v"), ("mM", "mM"), ("M", "M"),
    ("mmol/L", "mM"), ("mol/l", "M"), ("\u03bcM", "uM"), ("m", None), ("MM", None), ("x", None),
    ("U/mL", None),
])
def test_concentration_unit_table(unit, want):
    assert concentration_unit(unit) == want


def test_parse_quantity_forms():
    assert parse_quantity("500 g/L") == Quantity(500.0, "g/L")
    assert parse_quantity((3, "mL")) == Quantity(3.0, "mL")
    assert parse_quantity({"value": 0.5, "unit": "M"}) == Quantity(0.5, "M")
    for bad in ("3", "lots", None, 3.0, (True, "mL"), ("3", "mL"), (3, "")):
        assert parse_quantity(bad) is None


# -- conversion: goldens ------------------------------------------------------


def test_fixture_conversion_goldens():
    frame = sheet()
    res = convert(frame, STOCKS, "3 mL", MIXES)
    t = res.table
    assert len(t) == len(frame)  # no row dropped
    assert list(t.columns) == [*frame.columns, *OUT, "dispense_status", "dispense_provenance"]
    # the raw volume columns are kept unchanged; mixes and water get no column of their own
    assert t[VOLUME_COLUMNS].equals(frame[VOLUME_COLUMNS])
    assert not any(c.startswith(("Trace mix (", "Base medium (", "Water (")) for c in t.columns)
    want = {
        # M01: 30 uL, 15 uL, 60 uL trace, 2700 uL base, 0.015 mL methanol in 3 mL (3000 uL)
        0: {"Glycerol (g/L)": 500 * 30 / 3000 + 10 * 2700 / 3000,      # 5 + 9 = 14
            "Citric acid (g/L)": 100 * 15 / 3000 + 5 * 60 / 3000,      # 0.5 + 0.1 = 0.6
            "Zinc sulfate (g/L)": 1 * 60 / 3000,                       # 0.02
            "Biotin (g/L)": 0.02 * 60 / 3000,                          # 0.0004
            "Magnesium sulfate (g/L)": 5 * 2700 / 3000,                # 4.5
            "Methanol (% v/v)": 100 * 0.015 / 3},                      # 0.5
        # M02: no citric acid stock added; it still comes in with the trace mix
        1: {"Glycerol (g/L)": 19.0, "Citric acid (g/L)": 0.05, "Zinc sulfate (g/L)": 0.01,
            "Biotin (g/L)": 0.0002, "Magnesium sulfate (g/L)": 4.5, "Methanol (% v/v)": 1.0},
        # M03: 2850 uL added of 3000 uL (the remainder is reported, not filled)
        2: {"Glycerol (g/L)": 2.5 + 8.0, "Citric acid (g/L)": 1.0 + 0.15,
            "Zinc sulfate (g/L)": 0.03, "Biotin (g/L)": 0.0006,
            "Magnesium sulfate (g/L)": 4.0, "Methanol (% v/v)": 0.5},
    }
    for row, values in want.items():
        for col, value in values.items():
            assert t.at[row, col] == pytest.approx(value, rel=1e-12), (row, col)
    assert list(t["dispense_status"]) == [
        "converted", "converted", "converted", "refused_added_volume_exceeds_final",
        "refused_volume_negative", "refused_volume_not_numeric", "refused_volume_not_numeric"]
    assert t.loc[3:, OUT].isna().all().all()  # refused rows get no value, never a filled one
    assert t.at[3, "dispense_provenance"] == {}


def test_formulas_provenance_and_roles():
    res = convert(sheet(), STOCKS, "3 mL", MIXES)
    m = res.meta
    assert m["parser"] == "dispense/1" and m["converted"] is True
    assert m["formulas"]["Citric acid (g/L)"] == (
        "c = 100 g/L x V(Citric.acid..uL.to.add) / V_final"
        " + 5 g/L in Trace mix x V(Trace mix uL to add) / V_final")
    assert m["formulas"]["Methanol (% v/v)"] == "c = 100 % v/v x V(Methanol (mL)) / V_final"
    assert set(m["formulas"]) == set(OUT) == set(m["conversion"])
    src = m["conversion"]["Glycerol (g/L)"]["sources"]
    assert [(s["column"], s["kind"], s["declared_as"]) for s in src] == [
        ("Glycerol.uL.to.add", "stock", "Glycerol"), ("Base medium_uL", "mix", "Base medium")]
    assert m["conversion"]["Methanol (% v/v)"]["sources"][0]["volume_factor_to_L"] == 1e-3
    assert m["final_volume"] == {"value": 3.0, "unit": "mL"}
    assert m["stocks"]["Glycerol"] == {"value": 500.0, "unit": "g/L"}
    assert m["master_mixes"]["Water"] == {}
    # converted concentrations are recipe inputs; the raw volume columns get no suggestion
    assert m["role_suggestions"] == dict.fromkeys(OUT, "setting")
    prov = res.table.at[0, "dispense_provenance"]
    assert prov["Citric acid (g/L)"] == (
        "100 g/L x 15 uL / 3 mL + 5 g/L in Trace mix x 60 uL / 3 mL = 0.6 g/L")
    assert prov["Methanol (% v/v)"] == "100 % v/v x 0.015 mL / 3 mL = 0.5 % v/v"


def test_row_refusals_are_reported_by_row():
    res = convert(sheet(), STOCKS, "3 mL", MIXES)
    c = checks(res)
    assert c["volume_negative"].column == "Glycerol.uL.to.add" and c["volume_negative"].rows == (4,)
    not_num = checks_by_column(res, "volume_not_numeric")
    assert not_num["Citric.acid..uL.to.add"].rows == (5,)
    assert not_num["Citric.acid..uL.to.add"].detail["raw_values"] == ["n/a"]
    assert not_num["Trace mix uL to add"].rows == (6,)
    assert not_num["Trace mix uL to add"].detail["raw_values"] == ["(blank)"]  # never read as 0
    over = c["added_volume_exceeds_final"]
    # M04: 45 + 15 + 60 + 2700 + 15 + 300 = 3135 uL > 3000 uL
    assert over.severity == "error" and over.rows == (3,)
    assert over.detail["where"] == [{"row": 3, "total_added_mL": 3.135, "final_mL": 3.0}]
    assert c["volume_remainder_not_declared"].rows == (2,)
    assert c["volume_remainder_not_declared"].severity == "info"
    assert c["mix_declared_empty"].detail == {"mix": "Water"}
    assert c["volume_columns_kept"].detail["columns"] == VOLUME_COLUMNS
    assert "volumes_not_concentrations" not in c  # converted; the kept columns are noted instead
    for f in res.findings:
        assert f.message.endswith(".")


def test_per_row_final_volume_column():
    res = convert(sheet(), STOCKS, FinalVolumeColumn("Culture volume (mL)", "mL"), MIXES)
    t = res.table
    # M03 was made up to 2.85 mL: glycerol (500 x 15 + 10 x 2400) / 2850 = 11.0526...
    assert t.at[2, "Glycerol (g/L)"] == pytest.approx(31500 / 2850, rel=1e-12)
    assert t.at[2, "Citric acid (g/L)"] == pytest.approx((100 * 30 + 5 * 90) / 2850, rel=1e-12)
    assert t.at[0, "Glycerol (g/L)"] == pytest.approx(14.0)
    assert "volume_remainder_not_declared" not in checks(res)  # 2850 uL of 2.85 mL
    assert t.at[2, "dispense_provenance"]["Zinc sulfate (g/L)"].endswith("/ 2.85 mL = 0.0315789 g/L")
    assert res.meta["conversion"]["Glycerol (g/L)"]["final_volume"] == "Culture volume (mL) (mL)"
    # a final-volume cell that is blank or not positive refuses that row only
    frame = sheet()
    frame.loc[1, "Culture volume (mL)"] = ""
    frame.loc[2, "Culture volume (mL)"] = "0"
    res = convert(frame, STOCKS, FinalVolumeColumn("Culture volume (mL)", "mL"), MIXES)
    assert checks(res)["final_volume_invalid"].rows == (1, 2)
    assert list(res.table["dispense_status"][:3]) == [
        "converted", "refused_final_volume_invalid", "refused_final_volume_invalid"]


def test_final_volume_column_that_looks_like_a_volume_is_not_a_source():
    frame = pd.DataFrame({"Glycerol uL to add": [30, 60], "Volume (mL)": [3, 6]})
    res = convert(frame, {"Glycerol": "500 g/L"}, FinalVolumeColumn("Volume (mL)", "mL"))
    assert list(res.table["Glycerol (g/L)"]) == pytest.approx([5.0, 5.0])  # 500 x 30/3000, x 60/6000
    assert res.meta["role_suggestions"] == {"Glycerol (g/L)": "setting"}


# -- double counting ------------------------------------------------------------


def test_component_in_a_stock_and_in_a_mix_is_counted_once_per_column():
    frame = pd.DataFrame({"Citric acid uL to add": [30.0], "Trace mix uL to add": [60.0]})
    mixes = {"Trace mix": {"Citric acid": "5 g/L", "Zinc sulfate": "1 g/L"}}
    res = convert(frame, {"Citric acid": "100 g/L"}, "3 mL", mixes)
    t = res.table
    # stock: 100 x 30 / 3000 = 1.0; mix: 5 x 60 / 3000 = 0.1 -> 1.1, not 1.2 (mix twice)
    # and not 1.0 + 100 x 60 / 3000 (the mix read as the citric acid stock)
    assert t.at[0, "Citric acid (g/L)"] == pytest.approx(1.1, rel=1e-12)
    assert t.at[0, "Zinc sulfate (g/L)"] == pytest.approx(0.02, rel=1e-12)
    assert "Trace mix (g/L)" not in t.columns  # the mix is expanded, never its own ingredient
    assert len(res.meta["conversion"]["Citric acid (g/L)"]["sources"]) == 2


def test_double_counting_declarations_are_refused():
    frame = pd.DataFrame({"Citric acid uL to add": [30.0], "Trace mix uL to add": [60.0]})
    # a mix declared also as a stock
    res = convert(frame, {"Citric acid": "100 g/L", "Trace mix": "10 g/L"}, "3 mL",
                  {"Trace mix": {"Citric acid": "5 g/L"}})
    assert checks(res)["declaration_duplicated"].severity == "error"
    assert list(res.table.columns) == list(frame.columns) and res.meta["converted"] is False
    # a mix listing one component twice under two spellings
    res = convert(frame, {"Citric acid": "100 g/L"}, "3 mL",
                  {"Trace mix": {"Citric acid": "5 g/L", "citric.acid": "5 g/L"}})
    assert checks(res)["mix_component_duplicated"].detail["component"] == "citric.acid"
    # two columns that dispense the same stock (a copied column)
    dup = frame.assign(**{"Citric acid_uL": [30.0]})
    res = convert(dup, {"Citric acid": "100 g/L"}, "3 mL", {"Trace mix": {"Citric acid": "5 g/L"}})
    f = checks(res)["dispense_source_duplicated"]
    assert f.detail["columns"] == ["Citric acid uL to add", "Citric acid_uL"]
    # a mix inside a mix is not expanded
    res = convert(frame, {"Citric acid": "100 g/L"}, "3 mL",
                  {"Trace mix": {"Base": "1 g/L"}, "Base": {"Zinc sulfate": "1 g/L"}})
    assert "mix_nested" in checks(res)


# -- units in the conversion ------------------------------------------------------


def test_volume_units_convert_within_volume_only():
    frame = pd.DataFrame({"Glycerol (mL)": [10.0], "Trace mix (uL)": [500.0]})
    res = convert(frame, {"Glycerol": "500 g/L"}, "0.5 L", {"Trace mix": {"Zinc sulfate": "1 g/L"}})
    # 500 g/L x 10 mL / 500 mL = 10 g/L; 1 g/L x 0.5 mL / 500 mL = 0.001 g/L
    assert res.table.at[0, "Glycerol (g/L)"] == pytest.approx(10.0, rel=1e-12)
    assert res.table.at[0, "Zinc sulfate (g/L)"] == pytest.approx(0.001, rel=1e-12)


def test_concentration_units_convert_within_a_dimension():
    frame = pd.DataFrame({"Glycerol uL to add": [60.0], "Base medium uL to add": [300.0]})
    res = convert(frame, {"Glycerol": "50 % w/v"}, "3 mL", {"Base medium": {"Glycerol": "10 g/L"}})
    # output in the first source's unit: 50 % w/v x 60/3000 = 1; 10 g/L = 1 % w/v, x 300/3000 = 0.1
    assert res.table.at[0, "Glycerol (% w/v)"] == pytest.approx(1.1, rel=1e-12)
    src = res.meta["conversion"]["Glycerol (% w/v)"]["sources"]
    assert [s["factor_to_output_unit"] for s in src] == pytest.approx([1.0, 0.1])
    assert "10 g/L (= 1 % w/v) in Base medium" in res.meta["formulas"]["Glycerol (% w/v)"]


def test_mass_and_molar_only_with_a_declared_molecular_weight():
    frame = pd.DataFrame({"Phosphate uL to add": [30.0], "Base medium uL to add": [300.0],
                          "Glycerol uL to add": [30.0]})
    stocks = {"Phosphate": "1 M", "Glycerol": "500 g/L"}
    mixes = {"Base medium": {"Phosphate": "13.609 g/L"}}
    refused = convert(frame, stocks, "3 mL", mixes)
    f = checks(refused)["concentration_units_incompatible"]
    assert f.severity == "error" and "molecular weight" in f.message
    assert "Phosphate (M)" not in refused.table.columns
    assert refused.table.at[0, "Glycerol (g/L)"] == pytest.approx(5.0)  # the others still convert
    assert refused.meta["refused_components"] == {"Phosphate": "concentration_units_incompatible"}
    res = convert(frame, stocks, "3 mL", mixes, molecular_weights={"phosphate": 136.09})
    # 1 M x 30/3000 = 0.01 M; 13.609 g/L / 136.09 g/mol = 0.1 M, x 300/3000 = 0.01 M -> 0.02 M
    assert res.table.at[0, "Phosphate (M)"] == pytest.approx(0.02, rel=1e-12)
    assert "MW 136.09 g/mol" in res.meta["formulas"]["Phosphate (M)"]
    assert res.meta["molecular_weights"] == {"phosphate": 136.09}


def test_unknown_or_invalid_stock_refuses_that_component_only():
    frame = pd.DataFrame({"Glycerol uL to add": [30.0], "Trace mix uL to add": [3.0],
                          "Biotin uL to add": [3.0]})
    res = convert(frame, {"Glycerol": "500 g/L", "Biotin": "-1 g/L"}, "3 mL",
                  {"Trace mix": {"Zinc sulfate": "1000x"}})
    c = checks(res)
    assert c["concentration_unit_unknown"].detail["units"] == ["x"]
    assert c["stock_invalid"].detail["stock"] == "Biotin"
    assert res.meta["refused_components"] == {
        "Zinc sulfate": "concentration_unit_unknown", "Biotin": "stock_invalid"}
    assert [c for c in res.table.columns if "(" in c] == ["Glycerol (g/L)"]


def test_other_unit_and_declaration_refusals():
    frame = pd.DataFrame({"Methanol uL to add": [15.0], "Feed mix uL to add": [30.0]})
    # volume/volume and mass/volume never add up, with or without a molecular weight
    res = convert(frame, {"Methanol": "100 % v/v"}, "3 mL", {"Feed mix": {"Methanol": "790 g/L"}},
                  molecular_weights={"Methanol": 32.04})
    f = checks(res)["concentration_units_incompatible"]
    assert "molecular weight" not in f.message and "Methanol (% v/v)" not in res.table.columns
    # a bad component inside a mix refuses that component only
    res = convert(frame, {"Methanol": "100 % v/v"}, "3 mL", {"Feed mix": {"Glycerol": "0 g/L"}})
    assert checks(res)["stock_invalid"].detail["component"] == "Glycerol"
    assert res.table.at[0, "Methanol (% v/v)"] == pytest.approx(0.5)
    # a molecular weight that is not a positive number refuses the conversion
    res = convert(frame, {"Methanol": "100 % v/v"}, "3 mL", {"Feed mix": {}},
                  molecular_weights={"Methanol": 0})
    assert checks(res)["molecular_weight_invalid"].detail["components"] == ["Methanol"]
    assert res.meta["converted"] is False


def test_existing_output_column_is_not_overwritten():
    frame = pd.DataFrame({"Glycerol uL to add": [30.0], "Glycerol (g/L)": [7.0]})
    res = convert(frame, {"Glycerol": "500 g/L"}, "3 mL")
    assert checks(res)["output_column_exists"].column == "Glycerol (g/L)"
    assert res.table.at[0, "Glycerol (g/L)"] == 7.0


# -- refusals of the whole conversion ---------------------------------------------


def test_column_with_no_stock_refuses_everything_and_never_guesses():
    frame = sheet()
    mixes = {k: v for k, v in MIXES.items() if k != "Water"}
    res = convert(frame, STOCKS, "3 mL", mixes)
    c = checks(res)
    assert c["dispense_column_no_stock"].column == "Water uL to add"
    assert res.meta["converted"] is False and res.meta["role_suggestions"] == {}
    assert res.table.equals(frame)  # nothing added, nothing changed
    # the detection errors stay: the volume columns are still not concentrations
    assert {f.column for f in res.findings if f.check == "volumes_not_concentrations"} == set(
        VOLUME_COLUMNS)
    none = convert(frame, None, "3 mL")
    assert len([f for f in none.findings if f.check == "dispense_column_no_stock"]) == 6


@pytest.mark.parametrize(("final_volume", "check"), [
    (None, "final_volume_missing"),
    ("3", "final_volume_invalid"),
    ("3 g/L", "final_volume_invalid"),
    ("-1 mL", "final_volume_invalid"),
    ((0, "mL"), "final_volume_invalid"),
    (FinalVolumeColumn("Not a column", "mL"), "final_volume_missing"),
    (FinalVolumeColumn("Culture volume (mL)", "cups"), "final_volume_unit_unknown"),
    (FinalVolumeColumn("Culture volume (mL)", "uL"), "final_volume_unit_conflict"),
])
def test_final_volume_refusals(final_volume, check):
    res = convert(sheet(), STOCKS, final_volume, MIXES)
    assert checks(res)[check].severity == "error"
    assert res.meta["converted"] is False and "dispense_status" not in res.table.columns


def test_header_without_component_or_unit_is_refused():
    res = convert(pd.DataFrame({"Volume (mL)": [0.03]}), {"Glycerol": "500 g/L"}, "3 mL")
    assert checks(res)["dispense_component_unknown"].column == "Volume (mL)"
    res = convert(pd.DataFrame({"Glycerol volume": [30.0]}), {"Glycerol": "500 g/L"}, "3 mL")
    assert checks(res)["volume_unit_missing"].column == "Glycerol volume"


def test_unused_declaration_is_a_warning_and_names_match_loosely():
    frame = pd.DataFrame({"Citric.acid..uL.to.add": [30.0]})
    res = convert(frame, {"CITRIC ACID": "100 g/L", "Glucose": "500 g/L"}, "3 mL")
    assert res.table.at[0, "CITRIC ACID (g/L)"] == pytest.approx(1.0)
    assert checks(res)["declaration_not_used"].detail["names"] == ["Glucose"]


def test_no_dispense_columns_is_info_only():
    frame = pd.DataFrame({"Glycerol (g/L)": [5.0], "OD600": [10.0]})
    res = convert(frame, {"Glycerol": "500 g/L"}, "3 mL")
    assert [f.check for f in res.findings] == ["no_dispense_columns"]
    assert res.table.equals(frame) and detect_dispense_columns(frame) == ()


# -- contracts ---------------------------------------------------------------


RESERVED_CHECKS = {"outliers", "physical_bounds", "duplicate_ids", "result_duplicate_entry",
                   "target_leak", "derived_output"}


def _every_result():
    """One run of every path, so every finding check the module can emit is produced."""
    two = pd.DataFrame({"Citric acid uL to add": [30.0], "Trace mix uL to add": [60.0]})
    meoh = pd.DataFrame({"Methanol uL to add": [15.0], "Feed mix uL to add": [30.0]})
    yield convert(sheet(), STOCKS, "3 mL", MIXES)
    yield convert(sheet(), {}, None)
    yield convert(sheet(), STOCKS, "3", MIXES)
    yield convert(sheet(), STOCKS, FinalVolumeColumn("Culture volume (mL)", "cups"), MIXES)
    yield convert(sheet(), STOCKS, FinalVolumeColumn("Culture volume (mL)", "uL"), MIXES)
    frame = sheet()
    frame.loc[1, "Culture volume (mL)"] = ""
    yield convert(frame, STOCKS, FinalVolumeColumn("Culture volume (mL)", "mL"), MIXES)
    yield convert(two, {"Citric acid": "1 g/L", "Trace mix": "1 g/L"}, "3 mL",
                  {"Trace mix": {"Base": "1 g/L", "base": "1 g/L"}, "Base": {}})
    yield convert(two.assign(**{"Citric acid_uL": [1.0]}), {"Citric acid": "1 g/L", "Glucose": "1 g/L"},
                  "3 mL", {"Trace mix": {}}, molecular_weights={"Citric acid": -1})
    yield convert(pd.DataFrame({"Volume (mL)": [1.0], "Glycerol volume": [1.0]}), {"Glycerol": "1 g/L"},
                  "3 mL")
    yield convert(meoh, {"Methanol": "100 % v/v"}, "3 mL",
                  {"Feed mix": {"Methanol": "790 g/L", "Biotin": "1000x", "Zinc sulfate": "0 g/L"}})
    yield convert(pd.DataFrame({"Glycerol uL to add": [30.0], "Glycerol (g/L)": [7.0]}),
                  {"Glycerol": "500 g/L"}, "3 mL")
    yield convert(pd.DataFrame({"OD600": [1.0]}), {}, "3 mL")


def test_every_finding_is_a_validation_finding_and_every_check_is_exercised():
    seen: set[str] = set()
    for res in _every_result():
        for f in [*res.findings, *detect_dispense_columns(res.table)]:
            assert isinstance(f, Finding)
            assert re.fullmatch(r"[a-z][a-z0-9_]*", f.check) and f.check not in RESERVED_CHECKS
            assert f.severity in ("error", "warning", "info") and f.message.endswith(".")
            if f.severity == "error":
                # every refusal says what was kept or that nothing was changed
                assert re.search(r"kept|nothing was changed|still converted|not overwritten",
                                 f.message), f.check
            seen.add(f.check)
    emitted = set(re.findall(r'finding\(\s*"([a-z_]+)"', Path(dispense.__file__).read_text()))
    assert emitted and seen == emitted, sorted(emitted ^ seen)


def test_results_are_json_safe():
    for res in (convert(sheet(), STOCKS, "3 mL", MIXES),
                convert(sheet(), STOCKS, FinalVolumeColumn("Culture volume (mL)", "mL"), MIXES),
                convert(sheet(), None, None)):
        text = json.dumps(res.to_dict(), allow_nan=False)
        assert "NaN" not in text
    out = convert(sheet(), STOCKS, "3 mL", MIXES).to_dict()
    assert out["table"][3]["Glycerol (g/L)"] is None  # a refused row is null, not a number
    json.dumps([f.to_dict() for f in detect_dispense_columns(sheet())], allow_nan=False)


def test_deterministic():
    a = convert(sheet(), STOCKS, "3 mL", MIXES).to_dict()
    b = convert(sheet(), STOCKS, "3 mL", MIXES).to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_ingest_imports_stay_in_layer():
    """kalos/ingest imports only util, data, config, core, libs and runtime (scripts/import_layers.py)."""
    allowed = ("kalos.ingest", "kalos.util", "kalos.data", "kalos.config", "kalos.envconfig",
               "kalos.core", "kalos.features", "kalos.versions", "kalos.normalize", "kalos.domains",
               "kalos.validation", "kalos.scale", "kalos.providers", "kalos.runtime")
    root = Path(dispense.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for mod in mods:
                if mod.startswith("kalos"):
                    assert mod.startswith(allowed), f"{path.name} imports {mod}"
    assert (root / "__init__.py").read_text() == (
        '"""Upload parsing and its guards, cell typing, instrument and plate readers."""\n')
