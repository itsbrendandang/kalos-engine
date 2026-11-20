"""LabChip peak tables: product peak by a declared size window (kalos.ingest.instruments.labchip).

The fixture tests/fixtures/lab_ingest/labchip_peaktable.csv is synthetic: it
copies the header layout of a LabChip GX Protein Express PeakTable export with
made-up wells, sample names, sizes and concentrations (a ladder well, markers,
system peaks, a non-detect well, a two-peaks-in-window well, a blank
concentration cell, a standard). Smaller cases are built inline.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from kalos.ingest import findings as ingest_findings
from kalos.ingest.instruments import labchip
from kalos.ingest.instruments.labchip import (
    canonical_well,
    parse_peak_table_text,
    read_peak_table,
)
from kalos.validation.report import Finding

FIX = Path(__file__).parent / "fixtures" / "lab_ingest"
PEAKS = FIX / "labchip_peaktable.csv"
HEADER = ("Well Label,Sample Name,Type,Size [kDa],Conc. (ng/ul),% Purity,Migration Time(sec) - Start,"
          "Migration Time(sec) - End,Corr. Area,Date_Time,Plate File,Height")
DT, PF = "2030\\01\\15 09:30:00", "C:\\LabChip\\Data\\demo_plate_01.gxd"
WINDOW = {"lo_kda": 27.0, "hi_kda": 33.0}


def checks(result) -> dict[str, Finding]:
    return {f.check: f for f in result.findings}


def by_well(result) -> dict[str, dict]:
    return {r["well"]: r for r in result.results.to_dict(orient="records")}


def _csv(*peaks: tuple, header: str = HEADER, marker: bool = True) -> str:
    """Rows of (well, sample, type, size, conc, purity[, plate_file]); an LM peak first per well."""
    lines = [header]
    seen: set[tuple[str, str]] = set()
    for p in peaks:
        well, name, typ, size, conc, purity = p[:6]
        pf = p[6] if len(p) > 6 else PF
        if marker and (pf, well) not in seen and not well.startswith("Ladder"):
            lines.append(f"{well},{name},LM,1.0,,,15.1,15.3,90.0,{DT},{pf},40.0")
        seen.add((pf, well))
        lines.append(f"{well},{name},{typ},{size},{conc},{purity},22.0,22.3,100.0,{DT},{pf},50.0")
    return "\n".join(lines) + "\n"


# -- the fixture ---------------------------------------------------------------


def test_fixture_every_peak_kept_with_a_status():
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    t = res.table
    n_lines = len(PEAKS.read_text().splitlines()) - 1
    assert len(t) == n_lines == 34
    assert list(t["source_line"]) == list(range(2, 36))
    assert t["status"].notna().all() and (t["reason"][t["status"] != "ok"] != "").all()
    # the ladder well and its peaks (LM, Xsys, L, one with a blank size) are kept, excluded
    ladder = t[t["well"] == "Ladder01"]
    assert len(ladder) == 8 and set(ladder["status"]) == {"ladder"}
    assert ladder["well_canonical"].isna().all()
    assert set(t.loc[t["peak_type"] == "LM", "status"]) == {"ladder", "marker", "standard"}
    assert set(t.loc[t["peak_type"] == "Xsys", "status"]) == {"ladder", "system_peak", "standard"}
    c = checks(res)
    assert c["rows_excluded"].severity == "info"
    assert c["rows_excluded"].detail["counts"] == {
        "ladder": 8, "marker": 5, "standard": 3, "system_peak": 5, "unassigned_type": 1}
    assert c["blank_cells_in_excluded_rows"].severity == "info"
    # results: sample wells only (no ladder, no standard), in file order
    assert list(res.results["well"]) == ["A01", "A02", "A03", "A04", "A05"]
    assert list(res.results["well_canonical"]) == ["A1", "A2", "A3", "A4", "A5"]


def test_fixture_product_values_units_and_window_on_every_row():
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    r = by_well(res)
    a1 = r["A01"]
    assert a1["status"] == "ok" and a1["rule_applied"] == "single_peak"
    assert a1["product_conc"] == pytest.approx(412.5) and a1["conc_unit"] == "ng/ul"
    assert a1["product_conc_g_per_l"] == pytest.approx(0.4125)
    assert a1["product_purity_pct"] == pytest.approx(90.2)  # % Purity is not converted
    assert a1["product_size_kda"] == pytest.approx(30.1) and a1["sample_name"] == "R01-1"
    assert a1["conversion"] == labchip.CONVERSION
    # the peak used is traceable to its row in the long table
    assert res.table.loc[a1["peak_rows"][0], "size_kda"] == pytest.approx(30.1)
    # a '?' peak outside the window and a 52 kDa peak do not disturb A02
    assert r["A02"]["product_conc"] == pytest.approx(356.0)
    # the window and rule are stored on every row and in meta
    assert (res.results["window_lo_kda"] == 27.0).all() and (res.results["window_hi_kda"] == 33.0).all()
    assert res.results["multi_peak_rule"].isna().all()
    assert res.meta["size_window_kda"] == {"lo": 27.0, "hi": 33.0}
    assert res.meta["window_includes_bounds"] is True and res.meta["multi_peak_rule"] is None
    conv = res.meta["conversion"]
    assert conv["factor"] == 1e-3 and conv["from_unit"] == "ng/ul" and conv["to_unit"] == "g/L"
    assert res.meta["parser"] == "labchip/1" and res.meta["encoding"] == "utf-8"
    # date and plate file are kept as provenance text, untouched
    assert set(res.table["date_time"]) == {DT} and set(res.table["plate_file"]) == {PF}
    assert a1["plate_file"] == PF and a1["date_time"] == DT


def test_non_detect_is_nan_never_zero():
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    a3 = by_well(res)["A03"]
    assert a3["status"] == "no_peak_in_window" and a3["non_detect"] is True
    assert math.isnan(a3["product_conc"]) and math.isnan(a3["product_conc_g_per_l"])
    assert (res.results["product_conc"] != 0).all()
    f = checks(res)["no_peak_in_window"]
    assert f.severity == "warning" and f.detail["wells"] == ["A03"] and "not as 0" in f.message
    # rows point into the long peak table (all of A03's peaks); result_rows into the results
    assert f.rows == (18, 19, 20, 21) and f.detail["result_rows"] == [2]
    # a peak just outside the window is reported for the record, not used
    near = checks(res)["peak_near_window"]
    assert near.severity == "info" and near.detail["peaks"] == ["A03:25.6 kDa"]
    assert res.meta["near_margin_frac"] == 0.10


def test_two_peaks_in_window_refused_without_a_rule():
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    a4 = by_well(res)["A04"]
    assert a4["status"] == "multiple_peaks_in_window" and math.isnan(a4["product_conc"])
    assert a4["non_detect"] is False and a4["sizes_in_window_kda"] == [28.4, 31.7]
    f = checks(res)["multiple_peaks_in_window"]
    assert f.severity == "error" and f.detail["wells"] == {"A04": [28.4, 31.7]}


def test_declared_rule_sum_and_largest():
    total = read_peak_table(PEAKS, **WINDOW, standards=["A06"], multi_peak_rule="sum")
    a4 = by_well(total)["A04"]
    assert a4["status"] == "ok" and a4["rule_applied"] == "sum"
    assert a4["product_conc"] == pytest.approx(120.7 + 180.2)
    assert a4["product_purity_pct"] == pytest.approx(38.5 + 57.6)
    assert math.isnan(a4["product_size_kda"])  # no single size for a sum
    assert total.meta["multi_peak_rule"] == "sum" and set(total.results["multi_peak_rule"]) == {"sum"}
    assert checks(total)["multi_peak_rule_applied"].detail["wells"] == {"A04": [28.4, 31.7]}
    assert "multiple_peaks_in_window" not in checks(total)
    # single-peak wells are unchanged by the rule
    assert by_well(total)["A01"]["rule_applied"] == "single_peak"
    big = read_peak_table(PEAKS, **WINDOW, standards=["A06"], multi_peak_rule="largest")
    a4 = by_well(big)["A04"]
    assert a4["product_conc"] == pytest.approx(180.2) and a4["product_size_kda"] == pytest.approx(31.7)
    assert a4["rule_applied"] == "largest" and a4["peak_rows"] == [25]


def test_unknown_rule_is_refused_and_ignored():
    res = read_peak_table(PEAKS, **WINDOW, multi_peak_rule="first")
    assert checks(res)["multi_peak_rule_invalid"].severity == "error"
    assert by_well(res)["A04"]["status"] == "multiple_peaks_in_window"
    assert res.meta["multi_peak_rule"] is None


def test_blank_conc_cell_reported_and_product_left_empty():
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    a5 = by_well(res)["A05"]
    assert a5["status"] == "product_conc_missing" and math.isnan(a5["product_conc"])
    c = checks(res)
    assert c["product_conc_missing"].severity == "error"
    assert c["conc_not_numeric"].rows == (29,)
    assert c["conc_not_numeric"].detail["where"] == ["A05 line 31: ''"]
    assert res.table.at[29, "status"] == "conc_missing" and res.table.at[29, "conc_raw"] == ""


def test_window_change_is_a_method_change():
    narrow = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    wide = read_peak_table(PEAKS, lo_kda=24.5, hi_kda=36.0, standards=["A06"])
    assert narrow.meta["size_window_kda"] != wide.meta["size_window_kda"]
    # the 25.6 kDa peak enters the wider window: A03 is no longer a non-detect
    a3 = by_well(wide)["A03"]
    assert a3["status"] == "ok" and a3["product_conc"] == pytest.approx(5.1)
    assert set(wide.results["window_lo_kda"]) == {24.5}


# -- refusals ------------------------------------------------------------------


def test_no_window_is_a_refusal():
    res = read_peak_table(PEAKS)
    f = checks(res)["size_window_not_declared"]
    assert f.severity == "error" and "lo_kda" in f.message
    assert res.results.empty and list(res.results.columns) == labchip._RESULT_COLUMNS
    assert len(res.table) == 34 and res.table["in_window"].isna().all()
    assert res.meta["size_window_kda"] is None


@pytest.mark.parametrize(("lo", "hi"), [(33.0, 27.0), (30.0, 30.0), (-1.0, 30.0),
                                        (float("nan"), 30.0), (27.0, float("inf")), ("a", 30.0)])
def test_invalid_window_is_a_refusal(lo, hi):
    res = read_peak_table(PEAKS, lo_kda=lo, hi_kda=hi)
    assert checks(res)["size_window_invalid"].severity == "error" and res.results.empty
    json.dumps(res.to_dict(), allow_nan=False)


def test_only_one_bound_is_a_refusal():
    res = read_peak_table(PEAKS, lo_kda=27.0)
    assert "size_window_not_declared" in checks(res) and res.results.empty


@pytest.mark.parametrize("unit", ["ng/µl", "ng/μL", "NG/UL", "ng / ul"])
def test_ng_per_ul_spellings_convert(unit):
    text = _csv(("A1", "s", "", "30.0", "250.0", "90")).replace("(ng/ul)", f"({unit})")
    res = parse_peak_table_text(text, **WINDOW)
    a1 = by_well(res)["A1"]
    assert a1["product_conc_g_per_l"] == pytest.approx(0.25) and a1["conc_unit"] == unit
    assert "conc_unit_not_ng_per_ul" not in checks(res)


@pytest.mark.parametrize("unit", ["ug/ml", "mg/L", "ng"])
def test_other_conc_unit_refuses_conversion(unit):
    text = _csv(("A1", "s", "", "30.0", "250.0", "90")).replace("(ng/ul)", f"({unit})")
    res = parse_peak_table_text(text, **WINDOW)
    f = checks(res)["conc_unit_not_ng_per_ul"]
    assert f.severity == "error" and f.detail["unit"] == unit
    a1 = by_well(res)["A1"]
    assert a1["product_conc"] == pytest.approx(250.0) and a1["conc_unit"] == unit  # raw kept
    assert math.isnan(a1["product_conc_g_per_l"]) and a1["conversion"] is None
    assert res.meta["conversion"] is None
    assert res.meta["role_suggestions"]["product_conc"] == "outcome"
    assert "product_conc_g_per_l" not in res.meta["role_suggestions"]


def test_size_not_in_kda_is_a_refusal():
    text = _csv(("A1", "s", "", "30.0", "250.0", "90")).replace("Size [kDa]", "Size [bp]")
    res = parse_peak_table_text(text, **WINDOW)
    assert checks(res)["size_unit_not_kda"].severity == "error"
    assert res.results.empty and res.table["in_window"].isna().all()


def test_required_column_missing_is_a_refusal():
    text = "Well Label,Sample Name,Size [kDa]\nA1,s,30.0\nA2,t,31.0\n"
    res = parse_peak_table_text(text, **WINDOW)
    f = checks(res)["required_column_missing"]
    assert f.severity == "error" and f.detail["missing"] == ["peak_type", "conc"]
    assert len(res.table) == 2 and set(res.table["status"]) == {"not_read"} and res.results.empty
    empty = parse_peak_table_text("", **WINDOW)
    assert "required_column_missing" in checks(empty)


def test_header_only_file():
    res = parse_peak_table_text(HEADER + "\n", **WINDOW)
    assert checks(res)["no_peaks"].severity == "error" and res.results.empty


# -- cell and well checks ------------------------------------------------------


def test_size_not_numeric_refuses_the_well():
    text = _csv(("A1", "s", "", "30.0", "250.0", "90"), ("A1", "s", "", "n/a", "12.0", "10"),
                ("A2", "t", "", "", "5.0", "100"))
    res = parse_peak_table_text(text, **WINDOW)
    f = checks(res)["size_not_numeric"]
    assert f.severity == "warning" and f.detail["n_rows_affected"] == 2
    assert f.detail["where"] == ["A1 line 4: 'n/a'", "A2 line 6: ''"]
    r = by_well(res)
    assert r["A1"]["status"] == "peak_size_missing" and math.isnan(r["A1"]["product_conc"])
    assert r["A2"]["status"] == "peak_size_missing" and r["A2"]["non_detect"] is False
    assert checks(res)["peak_size_missing"].detail["wells"] == ["A1", "A2"]
    assert res.table.at[2, "size_raw"] == "n/a" and res.table.at[2, "status"] == "size_not_numeric"


def test_conc_text_and_purity_blank_reported():
    text = _csv(("A1", "s", "", "30.0", "<LOQ", "90"), ("A2", "t", "", "29.0", "40.0", ""))
    res = parse_peak_table_text(text, **WINDOW)
    c = checks(res)
    assert c["conc_not_numeric"].detail["where"] == ["A1 line 3: '<LOQ'"]
    assert by_well(res)["A1"]["status"] == "product_conc_missing"
    assert c["purity_not_numeric"].rows == (3,)
    a2 = by_well(res)["A2"]
    assert a2["product_conc"] == pytest.approx(40.0) and math.isnan(a2["product_purity_pct"])


def test_duplicate_well_size_rows_reported_and_never_summed():
    text = _csv(("A1", "s", "", "30.0", "250.0", "90"), ("A1", "s", "", "30.0", "250.0", "90"))
    res = parse_peak_table_text(text, **WINDOW, multi_peak_rule="sum")
    c = checks(res)
    assert c["duplicate_peak"].detail["pairs"] == ["A1:30 kDa"] and c["duplicate_peak"].rows == (1, 2)
    a1 = by_well(res)["A1"]
    assert a1["status"] == "duplicate_peak_in_window" and math.isnan(a1["product_conc"])
    assert c["duplicate_peak_in_window"].severity == "error"
    assert len(res.table) == 3


def test_unquantified_peak_in_window_is_not_a_non_detect():
    text = _csv(("A1", "s", "?", "30.0", "", ""), ("A1", "s", "", "60.0", "20.0", "100"))
    res = parse_peak_table_text(text, **WINDOW)
    a1 = by_well(res)["A1"]
    assert a1["status"] == "unquantified_peak_in_window" and a1["non_detect"] is False
    assert a1["peak_rows"] == [1]
    assert checks(res)["unquantified_peak_in_window"].severity == "warning"
    assert "no_peak_in_window" not in checks(res)


def test_largest_tie_refused():
    text = _csv(("A1", "s", "", "29.0", "100.0", "50"), ("A1", "s", "", "31.0", "100.0", "50"))
    res = parse_peak_table_text(text, **WINDOW, multi_peak_rule="largest")
    assert by_well(res)["A1"]["status"] == "largest_peak_tied"
    assert checks(res)["largest_peak_tied"].severity == "error"


def test_window_bounds_are_inclusive():
    text = _csv(("A1", "s", "", "27.0", "10.0", "100"), ("A2", "t", "", "33.0", "20.0", "100"))
    r = by_well(parse_peak_table_text(text, **WINDOW))
    assert r["A1"]["product_conc"] == 10.0 and r["A2"]["product_conc"] == 20.0


def test_near_margin_declared_or_invalid():
    text = _csv(("A1", "s", "", "34.0", "10.0", "100"))
    assert checks(parse_peak_table_text(text, **WINDOW))["peak_near_window"].detail["peaks"] == [
        "A1:34 kDa"]
    assert "peak_near_window" not in checks(parse_peak_table_text(text, **WINDOW, near_margin_frac=0.0))
    bad = parse_peak_table_text(text, **WINDOW, near_margin_frac=1.5)
    assert checks(bad)["near_margin_invalid"].severity == "warning"
    assert by_well(bad)["A1"]["status"] == "no_peak_in_window"


def test_lower_marker_missing_warned():
    text = _csv(("A1", "s", "", "30.0", "10.0", "100"), marker=False)
    res = parse_peak_table_text(text, **WINDOW)
    assert checks(res)["lower_marker_missing"].detail["wells"] == ["A1"]


def test_standards_declared_by_name_or_well_and_suspected():
    suspect = read_peak_table(PEAKS, **WINDOW)
    f = checks(suspect)["possible_standard_not_declared"]
    assert f.severity == "warning" and f.detail["wells"] == ["A06 (BSA std 250 ng/uL)"]
    assert "A06" in set(suspect.results["well"])  # treated as a sample until declared
    for decl in (["BSA std 250 ng/uL"], ["A6"], ["A06"]):
        res = read_peak_table(PEAKS, **WINDOW, standards=decl)
        assert "A06" not in set(res.results["well"])
        assert set(res.table.loc[res.table["well"] == "A06", "status"]) == {"standard"}
        assert "possible_standard_not_declared" not in checks(res)
    missing = read_peak_table(PEAKS, **WINDOW, standards=["A06", "QC-std"])
    assert checks(missing)["standard_not_found"].detail["standards"] == ["QC-std"]


def test_invalid_well_label_reported_and_ladder_label_is_not():
    text = _csv(("Ladder02", "ladder", "L", "20.0", "", ""), ("Z99", "s", "", "30.0", "10.0", "100"),
                ("A1", "t", "", "30.0", "11.0", "100"))
    res = parse_peak_table_text(text, **WINDOW)
    f = checks(res)["well_label_invalid"]
    assert f.severity == "error" and f.detail["labels"] == ["Z99"]
    assert list(res.results["well"]) == ["A1"] and len(res.table) == 5


def test_plate_files_kept_apart_and_names_checked():
    text = _csv(("A1", "s", "", "30.0", "10.0", "100", "run1.gxd"),
                ("A1", "s2", "", "30.0", "20.0", "100", "run2.gxd"),
                ("A2", "t", "", "29.0", "5.0", "60", "run1.gxd"),
                ("A2", "t-renamed", "", "50.0", "5.0", "40", "run1.gxd"))
    res = parse_peak_table_text(text, **WINDOW)
    c = checks(res)
    assert c["multiple_plate_files"].detail["n_plate_files"] == 2
    assert list(zip(res.results["plate_file"], res.results["well"], res.results["product_conc"],
                    strict=True)) == [("run1.gxd", "A1", 10.0), ("run2.gxd", "A1", 20.0),
                                      ("run1.gxd", "A2", 5.0)]
    assert c["sample_name_varies_in_well"].detail["wells"] == ["A2: ['t', 't-renamed']"]
    assert "duplicate_peak" not in c  # same well and size on two plates is not a duplicate


def test_ragged_rows_kept():
    # a short row (no Height) and a long row (one extra cell)
    text = HEADER + f"\nA1,s,,30.0,10.0,100,22.0,22.3,100.0,{DT},{PF}\nA1,s,LM,1.0,,,15.1,15.3,90.0,{DT},{PF},40.0,extra\n"
    res = parse_peak_table_text(text, **WINDOW)
    f = checks(res)["row_length_mismatch"]
    assert f.rows == (0, 1) and f.detail["lines"] == [2, 3]
    assert by_well(res)["A1"]["product_conc"] == 10.0


def test_tab_delimited_utf16_file(tmp_path):
    path = tmp_path / "peaks.txt"
    text = _csv(("A01", "s", "", "30.0", "250.0", "90")).replace(",", "\t")
    path.write_bytes(text.encode("utf-16"))
    res = read_peak_table(path, **WINDOW)
    assert res.meta["encoding"] == "utf-16" and res.meta["delimiter"] == "\t"
    assert by_well(res)["A01"]["product_conc_g_per_l"] == pytest.approx(0.25)


@pytest.mark.parametrize(("label", "want"), [
    ("A01", "A1"), ("a1", "A1"), (" H 12 ", "H12"), ("P24", "P24"), ("Q1", None), ("A25", None),
    ("A0", None), ("Ladder01", None), ("", None), (None, None),
])
def test_canonical_well(label, want):
    assert canonical_well(label) == want


# -- contracts -----------------------------------------------------------------


def test_roles_never_setting_and_no_area_or_size_suggestion():
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    roles = res.meta["role_suggestions"]
    assert roles == {"sample_name": "id", "well": "id", "well_canonical": "id",
                     "product_purity_pct": "outcome", "product_conc_g_per_l": "outcome",
                     "product_conc": "derived_from_target"}
    assert "setting" not in roles.values()
    assert not {"product_size_kda", "corr_area", "size_kda", "migration_start_s"} & set(roles)


def test_results_joinable_by_position():
    """The columns plates.join_by_position needs: well, plus a sample name to compare."""
    res = read_peak_table(PEAKS, **WINDOW, standards=["A06"])
    assert {"well", "well_canonical", "sample_name"} <= set(res.results.columns)
    assert "plate" not in res.results.columns  # the plate id is declared at join time, not guessed
    assert list(res.results["sample_name"]) == ["R01-1", "R01-2", "R02-1", "R02-2", "R03-1"]


def test_deterministic():
    a = read_peak_table(PEAKS, **WINDOW, standards=["A06"], multi_peak_rule="sum")
    b = read_peak_table(PEAKS, **WINDOW, standards=["A06"], multi_peak_rule="sum")
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize("kwargs", [{}, WINDOW, {**WINDOW, "multi_peak_rule": "sum"}])
def test_to_dict_is_json_safe(kwargs):
    res = read_peak_table(PEAKS, **kwargs)
    out = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert len(out["table"]) == 34 and isinstance(out["findings"], list)
    if kwargs:
        a3 = next(r for r in out["results"] if r["well"] == "A03")
        assert a3["product_conc"] is None and a3["non_detect"] is True


def test_findings_well_formed():
    res = read_peak_table(PEAKS, **WINDOW)
    assert len(res.findings) >= 6
    for f in res.findings:
        assert f.severity in ("error", "warning", "info")
        assert f.check.replace("_", "").isalpha() and f.check == f.check.lower()
        assert all(type(r) is int and 0 <= r < len(res.table) for r in f.rows)


@pytest.mark.parametrize("kwargs", [{}, WINDOW, {**WINDOW, "multi_peak_rule": "sum"}])
def test_every_finding_is_a_validation_finding(kwargs):
    res = read_peak_table(PEAKS, **kwargs)
    assert res.findings and all(type(f) is Finding for f in res.findings)
    assert ingest_findings.Finding is Finding
    f = ingest_findings.finding("x", "warning", "m", column="c", rows=range(25), where="A1")
    assert f.rows == tuple(range(20)) and f.detail == {"where": "A1", "n_rows_affected": 25}


def test_messages_say_what_to_do_and_what_was_kept():
    """Plain language: each message names an action or says nothing was changed or filled."""
    seen: dict[str, Finding] = {}
    for kwargs in ({}, WINDOW, {**WINDOW, "multi_peak_rule": "sum"}, {"lo_kda": 3, "hi_kda": 1}):
        for f in read_peak_table(PEAKS, **kwargs).findings:
            seen.setdefault(f.check, f)
    for f in seen.values():
        assert "calibrated" not in f.message.lower()
        assert any(k in f.message for k in ("nothing was", "Nothing was", "kept")), f.check


def test_reserved_check_codes_not_used():
    import inspect

    src = inspect.getsource(labchip)
    for code in ("outliers", "physical_bounds", "duplicate_ids", "result_duplicate_entry",
                 "target_leak", "derived_output"):
        assert f'"{code}"' not in src


def test_ingest_imports_stay_in_layer():
    """kalos/ingest may import util, data, config, core, libs, runtime and itself (layer order
    in scripts/import_layers.py; libs = normalize, domains, validation, scale, providers), never
    analysis, campaigns, portal, storage, jobs, assist or api."""
    allowed = ("kalos.ingest", "kalos.util", "kalos.data", "kalos.config", "kalos.core",
               "kalos.normalize", "kalos.domains", "kalos.validation", "kalos.scale",
               "kalos.providers", "kalos.runtime")
    root = Path(labchip.__file__).parents[1]
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
    assert (root / "instruments" / "__init__.py").read_text() == (
        '"""Instrument export readers: one parser per format, each a long table plus findings."""\n')


def test_pandas_frame_types():
    res = read_peak_table(PEAKS, **WINDOW)
    assert isinstance(res.table, pd.DataFrame) and isinstance(res.results, pd.DataFrame)
    assert res.results["product_conc"].dtype == float
