"""Plate grids, plate maps and the join by position (kalos.ingest.plates).

Fixtures in tests/fixtures/lab_ingest/ are synthetic: they copy the layout of
lab exports (a grid of column numbers 1..n and row letters; a two-level plate
map with culture and analytics plate/position columns) with made-up values.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from kalos.ingest import findings as ingest_findings
from kalos.ingest import plates
from kalos.ingest.plates import (
    MapLevel,
    PlateMapColumns,
    join_by_position,
    normalize_well,
    parse_grid_text,
    parse_plate_map,
    read_grid,
    read_plate_map,
)
from kalos.validation.report import Finding

FIX = Path(__file__).parent / "fixtures" / "lab_ingest"


def checks(result) -> dict[str, Finding]:
    return {f.check: f for f in result.findings}


# -- wells -------------------------------------------------------------------


@pytest.mark.parametrize(("label", "fmt", "want"), [
    ("A1", None, "A1"), ("A01", 96, "A1"), (" b 07 ", 96, "B7"), ("H12", 96, "H12"),
    ("I1", 96, None), ("A13", 96, None), ("D6", 24, "D6"), ("E1", 24, None), ("A7", 24, None),
    ("P24", 384, "P24"), ("Q1", 384, None), ("A0", None, None), ("", None, None),
    (None, None, None), (float("nan"), None, None), ("well 3", None, None), ("AA1", None, None),
])
def test_normalize_well(label, fmt, want):
    assert normalize_well(label, fmt) == want


def test_wells_row_major():
    assert plates.wells(24)[:7] == ["A1", "A2", "A3", "A4", "A5", "A6", "B1"]
    assert len(plates.wells(384)) == 384


# -- grids -------------------------------------------------------------------


def test_grid_96_every_well_kept_and_reported():
    res = read_grid(FIX / "plate_grid_96.tsv", measurement="OD600", plate="AP-1")
    t = res.table
    assert len(t) == 96 and list(t["well"][:2]) == ["A1", "A2"]
    assert set(t["plate_format"]) == {96} and set(t["measurement"]) == {"OD600"}
    assert t["unit"].isna().all()  # the unit is never guessed
    c = checks(res)
    # the overflow cell is kept with its raw text and reported by well
    over = t[t["well"] == "B5"].iloc[0]
    assert over["status"] == "not_numeric" and over["raw"] == "OVER" and pd.isna(over["value"])
    assert c["grid_value_not_numeric"].detail["wells"] == ["B5"]
    assert c["grid_value_not_numeric"].severity == "warning"
    # empty wells are kept, never filled
    assert (t["status"] == "empty").sum() == 44
    assert c["grid_empty_wells"].detail["n_rows_affected"] == 44
    assert len(c["grid_empty_wells"].rows) == ingest_findings.ROW_CAP
    assert t.loc[t["well"] == "A1", "value"].iloc[0] == pytest.approx(0.7210)
    assert res.meta["encoding"] == "utf-8" and res.meta["delimiter"] == "\t"
    assert res.meta["role_suggestions"] == {"plate": "batch_group", "well": "id"}


def test_grid_24_comma():
    res = read_grid(FIX / "plate_grid_24.csv", measurement="pH")
    assert len(res.table) == 24 and set(res.table["plate_format"]) == {24}
    assert res.table["status"].eq("ok").all() and not res.findings


def _grid(n_rows: int, n_cols: int, cell=lambda r, c: f"{r + c / 100:.2f}", sep="\t",
          corner="") -> str:
    head = sep.join([corner] + [str(c) for c in range(1, n_cols + 1)])
    body = [sep.join(["ABCDEFGHIJKLMNOP"[r]] + [cell(r, c) for c in range(1, n_cols + 1)])
            for r in range(n_rows)]
    return "\n".join([head, *body]) + "\n"


def test_grid_384_and_preamble_lines():
    text = "Plate reader export\nMode\tAbsorbance\n\n" + _grid(16, 24, corner="<>")
    res = parse_grid_text(text, measurement="A600")
    assert len(res.table) == 384 and set(res.table["plate_format"]) == {384}
    assert res.table["well"].iloc[-1] == "P24" and res.table["source_line"].iloc[0] == 5
    assert not res.findings


def test_grid_declared_format_mismatch_and_incomplete():
    text = _grid(5, 12)  # 12 columns = 96-well, but only rows A-E
    res = parse_grid_text(text, measurement="OD600", plate_format=24)
    c = checks(res)
    assert c["plate_format_mismatch"].severity == "error"
    assert c["grid_incomplete"].detail == {"block": 1, "line": 1, "rows_expected": 8, "rows_found": 5}
    assert len(res.table) == 60  # the rows that are there are kept


def test_no_grid_is_an_error_with_an_action():
    res = parse_grid_text("Well,Value\nA1,0.5\n", measurement="OD600")
    assert res.table.empty and checks(res)["no_grid_found"].severity == "error"


def test_decimal_comma_hint_then_read():
    text = _grid(4, 6, cell=lambda r, c: f"{r},{c}5", sep=";")
    res = parse_grid_text(text, measurement="OD600")
    f = checks(res)["grid_value_not_numeric"]
    assert "decimal=','" in f.message and f.detail["n_rows_affected"] == 24
    ok = parse_grid_text(text, measurement="OD600", decimal=",")
    assert ok.table["value"].iloc[1] == pytest.approx(0.25) and not ok.findings


def test_two_grids_keep_their_blocks_and_extra_cells_reported():
    lines = (_grid(4, 6) + "\n" + _grid(4, 6)).splitlines()
    lines[2] += "\t9.99"  # a value beyond column 6 on row B of grid 1
    res = parse_grid_text("\n".join(lines), measurement="OD600")
    assert sorted(res.table["block"].unique()) == [1, 2] and len(res.table) == 48
    c = checks(res)
    assert c["multiple_grids"].detail["n_blocks"] == 2
    assert c["grid_extra_cells"].detail["where"] == ["B (line 3)"]


# -- plate maps --------------------------------------------------------------


def test_plate_map_two_levels_detected():
    res = read_plate_map(FIX / "plate_map_two_level.tsv")
    t = res.table
    assert len(t) == 48 and not res.findings
    assert [lv["name"] for lv in res.meta["levels"]] == ["culture", "analytics"]
    assert [lv["plate_format"] for lv in res.meta["levels"]] == [24, 96]
    # "A01" in the map is the same well as "A1"
    assert t.loc[t["sample"] == "R08-2", "culture_well"].iloc[0] == "A1"
    assert list(t.columns) == ["sample", "culture_plate", "culture_well", "analytics_plate",
                               "analytics_well", "Media condition", "map_row"]
    assert res.meta["role_suggestions"] == {
        "sample": "id", "culture_plate": "batch_group", "analytics_plate": "batch_group",
        "culture_well": "id", "analytics_well": "id"}
    # text stays text: "1" is a media label, not a number
    assert t["Media condition"].iloc[1] == "1"


def _map(rows, headers=("Sample", "Culture plate", "Culture well", "Analytics plate",
                        "Analytics well")):
    return pd.DataFrame(rows, columns=list(headers))


def test_plate_map_duplicates_replicates_and_bad_labels():
    frame = _map([
        ["s1", "C1", "A1", "R1", "A1"],
        ["s1", "C1", "A1", "R1", "A2"],   # one culture read in two analytics wells: repeat reads
        ["s2", "C1", "A2", "R1", "A2"],   # two samples in analytics well A2: conflict
        ["s3", "C1", "Z9", "R1", "A3"],   # not a well
        ["s4", "C1", "", "R1", "A4"],     # no culture well
    ])
    res = parse_plate_map(frame)
    c = checks(res)
    assert c["technical_replicates"].severity == "info"
    assert c["technical_replicates"].detail["positions"] == ["C1:A1"]
    assert c["plate_map_duplicate_position"].detail["positions"] == ["R1:A2"]
    assert c["plate_map_duplicate_position"].rows == (1, 2)
    assert c["well_label_invalid"].detail["labels"] == ["Z9"]
    assert c["plate_map_missing_position"].rows == (4,)
    assert len(res.table) == 5  # nothing dropped


def test_plate_map_unclear_columns_refused():
    frame = pd.DataFrame({"Sample": ["a"], "Plate A": ["p"], "Plate B": ["q"], "Well": ["A1"]})
    res = parse_plate_map(frame)
    assert checks(res)["plate_map_columns_unclear"].severity == "error"
    declared = PlateMapColumns("Sample", (MapLevel("analytics", "Plate B", "Well", 96),))
    ok = parse_plate_map(frame, declared)
    assert ok.table[["analytics_plate", "analytics_well"]].iloc[0].tolist() == ["q", "A1"]


# -- join by position --------------------------------------------------------


def test_join_grid_to_map_mismatch_report():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    grid = read_grid(FIX / "plate_grid_96.tsv", measurement="OD600", plate="AP-1")
    res = join_by_position(pmap, grid.table, level="analytics")
    t, r = res.table, res.report
    assert r["matched"] == 48
    # rows E-H are not in the map: 44 empty wells, and column 12 was read on each row anyway
    c = checks(res)
    assert c["unmapped_empty_wells"].detail["n_rows_affected"] == 44
    assert r["measurement_only"] == [f"AP-1:{w}" for w in plates.wells(96)[48:]]
    assert c["measurement_position_not_in_map"].detail["positions"] == [
        "AP-1:E12", "AP-1:F12", "AP-1:G12", "AP-1:H12"]
    assert r["map_only"] == [] and "map_position_not_measured" not in c
    # every measurement row and every map row appears exactly once
    assert sorted(t["measurement_row"].dropna().astype(int)) == list(range(96))
    assert sorted(t["map_row"].dropna().astype(int)) == list(range(48))
    row = t[t["sample"] == "R08-2"].iloc[0]
    assert (row["analytics_well"], row["culture_plate"], row["culture_well"]) == ("C1", "CP-2", "A1")
    assert row["value"] == pytest.approx(float(grid.table.set_index("well").at["C1", "value"]))
    # the overflow well stays joined to its sample, with no value
    b5 = t[t["analytics_well"] == "B5"].iloc[0]
    assert b5["match"] == "matched" and b5["status"] == "not_numeric" and pd.isna(b5["value"])


def _instrument(pmap, swap=("B3", "B4")):
    """A long instrument table: well labels as 'B03', sample names from the map with two swapped."""
    m = pmap.table.set_index("analytics_well")["sample"].to_dict()
    names = dict(m)
    names[swap[0]], names[swap[1]] = m[swap[1]], m[swap[0]]
    wells_ = list(m)
    return pd.DataFrame({
        "well": [f"{w[0]}{int(w[1:]):02d}" for w in wells_],
        "Sample Name": [names[w] for w in wells_],
        "value": [float(i) for i in range(len(wells_))],
        "measurement": "product",
    })


def test_well_name_mismatch_caught_and_joined_by_position():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    meas = _instrument(pmap)
    res = join_by_position(pmap, meas, level="analytics", sample_column="Sample Name")
    c = checks(res)
    f = c["well_name_mismatch"]
    assert f.severity == "warning" and f.detail["n_rows_affected"] == 2
    b3 = next(m for m in f.detail["mismatches"] if m["position"] == "AP-1:B3")
    assert b3["map_sample"] != b3["instrument_sample"]
    assert b3["instrument_name_is_at"] == "AP-1:B4"
    # the plate was taken from the map (it has one analytics plate) and said so
    assert c["plate_from_map"].detail["plate"] == "AP-1"
    # joined by position: the value at B3 stays with the map's B3 sample
    t = res.table
    assert t.loc[t["analytics_well"] == "B3", "sample"].iloc[0] == b3["map_sample"]
    assert res.report["matched"] == 48


def test_no_mismatch_when_names_agree():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    res = join_by_position(pmap, _instrument(pmap, swap=("B3", "B3")), level="analytics",
                           sample_column="Sample Name")
    assert "well_name_mismatch" not in checks(res) and res.report["name_mismatches"] == []


def test_join_culture_level_with_two_plates_needs_a_plate():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    grid = read_grid(FIX / "plate_grid_24.csv", measurement="pH")
    refused = join_by_position(pmap, grid.table, level="culture")
    assert checks(refused)["plate_not_declared"].severity == "error"
    assert len(refused.table) == 24 and set(refused.table["match"]) == {"not_joined"}
    res = join_by_position(pmap, grid.table, level="culture", plate="CP-2")
    assert res.report["matched"] == 24 and res.report["map_plates_not_measured"] == ["CP-1"]
    assert checks(res)["map_plates_not_measured"].severity == "info"
    assert len(res.table) == 24


def test_join_reports_unmeasured_unknown_plate_duplicates_and_bad_wells():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    meas = pd.DataFrame({
        "plate": ["AP-1", "AP-1", "AP-1", "ap-1", "AP-1"],
        "well": ["A1", "A1", "A2", "A3", "K1"],
        "value": [1.0, 1.1, 2.0, 3.0, 4.0],
        "measurement": "titer",
    })
    res = join_by_position(pmap, meas, level="analytics")
    c, r = checks(res), res.report
    assert r["multiply_matched"] == ["AP-1:A1"]
    assert c["plate_not_in_map"].detail["plates"] == ["ap-1"]  # ids compared exactly
    assert c["well_label_invalid"].detail["labels"] == ["K1"]
    assert c["map_position_not_measured"].detail["n_rows_affected"] == 46
    assert sorted(res.table["measurement_row"].dropna().astype(int)) == [0, 1, 2, 3, 4]
    assert len(res.table) == 2 + 1 + 46 + 2  # A1 twice, A2, 46 unmeasured, ap-1:A3, K1


def test_join_unknown_level_refused():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    res = join_by_position(pmap, pd.DataFrame({"well": ["A1"]}), level="harvest")
    assert checks(res)["join_level_unknown"].detail["levels"] == ["analytics", "culture"]


# -- contracts ---------------------------------------------------------------


def test_parsers_return_the_validation_finding_type():
    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    res = join_by_position(pmap, _instrument(pmap), level="analytics", sample_column="Sample Name")
    assert res.findings and all(type(f) is Finding for f in res.findings)
    f = ingest_findings.finding("x", "warning", "m", column="c", rows=range(25), where="A1")
    assert f.rows == tuple(range(20)) and f.detail == {"where": "A1", "n_rows_affected": 25}


def test_results_are_json_safe():
    import json

    pmap = read_plate_map(FIX / "plate_map_two_level.tsv")
    res = join_by_position(pmap, _instrument(pmap), level="analytics", sample_column="Sample Name")
    json.dumps([f.to_dict() for f in res.findings])
    json.dumps(res.report)


def test_ingest_imports_stay_in_layer():
    """kalos/ingest may import util, data, config, core, libs, runtime and itself (layer order
    in scripts/import_layers.py; libs = normalize, domains, validation, scale, providers), never
    analysis, campaigns, portal, storage, jobs, assist or api."""
    allowed = ("kalos.ingest", "kalos.util", "kalos.data", "kalos.config", "kalos.core",
               "kalos.normalize", "kalos.domains", "kalos.validation", "kalos.scale",
               "kalos.providers", "kalos.runtime")
    root = Path(plates.__file__).parent
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
