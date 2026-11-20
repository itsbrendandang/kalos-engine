"""S.NEST DO/pH exports: long table, as-run conditions, per-well summaries (kalos.ingest.instruments.snest).

Fixtures in tests/fixtures/lab_ingest/snest_* are synthetic: they copy the
layout of the instrument's CSV export (metadata block, plate layout, pauses,
processing options, data block) with a made-up experiment, times and values.
In them, B2 has a gap (two missing DO readings, one missing pH reading), C3
dips below 30 for a while, D4 has only two valid DO readings, and A1/B1/C1/D1
are blanks listed as baseline reference wells.
"""
from __future__ import annotations

import ast
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from kalos.ingest import findings as ingest_findings
from kalos.ingest.instruments import snest
from kalos.ingest.instruments.snest import (
    MIN_VALID_POINTS,
    combine,
    parse_snest_text,
    read_snest,
    read_snest_csv,
    time_below,
)
from kalos.validation.report import Finding

FIX = Path(__file__).parent / "fixtures" / "lab_ingest"
DO_CSV, PH_CSV, DO_TXT = FIX / "snest_DO.csv", FIX / "snest_pH.csv", FIX / "snest_DO.txt"
ROLES = {"setting", "outcome", "covariate", "batch_group", "clone", "id", "derived_from_target"}


def checks(result) -> dict[str, Finding]:
    return {f.check: f for f in result.findings}


def well_row(summary: pd.DataFrame, well: str) -> pd.Series:
    return summary[summary["well_canonical"] == well].iloc[0]


@pytest.fixture(scope="module")
def plate():
    return read_snest([DO_CSV, PH_CSV], do_threshold=30.0)


def export(
    channel: str = "DO",
    wells: tuple[str, ...] = ("A1", "A2"),
    rows: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("2031/3/14 09:05", "5.0", ("90", "80")),
        ("2031/3/14 09:35", "35.0", ("85", "70")),
        ("2031/3/14 10:05", "65.0", ("80", "60")),
    ),
    *,
    fmt: str = "24-well plate",
    title: str | None = None,
    extra: str = "",
    layout: bool = True,
    name: str | None = "Synthetic_PlateZ",
) -> str:
    """A small export in the instrument's CSV layout."""
    lines = ["S.NEST Experiment"]
    if name is not None:
        lines.append(f"Experiment name:,{name}")
    lines += ["Start time:,2031-03-14 09:00:00", "", "***General setup***", f"Plate format:,{fmt}",
              "Working volume:,800,μL", "Temperature:,30,°C", "CO2:,0,%", ""]
    if extra:
        lines += extra.splitlines() + [""]
    if layout:
        lines += ["***Plate layout***", "Well ID,Group,Note"]
        lines += [f"{w},Medium A,{i}" for i, w in enumerate(wells)] + [""]
    lines += [title or f"***S.NEST {channel} data***", "",
              "Time,Running time(min)," + ",".join(f"{w}_Medium A_{i}" for i, w in enumerate(wells)) + ","]
    lines += [f"{clk},{run}," + ",".join(vals) for clk, run, vals in rows]
    return "\n".join(lines) + "\n"


# -- reading the export --------------------------------------------------------


def test_every_sample_is_kept_in_a_long_table(plate):
    t = plate.table
    assert len(t) == 24 * 13 * 2  # 24 wells x 13 times x 2 channels
    assert set(t["channel"]) == {"DO", "pH"}
    assert list(t.columns) == snest.TABLE_COLUMNS
    assert set(t["plate"]) == {"Synthetic_PlateA"}
    assert set(t["status"]) == {"ok", "missing"}
    a1 = t[(t["well"] == "A1") & (t["channel"] == "DO")]
    assert a1["time_h"].iloc[0] == pytest.approx(5 / 60)  # running time 5.0 min
    assert a1["timestamp"].iloc[0] == "2031/3/14 09:05"
    assert a1["value"].iloc[0] == pytest.approx(99.0) and a1["raw"].iloc[0] == "99.00"
    assert a1["source_line"].iloc[0] == 76
    assert "Running time converted to h" in plate.meta["time_basis"][0]


def test_units_come_from_the_file_or_the_user_never_guessed(plate):
    t = plate.table
    assert t.loc[t["channel"] == "DO", "unit"].isna().all()
    assert set(t.loc[t["channel"] == "pH", "unit"]) == {"pH"}
    f = checks(plate)["unit_not_stated"]
    assert f.severity == "warning" and "not assumed" in f.message and f.detail["source"] == "snest_DO.csv"
    declared = read_snest_csv(DO_CSV, do_unit="% air saturation")
    assert set(declared.table["unit"]) == {"% air saturation"} and declared.meta["unit_source"] == "declared"
    assert "unit_not_stated" not in checks(declared)
    in_file = parse_snest_text(export(title="***S.NEST DO data (% air saturation)***"), do_unit="mg/L")
    assert set(in_file.table["unit"]) == {"% air saturation"} and in_file.meta["unit_source"] == "file"
    assert checks(in_file)["unit_conflict"].severity == "error"


def test_metadata_block_is_kept_as_run_with_units_as_written(plate):
    run = plate.meta["plates"]["Synthetic_PlateA"]
    assert run["as_run"]["working_volume"] == {"key": "Working volume", "value": "800", "number": 800.0,
                                               "unit": "μL"}
    assert run["as_run"]["temperature"]["unit"] == "°C" and run["as_run"]["co2"]["unit"] == "%"
    assert run["as_run"]["mixing"]["value"] == "ON"
    assert run["as_run"]["sampling_interval"] == {"key": "Sampling rate", "value": "30", "number": 30.0,
                                                  "unit": "mins"}
    assert "not set points" in run["as_run_note"]
    # unknown keys are kept verbatim, with their section
    assert run["conditions"]["Plate brand"]["value"] == "ExampleBrand"
    assert run["conditions"]["Plate brand"]["section"] == "Sensing setup"
    assert run["conditions"]["Plate layout ID"]["value"] == "0 (24-well DO/pH)"
    assert run["start_time_recorded"] == "2031-03-14 09:00:00" and run["plate_format"] == 24
    assert run["pauses"][0]["from"] == "2031/3/14 14:00"
    one = read_snest_csv(DO_CSV)
    assert one.meta["layout"]["C3"] == {"well": "C3", "group": "Medium C", "note": "5"}
    assert one.meta["encoding"] == "utf-8" and one.meta["title"] == "S.NEST Experiment"


def test_gap_is_kept_and_reported_by_well_and_time(plate):
    t = plate.table
    b2 = t[(t["well"] == "B2") & (t["channel"] == "DO")]
    gap = b2[b2["status"] == "missing"]
    assert len(gap) == 2 and gap["value"].isna().all() and set(gap["raw"]) == {"N/A"}
    do_missing = next(f for f in plate.findings
                      if f.check == "reading_missing" and f.detail["source"] == "snest_DO.csv")
    assert do_missing.detail["where"][:2] == ["B2@2.08h", "B2@2.58h"]
    assert do_missing.detail["n_rows_affected"] == 2 + 11  # B2 gap + D4's 11 missing readings
    assert "never filled" in do_missing.message
    s = well_row(plate.summary, "B2")
    # 13 readings - 2 missing (DO) or 1 missing (pH) - 1 taken during the pause
    assert s["n_valid_do"] == 10 and s["n_valid_ph"] == 11
    # three 0.5 h steps touch a missing reading and two touch the in-pause reading: not
    # counted, reported as unassessed
    assert s["do_time_unassessed_h"] == pytest.approx(1.5 + 1.0)
    assert s["do_min"] == pytest.approx(64.0)


def test_dip_below_declared_threshold(plate):
    s = well_row(plate.summary, "C3")
    assert s["do_min"] == pytest.approx(20.0) and s["do_final"] == pytest.approx(80.0)
    # 40 -> 25 crosses 30 at 1/3 of the way back from 25: 0.5 * 10/15 h before the 25 reading;
    # 25 -> 35 crosses at half way. Below from t=1.8333 h to t=3.75 h (shifted by 5 min).
    assert s["do_time_below_threshold_h"] == pytest.approx(3.75 - (1.5 + 0.5 * 10 / 15))
    assert s["do_time_unassessed_h"] == pytest.approx(1.0)  # the two steps around the pause reading
    assert s["do_final_time_h"] == pytest.approx(6 + 5 / 60)
    assert (plate.summary.drop(index=plate.summary.index[plate.summary["well"].isin(["C3", "D4"])])
            ["do_time_below_threshold_h"] == 0).all()
    assert plate.meta["do_threshold"] == 30.0


def test_short_well_gets_nan_and_a_finding_never_extrapolated(plate):
    s = well_row(plate.summary, "D4")
    assert s["n_valid_do"] == 2
    for col in ("do_min", "do_final", "do_time_below_threshold_h", "do_time_unassessed_h"):
        assert math.isnan(s[col])
    assert s["ph_start"] == pytest.approx(7.05)  # the pH channel of the well is fine
    f = checks(plate)["too_few_points"]
    assert f.detail["wells"] == ["Synthetic_PlateA:D4:DO (2)"]
    assert f.detail["min_points"] == MIN_VALID_POINTS == 3
    # the readings stay in the table
    assert len(plate.table[(plate.table["well"] == "D4") & (plate.table["channel"] == "DO")]) == 13
    lower = read_snest([DO_CSV, PH_CSV], do_threshold=30.0, min_points=2)
    assert well_row(lower.summary, "D4")["do_min"] == pytest.approx(95.25)
    with pytest.raises(ValueError, match="at least 2"):
        read_snest([DO_CSV], min_points=1)


def test_ph_summaries(plate):
    s = well_row(plate.summary, "A2")
    assert s["ph_start"] == pytest.approx(7.05) and s["ph_final"] == pytest.approx(6.546)
    assert s["ph_drift"] == pytest.approx(6.546 - 7.05)
    assert s["ph_min"] == pytest.approx(6.546) and s["ph_max"] == pytest.approx(7.05)
    assert s["ph_start_time_h"] == pytest.approx(5 / 60) and s["ph_final_time_h"] == pytest.approx(6 + 5 / 60)
    assert s["n_valid_ph"] == 12  # the in-pause reading is not counted
    blank = well_row(plate.summary, "A1")
    assert blank["ph_drift"] == 0.0 and bool(blank["baseline_reference_well"])
    assert not bool(s["baseline_reference_well"])
    assert list(plate.summary.columns) == snest.SUMMARY_COLUMNS and len(plate.summary) == 24


def test_threshold_must_be_declared():
    res = read_snest([DO_CSV, PH_CSV])
    f = checks(res)["do_threshold_not_declared"]
    assert f.severity == "error" and "Declare do_threshold" in f.message and "none is assumed" in f.message
    assert res.summary["do_time_below_threshold_h"].isna().all()
    assert res.summary["do_min"].notna().sum() == 23  # the other DO summaries are still there
    assert res.meta["do_threshold"] is None
    assert well_row(res.summary, "B2")["do_time_unassessed_h"] == pytest.approx(2.5)
    bad = read_snest([DO_CSV], do_threshold=float("nan"))
    assert checks(bad)["do_threshold_invalid"].severity == "error"
    assert bad.summary["do_time_below_threshold_h"].isna().all()


def test_summary_formulas_are_declared(plate):
    cols = plate.meta["summary_columns"]
    for col in snest.SUMMARY_COLUMNS[6:]:
        assert cols[col]["formula"] and cols[col]["derived_from"], col
    assert cols["do_time_below_threshold_h"]["threshold"] == 30.0
    assert "straight-line interpolation" in cols["do_time_below_threshold_h"]["formula"]
    assert cols["ph_drift"]["formula"] == "ph_final - ph_start"
    assert cols["ph_drift"]["derived_from"] == ["ph_final", "ph_start"]
    assert plate.meta["min_valid_points"] == 3


def test_no_summary_or_metadata_column_is_ever_a_setting(plate):
    roles = plate.meta["role_suggestions"]
    assert set(roles.values()) <= ROLES - {"setting"}
    assert roles["plate"] == "batch_group" and roles["well"] == "id"
    summaries = ["do_min", "do_final", "do_time_below_threshold_h", "ph_start", "ph_final", "ph_drift",
                 "ph_min", "ph_max"]
    assert all(roles[c] in {"covariate", "outcome"} for c in summaries)
    assert roles["ph_start"] == "covariate" and roles["do_min"] == "outcome"
    run = plate.meta["plates"]["Synthetic_PlateA"]
    every = [*plate.table.columns, *plate.summary.columns, *run["as_run"], *run["conditions"]]
    assert all(roles.get(c) != "setting" for c in every)
    for spec in plate.meta["summary_columns"].values():
        assert spec["role"] != "setting"
    one = read_snest_csv(DO_CSV)
    assert "setting" not in one.meta["role_suggestions"].values()


def test_pauses_are_recorded_and_readings_inside_flagged(plate):
    t = plate.table
    inside = t[t["in_pause"]]
    assert set(inside["timestamp"]) == {"2031/3/14 14:05"} and len(inside) == 48
    f = checks(plate)["pauses_recorded"]
    assert f.severity == "info" and f.detail["n_rows_affected"] == 24


def test_sensor_calibration_and_processing_notes():
    res = parse_snest_text(export(extra="***Data Processing Options***\nMoving Average Smoothing,ON\n"
                                        "Baseline Correction,OFF\nSensor calibration:,ON/1 h after start"))
    c = checks(res)
    assert c["instrument_processing_on"].detail["options"] == ["Moving Average Smoothing"]
    assert c["sensor_calibration_after_start"].severity == "info"


def test_txt_export_is_refused_with_an_action():
    res = read_snest_csv(DO_TXT)
    f = checks(res)["snest_txt_not_supported"]
    assert f.severity == "error" and "Upload the CSV export" in f.message
    assert res.table.empty


def test_no_data_block_or_bad_header_is_an_error():
    res = parse_snest_text("Well,Value\nA1,0.5\n")
    assert checks(res)["no_data_section"].severity == "error" and res.table.empty
    bad = parse_snest_text(export().replace("Running time(min)", "Elapsed"))
    assert checks(bad)["data_header_unreadable"].severity == "error" and bad.table.empty


PAUSE = "***Pause information***\nPause No.,from,to\nPause 1,2031/3/14 10:00,2031/3/14 10:10"
SERIES = (("2031/3/14 09:05", "5", ("90", "80")), ("2031/3/14 09:35", "35", ("85", "75")),
          ("2031/3/14 10:05", "65", ("10", "5")), ("2031/3/14 10:35", "95", ("80", "70")),
          ("2031/3/14 11:05", "125", ("78", "68")))


def test_pause_dip_does_not_change_do_min_or_time_below():
    """During a pause the plate is off the shaker: a DO dip then is lost aeration, not the culture."""
    res = combine([parse_snest_text(export(rows=SERIES, extra=PAUSE))], do_threshold=30.0)
    t = res.table
    dip = t[t["timestamp"] == "2031/3/14 10:05"]
    assert dip["in_pause"].all() and (dip["status"] == "ok").all() and list(dip["value"]) == [10.0, 5.0]
    s = well_row(res.summary, "A1")
    assert s["do_min"] == 78.0 and s["do_time_below_threshold_h"] == 0.0
    assert s["n_valid_do"] == 4 and s["do_time_unassessed_h"] == pytest.approx(1.0)
    assert well_row(res.summary, "A2")["do_min"] == 68.0
    f = checks(res)["pause_readings_excluded"]
    assert f.detail["n_rows_affected"] == 2 and f.detail["wells"] == ["A1", "A2"]
    assert "in_pause" in res.meta["summary_columns"]["do_min"]["formula"]
    assert "table.in_pause" in res.meta["summary_columns"]["do_min"]["derived_from"]
    # without the pause block the same dip counts
    plain = combine([parse_snest_text(export(rows=SERIES))], do_threshold=30.0)
    assert well_row(plain.summary, "A1")["do_min"] == 10.0
    assert well_row(plain.summary, "A1")["do_time_below_threshold_h"] > 0
    assert "pause_readings_excluded" not in checks(plain)


def test_final_and_start_carry_their_time_and_early_ends_are_flagged():
    rows = tuple((clk, run, (a, "N/A" if k >= 3 else b)) for k, (clk, run, (a, b)) in enumerate(SERIES))
    res = combine([parse_snest_text(export(rows=rows))], do_threshold=30.0)
    a1, a2 = well_row(res.summary, "A1"), well_row(res.summary, "A2")
    assert a1["do_final_time_h"] == pytest.approx(125 / 60)
    assert a2["do_final_time_h"] == pytest.approx(65 / 60) and a2["do_final"] == 5.0
    f = checks(res)["series_ends_early"]
    assert f.severity == "warning" and f.detail["wells"] == [
        "Synthetic_PlateZ:A2:DO (last valid 1.08 h, plate 2.08 h)"]
    for col in ("do_final_time_h", "ph_start_time_h", "ph_final_time_h"):
        spec = res.meta["summary_columns"][col]
        assert spec["role"] is None and spec["formula"] and spec["derived_from"] and spec["unit"] == "h"
        assert col not in res.meta["role_suggestions"]
    # one step short is within a sampling interval: not flagged
    one_short = tuple((clk, run, (a, "N/A" if k == 4 else b)) for k, (clk, run, (a, b)) in enumerate(SERIES))
    assert "series_ends_early" not in checks(combine([parse_snest_text(export(rows=one_short))]))


def test_readings_before_sensor_calibration_are_counted_not_dropped(plate):
    f = next(f for f in plate.findings if f.check == "sensor_calibration_after_start")
    assert f.detail["calibration_after_h"] == pytest.approx(0.5)
    assert f.detail["n_rows_affected"] == 24  # the 5 min reading of each well
    assert "not be calibrated" in f.message and "nothing was changed" in f.message
    hour = parse_snest_text(export(extra="Sensor calibration:,ON/One hour after start"))
    g = checks(hour)["sensor_calibration_after_start"]
    assert g.detail["calibration_after_h"] == 1.0 and g.detail["n_rows_affected"] == 4
    vague = parse_snest_text(export(extra="Sensor calibration:,ON/after start"))
    assert checks(vague)["sensor_calibration_after_start"].detail["calibration_after_h"] is None


# -- wells ---------------------------------------------------------------------


def test_wells_outside_the_plate_format_are_reported_and_kept():
    rows = (("2031/3/14 09:05", "5", ("1", "2", "3", "4")), ("2031/3/14 09:35", "35", ("1", "2", "3", "4")))
    res = parse_snest_text(export(wells=("A1", "A7", "E1", "A01"), rows=rows, layout=False))
    c = checks(res)
    f = c["well_outside_plate_format"]
    assert f.severity == "error" and f.detail["wells"] == ["A7", "E1"]
    assert f.detail["n_rows_affected"] == 4
    assert c["duplicate_well_column"].detail["wells"] == ["A1"]  # "A01" is A1
    assert len(res.table) == 8 and set(res.table["well"]) == {"A1", "A7", "E1", "A01"}
    assert res.table.loc[res.table["well"] == "A01", "well_canonical"].iloc[0] == "A1"
    ok96 = parse_snest_text(export(wells=("H12", "I1"), rows=(("2031/3/14 09:05", "5", ("1", "2")),),
                                   fmt="96-well plate", layout=False))
    assert checks(ok96)["well_outside_plate_format"].detail["wells"] == ["I1"]


def test_invalid_label_layout_mismatch_and_formats():
    res = parse_snest_text(export(wells=("A1", "Temp"), layout=False))
    assert checks(res)["well_label_invalid"].detail["labels"] == ["Temp"]
    lay = parse_snest_text(export().replace("A2,Medium A,1", "B6,Medium A,1"))
    f = checks(lay)["layout_data_mismatch"]
    assert f.detail["only_in_data"] == ["A2"] and f.detail["only_in_layout"] == ["B6"]
    mism = parse_snest_text(export(), plate_format=96)
    assert checks(mism)["plate_format_mismatch"].severity == "error"
    unk = parse_snest_text(export(fmt="48-well plate"))
    assert checks(unk)["plate_format_unknown"].detail["plate_format"] == 48


# -- times and values --------------------------------------------------------------


def test_duplicated_and_backward_timestamps_reported():
    rows = (("2031/3/14 09:05", "5", ("90", "80")), ("2031/3/14 09:35", "35", ("85", "70")),
            ("2031/3/14 09:35", "35", ("84", "69")), ("2031/3/14 09:20", "20", ("88", "75")),
            ("2031/3/14 10:05", "65", ("80", "60")))
    res = parse_snest_text(export(rows=rows))
    c = checks(res)
    assert c["duplicate_timestamp"].detail["lines"] == [res.table["source_line"].iloc[2]]
    assert c["time_not_increasing"].detail["lines"] == [res.table["source_line"].iloc[3]]
    assert len(res.table) == 10  # all kept
    s, _f, _c = snest.summarize(res.table, do_threshold=50.0)
    # summaries use readings sorted by time: start/final are the 5 and 65 min readings
    assert s.loc[s["well"] == "A1", "do_final"].iloc[0] == 80.0


def test_time_not_numeric_and_clock_disagreement():
    rows = (("2031/3/14 09:05", "5", ("90", "80")), ("2031/3/14 09:35", "x", ("85", "70")),
            ("2031/3/14 11:05", "65", ("80", "60")), ("2031/3/14 10:35", "95", ("75", "50")))
    res = parse_snest_text(export(rows=rows))
    c = checks(res)
    assert c["time_not_numeric"].severity == "error"
    assert set(res.table.loc[res.table["running_time_raw"] == "x", "status"]) == {"no_time"}
    assert c["clock_time_disagrees"].detail["lines"] == [res.table["source_line"].iloc[2]]


def test_not_numeric_negative_and_out_of_range_values():
    rows = (("2031/3/14 09:05", "5", ("ERR", "-2")), ("2031/3/14 09:35", "35", ("85", "70")),
            ("2031/3/14 10:05", "65", ("", "60")))
    do = parse_snest_text(export(rows=rows))
    c = checks(do)
    assert c["reading_not_numeric"].detail["raw_values"] == ["ERR"]
    assert c["reading_not_numeric"].detail["where"] == ["A1@0.08h"]
    assert c["do_negative"].detail["where"] == ["A2@0.08h"]
    assert c["reading_missing"].detail["where"] == ["A1@1.08h"]
    ph_rows = (("2031/3/14 09:05", "5", ("7.0", "15.2")), ("2031/3/14 09:35", "35", ("6.9", "7.0")))
    ph = parse_snest_text(export("pH", rows=ph_rows))
    assert checks(ph)["ph_out_of_range"].detail["where"] == ["A2@0.08h"]
    assert set(ph.table["unit"]) == {"pH"}


def test_declared_start_shifts_time_and_is_stated():
    res = parse_snest_text(export(), start=datetime(2031, 3, 14, 8, 30))
    assert res.table["time_h"].iloc[0] == pytest.approx(0.5 + 5 / 60)
    assert "declared start" in res.meta["time_basis"]
    no_start = parse_snest_text(export().replace("Start time:,2031-03-14 09:00:00", "Start time:,--"),
                                start=datetime(2031, 3, 14, 8, 30))
    assert checks(no_start)["start_time_missing"].severity == "error"
    assert no_start.table["time_h"].iloc[0] == pytest.approx(5 / 60)


def test_plate_id_declared_or_from_file():
    assert parse_snest_text(export(), plate="P1").meta["plate_source"] == "declared"
    res = parse_snest_text(export(name=None))
    assert checks(res)["plate_not_declared"].severity == "error" and res.table["plate"].isna().all()


# -- several files ---------------------------------------------------------------


def test_two_files_of_one_channel_are_not_summarized():
    res = read_snest([DO_CSV, DO_CSV, PH_CSV], do_threshold=30.0)
    f = checks(res)["duplicate_channel_file"]
    assert f.severity == "error" and f.detail["channel"] == "DO"
    assert res.summary["do_min"].isna().all() and res.summary["ph_start"].notna().all()
    assert len(res.table) == 24 * 13 * 3


def test_metadata_differs_and_missing_channel():
    do = parse_snest_text(export(), source="a")
    ph = parse_snest_text(export("pH", rows=(("2031/3/14 09:05", "5", ("7", "7")),)).replace(
        "Temperature:,30", "Temperature:,37"), source="b")
    res = combine([do, ph], do_threshold=50.0)
    assert checks(res)["metadata_differs_between_files"].detail["fields"] == ["as_run"]
    only = combine([do], do_threshold=50.0)
    assert checks(only)["channel_not_loaded"].detail == {"plate": "Synthetic_PlateZ", "channel": "pH"}


def test_time_below_rule_with_irregular_sampling():
    # irregular steps; a pair touching a missing reading is not counted
    t = [0.0, 0.5, 2.0, 2.25, 3.0]
    v = [50.0, 10.0, 10.0, math.nan, 50.0]
    below, unassessed = time_below(t, v, [True, True, True, False, True], 30.0)
    assert below == pytest.approx(0.5 * 20 / 40 + 1.5)
    assert unassessed == pytest.approx(1.0)
    # duplicated times add nothing; equal to threshold is not below
    assert time_below([0.0, 0.0, 1.0], [10.0, 10.0, 10.0], [True] * 3, 30.0) == (1.0, 0.0)
    assert time_below([0.0, 1.0], [30.0, 30.0], [True, True], 30.0) == (0.0, 0.0)


# -- contracts ---------------------------------------------------------------------


def test_deterministic(plate):
    again = read_snest([DO_CSV, PH_CSV], do_threshold=30.0)
    pd.testing.assert_frame_equal(plate.table, again.table)
    pd.testing.assert_frame_equal(plate.summary, again.summary)
    assert plate.to_dict() == again.to_dict()


RESERVED = {"outliers", "physical_bounds", "duplicate_ids", "result_duplicate_entry", "target_leak",
            "derived_output"}


def _all_results():
    """Results that between them raise every check the parser has, for the contract test."""
    rows = (("2031/3/14 09:05", "5", ("ERR", "-2", "1", "1")), ("2031/3/14 09:35", "x", ("85", "70", "1", "1")),
            ("2031/3/14 09:35", "35", ("", "60", "1", "1")), ("2031/3/14 09:35", "35", ("1", "1", "1", "1")),
            ("2031/3/14 09:20", "20", ("80", "60", "1", "1")), ("2031/3/14 11:05", "65", ("80", "60", "1", "1")))
    yield read_snest([DO_CSV, PH_CSV])
    yield read_snest([DO_CSV, DO_CSV, PH_CSV], do_threshold=30.0)
    yield read_snest([DO_CSV], do_threshold=float("nan"))
    yield read_snest_csv(DO_TXT)
    yield parse_snest_text("Well,Value\nA1,0.5\n")
    yield parse_snest_text(export().replace("Running time(min)", "Elapsed"))
    yield parse_snest_text(
        export(wells=("A1", "A7", "Temp", "A01"), rows=rows, name=None,
               extra="Moving Average Smoothing,ON\nSensor calibration:,ON/1 h after start"),
        plate_format=96, start=datetime(2031, 3, 14))
    yield parse_snest_text(export().replace("A2,Medium A,1", "B6,Medium A,1"))
    yield parse_snest_text(export("pH", rows=(("2031/3/14 09:05", "5", ("7", "15")),)))
    yield parse_snest_text(export(title="***S.NEST DO data (%)***"), do_unit="mg/L")
    yield parse_snest_text(export(fmt="48-well plate"))
    yield parse_snest_text(export().replace("Start time:,2031-03-14 09:00:00", "Start time:,--"),
                           start=datetime(2031, 3, 14))
    yield combine([parse_snest_text(export(), source="a"), parse_snest_text(
        export("pH").replace("Temperature:,30", "Temperature:,37"), source="b")])
    yield combine([parse_snest_text(export(rows=SERIES, extra=PAUSE))])
    yield combine([parse_snest_text(export(rows=tuple(
        (clk, run, (a, "N/A" if k >= 3 else b)) for k, (clk, run, (a, b)) in enumerate(SERIES))))])
    yield combine([parse_snest_text(export(), do_unit="%", plate="P1"),
                   parse_snest_text(export(), do_unit="mg/L", plate="P2")])


def test_every_finding_is_a_validation_finding_with_a_stable_code():
    seen: set[str] = set()
    for res in _all_results():
        json.dumps(res.to_dict(), allow_nan=False)
        for f in res.findings:
            assert type(f) is Finding
            assert f.check == f.check.lower() and f.check.replace("_", "").isalpha()
            assert f.check not in RESERVED
            assert f.severity in {"error", "warning", "info"}
            assert len(f.rows) <= ingest_findings.ROW_CAP
            assert f.message.rstrip().endswith(".")
            seen.add(f.check)
    # every check the parser can emit is raised by one of the cases above
    emitted = set(re.findall(r'finding\(\s*"([a-z_]+)"', Path(snest.__file__).read_text()))
    assert len(emitted) >= 30
    assert emitted == seen, emitted ^ seen


def test_ingest_imports_stay_in_layer():
    """kalos/ingest may import util, data, config, core and libs only (layer rule)."""
    allowed = ("kalos.ingest", "kalos.util", "kalos.data", "kalos.config", "kalos.core", "kalos.libs",
               "kalos.normalize", "kalos.domains", "kalos.validation", "kalos.scale", "kalos.providers")
    root = Path(snest.__file__).parent.parent
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
