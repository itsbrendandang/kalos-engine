"""Findings about rows carry every affected row as a SHEET row number
(`sheet_rows`: header = row 1, first data row = 2), plus `values` for
one-column findings; `rows` stays the capped frame index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.validation import report_dict, validate_frame
from kalos.validation.report import ROW_CAP, SHEET_ROW_CAP, Finding


def _findings(df: pd.DataFrame, **kw) -> list[dict]:
    return report_dict(validate_frame(df, **kw))["findings"]


def _one(findings, check, column=None, severity=None):
    hits = [
        f for f in findings
        if f["check"] == check and (column is None or f["column"] == column)
        and (severity is None or f["severity"] == severity)
    ]
    assert len(hits) == 1, (check, column, [f["check"] for f in findings])
    return hits[0]


def _consistent(f):
    assert f["detail"]["n_rows_affected"] == len(f["sheet_rows"])
    assert f["sheet_rows_truncated"] is False
    assert f["rows"] == [r - 2 for r in f["sheet_rows"]][:ROW_CAP]


def test_outliers_return_all_43_rows_with_values():
    rng = np.random.default_rng(0)
    sp = rng.normal(10.0, 0.5, 200).round(3)
    bad = sorted(rng.choice(200, 43, replace=False).tolist())
    sp[bad] = 500.0
    f = _one(_findings(pd.DataFrame({"SP": sp})), "outliers", "SP")
    assert len(f["rows"]) == ROW_CAP  # unchanged: capped
    assert f["sheet_rows"] == [r + 2 for r in bad]
    _consistent(f)
    assert f["values"] == {str(r + 2): 500.0 for r in bad}


def test_physical_bounds_units_duplicates_and_blanks():
    n = 30
    df = pd.DataFrame({
        "pH": [7.0] * n,
        "temperature": ["37 C"] * n,
        "feed": [float(i) for i in range(n)],
        "notes": [None] * 20 + ["ok"] * 10,
    })
    df.loc[3, "pH"] = 20.0        # impossible pH -> sheet row 5
    df.loc[7, "temperature"] = "98 F"  # mixed units -> sheet row 9 among others
    df.loc[11] = df.loc[10]       # exact duplicate of row 10 -> sheet rows 12, 13
    findings = _findings(df)

    ph = _one(findings, "physical_bounds", "pH", "error")
    assert ph["sheet_rows"] == [5] and ph["values"] == {"5": 20.0}
    _consistent(ph)

    units = _one(findings, "units_consistency", "temperature", "error")
    assert len(units["sheet_rows"]) == n and units["values"]["9"] == "98 F"
    _consistent(units)

    dup = _one(findings, "duplicate_rows", severity="warning")
    assert dup["sheet_rows"] == [12, 13] and "values" not in dup
    _consistent(dup)

    blank = _one(findings, "missingness", "notes")
    assert blank["rows"] == []  # unchanged: this finding never listed rows
    assert blank["sheet_rows"] == list(range(2, 22))
    assert blank["detail"]["n_rows_affected"] == 20


def test_informative_missingness_lists_the_blank_rows():
    n = 40
    y = np.linspace(1, 100, n)
    x = np.linspace(0, 1, n)
    other = np.where(y < 30, np.nan, x)  # blank exactly where the result is low
    df = pd.DataFrame({"x": x, "other": other, "titer": y})
    f = _one(_findings(df, target="titer"), "informative_missingness", "other")
    blank = [int(i) + 2 for i in np.flatnonzero(np.isnan(other))]
    assert f["sheet_rows"] == blank
    assert f["detail"]["n_rows_affected"] == len(blank)


def test_sheet_row_cap_sets_the_flag():
    f = Finding(check="outliers", severity="warning", message="m", column="c",
                all_rows=tuple(range(SHEET_ROW_CAP + 7)))
    d = f.to_dict()
    assert len(d["sheet_rows"]) == SHEET_ROW_CAP and d["sheet_rows_truncated"] is True
    plain = Finding(check="x", severity="info", message="m").to_dict()
    assert "sheet_rows" not in plain  # not about rows: no new keys


# --- end to end through the upload parser ------------------------------------ #

pytest.importorskip("fastapi")

from kalos.portal.analysis import _analyze  # noqa: E402
from kalos.portal.uploads import parse_upload  # noqa: E402


def _csv_with_group_row_and_blank_line() -> bytes:
    """Line 1: a group-label row (so the names are on line 2). Line 7: blank.
    Data starts on line 3."""
    rng = np.random.default_rng(1)
    lines = ["Medium,,", "Methanol,pH,lipase_titer"]
    for i in range(24):
        m = round(float(rng.uniform(0, 4)), 3)
        ph = round(float(rng.uniform(5, 7)), 2)
        y = round(1.5 * m + float(rng.normal(0, 0.3)), 3)
        lines.append(f"{m},{ph},{y}")
        if i == 3:
            lines.append("")  # a blank line the parser skips
    lines[10] = lines[10].split(",")[0] + ",ERR," + lines[10].split(",")[2]  # line 11
    lines[14] = "999," + ",".join(lines[14].split(",")[1:])  # line 15: an outlier Methanol
    return ("\n".join(lines) + "\n").encode()


def test_header_offset_sheet_reports_rows_as_the_user_sees_them():
    parsed = parse_upload(_csv_with_group_row_and_blank_line(), filename="s.csv")
    df = parsed.df
    assert df.attrs["kalos_header"]["header_row"] == 1
    out = _analyze(df, target="lipase_titer")
    findings = out["validation"]["findings"]
    outlier = _one(findings, "outliers", "Methanol")
    assert outlier["sheet_rows"] == [15]
    assert outlier["values"] == {"15": 999.0}
    dropped = _one(findings, "dropped_cells", "pH")
    assert dropped["sheet_rows"] == [11] and dropped["values"] == {"11": "ERR"}
    assert dropped["detail"]["n_rows_affected"] == 1
    rows = _one(findings, "dropped_rows")
    assert rows["sheet_rows"] == [11] and rows["detail"]["n_rows_affected"] == 1
    # The counts agree with the findings, severity by severity.
    counts = out["validation"]["counts"]
    for sev in ("error", "warning", "info"):
        assert counts[sev] == sum(1 for f in findings if f["severity"] == sev)
    for f in findings:
        if f.get("sheet_rows") is not None and "n_rows_affected" in f["detail"]:
            assert f["detail"]["n_rows_affected"] == len(f["sheet_rows"]), f["check"]


def test_excel_header_on_row_2_with_a_blank_row():
    import io

    openpyxl = pytest.importorskip("openpyxl")
    from kalos.portal.analysis_parts.diagnostics import sheet_row_mapper

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Medium", None, None])          # row 1: group label
    ws.append(["Methanol", "pH", "titer"])     # row 2: names
    for i in range(30):                         # rows 3..
        ws.append([1.0 + 0.01 * i, 6.5, 10.0 + i])
    ws.append([None, None, None])               # a blank row (kept by the parser)
    ws.append([900.0, 6.5, 50.0])               # the outlier, on sheet row 34
    buf = io.BytesIO()
    wb.save(buf)
    df = parse_upload(buf.getvalue(), filename="s.xlsx").df
    assert df.attrs["kalos_header"]["header_row"] == 1
    mapper = sheet_row_mapper(df.attrs["kalos_header"], df.attrs.get("kalos_row_map"), False)
    findings = report_dict(validate_frame(df), sheet_row=mapper)["findings"]
    f = _one(findings, "outliers", "Methanol")
    assert f["sheet_rows"] == [34] and f["values"] == {"34": 900.0}
