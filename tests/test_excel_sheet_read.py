"""The per-sheet read in `kalos.portal.uploads._parse_excel` must return exactly
what `pd.ExcelFile.parse` returns for the same sheet.

It used to call `book.parse(name)`, which the pandas-stubs in CI (3.x) do not
declare, so mypy failed on `uploads.py` with `"ExcelFile" has no attribute
"parse"`. The fix is `pd.read_excel(book, sheet_name=name)`, which pandas
implements as exactly `ExcelFile.parse` with the same defaults. This pins that
equivalence on a workbook with mixed dtypes, NA cells and a second sheet.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

pytest.importorskip("openpyxl")


def test_read_excel_on_an_excelfile_matches_excelfile_parse():
    buf = io.BytesIO()
    frames = {
        "runs": pd.DataFrame({"temp": [30.0, None, 37.5], "ph": [7, 6, 5], "label": ["a", "b", None]}),
        "other": pd.DataFrame({"x": [1, 2]}),
    }
    with pd.ExcelWriter(buf) as writer:
        for name, df in frames.items():
            df.to_excel(writer, sheet_name=name, index=False)
    book = pd.ExcelFile(io.BytesIO(buf.getvalue()))
    for name in book.sheet_names:
        via_parse = book.parse(name)
        via_read = pd.read_excel(book, sheet_name=name)
        pd.testing.assert_frame_equal(via_read, via_parse)


def test_uploads_module_has_no_direct_excelfile_parse_call():
    """Guards the typing fix: the call site stays on the stub-declared API."""
    import ast
    from pathlib import Path

    import kalos.portal.uploads as uploads

    tree = ast.parse(Path(uploads.__file__).read_text())
    direct = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "parse"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "book"
    ]
    assert direct == []
