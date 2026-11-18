"""Torch-free coverage for the xlsx zip-bomb guard (`_reject_oversized_xlsx`).

The guard that stops a crafted `.xlsx` (small real content, enormous DECLARED
`<dimension>`, or a sparse cell planted far down a sheet) from being handed to
`pd.read_excel` is security-critical, but every existing test that exercises it
imports `kalos.portal.app`, which transitively pulls in gpytorch/torch. So on a
checkout without the heavy `ml` extra the guard has no coverage at all.

This module imports ONLY `kalos.portal.uploads` (no torch) and calls the guard
directly, so the pre-materialization ceiling is pinned in the lean test tier
too. It is the regression net for issue #14 (declared-dimension bomb slipping
past the guard under openpyxl 3.1.5): the declared shape is read straight from
the sheet XML, independent of openpyxl's lazy `max_row` / `max_column`.
"""
from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable

import pandas as pd
import pytest

from kalos.portal import uploads
from kalos.portal.uploads import UploadRejected


def _xlsx(sheet_xml_transform: Callable[[str], str] | None = None) -> bytes:
    """A valid tiny xlsx (3 real rows), optionally rewriting its worksheet XML
    so the DECLARED shape differs from the real content (a zip-bomb shape)."""
    pytest.importorskip("openpyxl")
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_excel(buf, index=False)
    zin = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    names = zin.namelist()
    sheet = next(n for n in names if n.startswith("xl/worksheets/sheet"))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            data = zin.read(n)
            if n == sheet and sheet_xml_transform is not None:
                data = sheet_xml_transform(data.decode()).encode()
            zout.writestr(n, data)
    return out.getvalue()


def _with_declared_dims(ref: str) -> bytes:
    return _xlsx(lambda xml: re.sub(r'<dimension ref="[^"]*"\s*/>', f'<dimension ref="{ref}"/>', xml))


def test_declared_dimension_bomb_rejected_at_real_cap():
    # Issue #14: declared ~524M cells (SF = col 500, within the column cap) but
    # only 3 real rows. Must trip the CELL cap from the declared <dimension>,
    # not be waved through because openpyxl materialized just 3 rows.
    assert uploads.MAX_XLSX_CELLS < 500 * 1_048_576
    with pytest.raises(UploadRejected) as exc:
        uploads._reject_oversized_xlsx(_with_declared_dims("A1:SF1048576"))
    assert exc.value.code == "too_many_cells"


def test_declared_columns_over_cap_rejected(monkeypatch):
    # A declared width past MAX_COLUMNS is refused as too_many_columns (distinct
    # machine code from the cell cap), again from the declared dimension alone.
    monkeypatch.setattr(uploads, "MAX_COLUMNS", 3)
    with pytest.raises(UploadRejected) as exc:
        uploads._reject_oversized_xlsx(_with_declared_dims("A1:H10"))
    assert exc.value.code == "too_many_columns"


def test_missing_dimension_sparse_bomb_rejected():
    # No <dimension> tag at all, but a single cell planted far down the sheet:
    # the streaming real-extent scan must still reject it before materialization.
    def strip_dimension_and_plant_far_cell(xml: str) -> str:
        xml = re.sub(r'<dimension ref="[^"]*"\s*/>', "", xml)
        return xml.replace(
            "<sheetData>",
            '<sheetData><row r="3000000"><c r="A3000000"><v>1</v></c></row>',
            1,
        )

    with pytest.raises(UploadRejected) as exc:
        uploads._reject_oversized_xlsx(_xlsx(strip_dimension_and_plant_far_cell))
    assert exc.value.code in {"too_many_cells", "too_many_columns"}


def test_honest_small_xlsx_accepted():
    # An honestly-small sheet passes and the guard reports the zip parts it saw.
    part_sizes = uploads._reject_oversized_xlsx(_xlsx())
    assert isinstance(part_sizes, dict) and part_sizes


def _zip(entries: dict[str, str]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return out.getvalue()


@pytest.mark.parametrize(
    ("entries", "hint"),
    [
        ({"word/document.xml": "<x/>"}, "Word document"),
        ({"ppt/presentation.xml": "<x/>"}, "PowerPoint file"),
        ({"notes.txt": "hi"}, "zip archive"),
    ],
)
def test_non_xlsx_zip_named_not_called_corrupt(entries, hint):
    # A well-formed zip that is not an Excel workbook (a .docx/.pptx mis-named
    # .xlsx, or a plain archive) is reported as unsupported with a human hint
    # about what it actually is - not as a generic "corrupt" error.
    with pytest.raises(UploadRejected) as exc:
        uploads._reject_oversized_xlsx(_zip(entries))
    assert exc.value.code == "unsupported_file_type"
    assert hint in str(exc.value)
