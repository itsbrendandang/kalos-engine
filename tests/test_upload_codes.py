"""Every size/shape refusal of an upload carries a stable machine code (review M4).

These raised `UploadRejected` with no `code`, so a client got `"code": null` and
could not tell "too big" from "too wide" or "too many rows". The caps are
monkeypatched small so each case is cheap to build.
"""
from __future__ import annotations

import io

import pytest

from kalos.portal import uploads
from kalos.portal.uploads import UploadRejected, parse_upload


def _xlsx(rows: list[list[object]]) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _rejected_code(raw: bytes, **kwargs) -> str | None:
    with pytest.raises(UploadRejected) as exc:
        parse_upload(raw, **kwargs)
    return exc.value.code


def test_byte_cap_is_payload_too_large(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 64)
    assert _rejected_code(b"a,b\n" + b"1,2\n" * 40, filename="x.csv") == "payload_too_large"


def test_too_many_columns_in_csv(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_COLUMNS", 4)
    header = ",".join(f"c{i}" for i in range(5))
    raw = (header + "\n" + ",".join(["1"] * 5) + "\n").encode()
    assert _rejected_code(raw, filename="w.csv") == "too_many_columns"


def test_too_many_columns_in_xlsx(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_COLUMNS", 3)
    raw = _xlsx([["a", "b", "c", "d"], [1, 2, 3, 4], [2, 3, 4, 5]])
    assert _rejected_code(raw, filename="w.xlsx") == "too_many_columns"


def test_too_many_cells_in_xlsx(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_XLSX_CELLS", 5)
    raw = _xlsx([["a", "b", "c"], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    assert _rejected_code(raw, filename="c.xlsx") == "too_many_cells"


def test_xlsx_uncompressed_size_is_payload_too_large(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_XLSX_UNCOMPRESSED_BYTES", 1024)
    raw = _xlsx([["a", "b"]] + [[i, i * 2] for i in range(2000)])
    assert _rejected_code(raw, filename="big.xlsx") == "payload_too_large"


def test_every_size_refusal_site_in_uploads_has_a_code():
    """Static guard: no `UploadRejected(_ERR_TOO_LARGE ...)` / `_ERR_TOO_MANY_COLUMNS`
    raise may be left without `code=`."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(uploads.__file__).read_text())
    bare = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "UploadRejected":
            first = node.args[0] if node.args else None
            if (
                isinstance(first, ast.Name)
                and first.id in {"_ERR_TOO_LARGE", "_ERR_TOO_MANY_COLUMNS"}
                and not any(k.arg == "code" for k in node.keywords)
            ):
                bare.append(node.lineno)
    assert bare == []
