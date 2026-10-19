"""Every way an upload can fail gets its own specific, actionable, safe error.

Regression for a QA finding: an empty file, a corrupt .xlsx, a .txt of prose, a
4-row sheet, a one-column sheet, and a workbook whose data is on sheet 2 all
returned the same "Could not parse the uploaded file" message. Each now returns
`{"error": <specific message>, "code": <machine code>, ...}` with HTTP 400, and
workbooks report `sheets` / `sheet_used` and accept a `sheet` parameter. No
response may contain exception text, paths, or stack details. All files are
built in memory from synthetic data.
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.portal.uploads import (  # noqa: E402
    MIN_FIT_ROWS,
    UploadRejected,
    parse_upload,
    table_shape_problem,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(portal_module, "_LATEST", {})
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app)
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


def _good(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3),
        "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)).round(3),
    })


def _xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


NOTES = pd.DataFrame({"About this workbook": ["synthetic demo", "data on sheet 2"]})

# Internal details that must never reach a client.
_LEAK = re.compile(
    r"Traceback|File \"|/Users/|/home/|site-packages|kalos/|\.py\b|Exception|"
    r"BadZipFile|ParserError|EmptyDataError|InvalidFileException|KeyError|line \d+"
)


def _post(client, content: bytes, filename: str = "runs.csv", **data):
    return client.post(
        "/api/run", files={"file": (filename, content, "application/octet-stream")}, data=data
    )


def _assert_error(resp, code: str) -> dict:
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == code, body
    assert isinstance(body["error"], str) and body["error"]
    assert not _LEAK.search(resp.text), resp.text
    return body


# --- file-level failures --------------------------------------------------------- #


@pytest.mark.parametrize("content", [b"", b"   \n\r\n\t ", b"\xef\xbb\xbf\n"])
def test_empty_file(client, content):
    body = _assert_error(_post(client, content), "empty_file")
    assert body["error"].startswith("The uploaded file is empty")


@pytest.mark.parametrize("content, filename, fragment", [
    (b"%PDF-1.7\n%binary", "runs.pdf", "PDF"),
    (b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "runs.csv", "PNG"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64, "old.xls", ".xls"),
    (b"a,b\n1,2\n", "runs.json", ".json"),
    (bytes(range(256)) * 4, "junk.csv", "binary"),
])
def test_unsupported_file_type(client, content, filename, fragment):
    body = _assert_error(_post(client, content, filename), "unsupported_file_type")
    assert fragment in body["error"]


def test_word_document_is_unsupported_not_corrupt(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<w:document/>")
    body = _assert_error(_post(client, buf.getvalue(), "report.docx"), "unsupported_file_type")
    assert "Word document" in body["error"]


@pytest.mark.parametrize("content", [
    b"PK\x03\x04" + b"\x00" * 300,                     # zip magic, broken archive
    _xlsx({"Data": _good()})[:2000],                   # a truncated real workbook
])
def test_corrupt_excel(client, content):
    body = _assert_error(_post(client, content, "book.xlsx"), "corrupt_excel")
    assert ".xlsx" in body["error"] and "CSV" in body["error"]


def test_prose_txt_is_rejected_specifically(client):
    prose = b"These are my notes from Tuesday.\nThe bioreactor ran fine.\nNo numbers here.\n"
    body = _assert_error(_post(client, prose, "notes.txt"), "too_few_columns")
    assert "commas or tabs" in body["error"]


def test_ragged_text_is_not_a_table(client):
    ragged = b"a,b\n1,2\n3,4,5,6\n7,8\n"
    body = _assert_error(_post(client, ragged, "notes.txt"), "not_a_table")
    assert "comma- or tab-separated" in body["error"]


# --- table-level failures ---------------------------------------------------------- #


def test_four_row_file_states_minimum_and_count(client):
    body = _assert_error(_post(client, _good(4).to_csv(index=False).encode()), "too_few_rows")
    assert body["n_rows"] == 4 and body["min_rows"] == MIN_FIT_ROWS == 6
    assert "4 data rows" in body["error"] and "at least 6" in body["error"]


def test_one_column_file(client):
    csv = pd.DataFrame({"lipase_titer": np.arange(10.0)}).to_csv(index=False).encode()
    body = _assert_error(_post(client, csv), "too_few_columns")
    assert body["n_columns"] == 1


def test_no_numeric_columns(client):
    df = pd.DataFrame({"strain": list("ABCDEFGH"), "notes": ["ok"] * 8})
    _assert_error(_post(client, df.to_csv(index=False).encode()), "no_numeric_inputs")


def test_no_process_inputs(client):
    # Only outputs and an id: nothing to optimize over.
    df = pd.DataFrame({
        "run_id": [f"R{i}" for i in range(8)],
        "biomass_od600": np.linspace(1, 3, 8),
        "lipase_titer": np.linspace(2, 9, 8),
    })
    _assert_error(_post(client, df.to_csv(index=False).encode()), "no_process_inputs")


def test_requested_target_missing(client):
    csv = _good().to_csv(index=False).encode()
    body = _assert_error(_post(client, csv, target="titer_typo"), "target_not_found")
    assert body["target"] == "titer_typo"
    assert body["available_targets"] == ["lipase_titer"]


def test_requested_target_all_blank(client):
    df = _good()
    df["product_titer"] = np.nan
    body = _assert_error(
        _post(client, df.to_csv(index=False).encode(), target="product_titer"), "target_empty"
    )
    assert body["target"] == "product_titer"


def test_too_few_rows_with_a_numeric_target(client):
    df = _good(10)
    df.loc[3:, "lipase_titer"] = np.nan  # only 3 measured rows
    body = _assert_error(_post(client, df.to_csv(index=False).encode()), "too_few_rows")
    assert body["n_rows"] == 3 and body["target"] == "lipase_titer"


# --- workbooks and sheets ------------------------------------------------------------- #


def test_data_on_second_sheet_is_found_and_reported(client):
    resp = _post(client, _xlsx({"Notes": NOTES, "Data": _good()}), "book.xlsx")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sheets"] == ["Notes", "Data"]
    assert body["sheet_used"] == "Data"
    assert body["target"] == "lipase_titer"


def test_first_usable_sheet_wins_when_several_are_usable(client):
    resp = _post(client, _xlsx({"Run1": _good(seed=1), "Run2": _good(seed=2)}), "book.xlsx")
    assert resp.status_code == 200, resp.text
    assert resp.json()["sheet_used"] == "Run1"


@pytest.mark.parametrize("how", ["form", "query"])
def test_explicit_sheet_parameter(client, how):
    content = _xlsx({"Run1": _good(seed=1), "Run2": _good(seed=2)})
    files = {"file": ("book.xlsx", content, "application/octet-stream")}
    if how == "form":
        resp = client.post("/api/run", files=files, data={"sheet": "Run2"})
    else:
        resp = client.post("/api/run?sheet=Run2", files=files)
    assert resp.status_code == 200, resp.text
    assert resp.json()["sheet_used"] == "Run2"
    assert resp.json()["sheets"] == ["Run1", "Run2"]


def test_explicit_unusable_sheet_says_which_and_lists_others(client):
    resp = _post(client, _xlsx({"Notes": NOTES, "Data": _good()}), "book.xlsx", sheet="Notes")
    body = _assert_error(resp, "too_few_columns")
    assert body["sheet_used"] == "Notes" and body["sheets"] == ["Notes", "Data"]
    assert "sheet 'Notes'" in body["error"] and "'Data'" in body["error"]


def test_unknown_sheet_name(client):
    resp = _post(client, _xlsx({"Notes": NOTES, "Data": _good()}), "book.xlsx", sheet="Dat")
    body = _assert_error(resp, "sheet_not_found")
    assert body["sheet"] == "Dat" and body["sheets"] == ["Notes", "Data"]
    assert body["sheet_used"] is None


def test_sheet_name_match_ignores_case_and_padding(client):
    resp = _post(client, _xlsx({"Notes": NOTES, "Data": _good()}), "book.xlsx", sheet=" data ")
    assert resp.status_code == 200, resp.text
    assert resp.json()["sheet_used"] == "Data"


def test_workbook_with_no_usable_sheet(client):
    resp = _post(client, _xlsx({"Notes": NOTES, "More": NOTES}), "book.xlsx")
    body = _assert_error(resp, "no_usable_sheet")
    assert body["sheets"] == ["Notes", "More"]
    assert "'Notes'" in body["error"] and "'More'" in body["error"]


def test_single_sheet_workbook_gives_the_specific_error(client):
    body = _assert_error(_post(client, _xlsx({"Only": _good(4)}), "book.xlsx"), "too_few_rows")
    assert body["sheets"] == ["Only"] and body["sheet_used"] == "Only"
    assert "This was sheet" not in body["error"]  # nothing else to choose


def test_csv_response_has_no_sheet_keys(client):
    resp = _post(client, _good().to_csv(index=False).encode(), sheet="ignored")
    assert resp.status_code == 200
    assert "sheets" not in resp.json() and "sheet_used" not in resp.json()


def test_utf16_unicode_text_export_is_read(client):
    content = _good().to_csv(index=False, sep="\t").encode("utf-16")
    resp = _post(client, content, "export.txt")
    assert resp.status_code == 200, resp.text
    assert resp.json()["target"] == "lipase_titer"


# --- unit level ------------------------------------------------------------------------- #


def test_parse_upload_reports_sheets_and_rule():
    parsed = parse_upload(_xlsx({"Notes": NOTES, "Data": _good()}))
    assert parsed.kind == "excel" and parsed.sheet_used == "Data"
    assert parsed.context() == {"sheets": ["Notes", "Data"], "sheet_used": "Data"}
    assert parse_upload(b"a,b\n1,2\n").context() == {}


def test_table_shape_problem():
    assert table_shape_problem(_good()) is None
    assert table_shape_problem(NOTES) == "too_few_columns"
    assert table_shape_problem(_good(5)) == "too_few_rows"
    assert table_shape_problem(pd.DataFrame({"a": list("abcdefg"), "b": list("hijklmn")})) == (
        "no_numeric_inputs"
    )


def test_upload_rejected_body_never_overrides_error_or_code():
    rej = UploadRejected("msg", code="c", context={"error": "x", "code": "y", "k": 1})
    assert rej.to_body() == {"error": "msg", "code": "c", "k": 1}
