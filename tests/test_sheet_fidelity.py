"""Sheet fidelity: what is stored and shown for a sample is what its cells say.

Four defects, each of which changed a sample's value or tag without telling the
user (found by probing tiny synthetic sheets, 2026-11):

1. A cell like `3.25<nbsp>` was "numeric" to the classifier (`float()`) but NaN
   to the converter (`pd.to_numeric`): in a blank-is-absent column it was fitted
   as 0.0 and not counted. One reader now does both (`cells.parse_number`).
2. A date column made `store.seed` fail JSON encoding; the upload still returned
   200 and the campaign held no rows. Dates are stored as ISO text, and the
   response carries `rows_stored`.
3. Run ids lost leading zeros (`001` -> `1`, so `001` and `0001` collided).
4. Numeric category codes with a blank were shown as `1.0`, `2.0`.

Synthetic sheets only.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from kalos.portal.cells import (
    NON_NUMERIC,
    NUMERIC,
    cell_label,
    classify_value,
    parse_number,
    to_numbers,
)
from kalos.portal.uploads import parse_upload

TRICKY = [
    "3.25", " 3.25 ", "3.25\u00a0", "\u20093.25", "-1e3", "+.5", "7.", "0", "001",
    "1_000", "\u0663", "\uff13", "inf", "-Infinity", "1,5", "1.2.3", "3 25", "0x10",
    "", "n/a", "<0.1", "ERR", "1e", ".", "--1",
]


# --- 1. one reader for "is it a number" and "which number" ------------------- #

@pytest.mark.parametrize("text", TRICKY)
def test_classifier_and_converter_always_agree(text):
    kind = classify_value(text)
    value = to_numbers(pd.Series([text], dtype=object)).iloc[0]
    if kind == NUMERIC:
        assert np.isfinite(value), (text, kind, value)
        assert value == parse_number(text)
    else:
        assert np.isnan(value), (text, kind, value)


def test_whitespace_is_trimmed_but_nothing_else_is_cleaned():
    assert parse_number("3.25\u00a0") == 3.25
    assert parse_number("\u20093.25") == 3.25
    for text in ("1_000", "\u0663", "\uff13", "inf", "1,5", "0x10"):
        assert parse_number(text) is None, text
        assert classify_value(text) == NON_NUMERIC, text


def test_a_number_with_an_invisible_space_is_fitted_as_itself_not_zero():
    pytest.importorskip("torch")
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(1)
    n = 16
    glucose = [f"{2 + 0.25 * i:.2f}" for i in range(n)]
    glucose[5] = "3.25\u00a0"   # the planted cell (2 + 0.25*5 = 3.25)
    glucose[3] = "ERR"          # a text cell, so the column stays as written
    df = pd.DataFrame({
        "glucose_g_L": glucose,  # a concentration: blank would mean absent (0)
        "temp_C": [30.0 + i % 5 for i in range(n)],
        "titer_g_L": [round(1 + 0.1 * i + rng.normal(0, 0.05), 3) for i in range(n)],
    })
    out = _analyze(df, target="titer_g_L")
    assert out["data_drops"]["cells_by_column"]["glucose_g_L"] == {"non_numeric": 1}
    assert out["data_drops"]["blank_cells_as_absent"] == {}
    lo, hi = out["feature_ranges"]["glucose_g_L"]
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2 + 0.25 * (n - 1))  # never widened to 0


# --- 3. ids keep their leading zeros ----------------------------------------- #

def _csv(rows: list[str]) -> bytes:
    return ("\n".join(rows) + "\n").encode()


def test_csv_run_ids_keep_leading_zeros_and_do_not_collide():
    p = parse_upload(_csv(["run_id,pH,titer", "001,6.5,1.0", "0001,6.6,1.1", "010,6.7,1.2"]),
                     filename="t.csv")
    assert p.df["run_id"].tolist() == ["001", "0001", "010"]
    assert p.df["pH"].tolist() == [6.5, 6.6, 6.7]          # other columns unchanged


def test_xlsx_text_run_ids_keep_leading_zeros():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["run_id", "pH", "titer"])
    for i, rid in enumerate(["001", "002", "010"]):
        ws.append([rid, 6.5 + 0.1 * i, 1.0 + 0.1 * i])
    buf = io.BytesIO()
    wb.save(buf)
    p = parse_upload(buf.getvalue(), filename="t.xlsx")
    assert p.df["run_id"].tolist() == ["001", "002", "010"]


def test_repeated_zero_padded_levels_are_still_numbers():
    # `05, 10, 05, 20`: repeated values are settings levels, not ids.
    p = parse_upload(_csv(["dose,titer", "05,1.0", "10,1.1", "05,1.2", "20,1.3"]), filename="t.csv")
    assert p.df["dose"].tolist() == [5, 10, 5, 20]


# --- 4. category labels read as the sheet writes them ------------------------ #

def test_cell_label_shows_whole_number_codes_without_a_decimal():
    assert [cell_label(v) for v in (1.0, 2.0, float("nan"), None, 2.5, " A ", 3)] == [
        "1", "2", "", "", "2.5", "A", "3"]


def test_numeric_category_codes_with_a_blank_are_shown_as_written():
    pytest.importorskip("torch")
    from kalos.domains import GENERIC_PROFILE, ColumnRoles
    from kalos.portal.analysis import _analyze

    rng = np.random.default_rng(2)
    rows = ["run,medium_code,pH,titer_g_L"]
    for i in range(16):
        code = "" if i == 4 else str(1 + i % 3)
        rows.append(f"R{i},{code},{6.5 + 0.1 * (i % 6):.1f},{1 + 0.1 * i + rng.normal(0, .05):.3f}")
    p = parse_upload(_csv(rows), filename="t.csv")
    roles = ColumnRoles.from_dict({"target": "titer_g_L", "features": ["pH"],
                                   "categoricals": ["medium_code"], "ids": ["run"]})
    out = _analyze(p.df, "titer_g_L", roles=roles, profile=GENERIC_PROFILE)
    text = json.dumps(out)
    assert '"1.0"' not in text and '"2.0"' not in text and '"3.0"' not in text
    for prop in out["proposals"]:
        assert prop["recipe"]["medium_code"] in {"1", "2", "3"}


# --- 2. a date column is stored, and the response says whether rows were saved #

def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from kalos.portal import app as portal_module
    from kalos.portal import campaign as campaign_module
    from kalos.portal.app import app
    from kalos.portal.campaign import CampaignStore, get_campaign_store

    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps([
        {"token_sha256": _sha("tok"), "subject": "a@acme", "tenant": "acme",
         "scopes": ["read", "write", "admin"]}]))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def test_a_date_column_is_stored_as_iso_text_and_rows_stored_is_true(client):
    from openpyxl import Workbook

    headers = {"Authorization": "Bearer tok"}
    rng = np.random.default_rng(1)
    wb = Workbook()
    ws = wb.active
    ws.append(["run_id", "medium", "glucose_g_L", "temp_C", "harvest_date", "titer_g_L"])
    for i in range(14):
        ws.append([f"R{i + 1}", "A" if i % 2 else "B", 2 + 0.25 * i, 30 + i % 5,
                   datetime.date(2026, 1, i + 1), round(1 + 0.1 * i + rng.normal(0, 0.05), 3)])
    buf = io.BytesIO()
    wb.save(buf)
    made = client.post("/api/campaigns", json={"name": "dates"}, headers=headers).json()
    cid = made.get("id") or made["campaign"]["id"]
    up = client.post(f"/api/campaigns/{cid}/upload", files={"file": ("t.xlsx", buf.getvalue())},
                     headers=headers)
    assert up.status_code == 200, up.text
    assert up.json()["rows_stored"] is True
    rows = client.get(f"/api/campaigns/{cid}/export/rows", headers=headers).json()
    original = rows["original_rows"]["rows"]
    assert len(original) == 14
    assert [r["harvest_date"][:10] for r in original[:3]] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    assert [r["run_id"] for r in original[:3]] == ["R1", "R2", "R3"]
