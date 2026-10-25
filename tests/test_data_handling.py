"""Silent data loss in analysis is reported, never silent (D5, D6, D7).

D5. With no declared `roles`, a text factor (a resin, a medium) was dropped as
    `dropped_sparse` and no proposal ever varied it. Low-cardinality repeated text
    columns are now categorical features, and each is reported as a finding so the
    uploader can confirm it or override it with `roles`.

D6. A run with a missing or non-numeric target/feature cell (`n/a`, `ERR`) was
    dropped with `n_dropped_incomplete: 0`. Every drop is now counted per column
    and reason, reported as findings, and echoed in the run record. Censored values
    (`<LOD`, `>100`, `<x`, `>x`) are excluded from the fit, counted separately, and
    never imputed.

D7. A merged-header workbook (a group-label row above the real header) produced
    pandas' `Unnamed: N` names. The next row is now used as the header when the
    first is mostly unnamed. If that is still ambiguous, the upload is refused with
    `ambiguous_header` and the detected columns, and `Unnamed` never reaches a client.
"""
from __future__ import annotations

import io
import json
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient

from kalos.domains import GENERIC_PROFILE, ColumnRoles
from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.analysis import _analyze
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.pipeline import compute_upload_analysis
from kalos.portal.uploads import UploadRejected, parse_upload

N = 24


def _sheet(n: int = N, seed: int = 0) -> pd.DataFrame:
    """A process sheet with a real text factor: three resins, each repeated."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(5.5, 7.5, n).round(3)
    temp = rng.uniform(25, 40, n).round(2)
    resin = ["ABC"[i % 3] for i in range(n)]
    titer = (1.2 * ph - 0.05 * (temp - 32) ** 2 + np.where(np.array(resin) == "B", 2.0, 0.0)
             + rng.normal(0, 0.1, n)).round(3)
    return pd.DataFrame({"pH": ph, "Temp": temp, "Resin": resin, "lipase_titer": titer}).astype(object)


def _features(out: dict) -> set[str]:
    return set(out["features"])


def _findings(out: dict, check: str) -> list[dict]:
    return [f for f in out["validation"]["findings"] if f["check"] == check]


# --- D5: text factors are categorical features ------------------------------ #


def test_text_factor_is_auto_detected_and_modeled():
    out = _analyze(_sheet(), target="lipase_titer")
    assert "Resin" in _features(out)
    assert "Resin" in out["categorical_features"]
    assert out["auto_categorical"] == ["Resin"]
    statuses = {p["name"]: p["status"] for p in out["provenance"]}
    assert statuses["Resin"] == "kept_feature"


def test_auto_categorical_is_reported_as_a_finding_per_column():
    out = _analyze(_sheet(), target="lipase_titer")
    found = _findings(out, "auto_categorical")
    assert [f["column"] for f in found] == ["Resin"]
    assert found[0]["severity"] == "info"
    # the message says how to confirm or override it
    assert "roles.categoricals" in found[0]["message"]


def test_auto_categorical_factor_is_varied_by_proposals():
    out = _analyze(_sheet(), target="lipase_titer")
    levels = {p["recipe"]["Resin"] for p in out["proposals"]}
    assert levels <= {"A", "B", "C"}
    # The factor is part of the design: the recipe carries every modeled feature.
    for p in out["proposals"]:
        assert "Resin" in p["recipe"]


def test_high_cardinality_text_is_not_auto_categorical():
    df = _sheet()
    df["Batch note"] = [f"observation {i} from the shift" for i in range(N)]
    df["Lot code"] = [f"L{i:03d}" for i in range(N)]  # unique per run: not a factor
    out = _analyze(df, target="lipase_titer")
    assert "Batch note" not in out["categorical_features"]
    assert "Lot code" not in out["categorical_features"]


def test_mixed_text_and_numbers_is_not_auto_categorical():
    df = _sheet()
    df["Code"] = ["A", "B", "7", "A", "B", "7"] * (N // 6)  # numbers mixed into text
    out = _analyze(df, target="lipase_titer")
    assert "Code" not in out["categorical_features"]


def test_auto_detection_is_off_when_roles_are_declared():
    """Declaring roles is the uploader's word: undeclared text stays as before."""
    roles = ColumnRoles(target="lipase_titer", features=("pH", "Temp"))
    out = _analyze(_sheet(), roles=roles, profile=GENERIC_PROFILE)
    assert out["auto_categorical"] == []
    assert "Resin" not in _features(out)
    assert _findings(out, "auto_categorical") == []


def test_declared_categorical_is_not_reported_as_auto():
    roles = ColumnRoles(target="lipase_titer", features=("pH", "Temp"), categoricals=("Resin",))
    out = _analyze(_sheet(), roles=roles, profile=GENERIC_PROFILE)
    assert "Resin" in out["categorical_features"]
    assert _findings(out, "auto_categorical") == []


# --- D6: every drop is counted and reported --------------------------------- #


def _with_bad_cells() -> pd.DataFrame:
    df = _sheet()
    df.loc[2, "lipase_titer"] = "n/a"     # missing target
    df.loc[5, "Temp"] = "ERR"             # non-numeric feature
    df.loc[8, "pH"] = "nan"               # a null token in a feature
    return df


def test_dropped_runs_are_counted_not_silent():
    out = _analyze(_with_bad_cells(), target="lipase_titer")
    assert out["n_dropped_incomplete"] == 3
    assert out["n"] == N - 3
    assert out["data_drops"]["rows_dropped"] == 3


def test_dropped_cells_are_counted_per_column_and_reason():
    out = _analyze(_with_bad_cells(), target="lipase_titer")
    by_column = out["data_drops"]["cells_by_column"]
    assert by_column["lipase_titer"] == {"missing": 1}
    assert by_column["Temp"] == {"non_numeric": 1}
    assert by_column["pH"] == {"missing": 1}
    assert out["data_drops"]["cells_by_reason"] == {"missing": 2, "non_numeric": 1, "censored": 0}


def test_each_drop_has_a_finding_naming_the_column():
    out = _analyze(_with_bad_cells(), target="lipase_titer")
    assert _findings(out, "dropped_rows")[0]["severity"] == "warning"
    by_col = {f["column"]: f for f in _findings(out, "dropped_cells")}
    assert set(by_col) == {"lipase_titer", "Temp", "pH"}
    assert "not imputed" in by_col["Temp"]["message"]
    assert out["validation"]["status"] == "pass_with_warnings"


def test_censored_values_are_excluded_counted_separately_and_not_imputed():
    df = _sheet()
    df.loc[4, "lipase_titer"] = "<LOD"
    df.loc[6, "Temp"] = ">100"
    out = _analyze(df, target="lipase_titer")
    assert out["n_censored_excluded"] == 2
    assert out["n_dropped_incomplete"] == 0
    assert out["n"] == N - 2
    assert out["data_drops"]["cells_by_reason"]["censored"] == 2
    cens = {f["column"]: f for f in _findings(out, "censored_values")}
    assert set(cens) == {"lipase_titer", "Temp"}
    assert "excluded" in cens["lipase_titer"]["message"]
    assert "not imputed" in cens["lipase_titer"]["message"]
    rows = _findings(out, "censored_rows")
    assert len(rows) == 1 and "excluded" in rows[0]["message"]


@pytest.mark.parametrize("token", ["<LOD", "<0.5", ">100", "≤2", "BLQ", " <LOQ"])
def test_each_censored_spelling_is_excluded(token):
    df = _sheet()
    df.loc[3, "lipase_titer"] = token
    out = _analyze(df, target="lipase_titer")
    assert out["n_censored_excluded"] == 1, token
    assert out["n"] == N - 1


def test_a_blank_feature_cell_drops_its_run_and_is_reported():
    """Review C1: a blank cell of an ordinary feature (here Temp) is UNKNOWN, not
    0.0. The run is dropped through the row accounting, counted per column, and
    nothing is imputed."""
    df = _sheet()
    df.loc[7, "Temp"] = None
    out = _analyze(df, target="lipase_titer")
    assert out["n"] == N - 1
    assert out["n_dropped_incomplete"] == 1
    assert out["data_drops"]["cells_by_column"]["Temp"] == {"missing": 1}
    assert out["data_drops"]["blank_cells_as_absent"] == {}
    assert _findings(out, "blank_as_absent") == []


def test_a_blank_concentration_cell_is_absent_and_reported():
    """A concentration-type column (glucose, methanol, a salt, ...) keeps the
    sparse-component convention: a blank is 0 (absent), the run is kept, counted."""
    df = _sheet()
    df["Glucose_g_L"] = np.linspace(1.0, 10.0, N)
    df.loc[7, "Glucose_g_L"] = None
    out = _analyze(df, target="lipase_titer")
    assert out["n"] == N
    assert out["n_dropped_incomplete"] == 0
    assert out["data_drops"]["blank_cells_as_absent"] == {"Glucose_g_L": 1}
    assert _findings(out, "blank_as_absent")[0]["column"] == "Glucose_g_L"


def test_too_few_complete_runs_is_a_specific_refusal():
    df = _sheet(n=10)
    df.loc[0:3, "lipase_titer"] = "n/a"   # 6 numeric targets remain ...
    df.loc[4, "pH"] = "ERR"                 # ... but one more run is unusable: 5 left
    with pytest.raises(UploadRejected) as info:
        _analyze(df, target="lipase_titer")
    assert info.value.code == "too_few_complete_rows"
    assert info.value.context["n_dropped_incomplete"] == 5
    assert info.value.context["n_rows"] == 5


def test_run_record_carries_the_drop_counts():
    csv = _with_bad_cells().to_csv(index=False).encode()
    df_csv = pd.read_csv(io.BytesIO(csv), keep_default_na=False, na_values=[""])
    assert df_csv.shape[0] == N
    _, result = compute_upload_analysis(csv, "lipase_titer", False, "runs.csv")
    run = result["run"]
    assert run["data_drops"]["rows_dropped"] == 3
    assert run["data_drops"]["censored_rows_excluded"] == 0
    assert run["data_drops"]["cells_by_reason"]["missing"] == 2
    assert run["data_drops"]["auto_categorical_count"] == 1


def test_anonymized_run_record_carries_counts_but_no_column_names():
    csv = _with_bad_cells().to_csv(index=False).encode()
    _, result = compute_upload_analysis(csv, "lipase_titer", True, "runs.csv")
    run_blob = json.dumps(result["run"])
    assert result["run"]["data_drops"]["rows_dropped"] == 3
    for name in ("Temp", "Resin", "pH"):
        assert name not in run_blob


def test_anonymize_leaves_drop_detail_for_model_columns_unchanged():
    """Only identifier names are aliased; the drop detail is about model inputs and
    the target, which are never identifiers, so the keys are the same as the owner's."""
    plain = _analyze(_with_bad_cells(), target="lipase_titer")
    anon = _analyze(_with_bad_cells(), target="lipase_titer", anonymize=True)
    assert anon["data_drops"]["cells_by_column"] == plain["data_drops"]["cells_by_column"]


# --- D7: merged headers -------------------------------------------------------- #


def _merged_xlsx(group_row: list | None, header_row: list | None, body: list[list]) -> bytes:
    """A workbook with an optional group-label row (each label merged across the
    columns up to the next label) above an optional header row, then data."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Run"
    r = 1
    if group_row is not None:
        labelled = [c for c, v in enumerate(group_row, start=1) if v is not None]
        for c, value in enumerate(group_row, start=1):
            if value is not None:
                ws.cell(row=r, column=c, value=value)
        for i, start in enumerate(labelled):
            end = labelled[i + 1] - 1 if i + 1 < len(labelled) else len(group_row)
            if end > start:
                ws.merge_cells(start_row=r, start_column=start, end_row=r, end_column=end)
        r += 1
    if header_row is not None:
        for c, value in enumerate(header_row, start=1):
            ws.cell(row=r, column=c, value=value)
        r += 1
    for row in body:
        for c, value in enumerate(row, start=1):
            ws.cell(row=r, column=c, value=value)
        r += 1
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _body(n: int = 12, seed: int = 1) -> list[list]:
    rng = np.random.default_rng(seed)
    ph = rng.uniform(5.5, 7.5, n).round(3)
    temp = rng.uniform(25, 40, n).round(1)
    flow = rng.uniform(0.5, 2.0, n).round(2)
    titer = (1.1 * ph + 0.1 * temp + rng.normal(0, 0.1, n)).round(3)
    return [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(ph, temp, flow, titer)]


# A merged group row over four sheet columns: 3 of 4 are blank cells under the merge.
_GROUPS = ["Process", None, None, "Analytics"]
_SUB = ["pH", "Temp", "Flow", "lipase_titer"]


def test_group_label_row_is_skipped_and_the_next_row_is_the_header():
    raw = _merged_xlsx(_GROUPS, _SUB, _body())
    parsed = parse_upload(raw, filename="runs.xlsx")
    assert list(parsed.df.columns) == _SUB
    assert parsed.df.attrs["kalos_header"]["header_row"] == 1
    assert not any("Unnamed" in c for c in parsed.df.columns)


def test_group_label_row_gets_a_finding_in_the_analysis():
    raw = _merged_xlsx(_GROUPS, _SUB, _body())
    parsed = parse_upload(raw, filename="runs.xlsx")
    out = _analyze(parsed.df, target="lipase_titer")
    assert out["header_row"] == 1
    assert _findings(out, "header_row")[0]["severity"] == "info"


def test_title_row_above_the_header_is_skipped():
    raw = _merged_xlsx(["Campaign 7 run sheet"], _SUB, _body())
    parsed = parse_upload(raw, filename="runs.xlsx")
    assert list(parsed.df.columns) == _SUB


def test_merged_header_with_a_group_row_in_csv_is_skipped():
    body = "\n".join(",".join(str(v) for v in row) for row in _body())
    text = ",Process,,Analytics\n" + ",".join(_SUB) + "\n" + body + "\n"
    text = text.replace(",Process,,Analytics", "Process,,,Analytics")
    parsed = parse_upload(text.encode(), filename="runs.csv")
    assert list(parsed.df.columns) == _SUB
    assert parsed.df.attrs["kalos_header"]["header_row"] == 1


def test_a_partial_group_row_is_refused_not_guessed():
    """One gap among four header cells: the sheet does not say whether row 0 is a
    group row (merged cells) or a header with a blank cell above a units row, so it
    is refused, never guessed."""
    raw = _merged_xlsx(["Process", None, "Analytics", "Extra"], _SUB, _body())
    with pytest.raises(UploadRejected) as info:
        parse_upload(raw, filename="runs.xlsx")
    assert info.value.code == "ambiguous_header"


def test_a_headerless_sheet_is_refused_with_its_detected_columns():
    """Blank first two rows: no header can be trusted, so refuse. The columns it
    did detect are listed, with no pandas `Unnamed` name in the message."""
    raw = _merged_xlsx(None, None, [[None, None, None, None]] + _body())
    from kalos.portal.uploads import parse_upload as parse

    with pytest.raises(UploadRejected) as info:
        parse(raw, filename="runs.xlsx")
    rej = info.value
    assert rej.code == "ambiguous_header"
    assert rej.context.get("columns")
    assert "Unnamed" not in rej.message and not any("Unnamed" in c for c in rej.context["columns"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(portal_module.app)
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)


_UNNAMED = re.compile(r"Unnamed")


def test_ambiguous_header_is_a_400_with_the_detected_columns(client):
    raw = _merged_xlsx(None, None, [[None, None, None, None]] + _body())
    resp = client.post(
        "/api/run", files={"file": ("runs.xlsx", raw, "application/octet-stream")},
        data={"target": "lipase_titer"},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == "ambiguous_header"
    assert isinstance(body["columns"], list) and body["columns"]
    assert not _UNNAMED.search(resp.text)


def test_merged_header_upload_analyzes_end_to_end(client):
    raw = _merged_xlsx(_GROUPS, _SUB, _body(n=14))
    resp = client.post(
        "/api/run", files={"file": ("runs.xlsx", raw, "application/octet-stream")},
        data={"target": "lipase_titer"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["run"]["header_row"] == 1
    assert set(body["features"]) == {"pH", "Temp", "Flow"}
    assert not _UNNAMED.search(resp.text)


def test_a_single_unlabelled_data_column_is_named_by_position_and_flagged():
    """A labelled sheet with one blank header cell over real data: never `Unnamed: N`."""
    head = ["pH", None, "lipase_titer"]
    rng = np.random.default_rng(3)
    body = [[float(a), float(b), float(c)] for a, b, c in zip(
        rng.uniform(5.5, 7.5, 12).round(3), rng.uniform(25, 40, 12).round(1),
        rng.uniform(1, 3, 12).round(3))]
    raw = _merged_xlsx(None, head, body)
    parsed = parse_upload(raw, filename="runs.xlsx")
    assert list(parsed.df.columns) == ["pH", "column 2", "lipase_titer"]
    out = _analyze(parsed.df, target="lipase_titer")
    unnamed = _findings(out, "unnamed_column")
    assert [f["column"] for f in unnamed] == ["column 2"]
    assert unnamed[0]["severity"] == "warning"
    assert not _UNNAMED.search(json.dumps(out))


# --- parsing keeps numbers numeric and text as written ----------------------- #


def test_numeric_column_with_null_tokens_is_still_numbers_and_tokens_stay_text():
    text = "pH,Label,lipase_titer\n6.5,A,1.2\nn/a,B,1.4\n6.8,A,<LOD\n7.1,B,1.9\n"
    df = parse_upload(text.encode(), filename="runs.csv").df
    assert df["pH"].tolist()[0] == 6.5 and isinstance(df["pH"].tolist()[0], float)
    assert df["pH"].tolist()[1] == "n/a"
    assert df["Label"].tolist() == ["A", "B", "A", "B"]  # text column untouched


def test_text_column_with_numbers_keeps_its_labels_as_written():
    text = "pH,Strain,lipase_titer\n6.5,2,1.2\n6.8,A,1.4\n7.1,2,1.9\n"
    df = parse_upload(text.encode(), filename="runs.csv").df
    assert df["Strain"].tolist() == ["2", "A", "2"]


def test_campaign_column_kinds_treat_null_tokens_and_censored_as_not_text():
    from kalos.portal.recipe_values import column_kinds

    rows = [{"pH": 6.5}, {"pH": "n/a"}, {"pH": "<LOD"}, {"pH": 7.0}]
    kinds = column_kinds(rows, ["pH"])
    assert kinds["pH"].kind == "numeric"
    assert set(kinds["pH"].levels) == {"6.5", "7.0"}


def test_campaign_column_kinds_still_treat_real_text_as_categorical():
    from kalos.portal.recipe_values import column_kinds

    kinds = column_kinds([{"Resin": "A"}, {"Resin": "n/a"}, {"Resin": "B"}], ["Resin"])
    assert kinds["Resin"].kind == "categorical"
    assert set(kinds["Resin"].levels) == {"A", "B"}
