"""Pull mapping on recorded-shape fixtures: a Bioprocess study (v3) and assay runs
(v2) become a Kalos run sheet with a column-provenance table and an import report.
Includes a 1,050-row study and a result schema whose target field has a unit."""
from __future__ import annotations

import csv
import io
import re

import pytest
from benchling_fake import (
    BASE,
    METHOD_VERSION,
    PARAMS,
    PROJECT_ID,
    RESULT_SCHEMA,
    RUN_SCHEMA,
    STUDY_ID,
    TITER_FIELD,
    FakeBenchling,
    bioprocess_study,
    v2_assay,
)

from kalos.integrations.benchling.client import (
    AppCredentials,
    BenchlingClient,
    ClientCredentialsToken,
    TokenBucket,
)
from kalos.integrations.benchling.pull import (
    RUN_ID_COLUMN,
    ImportRefused,
    fetch_assay,
    fetch_study,
    list_projects,
    list_studies,
    map_assay,
    map_study,
    parameter_value,
)


def _client(fake: FakeBenchling) -> BenchlingClient:
    return BenchlingClient(BASE, ClientCredentialsToken(AppCredentials(fake.client_id, fake.client_secret)),
                           transport=fake.transport(), bucket=TokenBucket(10_000, 30.0))


def _rows(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw.decode())))


def _sheet(fake: FakeBenchling, **kw):
    with _client(fake) as c:
        data = fetch_study(c, STUDY_ID, target_field_id=TITER_FIELD, **kw)
    return map_study(data, target_field_id=TITER_FIELD)


def test_lists_studies_v3_and_projects_v2():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=1, replicates=1)
    with _client(fake) as c:
        studies = list_studies(c)
        projects = list_projects(c)
    assert [s["id"] for s in studies] == [STUDY_ID, "stdy_other0001"]
    assert projects == [{"id": PROJECT_ID, "name": "Process development"}]


def test_study_maps_one_row_per_replicate_with_units_and_provenance():
    fake = bioprocess_study(FakeBenchling(), n_runs=2, conditions_per_run=3, replicates=2)
    sheet = _sheet(fake)
    # planned parameters (by name), then measured-only ones, then the target
    assert sheet.columns == [RUN_ID_COLUMN, "Feed rate [mL/h]", "Medium", "pH setpoint", "Temperature [degC]",
                             "Dissolved oxygen (measured) [%]", "Titer [g/L]"]
    assert sheet.target_column == "Titer [g/L]"
    assert len(sheet.rows) == 12 and sheet.report.rows_kept == 12 and sheet.report.skipped == {}
    prov = {p.column: p for p in sheet.provenance}
    # Benchling field id -> column, for every column
    assert prov["Temperature [degC]"].benchling_id == "schver_Rm4nTvWz"
    assert prov["Temperature [degC]"].kind == "planned_parameter"
    assert prov["Temperature [degC]"].template_method_version_ids == [METHOD_VERSION]
    assert prov["Dissolved oxygen (measured) [%]"].kind == "measured_parameter"
    assert prov["Titer [g/L]"].benchling_id == TITER_FIELD and prov["Titer [g/L]"].unit == "g/L"
    assert prov["Titer [g/L]"].role == "target"
    assert {p.benchling_id for p in sheet.provenance if p.role == "input"} == {p[0] for p in PARAMS}
    assert any("measured values only" in n for n in sheet.report.notes)
    # categorical stays text, numbers are numbers
    assert {r["Medium"] for r in sheet.rows} <= {"CD-CHO", "ActiPro"}
    assert all(isinstance(r["Temperature [degC]"], float) for r in sheet.rows)
    # replicates of one condition share the planned setpoints
    first, second = sheet.rows[0], sheet.rows[1]
    assert first["Temperature [degC]"] == second["Temperature [degC]"]
    assert sheet.row_sources[0]["condition_id"] and sheet.row_sources[0]["result_id"].startswith("result_")
    assert sheet.target["schema_id"] == RESULT_SCHEMA and sheet.target["project_id"] == PROJECT_ID
    assert sheet.flowchart_config_version_ids == ["wffccv_Lk3mQpXn"]


def test_filename_records_study_id_and_fetch_time():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    sheet = _sheet(fake)
    assert re.fullmatch(rf"benchling-{STUDY_ID}-\d{{8}}T\d{{6}}Z\.csv", sheet.filename)
    assert sheet.fetched_at.strftime("%Y%m%dT%H%M%SZ") in sheet.filename


def test_skipped_rows_are_counted_with_reasons():
    fake = bioprocess_study(FakeBenchling(), n_runs=2, conditions_per_run=5, replicates=2,
                            missing_target_every=7, duplicate_planned_every=5, text_target_every=9)
    sheet = _sheet(fake)
    r = sheet.report
    assert r.runs_seen == 20
    assert set(r.skipped) <= {"no_result_for_target", "conflicting_values_for_one_parameter", "target_not_numeric"}
    assert r.skipped["no_result_for_target"] >= 1
    assert r.skipped["conflicting_values_for_one_parameter"] >= 1
    assert r.skipped["target_not_numeric"] >= 1
    assert r.rows_kept + sum(r.skipped.values()) == r.runs_seen
    assert all(ex.startswith("prcr_") for exs in r.skipped_examples.values() for ex in exs)


def test_large_study_1050_rows_passes_the_upload_parser_and_preflight():
    fake = bioprocess_study(FakeBenchling(), n_runs=15, conditions_per_run=35, replicates=2, seed=3)
    with _client(fake) as c:
        data = fetch_study(c, STUDY_ID, target_field_id=TITER_FIELD, result_schema_id=RESULT_SCHEMA)
        sent = c.requests_sent
    sheet = map_study(data, target_field_id=TITER_FIELD)
    assert len(sheet.rows) == 1050
    raw = sheet.to_csv()
    assert len(_rows(raw)) == 1050
    # bounded call count: 2 per run + per 100 replicates/values + per 200 linked ids for results
    assert sent <= 115, sent
    # EARLY-ACCESS on every bioprocess call, never on the GA ones
    for call in fake.calls:
        beta = call.path.startswith("/api/v3/procedure") or call.path.endswith("/field-definitions/items")
        assert (call.headers.get("early-access") == "true") is beta, call.path
    # the existing upload checks accept it (parse + every pre-fit refusal; no GP fit)
    from kalos.portal.pipeline import preflight_upload

    parsed = preflight_upload(raw, sheet.target_column, False, sheet.filename)
    assert len(parsed.df) == 1050


def test_result_schema_found_by_scanning_when_not_given():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    sheet = _sheet(fake)
    assert sheet.target["schema_id"] == RESULT_SCHEMA


def test_non_numeric_or_unknown_target_field_is_refused():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    with _client(fake) as c:
        data = fetch_study(c, STUDY_ID, target_field_id=TITER_FIELD, result_schema_id=RESULT_SCHEMA)
    with pytest.raises(ImportRefused):
        map_study(data, target_field_id="tsf_kalos001")  # a text field
    with pytest.raises(ImportRefused):
        map_study(data, target_field_id="tsf_nope")
    with _client(fake) as c, pytest.raises(ImportRefused):
        fetch_study(c, STUDY_ID, target_field_id="tsf_nope")


def test_csv_round_trips_values_exactly():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=3, replicates=1)
    sheet = _sheet(fake)
    rows = _rows(sheet.to_csv())
    for got, want in zip(rows, sheet.rows):
        assert float(got["Titer [g/L]"]) == want["Titer [g/L]"]
        assert got[RUN_ID_COLUMN] == want[RUN_ID_COLUMN]


@pytest.mark.parametrize("raw, want", [
    ({"type": "ft_float", "floatValue": 10.0, "value": "10.0"}, 10.0),
    ({"value": {"type": "ft_float", "floatValue": 1.5, "value": "1.5"}}, 1.5),
    ({"type": "ft_string", "value": "CD-CHO"}, "CD-CHO"),
    ({"type": "ft_string", "value": "7"}, "7"),
    (3, 3.0), ("2.5", 2.5), ("ActiPro", "ActiPro"), (None, None), (True, None),
    ({"type": "ft_float", "value": "nan"}, None),
])
def test_parameter_value_shapes(raw, want):
    assert parameter_value(raw) == want


def test_v2_assay_runs_map_one_row_per_run():
    fake = v2_assay(FakeBenchling(), n_runs=12, multi_result_every=5)
    with _client(fake) as c:
        data = fetch_assay(c, RUN_SCHEMA, "assaysch_v2res001")
    sheet = map_assay(data, target_field="yield")
    assert sheet.columns == [RUN_ID_COLUMN, "temperature", "ph", "Yield [g/L]"]
    assert sheet.report.skipped == {"several_results_for_target": 3}  # runs 0, 5, 10
    assert sheet.report.rows_kept == 9
    assert any("not used" in n for n in sheet.report.notes)  # the text field "operator"
    prov = {p.column: p for p in sheet.provenance}
    assert prov["temperature"].benchling_id == "temperature" and prov["temperature"].kind == "run_field"
    assert prov["Yield [g/L]"].benchling_id == "yield"
    assert sheet.row_sources[0]["v3_id"].startswith("run_")
    assert sheet.filename.startswith(f"benchling-{RUN_SCHEMA}-")


def test_v2_target_missing_is_refused():
    fake = v2_assay(FakeBenchling(), n_runs=3)
    with _client(fake) as c:
        data = fetch_assay(c, RUN_SCHEMA, "assaysch_v2res001")
    with pytest.raises(ImportRefused):
        map_assay(data, target_field="nope")


# --- review fix: spreadsheet formula injection ------------------------------------------------- #


@pytest.mark.parametrize("text, want", [
    ("=HYPERLINK(\"http://x\")", "'=HYPERLINK(\"http://x\")"), ("+cmd", "'+cmd"), ("-abc", "'-abc"),
    ("@SUM(A1)", "'@SUM(A1)"), ("\tx", "'\tx"), ("\rx", "'\rx"), ("-2.5", "-2.5"), ("+1", "+1"),
    ("1e-3", "1e-3"), ("CD-CHO", "CD-CHO"), ("'=x", "'=x"), ("", ""),
])
def test_csv_safe(text, want):
    from kalos.integrations.benchling.pull import csv_safe

    assert csv_safe(text) == want


def test_formula_text_in_headers_and_cells_is_neutralized():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    for v in fake.param_versions:
        if v["name"] == "Medium":
            v["name"] = "=HYPERLINK(\"http://evil\",\"x\")"
    evil = ["=cmd|' /C calc'!A0", "@SUM(1+1)"]
    media = [p for p in fake.planned if p["parameterVersion"]["id"] == "schver_media001"]
    for p, level in zip(media, evil):
        p["value"] = {"type": "ft_string", "value": level}
    sheet = _sheet(fake)
    raw = sheet.to_csv().decode()
    header = next(csv.reader(io.StringIO(raw)))
    assert "'=HYPERLINK(\"http://evil\",\"x\")" in header
    cells = [c for row in csv.reader(io.StringIO(raw)) for c in row]
    assert not any(c[:1] in ("=", "+", "@", "\t", "\r") for c in cells)
    assert not any(c.startswith("-") and not c[1:2].isdigit() for c in cells)
    assert {"'=cmd|' /C calc'!A0", "'@SUM(1+1)"} <= set(cells)
    # rows and provenance hold the same text the CSV has
    col = next(p.column for p in sheet.provenance if p.benchling_id == "schver_media001")
    assert col.startswith("'=") and {r[col] for r in sheet.rows} == {"'=cmd|' /C calc'!A0", "'@SUM(1+1)"}


def test_pushed_levels_get_the_original_text_back():
    from kalos.integrations.benchling.push import _typed

    assert _typed("'=cmd")["value"]["value"] == "=cmd"
    assert _typed("'quoted")["value"]["value"] == "'quoted"


def test_results_are_fetched_only_for_this_study():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    # 500 results of the same schema that belong to other studies
    for i in range(500):
        fake.results.append({
            "__typename": "Result", "id": f"result_other_{i}", "archived": False, "schema": {"id": RESULT_SCHEMA},
            "modifiedAt": "2026-10-02T00:00:00+00:00", "schemaFields": [
                {"fieldDefinition": {"id": TITER_FIELD}, "value": {"__typename": "DecimalValue", "value": "9"}},
                {"fieldDefinition": {"id": "tsf_runlink1"},
                 "value": {"__typename": "ObjectLinkValue", "value": {"id": f"prcr_elsewhere_{i}"}}}]})
    with _client(fake) as c:
        data = fetch_study(c, STUDY_ID, target_field_id=TITER_FIELD, result_schema_id=RESULT_SCHEMA)
    assert len(data.results) == 2
    calls = [x for x in fake.calls if x.path == "/api/v3/result/items"]
    assert calls and all("schemaFieldLinkValue.linkedItemIds" in x.params for x in calls)
