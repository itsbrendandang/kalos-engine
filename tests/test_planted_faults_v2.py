"""Planted-fault catalogue v2 (kalos/bench/faults, PROTOCOL_V2.md): the pre-registration.

Torch-free. Marker `planted_faults`.

- the catalogue holds exactly the pre-registered cases, and v1 is untouched by it;
- the suite builds byte-identically and reproduces the committed truth and manifest;
- case folders leak no truth, and are in the v1 layout the blind runner reads;
- the clean bases have the structure PROTOCOL_V2.md section 2 describes;
- every planter changes only what its fault names, and the truth rows point at it.

No detector is run and nothing is scored here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.bench.faults import catalogue as v1
from kalos.bench.faults import catalogue_v2 as cat
from kalos.bench.faults.bases_v2 import (
    ASSAY_LOQ_G_L,
    CONTROL_WELLS,
    GLUCOSE_MW,
    PLATES,
    is_edge_well,
    load_bases_v2,
    well_names,
)
from kalos.bench.faults.cases import CASES_DIR, TRUTH_DIR
from kalos.bench.faults.cases_v2 import (
    MANIFEST_V2,
    TRUTH_V2,
    build_suite_v2,
    case_numbers_v2,
)
from kalos.bench.faults.plant_v2 import PLANTERS_V2, plant_fault_v2

pytestmark = pytest.mark.planted_faults

PKG = Path(cat.__file__).parent

# The pre-registered case list (WS-C PR 0).
PREREGISTERED = {
    "nondetects_as_zero", "placeholder_fill_constant", "mean_imputed_target",
    "volumes_not_concentrations", "stacked_regimes", "unrecorded_harvest_time",
    "failed_control_plate_kept", "sample_id_not_recipe", "well_name_mismatch",
    "derived_column_inconsistent", "contamination_in_notes", "derived_target_linear_combo",
    "label_encoded_by_column", "id_token_feature", "pair_id_across_campaigns",
    "flask_media_mixed", "feature_after_outcome", "edge_well_effect",
}


@pytest.fixture(scope="module")
def bases() -> dict:
    return load_bases_v2()


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("faults-v2")
    build_suite_v2(out)
    return out


# --- catalogue ------------------------------------------------------------------- #


def test_catalogue_holds_exactly_the_preregistered_cases() -> None:
    assert {f.id for f in cat.FAULTS_V2} == PREREGISTERED
    ids = [f.id for f in cat.FAULTS_V2] + [c.id for c in cat.CLEAN_V2]
    assert len(ids) == len(set(ids))
    assert not set(ids) & ({f.id for f in v1.FAULTS} | {c.id for c in v1.CLEAN})
    assert set(cat.REFERENCED_ELSEWHERE) == {"assay_method_changes_with_scale", "transposed_funnel_composite_ids"}
    assert not set(cat.REFERENCED_ELSEWHERE) & set(ids)


def test_every_fault_is_complete() -> None:
    clean = {c.id: c for c in cat.CLEAN_V2}
    for f in cat.FAULTS_V2:
        assert f.parent in clean and clean[f.parent].base == f.base, f.id
        assert f.id in PLANTERS_V2, f.id
        assert f.caught_if and f.expected and f.owner_module and f.owner_plan, f.id
        assert f.surface == "sheet" and f.tiers == ("fast", "full"), f.id
        for m in (*f.caught_if, *f.noticed_if, *f.harm_if):
            if m.planned:
                assert m.codes[0] in cat.PLANNED_CODES and m.source == "validation", f.id
    assert set(PLANTERS_V2) == PREREGISTERED
    used = {m.codes[0] for f in cat.FAULTS_V2 for m in f.caught_if if m.planned}
    assert used == set(cat.PLANNED_CODES)


def test_v2_false_alarm_detectors_include_planned_codes_and_v1() -> None:
    dets = cat.catch_detectors_v2()
    codes = {c for d in dets for c in d.codes}
    assert set(cat.PLANNED_CODES) <= codes
    for d in v1.catch_detectors():
        assert any(x.source == d.source and x.codes == d.codes for x in dets), d
    assert all(d.column is None and not d.rows and not d.planned for d in dets)
    assert all(v1.SEVERITY_RANK[d.min_severity] >= 1 for d in dets)


def test_v1_is_not_changed_by_v2() -> None:
    assert v1.PROTOCOL_VERSION == "1"
    assert len(v1.FAULTS) == 36 and len(v1.CLEAN) == 8
    assert "## Amendments\n\nNone." in (PKG / "PROTOCOL.md").read_text(encoding="utf-8")


def test_protocol_v2_names_every_case_and_code() -> None:
    text = (PKG / "PROTOCOL_V2.md").read_text(encoding="utf-8")
    for name in [*PREREGISTERED, *(c.id for c in cat.CLEAN_V2), *cat.PLANNED_CODES, *cat.REFERENCED_ELSEWHERE]:
        assert f"`{name}`" in text, name
    assert str(cat.SUITE_SEED_V2) in text


# --- build: reproducible, committed, blinded --------------------------------------- #


def test_build_reproduces_the_committed_truth_and_manifest(suite: Path) -> None:
    assert (suite / TRUTH_DIR / "truth.json").read_bytes() == TRUTH_V2.read_bytes()
    assert (suite / "manifest.json").read_bytes() == MANIFEST_V2.read_bytes()
    manifest = json.loads(MANIFEST_V2.read_text())["files"]
    on_disk = sorted(str(p.relative_to(suite / CASES_DIR)) for p in (suite / CASES_DIR).rglob("*") if p.is_file())
    assert sorted(manifest) == on_disk


def test_case_folders_carry_no_truth(suite: Path) -> None:
    cases = suite / CASES_DIR
    names = [f.id for f in cat.FAULTS_V2] + [c.id for c in cat.CLEAN_V2]
    words = {"fault", "planted", "truth", "clean_", "expected", "catalogue"}
    for path in cases.rglob("*"):
        if path.is_dir():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for token in [*names, *words]:
            assert token.lower() not in text, (path, token)
    for path in cases.rglob("case.json"):
        meta = json.loads(path.read_text())
        assert set(meta) == {"surface", "filename", "target"} and meta["surface"] == "sheet"
    truth = json.loads(TRUTH_V2.read_text())
    assert truth["protocol_version"] == cat.PROTOCOL_VERSION_V2 and truth["suite_seed"] == cat.SUITE_SEED_V2
    assert set(truth["cases"]) == set(case_numbers_v2().values())
    ordered = [case_numbers_v2()[f.id] for f in cat.FAULTS_V2]
    assert ordered != sorted(ordered)


def test_every_case_sheet_parses_with_its_target(suite: Path) -> None:
    """The upload parser reads every v2 sheet (no detector is run here)."""
    from kalos.portal.uploads import parse_upload

    for folder in sorted((suite / CASES_DIR).iterdir()):
        meta = json.loads((folder / "case.json").read_text())
        parsed = parse_upload((folder / "sheet.csv").read_bytes(), filename="sheet.csv")
        assert meta["target"] in parsed.df.columns, folder.name


# --- clean bases ---------------------------------------------------------------- #


def test_plate_base_layout(bases: dict) -> None:
    df = bases["plate368"].df
    assert len(df) == 368 and df["run_id"].is_unique
    assert not df.duplicated(["plate", "well"]).any()
    assert set(df["well"]) <= set(well_names())
    for _campaign, plate in PLATES:
        on = df[df["plate"] == plate]
        assert sorted(on.loc[on["sample_type"] == "control", "well"]) == sorted(CONTROL_WELLS)
        assert (on["sample_type"] == "sample").sum() == 88
    samples = df[df["sample_type"] == "sample"]
    per_recipe = samples.groupby("recipe_id")["plate"].agg(["nunique", "count"])
    assert (per_recipe["nunique"] == 2).all() and (per_recipe["count"] == 2).all()
    assert (samples.groupby("recipe_id")["campaign"].nunique() == 1).all()
    assert df["titer_g_L"].min() > 4 * ASSAY_LOQ_G_L
    notes = " ".join(df["notes"].astype(str)).lower()
    for word in ("contamin", "turbid", "fung", "fail", "discard", "no growth"):
        assert word not in notes
    assert any(is_edge_well(w) for w in samples["well"]) and not any(is_edge_well(w) for w in CONTROL_WELLS)


def test_derived_base_is_consistent(bases: dict) -> None:
    df = bases["plate368_derived"].df
    expected = np.round(df["glucose_gL"].to_numpy(float) / GLUCOSE_MW * 1000.0, 2)
    assert np.array_equal(df["glucose_mM"].to_numpy(float), expected)


def test_clone_base_is_crossed_with_labelled_censoring(bases: dict) -> None:
    df = bases["clone64"].df
    assert (df.groupby("clone")["flask_medium"].nunique() == 2).all()
    assert (df.groupby(["clone", "flask_medium"]).size() == 2).all()
    y = df["titer_g_L"].astype(str)
    censored = y == f"<{ASSAY_LOQ_G_L:g}"
    assert 0 < censored.sum() < len(df)
    assert (pd.to_numeric(y[~censored]) > ASSAY_LOQ_G_L).all()
    assert not (pd.to_numeric(y, errors="coerce") == 0).any()


def test_scaleup_feed_base_keeps_the_example_sheet(bases: dict) -> None:
    from kalos.bench.faults.bases import SCALEUP_CSV

    df = bases["scaleup_feed90"].df
    ref = pd.read_csv(SCALEUP_CSV)
    pd.testing.assert_frame_equal(df.drop(columns=["feed_pct"]), ref)
    assert df["feed_pct"].between(10.0, 30.0).all()


# --- planters -------------------------------------------------------------------- #

# Faults that keep the base's shape and columns: only the named cells may change.
_SAME_SHAPE = [
    "nondetects_as_zero", "placeholder_fill_constant", "mean_imputed_target",
    "unrecorded_harvest_time", "failed_control_plate_kept", "well_name_mismatch",
    "derived_column_inconsistent", "contamination_in_notes", "pair_id_across_campaigns",
    "edge_well_effect",
]
# Where the planted columns are not the columns whose cells change.
_CHANGED_COLUMNS = {
    "unrecorded_harvest_time": ["titer_g_L"],
    "failed_control_plate_kept": ["titer_g_L"],
    "derived_column_inconsistent": ["glucose_mM"],
    "edge_well_effect": ["titer_g_L"],
}


@pytest.mark.parametrize("fault_id", _SAME_SHAPE)
def test_planter_changes_only_planted_cells(fault_id: str, bases: dict) -> None:
    fault = cat.fault_by_id(fault_id)
    base = bases[fault.base]
    planted = plant_fault_v2(fault, base)
    clean, dirty = base.df, planted.df
    assert dirty.shape == clean.shape and list(dirty.columns) == list(clean.columns)
    assert planted.rows
    cols = _CHANGED_COLUMNS.get(fault_id, planted.columns)
    changed_rows = set()
    for col in clean.columns:
        for i in range(len(clean)):
            if str(clean.at[i, col]) != str(dirty.at[i, col]):
                assert col in cols and i in planted.rows, (fault_id, i, col)
                changed_rows.add(i)
    assert changed_rows, fault_id
    # Planted rows can be wider than the changed cells: a failed plate's noise can
    # round to no change, and a reused key links the untouched first-campaign rows.
    if fault_id not in ("failed_control_plate_kept", "pair_id_across_campaigns"):
        assert changed_rows == set(planted.rows), fault_id


@pytest.mark.parametrize("fault_id", ["volumes_not_concentrations", "derived_target_linear_combo",
                                      "label_encoded_by_column", "feature_after_outcome",
                                      "id_token_feature", "sample_id_not_recipe"])
def test_column_planters_touch_only_their_columns(fault_id: str, bases: dict) -> None:
    fault = cat.fault_by_id(fault_id)
    base = bases[fault.base]
    planted = plant_fault_v2(fault, base)
    clean, dirty = base.df, planted.df
    assert len(dirty) == len(clean)
    shared = [c for c in clean.columns if c in dirty.columns and c not in planted.columns]
    pd.testing.assert_frame_equal(dirty[shared], clean[shared], check_exact=False, atol=1e-3)
    assert set(dirty.columns) - set(clean.columns) <= set(planted.columns)


def test_specific_planted_structure(bases: dict) -> None:
    plate = bases["plate368"]
    # the linear combination is exact at the sheet's precision
    p = plant_fault_v2(cat.fault_by_id("derived_target_linear_combo"), plate).df
    assert np.allclose(p["supernatant_g_L"] + p["cell_assoc_g_L"], p["titer_g_L"], atol=1e-9)
    # the token is literally the number inside the run id
    p = plant_fault_v2(cat.fault_by_id("id_token_feature"), plate).df
    assert (p["run_id"].str[1:].astype(int) == p["tube_no"]).all() and p["run_id"].is_unique
    # sample ids: replicates share an id, and the planted ids each cover two recipes
    pl = plant_fault_v2(cat.fault_by_id("sample_id_not_recipe"), plate)
    key_cols = ["medium_base", "pH_initial", "glucose_gL", "glutamine_mM", "feed_pct", "osmolality_mOsm"]
    recipes = pl.df.groupby("sample_id")[key_cols].nunique().max(axis=1)
    assert set(pl.df.loc[pl.rows, "sample_id"]) == set(recipes[recipes > 1].index)
    # restarted recipe keys span both campaigns
    pr = plant_fault_v2(cat.fault_by_id("pair_id_across_campaigns"), plate)
    spans = pr.df[pr.df["sample_type"] == "sample"].groupby("recipe_id")["campaign"].nunique()
    assert (spans == 2).all()
    # stacked regimes: >100x span in working volume, two separated titer groups
    ps = plant_fault_v2(cat.fault_by_id("stacked_regimes"), plate).df
    vol = ps["working_volume_mL"]
    assert vol.max() / vol.min() > 100
    assert ps.loc[vol > 1, "titer_g_L"].median() > 2 * ps.loc[vol == 1, "titer_g_L"].median()
    # wells: three zero-padded duplicates and two impossible positions on one plate
    pw = plant_fault_v2(cat.fault_by_id("well_name_mismatch"), plate)
    norm = pw.df["well"].str[0] + pw.df["well"].str[1:].astype(int).astype(str)
    assert norm.isin(well_names()).sum() == len(pw.df) - 2
    assert pw.df.assign(w=norm).duplicated(["plate", "w"], keep=False).sum() == 6
    # flask media: every clone in one medium only
    pf = plant_fault_v2(cat.fault_by_id("flask_media_mixed"), bases["clone64"]).df
    assert (pf.groupby("clone")["flask_medium"].nunique() == 1).all() and pf["flask_medium"].nunique() == 2
    # the status column predicts the censored label at > 0.95 balanced accuracy, not exactly
    pc = plant_fault_v2(cat.fault_by_id("label_encoded_by_column"), bases["clone64"]).df
    label = pc["titer_g_L"].astype(str).str.startswith("<")
    pred = pc["expansion_status"] == "terminated"
    bal = 0.5 * ((pred & label).sum() / label.sum() + (~pred & ~label).sum() / (~label).sum())
    assert 0.95 < bal < 1.0
    # zeros replace exactly the labelled non-detects
    pz = plant_fault_v2(cat.fault_by_id("nondetects_as_zero"), bases["clone64"])
    assert (pd.to_numeric(pz.df["titer_g_L"]).iloc[pz.rows] == 0).all()


def test_truth_rows_are_the_planted_rows_after_shuffling(suite: Path, bases: dict) -> None:
    truth = json.loads(TRUTH_V2.read_text())["cases"]
    for rec in truth.values():
        if rec["kind"] != "fault" or not rec["rows"]:
            continue
        fault = cat.fault_by_id(rec["id"])
        planted = plant_fault_v2(fault, bases[fault.base])
        sheet = pd.read_csv(suite / CASES_DIR / rec["case"] / "sheet.csv", keep_default_na=False)
        assert len(sheet) == rec["n_sheet_rows"] == len(planted.df)
        want = sorted(planted.df["run_id"].iloc[planted.rows].astype(str))
        assert sorted(sheet["run_id"].iloc[rec["rows"]].astype(str)) == want, rec["id"]
