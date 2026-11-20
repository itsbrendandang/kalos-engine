"""F06: columns derived from the result are not settings.

A grade, tier or converted copy of the target (found on content, so a renamed
column is caught), an observed outcome such as contamination (by name), and any
column the user lists in `roles.outputs` are never fitted as inputs or proposed.
Each is reported with why, and an explicit declaration overrides the inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalos.domains import BIOPROCESS_PROFILE, ColumnRoles
from kalos.portal.analysis_parts.columns import _resolve_columns, derived_output_columns
from kalos.portal.analysis_parts.diagnostics import derived_output_findings
from kalos.portal.validate import column_provenance
from kalos.validation.checks import check_target_leak, target_leak_evidence

BIO = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"
T = "titer_g_L"


def _with_grade(name: str = "qc_grade") -> pd.DataFrame:
    df = pd.read_csv(BIO)
    y = df[T]
    q1, q2 = y.quantile(1 / 3), y.quantile(2 / 3)
    df.insert(len(df.columns) - 1, name, np.where(y > q2, "A", np.where(y > q1, "B", "C")))
    return df


def test_clean_example_has_no_leak() -> None:
    assert target_leak_evidence(pd.read_csv(BIO), T) == {}


def test_a_renamed_target_tertile_is_found_on_content() -> None:
    ev = target_leak_evidence(_with_grade("quality_band"), T)
    assert ev["quality_band"]["kind"] == "levels"
    assert ev["quality_band"]["levels_low_to_high"] == ["C", "B", "A"]
    assert ev["quality_band"]["exclude"] is True
    (f,) = check_target_leak(_with_grade("quality_band"), T)
    assert f.check == "target_leak" and f.column == "quality_band" and f.severity == "warning"
    assert "roles.features" in f.message


def test_the_result_in_other_units_is_found() -> None:
    df = pd.read_csv(BIO)
    df["product_mg_L"] = df[T] * 1000 + 0.001
    ev = target_leak_evidence(df, T)["product_mg_L"]
    assert ev["kind"] == "numeric"
    assert ev["exclude"] is False  # reported, never dropped on content: a clean dose-response looks the same
    (f,) = check_target_leak(df, T)
    assert "not excluded" in f.message and "roles.outputs" in f.message


def test_a_factor_named_column_with_a_huge_effect_is_reported_not_excluded() -> None:
    rng = np.random.default_rng(0)
    medium = rng.choice(["A", "B", "C"], 60)
    y = np.select([medium == "A", medium == "B"], [0.1, 0.5], 1.0) + rng.normal(0, 0.05, 60)
    df = pd.DataFrame({"medium": medium, "pH": rng.uniform(6, 7.5, 60), "titer": y})
    ev = target_leak_evidence(df, "titer", factor_hint=BIOPROCESS_PROFILE.group_hint)
    assert ev["medium"]["exclude"] is False
    (f,) = check_target_leak(df, "titer", factor_hint=BIOPROCESS_PROFILE.group_hint)
    assert "not excluded" in f.message and "roles.outputs" in f.message
    assert derived_output_columns(df, "titer", ["medium", "pH"], None, BIOPROCESS_PROFILE) == {}


def test_noisy_real_factor_is_not_a_leak() -> None:
    rng = np.random.default_rng(1)
    medium = rng.choice(["A", "B", "C"], 60)
    y = np.select([medium == "A", medium == "B"], [0.1, 0.5], 1.0) + rng.normal(0, 0.2, 60)
    assert target_leak_evidence(pd.DataFrame({"feed": medium, "titer": y}), "titer") == {}


def test_derived_columns_are_excluded_with_a_reason() -> None:
    df = _with_grade("quality_band")
    df.insert(1, "contamination", np.where(np.arange(len(df)) % 3 == 0, "Y", "N"))
    cands = [c for c in df.columns if c not in ("run_id", T)]
    derived = derived_output_columns(df, T, cands, None, BIOPROCESS_PROFILE)
    assert derived["contamination"]["reason"] == "observed"
    assert derived["quality_band"]["reason"] == "target_leak"
    assert "medium_base" not in derived and "pH" not in derived
    findings = derived_output_findings(derived)
    assert {f.column for f in findings} == {"contamination", "quality_band"}
    assert all("never proposed" in f.message for f in findings)


def test_explicit_feature_declaration_wins_and_outputs_can_be_declared() -> None:
    df = _with_grade("quality_band")
    roles = ColumnRoles.from_dict({"target": T, "categoricals": ["quality_band"], "outputs": ["DO_pct"]})
    derived = derived_output_columns(df, T, ["quality_band", "DO_pct", "pH"], roles, BIOPROCESS_PROFILE)
    assert derived == {"DO_pct": {"reason": "declared"}}
    num = [c for c in df.columns if c not in ("run_id", "medium_base", "quality_band")]
    _t, cont, _cats, _g, _d = _resolve_columns(df, num, T, roles, BIOPROCESS_PROFILE)
    assert "DO_pct" not in cont and "pH" in cont


@pytest.mark.parametrize("bad", [
    {"target": T, "outputs": [T]},
    {"target": T, "features": ["pH"], "outputs": ["pH"]},
    {"target": T, "categoricals": ["medium_base"], "outputs": ["medium_base"]},
])
def test_contradictory_output_declarations_are_refused(bad: dict) -> None:
    with pytest.raises(ValueError):
        ColumnRoles.from_dict(bad)


def test_provenance_reports_derived_columns_as_dropped_output() -> None:
    df = _with_grade()
    rows = column_provenance(
        df, target=T, features=["pH"], numeric_cols=["pH", T],
        is_identifier=lambda n: n == "run_id", outcome_hint=BIOPROCESS_PROFILE.outcome_hint,
        derived_outputs={"qc_grade"},
    )
    assert {r.name: r.status for r in rows}["qc_grade"] == "dropped_output"


@pytest.mark.slow
def test_analysis_never_fits_or_proposes_a_derived_column() -> None:
    from kalos.portal.analysis import _analyze

    df = _with_grade("quality_band").sample(40, random_state=0).reset_index(drop=True)
    df.insert(1, "contamination", np.where(np.arange(len(df)) % 3 == 0, "Y", "N"))
    out = _analyze(df[["run_id", "pH", "temperature_C", "DO_pct", "contamination", "quality_band", T]], T)
    status = {p["name"]: p["status"] for p in out["provenance"]}
    assert status["quality_band"] == "dropped_output"
    assert status["contamination"] == "dropped_output"
    assert "quality_band" not in out["features"] and "contamination" not in out["features"]
    for p in out.get("proposals") or []:
        assert "quality_band" not in p["recipe"] and "contamination" not in p["recipe"]
    checks = {(f["check"], f["column"]) for f in out["validation"]["findings"]}
    assert ("derived_output", "quality_band") in checks
    assert ("target_leak", "quality_band") in checks


def test_a_deterministic_response_to_few_level_settings_is_not_a_leak() -> None:
    # every recipe run twice with the same result: each pH level holds one value
    rng = np.random.default_rng(5)
    ph = rng.uniform(6.0, 7.5, 12).round(3)
    titer = (8 - 2 * (ph - 6.8) ** 2).round(4)
    df = pd.DataFrame({"pH": np.r_[ph, ph], "label": np.r_[ph, ph].astype(str), "titer": np.r_[titer, titer]})
    assert target_leak_evidence(df, "titer") == {}

