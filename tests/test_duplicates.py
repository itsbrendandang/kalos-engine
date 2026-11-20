"""BRE-116: duplicate run ids, and rows pasted again under a new id."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from kalos.domains import BIOPROCESS_PROFILE
from kalos.validation import report_dict, validate_frame
from kalos.validation.checks import check_duplicate_ids, check_duplicate_rows

BIO = Path(__file__).resolve().parents[1] / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"


def _sheet() -> pd.DataFrame:
    return pd.read_csv(BIO)


def test_clean_sheet_has_no_duplicate_findings() -> None:
    report = report_dict(validate_frame(_sheet(), target="titer_g_L"))
    assert not [f for f in report["findings"] if f["check"] in ("duplicate_rows", "duplicate_ids")]


def test_two_different_runs_sharing_an_id_are_flagged() -> None:
    df = _sheet()
    df.loc[40, "run_id"] = df.loc[7, "run_id"]
    (f,) = check_duplicate_ids(df)
    assert f.column == "run_id" and f.severity == "warning"
    assert set(f.rows) == {7, 40}
    assert f.detail["shared_by_different_rows"] == [df.loc[7, "run_id"]]
    assert "cannot be traced" in f.message


def test_rows_pasted_again_under_new_ids_are_flagged() -> None:
    df = _sheet()
    copies = df.iloc[[3, 50]].copy()
    copies["run_id"] = ["R161", "R162"]
    df = pd.concat([df, copies], ignore_index=True)
    findings = [f for f in check_duplicate_rows(df, outcome_hint=BIOPROCESS_PROFILE.outcome_hint)
                if f.severity == "warning"]
    assert len(findings) == 1
    assert set(findings[0].rows) == {3, 50, 160, 161}
    assert findings[0].column is None
    assert findings[0].detail["identifier_columns"] == ["run_id"]
    assert check_duplicate_ids(df) == []  # the ids themselves are unique


def test_exact_copies_are_reported_once_as_exact_and_by_their_shared_id() -> None:
    df = _sheet()
    df = pd.concat([df, df.iloc[[9]]], ignore_index=True)
    dup = check_duplicate_rows(df, outcome_hint=BIOPROCESS_PROFILE.outcome_hint)
    assert [f.severity for f in dup] == ["warning"]
    assert "exact copies" in dup[0].message
    (ids,) = check_duplicate_ids(df)
    assert ids.detail["shared_by_identical_rows"] == ["R010"]


def test_a_repeating_batch_column_is_not_an_id_problem() -> None:
    df = pd.DataFrame({"batch": ["B1", "B1", "B2", "B2", "B3", "B3"], "pH": [7, 7.1, 6.9, 7.0, 7.2, 6.8],
                       "titer": [1, 2, 3, 4, 5, 6]})
    assert check_duplicate_ids(df) == []


def test_replicates_differing_only_in_the_outcome_stay_info() -> None:
    df = pd.DataFrame({"run_id": ["R1", "R2", "R3"], "pH": [7.0, 7.0, 6.5], "titer": [1.0, 1.2, 0.8]})
    findings = check_duplicate_rows(df, outcome_hint=BIOPROCESS_PROFILE.outcome_hint)
    assert [f.severity for f in findings] == ["info"]
