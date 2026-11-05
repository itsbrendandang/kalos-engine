"""The blood-reactor example is a schema template: roles parse, no data ships,
and the schema runs end to end on a frame of random numbers made in the test."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kalos.domains import GENERIC_PROFILE, ColumnRoles

FOLDER = Path(__file__).resolve().parents[1] / "examples" / "blood_reactor_profile"
DATA_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".db", ".sqlite", ".pkl"}


def _roles_dict() -> dict:
    return json.loads((FOLDER / "roles.json").read_text())


def test_roles_parse_with_eight_features_and_target():
    roles = ColumnRoles.from_dict(_roles_dict())
    assert roles.target == "leukemic_fold_change"
    assert len(roles.features) == 8
    assert len(set(roles.features)) == 8
    assert roles.target not in roles.features
    assert roles.groups == "replicate_group"
    assert set(roles.sparse) <= set(roles.features)


def test_bounds_cover_features_and_are_marked_placeholder():
    b = json.loads((FOLDER / "bounds.json").read_text())
    assert b["placeholder"] is True
    roles = ColumnRoles.from_dict(_roles_dict())
    assert set(b["bounds"]) == set(roles.features)
    for lo, hi in b["bounds"].values():
        assert 0 <= lo < hi


def test_no_data_rows_in_folder():
    for path in FOLDER.rglob("*"):
        if path.suffix.lower() in {".csv", ".tsv"}:
            with path.open(newline="") as fh:
                rows = [r for r in csv.reader(fh) if any(cell.strip() for cell in r)]
            assert len(rows) <= 1, f"{path} has data rows; this folder is schema only"
        else:
            assert path.suffix.lower() not in DATA_SUFFIXES, f"data file {path} found"


def test_readme_states_schema_only():
    text = (FOLDER / "README.md").read_text().lower()
    assert "schema" in text
    assert "no wet-lab data" in text
    assert "not included" in text and "must not" in text


def test_make_template_writes_header_only(tmp_path):
    out = tmp_path / "template.csv"
    subprocess.run(
        [sys.executable, str(FOLDER / "make_template.py"), str(out)],
        check=True, capture_output=True,
    )
    with out.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 1
    roles = ColumnRoles.from_dict(_roles_dict())
    assert set(rows[0]) == {*roles.features, roles.target, roles.groups, *roles.ids}


def _synthetic_frame(n_recipes: int = 10, reps: int = 2, seed: int = 0) -> pd.DataFrame:
    """Random cytokine levels and a MADE-UP smooth response; not biology."""
    rng = np.random.default_rng(seed)
    roles = _roles_dict()
    bounds = json.loads((FOLDER / "bounds.json").read_text())["bounds"]
    feats = roles["features"]
    lo = np.array([bounds[f][0] for f in feats])
    hi = np.array([bounds[f][1] for f in feats])
    X = lo + (hi - lo) * rng.random((n_recipes, len(feats)))
    u = (X - lo) / (hi - lo)
    w = np.linspace(1.0, -0.5, len(feats))
    fc = 1.0 + 2.0 * np.exp(-((u - 0.6) ** 2 @ np.abs(w))) + u @ w * 0.3
    rows = []
    for i in range(n_recipes):
        for r in range(reps):
            rows.append({
                "run_id": f"r{i}_{r}",
                "replicate_group": f"g{i}",
                **dict(zip(feats, X[i], strict=True)),
                "leukemic_fold_change": fc[i] + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_schema_runs_through_analyze_on_synthetic_frame():
    from kalos.portal.analysis import _analyze

    df = _synthetic_frame()
    roles = ColumnRoles.from_dict(_roles_dict())
    out = _analyze(df, roles=roles, profile=GENERIC_PROFILE)
    assert out["target"] == "leukemic_fold_change"
    assert set(out["features"]) == set(roles.features)
    assert len(out["proposals"]) > 0
    assert set(out["proposals"][0]["recipe"]) == set(roles.features)
