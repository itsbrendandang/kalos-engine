"""Loader tests for kalos.bench.open_datasets on tiny fakes in the published layouts.
No real dataset is read or committed: every file is written into tmp_path."""
from __future__ import annotations

import pandas as pd
import pytest

from kalos.bench.open_datasets import load_pichia, open_data_dir


def test_open_data_dir_requires_env(monkeypatch):
    monkeypatch.delenv("KALOS_OPEN_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="KALOS_OPEN_DATA"):
        open_data_dir()


def test_open_data_dir_rejects_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        open_data_dir(tmp_path / "nope")


def test_pichia_drops_index_and_od_outputs(tmp_path):
    d = tmp_path / "AllMolecule"
    d.mkdir()
    pd.DataFrame({
        "": [0, 1, 2],
        "CarbonSource": [14, 10, 0],
        "Carbon_conc": [49.3, 36.7, 0.0],
        "Gly": [9.6, 0.4, 4.0],
        "Met": [3.2, 0.8, 1.5],
        "SP": [2.2, 2.6, 3.7],
        "Titer": [25.9, 31.1, 0.0],
        "OD_OG": [1.3, 4.9, 0.7],
        "OD_Prod": [11.7, 11.9, 4.8],
    }).to_csv(d / "HSA_CoCaBO_Prod_AllData.csv", index=False)
    s = load_pichia("HSA", tmp_path)
    assert list(s.df.columns) == ["CarbonSource", "Carbon_conc", "Gly", "Met", "SP", "Titer"]
    assert s.target == "Titer"
    assert s.roles == {"target": "Titer", "categoricals": ["CarbonSource"]}
    assert s.df["CarbonSource"].tolist() == ["src_14", "src_10", "src_00"]
    assert int((s.df["Titer"] <= 0).sum()) == 1


def test_pichia_unknown_molecule_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown Pichia molecule"):
        load_pichia("NOPE", tmp_path)
