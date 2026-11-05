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


def test_scfv_loader_joins_doe1_and_bo_with_seed_mask(tmp_path):
    from kalos.bench.open_datasets import load_scfv_refolding
    doe = pd.DataFrame({
        "Exp No": [1, 2], "Exp Name": ["N1", "N2"], "Run Order": [1, 2], "Incl/Excl": ["Incl", "Incl"],
        "pH": [8.0, 11.0], "final Urea (M)": [6.0, 0.2], "Dilution Factor (-)": [2, 40],
        "GSSG (mM)": [0.0, 2.5], "DTT in solubilized protein (mM)": [0.0, 25.0],
        "Solubilized scFvM concentration (g/L)": [8.2, 8.5], "Refolded scFvM (g/L)": [0.8, 0.0],
        "Refolding Yield (%)": [20.1, 0.0], "Delta AEW (nm)": [3.6, -0.5],
    })
    bo = pd.DataFrame({
        "BO-Iteration": [0, 1], "Experiment ID": ["Exp1", "Exp10"], "DTT [mM]": [3.3, 5.3],
        "GSSG [mM]": [1.3, 2.5], "Dilution Factor": [24.8, 5.3], "pH": [10.3, 10.9],
        "Final Urea [M]": [5.1, 3.0], "Delta AEW [nm]": [7.9, 8.0], "P_Proxy [nm*g/L]": [2.5, 3.0],
        "Refolding yield (HPLC based) [%]": [66.9, 41.8],
        "Native product monomer (HPLC based) [mg/L]": [211.0, 620.3],
        "Yield adjusted concentration [g/L]": [0.14, 0.26],
    })
    d = tmp_path / "Results_Tabular"
    d.mkdir()
    with pd.ExcelWriter(d / "DoE_Data.xlsx") as w:
        doe.to_excel(w, sheet_name="DoE1", index=False)
    bo.to_excel(d / "BO_Data.xlsx", index=False)
    s = load_scfv_refolding(tmp_path)
    assert s.target == "Refolding yield (%)"
    assert s.seed_mask == [True, True, False, False]
    assert list(s.df.columns) == ["pH", "Final Urea (M)", "Dilution Factor", "GSSG (mM)", "DTT (mM)",
                                  "Refolding yield (%)"]
    assert s.df.loc[2, "Refolding yield (%)"] == pytest.approx(66.9)


def test_rudt_decoding_anchors(tmp_path):
    from kalos.bench.open_datasets import (
        RUDT_PH_RANGE,
        RUDT_T_RANGE,
        load_rudt_ib_refolding,
    )
    pd.DataFrame({
        "reactor": ["V01", "V02", "V03"], "ambr_block": [0, 0, 1],
        "is_part_of_ref_block_1": [0, 0, 0], "pH": [-1.0, 0.230769231, 1.0],
        "temperature": [1.0, 0.411764706, -1.0], "feed_rate": [0.0, -0.03, 1.0],
        "refolded_concentration": [0.0004, 0.001, 0.0005],
    }).to_csv(tmp_path / "master table.csv", sep=";", index=False)
    s = load_rudt_ib_refolding(tmp_path)
    assert RUDT_PH_RANGE == (6.2, 7.5) and RUDT_T_RANGE == (18.0, 35.0)
    assert s.df["pH"].tolist() == pytest.approx([6.2, 7.0, 7.5], abs=1e-3)
    assert s.df["temperature_C"].tolist() == pytest.approx([35.0, 30.0, 18.0], abs=1e-3)
    assert s.df["feed_rate_coded"].tolist() == [0.0, -0.03, 1.0]  # not decoded (pseudonymised)


def test_cho_ivcd_is_the_trapezoid_of_offline_vcd(tmp_path):
    from kalos.bench.open_datasets import load_cho_bioreactor
    t = [0.0, 5.0, 10.0]
    vcd = [1e6, 1e6, 1e6]  # constant 1e6 cells/mL over 10 h -> IVCD = 10 (1e6 cells h / mL)
    pp = pd.DataFrame({"Time (h)": t, "pH": [7.0] * 3, "Dissolved O2 (% air sat)": [30.0] * 3,
                       "Temperature (°C)": [37.0] * 3, "Agitation rate (rpm)": [80.0] * 3,
                       "Aeration rate total (sL/h)": [5.0] * 3, "O2 fraction inlet (%)": [21.0] * 3,
                       "CO2 fraction inlet (%)": [1.0] * 3})
    feed = pd.DataFrame({"Time (h)": t, "Cumulative feed weight (g)": [0.0, 0.0, 0.0]})
    naoh = pd.DataFrame({"Time (h)": t, "Cumulative NaOH weight (g)": [0.0, 0.0, 0.0]})
    cell = pd.DataFrame({"Time (h)": t, "VCD Mean (cells/mL)": vcd, "Viability Mean (%)": [90.0, 90.0, 90.0]})
    with pd.ExcelWriter(tmp_path / "EXP001_batch.xlsx") as w:
        for r in ("R01", "R02"):
            pp.to_excel(w, sheet_name=f"{r}_inline_ProcessParameters", index=False)
            feed.to_excel(w, sheet_name=f"{r}_online_Feed", index=False)
            naoh.to_excel(w, sheet_name=f"{r}_online_NaOH", index=False)
            cell.to_excel(w, sheet_name=f"{r}_offline_CellAnalysis", index=False)
    s = load_cho_bioreactor(tmp_path)
    assert s.target == "IVCD_1e6_cells_h_mL"
    assert s.df["IVCD_1e6_cells_h_mL"].tolist() == pytest.approx([10.0, 10.0])
    assert "run_length_h" not in s.df.columns and "NaOH_total_g" not in s.df.columns
