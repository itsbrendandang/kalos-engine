"""Loaders that shape open published datasets into run sheets.

Used by `examples/open_replay_e2e.py` (end-to-end HTTP run) and by the pool replay.
Nothing here bundles data. Every loader reads files from the directory named by
`KALOS_OPEN_DATA` (or an explicit path), and fails with a clear message if the
files are absent. The Amii sets use `KALOS_AMII_DATA` (see `kalos.bench.amii`).

Each loader returns `OpenSheet(df, target, roles, notes)`:
  - `df`: one row per measured run; feature columns plus the target column.
    Identifiers, other measured outputs (OD, sd, replicate reads) and index
    columns are NOT included. Each dropped column is listed in `notes`.
  - `target`: the outcome column the engine should maximize.
  - `roles`: a `roles` dict for `POST /api/run` (categoricals, sparse), or None.

Datasets and their DOI / licence strings are documented in benchmarks/OPEN_REPLAY.md.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

OPEN_DATA_ENV = "KALOS_OPEN_DATA"


@dataclass
class OpenSheet:
    df: pd.DataFrame
    target: str
    roles: dict | None
    notes: list[str] = field(default_factory=list)


def open_data_dir(explicit: str | Path | None = None) -> Path:
    raw = explicit or os.environ.get(OPEN_DATA_ENV)
    if not raw:
        raise FileNotFoundError(
            f"set {OPEN_DATA_ENV} to the folder holding the downloaded open datasets "
            "(see benchmarks/OPEN_REPLAY.md for file names and DOIs)"
        )
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"{OPEN_DATA_ENV}={raw!r} is not a directory")
    return p


def _find(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern!r} under {root}")
    return hits[0]


# --- Pichia pastoris CoCaBO media (zenodo 15466161) ---------------------------- #

PICHIA_MOLECULES = {"HSA": "HSA", "HerceptinA": "HerceptinA", "RBDJ": "RBDJ"}
# CarbonSource is an integer code (0-18) in the source. Its name table is in
# CarbonSourceInfo.xlsx; the code is kept as a labelled text level here so the
# engine treats it as the categorical it is.
_PICHIA_OUTPUTS = ("OD_OG", "OD_Prod")


def load_pichia(molecule: str, root: Path | None = None) -> OpenSheet:
    if molecule not in PICHIA_MOLECULES:
        raise ValueError(f"unknown Pichia molecule {molecule!r}; choose from {sorted(PICHIA_MOLECULES)}")
    root = root or open_data_dir()
    path = _find(root, f"{molecule}_CoCaBO_Prod_AllData.csv")
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "CarbonSource": "src_" + raw["CarbonSource"].astype(int).astype(str).str.zfill(2),
        "Carbon_conc": raw["Carbon_conc"],
        "Gly": raw["Gly"],
        "Met": raw["Met"],
        "SP": raw["SP"],
        "Titer": raw["Titer"],
    })
    notes = [
        f"dropped index column 'Unnamed: 0' and measured outputs {list(_PICHIA_OUTPUTS)} (not inputs)",
        "CarbonSource integer code (0-18) mapped to text level 'src_NN' and declared categorical",
        f"{int((df['Titer'] <= 0).sum())} of {len(df)} runs have Titer == 0 (non-producers, kept)",
    ]
    return OpenSheet(df, "Titer", {"target": "Titer", "categoricals": ["CarbonSource"]}, notes)


# --- Trichoderma ambr250 (zenodo 5846028, Tamminen 2022) ----------------------- #

_AMBR_SHEET = re.compile(r"^(D-161646|T2) (.+) \((1|2)\)$")


def load_trichoderma_ambr(root: Path | None = None, *, include_strain: bool = True) -> OpenSheet:
    """One row per culture (10 conditions x 2 duplicates). The endpoint is the
    LAST offline measurement of 'Mucic acid (g/L)' in each culture's sheet."""
    root = root or open_data_dir()
    path = _find(root, "*Tamminen2022_ambr*.xlsx")
    book = pd.ExcelFile(path)
    rows, skipped = [], []
    for sheet in (str(x) for x in book.sheet_names):
        m = _AMBR_SHEET.match(sheet.strip())
        if not m:
            continue  # 'Description', 'Feed composition'
        raw = book.parse(sheet, header=0)  # type: ignore[attr-defined]
        col = [c for c in raw.columns if str(c).strip().startswith("Mucic acid")]
        if not col:
            skipped.append(sheet)
            continue
        s = pd.to_numeric(raw[col[0]], errors="coerce").dropna()
        if s.empty:
            skipped.append(sheet)
            continue
        end_idx = s.index[-1]
        rows.append({
            "strain": m.group(1),
            "condition": f"{m.group(1)} {m.group(2)}",
            "replicate": int(m.group(3)),
            "end_time_h": float(pd.to_numeric(raw.loc[end_idx, "Time (h)"], errors="coerce")),
            "mucic_acid_g_L": float(s.iloc[-1]),
        })
    df = pd.DataFrame(rows)
    notes = [
        "one row per culture sheet; endpoint = last offline 'Mucic acid (g/L)' value",
        "condition = sheet-name label (strain + condition); YE/glucose/pH are NOT parsed numerically "
        "because the baseline medium of the pH-only and YE/glucose-only sheets is not stated",
        "end_time_h (time of the endpoint) is not an input and is left out of the sheet",
        "CDW (g/L) is a second endpoint in the same sheets and is not modelled here",
    ]
    if skipped:
        notes.append(f"sheets with no mucic-acid endpoint, skipped: {skipped}")
    df = df.drop(columns=["end_time_h", "replicate"])
    if include_strain:
        # The condition label already names the strain ('T2 pH 4'), so strain and
        # condition are NOT independent factors: a sheet that declares both lets the
        # engine propose combinations that were never run (see RUN-REPORT).
        return OpenSheet(
            df, "mucic_acid_g_L", {"target": "mucic_acid_g_L", "categoricals": ["strain", "condition"]},
            notes + ["variant: strain + condition as two categoricals (as published columns)"],
        )
    # Keep the strain column but declare it an identifier: a 2-column sheet (label +
    # target) is parsed as transposed by the engine and refused (see RUN-REPORT).
    return OpenSheet(
        df, "mucic_acid_g_L",
        {"target": "mucic_acid_g_L", "categoricals": ["condition"], "ids": ["strain"]},
        notes + ["variant: condition is the only factor; strain is declared an id (implied by the label)"],
    )


# --- Candida utilis RSM Nr1 (zenodo 22277916) ---------------------------------- #

def load_candida_rsm(root: Path | None = None) -> OpenSheet:
    """RSM Nr1: 2 factors (molasses 80-120 g/L, ammonium 5-20 g/L), 13 runs
    (4 factorial + 4 axial + 5 centre points), target biomass (g/L) at day 5."""
    root = root or open_data_dir()
    path = _find(root, "*Cutilis*.xlsx")
    book = pd.ExcelFile(path)
    sheet = next(str(s) for s in book.sheet_names if str(s).strip().startswith("6.") and "RSM Nr1" in str(s))
    raw = book.parse(sheet, header=None)  # type: ignore[attr-defined]
    # Columns (0-based, header=None): 2 'Std.', 3 'Speak' (run order), 4 molasses,
    # 5 ammonium, 6 biomass (response 1), 7 proteins (response 2). The header is
    # followed by a units row ('g/l'), then the runs until the first non-numeric Std.
    hdr = next(i for i in range(len(raw)) if str(raw.iat[i, 2]).strip() == "Std.")
    rows = []
    for i in range(hdr + 1, len(raw)):
        std = pd.to_numeric(raw.iat[i, 2], errors="coerce")
        if pd.isna(std):
            continue
        rows.append({
            "molasses_g_L": float(raw.iat[i, 4]),
            "ammonium_g_L": float(raw.iat[i, 5]),
            "biomass_g_L": float(raw.iat[i, 6]),
        })
    df = pd.DataFrame(rows)
    notes = [
        f"{len(df)} runs from sheet {sheet.strip()!r}; the 'Std.' and 'Speak' run-order columns and the "
        "protein responses ('Proteins', 'Protein %') are dropped (identifier / other outputs)",
        "5 centre points (molasses 100, ammonium 12.5) are the replicate-noise estimate",
    ]
    return OpenSheet(df, "biomass_g_L", {"target": "biomass_g_L"}, notes)


# --- Amii sets (kalos.bench.amii REGISTRY; KALOS_AMII_DATA) -------------------- #

def load_amii_sheet(name: str, data_dir: str | Path | None = None) -> OpenSheet:
    """The Amii set as a run sheet: the same feature/target choices as
    `kalos.bench.amii.REGISTRY`, but one row per raw measurement (no averaging)."""
    from kalos.bench.amii import REGISTRY, amii_data_dir

    spec = REGISTRY[name]
    root = amii_data_dir(data_dir)
    if root is None:
        raise FileNotFoundError("set KALOS_AMII_DATA to the folder holding the Amii CSVs")
    df = pd.read_csv(root / spec.filename)
    notes: list[str] = []
    if spec.fidelity_col:
        top = df[spec.fidelity_col].max()
        notes.append(f"kept Fidelity == {top} rows only ({int((df[spec.fidelity_col] == top).sum())} of {len(df)})")
        df = df[df[spec.fidelity_col] == top]
    if spec.transform == "log1p_over_poscont":
        target_col = "log_cell_count_over_poscont"
        df[target_col] = np.log1p(
            pd.to_numeric(df[spec.target], errors="coerce") / pd.to_numeric(df["PosCont FC"], errors="coerce")
        )
        notes.append("target = log1p(Cell count / PosCont FC), as in the registry")
    elif spec.transform == "log1p":
        target_col = "log1p_" + re.sub(r"\W+", "_", spec.target).strip("_")
        df[target_col] = np.log1p(pd.to_numeric(df[spec.target], errors="coerce"))
        notes.append(f"target = log1p({spec.target}), as in the registry")
    else:
        target_col = spec.target
    feats = list(spec.features) if spec.features else [
        c for c in df.columns if c not in set(spec.drop) | set(spec.label_cols) | {target_col}
    ]
    if spec.fidelity_col and spec.fidelity_col in feats:
        feats.remove(spec.fidelity_col)
    dropped = [c for c in df.columns if c not in feats and c != target_col]
    notes.append(f"dropped non-input columns: {dropped}")
    out = df[feats + [target_col]].copy()
    out.columns = [str(c).strip() for c in out.columns]
    if spec.name == "DBO_rat_myocyte":
        notes.append(f"IS_indicator (info-source, values {sorted(df['IS_indicator'].unique().tolist())}) "
                     "is NOT a feature: sources are mixed, so results are unresolved (see amii.py)")
    tgt = target_col
    return OpenSheet(out, tgt, None, notes)


# --- registry ------------------------------------------------------------------ #

OPEN_SETS: dict[str, Callable[[Path | None], OpenSheet]] = {
    "pichia_HSA": partial(load_pichia, "HSA"),
    "pichia_HerceptinA": partial(load_pichia, "HerceptinA"),
    "pichia_RBDJ": partial(load_pichia, "RBDJ"),
    "trichoderma_ambr": load_trichoderma_ambr,
    "trichoderma_ambr_condition": partial(load_trichoderma_ambr, include_strain=False),
    "candida_rsm_nr1": load_candida_rsm,
}
AMII_SETS = ("DBO_rat_myocyte", "MOBO_rat_myocyte", "HeLa_regular", "HeLa_timesaving",
             "Tcell_expansion", "TF1_expansion")


def load_open_sheet(name: str, root: str | Path | None = None) -> OpenSheet:
    """Load any registered set by name: `OPEN_SETS` keys (root = folder of the
    open files, default `KALOS_OPEN_DATA`) or `AMII_SETS` names (root = folder of
    the Amii CSVs, default `KALOS_AMII_DATA`)."""
    if name in OPEN_SETS:
        return OPEN_SETS[name](Path(root) if root else None)
    if name in AMII_SETS:
        return load_amii_sheet(name, root)
    raise KeyError(f"unknown dataset {name!r}")
