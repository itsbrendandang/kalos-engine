"""Public in-silico fed-batch runs as a benchmark (QUEUE round 3 item 10).

Source: DataHow AG, `hybrid-modeling-summer-school`, MIT License,
Copyright (c) 2023 DataHow, https://github.com/DataHow/hybrid-modeling-summer-school
(checked at commit a002db21de9f42a9fb87da22f860e70c0cc953e9). Files
`data/insilico_{fixed,noisy}_A_{train,test}.csv`: simulated (in-silico) CHO-like
fed-batch runs, 20 train and 100 test runs x 15 daily steps, with states
(`X:`), controls (`W:` temperature, pH; `Z:` stirring, DO) and bolus feeds.

The data is NOT in this repository. Download the four CSV files into a folder
and point `KALOS_DATAHOW_DATA` at it (benchmarks/INSILICO_HYBRID.md). All rows
are simulated, so results on them are evidence about the optimizer on this
simulator, not about any real process.

`endpoint_table` turns the long time series into one row per run, the run-sheet
shape Kalos works on: the settings a lab chooses before or during the run
(temperature and pH at the start and after their one shift, stirring, DO,
starting glucose and glutamine, the bolus feed sizes) and the harvest titer
(the last step) as the target.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = {
    "name": "hybrid-modeling-summer-school (DataHow AG)",
    "url": "https://github.com/DataHow/hybrid-modeling-summer-school",
    "commit": "a002db21de9f42a9fb87da22f860e70c0cc953e9",
    "license": "MIT License, Copyright (c) 2023 DataHow",
    "files": ["insilico_{variant}_A_train.csv", "insilico_{variant}_A_test.csv"],
}

INPUTS = [
    "temp_start", "temp_end", "pH_start", "pH_end", "stir", "DO",
    "Glc_start", "Gln_start", "feed_Glc_bolus", "feed_Gln_bolus",
]
TARGET = "titer_harvest"


def endpoint_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per run from the long format (columns as in the source files)."""
    rows = []
    for _run, g in long.sort_values(["run", "timesteps"]).groupby("run", sort=True):
        first, last = g.iloc[0], g.iloc[-1]
        rows.append({
            "run": int(g["run"].iloc[0]),
            "temp_start": float(first["W:temp"]), "temp_end": float(last["W:temp"]),
            "pH_start": float(first["W:pH"]), "pH_end": float(last["W:pH"]),
            "stir": float(first["Z:Stir"]), "DO": float(first["Z:DO"]),
            "Glc_start": float(first["X:Glc"]), "Gln_start": float(first["X:Gln"]),
            "feed_Glc_bolus": float(g["BolusFeed_Glc:Glc"].max()),
            "feed_Gln_bolus": float(g["BolusFeed_Gln:Gln"].max()),
            TARGET: float(last["X:Titer"]),
        })
    return pd.DataFrame(rows)


def data_dir() -> Path | None:
    raw = os.environ.get("KALOS_DATAHOW_DATA", "").strip()
    return Path(raw).expanduser() if raw else None


def load_runs(directory: Path | str | None = None, *, variant: str = "noisy") -> pd.DataFrame:
    """Train and test runs as one endpoint table with a `split` column.
    Raises FileNotFoundError naming the missing file."""
    if variant not in {"noisy", "fixed"}:
        raise ValueError('variant must be "noisy" or "fixed"')
    base = Path(directory) if directory is not None else data_dir()
    if base is None:
        raise FileNotFoundError("set KALOS_DATAHOW_DATA to the folder with the four CSV files")
    parts = []
    for split in ("train", "test"):
        path = base / f"insilico_{variant}_A_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (see benchmarks/INSILICO_HYBRID.md)")
        table = endpoint_table(pd.read_csv(path))
        table.insert(0, "split", split)
        table["run"] = [f"{split}-{r}" for r in table["run"]]
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def varying_inputs(df: pd.DataFrame) -> list[str]:
    """The inputs that actually vary across runs (constant ones carry no signal)."""
    return [c for c in INPUTS if c in df.columns and float(np.nanstd(df[c].to_numpy(float))) > 0]


__all__ = ["INPUTS", "SOURCE", "TARGET", "data_dir", "endpoint_table", "load_runs", "varying_inputs"]
