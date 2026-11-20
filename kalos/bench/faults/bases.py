"""Clean bases: deterministic frames built from the repo's example sheets
(plus the optional local Kaggle sheet), exactly as PROTOCOL.md section 2 says.

Every random draw uses `np.random.default_rng((SUITE_SEED, k))` with a fixed `k`
per use, so a base never depends on the order bases are built in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogue import SUITE_SEED

REPO_ROOT = Path(__file__).resolve().parents[3]
BIOPROCESS_CSV = REPO_ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"
SCALEUP_CSV = REPO_ROOT / "examples" / "synthetic_scaleup" / "synthetic_scaleup.csv"
KAGGLE_NAME = "kaggle_fermentation_dataset_0.csv"

BIOPROCESS_TARGET = "titer_g_L"
SCALEUP_TARGET = "titer_g_per_L"
KAGGLE_TARGET = "max_yield_fpu_ml"
SCALE_PROCESS_COLUMNS = ("ph_setpoint", "temperature_C")
KAGGLE_KEEP = (
    "run_id", "temp_c", "ph", "inoculum_vol_ml", "rpm", "lactose_feed_ml", "duration_days",
    KAGGLE_TARGET,
)
# Removed from the Kaggle base (PROTOCOL.md section 2) and kept aside so a fault
# can plant them back: a response, a derived total, an outcome, a target-derived
# label and a provenance flag.
KAGGLE_EXTRAS = ("min_do_pct", "total_media_vol_ml", "contamination", "yield_tier", "data_type")
RESULTS_HOLDOUT = 5
RESULTS_MIN_TITER = 0.3

# Fixed stream ids for `default_rng((SUITE_SEED, k))`.
_RNG_BIOPROCESS60 = 1
_RNG_BIOPROCESS30 = 2
_RNG_RESULTS = 3
_RNG_KAGGLE = 4


def rng_for(k: int) -> np.random.Generator:
    return np.random.default_rng((SUITE_SEED, k))


def default_datasets_dir() -> Path:
    """`KALOS_DATASETS_DIR`, else `~/kalos/datasets` (where the Kaggle sheet lives)."""
    return Path(os.environ.get("KALOS_DATASETS_DIR") or "~/kalos/datasets").expanduser()


@dataclass(frozen=True)
class Base:
    """A clean frame plus what a planter may need besides it."""

    id: str
    df: pd.DataFrame
    target: str
    holdout: pd.DataFrame | None = None
    extras: dict[str, pd.Series] = field(default_factory=dict)


def _sample(df: pd.DataFrame, n: int, k: int) -> pd.DataFrame:
    idx = np.sort(rng_for(k).choice(len(df), size=n, replace=False))
    return df.iloc[idx].reset_index(drop=True)


def _kaggle(datasets_dir: Path | None) -> Base | None:
    path = (datasets_dir or default_datasets_dir()) / KAGGLE_NAME
    if not path.is_file():
        return None
    raw = pd.read_csv(path)
    raw = raw.drop_duplicates(subset="run_id", keep="first").reset_index(drop=True)
    real = np.flatnonzero((raw["data_type"] == "Real").to_numpy())
    synthetic = np.flatnonzero((raw["data_type"] != "Real").to_numpy())
    n_syn = 150 - len(real)
    pick = rng_for(_RNG_KAGGLE).choice(synthetic, size=n_syn, replace=False)
    idx = np.sort(np.concatenate([real, pick]))
    chosen = raw.iloc[idx].reset_index(drop=True)
    extras = {c: chosen[c].copy() for c in KAGGLE_EXTRAS}
    return Base("kaggle150", chosen[list(KAGGLE_KEEP)].copy(), KAGGLE_TARGET, extras=extras)


def load_bases(datasets_dir: Path | None = None) -> dict[str, Base | None]:
    """Every base the catalogue names; an unavailable one maps to None."""
    bio = pd.read_csv(BIOPROCESS_CSV)
    scale = pd.read_csv(SCALEUP_CSV)

    producers = np.flatnonzero((bio[BIOPROCESS_TARGET] > RESULTS_MIN_TITER).to_numpy())
    held = np.sort(rng_for(_RNG_RESULTS).choice(producers, size=RESULTS_HOLDOUT, replace=False))
    keep = np.setdiff1d(np.arange(len(bio)), held)

    bases: dict[str, Base | None] = {
        "bioprocess160": Base("bioprocess160", bio.copy(), BIOPROCESS_TARGET),
        "bioprocess60": Base("bioprocess60", _sample(bio, 60, _RNG_BIOPROCESS60), BIOPROCESS_TARGET),
        "bioprocess30": Base("bioprocess30", _sample(bio, 30, _RNG_BIOPROCESS30), BIOPROCESS_TARGET),
        "results155": Base(
            "results155", bio.iloc[keep].reset_index(drop=True), BIOPROCESS_TARGET,
            holdout=bio.iloc[held].reset_index(drop=True),
        ),
        "scaleup90": Base("scaleup90", scale.copy(), SCALEUP_TARGET),
        "kaggle150": _kaggle(datasets_dir),
    }
    for limit in (10, 300, 1000):
        sub = scale[scale["scale_L"] <= limit].reset_index(drop=True)
        bases[f"scaleup_le{limit}"] = Base(f"scaleup_le{limit}", sub, SCALEUP_TARGET)
    return bases


__all__ = [
    "BIOPROCESS_TARGET", "KAGGLE_TARGET", "REPO_ROOT", "SCALEUP_TARGET", "SCALE_PROCESS_COLUMNS",
    "Base", "default_datasets_dir", "load_bases", "rng_for",
]
