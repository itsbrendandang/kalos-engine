"""Clean bases for catalogue v2 (PROTOCOL_V2.md section 2).

Every v2 base is generated here from a fixed seed, or derived from a sheet that
ships in `examples/`. Nothing is read from outside the repo. The generators are
synthetic benchmark fixtures: their outcomes are computed, not measured, and
they only ever feed this suite.

- `plate368`: a two-round media screen in 96-deep-well plates. Two campaigns,
  two plates each; every recipe sits in one well on each plate of its campaign
  (replicates on different plates), four reference-recipe control wells per
  plate, wells assigned at random, a free-text notes column.
- `plate368_derived`: `plate368` plus `glucose_mM`, computed from `glucose_gL`.
- `clone64`: a shake-flask clone comparison. 16 clones, each in both flask media
  (crossed), duplicate flasks, the previous-stage (96-deep-well) titer, and
  non-producing flasks recorded as `<0.05` (below the assay's quantitation limit).
- `scaleup_feed90`: `examples/synthetic_scaleup` plus a feed column in % v/v of
  the starting working volume (seeded; no effect on titer in this base).

Random draws use `np.random.default_rng((SUITE_SEED_V2, k))` with a fixed `k` per
use, so a base never depends on the order bases are built in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bases import SCALEUP_CSV, Base
from .catalogue_v2 import SUITE_SEED_V2

PLATE_TARGET = "titer_g_L"
CLONE_TARGET = "titer_g_L"
SCALEUP_TARGET = "titer_g_per_L"

# Reporting limit of the (synthetic) titer assay, in g/L. Written into the clone
# sheet as `<0.05`; never written as a column anywhere.
ASSAY_LOQ_G_L = 0.05
GLUCOSE_MW = 180.156  # g/mol

ROWS = "ABCDEFGH"
N_COLS = 12
CONTROL_WELLS = ("B3", "C10", "F4", "G9")  # interior, spread over the plate
PLATES = (("C1", "P1"), ("C1", "P2"), ("C2", "P3"), ("C2", "P4"))
RECIPES_PER_CAMPAIGN = 88

PLATE_FACTORS: dict[str, tuple[float, float, int]] = {  # name -> (low, high, decimals)
    "pH_initial": (6.80, 7.40, 2),
    "glucose_gL": (4.0, 12.0, 2),
    "glutamine_mM": (2.0, 8.0, 2),
    "feed_pct": (0.0, 15.0, 1),
    "osmolality_mOsm": (280.0, 340.0, 0),
}
CONTROL_RECIPE = {"medium_base": "basal_A", "pH_initial": 7.10, "glucose_gL": 8.0,
                  "glutamine_mM": 5.0, "feed_pct": 9.0, "osmolality_mOsm": 305.0}
BENIGN_NOTES = (
    "breathable seal replaced after sampling",
    "extra sample taken for metabolites",
    "plate map checked against pipetting log",
    "foam on day 3, settled without antifoam",
)

# Fixed stream ids for `default_rng((SUITE_SEED_V2, k))`.
_RNG_DESIGN = 1
_RNG_LAYOUT = 2
_RNG_NOISE = 3
_RNG_NOTES = 4
_RNG_CLONES = 5
_RNG_FEED = 6


def rng_v2(k: int) -> np.random.Generator:
    return np.random.default_rng((SUITE_SEED_V2, k))


def well_names() -> list[str]:
    return [f"{r}{c}" for r in ROWS for c in range(1, N_COLS + 1)]


def is_edge_well(well: str) -> bool:
    return well[0] in (ROWS[0], ROWS[-1]) or int(well[1:]) in (1, N_COLS)


def _lhs(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Latin hypercube on [0, 1)^d: one point per stratum in every dimension."""
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def _bump(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.asarray(np.exp(-0.5 * ((x - mu) / sd) ** 2))


def plate_mean_titer(df: pd.DataFrame) -> np.ndarray:
    """Noise-free titer (g/L) of a recipe in this synthetic 96-deep-well screen.

    A smooth response with an optimum near pH 7.1, saturating in glucose; the
    shape follows `examples/synthetic_bioprocess/make_synthetic.py`. Lowest value
    over the design box is about 0.4 g/L, well above `ASSAY_LOQ_G_L`.
    """
    ph = df["pH_initial"].to_numpy(float)
    glc = df["glucose_gL"].to_numpy(float)
    gln = df["glutamine_mM"].to_numpy(float)
    feed = df["feed_pct"].to_numpy(float)
    osm = df["osmolality_mOsm"].to_numpy(float)
    medium = np.where(df["medium_base"].astype(str).to_numpy() == "basal_B", 0.88, 1.0)
    return np.asarray(
        2.4 * _bump(ph, 7.1, 0.3) * glc / (glc + 2.5) * (0.65 + 0.35 * _bump(gln, 5.0, 2.2))
        * (0.75 + 0.25 * _bump(feed, 9.0, 5.0)) * (0.8 + 0.2 * _bump(osm, 305.0, 25.0)) * medium
    )


def _plate_design() -> pd.DataFrame:
    """176 recipes (88 per campaign), each campaign its own Latin hypercube."""
    rng = rng_v2(_RNG_DESIGN)
    frames = []
    for c, campaign in enumerate(("C1", "C2")):
        u = _lhs(rng, RECIPES_PER_CAMPAIGN, len(PLATE_FACTORS))
        rec = pd.DataFrame({
            name: np.round(lo + u[:, j] * (hi - lo), dec)
            for j, (name, (lo, hi, dec)) in enumerate(PLATE_FACTORS.items())
        })
        media = np.array(["basal_A", "basal_B"] * (RECIPES_PER_CAMPAIGN // 2))
        rec.insert(0, "medium_base", media[rng.permutation(RECIPES_PER_CAMPAIGN)])
        rec.insert(0, "recipe_id", [f"M{c * RECIPES_PER_CAMPAIGN + i + 1:03d}"
                                    for i in range(RECIPES_PER_CAMPAIGN)])
        rec.insert(0, "campaign", campaign)
        frames.append(rec)
    return pd.concat(frames, ignore_index=True)


def plate_screen() -> pd.DataFrame:
    design = _plate_design()
    layout_rng, noise_rng, notes_rng = rng_v2(_RNG_LAYOUT), rng_v2(_RNG_NOISE), rng_v2(_RNG_NOTES)
    free = [w for w in well_names() if w not in CONTROL_WELLS]
    order = {w: i for i, w in enumerate(well_names())}
    plate_offset = {plate: float(noise_rng.normal(1.0, 0.03)) for _c, plate in PLATES}
    rows = []
    for campaign, plate in PLATES:
        recipes = design[design["campaign"] == campaign]
        wells = [free[int(i)] for i in layout_rng.permutation(len(free))[: len(recipes)]]
        for well, (_, rec) in zip(wells, recipes.iterrows()):
            rows.append({"campaign": campaign, "plate": plate, "well": well, "sample_type": "sample",
                         **{str(k): v for k, v in rec.drop(labels=["campaign"]).items()}})
        for well in CONTROL_WELLS:
            rows.append({"campaign": campaign, "plate": plate, "well": well, "sample_type": "control",
                         "recipe_id": "REF", **CONTROL_RECIPE})
    df = pd.DataFrame(rows)
    df["_order"] = [order[w] for w in df["well"]]
    df = df.sort_values(["plate", "_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    mu = plate_mean_titer(df) * df["plate"].map(plate_offset).to_numpy(float)
    titer = mu * np.exp(noise_rng.normal(0.0, 0.08, size=len(df)))
    notes = np.where(notes_rng.random(len(df)) < 0.08,
                     np.asarray(BENIGN_NOTES, dtype=object)[notes_rng.integers(0, len(BENIGN_NOTES), len(df))],
                     "")
    df.insert(0, "run_id", [f"R{i + 1:04d}" for i in range(len(df))])
    df["notes"] = notes
    df[PLATE_TARGET] = np.round(titer, 3)
    cols = ["run_id", "campaign", "plate", "well", "sample_type", "recipe_id", "medium_base",
            *PLATE_FACTORS, "notes", PLATE_TARGET]
    return df[cols]


def with_glucose_mM(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pos = out.columns.get_loc("glucose_gL")
    assert isinstance(pos, int)
    out.insert(pos + 1, "glucose_mM", np.round(out["glucose_gL"].to_numpy(float) / GLUCOSE_MW * 1000.0, 2))
    return out


def clone_flasks() -> pd.DataFrame:
    """16 clones x 2 flask media x duplicate flasks; 5 clones lost expression."""
    rng = rng_v2(_RNG_CLONES)
    n_clones = 16
    clone_mu = np.exp(rng.normal(np.log(2.0), 0.35, size=n_clones))
    lost = np.sort(rng.choice(n_clones, size=5, replace=False))
    early = clone_mu * 0.6 * np.exp(rng.normal(0.0, 0.3, size=n_clones))
    rows = []
    for c in range(n_clones):
        for medium, effect in (("medium_A", 1.0), ("medium_B", 1.25)):
            for _rep in range(2):
                value: object
                if c in lost:
                    value = f"<{ASSAY_LOQ_G_L:g}"
                else:
                    value = round(float(clone_mu[c] * effect * np.exp(rng.normal(0.0, 0.10))), 3)
                rows.append({
                    "clone": f"CL{c + 1:02d}", "flask_medium": medium,
                    "seed_vcd_e6_per_mL": round(float(rng.normal(0.30, 0.015)), 3),
                    "titer_96dwp_g_L": round(float(early[c]), 3),
                    CLONE_TARGET: value,
                })
    df = pd.DataFrame(rows)
    df.insert(0, "run_id", [f"F{i + 1:03d}" for i in range(len(df))])
    return df


def scaleup_feed() -> pd.DataFrame:
    df = pd.read_csv(SCALEUP_CSV)
    feed = np.round(rng_v2(_RNG_FEED).uniform(10.0, 30.0, size=len(df)), 1)
    df.insert(df.columns.get_loc(SCALEUP_TARGET), "feed_pct", feed)  # type: ignore[arg-type]
    return df


def load_bases_v2() -> dict[str, Base]:
    plate = plate_screen()
    return {
        "plate368": Base("plate368", plate, PLATE_TARGET),
        "plate368_derived": Base("plate368_derived", with_glucose_mM(plate), PLATE_TARGET),
        "clone64": Base("clone64", clone_flasks(), CLONE_TARGET),
        "scaleup_feed90": Base("scaleup_feed90", scaleup_feed(), SCALEUP_TARGET),
    }


__all__ = [
    "ASSAY_LOQ_G_L", "CLONE_TARGET", "CONTROL_RECIPE", "CONTROL_WELLS", "GLUCOSE_MW", "PLATES",
    "PLATE_FACTORS", "PLATE_TARGET", "SCALEUP_TARGET", "clone_flasks", "is_edge_well",
    "load_bases_v2", "plate_mean_titer", "plate_screen", "rng_v2", "scaleup_feed", "well_names",
    "with_glucose_mM",
]
