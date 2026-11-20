"""Catalogue v2 planters: a clean v2 base in, a faulted frame plus its truth out.

Each planter works on a COPY of its base and changes only what its fault names
(tests/test_planted_faults_v2.py checks this per planter). Planted rows are
positions in the returned frame, before the case builder shuffles row order.
Randomness comes from `case_rng_v2(case_id)`, so a planter's draws depend only
on the v2 suite seed and its own id.
"""
from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from .bases import Base
from .bases_v2 import is_edge_well, plate_mean_titer
from .catalogue import CleanCase, Fault
from .catalogue_v2 import SUITE_SEED_V2
from .plant import Planted

ID_COLUMN = "run_id"
CONTAMINATION_NOTES = (
    "turbid on day 5, suspected bacterial contamination",
    "contaminated - cloudy with a sour smell",
    "fungal growth on the well wall",
    "contamination seen at harvest, kept for now",
)


def case_rng_v2(case_id: str) -> np.random.Generator:
    return np.random.default_rng((SUITE_SEED_V2, 10_000 + zlib.crc32(case_id.encode()) % 1_000_000))


def _pick(rng: np.random.Generator, pool: np.ndarray | list[int] | range, n: int) -> list[int]:
    return sorted(int(i) for i in rng.choice(np.asarray(pool), size=n, replace=False))


def _rows_where(mask: pd.Series | np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(mask))]


def _samples(df: pd.DataFrame) -> np.ndarray:
    return np.flatnonzero((df["sample_type"] == "sample").to_numpy())


# --- censored and filled results ------------------------------------------------------ #


def _nondetects_as_zero(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    rows = _rows_where(df[col].astype(str) == p["censored_token"])
    df[col] = pd.to_numeric(df[col].where(df[col].astype(str) != p["censored_token"], p["replacement"]))
    return Planted(df, base.target, rows=rows, columns=[col])


def _placeholder_fill(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    rows = _pick(rng, _samples(df), p["n_rows"])
    for i in rows:
        df.at[i, col] = float(p["value"])
    return Planted(df, base.target, rows=rows, columns=[col])


def _mean_imputed(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    rows = _pick(rng, _samples(df), p["n_rows"])
    kept = df[col].drop(index=rows)
    fill = round(float(kept.mean()), int(p["decimals"]))
    for i in rows:
        df.at[i, col] = fill
    return Planted(df, base.target, rows=rows, columns=[col])


# --- units, regimes, missing context ---------------------------------------------------- #


def _volumes(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df = base.df.copy()
    pct = df[p["source_column"]].to_numpy(float)
    vessel_mL = df[p["scale_column"]].to_numpy(float) * 1000.0
    pos = df.columns.get_loc(p["source_column"])
    assert isinstance(pos, int)
    df = df.drop(columns=[p["source_column"]])
    df.insert(pos, p["column"], np.round(pct / 100.0 * vessel_mL, 1))
    return Planted(df, base.target, rows=[], columns=[p["column"]])


def _stacked_regimes(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    target = base.target
    pos = df.columns.get_loc(target)
    assert isinstance(pos, int)
    df.insert(pos, col, float(p["plate_volume_mL"]))
    recipes = df[df["sample_type"] == "sample"].drop_duplicates("recipe_id")
    chosen = recipes.iloc[_pick(rng, range(len(recipes)), p["n_recipes"])]
    reps = int(p["n_runs"]) // int(p["n_recipes"])
    extra = pd.concat([chosen] * reps, ignore_index=True)
    mu = plate_mean_titer(extra) * float(p["regime_factor"])
    extra[target] = np.round(mu * np.exp(rng.normal(0.0, p["noise_sd"], size=len(extra))), 3)
    extra["plate"] = ""
    extra["well"] = ""
    extra["notes"] = ""
    extra["campaign"] = "C2"
    extra[col] = float(p["reactor_volume_mL"])
    extra[ID_COLUMN] = [f"R{len(df) + k + 1:04d}" for k in range(len(extra))]
    out = pd.concat([df, extra[df.columns]], ignore_index=True)
    rows = list(range(len(df), len(out)))
    return Planted(out, target, rows=rows, columns=[col, target])


def _harvest(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df = base.df.copy()
    mask = df["plate"].isin(p["plates"])
    df.loc[mask, base.target] = np.round(df.loc[mask, base.target].to_numpy(float) * p["factor"], 3)
    return Planted(df, base.target, rows=_rows_where(mask), columns=["plate"])


# --- plates, controls, identifiers ------------------------------------------------------ #


def _failed_controls(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, t = base.df.copy(), base.target
    on_plate = (df["plate"] == p["plate"]).to_numpy()
    control = on_plate & (df["sample_type"] == "control").to_numpy()
    sample = on_plate & ~control
    y = df[t].to_numpy(float)
    y = np.where(control, y * p["control_factor"], y)
    y = np.where(sample, y * np.exp(rng.normal(0.0, p["sample_noise_sd"], size=len(y))), y)
    df[t] = np.round(y, 3)
    return Planted(df, t, rows=_rows_where(on_plate), columns=["plate"])


def _sample_id(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    """recipe_id becomes sample numbers S001..S176 (replicates share one); on one
    plate the numbering of a few wells is shifted, so those ids now also cover a
    different recipe."""
    p = f.params
    df = base.df.copy()
    pos = df.columns.get_loc(p["source_column"])
    assert isinstance(pos, int)
    key = df[p["source_column"]].astype(str)
    ids = sorted(key[key != "REF"].unique())
    number = {rid: f"S{i + 1:03d}" for i, rid in enumerate(ids)}
    sample_id = key.map(lambda k: number.get(k, "REF")).to_numpy(object)
    on_plate = np.flatnonzero(((df["plate"] == p["plate"]) & (df["sample_type"] == "sample")).to_numpy())
    rows = _pick(rng, on_plate, int(p["n_conflicts"]))
    pool = sorted({sample_id[i] for i in on_plate})
    for i in rows:
        current = sample_id[i]
        sample_id[i] = pool[(pool.index(current) + 1) % len(pool)]
    df = df.drop(columns=[p["source_column"]])
    df.insert(pos, p["column"], sample_id)
    return Planted(df, base.target, rows=rows, columns=[p["column"]])


def _well_mismatch(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    on_plate = np.flatnonzero(((df["plate"] == p["plate"]) & (df["sample_type"] == "sample")).to_numpy())
    n_dup, impossible = int(p["n_duplicates"]), list(p["impossible"])
    rows = _pick(rng, on_plate, n_dup + len(impossible))
    others = [int(i) for i in on_plate if int(i) not in rows]
    targets = _pick(rng, others, n_dup)
    for i, j in zip(rows[:n_dup], targets):
        w = str(df.at[j, col])
        df.at[i, col] = f"{w[0]}{int(w[1:]):02d}"  # the same well, written zero-padded
    for i, w in zip(rows[n_dup:], impossible):
        df.at[i, col] = w
    return Planted(df, base.target, rows=rows, columns=[col])


def _derived_inconsistent(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    rows = _pick(rng, range(len(df)), int(p["n_rows"]))
    vals = df[col].to_numpy(float)
    for i in rows:
        df.at[i, col] = round(float(vals[i]) * float(p["stale_factor"]), 2)
    return Planted(df, base.target, rows=rows, columns=[col, p["source_column"]])


def _contamination(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    pool = [int(i) for i in _samples(df) if not str(df.at[int(i), col])]
    rows = _pick(rng, pool, int(p["n_rows"]))
    y = df[base.target].to_numpy(float)
    for k, i in enumerate(rows):
        df.at[i, col] = CONTAMINATION_NOTES[k % len(CONTAMINATION_NOTES)]
        df.at[i, base.target] = round(float(y[i]) * float(p["titer_factor"]), 3)
    return Planted(df, base.target, rows=rows, columns=[col, base.target])


# --- leaks --------------------------------------------------------------------------- #


def _linear_combo(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, t = base.df.copy(), base.target
    a, b = p["columns"]
    y = df[t].to_numpy(float)
    share = rng.uniform(p["low"], p["high"], size=len(df))
    sup = np.round(y * share, 3)
    cell = np.round(y - sup, 3)
    pos = df.columns.get_loc(t)
    assert isinstance(pos, int)
    df.insert(pos, b, cell)
    df.insert(pos, a, sup)
    df[t] = np.round(sup + cell, 3)
    return Planted(df, t, rows=[], columns=[a, b, t])


def _label_column(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, t = base.df.copy(), base.target
    censored = df[t].astype(str).str.startswith("<").to_numpy()
    status = np.where(censored, "terminated", "expanded").astype(object)
    producers = np.flatnonzero(~censored)
    lowest = producers[np.argsort(pd.to_numeric(df[t].iloc[producers]).to_numpy(float), kind="mergesort")]
    for i in lowest[: int(p["n_flipped"])]:
        status[i] = "terminated"  # low producers culled on the same readout
    pos = df.columns.get_loc(t)
    assert isinstance(pos, int)
    df.insert(pos, p["column"], status)
    return Planted(df, t, rows=[], columns=[p["column"]])


def _id_token(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    """Run ids renumbered in the order of a noisy pre-screen ranking (best first),
    and the number parsed back out of the id offered as `tube_no`."""
    p = f.params
    df, t = base.df.copy(), base.target
    noisy = df[t].to_numpy(float) * np.exp(rng.normal(0.0, p["rank_noise_sd"], size=len(df)))
    rank = np.empty(len(df), dtype=int)
    rank[np.argsort(-noisy, kind="mergesort")] = np.arange(1, len(df) + 1)
    df[p["id_column"]] = [f"R{r:04d}" for r in rank]
    pos = df.columns.get_loc(t)
    assert isinstance(pos, int)
    df.insert(pos, p["column"], rank)
    return Planted(df, t, rows=[], columns=[p["column"], p["id_column"]])


def _pair_id(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    key = df[col].astype(str)
    first = sorted(key[(df["campaign"] != p["campaign"]) & (key != "REF")].unique())
    second = sorted(key[(df["campaign"] == p["campaign"]) & (key != "REF")].unique())
    remap = {old: first[i] for i, old in enumerate(second)}
    mask = (df["campaign"] == p["campaign"]) & (key != "REF")
    df.loc[mask, col] = key[mask].map(remap)
    reused = set(remap.values())
    rows = _rows_where((df["sample_type"] == "sample") & df[col].isin(reused))
    return Planted(df, base.target, rows=rows, columns=[col])


# --- design structure and time order ---------------------------------------------------- #


def _flask_media(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df = base.df.copy()
    clones = sorted(df[p["clone_column"]].unique())
    half = set(clones[: len(clones) // 2])
    first = df[p["clone_column"]].isin(half)
    keep = (first & (df[p["medium_column"]] == p["first_half_medium"])) | (
        ~first & (df[p["medium_column"]] == p["second_half_medium"]))
    out = df[keep.to_numpy()].reset_index(drop=True)
    return Planted(out, base.target, rows=[], columns=[p["clone_column"], p["medium_column"]])


def _after_outcome(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, t = base.df.copy(), base.target
    supplied = df["glucose_gL"].to_numpy(float) + p["feed_glucose_frac"] * df["feed_pct"].to_numpy(float)
    residual = supplied - p["uptake_per_titer"] * df[t].to_numpy(float) + rng.normal(0.0, p["noise_sd"], len(df))
    pos = df.columns.get_loc("notes")
    assert isinstance(pos, int)
    df.insert(pos, p["column"], np.round(np.clip(residual, 0.1, None), 2))
    return Planted(df, t, rows=[], columns=[p["column"]])


def _edge(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, t = base.df.copy(), base.target
    edge = df["well"].astype(str).map(is_edge_well).to_numpy(bool)
    df.loc[edge, t] = np.round(df.loc[edge, t].to_numpy(float) * f.params["factor"], 3)
    return Planted(df, t, rows=_rows_where(edge), columns=["well", t])


PLANTERS_V2: dict[str, Callable[[Base, Fault, np.random.Generator], Planted]] = {
    "nondetects_as_zero": _nondetects_as_zero,
    "placeholder_fill_constant": _placeholder_fill,
    "mean_imputed_target": _mean_imputed,
    "volumes_not_concentrations": _volumes,
    "stacked_regimes": _stacked_regimes,
    "unrecorded_harvest_time": _harvest,
    "failed_control_plate_kept": _failed_controls,
    "sample_id_not_recipe": _sample_id,
    "well_name_mismatch": _well_mismatch,
    "derived_column_inconsistent": _derived_inconsistent,
    "contamination_in_notes": _contamination,
    "derived_target_linear_combo": _linear_combo,
    "label_encoded_by_column": _label_column,
    "id_token_feature": _id_token,
    "pair_id_across_campaigns": _pair_id,
    "flask_media_mixed": _flask_media,
    "feature_after_outcome": _after_outcome,
    "edge_well_effect": _edge,
}


def plant_fault_v2(fault: Fault, base: Base) -> Planted:
    return PLANTERS_V2[fault.id](base, fault, case_rng_v2(fault.id))


def plant_clean_v2(clean: CleanCase, base: Base) -> Planted:
    return Planted(base.df.copy(), base.target)


__all__ = ["CONTAMINATION_NOTES", "PLANTERS_V2", "case_rng_v2", "plant_clean_v2", "plant_fault_v2"]
