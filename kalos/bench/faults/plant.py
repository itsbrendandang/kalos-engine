"""Fault planters: clean base in, faulted frame plus the truth about it out.

Each planter takes a COPY of a clean base and changes only the cells its fault
names (tests/test_planted_faults.py checks that cell by cell). Planted rows are
positions in the returned frame, before the case builder shuffles row order.
Randomness comes from `case_rng(case_id)`, so a planter's draws depend only on
the suite seed and its own id.
"""
from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .bases import SCALE_PROCESS_COLUMNS, Base
from .catalogue import SUITE_SEED, CleanCase, Fault

ID_COLUMN = "run_id"


def case_rng(case_id: str) -> np.random.Generator:
    return np.random.default_rng((SUITE_SEED, 10_000 + zlib.crc32(case_id.encode()) % 1_000_000))


@dataclass
class Planted:
    """A faulted (or clean) input and its truth.

    `rows` are planted row positions in `df` (sheet surface) and `entries` the
    result entries (result surface); `entry_faults` / `watch_entries` index into
    `entries`. `request` is the scale readout request (scale surface).
    """

    df: pd.DataFrame
    target: str
    rows: list[int] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    entry_faults: list[int] = field(default_factory=list)
    watch_entries: list[int] = field(default_factory=list)
    request: dict[str, Any] | None = None
    transposed: bool = False


def _producers_above_median(y: pd.Series) -> np.ndarray:
    pos = y[y > 0]
    return np.flatnonzero((y > float(pos.median())).to_numpy())


def _pick(rng: np.random.Generator, pool: np.ndarray | range, n: int) -> list[int]:
    return sorted(int(i) for i in rng.choice(np.asarray(pool), size=n, replace=False))


def _object_column(df: pd.DataFrame, col: str) -> None:
    df[col] = df[col].astype(object)


# --- sheet faults ---------------------------------------------------------------- #


def _mixed_units_labeled(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    df, col = base.df.copy(), p["column"]
    rows = _pick(rng, range(len(df)), round(p["fraction"] * len(df)))
    vals = df[col].to_numpy(float)
    _object_column(df, col)
    for i in range(len(df)):
        if i in rows:
            df.at[i, col] = f"{vals[i] * p['factor']:.1f} {p['alt_unit']}"
        else:
            df.at[i, col] = f"{vals[i]:.3f} {p['unit']}"
    return Planted(df, base.target, rows=rows, columns=[col])


def _mixed_units_unlabeled(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    start = len(df) - len(df) // 4
    y = df[col].to_numpy(float)
    rows = [i for i in range(start, len(df)) if y[i] > 0]
    for i in rows:
        df.at[i, col] = round(y[i] * f.params["factor"], 3)
    return Planted(df, base.target, rows=rows, columns=[col])


def _fahrenheit(labeled: bool) -> Callable[[Base, Fault, np.random.Generator], Planted]:
    def plant(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
        df, col = base.df.copy(), f.params["column"]
        rows = _pick(rng, range(len(df)), round(f.params["fraction"] * len(df)))
        c = df[col].to_numpy(float)
        if labeled:
            _object_column(df, col)
        for i in rows:
            fahrenheit = round(c[i] * 9.0 / 5.0 + 32.0, 2)
            df.at[i, col] = f"{fahrenheit:.2f} F" if labeled else fahrenheit
        return Planted(df, base.target, rows=rows, columns=[col])

    return plant


def _transposed(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    return Planted(base.df.copy(), base.target, transposed=True)


def _typo(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    pool = _producers_above_median(df[col]) if col == base.target else np.arange(len(df))
    rows = _pick(rng, pool, f.params["n_rows"])
    vals = df[col].to_numpy(float)
    for i in rows:
        df.at[i, col] = round(float(vals[i]) * f.params["factor"], 3)
    return Planted(df, base.target, rows=rows, columns=[col])


def _swapped_columns(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df = base.df.copy()
    a, b = f.params["columns"]
    rows = _pick(rng, range(len(df)), f.params["n_rows"])
    va, vb = df[a].to_numpy(float), df[b].to_numpy(float)
    for i in rows:
        df.at[i, a], df.at[i, b] = vb[i], va[i]
    return Planted(df, base.target, rows=rows, columns=[a, b])


def _swapped_levels(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    n = f.params["block_rows"]
    start = int(rng.integers(0, len(df) - n + 1))
    rotation = f.params["rotation"]
    rows = list(range(start, start + n))
    labels = df[col].astype(str).tolist()
    for i in rows:
        df.at[i, col] = rotation[labels[i]]
    return Planted(df, base.target, rows=rows, columns=[col])


def _duplicated(new_id: bool) -> Callable[[Base, Fault, np.random.Generator], Planted]:
    def plant(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
        df = base.df.copy()
        src = _pick(rng, range(len(df)), f.params["n_rows"])
        copies = df.iloc[src].copy()
        if new_id:
            copies[ID_COLUMN] = [f"R{len(df) + 1 + k:03d}" for k in range(len(src))]
        out = pd.concat([df, copies], ignore_index=True)
        rows = src + list(range(len(df), len(out)))
        return Planted(out, base.target, rows=rows, columns=[])

    return plant


def _duplicate_run_id(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df = base.df.copy()
    a, b = _pick(rng, range(len(df)), 2)
    df.at[b, ID_COLUMN] = str(df[ID_COLUMN].iloc[a])
    return Planted(df, base.target, rows=[a, b], columns=[ID_COLUMN])


def _too_few_rows(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    rows = _pick(rng, range(len(base.df)), f.params["n_rows"])
    return Planted(base.df.iloc[rows].reset_index(drop=True), base.target)


def _constant_target(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df = base.df.copy()
    df[f.params["column"]] = f.params["value"]
    return Planted(df, base.target, columns=[f.params["column"]])


def _impossible_value(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    (row,) = _pick(rng, range(len(df)), 1)
    df.at[row, col] = f.params["value"]
    return Planted(df, base.target, rows=[row], columns=[col])


# --- result faults ---------------------------------------------------------------- #


def _result_entries(base: Base) -> list[dict[str, Any]]:
    assert base.holdout is not None
    process = [c for c in base.df.columns if c not in (ID_COLUMN, base.target)]
    entries = []
    for _, row in base.holdout.iterrows():
        recipe = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in process}
        entries.append({"recipe": recipe, "value": float(row[base.target])})
    return entries


def _result_fault(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    entries = _result_entries(base)
    k = int(f.params["entry"])
    v = entries[k]["value"]
    if f.id == "result_negative":
        entries[k]["value"] = -v
    elif f.id == "result_copy_paste":
        entries[k]["value"] = entries[int(f.params["copy_of"])]["value"]
    else:
        entries[k]["value"] = round(v * float(f.params["factor"]), 6)
    return Planted(base.df.copy(), base.target, entries=entries, entry_faults=[k], columns=[base.target])


# --- model faults ---------------------------------------------------------------- #


def _pure_noise(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    perm = np.random.default_rng((SUITE_SEED, 500 + int(f.params["shuffle_seed"]))).permutation(len(df))
    df[col] = df[col].to_numpy()[perm]
    return Planted(df, base.target, columns=[col])


def _kaggle_extra(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    df.insert(len(df.columns) - 1, col, base.extras[col].to_numpy())
    return Planted(df, base.target, columns=[col])


def _leak_renamed(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    y = df[base.target]
    q1, q2 = float(y.quantile(1 / 3)), float(y.quantile(2 / 3))
    grade = np.where(y > q2, "A", np.where(y > q1, "B", "C"))
    df.insert(len(df.columns) - 1, col, grade)
    return Planted(df, base.target, columns=[col])


def _clone(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    levels, offsets = f.params["levels"], f.params["offsets"]
    which = rng.integers(0, len(levels), size=len(df))
    y = df[base.target].to_numpy(float)
    df[base.target] = np.round(np.where(y > 0, y + np.asarray(offsets)[which], y), 3)
    df.insert(1, col, np.asarray(levels)[which])
    return Planted(df, base.target, columns=[col])


def _passage(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    df, col = base.df.copy(), f.params["column"]
    p = rng.integers(f.params["low"], f.params["high"] + 1, size=len(df))
    y = df[base.target].to_numpy(float)
    df[base.target] = np.round(y * (1.0 + f.params["slope"] * (p - f.params["low"])), 3)
    df.insert(1, col, p)
    return Planted(df, base.target, columns=[col])


def _batch_offset(with_column: bool) -> Callable[[Base, Fault, np.random.Generator], Planted]:
    def plant(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
        p = f.params
        n = p["n_batches"] * p["per_batch"]
        df = base.df.sort_values("pH", kind="mergesort").head(n).reset_index(drop=True)
        batch = np.repeat(np.arange(p["n_batches"]), p["per_batch"])
        offsets = 0.6 + rng.normal(0.0, p["offset_sd"], size=p["n_batches"])
        y = offsets[batch] + rng.normal(0.0, p["noise_sd"], size=n)
        df[base.target] = np.round(np.clip(y, 0.0, None), 4)
        cols: list[str] = []
        if with_column:
            df.insert(1, "batch", [f"B{b + 1}" for b in batch])
            cols = ["batch"]
        return Planted(df, base.target, columns=cols)

    return plant


def _jittered_replicates(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    src = base.df.iloc[_pick(rng, range(len(base.df)), p["n_recipes"])].reset_index(drop=True)
    numeric = [c for c in src.columns
               if c not in (ID_COLUMN, base.target) and pd.api.types.is_numeric_dtype(src[c])]
    span = {c: float(base.df[c].max() - base.df[c].min()) for c in numeric}
    effects = np.clip(0.6 + rng.normal(0.0, p["effect_sd"], size=len(src)), 0.0, None)
    rows = []
    for r in range(len(src)):
        for k in range(p["replicates"]):
            row = src.iloc[r].to_dict()
            for c in numeric:
                row[c] = round(float(row[c]) + rng.uniform(-1, 1) * p["jitter_frac"] * span[c], 4)
            row[ID_COLUMN] = f"J{r + 1:02d}-{k + 1}"
            row[base.target] = round(max(0.0, effects[r] + rng.normal(0.0, p["noise_sd"])), 4)
            rows.append(row)
    return Planted(pd.DataFrame(rows, columns=list(base.df.columns)), base.target)


# --- scale faults ----------------------------------------------------------------- #


def scale_request(base: Base, target_scale: float, rpm: float, airflow: float) -> dict[str, Any]:
    params = {c: round(float(base.df[c].median()), 4) for c in SCALE_PROCESS_COLUMNS}
    return {
        "target_column": base.target,
        "process_columns": list(SCALE_PROCESS_COLUMNS),
        "target": {
            "scale_L": float(target_scale), "process_params": params,
            "agitation_rpm": float(rpm), "airflow_L_per_min": float(airflow),
        },
    }


def _scale_fault(base: Base, f: Fault, rng: np.random.Generator) -> Planted:
    p = f.params
    req = scale_request(base, p["target_scale_L"], p["agitation_rpm"], p["airflow_L_per_min"])
    return Planted(base.df.copy(), base.target, request=req, columns=["scale_L"])


PLANTERS: dict[str, Callable[[Base, Fault, np.random.Generator], Planted]] = {
    "mixed_units_labeled": _mixed_units_labeled,
    "mixed_units_unlabeled": _mixed_units_unlabeled,
    "fahrenheit_rows_bare": _fahrenheit(labeled=False),
    "fahrenheit_rows_labeled": _fahrenheit(labeled=True),
    "transposed_sheet": _transposed,
    "transposed_sheet_no_id": _transposed,
    "typo_x10_target": _typo,
    "typo_x1000_target": _typo,
    "typo_x10_feature": _typo,
    "typo_x1000_feature": _typo,
    "swapped_columns": _swapped_columns,
    "swapped_levels": _swapped_levels,
    "duplicated_rows_exact": _duplicated(new_id=False),
    "duplicated_rows_new_id": _duplicated(new_id=True),
    "duplicate_run_id": _duplicate_run_id,
    "too_few_rows_5": _too_few_rows,
    "too_few_rows_8": _too_few_rows,
    "constant_target": _constant_target,
    "impossible_value": _impossible_value,
    "result_x1000": _result_fault,
    "result_x10": _result_fault,
    "result_negative": _result_fault,
    "result_copy_paste": _result_fault,
    "pure_noise_target_s1": _pure_noise,
    "pure_noise_target_s2": _pure_noise,
    "pure_noise_target_s3": _pure_noise,
    "leaked_column_named": _kaggle_extra,
    "leaked_column_renamed": _leak_renamed,
    "id_like_numeric_categorical": _clone,
    "covariate_as_setpoint": _passage,
    "batch_offset_with_column": _batch_offset(with_column=True),
    "batch_offset_no_column": _batch_offset(with_column=False),
    "jittered_replicates": _jittered_replicates,
    "outcome_as_factor": _kaggle_extra,
    "scale_physics_extrapolation": _scale_fault,
    "scale_far_extrapolation": _scale_fault,
}


def plant_fault(fault: Fault, base: Base) -> Planted:
    return PLANTERS[fault.id](base, fault, case_rng(fault.id))


def plant_clean(clean: CleanCase, base: Base) -> Planted:
    """The clean input for a clean case (a copy; scale and result cases get their
    request / entries built the same way the faulted ones are)."""
    if clean.surface == "result":
        entries = _result_entries(base)
        watch = list(range(len(entries)))
        if "factor_of_best" in clean.params:
            k = int(clean.params["entry"])
            best = float(base.df[base.target].max())
            entries[k]["value"] = round(best * float(clean.params["factor_of_best"]), 6)
            watch = [k]
        return Planted(base.df.copy(), base.target, entries=entries, watch_entries=watch)
    if clean.surface == "scale":
        top = base.df[base.df["scale_L"] == float(clean.params["target_scale_L"])]
        req = scale_request(
            base, clean.params["target_scale_L"],
            round(float(top["agitation_rpm"].median()), 2),
            round(float(top["airflow_L_per_min"].median()), 3),
        )
        return Planted(base.df.copy(), base.target, request=req)
    return Planted(base.df.copy(), base.target)


__all__ = ["PLANTERS", "Planted", "case_rng", "plant_clean", "plant_fault", "scale_request"]
