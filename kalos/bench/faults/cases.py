"""Write the blinded suite: case folders without truth, and the truth key apart.

Layout under `out/`::

    cases/case-NN/sheet.csv      what a user uploads (rows shuffled with a seed)
    cases/case-NN/case.json      surface, file name, target column
    cases/case-NN/entries.json   result surface: one entry per started run
    cases/case-NN/request.json   scale surface: the readout request
    manifest.json                sha256 of every case file (no truth)
    evaluator/truth.json         fault id, parent, planted columns and rows

Case numbers come from a seeded permutation of every catalogue id, so they say
nothing about the fault and stay the same when an optional base is missing (its
case folder is simply not written, and truth marks it skipped). Two builds with
the same seed are byte-identical.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bases import load_bases
from .catalogue import CLEAN, FAULTS, PROTOCOL_VERSION, SUITE_SEED, CleanCase, Fault
from .plant import Planted, case_rng, plant_clean, plant_fault

SHEET = "sheet.csv"
TRUTH_DIR = "evaluator"
CASES_DIR = "cases"


@dataclass(frozen=True)
class SuiteInfo:
    out: Path
    n_cases: int
    n_skipped: int


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def case_numbers() -> dict[str, str]:
    """Catalogue id -> `case-NN`, from a seeded permutation."""
    ids = [f.id for f in FAULTS] + [c.id for c in CLEAN]
    order = np.random.default_rng((SUITE_SEED, 0)).permutation(len(ids))
    return {ids[int(j)]: f"case-{i + 1:02d}" for i, j in enumerate(order)}


def _transposed_csv(df: pd.DataFrame) -> str:
    """Runs as columns: the first column holds the parameter names. A sheet with
    a run-id column puts the ids in the header row; one without numbers the runs."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if "run_id" in df.columns:
        body = df.drop(columns=["run_id"])
        w.writerow(["run_id", *df["run_id"].astype(str)])
    else:
        body = df
        w.writerow(["parameter", *[f"run_{i + 1}" for i in range(len(df))]])
    for col in body.columns:
        w.writerow([col, *[repr(v) if isinstance(v, float) else str(v) for v in body[col]]])
    return buf.getvalue()


def _shuffle(planted: Planted, case_id: str) -> tuple[pd.DataFrame, list[int]]:
    perm = case_rng(case_id + "/rows").permutation(len(planted.df))
    shuffled = planted.df.iloc[perm].reset_index(drop=True)
    where = {int(old): new for new, old in enumerate(perm)}
    return shuffled, sorted(where[r] for r in planted.rows)


def _write_case(folder: Path, surface: str, planted: Planted, case_id: str) -> dict[str, Any]:
    """Write one case folder; return the truth fields that depend on the layout."""
    folder.mkdir(parents=True)
    df, rows = _shuffle(planted, case_id)
    if planted.transposed:
        text = _transposed_csv(df)
    else:
        text = df.to_csv(index=False, lineterminator="\n")
    (folder / SHEET).write_text(text, encoding="utf-8")
    meta: dict[str, Any] = {"surface": surface, "filename": SHEET, "target": planted.target}
    truth: dict[str, Any] = {"rows": rows, "n_sheet_rows": len(df)}
    if surface == "result":
        (folder / "entries.json").write_text(_dump(planted.entries), encoding="utf-8")
        meta["entries"] = "entries.json"
        # The fold appends each entry after the base rows, in entry order.
        truth["rows"] = [len(df) + k for k in planted.entry_faults]
        truth["watch_rows"] = [len(df) + k for k in planted.watch_entries]
    if surface == "scale":
        (folder / "request.json").write_text(_dump(planted.request), encoding="utf-8")
        meta["request"] = "request.json"
    (folder / "case.json").write_text(_dump(meta), encoding="utf-8")
    return truth


def build_suite(out: Path, *, datasets_dir: Path | None = None) -> SuiteInfo:
    """(Re)build the suite under `out`. Existing `cases/` and `evaluator/` are
    replaced; observations and scorecards are left alone."""
    out = Path(out)
    for sub in (CASES_DIR, TRUTH_DIR):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / CASES_DIR).mkdir(parents=True)
    (out / TRUTH_DIR).mkdir(parents=True)
    bases = load_bases(datasets_dir)
    numbers = case_numbers()
    entries: list[tuple[str, Fault | CleanCase]] = [(f.id, f) for f in FAULTS] + [(c.id, c) for c in CLEAN]
    truth: dict[str, dict[str, Any]] = {}
    manifest: dict[str, str] = {}
    skipped = 0
    for cid, item in sorted(entries, key=lambda e: numbers[e[0]]):
        case = numbers[cid]
        is_fault = isinstance(item, Fault)
        record: dict[str, Any] = {
            "case": case, "id": cid, "kind": "fault" if is_fault else "clean",
            "base": item.base, "surface": item.surface, "tiers": list(item.tiers),
        }
        if is_fault:
            assert isinstance(item, Fault)
            record["parent"] = item.parent
            record["parent_case"] = numbers[item.parent]
        base = bases.get(item.base)
        if base is None:
            record["skipped"] = f"base {item.base} unavailable"
            truth[case] = record
            skipped += 1
            continue
        planted = plant_fault(item, base) if isinstance(item, Fault) else plant_clean(item, base)
        layout = _write_case(out / CASES_DIR / case, item.surface, planted, cid)
        record.update(layout)
        record["columns"] = list(planted.columns)
        record["skipped"] = None
        truth[case] = record
        for path in sorted((out / CASES_DIR / case).iterdir()):
            manifest[f"{case}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / TRUTH_DIR / "truth.json").write_text(
        _dump({"protocol_version": PROTOCOL_VERSION, "suite_seed": SUITE_SEED, "cases": truth}),
        encoding="utf-8",
    )
    (out / "manifest.json").write_text(
        _dump({"protocol_version": PROTOCOL_VERSION, "suite_seed": SUITE_SEED, "files": manifest}),
        encoding="utf-8",
    )
    return SuiteInfo(out=out, n_cases=len(truth) - skipped, n_skipped=skipped)


def load_truth(out: Path) -> dict[str, Any]:
    return dict(json.loads((Path(out) / TRUTH_DIR / "truth.json").read_text(encoding="utf-8")))


__all__ = ["CASES_DIR", "SHEET", "TRUTH_DIR", "SuiteInfo", "build_suite", "case_numbers", "load_truth"]
