"""Write the blinded catalogue-v2 suite, in the v1 layout (PROTOCOL_V2.md section 3).

    cases/case-NN/sheet.csv   what a user uploads (rows shuffled with a seed)
    cases/case-NN/case.json   surface, file name, target column
    manifest.json             sha256 of every case file (no truth)
    evaluator/truth.json      fault id, parent, planted columns and rows

The folder is read by the v1 blind runner (`runner.observe_case`) unchanged.
The pre-registered truth and manifest are committed next to this module
(`truth_v2.json`, `manifest_v2.json`); `build_suite_v2` must reproduce them byte
for byte (tests/test_planted_faults_v2.py).

    python -m kalos.bench.faults.cases_v2 OUT        build the suite under OUT
    python -m kalos.bench.faults.cases_v2 --freeze   rewrite the committed truth and
                                                     manifest (pre-registration only;
                                                     afterwards only by amendment)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bases_v2 import load_bases_v2
from .cases import CASES_DIR, SHEET, TRUTH_DIR, _dump
from .catalogue import CleanCase, Fault
from .catalogue_v2 import CLEAN_V2, FAULTS_V2, PROTOCOL_VERSION_V2, SUITE_SEED_V2
from .plant import Planted
from .plant_v2 import case_rng_v2, plant_clean_v2, plant_fault_v2

PKG = Path(__file__).parent
TRUTH_V2 = PKG / "truth_v2.json"
MANIFEST_V2 = PKG / "manifest_v2.json"


@dataclass(frozen=True)
class SuiteInfoV2:
    out: Path
    n_cases: int


def case_numbers_v2() -> dict[str, str]:
    """Catalogue v2 id -> `case-NN`, from a seeded permutation."""
    ids = [f.id for f in FAULTS_V2] + [c.id for c in CLEAN_V2]
    order = np.random.default_rng((SUITE_SEED_V2, 0)).permutation(len(ids))
    return {ids[int(j)]: f"case-{i + 1:02d}" for i, j in enumerate(order)}


def _write_case(folder: Path, planted: Planted, case_id: str) -> dict[str, Any]:
    folder.mkdir(parents=True)
    perm = case_rng_v2(case_id + "/rows").permutation(len(planted.df))
    df = planted.df.iloc[perm].reset_index(drop=True)
    where = {int(old): new for new, old in enumerate(perm)}
    (folder / SHEET).write_text(df.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    meta = {"surface": "sheet", "filename": SHEET, "target": planted.target}
    (folder / "case.json").write_text(_dump(meta), encoding="utf-8")
    return {"rows": sorted(where[r] for r in planted.rows), "n_sheet_rows": len(df)}


def build_suite_v2(out: Path) -> SuiteInfoV2:
    """(Re)build the v2 suite under `out`; existing `cases/` and `evaluator/` are replaced."""
    out = Path(out)
    for sub in (CASES_DIR, TRUTH_DIR):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / CASES_DIR).mkdir(parents=True)
    (out / TRUTH_DIR).mkdir(parents=True)
    bases = load_bases_v2()
    numbers = case_numbers_v2()
    entries: list[tuple[str, Fault | CleanCase]] = [(f.id, f) for f in FAULTS_V2] + [(c.id, c) for c in CLEAN_V2]
    truth: dict[str, dict[str, Any]] = {}
    manifest: dict[str, str] = {}
    for cid, item in sorted(entries, key=lambda e: numbers[e[0]]):
        case = numbers[cid]
        record: dict[str, Any] = {
            "case": case, "id": cid, "kind": "fault" if isinstance(item, Fault) else "clean",
            "base": item.base, "surface": item.surface, "tiers": list(item.tiers),
        }
        base = bases[item.base]
        if isinstance(item, Fault):
            record["parent"] = item.parent
            record["parent_case"] = numbers[item.parent]
            planted = plant_fault_v2(item, base)
        else:
            planted = plant_clean_v2(item, base)
        record.update(_write_case(out / CASES_DIR / case, planted, cid))
        record["columns"] = list(planted.columns)
        record["skipped"] = None
        truth[case] = record
        for path in sorted((out / CASES_DIR / case).iterdir()):
            manifest[f"{case}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    head = {"protocol_version": PROTOCOL_VERSION_V2, "suite_seed": SUITE_SEED_V2}
    (out / TRUTH_DIR / "truth.json").write_text(_dump({**head, "cases": truth}), encoding="utf-8")
    (out / "manifest.json").write_text(_dump({**head, "files": manifest}), encoding="utf-8")
    return SuiteInfoV2(out=out, n_cases=len(truth))


def freeze() -> None:
    """Write the committed truth and manifest from a fresh build."""
    with tempfile.TemporaryDirectory(prefix="kalos-faults-v2-") as tmp:
        build_suite_v2(Path(tmp))
        TRUTH_V2.write_bytes((Path(tmp) / TRUTH_DIR / "truth.json").read_bytes())
        MANIFEST_V2.write_bytes((Path(tmp) / "manifest.json").read_bytes())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kalos.bench.faults.cases_v2")
    ap.add_argument("out", nargs="?", type=Path)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)
    if args.freeze:
        freeze()
        print(f"wrote {TRUTH_V2} and {MANIFEST_V2}")
        return 0
    if args.out is None:
        ap.error("give OUT or --freeze")
    info = build_suite_v2(args.out)
    print(f"built {info.n_cases} v2 case(s) under {info.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MANIFEST_V2", "TRUTH_V2", "SuiteInfoV2", "build_suite_v2", "case_numbers_v2", "freeze"]
