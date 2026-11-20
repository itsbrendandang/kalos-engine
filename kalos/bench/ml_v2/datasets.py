"""Real and fixture datasets for pool replay, emulators and real snapshots
(PROTOCOL.md section 3.2; licences and provenance in benchmarks/ml_v2/DATASETS.md).

No data file is stored in the repository. Each loader reads from a folder:

- Amii media datasets: `KALOS_AMII_DATA` (the six CSVs of
  github.com/Amii-Applied-AI/amii-cell-ag-tools, active-learning-for-cell-media/Data).
- Downloaded open datasets (Zenodo, Olympus): `KALOS_BENCH_DATA`
  (default ~/kalos-bench-data); `python -m kalos.bench.ml_v2 fetch` downloads them
  and checks their SHA-256.
- The Kaggle fermentation sheet (information only): `KALOS_DATASET_DIR` or
  ~/kalos/datasets.
- Repo fixtures: examples/synthetic_bioprocess and examples/synthetic_scaleup.
- A user folder (`--user-data DIR`): every `*.csv` with an optional
  `<stem>.json` sidecar (see `load_user_folder`).

A missing dataset is reported as skipped, never as a result.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .space import Space

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RealDataset:
    name: str
    group: str  # real | ood | fixture | info_only | user
    frame: pd.DataFrame  # inputs + target, one row per (aggregated) recipe
    target: str
    continuous: list[str]
    categorical: list[str] = field(default_factory=list)
    goal: str = "maximize"
    source: str = ""
    license: str = ""
    measured: str = "measured"
    notes: str = ""

    @property
    def y(self) -> np.ndarray:
        return self.frame[self.target].to_numpy(float)

    def space(self) -> Space:
        cont = {c: (float(self.frame[c].min()), float(self.frame[c].max())) for c in self.continuous}
        cats = {c: tuple(sorted(self.frame[c].astype(str).unique())) for c in self.categorical}
        return Space(continuous=cont, categorical=cats)

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "group": self.group, "n_rows": len(self.frame), "target": self.target,
                "n_continuous": len(self.continuous), "n_categorical": len(self.categorical),
                "goal": self.goal, "source": self.source, "license": self.license,
                "measured": self.measured, "notes": self.notes}


def bench_data_dir() -> Path:
    return Path(os.environ.get("KALOS_BENCH_DATA", "~/kalos-bench-data")).expanduser()


def _finish(df: pd.DataFrame, target: str, cont: list[str], cats: list[str]) -> pd.DataFrame:
    """Finite rows only, constant inputs dropped, replicate recipes averaged."""
    df = df[[*cont, *cats, target]].copy()
    for c in [*cont, target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cats:
        df[c] = df[c].astype(str)
    df = df.dropna()
    keys = [c for c in [*cont, *cats] if df[c].nunique() > 1]
    return pd.DataFrame(df.groupby(keys, sort=False, as_index=False)[target].mean())


# -------------------------------------------------------------------- Amii
def _amii(name: str) -> RealDataset:
    from kalos.bench.amii import REGISTRY, load_amii_pool

    x, y, info = load_amii_pool(name)
    feats = list(info["features"])
    df = pd.DataFrame(x, columns=feats)
    df["y"] = y
    spec = REGISTRY[name]
    return RealDataset(f"amii/{name}", "real", df, "y", feats, source=f"Amii active-learning-for-cell-media, {spec.filename}",
                       license="open-access article data, reuse with attribution (datasets keep their papers' terms)",
                       notes=spec.objective)


# ---------------------------------------------------------------- refolding
REFOLD_INPUTS = ["pH", "urea_M", "dilution", "GSSG_mM", "DTT_mM"]


def _refold() -> RealDataset:
    root = bench_data_dir() / "zenodo" / "refold" / "Results_Tabular"
    doe = pd.read_excel(root / "DoE_Data.xlsx", sheet_name="DoE1")
    doe = doe[doe["Incl/Excl"].astype(str).str.strip() == "Incl"]
    a = pd.DataFrame({"pH": doe["pH"], "urea_M": doe["final Urea (M)"], "dilution": doe["Dilution Factor (-)"],
                      "GSSG_mM": doe["GSSG (mM)"], "DTT_mM": doe["DTT in solubilized protein (mM)"],
                      "yield_pct": doe["Refolding Yield (%)"]})
    bo = pd.read_excel(root / "BO_Data.xlsx")
    b = pd.DataFrame({"pH": bo["pH"], "urea_M": bo["Final Urea [M]"], "dilution": bo["Dilution Factor"],
                      "GSSG_mM": bo["GSSG [mM]"], "DTT_mM": bo["DTT [mM]"],
                      "yield_pct": bo["Refolding yield (HPLC based) [%]"]})
    df = _finish(pd.concat([a, b], ignore_index=True), "yield_pct", REFOLD_INPUTS, [])
    return RealDataset("zenodo/refold_scfv", "real", df, "yield_pct", [c for c in REFOLD_INPUTS if c in df],
                       source="Zenodo 10.5281/zenodo.18270843 (inclusion-body refolding, DoE1 + BO rounds)",
                       license="CC-BY-4.0", notes="downstream refolding yield; 2 campaigns pooled")


# ------------------------------------------------------------------ Olympus
OLYMPUS_COMMIT = "440b6b58ebfcaa2391cff7e94b570fb4fda98d68"


def _olympus_hplc() -> RealDataset:
    cols = ["sample_loop", "additional_volume", "tubing_volume", "sample_flow", "push_speed", "wait_time"]
    df = pd.read_csv(bench_data_dir() / "olympus" / "hplc.csv", header=None, names=[*cols, "peak_area"])
    df = _finish(df, "peak_area", cols, [])
    return RealDataset("olympus/hplc", "ood", df, "peak_area", cols,
                       source=f"Olympus (aspuru-guzik-group/olympus@{OLYMPUS_COMMIT[:7]}) dataset_hplc",
                       license="MIT (Olympus repository)", notes="automated HPLC process parameters; not bioprocess")


def _olympus_lnp3() -> RealDataset:
    cols = ["drug_input", "solid_lipid", "solid_lipid_input", "liquid_lipid_input", "surfactant_input",
            "drug_loading", "encap_efficiency", "particle_diameter"]
    df = pd.read_csv(bench_data_dir() / "olympus" / "lnp3.csv", header=None, names=cols)
    cont = ["drug_input", "solid_lipid_input", "liquid_lipid_input", "surfactant_input"]
    df = _finish(df, "drug_loading", cont, ["solid_lipid"])
    return RealDataset("olympus/lnp3", "ood", df, "drug_loading", cont, ["solid_lipid"],
                       source=f"Olympus (aspuru-guzik-group/olympus@{OLYMPUS_COMMIT[:7]}) dataset_lnp3",
                       license="MIT (Olympus repository)",
                       notes="lipid-nanoparticle formulation grid; discrete levels treated as numeric")


# ------------------------------------------------------------------ fixtures
def _fixture_bioprocess() -> RealDataset:
    df = pd.read_csv(REPO_ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv")
    cont = list(BIO_INPUTS)
    df = _finish(df, "titer_g_L", cont, ["medium_base"])
    return RealDataset("fixture/synthetic_bioprocess", "fixture", df, "titer_g_L", cont, ["medium_base"],
                       source="examples/synthetic_bioprocess (repo)", license="repo (Apache-2.0)",
                       measured="synthetic (generated)", notes="same generator as the bioprocess family")


BIO_INPUTS = ("pH", "temperature_C", "glucose_gL", "glutamine_mM", "feed_rate_mLh", "DO_pct", "agitation_rpm",
              "inoculum_1e6", "antifoam_mgL", "osmolality_mOsm")


def _fixture_scaleup() -> RealDataset:
    df = pd.read_csv(REPO_ROOT / "examples" / "synthetic_scaleup" / "synthetic_scaleup.csv")
    cont = ["scale_L", "agitation_rpm", "airflow_L_per_min", "ph_setpoint", "temperature_C"]
    df = _finish(df, "titer_g_per_L", cont, [])
    return RealDataset("fixture/synthetic_scaleup", "fixture", df, "titer_g_per_L", cont,
                       source="examples/synthetic_scaleup (repo)", license="repo (Apache-2.0)",
                       measured="synthetic (generated)", notes="scale is an input here (a pool over scales)")


def _kaggle() -> RealDataset:
    root = Path(os.environ.get("KALOS_DATASET_DIR", "~/kalos/datasets")).expanduser()
    df = pd.read_csv(root / "kaggle_fermentation_dataset_0.csv")
    df = df.drop_duplicates("run_id")
    cont = ["temp_c", "ph", "inoculum_vol_ml", "rpm", "lactose_feed_ml", "duration_days"]
    df = _finish(df, "max_yield_fpu_ml", cont, [])
    return RealDataset("kaggle/fermentation", "info_only", df, "max_yield_fpu_ml", cont,
                       source="Kaggle 'Fermentation optimization' (dataset_0.csv)",
                       license="unclear (uploader lists Apache-2.0; rights to the runs unverified)",
                       measured="mostly machine-generated (696 of 715 rows flagged Synthetic)",
                       notes="INFORMATION ONLY: never pooled, never decides a criterion; leaked columns dropped")


AMII_NAMES = ("DBO_rat_myocyte", "MOBO_rat_myocyte", "HeLa_regular", "HeLa_timesaving",
              "Tcell_expansion", "TF1_expansion")
LOADERS: dict[str, Callable[[], RealDataset]] = {
    **{f"amii/{n}": functools.partial(_amii, n) for n in AMII_NAMES},
    "zenodo/refold_scfv": _refold,
    "olympus/hplc": _olympus_hplc,
    "olympus/lnp3": _olympus_lnp3,
    "fixture/synthetic_bioprocess": _fixture_bioprocess,
    "fixture/synthetic_scaleup": _fixture_scaleup,
    "kaggle/fermentation": _kaggle,
}
# Which suites each dataset enters (PROTOCOL.md section 3.2).
POOL_DATASETS = tuple(LOADERS)
EMULATOR_DATASETS = (*(f"amii/{n}" for n in AMII_NAMES), "zenodo/refold_scfv", "olympus/hplc", "olympus/lnp3")
SNAPSHOT_DATASETS = POOL_DATASETS

# Downloads: (relative path under KALOS_BENCH_DATA, url, sha256)
DOWNLOADS: tuple[tuple[str, str, str], ...] = (
    ("zenodo/refold_tab.zip", "https://zenodo.org/api/records/18270843/files/Results_Tabular.zip/content",
     "dbd820bb63b18e8384ca83091f55b85a54a41c6cea8fd13339cb69302ac88747"),
    ("olympus/hplc.csv", f"https://raw.githubusercontent.com/aspuru-guzik-group/olympus/{OLYMPUS_COMMIT}/"
     "src/olympus/datasets/dataset_hplc/data.csv",
     "9c94222798229c1391f75445f44d9c0ed285e83c1b1e0608ab76b28bf05decef"),
    ("olympus/lnp3.csv", f"https://raw.githubusercontent.com/aspuru-guzik-group/olympus/{OLYMPUS_COMMIT}/"
     "src/olympus/datasets/dataset_lnp3/data.csv",
     "69e8847e30f8b8b8720884676cd20d354152b7093309d278ee9910f9924b48ba"),
)


def fetch(log: Callable[[str], None] = print) -> None:
    """Download the open datasets into KALOS_BENCH_DATA and verify their SHA-256."""
    import zipfile

    root = bench_data_dir()
    for rel, url, sha in DOWNLOADS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            if not url.startswith("https://"):
                raise ValueError(f"refusing non-https url {url}")
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https urls
                path.write_bytes(resp.read())
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != sha:
            raise RuntimeError(f"{rel}: sha256 {got} != expected {sha}")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                zf.extractall(path.parent / "refold")
        log(f"ok {rel}")


def load(name: str) -> RealDataset:
    if name.startswith("user/"):
        raise KeyError("user datasets come from load_user_folder")
    return LOADERS[name]()


def available(names: tuple[str, ...] | list[str]) -> tuple[list[RealDataset], dict[str, str]]:
    """(loaded datasets, {name: reason skipped})."""
    out, skipped = [], {}
    for n in names:
        try:
            out.append(load(n))
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            skipped[n] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return out, skipped


def load_user_folder(folder: str | Path) -> tuple[list[RealDataset], dict[str, str]]:
    """Every `*.csv` in `folder`. A sidecar `<stem>.json` may give
    `{"target": col, "inputs": [...], "categorical": [...], "goal": "maximize"|"minimize",
    "measured": "measured"|"synthetic", "license": "...", "source": "..."}`.
    Without a sidecar the target must be a column named `target`, `titer`, `yield` or
    `y`; inputs default to every other numeric column. A sheet whose target cannot be
    resolved is skipped with the reason."""
    folder = Path(folder).expanduser()
    out, skipped = [], {}
    for path in sorted(folder.glob("*.csv")):
        name = f"user/{path.stem}"
        try:
            df = pd.read_csv(path)
            side = path.with_suffix(".json")
            meta: dict[str, Any] = json.loads(side.read_text()) if side.is_file() else {}
            target = meta.get("target") or next((c for c in ("target", "titer", "yield", "y") if c in df), None)
            if target is None or target not in df:
                skipped[name] = "no target: add a sidecar <stem>.json with {\"target\": column}"
                continue
            cats = [str(c) for c in meta.get("categorical", [])]
            inputs = [str(c) for c in meta.get("inputs", [])] or [
                c for c in df.columns if c != target and c not in cats and pd.api.types.is_numeric_dtype(df[c])]
            cont = [c for c in inputs if c not in cats]
            frame = _finish(df, target, cont, cats)
            cont = [c for c in cont if c in frame]
            cats = [c for c in cats if c in frame]
            if len(frame) < 30 or not (cont or cats):
                skipped[name] = f"{len(frame)} usable recipes (needs 30) or no inputs"
                continue
            out.append(RealDataset(name, "user", frame, target, cont, cats, goal=meta.get("goal", "maximize"),
                                   source=meta.get("source", str(path.name)), license=meta.get("license", "user"),
                                   measured=meta.get("measured", "measured")))
        except (OSError, ValueError, KeyError) as exc:
            skipped[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return out, skipped
