"""Real-data validation: replay a BO loop on a recorded fermentation run sheet.

The dataset (examples/datasets/dataset_0.csv) is a fixed table, so there is no
live assay to query. The loop is therefore a *pool replay*: each round fits a
`Surrogate`, asks `propose` for `q` points in the normalized design box, snaps
each to the nearest not-yet-run recorded row, and "measures" it by revealing
that row's recorded titer. The same loop runs on the synthetic fixture so the
two convergence curves are directly comparable.

Pure functions, no network, no writes except `write_report` (caller-chosen
paths). Torch/botorch are imported lazily so importing this module is cheap.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kalos.domains.profile import ColumnRoles


@dataclass(frozen=True)
class RealDataMapping:
    """Which columns play which role. `timepoint` is a process input here
    (harvest duration), so it is modelled as one of the features."""

    titer: str
    timepoint: str
    drivers: tuple[str, ...]
    ids: tuple[str, ...] = ()
    contamination: str | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return (*self.drivers, self.timepoint)

    def to_column_roles(self) -> ColumnRoles:
        return ColumnRoles(target=self.titer, features=self.features, ids=self.ids)


# dataset_0.csv: the target is the cellulase yield; there is no time series, so
# duration_days (harvest time) is the only timepoint-like column.
DATASET_0 = RealDataMapping(
    titer="max_yield_fpu_ml",
    timepoint="duration_days",
    drivers=(
        "temp_c", "ph", "min_do_pct", "inoculum_vol_ml", "rpm",
        "lactose_feed_ml", "total_media_vol_ml",
    ),
    ids=("run_id", "yield_tier", "data_type", "contamination"),
    contamination="contamination",
)


def load_real_frame(
    path: str | Path, mapping: RealDataMapping = DATASET_0, *, exclude_contaminated: bool = True
) -> pd.DataFrame:
    """Read the CSV and keep rows with finite values in every modelled column.

    `exclude_contaminated` drops rows flagged `Y` in the contamination column
    (those are not "clean" runs).
    """
    p = Path(path).resolve()
    if p.suffix.lower() != ".csv" or not p.is_file():
        raise ValueError("dataset path must be an existing .csv file")
    df = pd.read_csv(p)
    needed = [mapping.titer, *mapping.features]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if exclude_contaminated and mapping.contamination in df.columns:
        df = df[df[mapping.contamination].astype(str).str.upper() != "Y"]
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed)
    return df.reset_index(drop=True)


def frame_to_pool(df: pd.DataFrame, mapping: RealDataMapping) -> tuple[np.ndarray, np.ndarray]:
    return df[list(mapping.features)].to_numpy(float), df[mapping.titer].to_numpy(float)


def _normalize(X: np.ndarray) -> np.ndarray:
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def run_pool_bo(
    X: np.ndarray, y: np.ndarray, *, rounds: int = 10, q: int = 2, n_init: int = 6, seed: int = 0
) -> dict:
    """One BO pool-replay trial. Returns best-so-far after each round (index 0 =
    after the initial random design) plus a random-search baseline with the
    same evaluation budget."""
    import torch

    from kalos.core.optimize import propose
    from kalos.core.surrogate import FitError, Surrogate

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < n_init + rounds * q:
        raise ValueError(f"pool of {n} rows is too small for n_init={n_init}, rounds={rounds}, q={q}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    Xn = _normalize(X)
    bounds = np.vstack([np.zeros(X.shape[1]), np.ones(X.shape[1])])
    order = rng.permutation(n)
    init = [int(i) for i in order[:n_init]]

    evaluated = list(init)
    remaining = set(int(i) for i in order[n_init:])
    bo_curve = [float(y[evaluated].max())]
    fit_failures = 0
    for r in range(rounds):
        picks: list[int] = []
        try:
            s = Surrogate().fit(Xn[evaluated], y[evaluated], bounds=bounds)
            props = propose(s, bounds, q=q, seed=seed * 1000 + r)
            for p in np.atleast_2d(props):
                rem = np.array(sorted(remaining - set(picks)))
                picks.append(int(rem[np.argmin(((Xn[rem] - p) ** 2).sum(1))]))
        except FitError:
            fit_failures += 1
        while len(picks) < q:  # fallback: random fill keeps the budget equal
            picks.append(int(rng.choice(sorted(remaining - set(picks)))))
        for i in picks:
            remaining.discard(i)
            evaluated.append(i)
        bo_curve.append(float(y[evaluated].max()))

    rand_rng = np.random.default_rng(seed + 10_000)
    rorder = rand_rng.permutation(n)
    rand_eval = list(rorder[:n_init])
    rand_curve = [float(y[rand_eval].max())]
    pos = n_init
    for _ in range(rounds):
        rand_eval += list(rorder[pos:pos + q])
        pos += q
        rand_curve.append(float(y[rand_eval].max()))

    # LHS baseline: a space-filling design in the normalized box, each point
    # snapped to its nearest distinct recorded row (same budget as BO).
    from scipy.stats import qmc

    n_total = n_init + rounds * q
    lhs_pts = qmc.LatinHypercube(d=X.shape[1], seed=seed + 20_000).random(n_total)
    taken: list[int] = []
    for p in lhs_pts:
        avail = np.setdiff1d(np.arange(n), taken)
        taken.append(int(avail[np.argmin(((Xn[avail] - p) ** 2).sum(1))]))
    lhs_curve = [float(y[taken[: n_init + k * q]].max()) for k in range(rounds + 1)]

    return {"bo": bo_curve, "random": rand_curve, "lhs": lhs_curve, "fit_failures": fit_failures}


def _first_round_reaching(curve: np.ndarray, frac: float) -> int | None:
    hit = np.nonzero(curve >= frac)[0]
    return int(hit[0]) if hit.size else None


def convergence_summary(
    X: np.ndarray, y: np.ndarray, *, rounds: int, q: int, n_init: int, seeds: list[int]
) -> dict:
    """Multi-seed pool replay, curves normalized to the pool's best titer (1.0 =
    found the global best recorded row) so datasets with different units compare."""
    y_max = float(np.max(y))
    if not np.isfinite(y_max) or y_max <= 0:
        raise ValueError("pool has no positive titer to normalize against")
    trials = [run_pool_bo(X, y, rounds=rounds, q=q, n_init=n_init, seed=s) for s in seeds]
    bo = np.array([t["bo"] for t in trials]) / y_max
    rd = np.array([t["random"] for t in trials]) / y_max
    lh = np.array([t["lhs"] for t in trials]) / y_max
    bo_mean = bo.mean(0)
    return {
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "pool_best": y_max,
        "seeds": list(seeds),
        "rounds": rounds,
        "q": q,
        "n_init": n_init,
        "bo_mean": bo_mean.tolist(),
        "bo_std": bo.std(0).tolist(),
        "random_mean": rd.mean(0).tolist(),
        "random_std": rd.std(0).tolist(),
        "lhs_mean": lh.mean(0).tolist(),
        "lhs_std": lh.std(0).tolist(),
        "final_bo": float(bo_mean[-1]),
        "final_bo_std": float(bo[:, -1].std()),
        "final_random": float(rd.mean(0)[-1]),
        "final_lhs": float(lh.mean(0)[-1]),
        "final_per_seed": {"bo": bo[:, -1].tolist(), "random": rd[:, -1].tolist(), "lhs": lh[:, -1].tolist()},
        "bo_gain_over_random": float(bo_mean[-1] - rd.mean(0)[-1]),
        "round_to_95pct_best": _first_round_reaching(bo_mean, 0.95),
        "fit_failures": int(sum(t["fit_failures"] for t in trials)),
    }


def compare(
    real: tuple[np.ndarray, np.ndarray],
    synthetic: tuple[np.ndarray, np.ndarray],
    *,
    rounds: int = 10,
    q: int = 2,
    n_init: int = 6,
    seeds: list[int] | None = None,
) -> dict:
    seeds = list(seeds) if seeds is not None else list(range(5))
    if len(seeds) < 5:
        raise ValueError("use at least 5 seeds so seed variance is meaningful")
    out = {
        "real": convergence_summary(*real, rounds=rounds, q=q, n_init=n_init, seeds=seeds),
        "synthetic": convergence_summary(*synthetic, rounds=rounds, q=q, n_init=n_init, seeds=seeds),
    }
    out["final_gap_real_minus_synthetic"] = out["real"]["final_bo"] - out["synthetic"]["final_bo"]
    return out


def plotly_html(summary: dict, title: str = "Kalos BO convergence: hybrid/semi-synthetic fermentation data vs synthetic fixture") -> str:
    """Self-contained Plotly chart (plotly.js from the CDN; no Python plotly dependency)."""
    traces = []
    for name, color in (("real", "#2a9d8f"), ("synthetic", "#e76f51")):
        s = summary[name]
        x = list(range(len(s["bo_mean"])))
        traces.append({"x": x, "y": s["bo_mean"], "name": f"{name} BO", "mode": "lines+markers",
                       "line": {"color": color},
                       "error_y": {"type": "data", "array": s["bo_std"], "visible": True}})
        traces.append({"x": x, "y": s["random_mean"], "name": f"{name} random", "mode": "lines",
                       "line": {"color": color, "dash": "dot"}})
        if "lhs_mean" in s:
            traces.append({"x": x, "y": s["lhs_mean"], "name": f"{name} LHS", "mode": "lines",
                           "line": {"color": color, "dash": "dash"}})
    layout = {"title": title, "xaxis": {"title": "BO round (q per round)"},
              "yaxis": {"title": "best-so-far / pool best", "range": [0, 1.05]}}
    # json.dumps output is safe inside <script> once "</" is neutralized.
    payload = json.dumps({"data": traces, "layout": layout}).replace("</", "<\\/")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>"
        "<body><div id='chart' style='width:100%;height:560px'></div>"
        f"<script>const f={payload};Plotly.newPlot('chart',f.data,f.layout);</script>"
        "</body></html>"
    )


def write_report(summary: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jp, hp = out / "real_data_summary.json", out / "real_data_convergence.html"
    jp.write_text(json.dumps(summary, indent=2))
    hp.write_text(plotly_html(summary))
    return jp, hp


# --- IndPenSim (optional, never committed) -------------------------------- #
# Mendeley DOI 10.17632/pdnjz7zz5x.2, CC BY 4.0: 100 simulated 100,000 L
# penicillin batches. The file is very large, so it is read from disk only when
# the caller points at it. Column defaults below are NOT verified against the
# real file (it was not downloaded here); override them via the arguments.
INDPENSIM_BATCH_COL = "Batch ID"
INDPENSIM_TARGET_COL = "Penicillin concentration(P:g/L)"
INDPENSIM_RECIPE_COLS = (
    "Temperature(T:K)", "pH(pH:pH)", "Aeration rate(Fg:L/h)", "Agitator RPM(RPM:RPM)",
    "Sugar feed rate(Fs:L/h)",
)


def indpensim_path(explicit: str | Path | None = None) -> Path | None:
    """Explicit path, else $KALOS_INDPENSIM_PATH, else a CSV under $KALOS_MEDIA_DATA."""
    import os

    cand = explicit or os.environ.get("KALOS_INDPENSIM_PATH")
    if cand:
        p = Path(cand).expanduser().resolve()
        return p if p.is_file() else None
    media = os.environ.get("KALOS_MEDIA_DATA")
    if media:
        for p in sorted(Path(media).expanduser().resolve().glob("*IndPenSim*.csv")):
            return p
    return None


def load_indpensim_pool(
    path: str | Path | None = None,
    *,
    batch_col: str = INDPENSIM_BATCH_COL,
    target_col: str = INDPENSIM_TARGET_COL,
    recipe_cols: tuple[str, ...] = INDPENSIM_RECIPE_COLS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aggregate per batch: recipe = per-batch mean of each recipe column,
    target = final penicillin value of the batch. Returns (X, y, feature_names).
    Streams only the needed columns, so memory stays bounded.
    """
    p = indpensim_path(path)
    if p is None or p.suffix.lower() != ".csv":
        raise FileNotFoundError(
            "IndPenSim CSV not found: pass a path or set KALOS_INDPENSIM_PATH / KALOS_MEDIA_DATA"
        )
    cols = [batch_col, target_col, *recipe_cols]
    header = pd.read_csv(p, nrows=0).columns
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(f"IndPenSim file is missing columns {missing}; override the column names")
    df = pd.read_csv(p, usecols=cols)
    for c in (target_col, *recipe_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    g = df.dropna(subset=[target_col]).groupby(batch_col, sort=False)
    y = g[target_col].last()
    X = g[list(recipe_cols)].mean().loc[y.index]
    keep = np.isfinite(X.to_numpy()).all(1) & np.isfinite(y.to_numpy())
    return X.to_numpy(float)[keep], y.to_numpy(float)[keep], list(recipe_cols)
