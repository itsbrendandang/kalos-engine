"""Retrospective time-ordered backtest on real data (PROTOCOL.md section 5.6).

Hide the later experiments, let the arm propose from the earlier ones, and score
whether its picks point at the recipes that actually performed best. Results are
labelled "retrospective, pool-restricted": never a wet-lab win.

Per (dataset, ordering, cut): history = recipes before the cut (all their rows),
pool = recipes after it that are not in the history. One arm call, q = Q picks:
the arm's proposals are snapped greedily to the pool (unit-scaled distance, as in
pool replay); arms may propose pool rows directly from `ctx.candidates`.

Metrics (identical to ~/.kalos-dev/REAL_BACKTEST_PLAN.md, which is normative for
the Anagram component):
- M1 normalized simple regret = (max pool truth - max truth among picks) /
  (max pool truth - min pool truth); truth = recipe mean.
- M2 rank of the true best in the arm's ordering of the pool (by `predict`),
  scaled to [0, 1]; None without `predict`.
- M3 Spearman between `predict` and truth over the pool.
- M4 median over the arm's UNSNAPPED proposals of the distance (z-scored over
  all recipes, varying inputs only) to the nearest of the overall top-3 recipes,
  divided by the same median for 200 uniform points in the history's box (< 1
  means closer than random).
Baselines computed here: random (mean M1 over 200 seeded draws), OFAT proxy
(pool recipes ordered by the number of inputs differing from the incumbent by
> 0.5 SD, then standardized distance), the lab's sequence (real orders only: the
first Q pool rows in recorded order).

Orders: real where it exists (HeLa `Round`; T-cell `gen #`; TF-1 (`exp #`,
`gen #`) lexicographic; Anagram S26APR07 -> S26APR29). Otherwise seeded random
orderings of the recipes (`numpy.random.default_rng(seed).permutation`). Cuts: for
seeded orderings 25/50/75 % of recipes; for real orders the distinct step
boundaries nearest to 25/50/75 % of rows (deduplicated). History is capped at the
latest HISTORY_CAP recipes (engine runtime).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .adapter import Arm, RoundContext
from .protocol import LEVELS, Q
from .space import Space

HISTORY_CAP = 400
N_RANDOM = 200
LABEL = "retrospective, pool-restricted"


@dataclass
class Ordered:
    name: str
    group: str  # real | internal
    rows: pd.DataFrame  # raw rows: inputs, target, `_recipe` id, `_step` order key (NaN = no order)
    target: str
    continuous: list[str]
    categorical: list[str] = field(default_factory=list)
    real_order: bool = True
    label: str = ""


def _recipe_ids(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pd.factorize(df[cols].round(9).astype(str).agg("|".join, axis=1))[0]


def _amii_ordered(name: str) -> Ordered:
    from kalos.bench.amii import REGISTRY, amii_data_dir

    spec = REGISTRY[name]
    root = amii_data_dir()
    if root is None or not (root / spec.filename).is_file():
        raise FileNotFoundError(f"{spec.filename}: set KALOS_AMII_DATA")
    df = pd.read_csv(root / spec.filename)
    if spec.fidelity_col:
        df = df[df[spec.fidelity_col] == df[spec.fidelity_col].max()]
    if spec.transform == "log1p_over_poscont":
        y = np.log1p(pd.to_numeric(df[spec.target], errors="coerce").to_numpy(float)
                     / pd.to_numeric(df["PosCont FC"], errors="coerce").to_numpy(float))
    elif spec.transform == "log1p":
        y = np.log1p(pd.to_numeric(df[spec.target], errors="coerce").to_numpy(float))
    else:
        y = pd.to_numeric(df[spec.target], errors="coerce").to_numpy(float)
    cols = list(spec.features) if spec.features else [c for c in df.columns if c not in set(spec.drop) | set(spec.label_cols)]
    cols = [c for c in cols if c not in ("Round", "IS_indicator")]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    if "Round" in df:
        step = df["Round"].astype(float)
    elif "gen #" in df:
        exp_col = "exp #" if "exp #" in df else "exp#"
        step = df[exp_col].astype(float) * 100 + df["gen #"].astype(float)
    else:
        step = pd.Series(np.nan, index=df.index)
    out = x.assign(y=y, _step=step.to_numpy())
    out = out[np.isfinite(out[[*cols, "y"]].to_numpy(float)).all(1)]
    cols = [c for c in cols if out[c].nunique() > 1]
    out["_recipe"] = _recipe_ids(out, cols)
    return Ordered(f"amii/{name}", "real", out.reset_index(drop=True), "y", cols,
                   real_order=bool(np.isfinite(out["_step"]).any()))


def _external_root() -> Path:
    return Path(os.environ.get("KALOS_EXTERNAL_DATA", "~/kalos/external")).expanduser()


def _anagram_jan06() -> Ordered:
    """S26JAN06 bench rows (3 mL), recipe = sample ID; no real order (one plate)."""
    path = _external_root() / "bioqore" / "bioqore-synthetic-data-project" / "data" / "raw" / "anagram_compiled.csv"
    if not path.is_file():
        raise FileNotFoundError("anagram_compiled.csv: set KALOS_EXTERNAL_DATA (internal data)")
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["Culture_Volume.mL"], errors="coerce") == 3]
    drop = {"sample", "production_OD", "lipase_g.L"}
    cols = [c for c in df.columns if c not in drop and pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out["lipase_g_L"] = pd.to_numeric(df["lipase_g.L"], errors="coerce")
    out["_recipe"] = pd.factorize(df["sample"])[0]
    out["_step"] = np.nan
    out = out.dropna(subset=[*cols, "lipase_g_L"])
    return Ordered("internal/anagram_S26JAN06", "internal", out.reset_index(drop=True), "lipase_g_L", cols,
                   real_order=False, label="internal, data rights unconfirmed")


def _anagram_apr() -> Ordered:
    """S26APR07 (24 h plate only) then S26APR29: the one real chronological cut."""
    path = (_external_root() / "bioqore" / "ad_hoc_data_analyses" / "temp_data_storage"
            / "Combined_S26APR07_and_S26APR29_protein_expression_and_media_composition.tsv")
    if not path.is_file():
        raise FileNotFoundError("combined APR07/APR29 tsv: set KALOS_EXTERNAL_DATA (internal data)")
    df = pd.read_csv(path, sep="\t")
    name = df["Sample Name"].astype(str)
    df = df[~name.str.contains("BMMY|Blank", case=False) & ~name.str.contains("8 hours")]
    for c in ("Potassium_hydroxide", "Ammonia_hydroxide", "PPG-2000"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    skip = {"Sample Name", "24 well culture plate", "24 well culture plate position", "96 well analytics plate",
            "96 well analytics plate position", "Revvity Well", "Experiment", "Size [kDa]", "Conc. (ng/ul)",
            "% Purity", "Medium", "pH Before Adjustment", "Lipase_g.L"}
    cols = [c for c in df.columns if c not in skip and pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out["lipase_g_L"] = pd.to_numeric(df["Lipase_g.L"], errors="coerce")
    out["_step"] = (df["Experiment"].astype(str) == "S26APR29").astype(float).to_numpy()
    out = out.dropna(subset=[*cols, "lipase_g_L"])
    out["_recipe"] = _recipe_ids(out, cols)
    return Ordered("internal/anagram_APR07_to_APR29", "internal", out.reset_index(drop=True), "lipase_g_L", cols,
                   real_order=True, label="internal, data rights unconfirmed; zeros are censored non-detects")


LOADERS = {
    "amii/HeLa_regular": lambda: _amii_ordered("HeLa_regular"),
    "amii/HeLa_timesaving": lambda: _amii_ordered("HeLa_timesaving"),
    "amii/Tcell_expansion": lambda: _amii_ordered("Tcell_expansion"),
    "amii/TF1_expansion": lambda: _amii_ordered("TF1_expansion"),
    "amii/DBO_rat_myocyte": lambda: _amii_ordered("DBO_rat_myocyte"),
    "amii/MOBO_rat_myocyte": lambda: _amii_ordered("MOBO_rat_myocyte"),
    "internal/anagram_S26JAN06": _anagram_jan06,
    "internal/anagram_APR07_to_APR29": _anagram_apr,
}
TUNING_BACKTESTS = ("amii/HeLa_timesaving", "amii/Tcell_expansion", "amii/DBO_rat_myocyte")
HELDOUT_BACKTESTS = ("amii/HeLa_regular", "amii/TF1_expansion", "amii/MOBO_rat_myocyte",
                     "internal/anagram_S26JAN06", "internal/anagram_APR07_to_APR29")


def _cuts(d: Ordered, seed: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(cut name, history recipe ids, pool recipe ids)]."""
    rec = d.rows["_recipe"].to_numpy()
    if d.real_order:
        first = d.rows.groupby("_recipe")["_step"].min()
        steps = np.sort(first.unique())
        if len(steps) < 2:
            return []
        n_rows = d.rows.groupby("_recipe").size()
        cum = np.array([n_rows[first.index[first <= s]].sum() for s in steps[:-1]]) / len(d.rows)
        chosen = sorted({int(np.argmin(np.abs(cum - f))) for f in (0.25, 0.5, 0.75)})
        out = []
        for k in chosen:
            s = steps[k]
            hist = first.index[first <= s].to_numpy()
            pool = first.index[first > s].to_numpy()
            out.append((f"after_step_{s:g}", hist, pool))
        return out
    ids = np.unique(rec)
    order = np.random.default_rng(seed).permutation(ids)
    return [(f"k{int(f * 100)}pct", order[: int(round(f * len(ids)))], order[int(round(f * len(ids))):])
            for f in (0.25, 0.5, 0.75)]


def _space(d: Ordered, frame: pd.DataFrame) -> Space:
    return Space(continuous={c: (float(frame[c].min()), float(frame[c].max())) for c in d.continuous},
                 categorical={c: tuple(sorted(frame[c].astype(str).unique())) for c in d.categorical})


def _m1(truth_pool: np.ndarray, picked: np.ndarray) -> float | None:
    rng_ = truth_pool.max() - truth_pool.min()
    if rng_ <= 0 or not len(picked):
        return None
    return float((truth_pool.max() - picked.max()) / rng_)


def run_backtest(arm: Arm, task: str, seed: int) -> dict[str, Any]:
    d = LOADERS[task]()
    if d.real_order and seed not in (0, 1000, 5001, 9002):
        return {"skipped": "real order: one history per split (first seed only)", "group": "real_backtest"}
    rows = d.rows
    recipe_truth = rows.groupby("_recipe")[d.target].mean()
    recipe_x = rows.groupby("_recipe")[[*d.continuous, *d.categorical]].first()
    z_cols = [c for c in d.continuous if recipe_x[c].std() > 0]
    zx = (recipe_x[z_cols] - recipe_x[z_cols].mean()) / recipe_x[z_cols].std()
    top3 = recipe_truth.sort_values(ascending=False).index[:3]
    sd_all = recipe_x[d.continuous].std().replace(0, 1.0)
    cuts_out = []
    for cut, hist_ids, pool_ids in _cuts(d, seed):
        hist_ids = hist_ids[-HISTORY_CAP:]
        if len(pool_ids) < Q + 1 or len(hist_ids) < 5:
            continue
        hist = rows[rows["_recipe"].isin(hist_ids)]
        sheet = hist[[*d.continuous, *d.categorical, d.target]].reset_index(drop=True)
        pool_x = recipe_x.loc[pool_ids].reset_index(drop=True)
        truth_pool = recipe_truth.loc[pool_ids].to_numpy(float)
        sp = _space(d, rows)
        roles = {"categoricals": list(d.categorical)} if d.categorical else {}
        ctx = RoundContext("backtest", task, seed, 1, sheet, d.target, sp, Q, LEVELS, roles, candidates=pool_x)
        from .suites import _call, _verdict_rec

        out, cost = _call(arm, ctx)
        recs = [sp.project(p.recipe) for p in out.proposals[:Q]]
        # greedy snap to the pool
        pu = sp.unit(pool_x)
        free = np.ones(len(pool_x), bool)
        picks: list[int] = []
        for r in recs:
            dd = ((pu - sp.unit(pd.DataFrame([r]))) ** 2).sum(1)
            dd = np.where(free, dd, np.inf)
            j = int(np.argmin(dd))
            picks.append(j)
            free[j] = False
        m1 = _m1(truth_pool, truth_pool[picks]) if picks else None
        m2 = m3 = None
        if out.predict is not None:
            try:
                score = np.asarray(out.predict(pool_x), float)
                order = np.argsort(-score)
                best = int(np.argmax(truth_pool))
                m2 = float(np.flatnonzero(order == best)[0] / max(1, len(order) - 1))
                if np.ptp(score) > 0:
                    m3 = float(spearmanr(score, truth_pool).statistic)
            except Exception:  # noqa: BLE001 - a failing predict leaves M2/M3 unavailable
                m2 = m3 = None
        m4 = None
        if recs and z_cols:
            pz = (pd.DataFrame(recs)[z_cols] - recipe_x[z_cols].mean()) / recipe_x[z_cols].std()
            dist = np.sqrt(((pz.to_numpy()[:, None, :] - zx.loc[top3].to_numpy()[None]) ** 2).sum(-1)).min(1)
            hsp = _space(d, hist)
            rnd = hsp.sample(N_RANDOM, np.random.default_rng([seed, 99]))
            rz = (rnd[z_cols] - recipe_x[z_cols].mean()) / recipe_x[z_cols].std()
            rdist = np.sqrt(((rz.to_numpy()[:, None, :] - zx.loc[top3].to_numpy()[None]) ** 2).sum(-1)).min(1)
            m4 = float(np.median(dist) / max(np.median(rdist), 1e-12))
        rng = np.random.default_rng([seed, 7])
        rand_m1 = [_m1(truth_pool, truth_pool[rng.choice(len(truth_pool), Q, replace=False)]) for _ in range(N_RANDOM)]
        # OFAT proxy from the incumbent
        inc = recipe_truth.loc[hist_ids].idxmax()
        inc_row = recipe_x.loc[[inc], d.continuous].iloc[0] if d.continuous else None
        diff = (np.abs(pool_x[d.continuous] - inc_row) / sd_all) if inc_row is not None else None
        ofat_m1 = None
        if diff is not None:
            n_diff = (diff > 0.5).sum(1).to_numpy()
            dist_o = np.sqrt((diff ** 2).sum(1)).to_numpy()
            ofat = np.lexsort((dist_o, n_diff))[:Q]
            ofat_m1 = _m1(truth_pool, truth_pool[ofat])
        lab_m1 = None
        if d.real_order:
            pool_set = set(pool_ids.tolist())
            lab_ids = [i for i in rows["_recipe"].drop_duplicates() if i in pool_set][:Q]
            lab_m1 = _m1(truth_pool, recipe_truth.loc[lab_ids].to_numpy(float))
        cuts_out.append({
            "cut": cut, "n_history_recipes": int(len(hist_ids)), "n_history_rows": int(len(hist)),
            "n_pool": int(len(pool_ids)), **cost, **_verdict_rec(out), "n_arm_proposals": len(recs),
            "M1": m1, "M2": m2, "M3": m3, "M4": m4,
            "M1_random_mean": float(np.mean([v for v in rand_m1 if v is not None])),
            "M1_ofat": ofat_m1, "M1_lab": lab_m1,
            "top3_in_history": int(np.isin(top3, hist_ids).sum()),
        })
    return {"label": LABEL + ("; " + d.label if d.label else ""), "dataset": d.name, "real_order": d.real_order,
            "group": "internal" if d.group == "internal" else "real_backtest", "cuts": cuts_out}
