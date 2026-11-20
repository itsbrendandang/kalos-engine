"""Internal multi-scale CHO clone data (LeadGene / Cytena via the Voyager repo).

INTERNAL, DATA RIGHTS UNCONFIRMED. The file is never read from or written to
this repository. Point `KALOS_EXTERNAL_DATA` at the folder that holds
`leadgene_cho_dynamics.csv` (or pass `--leadgene PATH`). Results that use it are
labelled `INTERNAL_LABEL`, carry only aggregate metrics (no clone-level rows),
are report-only, and decide no criterion (PROTOCOL.md section 3.4).

Layout of the file: transposed, one column per clone (S1: 96, S2: 384, S3: 24),
one row per measurement. Clones go 384-well -> 96-well -> 24-well -> SF125
flasks; only the clones the lab ranked highest advance, so every larger-scale
outcome exists only for a SELECTED subset (range restriction: within-subset
rank correlations understate the association over all clones).

Tasks built here:
- `rank_96to24`: per clone, the 384w confluence and the 96w titer signal and
  confluence predict the 24w titer signal. Rows: clones measured at 24w (S1 24,
  S2 24, S3 24). Scored on the held-out clones: Spearman (ranking), top-8
  precision (the next selection), interval coverage and width. Baselines:
  carry-forward (the 96w titer signal itself as the score) and random.
- `transfer_loso`: a scale-transfer ladder in within-series percentile units
  (pct of the primary readout at the smaller scale, pct of its growth readout,
  the transition index) predicting the next-scale primary-readout percentile.
  Transitions: 384w->96w, 96w->24w, 24w->SF125 (flask1 signal for S1, flask2
  mg/L for S1 and S2). Leave-one-transition-out (LOSO).
- `flask_small`: 24w -> SF125 with 8-11 clones. Too few for a model: an honest
  arm refuses or reports not trusted; recorded, never pooled.

Splits (a series S1/S2/S3 is a campaign): tuning uses S1 and S2 only
(train on one, test on the other, and seeded 4-fold clone splits); the held-out
evaluation is leave-one-campaign-out over S1, S2, S3 (rank_96to24) and
leave-one-transition-out (transfer_loso). Mandatory baseline: the previous-scale
readout (carry-forward rank). Interval coverage under the campaign split is a
trust metric (coverage under campaign shift).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

INTERNAL_LABEL = "internal, data rights unconfirmed"
FILENAME = "leadgene_cho_dynamics.csv"
RANK_FEATURES = ["conf384", "conf96", "signal96"]
RANK_TARGET = "signal24"
TRANSFER_FEATURES = ["prev_primary_pct", "prev_growth_pct", "transition"]
TRANSFER_TARGET = "next_primary_pct"


def leadgene_path(explicit: str | Path | None = None) -> Path | None:
    cands = [Path(explicit).expanduser()] if explicit else []
    if os.environ.get("KALOS_EXTERNAL_DATA"):
        root = Path(os.environ["KALOS_EXTERNAL_DATA"]).expanduser()
        cands += [root / FILENAME, root / "voyager" / FILENAME]
    for p in cands:
        if p.is_file():
            return p.resolve()
    return None


def load_clones(path: Path) -> pd.DataFrame:
    """One row per clone, numeric readouts, `series` S1/S2/S3. Lab ranks and
    media are kept out of the features (ranks are the lab's decisions)."""
    t = pd.read_csv(path, index_col=0)
    df = t.T
    out = pd.DataFrame(index=df.index)
    out["series"] = df.index.str.extract(r"^(S\d+)", expand=False).to_numpy()

    def num(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)

    out["conf384"] = num("Confluence_%_384w")
    out["conf96"] = num("Confluence_%_96w")
    out["signal96"] = num("F.quant_titerSignal_96w")
    out["signal24"] = num("F.quant_titerSignal_24w")
    out["vcd24"] = num("VCD_24w")
    out["flask_signal"] = num("F.quant_titerSignal_SF125_flask1")
    out["flask_mgl"] = num("F.quant_titer_mgL_SF125_flask2")
    out["lab_rank96"] = num("rank_96w")
    return out.reset_index(drop=True)


def rank_frame(clones: pd.DataFrame) -> pd.DataFrame:
    keep = clones[RANK_FEATURES + [RANK_TARGET]].notna().all(axis=1)
    return clones.loc[keep, ["series", *RANK_FEATURES, RANK_TARGET]].reset_index(drop=True)


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def transfer_frame(clones: pd.DataFrame) -> pd.DataFrame:
    """Rows (clone, transition) in within-series percentile units."""
    rows = []
    for _series, g in clones.groupby("series"):
        steps = [
            (0, "conf384", "conf384", "signal96"),
            (1, "signal96", "conf96", "signal24"),
            (2, "signal24", "vcd24", "flask_signal"),
            (2, "signal24", "vcd24", "flask_mgl"),
        ]
        for k, prev, growth, nxt in steps:
            sub = g[g[[prev, growth, nxt]].notna().all(axis=1)]
            if len(sub) < 3:
                continue
            rows.append(pd.DataFrame({
                "series": sub["series"], "transition": float(k),
                "prev_primary_pct": _pct(sub[prev]), "prev_growth_pct": _pct(sub[growth]),
                TRANSFER_TARGET: _pct(sub[nxt]),
            }))
    return pd.concat(rows, ignore_index=True)


def flask_frame(clones: pd.DataFrame) -> pd.DataFrame:
    sub = clones[clones[["conf384", "signal96", "signal24", "vcd24", "flask_mgl"]].notna().all(axis=1)]
    return pd.DataFrame(sub[["series", "conf384", "signal96", "signal24", "vcd24", "flask_mgl"]]).reset_index(drop=True)


def splits(task: str, split: str, seed: int, frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(fold name, train index, test index)] for one (task, split, seed)."""
    idx = np.arange(len(frame))
    ser = frame["series"].to_numpy()
    if task == "rank_96to24":
        if split == "heldout":  # leave-one-campaign-out over the three series
            return [(f"loco_{s}", idx[ser != s], idx[ser == s]) for s in ("S1", "S2", "S3")]
        tune = idx[ser != "S3"]
        folds = [("train_S1_test_S2", idx[ser == "S1"], idx[ser == "S2"]),
                 ("train_S2_test_S1", idx[ser == "S2"], idx[ser == "S1"])]
        rng = np.random.default_rng([seed, 4242])
        perm = rng.permutation(tune)
        for k in range(4):
            te = np.sort(perm[k::4])
            folds.append((f"kfold{k}", np.setdiff1d(tune, te), te))
        return folds
    if task == "transfer_loso":
        tr = frame["transition"].to_numpy()
        if split == "heldout":
            return [(f"loso_t{int(k)}", idx[tr != k], idx[tr == k]) for k in sorted(set(tr))]
        tune = idx[ser != "S3"]
        rng = np.random.default_rng([seed, 4343])
        perm = rng.permutation(tune)
        return [(f"kfold{k}", np.setdiff1d(tune, np.sort(perm[k::4])), np.sort(perm[k::4])) for k in range(4)]
    if task == "flask_small":
        return [(f"loo{i}", np.setdiff1d(idx, [i]), np.array([i])) for i in idx]
    raise KeyError(task)


TASKS = {
    "rank_96to24": (rank_frame, RANK_FEATURES, RANK_TARGET),
    "transfer_loso": (transfer_frame, TRANSFER_FEATURES, TRANSFER_TARGET),
    "flask_small": (flask_frame, ["conf384", "signal96", "signal24", "vcd24"], "flask_mgl"),
}
