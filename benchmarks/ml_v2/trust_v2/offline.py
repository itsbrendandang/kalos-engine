"""Offline scoring of interval variants on captured reference trajectories (TUNING only).

Each variant is a calibration `cal(reps, level) -> (q, floor, kind)` computed from the
binding grouped-CV OOF repeats exactly as the engine would, and a band
`mu +/- half(sd)`. Scored like `report.py`: coverage at proposals per task with a
cluster bootstrap over instances, median width / (f* - f_med), held-out coverage.
"""
from __future__ import annotations

import gzip
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEVELS = (0.9, 0.95)


def same_partition(a, b) -> bool:
    fa = pd.factorize(np.asarray([str(v) for v in a]))[0]
    fb = pd.factorize(np.asarray([str(v) for v in b]))[0]
    return len(fa) == len(fb) and bool(np.array_equal(fa, fb))


def binding_reps(call: dict[str, Any]) -> list[dict[str, Any]]:
    bg = call.get("binding_groups")
    out = []
    for o in call["oof"]:
        if bg is not None and same_partition(o["groups"], bg):
            out.append(o)
    # one grouped_cv_report per partition: keep the first run of repeats (shuffle False first)
    if not out:
        return []
    first = [out[0]]
    for o in out[1:]:
        if not o["shuffle"]:
            break
        first.append(o)
    return first


def load(dirs: Iterable[Path]) -> list[dict[str, Any]]:
    recs = []
    for d in dirs:
        for f in sorted(Path(d).rglob("*.pkl.gz")):
            with gzip.open(f, "rb") as fh:
                recs.append(pickle.load(fh))
    return recs


# ------------------------------------------------------------- calibrations
def _rank(n: int, lv: float) -> int:
    return int(np.ceil((n + 1) * lv - 1e-9))


def select(reps, top_fraction: float, rank_key: str = "mean", kappa: float = 0.0):
    """Top `top_fraction` of each fold's held-out runs by the fold model's ranking."""
    idx_all = []
    for ri, rep in enumerate(reps):
        p, s, f = rep["pred"], rep["sd"], rep["fold"]
        key = p + kappa * s if rank_key == "ucb" else p
        for k in np.unique(f):
            idx = np.where(f == k)[0]
            m = int(np.ceil(top_fraction * len(idx) - 1e-9))
            top = idx[np.argsort(-key[idx], kind="stable")[:m]]
            idx_all.extend((ri, int(t)) for t in top)
    return idx_all


def cal_reference(reps, lv):
    """The shipped F15 band and its plain normalized fallback (repeat 0)."""
    all_sd = np.concatenate([r["sd"] for r in reps])
    floor = float(np.quantile(all_sd, 0.9))
    sel = select(reps, 0.3)
    sc = np.array([abs(reps[ri]["actual"][i] - reps[ri]["pred"][i]) / reps[ri]["sd"][i] for ri, i in sel])
    k = _rank(len(sc), lv)
    if len(sc) and k <= len(sc):
        return {"q": float(np.sort(sc)[k - 1]), "floor": floor, "kind": "sel"}
    r0 = reps[0]
    s0 = np.abs(r0["actual"] - r0["pred"]) / r0["sd"]
    k = _rank(len(s0), lv)
    if k <= len(s0):
        return {"q": float(np.sort(s0)[k - 1]), "floor": 0.0, "kind": "norm"}
    return None


def make_cal_floored(floor_q: float = 0.9, top_fraction: float = 0.3, rank_key: str = "mean", kappa: float = 0.0,
                     fallback: str = "expand", score: str = "floored", extra_ratio: float | None = None):
    """Consistent selection-aware score: s' = max(s, floor) in calibration AND in the band.

    fallback when too few selected runs for the level's rank:
      - "expand": grow the top fraction (0.3 -> 0.5 -> 1.0) until the rank exists;
      - "ratio": q(lv) = q(0.90) * extra_ratio (Gaussian ratio by default);
      - "ref": the shipped fallback (plain normalized band).
    """
    def qfor(reps, lv, tf):
        all_sd = np.concatenate([r["sd"] for r in reps])
        floor = float(np.quantile(all_sd, floor_q)) if floor_q > 0 else 0.0
        sel = select(reps, tf, rank_key, kappa)
        if score == "abs":
            sc = np.array([abs(reps[ri]["actual"][i] - reps[ri]["pred"][i]) for ri, i in sel])
        else:
            sc = np.array([abs(reps[ri]["actual"][i] - reps[ri]["pred"][i]) / max(reps[ri]["sd"][i], floor)
                           for ri, i in sel])
        k = _rank(len(sc), lv)
        if len(sc) and k <= len(sc):
            return {"q": float(np.sort(sc)[k - 1]), "floor": floor, "kind": "abs" if score == "abs" else "sel",
                    "n_sel": len(sc)}
        return None

    def cal(reps, lv):
        out = qfor(reps, lv, top_fraction)
        if out is not None:
            return out
        if fallback == "expand":
            for tf in (0.5, 1.0):
                if tf <= top_fraction:
                    continue
                out = qfor(reps, lv, tf)
                if out is not None:
                    out["kind"] += f"_tf{tf}"
                    return out
            return None
        if fallback == "ratio":
            base = qfor(reps, 0.9, top_fraction)
            if base is None:
                return None
            from scipy.stats import norm
            r = extra_ratio if extra_ratio is not None else norm.ppf(0.5 + lv / 2) / norm.ppf(0.95)
            base["q"] *= r
            base["kind"] += "_ratio"
            return base
        return cal_reference(reps, lv)

    return cal


def half_width(c, sd):
    if c is None:
        return None
    if c["kind"].startswith("abs"):
        return np.full_like(sd, c["q"])
    return c["q"] * np.maximum(sd, c["floor"])


# ------------------------------------------------------------------ scoring
def evaluate(recs, cal: Callable, levels=LEVELS) -> dict[str, Any]:
    """Per task: per-instance (hits, total) at proposals and held-out, widths."""
    rows = []
    for rec in recs:
        if rec.get("error") or rec["suite"] not in ("closed_loop", "emulator"):
            continue
        scale = rec["scale"]
        for c in rec["calls"]:
            if c.get("error") or "obs_props_post" not in c:
                continue
            reps = binding_reps(c)
            if not reps:
                continue
            n_props = c["n_props"]
            pp, y = c["obs_props_post"], c["obs_props"][:n_props]
            hp, yh = c.get("obs_ho_post"), c.get("obs_ho")
            for lv in levels:
                cc = cal(reps, lv)
                if cc is None or n_props == 0:
                    continue
                h = half_width(cc, pp["sd"][:n_props])
                mu = pp["mu"][:n_props]
                ok = np.isfinite(h) & np.isfinite(mu)
                ins = ((y >= mu - h) & (y <= mu + h))[ok]
                row = {"suite": rec["suite"], "task": rec["task"], "seed": rec["seed"], "round": c["round"],
                       "lv": lv, "hits": int(ins.sum()), "tot": int(ok.sum()),
                       "widths": list(2 * h[ok] / scale), "kind": cc["kind"]}
                if hp is not None:
                    hh = half_width(cc, hp["sd"])
                    okh = np.isfinite(hh) & np.isfinite(hp["mu"])
                    insh = ((yh >= hp["mu"] - hh) & (yh <= hp["mu"] + hh))[okh]
                    row.update(ho_hits=int(insh.sum()), ho_tot=int(okh.sum()),
                               ho_width=float(np.median(2 * hh[okh] / scale)) if okh.any() else np.nan)
                rows.append(row)
    return rows


def cluster_ci(cl, n_boot=2000, seed=20261120):
    c = [(h, t) for h, t in cl if t > 0]
    if len(c) < 2:
        return (np.nan, np.nan)
    h = np.array([a for a, _ in c], float)
    t = np.array([b for _, b in c], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    bs = h[idx].sum(1) / t[idx].sum(1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def summarize(rows, ref_rows=None, suites=("closed_loop", "emulator")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    df = pd.DataFrame(rows)
    if df.empty:
        return out
    for lv in LEVELS:
        d = df[(df.lv == lv) & df.suite.isin(suites)]
        inst = d.groupby(["suite", "task", "seed"]).agg(hits=("hits", "sum"), tot=("tot", "sum"),
                                                         ho_hits=("ho_hits", "sum"), ho_tot=("ho_tot", "sum"))
        cl = list(zip(inst.hits, inst.tot))
        pooled = inst.hits.sum() / max(inst.tot.sum(), 1)
        ho = inst.ho_hits.sum() / max(inst.ho_tot.sum(), 1)
        per = {}
        worst = []
        for (s, t), g in inst.groupby(level=[0, 1]):
            ci = cluster_ci(list(zip(g.hits, g.tot)))
            cov = g.hits.sum() / max(g.tot.sum(), 1)
            per[f"{s}|{t}"] = (round(cov, 3), tuple(round(v, 3) for v in ci), int(g.tot.sum()))
            if ci[1] < lv:
                worst.append(f"{s}|{t}")
        wl = [w for ws in d[d.suite == "closed_loop"].widths for w in ws]
        out[lv] = {"pooled": round(pooled, 4), "ci": tuple(round(v, 3) for v in cluster_ci(cl)),
                   "n": int(inst.tot.sum()), "heldout": round(ho, 4),
                   "median_width_cl": round(float(np.median(wl)), 3) if wl else None,
                   "tasks_ci_below": worst, "per_task": per}
    return out


def width_ratio(rows, ref_rows, lv=0.9, n_boot=1000, seed=20261120):
    """Median width at proposals (closed loop) candidate / reference, bootstrap over instances."""
    def by_inst(rs):
        m = defaultdict(list)
        for r in rs:
            if r["suite"] == "closed_loop" and r["lv"] == lv:
                m[(r["task"], r["seed"])].extend(r["widths"])
        return m
    a, b = by_inst(rows), by_inst(ref_rows)
    keys = sorted(set(a) & set(b))
    wa, wb = [a[k] for k in keys], [b[k] for k in keys]
    ratio = np.median(np.concatenate(wa)) / np.median(np.concatenate(wb))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(keys), len(keys))
        x = [w for k in i for w in wa[k]]
        y = [w for k in i for w in wb[k]]
        if x and y:
            bs.append(np.median(x) / np.median(y))
    return round(float(ratio), 3), (round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3))
