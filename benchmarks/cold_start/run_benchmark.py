"""Cold-start design benchmark: which starting design, and how many starting runs?

Question. A campaign starts with no data. Kalos plans the first round from a
space-filling design (`kalos.core.design`), then switches to model-guided rounds.
Which design (scrambled Sobol, LHS-maximin, random, BayOp-style LHS) and which
starting-round size (8, 12, 20 runs) leaves the campaign best off after the
model-guided rounds that follow, at the same total budget, and how soon does the
model reach the engine's own trust verdict?

Protocol (fixed before the first scored run; see README.md next to this file):

- Total budget N = 32 runs. Starting round n0 in {8, 12, 20}; then model-guided
  rounds of q = 4 until N (6, 5 and 3 rounds).
- Design arms: `sobol`, `lhs_maximin`, `random` (i.i.d. uniform; categoricals
  i.i.d. too), `lhs` (plain Latin hypercube: BayOp-style without its refinement)
  and `lhs_cd` (Latin hypercube refined for centered discrepancy: BayOp's actual
  default, `optimization="random-cd"`). Every arm except `random` goes through
  `kalos.core.design.generate_design` (snapping, constraints, balanced levels).
- Model-guided rounds, synthetic tasks: the production path
  (`Surrogate().fit` + `kalos.core.optimize.propose`, q = 4, seeded), the same
  for every arm, as in `kalos/bench/vs_baselines.py`'s Kalos arm.
  Pool replays (example sheets): the same surrogate scores every unrun pool row
  by expected improvement and runs the top 4 (pool BO, as `kalos/bench/pool.py`).
  The starting design is mapped to the nearest unused pool rows.
- Scores: normalized simple regret at N (synthetic: (f* - best TRUE value among
  the runs) / (f* - median f over the region); pool: the same on the measured
  values) and trust: the engine's own verdict (`kalos.portal.verdict.model_verdict`
  on `grouped_cv_report` with the engine's settings: 5 folds, 2 repeats, floor 0.20)
  after the starting round and after each model-guided round. "Reached trust" =
  verdict `reliable` (which needs >= 20 runs and >= 8 groups by construction).
- Seeds 1000..1019 (20) for every cell; 95% CIs by percentile bootstrap over
  seeds (10,000 resamples), and paired differences against `sobol` with the same
  seeds.

Run:  KALOS_TORCH_THREADS=1 .venv/bin/python benchmarks/cold_start/run_benchmark.py --workers 10
      (--quick: 2 seeds, two tasks, prints only)
The Kaggle replay needs KALOS_DATASET_DIR (the file is never committed; see
examples/datasets/README.md); without it that task is skipped and says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
SEEDS = tuple(range(1000, 1020))
N_TOTAL = 32
Q = 4
N0S = (8, 12, 20)
ARMS = ("sobol", "lhs_maximin", "random", "lhs")
# BayOp's own default (LHS refined for centered discrepancy) is scored on the
# starting design's quality only (`design_quality_only`), not run through the
# model-guided rounds: the full run was too slow on a shared machine.
QUALITY_ONLY_ARMS = ("lhs_cd",)
NOISE_H6 = 0.1


@dataclass(frozen=True)
class Synthetic:
    name: str
    box: np.ndarray  # (2, d) continuous box (categorical dims follow as codes)
    f: Callable[[np.ndarray], np.ndarray]
    f_star: float
    obs: str  # "gauss:<sd>" or "bio"
    cat_cards: tuple[int, ...] = ()
    steps: tuple[float | None, ...] | None = None
    constraint: tuple[tuple[int, ...], tuple[float, ...], float] | None = None  # sum c x <= rhs


def _tasks() -> dict[str, Synthetic]:
    from kalos.bench.coverage_at_proposals import _BIO_BOX, bioprocess, hartmann6
    from kalos.bench.objectives import bioprocess_like, gaussian_bump

    bl = bioprocess_like()
    gb = gaussian_bump(4)
    return {
        "hartmann6": Synthetic("hartmann6", np.array([[0.0] * 6, [1.0] * 6]), hartmann6, 3.32237,
                               f"gauss:{NOISE_H6}"),
        # The generator of examples/synthetic_bioprocess (10 settings, zero-inflated,
        # 12% multiplicative assay noise), with declared setpoint steps.
        "bioprocess_sheet_sim": Synthetic(
            "bioprocess_sheet_sim", _BIO_BOX, bioprocess, float("nan"), "bio",
            steps=(0.05, 0.5, 0.5, 0.5, 0.1, 5.0, 10.0, 0.1, 5.0, 5.0),
        ),
        "bioprocess_like_mixed": Synthetic(
            "bioprocess_like_mixed", bl.cont_bounds, bl._f, bl.optimum, f"gauss:{0.15 * bl.scale}",
            cat_cards=bl.cat_cardinalities,
        ),
        "gaussian4d_constrained": Synthetic(
            "gaussian4d_constrained", gb.bounds, gb._f, 10.0 * float(np.exp(-4.0 * 0.02)),
            f"gauss:{0.15 * gb.scale}", constraint=((0, 1), (1.0, 1.0), 1.0),
        ),
    }


# ----------------------------------------------------------------- helpers
def _feasible(task: Synthetic, x: np.ndarray) -> np.ndarray:
    if task.constraint is None:
        return np.ones(len(x), bool)
    idx, coef, rhs = task.constraint
    return (x[:, list(idx)] @ np.asarray(coef)) <= rhs + 1e-9


def _f_true(task: Synthetic, x: np.ndarray) -> np.ndarray:
    return np.asarray(task.f(x), float).reshape(-1)


def _observe(task: Synthetic, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = _f_true(task, x)
    if task.obs == "bio":
        return np.clip(np.where(f > 0, f * rng.normal(1.0, 0.12, len(f)), 0.0), 0, None)
    sd = float(task.obs.split(":")[1])
    return f + rng.normal(0, sd, len(f))


_REF: dict[str, tuple[float, float]] = {}


def _reference(task: Synthetic) -> tuple[float, float]:
    """(f_star, f_median) over the feasible region (2^15 Sobol / uniform codes)."""
    if task.name in _REF:
        return _REF[task.name]
    from scipy.stats import qmc

    d = task.box.shape[1]
    u = qmc.Sobol(d, scramble=True, seed=12345).random(1 << 15)
    x = task.box[0] + u * (task.box[1] - task.box[0])
    if task.cat_cards:
        r = np.random.default_rng(12345)
        x = np.hstack([x, np.column_stack([r.integers(0, k, len(x)) for k in task.cat_cards]).astype(float)])
    x = x[_feasible(task, x)]
    f = _f_true(task, x)
    f_star = task.f_star
    if not np.isfinite(f_star):
        # bioprocess simulator: optimum by dense search + local polish (scipy).
        from scipy.optimize import minimize

        best = x[np.argsort(-f)[:20]]
        vals = []
        for x0 in best:
            res = minimize(lambda z: -_f_true(task, z[None])[0], x0, method="Nelder-Mead",
                           options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9})
            z = np.clip(res.x, task.box[0], task.box[1])
            vals.append(_f_true(task, z[None])[0])
        f_star = float(max(max(vals), f.max()))
    _REF[task.name] = (f_star, float(np.median(f)))
    return _REF[task.name]


def _initial_design(task: Synthetic, arm: str, n0: int, seed: int) -> np.ndarray:
    from kalos.core.design import (
        CategoricalFactor,
        ContinuousFactor,
        LinearConstraint,
        generate_design,
    )

    d = task.box.shape[1]
    rng = np.random.default_rng([seed, n0, 31])
    if arm == "random":
        out = []
        while len(out) < n0:
            x = task.box[0] + rng.random(d) * (task.box[1] - task.box[0])
            if task.steps:
                x = np.array([
                    v if s is None else task.box[0, j] + np.round((v - task.box[0, j]) / s) * s
                    for j, (v, s) in enumerate(zip(x, task.steps))
                ])
                x = np.clip(x, task.box[0], task.box[1])
            if _feasible(task, x[None])[0]:
                out.append(x)
        X = np.array(out)
        if task.cat_cards:
            X = np.hstack([X, np.column_stack([rng.integers(0, k, n0) for k in task.cat_cards]).astype(float)])
        return X
    names = [f"x{j}" for j in range(d)]
    cont = [
        ContinuousFactor(names[j], float(task.box[0, j]), float(task.box[1, j]),
                         step=(task.steps[j] if task.steps else None))
        for j in range(d)
    ]
    cats = [CategoricalFactor(f"c{k}", tuple(str(i) for i in range(card)))
            for k, card in enumerate(task.cat_cards)]
    cons = []
    if task.constraint is not None:
        idx, coef, rhs = task.constraint
        cons.append(LinearConstraint({names[i]: c for i, c in zip(idx, coef)}, "<=", rhs))
    res = generate_design(cont, cats, cons, n=n0, method=arm, seed=seed)
    X = np.array([[r[nm] for nm in names] + [float(r[c.name]) for c in cats] for r in res.rows])
    return X


def _trust(X: np.ndarray, y: np.ndarray, bounds: np.ndarray, cat_dims: list[int] | None) -> str:
    from kalos.core.evaluation import grouped_cv_report
    from kalos.portal.analysis_parts.policy import (
        CV_N_REPEATS,
        RELIABILITY_SPEARMAN_FLOOR,
    )
    from kalos.portal.verdict import MIN_RUNS_FOR_RELIABLE, model_verdict

    if len(y) < MIN_RUNS_FOR_RELIABLE:
        return "not_yet_below_min_runs"
    keys = [tuple(np.round(r, 9)) for r in X]
    groups = [str(keys.index(k)) for k in keys]
    if np.ptp(y) <= 0:
        return "refused"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = grouped_cv_report(X, y, groups=groups, n_splits=5, bounds=bounds,
                                cat_dims=cat_dims, n_repeats=CV_N_REPEATS)
    rho = rep["spearman"]
    lo, hi = rep["ci95"]
    ci = None if (rho != rho or lo != lo or hi != hi) else [lo, hi]
    v = model_verdict(n_runs=len(y), n_groups=int(rep["n_groups"]),
                      spearman=None if rho != rho else float(rho), ci95=ci,
                      floor=RELIABILITY_SPEARMAN_FLOOR)
    return str(v["state"])


def _unit_quality(X: np.ndarray, box: np.ndarray) -> dict[str, float]:
    from scipy.spatial.distance import pdist
    from scipy.stats import qmc

    u = np.clip((X - box[0]) / (box[1] - box[0]), 0, 1)
    return {"min_dist": float(pdist(u).min()), "cd": float(qmc.discrepancy(u, method="CD"))}


# ----------------------------------------------------------------- synthetic unit
def run_synthetic(task_name: str, arm: str, n0: int, seed: int) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    from kalos.core.optimize import propose
    from kalos.core.surrogate import FitError, Surrogate

    task = _tasks()[task_name]
    f_star, f_med = _reference(task)
    rng_obs = np.random.default_rng([seed, 99])
    d = task.box.shape[1]
    ncat = len(task.cat_cards)
    full_box = task.box if not ncat else np.hstack([task.box, np.array([[0.0] * ncat, [k - 1.0 for k in task.cat_cards]])])
    cat_dims = list(range(d, d + ncat)) or None
    lin = None
    if task.constraint is not None:
        idx, coef, rhs = task.constraint
        lin = [(list(idx), [-c for c in coef], -rhs)]  # BoTorch form: sum >= rhs
    t0 = time.time()
    X = _initial_design(task, arm, n0, seed)
    q0 = _unit_quality(X[:, :d], task.box)
    y = _observe(task, X, rng_obs)
    curve = [float(_f_true(task, X).max())]
    ns = [len(y)]
    trust = [_trust(X, y, full_box, cat_dims)]
    rnd = 0
    while len(y) < N_TOTAL:
        rnd += 1
        try:
            s = Surrogate().fit(X, y, bounds=full_box, cat_dims=cat_dims)
            P = propose(s, full_box, q=Q, cat_dims=cat_dims,
                        cat_cardinalities=list(task.cat_cards) or None,
                        seed=seed * 100 + rnd, linear_constraints=lin)
            P = np.asarray(P, float).reshape(-1, full_box.shape[1])
        except FitError:
            P = np.empty((0, full_box.shape[1]))
        if len(P) < Q:  # short batch: top up uniformly (counted)
            r = np.random.default_rng([seed, rnd, 5])
            extra = []
            while len(P) + len(extra) < Q:
                x = full_box[0] + r.random(full_box.shape[1]) * (full_box[1] - full_box[0])
                if ncat:
                    x[d:] = np.rint(x[d:])
                if _feasible(task, x[None])[0]:
                    extra.append(x)
            P = np.vstack([P, np.array(extra)]) if len(P) else np.array(extra)
        P = P[: N_TOTAL - len(y)]
        X = np.vstack([X, P])
        y = np.concatenate([y, _observe(task, P, rng_obs)])
        curve.append(float(max(curve[-1], _f_true(task, P).max())))
        ns.append(len(y))
        trust.append(_trust(X, y, full_box, cat_dims))
    regret = [(f_star - c) / (f_star - f_med) for c in curve]
    first = next((n for n, t in zip(ns, trust) if t == "reliable"), None)
    return {"task": task_name, "arm": arm, "n0": n0, "seed": seed, "ns": ns, "regret": regret,
            "trust": trust, "trust_first_n": first, "init_quality": q0, "seconds": time.time() - t0}


# ----------------------------------------------------------------- pool replay unit
def _load_pool(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    import pandas as pd

    from kalos.bench.pool import pool_from_frame

    if name == "pool_synthetic_bioprocess":
        df = pd.read_csv(ROOT / "examples/synthetic_bioprocess/synthetic_bioprocess.csv")
        X, y, _ = pool_from_frame(df, "titer_g_L")
    elif name == "pool_synthetic_scaleup":
        df = pd.read_csv(ROOT / "examples/synthetic_scaleup/synthetic_scaleup.csv")
        X, y, _ = pool_from_frame(df, "titer_g_per_L")
    elif name == "pool_kaggle_fermentation":
        base = os.environ.get("KALOS_DATASET_DIR")
        path = Path(base) / "kaggle_fermentation_dataset_0.csv" if base else None
        if path is None or not path.exists():
            return None
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in ("yield_tier", "contamination", "data_type") if c in df.columns])
        X, y, _ = pool_from_frame(df, "max_yield_fpu_ml")
    else:
        raise ValueError(name)
    return np.asarray(X, float), np.asarray(y, float)


def run_pool(task_name: str, arm: str, n0: int, seed: int) -> dict[str, Any]:
    import torch
    from scipy.spatial.distance import cdist
    from scipy.stats import norm

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    from kalos.core.surrogate import FitError, Surrogate

    loaded = _load_pool(task_name)
    assert loaded is not None
    Xp, yp = loaded
    lo, hi = Xp.min(0), Xp.max(0)
    U = (Xp - lo) / np.where(hi > lo, hi - lo, 1.0)
    box = np.vstack([np.zeros(U.shape[1]), np.ones(U.shape[1])])
    t0 = time.time()
    task = Synthetic(task_name, box, lambda x: x[:, 0], 1.0, "gauss:0")
    D = _initial_design(task, arm, n0, seed)
    chosen: list[int] = []
    free = np.ones(len(U), bool)
    for x in D:  # nearest unused pool row
        dist = cdist(x[None], U)[0]
        dist[~free] = np.inf
        i = int(np.argmin(dist))
        chosen.append(i)
        free[i] = False
    y_best, y_med = float(yp.max()), float(np.median(yp))
    q0 = _unit_quality(U[chosen], box)
    curve = [float(yp[chosen].max())]
    ns = [len(chosen)]
    trust = [_trust(U[chosen], yp[chosen], box, None)]
    while len(chosen) < N_TOTAL:
        idx = np.array(chosen)
        rem = np.flatnonzero(free)
        try:
            s = Surrogate().fit(U[idx], yp[idx], bounds=box)
            mu, sd = s.posterior(U[rem])
            mu = np.asarray(mu, float).reshape(-1)
            sd = np.maximum(np.asarray(sd, float).reshape(-1), 1e-9)
            z = (mu - yp[idx].max()) / sd
            ei = (mu - yp[idx].max()) * norm.cdf(z) + sd * norm.pdf(z)
            picks = rem[np.argsort(-ei, kind="stable")[: min(Q, N_TOTAL - len(chosen))]]
        except FitError:
            r = np.random.default_rng([seed, len(chosen)])
            picks = r.choice(rem, size=min(Q, N_TOTAL - len(chosen)), replace=False)
        for i in picks:
            chosen.append(int(i))
            free[int(i)] = False
        curve.append(float(max(curve[-1], yp[picks].max())))
        ns.append(len(chosen))
        trust.append(_trust(U[chosen], yp[chosen], box, None))
    regret = [(y_best - c) / (y_best - y_med) if y_best > y_med else 0.0 for c in curve]
    first = next((n for n, t in zip(ns, trust) if t == "reliable"), None)
    return {"task": task_name, "arm": arm, "n0": n0, "seed": seed, "ns": ns, "regret": regret,
            "trust": trust, "trust_first_n": first, "init_quality": q0, "seconds": time.time() - t0}


def _quality_only(task: str, arm: str, n0: int, seed: int) -> dict[str, Any]:
    """Starting-design quality on the task's continuous box (no model, cheap)."""
    if task.startswith("pool_"):
        loaded = _load_pool(task)
        assert loaded is not None
        d = loaded[0].shape[1]
        syn = Synthetic(task, np.vstack([np.zeros(d), np.ones(d)]), lambda x: x[:, 0], 1.0, "gauss:0")
    else:
        syn = _tasks()[task]
    X = _initial_design(syn, arm, n0, seed)
    return {"task": task, "arm": arm, "n0": n0, "seed": seed,
            **_unit_quality(X[:, : syn.box.shape[1]], syn.box)}


def _summarise_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for n0 in N0S:
        for arm in (*ARMS, *QUALITY_ONLY_ARMS):
            sel = [r for r in rows if r["n0"] == n0 and r["arm"] == arm]
            if sel:
                out.append({"n0": n0, "arm": arm, "n": len(sel),
                            "mean_min_dist": float(np.mean([r["min_dist"] for r in sel])),
                            "mean_cd": float(np.mean([r["cd"] for r in sel]))})
    return out


def _unit(args: tuple[str, str, int, int]) -> dict[str, Any]:
    os.environ.setdefault("KALOS_TORCH_THREADS", "1")
    task, arm, n0, seed = args
    warnings.simplefilter("ignore")
    try:
        if task.startswith("pool_"):
            return run_pool(task, arm, n0, seed)
        return run_synthetic(task, arm, n0, seed)
    except Exception as exc:  # recorded, never silently dropped
        return {"task": task, "arm": arm, "n0": n0, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}


# ----------------------------------------------------------------- analysis
def _boot_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(units: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(20261120)
    ok = [u for u in units if "error" not in u]
    tasks = sorted({u["task"] for u in ok})
    out: dict[str, Any] = {"cells": [], "paired_vs_sobol": [], "errors": [u for u in units if "error" in u]}
    for t in tasks:
        for n0 in N0S:
            by_arm = {a: {u["seed"]: u for u in ok if u["task"] == t and u["n0"] == n0 and u["arm"] == a}
                      for a in ARMS}
            for a in ARMS:
                us = list(by_arm[a].values())
                if not us:
                    continue
                final = np.array([u["regret"][-1] for u in us])
                init = np.array([u["regret"][0] for u in us])
                reached = np.array([u["trust_first_n"] is not None for u in us], float)
                firsts = [u["trust_first_n"] for u in us if u["trust_first_n"] is not None]
                out["cells"].append({
                    "task": t, "n0": n0, "arm": a, "n_seeds": len(us),
                    "final_regret": _boot_ci(final, rng),
                    "regret_after_start": _boot_ci(init, rng),
                    "trust_reached_rate": _boot_ci(reached, rng),
                    "median_n_at_trust": float(np.median(firsts)) if firsts else None,
                    "init_min_dist": float(np.mean([u["init_quality"]["min_dist"] for u in us])),
                    "init_cd": float(np.mean([u["init_quality"]["cd"] for u in us])),
                    "mean_seconds": float(np.mean([u["seconds"] for u in us])),
                })
                if a != "sobol":
                    common = sorted(set(by_arm[a]) & set(by_arm["sobol"]))
                    diff = np.array([by_arm[a][s]["regret"][-1] - by_arm["sobol"][s]["regret"][-1] for s in common])
                    out["paired_vs_sobol"].append({"task": t, "n0": n0, "arm": a, "n": len(common),
                                                   "final_regret_minus_sobol": _boot_ci(diff, rng)})
    # pooled over tasks (each task weighted equally): mean final regret per (arm, n0)
    pooled = []
    for n0 in N0S:
        for a in ARMS:
            per_seed = []
            for s in SEEDS:
                vals = [u["regret"][-1] for u in ok if u["n0"] == n0 and u["arm"] == a and u["seed"] == s]
                if len(vals) == len(tasks):
                    per_seed.append(float(np.mean(vals)))
            reach = [u["trust_first_n"] is not None for u in ok if u["n0"] == n0 and u["arm"] == a]
            pooled.append({"n0": n0, "arm": a, "n_seeds": len(per_seed),
                           "mean_final_regret_over_tasks": _boot_ci(np.array(per_seed), rng),
                           "trust_reached_rate": float(np.mean(reach)) if reach else None})
    out["pooled"] = pooled
    return out


def _fmt(t: Any, digits: int = 3) -> str:
    if t is None:
        return "n/a"
    m, lo, hi = t
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def render(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = ["# Cold-start design benchmark: results", "",
             f"Seeds {meta['seeds'][0]}..{meta['seeds'][-1]} ({len(meta['seeds'])}), total budget N={N_TOTAL}, "
             f"q={Q}. Means with 95% bootstrap CIs over seeds. Regret is normalized "
             "((best possible - best found) / (best possible - median)); lower is better. "
             "Trust = the engine's `reliable` verdict on grouped CV.", ""]
    lines += ["## Pooled over tasks (each task weighted equally)", "",
              "| n0 | arm | mean final regret [95% CI] | trust reached by N=32 |", "|---|---|---|---|"]
    for p in summary["pooled"]:
        lines.append(f"| {p['n0']} | {p['arm']} | {_fmt(p['mean_final_regret_over_tasks'])} | "
                     f"{p['trust_reached_rate']:.2f} |" if p["trust_reached_rate"] is not None else
                     f"| {p['n0']} | {p['arm']} | {_fmt(p['mean_final_regret_over_tasks'])} | n/a |")
    lines += ["", "## Per task", "",
              "| task | n0 | arm | final regret | regret after start | trust reached | median n at trust | start min-dist | start CD |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in summary["cells"]:
        lines.append(
            f"| {c['task']} | {c['n0']} | {c['arm']} | {_fmt(c['final_regret'])} | {_fmt(c['regret_after_start'])} | "
            f"{_fmt(c['trust_reached_rate'], 2)} | {c['median_n_at_trust'] if c['median_n_at_trust'] is not None else 'n/a'} | "
            f"{c['init_min_dist']:.3f} | {c['init_cd']:.3f} |")
    lines += ["", "## Paired difference in final regret vs `sobol` (same seeds; negative = better than sobol)", "",
              "| task | n0 | arm | diff [95% CI] |", "|---|---|---|---|"]
    for p in summary["paired_vs_sobol"]:
        lines.append(f"| {p['task']} | {p['n0']} | {p['arm']} | {_fmt(p['final_regret_minus_sobol'])} |")
    lines += ["", "## Starting-design quality only (no model rounds; all tasks pooled, unit box)", "",
              "| n0 | arm | mean min pairwise distance | mean centered L2 discrepancy |", "|---|---|---|---|"]
    for q in summary.get("design_quality_only", []):
        lines.append(f"| {q['n0']} | {q['arm']} | {q['mean_min_dist']:.3f} | {q['mean_cd']:.3f} |")
    if summary["errors"]:
        lines += ["", f"## Errors ({len(summary['errors'])})", ""]
        lines += [f"- {e['task']} {e['arm']} n0={e['n0']} seed={e['seed']}: {e['error']}" for e in summary["errors"]]
    if meta.get("skipped"):
        lines += ["", "## Skipped", ""] + [f"- {s}" for s in meta["skipped"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--journal", default=str(OUT / "units.jsonl"),
                    help="append each finished unit here; units already in it are not rerun")
    args = ap.parse_args(argv)
    seeds = SEEDS[:2] if args.quick else SEEDS
    tasks = list(_tasks()) + ["pool_synthetic_bioprocess", "pool_synthetic_scaleup", "pool_kaggle_fermentation"]
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks.split(",")]
    if args.quick:
        tasks = tasks[:1] + [t for t in tasks if t.startswith("pool_")][:1]
    skipped = []
    if "pool_kaggle_fermentation" in tasks and _load_pool("pool_kaggle_fermentation") is None:
        tasks.remove("pool_kaggle_fermentation")
        skipped.append("pool_kaggle_fermentation: KALOS_DATASET_DIR not set (file is not committed)")
    for t in tasks:  # warm the reference cache once per task in the parent (and print it)
        if not t.startswith("pool_"):
            print(t, "f*, f_med =", _reference(_tasks()[t]), flush=True)
    jobs = [(t, a, n0, s) for t in tasks for n0 in N0S for a in ARMS for s in seeds]
    # Cheapest tasks first (pool replays, then 4-6 D), the 10-D simulator last, so
    # a partial run already covers most tasks; seeds in order within a task.
    cost = {"pool_": 0, "gaussian4d_constrained": 1, "bioprocess_like_mixed": 2, "hartmann6": 3,
            "bioprocess_sheet_sim": 4}
    jobs.sort(key=lambda j: (next((v for k, v in cost.items() if j[0].startswith(k)), 5), j[3], j[0], j[2], j[1]))
    journal = Path(args.journal)
    done: dict[tuple, dict[str, Any]] = {}
    if journal.exists() and not args.quick:
        for line in journal.read_text().splitlines():
            u = json.loads(line)
            if "error" not in u:
                done[(u["task"], u["arm"], u["n0"], u["seed"])] = u
    todo = [j for j in jobs if j not in done]
    print(f"{len(done)} units from the journal, {len(todo)} to run", flush=True)
    t0 = time.time()
    units: list[dict[str, Any]] = [done[j] for j in jobs if j in done]
    if not args.quick:
        journal.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, u in enumerate(ex.map(_unit, todo, chunksize=1), 1):
            units.append(u)
            if not args.quick:
                with journal.open("a") as fh:
                    fh.write(json.dumps(u, default=float) + "\n")
            if k % 25 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} units, {time.time() - t0:.0f} s", flush=True)
    quality_only = [_quality_only(t, a, n0, s) for t in tasks for n0 in N0S
                    for a in (*ARMS, *QUALITY_ONLY_ARMS) for s in seeds]
    meta = {"seeds": list(seeds), "n_total": N_TOTAL, "q": Q, "n0s": list(N0S), "arms": list(ARMS),
            "tasks": tasks, "skipped": skipped, "wall_seconds": time.time() - t0,
            "references": {t: _reference(_tasks()[t]) for t in tasks if not t.startswith("pool_")}}
    summary = summarise(units)
    summary["design_quality_only"] = _summarise_quality(quality_only)
    md = render(summary, meta)
    print(md)
    if not args.quick:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps({"meta": meta, **summary}, indent=1, default=float))
        (OUT / "RESULTS.md").write_text(md)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main(sys.argv[1:]))
