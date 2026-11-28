"""Head-to-head benchmark harness: Baseline vs Frontier vs LHS across 5 arenas.

    python -m kalos.bench.frontier_race --seeds 5 --quick
    python -m kalos.bench.frontier_race --seeds 10
    python -m kalos.bench.frontier_race --seeds 5 --quick --json

Imports ``kalos.core.frontier.*``, ``kalos.core.surrogate``, ``kalos.core.optimize``,
``kalos.portal.analysis_parts.intervals``, and ``torch``/BoTorch lazily inside arena
functions so module import stays lightweight.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import qmc, spearmanr


def _lhs(n: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    d = int(bounds.shape[1])
    u = qmc.LatinHypercube(d=d, seed=int(seed)).random(n)
    return bounds[0] + u * (bounds[1] - bounds[0])


def _set_torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


# ---------------------------------------------------------------- Arena 1 --
def _arena1_truth(X: np.ndarray) -> np.ndarray:
    """25-D sparse media: 4 active Hill / Gaussian drivers + 21 inert dims.

    Optimum on [0, 1]^25 is achieved at x0=1, x1=1, x2=0.7, x3=0.3:
    10 * (1 / 1.0625) * (1 / 1.2) * 1 * 1 = 7.8431372549...
    """
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    x3 = X[:, 3]
    hill0 = (x0 ** 2) / (0.25 ** 2 + x0 ** 2)
    mm1 = x1 / (0.20 + x1)
    g2 = np.exp(-4.0 * (x2 - 0.70) ** 2)
    g3 = np.exp(-4.0 * (x3 - 0.30) ** 2)
    return 10.0 * hill0 * mm1 * g2 * g3


def run_arena1_high_d_sparse(*, seeds: int, quick: bool) -> dict[str, Any]:
    """Arena 1: High-D Sparse Media (d=25, 4 active drivers, n_init=12, budget=18)."""
    from kalos.core.frontier.acquisition import AdaptiveTuRBO, TrustRegionState
    from kalos.core.frontier.warped_sparse_gp import WarpedSparseGP
    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    _set_torch_threads()
    d = 25
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    optimum = float(10.0 * (1.0 / (0.25 ** 2 + 1.0)) * (1.0 / 1.20))
    scale = optimum
    n_init = 12
    q = 3
    rounds = 4 if quick else 6  # 12 in quick, 18 in full
    noise_sd = 0.10 * scale

    regrets: dict[str, list[float]] = {"baseline": [], "frontier": [], "lhs": []}
    times: dict[str, list[float]] = {"baseline": [], "frontier": [], "lhs": []}

    for s in range(seeds):
        seed = 100 + s
        X_init = _lhs(n_init, bounds, seed=seed)
        rng_init = np.random.default_rng([seed, 1, 0])
        y_init_true = _arena1_truth(X_init)
        y_init_obs = y_init_true + rng_init.normal(0.0, noise_sd, size=n_init)

        # Baseline: Surrogate + propose
        t0 = time.perf_counter()
        X_b, y_b_obs, y_b_true = X_init.copy(), y_init_obs.copy(), y_init_true.copy()
        rng_b = np.random.default_rng([seed, 1, 1])
        for r in range(rounds):
            sur = Surrogate().fit(X_b, y_b_obs, bounds=bounds)
            X_next = propose(sur, bounds, q=q, raw_samples=128 if quick else 256, seed=seed * 100 + r)
            y_nt = _arena1_truth(X_next)
            y_no = y_nt + rng_b.normal(0.0, noise_sd, size=len(X_next))
            X_b = np.vstack([X_b, X_next])
            y_b_obs = np.concatenate([y_b_obs, y_no])
            y_b_true = np.concatenate([y_b_true, y_nt])
        times["baseline"].append(time.perf_counter() - t0)
        regrets["baseline"].append(optimum - float(np.max(y_b_true)))

        # Frontier: WarpedSparseGP + AdaptiveTuRBO
        t0 = time.perf_counter()
        X_f, y_f_obs, y_f_true = X_init.copy(), y_init_obs.copy(), y_init_true.copy()
        rng_f = np.random.default_rng([seed, 1, 1])
        tr = TrustRegionState(bounds, best_value=float(np.max(y_f_obs)))
        turbo = AdaptiveTuRBO(n_candidates=256 if quick else 512)
        for r in range(rounds):
            gp = WarpedSparseGP(sparsity_strength=0.15).fit(X_f, y_f_obs, bounds, seed=seed * 100 + r)
            X_next = turbo.propose(
                gp, bounds, q=q, X_train=X_f, y_train=y_f_obs, tr_state=tr, seed=seed * 100 + r
            )
            y_nt = _arena1_truth(X_next)
            y_no = y_nt + rng_f.normal(0.0, noise_sd, size=len(X_next))
            X_f = np.vstack([X_f, X_next])
            y_f_obs = np.concatenate([y_f_obs, y_no])
            y_f_true = np.concatenate([y_f_true, y_nt])
            tr.update(float(np.max(y_f_obs)))
        times["frontier"].append(time.perf_counter() - t0)
        regrets["frontier"].append(optimum - float(np.max(y_f_true)))

        # LHS
        t0 = time.perf_counter()
        X_l = np.vstack([X_init, _lhs(rounds * q, bounds, seed=seed * 1000 + 7)])
        y_l_true = _arena1_truth(X_l)
        times["lhs"].append(time.perf_counter() - t0)
        regrets["lhs"].append(optimum - float(np.max(y_l_true)))

    return _summarize_regret_arena("arena1_high_d_sparse", optimum, regrets, times, seeds)


# ---------------------------------------------------------------- Arena 2 --
def _arena2_truth(X: np.ndarray) -> np.ndarray:
    """6-D positive bioprocess peak in [0, 1]^6. Max = 10.5 at c."""
    c = np.array([0.65, 0.35, 0.70, 0.30, 0.55, 0.45])
    d2 = ((X - c) ** 2).sum(axis=1)
    return 0.5 + 10.0 * np.exp(-3.5 * d2)


def _arena2_observe(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative log-normal noise (CV=25%) + 10% +3.0-sigma outlier spikes."""
    sigma_ln = math.sqrt(math.log(1.0 + 0.25 ** 2))
    eps = rng.normal(0.0, sigma_ln, size=len(y_true))
    y_obs = y_true * np.exp(eps - 0.5 * sigma_ln ** 2)
    outliers = rng.random(len(y_true)) < 0.10
    y_obs = np.where(outliers, y_obs + 3.0 * 0.25 * y_true, y_obs)
    return y_obs


def run_arena2_heteroscedastic_outliers(*, seeds: int, quick: bool) -> dict[str, Any]:
    """Arena 2: Heteroscedastic Log-Normal (CV=25%) + 10% +3sigma Outliers."""
    from kalos.core.frontier.acquisition import AdaptiveTuRBO, TrustRegionState
    from kalos.core.frontier.warped_sparse_gp import WarpedSparseGP
    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    _set_torch_threads()
    d = 6
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    optimum = 10.5
    n_init = 12
    q = 3
    rounds = 4 if quick else 5  # 12 in quick, 15 in full

    regrets: dict[str, list[float]] = {"baseline": [], "frontier": [], "lhs": []}
    times: dict[str, list[float]] = {"baseline": [], "frontier": [], "lhs": []}

    for s in range(seeds):
        seed = 200 + s
        X_init = _lhs(n_init, bounds, seed=seed)
        rng_init = np.random.default_rng([seed, 2, 0])
        y_init_true = _arena2_truth(X_init)
        y_init_obs = _arena2_observe(y_init_true, rng_init)

        # Baseline
        t0 = time.perf_counter()
        X_b, y_b_obs, y_b_true = X_init.copy(), y_init_obs.copy(), y_init_true.copy()
        rng_b = np.random.default_rng([seed, 2, 1])
        for r in range(rounds):
            sur = Surrogate().fit(X_b, y_b_obs, bounds=bounds)
            X_next = propose(sur, bounds, q=q, raw_samples=128 if quick else 256, seed=seed * 100 + r)
            y_nt = _arena2_truth(X_next)
            y_no = _arena2_observe(y_nt, rng_b)
            X_b = np.vstack([X_b, X_next])
            y_b_obs = np.concatenate([y_b_obs, y_no])
            y_b_true = np.concatenate([y_b_true, y_nt])
        times["baseline"].append(time.perf_counter() - t0)
        regrets["baseline"].append(optimum - float(np.max(y_b_true)))

        # Frontier
        t0 = time.perf_counter()
        X_f, y_f_obs, y_f_true = X_init.copy(), y_init_obs.copy(), y_init_true.copy()
        rng_f = np.random.default_rng([seed, 2, 1])
        tr = TrustRegionState(bounds, best_value=float(np.max(y_f_obs)))
        turbo = AdaptiveTuRBO(n_candidates=256 if quick else 512)
        for r in range(rounds):
            gp = WarpedSparseGP(use_output_warp=True, robust=True).fit(
                X_f, y_f_obs, bounds, seed=seed * 100 + r
            )
            X_next = turbo.propose(
                gp, bounds, q=q, X_train=X_f, y_train=y_f_obs, tr_state=tr, seed=seed * 100 + r
            )
            y_nt = _arena2_truth(X_next)
            y_no = _arena2_observe(y_nt, rng_f)
            X_f = np.vstack([X_f, X_next])
            y_f_obs = np.concatenate([y_f_obs, y_no])
            y_f_true = np.concatenate([y_f_true, y_nt])
            tr.update(float(np.max(y_f_obs)))
        times["frontier"].append(time.perf_counter() - t0)
        regrets["frontier"].append(optimum - float(np.max(y_f_true)))

        # LHS
        t0 = time.perf_counter()
        X_l = np.vstack([X_init, _lhs(rounds * q, bounds, seed=seed * 1000 + 7)])
        y_l_true = _arena2_truth(X_l)
        times["lhs"].append(time.perf_counter() - t0)
        regrets["lhs"].append(optimum - float(np.max(y_l_true)))

    return _summarize_regret_arena("arena2_heteroscedastic_outliers", optimum, regrets, times, seeds)


# ---------------------------------------------------------------- Arena 3 --
_ARENA3_LOD = 0.5


def _arena3_truth(X: np.ndarray) -> np.ndarray:
    """5-D surface with ~30% dead/below-LOD region bordering a sharp cliff optimum.

    Dead zone: x0 < 0.30 (30% of [0, 1]^5 volume), where truth is 0.0 (< LOD).
    Peak sits right beside the cliff at x0=0.38, x1=0.65, x2=0.50, x3=0.50, x4=0.50 (max=10.0).
    """
    viable = X[:, 0] >= 0.30
    c = np.array([0.38, 0.65, 0.50, 0.50, 0.50])
    d2 = ((X - c) ** 2).sum(axis=1)
    titer = 10.0 * np.exp(-6.0 * d2)
    return np.where(viable, titer, 0.0)


def _arena3_observe(y_true: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y_raw = y_true + rng.normal(0.0, 0.35, size=len(y_true))
    censored = y_raw <= _ARENA3_LOD
    y_obs = np.where(censored, _ARENA3_LOD, y_raw)
    return y_obs, censored


def run_arena3_censored_cliff(*, seeds: int, quick: bool) -> dict[str, Any]:
    """Arena 3: Censored < LOD Cliff (d=5, 30% dead region, n_init=15, budget=15)."""
    from kalos.core.frontier.acquisition import AdaptiveTuRBO, TrustRegionState
    from kalos.core.frontier.warped_sparse_gp import WarpedSparseGP
    from kalos.core.optimize import propose
    from kalos.core.surrogate import Surrogate

    _set_torch_threads()
    d = 5
    bounds = np.vstack([np.zeros(d), np.ones(d)])
    optimum = 10.0
    n_init = 15
    q = 3
    rounds = 4 if quick else 5  # 12 in quick, 15 in full

    regrets: dict[str, list[float]] = {"baseline": [], "frontier": [], "lhs": []}
    times: dict[str, list[float]] = {"baseline": [], "frontier": [], "lhs": []}

    for s in range(seeds):
        seed = 300 + s
        X_init = _lhs(n_init, bounds, seed=seed)
        rng_init = np.random.default_rng([seed, 3, 0])
        y_init_true = _arena3_truth(X_init)
        y_init_obs, cens_init = _arena3_observe(y_init_true, rng_init)

        # Baseline: drops < LOD rows
        t0 = time.perf_counter()
        X_b, y_b_obs, c_b, y_b_true = (
            X_init.copy(),
            y_init_obs.copy(),
            cens_init.copy(),
            y_init_true.copy(),
        )
        rng_b = np.random.default_rng([seed, 3, 1])
        for r in range(rounds):
            unc = ~c_b
            X_fit = X_b[unc] if int(np.sum(unc)) >= 4 else X_b
            y_fit = y_b_obs[unc] if int(np.sum(unc)) >= 4 else y_b_obs
            sur = Surrogate().fit(X_fit, y_fit, bounds=bounds)
            X_next = propose(sur, bounds, q=q, raw_samples=128 if quick else 256, seed=seed * 100 + r)
            y_nt = _arena3_truth(X_next)
            y_no, c_n = _arena3_observe(y_nt, rng_b)
            X_b = np.vstack([X_b, X_next])
            y_b_obs = np.concatenate([y_b_obs, y_no])
            c_b = np.concatenate([c_b, c_n])
            y_b_true = np.concatenate([y_b_true, y_nt])
        times["baseline"].append(time.perf_counter() - t0)
        regrets["baseline"].append(optimum - float(np.max(y_b_true)))

        # Frontier: WarpedSparseGP with censored_mask & lod + AdaptiveTuRBO
        t0 = time.perf_counter()
        X_f, y_f_obs, c_f, y_f_true = (
            X_init.copy(),
            y_init_obs.copy(),
            cens_init.copy(),
            y_init_true.copy(),
        )
        rng_f = np.random.default_rng([seed, 3, 1])
        tr = TrustRegionState(bounds, best_value=float(np.max(y_f_obs)))
        turbo = AdaptiveTuRBO(n_candidates=256 if quick else 512)
        for r in range(rounds):
            gp = WarpedSparseGP().fit(
                X_f, y_f_obs, bounds, censored_mask=c_f, lod=_ARENA3_LOD, seed=seed * 100 + r
            )
            X_next = turbo.propose(
                gp, bounds, q=q, X_train=X_f, y_train=y_f_obs, tr_state=tr, seed=seed * 100 + r
            )
            y_nt = _arena3_truth(X_next)
            y_no, c_n = _arena3_observe(y_nt, rng_f)
            X_f = np.vstack([X_f, X_next])
            y_f_obs = np.concatenate([y_f_obs, y_no])
            c_f = np.concatenate([c_f, c_n])
            y_f_true = np.concatenate([y_f_true, y_nt])
            tr.update(float(np.max(y_f_obs)))
        times["frontier"].append(time.perf_counter() - t0)
        regrets["frontier"].append(optimum - float(np.max(y_f_true)))

        # LHS
        t0 = time.perf_counter()
        X_l = np.vstack([X_init, _lhs(rounds * q, bounds, seed=seed * 1000 + 7)])
        y_l_true = _arena3_truth(X_l)
        times["lhs"].append(time.perf_counter() - t0)
        regrets["lhs"].append(optimum - float(np.max(y_l_true)))

    return _summarize_regret_arena("arena3_censored_lod_cliff", optimum, regrets, times, seeds)


# ---------------------------------------------------------------- Arena 4 --
def _arena4_truth(X: np.ndarray) -> np.ndarray:
    """5-D smooth process surface f(x) in ~[1, 9]."""
    return (
        3.0 * np.sin(np.pi * X[:, 0]) * np.cos(0.5 * np.pi * X[:, 1])
        + 2.5 * np.exp(-3.0 * ((X[:, 2] - 0.6) ** 2 + (X[:, 3] - 0.4) ** 2))
        + 1.5 * X[:, 4]
        + 4.0
    )


def run_arena4_batch_drift(*, seeds: int, quick: bool) -> dict[str, Any]:
    """Arena 4: Multi-Round Batch Drift (d=5, 5 batches, tau_batch=2.0, n=50)."""
    from kalos.core.frontier.hierarchical_gp import HierarchicalBatchGP
    from kalos.core.surrogate import Surrogate

    _set_torch_threads()
    d = 5
    n_batches = 5
    per_batch = 10
    n = n_batches * per_batch
    tau_batch = 2.0
    assay_sd = 0.2
    bounds = np.vstack([np.zeros(d), np.ones(d)])

    metrics: dict[str, dict[str, list[float]]] = {
        "baseline": {"spearman_latent": [], "rmse_latent": [], "spearman_obs": [], "time": []},
        "frontier": {"spearman_latent": [], "rmse_latent": [], "spearman_obs": [], "time": []},
    }

    for s in range(seeds):
        seed = 400 + s
        rng = np.random.default_rng(seed)
        X = _lhs(n, bounds, seed=seed)
        batches = np.repeat([f"batch_{k}" for k in range(n_batches)], per_batch)
        batch_offsets = rng.normal(0.0, tau_batch, size=n_batches)
        batch_idx = np.repeat(np.arange(n_batches), per_batch)
        f_true = _arena4_truth(X)
        y_obs = f_true + batch_offsets[batch_idx] + rng.normal(0.0, assay_sd, size=n)

        # Leave-one-batch-out CV across the 5 unseen batches
        for arm in ("baseline", "frontier"):
            t0 = time.perf_counter()
            oof_pred = np.empty(n, dtype=float)
            for k in range(n_batches):
                te = batch_idx == k
                tr = ~te
                if arm == "baseline":
                    sur = Surrogate().fit(X[tr], y_obs[tr], bounds=bounds)
                    mu, _ = sur.posterior(X[te])
                else:
                    hgp = HierarchicalBatchGP().fit(
                        X[tr],
                        y_obs[tr],
                        batches[tr],
                        bounds,
                        seed=seed * 10 + k,
                        n_restarts=2 if quick else 5,
                    )
                    mu, _ = hgp.posterior(X[te], group=None)
                oof_pred[te] = np.asarray(mu, dtype=float).reshape(-1)
            elapsed = time.perf_counter() - t0
            metrics[arm]["time"].append(elapsed)
            metrics[arm]["spearman_latent"].append(float(spearmanr(oof_pred, f_true).statistic))
            metrics[arm]["rmse_latent"].append(float(np.sqrt(np.mean((oof_pred - f_true) ** 2))))
            metrics[arm]["spearman_obs"].append(float(spearmanr(oof_pred, y_obs).statistic))

    summary: dict[str, Any] = {"arena": "arena4_batch_drift", "n_seeds": seeds}
    for arm, d_m in metrics.items():
        summary[arm] = {
            "oof_spearman_latent": round(float(np.mean(d_m["spearman_latent"])), 4),
            "oof_rmse_latent": round(float(np.mean(d_m["rmse_latent"])), 4),
            "oof_spearman_obs": round(float(np.mean(d_m["spearman_obs"])), 4),
            "mean_seconds": round(float(np.mean(d_m["time"])), 3),
        }
    return summary


# ---------------------------------------------------------------- Arena 5 --
def run_arena5_proposal_intervals(*, seeds: int, quick: bool) -> dict[str, Any]:
    """Arena 5: Proposal Interval Coverage vs Width on Hartmann-6 & Bioprocess."""
    import pandas as pd

    from kalos.bench.coverage_at_proposals import (
        _BIO_BOX,
        _H6_BOX,
        _bio_obs,
        _noise_sd,
        bioprocess,
        hartmann6,
    )
    from kalos.core.evaluation import _oof, grouped_cv_report
    from kalos.core.frontier.shift_conformal import ShiftWeightedConformal
    from kalos.core.optimize import propose
    from kalos.core.splits import row_hash_groups
    from kalos.core.surrogate import Surrogate
    from kalos.portal.analysis_parts.intervals import calibrate_selection_aware

    _set_torch_threads()
    n_train = 40
    q = 5
    n_seeds = seeds if quick else max(seeds, 20)
    scenarios = [
        ("hartmann6", hartmann6, _noise_sd(0.1), _H6_BOX),
        ("bioprocess", bioprocess, _bio_obs, _BIO_BOX),
    ]

    results: dict[str, Any] = {}
    for sc_name, truth_fn, obs_fn, box in scenarios:
        base_cov: list[bool] = []
        base_width: list[float] = []
        base_avail: list[bool] = []
        base_time: list[float] = []

        fron_cov: list[bool] = []
        fron_width: list[float] = []
        fron_avail: list[bool] = []
        fron_time: list[float] = []

        for s in range(n_seeds):
            seed = 500 + s
            rng = np.random.default_rng(seed)
            X = _lhs(n_train, box, seed=seed)
            y = obs_fn(X, truth_fn(X), rng)
            yrange = float(np.ptp(y)) or 1.0

            sur = Surrogate().fit(X, y, bounds=box)
            props = propose(sur, box, q=q, raw_samples=128 if quick else 256, seed=seed)
            mu_prop, sd_prop = sur.posterior(props, observation_noise=True)
            y_prop = obs_fn(props, truth_fn(props), rng)

            # Shared repeated CV OOF pool (3 repeats)
            groups = row_hash_groups(pd.DataFrame(X))
            rep = grouped_cv_report(
                X, y, groups=groups, n_splits=5, bounds=box, n_boot=0, n_repeats=3, random_state=0
            )

            # Baseline: calibrate_selection_aware
            t0 = time.perf_counter()
            sel = calibrate_selection_aware(rep["oof_repeats"], [0.90])
            lv = sel["levels"][0]
            if lv["available"]:
                half = float(lv["_q"]) * np.maximum(sd_prop, float(sel["_floor"]))
                lo_b, hi_b = mu_prop - half, mu_prop + half
                base_avail.extend([True] * q)
                base_cov.extend(((y_prop >= lo_b) & (y_prop <= hi_b)).tolist())
                base_width.extend((2.0 * half / yrange).tolist())
            else:
                base_avail.extend([False] * q)
            base_time.append(time.perf_counter() - t0)

            # Frontier: ShiftWeightedConformal over the same 3-repeat OOF pool
            t0 = time.perf_counter()
            all_actual: list[np.ndarray] = []
            all_pred: list[np.ndarray] = []
            all_std: list[np.ndarray] = []
            all_X: list[np.ndarray] = []
            for r_idx in range(len(rep["oof_repeats"])):
                r_dict = rep["oof_repeats"][r_idx]
                all_actual.append(np.asarray(r_dict["actual"], dtype=float))
                all_pred.append(np.asarray(r_dict["pred"], dtype=float))
                all_std.append(np.asarray(r_dict["std"], dtype=float))
                if r_idx == 0:
                    all_X.append(X[np.asarray(rep["oof_index"], dtype=int)])
                else:
                    idx_r: list[int] = []
                    _oof(
                        X,
                        y,
                        groups,
                        5,
                        box,
                        shuffle=True,
                        random_state=r_idx,
                        index_out=idx_r,
                    )
                    all_X.append(X[np.asarray(idx_r, dtype=int)])
            swc = ShiftWeightedConformal(beta=1.0, w_min=0.1, w_max=10.0, min_n_eff=8.0)
            cal = swc.calibrate(
                np.concatenate(all_actual),
                np.concatenate(all_pred),
                np.concatenate(all_std),
                X_cal=np.vstack(all_X),
                X_proposals=props,
                levels=(0.90,),
            )
            iv = swc.interval(mu_prop, sd_prop, pred_X=props, cal_result=cal, coverage=0.90)
            avail_f = np.asarray(iv["available"], dtype=bool)
            fron_avail.extend(avail_f.tolist())
            for i in range(q):
                if avail_f[i]:
                    fron_cov.append(bool(iv["lo"][i] <= y_prop[i] <= iv["hi"][i]))
                    fron_width.append(float((iv["hi"][i] - iv["lo"][i]) / yrange))
            fron_time.append(time.perf_counter() - t0)

        results[sc_name] = {
            "n_seeds": n_seeds,
            "baseline": {
                "coverage": round(float(np.mean(base_cov)), 4) if base_cov else None,
                "median_width": round(float(np.median(base_width)), 4) if base_width else None,
                "availability": round(float(np.mean(base_avail)), 4) if base_avail else 0.0,
                "mean_seconds": round(float(np.mean(base_time)), 4),
            },
            "frontier": {
                "coverage": round(float(np.mean(fron_cov)), 4) if fron_cov else None,
                "median_width": round(float(np.median(fron_width)), 4) if fron_width else None,
                "availability": round(float(np.mean(fron_avail)), 4) if fron_avail else 0.0,
                "mean_seconds": round(float(np.mean(fron_time)), 4),
            },
        }

    return {"arena": "arena5_proposal_intervals", "scenarios": results}


def _summarize_regret_arena(
    name: str,
    optimum: float,
    regrets: dict[str, list[float]],
    times: dict[str, list[float]],
    seeds: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"arena": name, "n_seeds": seeds, "optimum": round(optimum, 4)}
    for arm in ("baseline", "frontier", "lhs"):
        arr = np.asarray(regrets[arm], dtype=float)
        t_arr = np.asarray(times[arm], dtype=float)
        out[arm] = {
            "mean_regret": round(float(np.mean(arr)), 4),
            "median_regret": round(float(np.median(arr)), 4),
            "mean_seconds": round(float(np.mean(t_arr)), 3),
        }
    return out


def _print_markdown_report(summary: dict[str, Any]) -> None:
    print("# Kalos Frontier Race Benchmark Summary\n")
    regret_arenas = [
        k for k in ("arena1", "arena2", "arena3") if k in summary
    ]
    if regret_arenas:
        print("## Arenas 1-3: Optimization Simple Regret (`optimum - best_true`) & Runtime\n")
        print(
            "| Arena | Arm | Mean Regret | Median Regret | Mean Runtime (s) |"
        )
        print("|---|---|---:|---:|---:|")
        for key in regret_arenas:
            a = summary[key]
            for arm in ("baseline", "frontier", "lhs"):
                m = a[arm]
                print(
                    f"| {a['arena']} | {arm} | {m['mean_regret']:.4f} | "
                    f"{m['median_regret']:.4f} | {m['mean_seconds']:.3f} |"
                )
        print()

    if "arena4" in summary:
        a4 = summary["arena4"]
        print("## Arena 4: Multi-Round Batch Drift (Unseen-Batch Leave-One-Batch-Out CV)\n")
        print("| Arm | OOF Spearman (Latent) | OOF RMSE (Latent) | OOF Spearman (Obs) | Mean Runtime (s) |")
        print("|---|---:|---:|---:|---:|")
        for arm in ("baseline", "frontier"):
            m = a4[arm]
            print(
                f"| {arm} | {m['oof_spearman_latent']:.4f} | {m['oof_rmse_latent']:.4f} | "
                f"{m['oof_spearman_obs']:.4f} | {m['mean_seconds']:.3f} |"
            )
        print()

    if "arena5" in summary:
        a5 = summary["arena5"]
        print("## Arena 5: Proposal 90% Interval Coverage vs Relative Width\n")
        print("| Scenario | Method | Coverage | Median Width / Range | Availability | Mean Cal+Int (s) |")
        print("|---|---|---:|---:|---:|---:|")
        for sc_name, sc in a5["scenarios"].items():
            for arm_label, arm_key in (("selection_aware", "baseline"), ("ShiftWeightedConformal", "frontier")):
                m = sc[arm_key]
                cov_s = f"{m['coverage']:.4f}" if m["coverage"] is not None else "n/a"
                w_s = f"{m['median_width']:.4f}" if m["median_width"] is not None else "n/a"
                print(
                    f"| {sc_name} | {arm_label} | {cov_s} | {w_s} | "
                    f"{m['availability']:.4f} | {m['mean_seconds']:.4f} |"
                )
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frontier vs Baseline head-to-head race")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds per arena")
    parser.add_argument("--quick", action="store_true", help="Run reduced-budget quick check")
    parser.add_argument(
        "--arenas",
        default="1,2,3,4,5",
        help="Comma-separated arena numbers to run (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        help="Output JSON summary to stdout ('--json') or a file ('--json path.json')",
    )
    args = parser.parse_args(argv)

    selected = {s.strip() for s in args.arenas.split(",") if s.strip()}
    runners = {
        "1": ("arena1", run_arena1_high_d_sparse),
        "2": ("arena2", run_arena2_heteroscedastic_outliers),
        "3": ("arena3", run_arena3_censored_cliff),
        "4": ("arena4", run_arena4_batch_drift),
        "5": ("arena5", run_arena5_proposal_intervals),
    }

    summary: dict[str, Any] = {"seeds": int(args.seeds), "quick": bool(args.quick)}
    for k in ("1", "2", "3", "4", "5"):
        if k in selected:
            key, fn = runners[k]
            summary[key] = fn(seeds=int(args.seeds), quick=bool(args.quick))

    if args.json == "-":
        print(json.dumps(summary, indent=2))
    else:
        _print_markdown_report(summary)
        if args.json is not None:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
                fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
