"""The suites (PROTOCOL.md section 5): one job = (suite, task, seed) for one arm.

- `closed_loop`: a synthetic instance; LHS initial design of n0, then ROUNDS
  rounds of Q proposals measured by the simulator (fresh noise).
- `emulator`: the same loop on an emulator of a real dataset.
- `pool`: pool replay on a real dataset; proposals snap to the nearest not-yet-run row.
- `snapshot`: one analysis of an LHS (signal family) or null design of size n.
- `snapshot_real`: one analysis of n random rows of a real dataset (`true`) or
  of the same rows with the target permuted (`perm`, a null on real inputs).
- `internal`: LeadGene clone ranking / scale transfer (report-only).

Every job returns a JSON-serializable record with per-round measurements; the
report computes every metric from these records.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import datasets as dsm
from . import leadgene
from .adapter import Arm, ArmOutput, RoundContext, Verdict
from .emulator import EMULATOR_KINDS, emulator_problem
from .problems import NULL_FAMILIES, SIGNAL_FAMILIES, make, null_design
from .protocol import (
    HELDOUT_DATASETS,
    LEVELS,
    N0_REGIMES,
    N_HELDOUT_POINTS,
    ROUNDS,
    SNAPSHOT_SIZES,
    TUNING_DATASETS,
    Q,
)
from .space import Problem, Space

SUITES = ("closed_loop", "emulator", "pool", "snapshot", "snapshot_real", "backtest", "scale", "internal")


def datasets_for(split: str) -> tuple[str, ...]:
    if split == "heldout":
        return tuple(dict.fromkeys((*TUNING_DATASETS, *HELDOUT_DATASETS)))
    return TUNING_DATASETS


def tasks_for(suite: str, split: str) -> list[str]:
    if suite == "closed_loop":
        return list(SIGNAL_FAMILIES)
    if suite == "snapshot":
        return [*SIGNAL_FAMILIES, *NULL_FAMILIES]
    if suite == "pool":
        return list(datasets_for(split))
    if suite == "emulator":
        return [f"{k}:{d}" for d in datasets_for(split) if d in dsm.EMULATOR_DATASETS for k in EMULATOR_KINDS]
    if suite == "snapshot_real":
        return [f"{v}:{d}" for d in datasets_for(split) for v in ("true", "perm")]
    if suite == "internal":
        return list(leadgene.TASKS)
    if suite == "backtest":
        from .backtest import HELDOUT_BACKTESTS, TUNING_BACKTESTS

        return list(TUNING_BACKTESTS) if split != "heldout" else [*TUNING_BACKTESTS, *HELDOUT_BACKTESTS]
    if suite == "scale":
        return list(SCALE_TASKS) if split == "heldout" else [t for t in SCALE_TASKS if not t.startswith("internal")]
    raise KeyError(suite)


SCALE_TASKS = ("loso:fixture/synthetic_scaleup", "internal/pichia_2L_to_35m3")


def _ss(*key: int) -> np.random.Generator:
    return np.random.default_rng(list(key))


def _tkey(task: str) -> int:
    return int.from_bytes(task.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)


def _safe(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _call(arm: Arm, ctx: RoundContext) -> tuple[ArmOutput, dict[str, Any]]:
    """Run one round; an arm exception is recorded and the round is spent on
    random proposals (counted as a fallback)."""
    t0, c0 = time.perf_counter(), time.process_time()
    err = None
    try:
        out = arm.propose(ctx)
    except Exception as exc:  # noqa: BLE001 - recorded, the run continues
        tb = traceback.extract_tb(exc.__traceback__)
        where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
        err = f"{type(exc).__name__} at {where}: {str(exc)[:200]}"
        out = ArmOutput([], Verdict(False, "arm_error", True))
    return out, {"wall_s": round(time.perf_counter() - t0, 3), "cpu_s": round(time.process_time() - c0, 3),
                 "error": err}


def _bands(out: ArmOutput, df: pd.DataFrame) -> dict[float, tuple[np.ndarray, np.ndarray] | None]:
    res: dict[float, tuple[np.ndarray, np.ndarray] | None] = {}
    for lv in LEVELS:
        b = None
        if out.band_at is not None and len(df):
            try:
                b = out.band_at(df, lv)
            except Exception:  # noqa: BLE001 - a band that cannot be built is unavailable
                b = None
        res[lv] = b
    return res


def _cov_record(y: np.ndarray, bands: dict[float, Any], scale: float) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for lv, b in bands.items():
        if b is None:
            rec[str(lv)] = None
            continue
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if not ok.any():
            rec[str(lv)] = None
            continue
        inside = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
        rec[str(lv)] = {"n": int(ok.sum()), "inside": int(inside.sum()),
                        "median_width": _safe(float(np.median((hi[ok] - lo[ok]) / scale)))}
    return rec


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return None
    return _safe(float(spearmanr(a[ok], b[ok]).statistic))


def _auc(score: np.ndarray | None, label: np.ndarray) -> float | None:
    if score is None or label.all() or not label.any():
        return None
    from scipy.stats import rankdata

    ok = np.isfinite(score)
    if ok.sum() < 4 or label[ok].all() or not label[ok].any():
        return None
    r = rankdata(score[ok])
    n1 = int(label[ok].sum())
    n0 = int(ok.sum() - n1)
    return _safe(float((r[label[ok]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)))


def _verdict_rec(out: ArmOutput) -> dict[str, Any]:
    v = out.verdict
    return {"trusted": bool(v.trusted), "state": v.state, "withheld": bool(v.withheld),
            "cv_spearman": _safe(v.cv_spearman), "clears_floor": v.clears_floor}


def _prop_cov(out: ArmOutput, y: np.ndarray, scale: float, n_arm: int) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for lv in LEVELS:
        ins, w = [], []
        for i, p in enumerate(out.proposals[:n_arm]):
            b = p.bands.get(lv)
            if b is not None and b.available:
                ins.append(bool(b.lo <= y[i] <= b.hi))  # type: ignore[operator]
                w.append((b.hi - b.lo) / scale)  # type: ignore[operator]
        rec[str(lv)] = {"inside": ins, "widths": [float(x) for x in w]}
    return rec


# ---------------------------------------------------------------- loops
def _loop(arm: Arm, prob: Problem, suite: str, task: str, seed: int, n0: int) -> dict[str, Any]:
    sp = prob.space
    tk = _tkey(task)
    rng_init, rng_noise, rng_ho, rng_fill = _ss(seed, tk, 1), _ss(seed, tk, 2), _ss(seed, tk, 3), _ss(seed, tk, 4)
    x = sp.sample(n0, rng_init, lhs=True, fixed=False)
    cov = sp.draw_covariates(n0, rng_init)
    x = pd.concat([x, cov], axis=1)
    rounds = np.zeros(n0, int)
    y = prob.observe(x, rng_noise, rounds)
    scale = prob.f_star - prob.f_med
    hist: list[dict[str, Any]] = []

    def best_obs_true() -> tuple[float, float]:
        # Feasible rows only; with a pinned level (the production clone) only rows
        # at that level count, since that is the process being optimized.
        feas = sp.feasible(x)
        tr = prob.truth(x)
        if not feas.any():
            return float(prob.f_med), float(prob.f_med)
        i = int(np.argmax(np.where(feas, y, -np.inf)))
        return float(tr[i]), float(np.max(np.where(feas, tr, -np.inf)))

    b0, bt0 = best_obs_true()
    hist.append({"round": 0, "n_total": int(len(x)), "regret_obs": _safe(prob.norm_regret(b0)),
                 "regret_best": _safe(prob.norm_regret(bt0))})
    for r in range(1, ROUNDS + 1):
        sheet = x.copy()
        sheet[prob.target] = y
        ctx = RoundContext(suite, task, seed, r, sheet, prob.target, sp, Q, LEVELS, dict(prob.roles),
                           tuple(int(v) for v in rounds))
        out, cost = _call(arm, ctx)
        recs = [sp.project(p.recipe) for p in out.proposals[:Q]]
        n_arm = len(recs)
        n_fill = Q - n_arm
        if n_fill:
            recs += [{str(k): v for k, v in r.items()} for r in sp.sample(n_fill, rng_fill).to_dict("records")]
        new = pd.DataFrame(recs)[sp.inputs]
        infeasible = int((~sp.feasible(new)).sum())
        infeasible_strict = int((~sp.feasible(new, lin_tol=1e-9)).sum())
        new = pd.concat([new.reset_index(drop=True), sp.draw_covariates(Q, rng_noise)], axis=1)
        y_new = prob.observe(new, rng_noise, np.full(Q, r))
        ho = sp.sample(N_HELDOUT_POINTS, rng_ho)
        ho = pd.concat([ho, sp.draw_covariates(len(ho), rng_ho)], axis=1)
        y_ho = prob.observe(ho, rng_ho, np.full(len(ho), r))
        pred_ho = None
        if out.predict is not None:
            try:
                pred_ho = _spearman(np.asarray(out.predict(ho), float), prob.truth(ho))
            except Exception:  # noqa: BLE001
                pred_ho = None
        rec = {
            "round": r, **cost, "n_train": int(len(x)), "n_arm_proposals": n_arm, "n_filled": n_fill,
            "n_infeasible": infeasible, "n_infeasible_strict": infeasible_strict, **_verdict_rec(out),
            "at_proposals": _prop_cov(out, y_new, scale, n_arm),
            "heldout": _cov_record(y_ho, _bands(out, ho), scale),
            "prospective_spearman": pred_ho, "info": out.info,
        }
        x = pd.concat([x, new], ignore_index=True)
        y = np.concatenate([y, y_new])
        rounds = np.concatenate([rounds, np.full(Q, r)])
        b, bt = best_obs_true()
        rec.update({"n_total": int(len(x)), "regret_obs": _safe(prob.norm_regret(b)),
                    "regret_best": _safe(prob.norm_regret(bt))})
        hist.append(rec)
    return {"n0": n0, "f_star": prob.f_star, "f_med": prob.f_med, "meta": _jsonable(prob.meta), "rounds": hist}


def _jsonable(d: Any) -> Any:
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, np.ndarray):
        return _jsonable(d.tolist())
    return _safe(d)


def _snap(cands: pd.DataFrame, ds: dsm.RealDataset, recs: list[dict[str, Any]], avail: np.ndarray) -> list[int]:
    """Greedy nearest not-yet-run row for each proposal (unit-scaled continuous
    distance + 1 per categorical mismatch)."""
    sp = ds.space()
    xu = sp.unit(cands) if ds.continuous else np.zeros((len(cands), 0))
    free = avail.copy()
    picks = []
    for rec in recs:
        p = pd.DataFrame([sp.project(rec)])
        pu = sp.unit(p) if ds.continuous else np.zeros((1, 0))
        d = ((xu - pu) ** 2).sum(1)
        for c in ds.categorical:
            d = d + (cands[c].astype(str).to_numpy() != str(p[c].iloc[0])).astype(float)
        d = np.where(free, d, np.inf)
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            break
        picks.append(j)
        free[j] = False
    return picks


def _prep_dataset(ds: dsm.RealDataset) -> dsm.RealDataset:
    if ds.goal == "minimize":  # the harness maximizes; a minimized target is negated, and said so
        f = ds.frame.copy()
        f[ds.target] = -f[ds.target]
        return replace(ds, frame=f, notes=(ds.notes + "; target negated (goal minimize)").strip("; "))
    return ds


def _pool(arm: Arm, ds: dsm.RealDataset, task: str, seed: int, n0: int) -> dict[str, Any]:
    ds = _prep_dataset(ds)
    cands = ds.frame[[*ds.continuous, *ds.categorical]].reset_index(drop=True)
    yv = ds.y
    n = len(yv)
    n0 = int(min(n0, n - ROUNDS * Q - 10))
    tk = _tkey(task)
    rng = _ss(seed, tk, 1)
    run = np.zeros(n, bool)
    run[rng.choice(n, n0, replace=False)] = True
    order: list[int] = [int(v) for v in np.flatnonzero(run)]
    rounds = [0] * n0
    f_star, f_med = float(yv.max()), float(np.median(yv))
    scale = f_star - f_med or float(np.std(yv)) or 1.0
    sp = ds.space()
    roles = {"categoricals": list(ds.categorical)} if ds.categorical else {}
    hist: list[dict[str, Any]] = [{"round": 0, "n_total": n0, "best": float(yv[run].max()),
                                   "regret_obs": _safe((f_star - yv[run].max()) / scale)}]
    rng_ho = _ss(seed, tk, 3)
    for r in range(1, ROUNDS + 1):
        sheet = cands.iloc[order].copy()
        sheet[ds.target] = yv[order]
        ctx = RoundContext("pool", task, seed, r, sheet.reset_index(drop=True), ds.target, sp, Q, LEVELS, roles,
                           tuple(rounds), candidates=cands[~run].reset_index(drop=True))
        out, cost = _call(arm, ctx)
        recs = [dict(p.recipe) for p in out.proposals[:Q]]
        picks = _snap(cands, ds, recs, ~run)
        n_arm = len(picks)
        if len(picks) < Q:
            rest = np.flatnonzero(~run)
            rest = rest[~np.isin(rest, picks)]
            picks += [int(v) for v in rng.choice(rest, Q - len(picks), replace=False)]
        picks = [int(p) for p in picks]
        snapped = cands.iloc[picks].reset_index(drop=True)
        y_new = yv[picks]
        free = np.flatnonzero(~run)
        free = free[~np.isin(free, picks)]
        ho_idx = rng_ho.choice(free, min(N_HELDOUT_POINTS, len(free)), replace=False)
        ho = cands.iloc[ho_idx].reset_index(drop=True)
        bands_snap = _bands(out, snapped.iloc[:n_arm])
        pred_ho = None
        if out.predict is not None:
            try:
                pred_ho = _spearman(np.asarray(out.predict(ho), float), yv[ho_idx])
            except Exception:  # noqa: BLE001
                pred_ho = None
        rec = {
            "round": r, **cost, "n_train": len(order), "n_arm_proposals": n_arm, "n_filled": Q - n_arm,
            **_verdict_rec(out),
            # coverage at the runs actually made (the snapped rows), from band_at;
            # arms without band_at fall back to the band they put on their proposal
            "at_proposals": (_cov_from_bands(y_new[:n_arm], bands_snap, scale) if out.band_at is not None
                             else _prop_cov(out, y_new, scale, n_arm)),
            "heldout": _cov_record(yv[ho_idx], _bands(out, ho), scale),
            "prospective_spearman": pred_ho, "info": out.info,
        }
        run[picks] = True
        order += [int(p) for p in picks]
        rounds += [r] * Q
        rec.update({"n_total": len(order), "best": float(yv[run].max()),
                    "regret_obs": _safe((f_star - yv[run].max()) / scale)})
        hist.append(rec)
    top = float(np.quantile(yv, 1 - 0.05))
    return {"n0": n0, "f_star": f_star, "f_med": f_med, "top5_threshold": top, "pool_n": n,
            "dataset": ds.info(), "order_values": [float(yv[i]) for i in order], "rounds": hist}


def _cov_from_bands(y: np.ndarray, bands: dict[float, Any], scale: float) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for lv, b in bands.items():
        if b is None:
            rec[str(lv)] = {"inside": [], "widths": []}
            continue
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        rec[str(lv)] = {"inside": [bool(v) for v in ((y >= lo) & (y <= hi))[ok]],
                        "widths": [float(v) for v in ((hi - lo) / scale)[ok]]}
    return rec


def _snapshot(arm: Arm, task: str, seed: int) -> dict[str, Any]:
    n = SNAPSHOT_SIZES[seed % len(SNAPSHOT_SIZES)]
    tk = _tkey(task)
    rng, rng_noise, rng_ho = _ss(seed, tk, 1), _ss(seed, tk, 2), _ss(seed, tk, 3)
    round_index: tuple[int, ...] | None = None
    if task in NULL_FAMILIES:
        sheet, rounds, prob = null_design(task, seed, n)
        if prob.meta.get("round_attr"):
            round_index = tuple(int(v) for v in rounds)
        scale = float(np.std(sheet["y"])) or 1.0
    else:
        prob = make(task, seed)
        x = prob.space.sample(n, rng, lhs=True, fixed=False)
        x = pd.concat([x, prob.space.draw_covariates(n, rng)], axis=1)
        sheet = x.copy()
        sheet[prob.target] = prob.observe(x, rng_noise, np.zeros(n, int))
        scale = prob.f_star - prob.f_med
    sp = prob.space
    ctx = RoundContext("snapshot", task, seed, 1, sheet, prob.target, sp, Q, LEVELS, dict(prob.roles), round_index)
    out, cost = _call(arm, ctx)
    recs = [sp.project(p.recipe) for p in out.proposals[:Q]]
    rec: dict[str, Any] = {"n": n, "signal": prob.signal, **cost, **_verdict_rec(out),
                           "n_arm_proposals": len(recs), "info": out.info, "meta": _jsonable(prob.meta)}
    if recs:
        new = pd.concat([pd.DataFrame(recs)[sp.inputs], sp.draw_covariates(len(recs), rng_noise)], axis=1)
        rec["at_proposals"] = _prop_cov(out, prob.observe(new, rng_noise, np.ones(len(recs), int)), scale, len(recs))
    ho = sp.sample(N_HELDOUT_POINTS, rng_ho)
    ho = pd.concat([ho, sp.draw_covariates(len(ho), rng_ho)], axis=1)
    rec["heldout"] = _cov_record(prob.observe(ho, rng_ho, np.zeros(len(ho), int)), _bands(out, ho), scale)
    if prob.signal and out.predict is not None:
        try:
            rec["prospective_spearman"] = _spearman(np.asarray(out.predict(ho), float), prob.truth(ho))
        except Exception:  # noqa: BLE001
            rec["prospective_spearman"] = None
    return rec


def _snapshot_real(arm: Arm, ds: dsm.RealDataset, task: str, seed: int) -> dict[str, Any]:
    ds = _prep_dataset(ds)
    variant = task.split(":", 1)[0]
    n_all = len(ds.frame)
    n = int(min(SNAPSHOT_SIZES[seed % len(SNAPSHOT_SIZES)], n_all // 2))
    tk = _tkey(task.split(":", 1)[1])  # same rows for true and perm
    rng = _ss(seed, tk, 1)
    idx = rng.choice(n_all, n, replace=False)
    cands = ds.frame[[*ds.continuous, *ds.categorical]].reset_index(drop=True)
    yv = ds.y
    sheet = cands.iloc[idx].reset_index(drop=True)
    ys = yv[idx].copy()
    if variant == "perm":
        ys = _ss(seed, tk, 5).permutation(ys)
    sheet[ds.target] = ys
    roles = {"categoricals": list(ds.categorical)} if ds.categorical else {}
    ctx = RoundContext("snapshot_real", task, seed, 1, sheet, ds.target, ds.space(), Q, LEVELS, roles)
    out, cost = _call(arm, ctx)
    scale = float(yv.max() - np.median(yv)) or float(np.std(yv)) or 1.0
    rest = np.setdiff1d(np.arange(n_all), idx)
    ho_idx = _ss(seed, tk, 3).choice(rest, min(N_HELDOUT_POINTS, len(rest)), replace=False)
    ho = cands.iloc[ho_idx].reset_index(drop=True)
    rec: dict[str, Any] = {"n": n, "signal": variant == "true", "dataset": ds.info(), **cost, **_verdict_rec(out),
                           "n_arm_proposals": len(out.proposals), "info": out.info}
    if variant == "true":
        rec["heldout"] = _cov_record(yv[ho_idx], _bands(out, ho), scale)
        if out.predict is not None:
            try:
                rec["prospective_spearman"] = _spearman(np.asarray(out.predict(ho), float), yv[ho_idx])
            except Exception:  # noqa: BLE001
                rec["prospective_spearman"] = None
    return rec


def _internal(arm: Arm, task: str, split: str, seed: int, path: Path) -> dict[str, Any]:
    """LeadGene tasks: aggregate metrics per fold only (no clone-level values)."""
    clones = leadgene.load_clones(path)
    build, feats, target = leadgene.TASKS[task]
    frame = build(clones)
    folds = []
    for name, tr, te in leadgene.splits(task, split, seed, frame):
        train, test = frame.iloc[tr].reset_index(drop=True), frame.iloc[te].reset_index(drop=True)
        cont = {c: (float(frame[c].min()), float(frame[c].max())) for c in feats}
        sp = Space(continuous=cont)
        ctx = RoundContext("internal", task, seed, 1, train[[*feats, target]], target, sp, Q, LEVELS,
                           {"features": list(feats)})
        out, cost = _call(arm, ctx)
        yt = test[target].to_numpy(float)
        fold: dict[str, Any] = {"fold": name, "n_train": len(train), "n_test": len(test), **cost,
                                **_verdict_rec(out)}
        scale = float(np.ptp(frame[target])) or 1.0
        pred = None
        if out.predict is not None:
            try:
                pred = np.asarray(out.predict(test[feats]), float)
            except Exception:  # noqa: BLE001
                pred = None
        fold["spearman"] = _spearman(pred, yt) if pred is not None and len(yt) >= 5 else None
        fold["coverage"] = _cov_record(yt, _bands(out, test[feats]), scale)
        if task == "rank_96to24" and pred is not None and len(yt) >= 16:
            k = 8
            top_true = set(np.argsort(-yt)[:k])
            fold["top8_precision"] = len(top_true & set(np.argsort(-pred)[:k])) / k
            fold["top8_precision_carry_forward"] = len(top_true & set(np.argsort(-test["signal96"].to_numpy())[:k])) / k
            fold["spearman_carry_forward"] = _spearman(test["signal96"].to_numpy(float), yt)
        if task == "transfer_loso":
            fold["spearman_carry_forward"] = _spearman(test["prev_primary_pct"].to_numpy(float), yt)
        if task in ("rank_96to24", "transfer_loso") and len(yt) >= 6:
            # classification view (as voyager-ml-pipeline): "top third at the next scale",
            # scored by the arm's predicted mean (AUC) and by carry-forward
            lab = yt >= np.quantile(yt, 2 / 3)
            carry = test["signal96" if task == "rank_96to24" else "prev_primary_pct"].to_numpy(float)
            fold["auc_top_third"] = _auc(pred, lab) if pred is not None else None
            fold["auc_top_third_carry_forward"] = _auc(carry, lab)
        if task == "flask_small" and pred is not None:
            fold["abs_err"] = _safe(float(abs(pred[0] - yt[0])))
        folds.append(fold)
    return {"label": leadgene.INTERNAL_LABEL, "n_rows": len(frame), "folds": folds}


def _scale(arm: Arm, task: str, seed: int) -> dict[str, Any]:
    """Leave-one-scale-out on a sheet with a scale column; the held-out scale's
    rows are predicted by an arm trained on the other scales. Directions: `up`
    (held-out scale above every training scale), `down`, `interior`. The Pichia
    task (5 fed-batch runs, 3 L to 35 m3) expects a refusal or an untrusted verdict."""
    if seed not in (0, 1000, 5001, 9002):
        return {"skipped": "deterministic: one evaluation per split", "group": "scale"}
    if task.startswith("loso:"):
        ds = dsm.load(task.split(":", 1)[1])
        frame, target, feats, scol = ds.frame, ds.target, list(ds.continuous), "scale_L"
        group, label = "fixture_scale", "synthetic (generated) scale-up fixture"
    else:
        path = (Path(os.environ.get("KALOS_EXTERNAL_DATA", "~/kalos/external")).expanduser() / "bioqore"
                / "bioqore-synthetic-data-project" / "data" / "raw" / "anagram_compiled.csv")
        if not path.is_file():
            raise FileNotFoundError("anagram_compiled.csv: set KALOS_EXTERNAL_DATA (internal data)")
        raw = pd.read_csv(path)
        raw = raw[pd.to_numeric(raw["Culture_Volume.mL"], errors="coerce") > 3]
        frame = pd.DataFrame({"scale_L": pd.to_numeric(raw["Culture_Volume.mL"], errors="coerce") / 1000.0,
                              "time_h": pd.to_numeric(raw["total_time_h_max"], errors="coerce"),
                              "lipase_g_L": pd.to_numeric(raw["lipase_g.L"], errors="coerce")}).dropna()
        target, feats, scol = "lipase_g_L", ["scale_L", "time_h"], "scale_L"
        group, label = "internal", "internal, data rights unconfirmed; n=5 single runs, refusal expected"
    scales = np.sort(frame[scol].unique())
    folds = []
    for s_out in scales:
        tr, te = frame[frame[scol] != s_out], frame[frame[scol] == s_out]
        tr_scales = tr[scol].to_numpy()
        direction = "up" if s_out > tr_scales.max() else "down" if s_out < tr_scales.min() else "interior"
        sp = Space(continuous={c: (float(frame[c].min()), float(frame[c].max())) for c in feats})
        ctx = RoundContext("scale", task, seed, 1, tr[[*feats, target]].reset_index(drop=True), target, sp, Q,
                           LEVELS, {"features": feats})
        out, cost = _call(arm, ctx)
        yt = te[target].to_numpy(float)
        pred = None
        if out.predict is not None:
            try:
                pred = np.asarray(out.predict(te[feats]), float)
            except Exception:  # noqa: BLE001
                pred = None
        scale_y = float(np.ptp(frame[target])) or 1.0
        folds.append({"held_out_scale": float(s_out), "direction": direction, "n_train": len(tr), "n_test": len(te),
                      **cost, **_verdict_rec(out),
                      "spearman": _spearman(pred, yt) if pred is not None else None,
                      "mae_norm": _safe(float(np.nanmean(np.abs(pred - yt)) / scale_y)) if pred is not None else None,
                      "coverage": _cov_record(yt, _bands(out, te[feats]), scale_y)})
    return {"label": label, "group": group, "folds": folds}


def run_job(arm: Arm, suite: str, task: str, seed: int, split: str,
            cache_dir: Path | None = None, user_data: Path | None = None,
            leadgene_path: Path | None = None) -> dict[str, Any]:
    """One job; returns the record (never raises for an arm failure)."""
    t0 = time.perf_counter()
    base = {"suite": suite, "task": task, "seed": seed, "split": split, "arm": arm.name}
    if suite == "closed_loop":
        prob = make(task, seed)
        body = _loop(arm, prob, suite, task, seed, N0_REGIMES[seed % len(N0_REGIMES)])
        body["group"] = prob.group
    elif suite == "emulator":
        kind, dname = task.split(":", 1)
        ds = _load_ds(dname, user_data)
        prob = emulator_problem(_prep_dataset(ds), kind, cache_dir)
        body = _loop(arm, prob, suite, task, seed, N0_REGIMES[seed % len(N0_REGIMES)])
        body["group"] = prob.group
        body["caveat"] = "emulator: a model of the data, not the lab (PROTOCOL.md 3.3)"
    elif suite == "pool":
        ds = _load_ds(task, user_data)
        body = _pool(arm, ds, task, seed, N0_REGIMES[seed % len(N0_REGIMES)])
        body["group"] = _pool_group(ds.group)
    elif suite == "snapshot":
        body = _snapshot(arm, task, seed)
        body["group"] = "synthetic"
    elif suite == "snapshot_real":
        ds = _load_ds(task.split(":", 1)[1], user_data)
        body = _snapshot_real(arm, ds, task, seed)
        body["group"] = _pool_group(ds.group)
    elif suite == "backtest":
        from .backtest import run_backtest

        body = run_backtest(arm, task, seed)
    elif suite == "scale":
        body = _scale(arm, task, seed)
    elif suite == "internal":
        if leadgene_path is None:
            raise FileNotFoundError("internal suite needs KALOS_EXTERNAL_DATA or --leadgene")
        body = _internal(arm, task, split, seed, leadgene_path)
        body["group"] = "internal"
    else:
        raise KeyError(suite)
    return {**base, **body, "job_seconds": round(time.perf_counter() - t0, 2)}


def _pool_group(g: str) -> str:
    return {"real": "real_pool", "ood": "ood_pool", "fixture": "fixture_pool"}.get(g, g)


def _load_ds(name: str, user_data: Path | None) -> dsm.RealDataset:
    if name.startswith("user/"):
        if user_data is None:
            raise FileNotFoundError("user dataset without --user-data")
        found, _ = dsm.load_user_folder(user_data)
        for d in found:
            if d.name == name:
                return d
        raise FileNotFoundError(name)
    return dsm.load(name)
