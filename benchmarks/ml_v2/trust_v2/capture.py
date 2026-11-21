"""ml-trust research capture (TUNING seeds only): run the reference engine through the
protocol's own suite loops and keep everything an interval / trust-gate variant needs,
so variants can be scored offline without refitting a GP.

Per analysis it keeps: every grouped-CV `_oof` call (groups, fold, OOF mean / sd /
actual, row index), which grouping was binding, the shipped interval method, the
verdict, and the served model's posterior (mean, latent sd, served predictive sd) at the
proposals and the 200 held-out runs, plus the MEASURED values there (captured from the
problem's `observe`, i.e. the same noise draws the harness uses).

Interval-only patches do not change proposals, so for them this capture is the exact
trajectory the harness would see (up to the engine's documented non-determinism).

    python benchmarks/ml_v2/trust_v2/capture.py --suite closed_loop --out /tmp/cap --workers 8
    [--patch cv_served]   # S2: CV folds fit like the served model (changes OOF only)
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gzip
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

CACHE = Path.home() / "kalos/kalos-engine-mlbench/benchmarks/ml_v2/results/reference_tuning/_cache"


def _setup_threads() -> None:
    os.environ.setdefault("KALOS_TORCH_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.cur: dict[str, Any] | None = None
        self.obs: list[tuple[Any, np.ndarray]] = []


REC = Recorder()


@contextlib.contextmanager
def _engine_taps(patch: str | None):
    import kalos.core.evaluation as ev
    import kalos.portal.analysis as an
    from kalos.bench.ml_v2.arms import kalos_ref as kr

    orig_oof = ev._oof
    orig_lf = an.leak_free_cv
    orig_band = kr._band_fn
    served_oof = None
    if patch and "cv_served" in patch:
        from kalos.bench.ml_v2.arms.trust_v2 import served_oof as served_oof  # noqa: PLC0415

    def oof(X, y, groups, n_splits, bounds, cat_dims=None, shuffle=False, random_state=42,
            index_out=None, fold_out=None):
        idx: list[int] = []
        fold: list[int] = []
        fn = served_oof if served_oof is not None else orig_oof
        out = fn(X, y, groups, n_splits, bounds, cat_dims=cat_dims, shuffle=shuffle, random_state=random_state,
                 index_out=idx, fold_out=fold)
        if index_out is not None:
            index_out.extend(idx)
        if fold_out is not None:
            fold_out.extend(fold)
        if REC.cur is not None:
            REC.cur.setdefault("X", np.asarray(X, float))
            REC.cur.setdefault("y", np.asarray(y, float))
            REC.cur.setdefault("bounds", None if bounds is None else np.asarray(bounds, float))
            REC.cur["oof"].append({
                "groups": [str(g) for g in np.asarray(groups).tolist()], "shuffle": bool(shuffle),
                "random_state": int(random_state), "idx": np.asarray(idx, int), "fold": np.asarray(fold, int),
                "pred": np.asarray(out[0], float), "sd": np.asarray(out[1], float),
                "actual": np.asarray(out[2], float),
            })
        return out

    def lf(primary_rep, primary_groups, primary_basis, primary_column, candidates, run_cv, **kw):
        block, rep, groups, basis = orig_lf(primary_rep, primary_groups, primary_basis, primary_column,
                                            candidates, run_cv, **kw)
        if REC.cur is not None:
            REC.cur["binding_basis"] = basis
            REC.cur["binding_groups"] = [str(g) for g in np.asarray(groups).tolist()]
            REC.cur["primary_groups"] = [str(g) for g in np.asarray(primary_groups).tolist()]
            REC.cur["candidates"] = [{"basis": c["basis"], "column": c["column"],
                                      "groups": [str(g) for g in np.asarray(c["groups"]).tolist()]}
                                     for c in candidates]
            REC.cur["primary_basis"] = primary_basis
            from kalos.bench.ml_v2.arms import trust_v2 as tv
            REC.cur["order_check"] = dict(tv.LAST_ORDER_CHECK)
        return block, rep, groups, basis

    def band(cap, res):
        if REC.cur is not None:
            REC.cur["cap"] = cap
            REC.cur["res"] = {k: res.get(k) for k in (
                "interval_method", "reliability", "verdict", "cv_grouping", "proposal_trust", "cv_spearman",
                "cv_spearman_per_repeat", "noise", "n_rows", "cv_ci95")}
            REC.cur["props"] = [
                {"recipe": p.get("recipe"), "pred": p.get("pred"), "predictive_std": p.get("predictive_std"),
                 "intervals": p.get("intervals")} for p in (res.get("proposals") or [])]
        return orig_band(cap, res)

    ev._oof = oof
    an.leak_free_cv = lf
    kr._band_fn = band
    try:
        yield
    finally:
        ev._oof = orig_oof
        an.leak_free_cv = orig_lf
        kr._band_fn = orig_band


def _posterior_fn(cap: dict[str, Any], res: dict[str, Any]):
    """mean, latent sd, served predictive sd and the encoded design at arbitrary rows
    (the same encoding and sigma source as `kalos_ref._band_fn`)."""
    import pandas as pd

    import kalos.portal.analysis as an

    s = cap.get("surrogate")
    encoder = cap.get("encoder")
    if s is None or encoder is None:
        return None
    cont, cats, code_maps, width, absent = encoder
    method = res.get("interval_method") or {}
    noise_floor = float((res.get("noise") or {}).get("noise_sd") or 0.0)

    def post(df: pd.DataFrame):
        frame = df.copy()
        ok = np.ones(len(frame), bool)
        for c in cont:
            v = pd.to_numeric(frame[c], errors="coerce") if c in frame else pd.Series(np.nan, index=frame.index)
            ok &= np.isfinite(v.to_numpy(float)) | (c in absent)
        for c in cats:
            lab = frame[c].astype(str).str.strip() if c in frame else pd.Series("", index=frame.index)
            ok &= lab.isin(code_maps[c]).to_numpy()
        n = len(df)
        mu, lat, sp = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        xenc = None
        if ok.any():
            x, _ = an._encode_pending(frame[ok], cont, cats, code_maps, width, absent_zero=absent)
            m, la = s.posterior(x)
            la = np.asarray(la, float).reshape(-1)
            if method.get("sigma_source") == "latent_plus_assay_noise_floor":
                spp = np.sqrt(la ** 2 + noise_floor ** 2)
            else:
                _, spp = s.posterior(x, observation_noise=True)
            mu[ok], lat[ok], sp[ok] = np.asarray(m, float).reshape(-1), la, np.asarray(spp, float).reshape(-1)
            xenc = np.full((n, np.asarray(x).shape[1]), np.nan)
            xenc[ok] = np.asarray(x, float)
        return {"mu": mu, "lat": lat, "sd": sp, "x": xenc}

    return post


def _make_arm(patch: str | None):
    from kalos.bench.ml_v2.arms.trust_v2 import TrustV2Arm

    class CaptureArm(TrustV2Arm):
        name = "trust_capture"

        def propose(self, ctx):  # type: ignore[no-untyped-def]
            REC.cur = {"round": ctx.round, "oof": [], "n_sheet": len(ctx.sheet),
                       "round_index": None if ctx.round_index is None else list(ctx.round_index),
                       "sheet_y": np.asarray(ctx.sheet[ctx.target], float)}
            t0, c0 = time.perf_counter(), time.process_time()
            try:
                out = super().propose(ctx)
            except Exception as exc:
                cur = REC.cur or {}
                REC.cur = None
                cur.pop("cap", None)
                cur["error"] = f"{type(exc).__name__}: {exc}"
                cur["n_props"] = 0
                REC.calls.append(cur)
                raise
            cur = REC.cur
            REC.cur = None
            cur["wall_s"], cur["cpu_s"] = time.perf_counter() - t0, time.process_time() - c0
            cur["post"] = _posterior_fn(cur.pop("cap", {}) or {}, cur.get("res") or {})
            cur["trusted"] = out.verdict.trusted
            cur["state"] = out.verdict.state
            cur["n_props"] = len(out.proposals)
            cur["prop_bands"] = [{str(k): (b.lo, b.hi) for k, b in p.bands.items()} for p in out.proposals]
            REC.calls.append(cur)
            return out

    return CaptureArm()


def _wrap_problem(prob):
    orig = prob.observe

    def observe(df, rng, rounds):
        y = orig(df, rng, rounds)
        REC.obs.append((df.reset_index(drop=True).copy(), np.asarray(y, float)))
        return y

    return dataclasses.replace(prob, observe=observe)


def run_one(suite: str, task: str, seed: int, out_dir: str, patch: str | None) -> str:
    _setup_threads()
    from kalos.bench.ml_v2 import suites as S
    from kalos.bench.ml_v2.emulator import emulator_problem
    from kalos.bench.ml_v2.protocol import N0_REGIMES

    out = Path(out_dir) / suite / f"{task.replace('/', '__').replace(':', '--')}_s{seed}.pkl.gz"
    if out.is_file():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    REC.calls, REC.obs = [], []
    arm = _make_arm(patch)
    arm.config = {"order": bool(patch and "order" in patch)}
    rec: dict[str, Any] = {"suite": suite, "task": task, "seed": seed, "patch": patch}
    orig_make, orig_null = S.make, S.null_design
    S.make = lambda t, s: _wrap_problem(orig_make(t, s))  # type: ignore[assignment]

    def nd(fam, s, n):
        df, rounds, prob = orig_null(fam, s, n)
        return df, rounds, _wrap_problem(prob)

    S.null_design = nd  # type: ignore[assignment]
    try:
        with _engine_taps(patch):
            if suite == "closed_loop":
                prob = S.make(task, seed)
                body = S._loop(arm, prob, suite, task, seed, N0_REGIMES[seed % len(N0_REGIMES)])
                rec["scale"] = prob.f_star - prob.f_med
            elif suite == "emulator":
                kind, dname = task.split(":", 1)
                prob = _wrap_problem(emulator_problem(S._prep_dataset(S._load_ds(dname, None)), kind, CACHE))
                body = S._loop(arm, prob, suite, task, seed, N0_REGIMES[seed % len(N0_REGIMES)])
                rec["scale"] = prob.f_star - prob.f_med
            elif suite == "snapshot":
                body = S._snapshot(arm, task, seed)
            elif suite == "snapshot_real":
                ds = S._load_ds(task.split(":", 1)[1], None)
                body = S._snapshot_real(arm, ds, task, seed)
            else:
                raise KeyError(suite)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
        body = {}
    finally:
        S.make, S.null_design = orig_make, orig_null
    rec["body"] = body
    # Pair each analysis with what was measured after it (closed loop: proposals then
    # held-out; snapshot: [proposals], held-out). Initial-design observe comes first.
    obs = list(REC.obs)
    calls = REC.calls
    if suite in ("closed_loop", "emulator"):
        obs = obs[1:]
        for i, c in enumerate(calls):
            c["obs_props"], c["obs_ho"] = obs[2 * i], obs[2 * i + 1]
    elif suite == "snapshot":
        if calls:
            c = calls[0]
            if c["n_props"] and len(obs) >= 2:
                c["obs_props"], c["obs_ho"] = obs[-2], obs[-1]
            elif obs:
                c["obs_ho"] = obs[-1]
    for c in calls:
        post = c.pop("post", None)
        for key in ("obs_props", "obs_ho"):
            if key in c and post is not None:
                df, y = c[key]
                c[key + "_post"] = post(df)
                c[key] = y
            elif key in c:
                c[key] = c[key][1]
    rec["calls"] = calls
    with gzip.open(out, "wb") as fh:
        pickle.dump(rec, fh)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patch", default=None)
    a = ap.parse_args()
    from kalos.bench.ml_v2.protocol import TUNING_SEEDS
    from kalos.bench.ml_v2.suites import tasks_for

    tasks = a.tasks or tasks_for(a.suite, "tuning")
    seeds = a.seeds if a.seeds is not None else list(TUNING_SEEDS[a.suite])
    assert all(s in TUNING_SEEDS[a.suite] for s in seeds), "tuning seeds only"
    jobs = [(a.suite, t, s, a.out, a.patch) for s in seeds for t in tasks]
    t0 = time.perf_counter()
    if a.workers <= 1:
        for j in jobs:
            print(run_one(*j), flush=True)
        return
    import multiprocessing as mp

    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(run_one, *j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
                print(f"[{k}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {futs[f][1]} s{futs[f][2]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {futs[f]}: {exc}", flush=True)


if __name__ == "__main__":
    main()
