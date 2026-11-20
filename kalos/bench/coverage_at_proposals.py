"""Coverage of the per-proposal intervals AT THE PROPOSALS (ML audit B1, REORG F02/F15).

The question: when Kalos says "90% range" next to a proposed recipe, how often
does the measurement of that recipe land inside it? The held-out calibration
that sets the band uses past runs; proposals are chosen BECAUSE the model
predicts them high, so their errors are not exchangeable with past runs.

Protocol, per (scenario, seed):
  1. draw a Latin-hypercube design of n runs from a known simulator, observe
     it with the simulator's noise;
  2. run the production analysis (`kalos.portal.analysis._analyze`);
  3. MEASURE each returned proposal once with fresh noise and check whether
     the value lies inside the shipped `proposals[].intervals` [lo, hi];
  4. rebuild the same band at 300 uniform points in the design box (the
     served surrogate's mean and predictive sd, the response's
     `interval_method`) and measure those too: coverage "on runs like past
     runs" for comparison.

Widths are reported as a fraction of the training target's range. The five
proposals of one seed are correlated, so the 95% CI is a seed-cluster
bootstrap, not a binomial interval.

Scenarios (simulators, all synthetic): Hartmann-6 (n=30, 80; noise sd 0.1),
the bioprocess simulator behind examples/synthetic_bioprocess (10 inputs,
zero-inflated, 12% multiplicative noise; n=40, 100), a heteroscedastic 3-D bump
(noise sd 0.05 + 0.3 * f; n=40) and Branin (n=20, noise sd 10) as a control.

    python -m kalos.bench.coverage_at_proposals --seeds 40 --procs 6 \
        --out benchmarks/results/coverage_at_proposals.json

`--seed-offset` draws fresh seeds (the selection-aware method was chosen on
seeds 0-39; its reported numbers come from seeds 100-139).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

# --------------------------------------------------------------- simulators
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def hartmann6(x: np.ndarray) -> np.ndarray:
    """Hartmann-6 on [0,1]^6, negated to a maximization (max 3.32237)."""
    inner = ((x[:, None, :] - _H6_P[None]) ** 2 * _H6_A[None]).sum(-1)
    return (_H6_ALPHA[None] * np.exp(-inner)).sum(-1)


def branin(x: np.ndarray) -> np.ndarray:
    """Branin on [-5,10] x [0,15], negated (max -0.397887)."""
    x1, x2 = x[:, 0], x[:, 1]
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return -((x2 - b * x1 ** 2 + c * x1 - 6.0) ** 2 + 10.0 * (1 - t) * np.cos(x1) + 10.0)


BIO_SPECS = {
    "pH": (6.0, 7.6), "temperature_C": (30.0, 39.0), "glucose_gL": (2.0, 12.0),
    "glutamine_mM": (1.0, 8.0), "feed_rate_mLh": (0.1, 2.0), "DO_pct": (20.0, 80.0),
    "agitation_rpm": (100.0, 400.0), "inoculum_1e6": (0.2, 2.0), "antifoam_mgL": (0.0, 50.0),
    "osmolality_mOsm": (280.0, 360.0),
}


def _bump(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def bioprocess(x: np.ndarray) -> np.ndarray:
    """The generator of examples/synthetic_bioprocess (same truth)."""
    d = dict(zip(BIO_SPECS, x.T))
    t = (3.2 * _bump(d["pH"], 7.0, 0.35) * _bump(d["temperature_C"], 36.5, 1.6)
         * _bump(d["DO_pct"], 52.0, 14.0) * (d["glucose_gL"] / (d["glucose_gL"] + 3.0))
         * (0.6 + 0.4 * _bump(d["glutamine_mM"], 5.0, 2.0))
         * (0.7 + 0.3 * _bump(d["feed_rate_mLh"], 1.2, 0.5)))
    viable = ((d["pH"] >= 6.25) & (d["pH"] <= 7.45) & (d["temperature_C"] < 38.4)
              & (d["osmolality_mOsm"] < 352) & (d["DO_pct"] > 24))
    return np.where(viable, t, 0.0)


def hetero_bump(x: np.ndarray) -> np.ndarray:
    return 5 * np.exp(-8 * ((x - 0.6) ** 2).sum(1))


def _noise_sd(sd: float) -> Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]:
    def obs(_x: np.ndarray, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return f + rng.normal(0, sd, len(f))
    return obs


def _bio_obs(_x: np.ndarray, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(np.where(f > 0, f * rng.normal(1.0, 0.12, len(f)), 0.0), 0, None)


def _het_obs(_x: np.ndarray, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return f + rng.normal(0, 0.05 + 0.3 * f, len(f))


_H6_BOX = np.array([[0.0] * 6, [1.0] * 6])
_BIO_BOX = np.array([[v[0] for v in BIO_SPECS.values()], [v[1] for v in BIO_SPECS.values()]])
_UNIT3 = np.array([[0.0] * 3, [1.0] * 3])
_BRANIN_BOX = np.array([[-5.0, 0.0], [10.0, 15.0]])

# name: (truth, observe, box, input names, n runs)
SCENARIOS: dict[str, tuple[Callable, Callable, np.ndarray, list[str], int]] = {
    "hartmann6_n30": (hartmann6, _noise_sd(0.1), _H6_BOX, [f"x{i}" for i in range(1, 7)], 30),
    "hartmann6_n80": (hartmann6, _noise_sd(0.1), _H6_BOX, [f"x{i}" for i in range(1, 7)], 80),
    "bioprocess_n40": (bioprocess, _bio_obs, _BIO_BOX, list(BIO_SPECS), 40),
    "bioprocess_n100": (bioprocess, _bio_obs, _BIO_BOX, list(BIO_SPECS), 100),
    "heteroscedastic_n40": (hetero_bump, _het_obs, _UNIT3, ["x1", "x2", "x3"], 40),
    "branin_n20_control": (branin, _noise_sd(10.0), _BRANIN_BOX, ["x1", "x2"], 20),
}
LEVELS = (0.9, 0.95)
N_RANDOM = 300


def _lhs(n: int, box: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = box.shape[1]
    u = (np.argsort(rng.random((n, d)), axis=0) + rng.random((n, d))) / n
    return box[0] + u * (box[1] - box[0])


# ------------------------------------------------------------- one analysis
_SERVED: dict[str, Any] = {}


def _capture_served_surrogate() -> None:
    """Keep a handle on the PROPOSAL surrogate (`check_conditioning=True` is the
    one served fit). In-process wrapper; nothing in the engine is edited."""
    from kalos.core import surrogate as s_mod

    orig = s_mod.Surrogate.fit
    if getattr(orig, "_coverage_bench", False):
        return

    def fit(self, *a, **kw):  # type: ignore[no-untyped-def]
        out = orig(self, *a, **kw)
        if kw.get("check_conditioning"):
            _SERVED["s"] = self
        return out

    fit._coverage_bench = True  # type: ignore[attr-defined]
    s_mod.Surrogate.fit = fit  # type: ignore[method-assign]


def run_one(name: str, seed: int) -> dict[str, Any]:
    """One analysis: per-level inside flags at the proposals and at random points."""
    import pandas as pd
    import torch

    from kalos.portal.analysis import _analyze
    from kalos.portal.analysis_parts.intervals import band_from_summary

    torch.set_num_threads(1)
    _capture_served_surrogate()
    truth, observe, box, names, n = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    x = _lhs(n, box, rng)
    y = observe(x, truth(x), rng)
    df = pd.DataFrame(x, columns=names)
    df["titer"] = y
    t0 = time.perf_counter()
    try:
        res = _analyze(df, target="titer")
    except Exception as exc:  # noqa: BLE001 - a refusal is a result here
        return {"scenario": name, "seed": seed, "error": f"{type(exc).__name__}: {exc}"[:300]}
    secs = time.perf_counter() - t0
    s = _SERVED["s"]
    yrange = float(np.ptp(y)) or 1.0
    out: dict[str, Any] = {"scenario": name, "seed": seed, "seconds": round(secs, 2),
                           "n_proposals": len(res["proposals"]),
                           "verdict": (res.get("verdict") or {}).get("state"),
                           "levels": {}}
    props = res["proposals"]
    p_x = np.array([[p["recipe"][k] for k in names] for p in props], float).reshape(-1, len(names))
    p_y = observe(p_x, truth(p_x), rng) if len(p_x) else np.empty(0)
    lo_box = np.array([res["feature_ranges"][k][0] for k in names])
    hi_box = np.array([res["feature_ranges"][k][1] for k in names])
    r_x = lo_box + rng.random((N_RANDOM, len(names))) * (hi_box - lo_box)  # inside the fitted box
    r_y = observe(r_x, truth(r_x), rng)
    r_mu, r_lat = s.posterior(r_x)
    _, r_obs = s.posterior(r_x, observation_noise=True)
    r_sd = r_obs if (res["interval_method"].get("sigma_source") != "latent_plus_assay_noise_floor") else \
        np.sqrt(np.asarray(r_lat) ** 2 + float(res["noise"]["noise_sd"]) ** 2)
    for lv in LEVELS:
        inside, widths, avail = [], [], []
        for i, p in enumerate(props):
            iv = next((e for e in p.get("intervals", []) if abs(float(e["coverage"]) - lv) < 1e-9), None)
            ok = bool(iv and iv.get("available") and iv.get("lo") is not None)
            avail.append(ok)
            if ok and iv is not None:
                inside.append(bool(iv["lo"] <= p_y[i] <= iv["hi"]))
                widths.append((iv["hi"] - iv["lo"]) / yrange)
        lo, hi = band_from_summary(np.asarray(r_mu), np.asarray(r_sd), res["interval_method"], lv)
        r_in = None if lo is None else float(np.mean((r_y >= lo) & (r_y <= hi)))
        r_w = None if lo is None else float(np.median((hi - lo) / yrange))
        method = next((e.get("method") for p in props for e in p.get("intervals", [])
                       if abs(float(e["coverage"]) - lv) < 1e-9 and e.get("available")), None)
        # Reference bands at the same proposals: the GP's own (model-based) band
        # and, at 90%, the global +/- conformal_q band.
        gp_in = []
        for i, p in enumerate(props):
            g = next((e for e in p.get("gp_intervals", []) if abs(float(e["coverage"]) - lv) < 1e-9), None)
            if g and g.get("available"):
                gp_in.append(bool(g["lo"] <= p_y[i] <= g["hi"]))
        # The plain normalized band (the fallback when the selection-aware band is
        # unavailable) at the same proposals: pred +/- q_norm * predictive sd.
        # Deterministic given the response; draws no random numbers.
        q_norm = next((float(c["q_norm"]) for c in res["interval_method"].get("levels", [])
                       if abs(float(c["coverage"]) - lv) < 1e-9 and c.get("available")), None)
        plain_in, plain_w = [], []
        for i, p in enumerate(props):
            sp = p.get("predictive_std")
            if q_norm is not None and sp is not None and np.isfinite(sp) and sp > 0:
                plain_in.append(bool(abs(p_y[i] - p["pred"]) <= q_norm * sp))
                plain_w.append(2 * q_norm * sp / yrange)
        cq = res.get("conformal_q")
        cq_in = ([bool(abs(p_y[i] - p["pred"]) <= cq) for i, p in enumerate(props)]
                 if cq is not None and abs(float(res.get("conformal_coverage", 0)) - lv) < 1e-9 else [])
        out["levels"][str(lv)] = {"inside": inside, "width": widths, "available": avail,
                                  "random_coverage": r_in, "random_width": r_w, "method": method,
                                  "gp_inside": gp_in, "conformal_q_inside": cq_in,
                                  "plain_inside": plain_in, "plain_width": plain_w}
    return out


# ------------------------------------------------------------------ summary
def _cluster_ci(per_seed: list[list[bool]], n_boot: int = 2000, seed: int = 0) -> list[float] | None:
    seeds = [s for s in per_seed if s]
    if not seeds:
        return None
    rng = np.random.default_rng(seed)
    hits = np.array([sum(s) for s in seeds], float)
    tot = np.array([len(s) for s in seeds], float)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(seeds), len(seeds))
        bs.append(hits[i].sum() / tot[i].sum())
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in SCENARIOS:
        rs = [r for r in rows if r.get("scenario") == name and "error" not in r]
        errs = [r for r in rows if r.get("scenario") == name and "error" in r]
        if not rs and not errs:
            continue
        sc: dict[str, Any] = {"n_seeds": len(rs), "n_errors": len(errs),
                              "n_proposals": int(sum(r["n_proposals"] for r in rs)),
                              "n_seeds_without_proposals": int(sum(r["n_proposals"] == 0 for r in rs)),
                              "median_seconds": round(float(np.median([r["seconds"] for r in rs])), 2) if rs else None,
                              "levels": {}}
        for lv in LEVELS:
            k = str(lv)
            per_seed = [r["levels"][k]["inside"] for r in rs]
            flat = [b for s in per_seed for b in s]
            av = [b for r in rs for b in r["levels"][k]["available"]]
            w = [x for r in rs for x in r["levels"][k]["width"]]
            rc = [r["levels"][k]["random_coverage"] for r in rs if r["levels"][k]["random_coverage"] is not None]
            rw = [r["levels"][k]["random_width"] for r in rs if r["levels"][k]["random_width"] is not None]
            methods = sorted({r["levels"][k]["method"] for r in rs if r["levels"][k]["method"]})
            gp = [b for r in rs for b in r["levels"][k].get("gp_inside", [])]
            cqi = [b for r in rs for b in r["levels"][k].get("conformal_q_inside", [])]
            plain_seed = [r["levels"][k].get("plain_inside", []) for r in rs]
            plain = [b for s in plain_seed for b in s]
            plain_w = [x for r in rs for x in r["levels"][k].get("plain_width", [])]
            sc["levels"][k] = {
                "coverage_at_proposals": round(float(np.mean(flat)), 3) if flat else None,
                "ci95_seed_cluster": _cluster_ci(per_seed),
                "n_measured": len(flat),
                "availability": round(float(np.mean(av)), 3) if av else None,
                "median_width_at_proposals": round(float(np.median(w)), 3) if w else None,
                "coverage_random_in_box": round(float(np.mean(rc)), 3) if rc else None,
                "median_width_random_in_box": round(float(np.median(rw)), 3) if rw else None,
                "methods": methods,
                "gp_posterior_coverage_at_proposals": round(float(np.mean(gp)), 3) if gp else None,
                "conformal_q_coverage_at_proposals": round(float(np.mean(cqi)), 3) if cqi else None,
                "plain_normalized_coverage_at_proposals": round(float(np.mean(plain)), 3) if plain else None,
                "plain_normalized_ci95_seed_cluster": _cluster_ci(plain_seed),
                "plain_normalized_median_width_at_proposals": round(float(np.median(plain_w)), 3)
                if plain_w else None,
            }
        out[name] = sc
    return out


def _worker(args: tuple[str, int]) -> dict[str, Any]:
    return run_one(*args)


def environment() -> dict[str, Any]:
    """Where a results file came from: commit, stack and thread counts.

    The coverage at the proposals depends on the exact proposals, and those move
    with the code and the numerical stack (VERIFY_MAIN claim 3: a file recorded
    on another stack reproduced 1/240 rows on main). Every results file carries
    this block so a rerun can be compared like for like."""
    import hashlib
    import platform
    import subprocess
    from importlib import metadata
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]

    def git(*a: str) -> str | None:
        try:
            return subprocess.run(  # noqa: S603 - fixed git argv, no shell
                ["git", *a], cwd=root, capture_output=True, text=True, check=True, timeout=10,  # noqa: S607
            ).stdout.strip()
        except subprocess.TimeoutExpired:
            return None
        except (OSError, subprocess.CalledProcessError):
            return None

    def version(dist: str) -> str | None:
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    lock = root / "uv.lock"
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {d: version(d) for d in ("torch", "botorch", "gpytorch", "numpy", "scipy",
                                             "pandas", "scikit-learn")},
        # run_one pins torch to 1 thread in every worker process.
        "cpu_count": os.cpu_count(),
        "load_avg_at_start": [round(x, 1) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
        "torch_threads_per_worker": 1,
        "env_threads": {k: os.environ.get(k) for k in ("KALOS_TORCH_THREADS", "OMP_NUM_THREADS",
                                                        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--out", default=None, help="write {summary, rows, meta} JSON here")
    args = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    env = environment()  # before the run: the commit the rows come from
    t_start = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
    jobs = [(nm, args.seed_offset + s) for nm in args.scenarios.split(",") for s in range(args.seeds)]
    if args.procs > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.procs) as pool:
            rows = pool.map(_worker, jobs, chunksize=1)
    else:
        rows = [_worker(j) for j in jobs]
    summary = summarize(rows)
    from kalos import __version__

    meta = {"seeds": args.seeds, "seed_offset": args.seed_offset, "engine_version": __version__,
            "n_random_points": N_RANDOM, "levels": list(LEVELS), "procs": args.procs,
            "command": "python -m kalos.bench.coverage_at_proposals " + " ".join(argv or sys.argv[1:]),
            "started_utc": started, "seconds_wall": round(time.time() - t_start, 1),
            "environment": env}
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "summary": summary, "rows": rows}, fh)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
