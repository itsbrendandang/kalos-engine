"""Alternative-surrogate benchmark harness under the "must beat the GP" rule.

READ THIS FIRST: `kalos.core.surrogate.Surrogate` (a BoTorch SingleTaskGP) REMAINS
THE DEFAULT. This module only *reports*; nothing here changes, or is wired into, the
default model (see "Surrogate guardrail" in BENCHMARK.md and
tests/test_surrogate_guardrail.py). An alternative surrogate only ships as a default
if it beats the GP, with the same initial design and at least 5 seeds, on the six
AMII datasets - and even then the switch is a deliberate edit next to that note.

Arms (each behind the same tiny `SurrogateArm` protocol):
  - ``gp``                kalos `Surrogate`, predictive (observation-noise) mean/std.
  - ``sk_gbr_quantile``   sklearn GradientBoostingRegressor, quantile loss at
                          alpha/2, 0.5, 1-alpha/2. Always available (no extras).
  - ``lgbm_quantile``     LightGBM LGBMRegressor(objective="quantile").   optional
  - ``catboost_quantile`` CatBoostRegressor(loss "Quantile:alpha=...").   optional
  - ``ngboost``           NGBRegressor(Dist=Normal) -> mean/std.          optional
Optional arms import lazily and are reported as ``skipped: not installed`` when
`importlib.util.find_spec` cannot find their package.

Protocol (identical for every arm, so the comparison is paired):
  - For each seed, ONE list of folds is built up front (seeded KFold, or GroupKFold
    with shuffling when `groups` is given) and every arm iterates that exact list.
  - Inside each training fold a calibration split (`cal_frac`, grouped when
    `groups` is given) is held out; the arm is fit on the remaining "proper"
    training rows only. This is plain SPLIT conformal (not CV+/jackknife+): cheap,
    finite-sample valid for exchangeable rows, at the cost of fitting on fewer rows.
  - Calibration: arms with a std (gp, ngboost) use normalised residuals
    |y - mu| / sd -> interval mu +/- q * sd (via `kalos.core.conformal`); quantile
    arms use CQR (Romano et al. 2019) scores max(lo - y, y - hi) -> [lo - q, hi + q].
  - The test fold is never seen by fit or calibration. Out-of-fold predictions are
    pooled per seed, and per seed we score RMSE, Spearman, PICP at the nominal level
    (default 90%) and mean interval width; the table reports mean +/- sd over seeds.

Verdict per non-GP arm: ``beats_gp`` is True only when there are >= 5 seeds, the
arm's mean RMSE is below the GP's, AND its mean PICP is within `picp_tol` of
nominal. With fewer seeds the verdict is ``insufficient_seeds`` and ``beats_gp`` is
None. A "True" on synthetic surfaces is NOT the guardrail's evidence; only the six
AMII datasets are.
"""
from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from kalos.core.conformal import q_from_residuals

MIN_SEEDS_FOR_VERDICT = 5
GP_ARM = "gp"
DEFAULT_ARMS: tuple[str, ...] = ("gp", "sk_gbr_quantile", "lgbm_quantile", "catboost_quantile", "ngboost")


# --------------------------------------------------------------------------- #
# Arm protocol + adapters
# --------------------------------------------------------------------------- #


@dataclass
class ArmPrediction:
    """Raw (uncalibrated) prediction. Either `std` (Gaussian arms) or `lo`/`hi`
    (quantile arms at alpha/2 and 1-alpha/2) is set; `mean` is always set."""

    mean: np.ndarray
    std: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None


class SurrogateArm(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, bounds: np.ndarray) -> SurrogateArm: ...

    def predict(self, X: np.ndarray) -> ArmPrediction: ...


class GPArm:
    """The kalos default surrogate, unchanged. Predictive std (observation noise
    on), because held-out targets are noisy measurements."""

    name = "gp"

    def __init__(self, alpha: float = 0.1, seed: int = 0) -> None:
        self.seed = seed
        self._s = None

    def fit(self, X, y, bounds):
        import torch

        from kalos.core.surrogate import Surrogate

        torch.manual_seed(self.seed)
        self._s = Surrogate().fit(X, y, bounds=bounds)
        return self

    def predict(self, X):
        assert self._s is not None, "fit() first"
        mean, std = self._s.posterior(X, observation_noise=True)
        return ArmPrediction(mean=np.asarray(mean, float), std=np.asarray(std, float))


class _QuantileArm:
    """Three models at alpha/2, 0.5, 1-alpha/2; the median is the point estimate."""

    name = "quantile"

    def __init__(self, alpha: float = 0.1, seed: int = 0) -> None:
        self.alpha = alpha
        self.seed = seed
        self._models: list = []

    def _make(self, q: float):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y, bounds):
        qs = (self.alpha / 2, 0.5, 1 - self.alpha / 2)
        self._models = [self._make(q).fit(np.asarray(X, float), np.asarray(y, float)) for q in qs]
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        lo, med, hi = (np.asarray(m.predict(X), float).reshape(-1) for m in self._models)
        # Independently fit quantiles can cross; sort so lo <= hi.
        return ArmPrediction(mean=med, lo=np.minimum(lo, hi), hi=np.maximum(lo, hi))


class SkGBRQuantileArm(_QuantileArm):
    name = "sk_gbr_quantile"

    def _make(self, q):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=200, learning_rate=0.05,
            max_depth=3, min_samples_leaf=3, random_state=self.seed,
        )


class LGBMQuantileArm(_QuantileArm):
    name = "lgbm_quantile"

    def _make(self, q):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=200, learning_rate=0.05,
            num_leaves=15, min_child_samples=5, random_state=self.seed, n_jobs=1, verbose=-1,
        )


class CatBoostQuantileArm(_QuantileArm):
    name = "catboost_quantile"

    def _make(self, q):
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function=f"Quantile:alpha={q}", iterations=300, depth=4, learning_rate=0.05,
            random_seed=self.seed, verbose=0, thread_count=2, allow_writing_files=False,
        )


class NGBoostArm:
    name = "ngboost"

    def __init__(self, alpha: float = 0.1, seed: int = 0) -> None:
        self.seed = seed
        self._m = None

    def fit(self, X, y, bounds):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        self._m = NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.03,
                               verbose=False, random_state=self.seed)
        self._m.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        assert self._m is not None, "fit() first"
        dist = self._m.pred_dist(np.asarray(X, float))
        return ArmPrediction(mean=np.asarray(dist.loc, float), std=np.asarray(dist.scale, float))


# name -> (class, required package or None)
ARMS: dict[str, tuple[type, str | None]] = {
    "gp": (GPArm, None),
    "sk_gbr_quantile": (SkGBRQuantileArm, None),
    "lgbm_quantile": (LGBMQuantileArm, "lightgbm"),
    "catboost_quantile": (CatBoostQuantileArm, "catboost"),
    "ngboost": (NGBoostArm, "ngboost"),
}


def arm_availability(arms: Iterable[str] = DEFAULT_ARMS) -> dict[str, str]:
    """{arm: "available" | "skipped: not installed"}. Unknown names raise."""
    out: dict[str, str] = {}
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"unknown arm {a!r}; choose from {sorted(ARMS)}")
        pkg = ARMS[a][1]
        out[a] = "available" if pkg is None or importlib.util.find_spec(pkg) is not None else "skipped: not installed"
    return out


# --------------------------------------------------------------------------- #
# Folds + conformal + metrics
# --------------------------------------------------------------------------- #


def make_folds(n: int, seed: int, *, groups=None, n_splits: int = 5, cal_frac: float = 0.25):
    """[(proper_train, cal, test), ...] for one seed. Built once and shared by every arm."""
    idx = np.arange(n)
    if groups is not None:
        groups = np.asarray(groups)
        outer = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx, groups=groups)
    else:
        outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(idx)
    folds = []
    for k, (tr, te) in enumerate(outer):
        if groups is not None and len(np.unique(groups[tr])) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=cal_frac, random_state=seed * 1000 + k)
            fit_i, cal_i = next(gss.split(tr, groups=groups[tr]))
        else:
            perm = np.random.default_rng(seed * 1000 + k).permutation(len(tr))
            n_cal = max(2, round(cal_frac * len(tr)))
            cal_i, fit_i = perm[:n_cal], perm[n_cal:]
        folds.append((np.sort(tr[fit_i]), np.sort(tr[cal_i]), np.sort(te)))
    return folds


def _cqr_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of SIGNED scores (CQR scores can be < 0,
    so `q_from_residuals`, which takes |.|, does not apply)."""
    s = np.asarray(scores, float).reshape(-1)
    level = min(1.0, np.ceil((len(s) + 1) * (1 - alpha)) / len(s))
    return float(np.quantile(s, level))


def conformalize(cal: ArmPrediction, y_cal, test: ArmPrediction, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Split-conformal [lo, hi] on the test rows from the calibration rows."""
    y_cal = np.asarray(y_cal, float).reshape(-1)
    if cal.std is not None and test.std is not None:
        sd_cal = np.maximum(cal.std, 1e-12)
        q = q_from_residuals((y_cal - cal.mean) / sd_cal, alpha)
        sd = np.maximum(test.std, 1e-12)
        return test.mean - q * sd, test.mean + q * sd
    assert cal.lo is not None and cal.hi is not None and test.lo is not None and test.hi is not None
    q = _cqr_quantile(np.maximum(cal.lo - y_cal, y_cal - cal.hi), alpha)
    return test.lo - q, test.hi + q


def raw_interval(p: ArmPrediction, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if p.std is not None:
        z = float(norm.ppf(1 - alpha / 2))
        return p.mean - z * p.std, p.mean + z * p.std
    assert p.lo is not None and p.hi is not None
    return p.lo, p.hi


def _metrics(y, mu, lo, hi) -> dict:
    y, mu, lo, hi = (np.asarray(a, float).reshape(-1) for a in (y, mu, lo, hi))
    sp = float("nan")
    if len(y) >= 3 and np.std(mu) > 0 and np.std(y) > 0:
        sp = float(spearmanr(mu, y).statistic)
    return {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "spearman": sp,
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "width": float(np.mean(hi - lo)),
    }


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def run_surrogate_bench(
    X,
    y,
    *,
    arms: Sequence[str] = ("gp", "sk_gbr_quantile"),
    seeds: Iterable[int] = range(5),
    groups=None,
    bounds=None,
    n_splits: int = 5,
    cal_frac: float = 0.25,
    alpha: float = 0.1,
    picp_tol: float = 0.05,
    conformal: bool = True,
) -> dict:
    """Paired CV benchmark of surrogate arms on (X, y). Reports only; never
    changes the default surrogate (`kalos.core.surrogate.Surrogate` stays the GP).

    Returns {"arms": {name: summary}, "per_seed": {name: [metrics per seed]},
    "folds": {seed: [(fit, cal, test), ...]}, "fold_log": {name: [(seed, test_idx), ...]},
    "availability": {...}, "config": {...}}.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)
    seeds = list(seeds)
    if bounds is None:
        bounds = np.vstack([X.min(axis=0), X.max(axis=0)])
    avail = arm_availability(arms)
    folds = {s: make_folds(len(y), s, groups=groups, n_splits=n_splits, cal_frac=cal_frac) for s in seeds}

    per_seed: dict[str, list[dict]] = {}
    fold_log: dict[str, list[tuple[int, tuple[int, ...]]]] = {}
    errors: dict[str, str] = {}
    for name in arms:
        if avail[name] != "available":
            continue
        cls = ARMS[name][0]
        per_seed[name], fold_log[name] = [], []
        try:
            for s in seeds:
                ys, mus, los, his = [], [], [], []
                for fit_i, cal_i, te_i in folds[s]:
                    fold_log[name].append((s, tuple(int(i) for i in te_i)))
                    arm = cls(alpha=alpha, seed=s).fit(X[fit_i], y[fit_i], bounds)
                    p_te = arm.predict(X[te_i])
                    if conformal:
                        lo, hi = conformalize(arm.predict(X[cal_i]), y[cal_i], p_te, alpha)
                    else:
                        lo, hi = raw_interval(p_te, alpha)
                    ys.append(y[te_i])
                    mus.append(p_te.mean)
                    los.append(lo)
                    his.append(hi)
                per_seed[name].append(
                    {"seed": s, **_metrics(*(np.concatenate(v) for v in (ys, mus, los, his)))}
                )
        except Exception as exc:  # one broken arm must not sink the report
            errors[name] = f"error: {type(exc).__name__}: {exc}"
            per_seed.pop(name, None)

    summary: dict[str, dict] = {}
    nominal = 1 - alpha
    gp = per_seed.get(GP_ARM)
    gp_rmse = float(np.mean([r["rmse"] for r in gp])) if gp else None
    for name in arms:
        if name not in per_seed:
            summary[name] = {"status": errors.get(name, avail[name])}
            continue
        rows = per_seed[name]
        agg = {"status": "ok", "n_seeds": len(rows)}
        for m in ("rmse", "spearman", "picp", "width"):
            vals = np.array([r[m] for r in rows], float)
            agg[m] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            agg[f"{m}_sd"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan")
        if name != GP_ARM:
            agg.update(_verdict(rows, gp, gp_rmse, agg, nominal, picp_tol))
        summary[name] = agg

    return {
        "arms": summary,
        "per_seed": per_seed,
        "folds": folds,
        "fold_log": fold_log,
        "availability": avail,
        "config": {"seeds": seeds, "n_splits": n_splits, "cal_frac": cal_frac, "alpha": alpha,
                   "picp_tol": picp_tol, "conformal": conformal, "grouped": groups is not None,
                   "n": len(y), "d": X.shape[1]},
    }


def _verdict(rows, gp_rows, gp_rmse, agg, nominal, picp_tol) -> dict:
    if gp_rows is None or gp_rmse is None:
        return {"beats_gp": None, "verdict": "no_gp_reference"}
    wins = sum(r["rmse"] < g["rmse"] for r, g in zip(rows, gp_rows, strict=True))
    out = {"rmse_wins_vs_gp": f"{wins}/{len(rows)}"}
    if len(rows) < MIN_SEEDS_FOR_VERDICT:
        return {**out, "beats_gp": None, "verdict": "insufficient_seeds"}
    calibrated = abs(agg["picp"] - nominal) <= picp_tol
    beats = bool(agg["rmse"] < gp_rmse and calibrated)
    return {**out, "beats_gp": beats, "verdict": "beats_gp" if beats else "does_not_beat_gp"}


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


def run_synthetic(
    *,
    arms: Sequence[str] = DEFAULT_ARMS,
    seeds: Iterable[int] = range(5),
    n: int = 60,
    noise: float = 0.1,
    dim: int = 4,
    data_seed: int = 0,
    **kw,
) -> dict[str, dict]:
    """gaussian_bump(dim) and ackley(dim) with `noise` (fraction of scale). The
    design (X, y) is drawn ONCE per surface with `data_seed`, so every seed and
    every arm sees the same data; seeds vary only the folds and model seeds."""
    from kalos.bench.objectives import ackley, gaussian_bump

    out = {}
    for obj in (gaussian_bump(dim), ackley(dim)):
        rng = np.random.default_rng(data_seed)
        X = rng.uniform(obj.bounds[0], obj.bounds[1], size=(n, obj.dim))
        _, y = obj.evaluate(X, noise=noise, rng=rng)
        out[obj.name] = run_surrogate_bench(X, y, arms=arms, seeds=seeds, bounds=obj.bounds, **kw)
    return out


def run_amii(
    *, arms: Sequence[str] = DEFAULT_ARMS, seeds: Iterable[int] = range(5), data_dir=None, **kw
) -> dict[str, dict]:
    """The six AMII datasets (the guardrail's actual evidence). Returns
    {"_skipped": reason} when KALOS_AMII_DATA is not set / the CSVs are absent."""
    from kalos.bench.amii import REGISTRY, amii_data_dir, load_amii_pool

    if amii_data_dir(data_dir) is None:
        return {"_skipped": "KALOS_AMII_DATA not set (data is not redistributed)"}
    out: dict[str, dict] = {}
    for name in REGISTRY:
        try:
            X, y, _ = load_amii_pool(name, data_dir)
        except FileNotFoundError as exc:
            out[name] = {"_skipped": str(exc)}
            continue
        out[name] = run_surrogate_bench(X, y, arms=arms, seeds=seeds, **kw)
    return out


def format_table(results: dict[str, dict]) -> str:
    lines = []
    if "_skipped" in results:  # whole suite skipped (e.g. no KALOS_AMII_DATA)
        return f"amii skipped: {results['_skipped']}"
    hdr = f"{'dataset':<20} {'arm':<18} {'RMSE':>14} {'Spearman':>14} {'PICP':>14} {'width':>14}  verdict"
    for ds, res in results.items():
        if "_skipped" in res:
            lines.append(f"{ds:<20} skipped: {res['_skipped']}")
            continue
        lines.append(hdr)
        for arm, a in res["arms"].items():
            if a.get("status") != "ok":
                lines.append(f"{ds:<20} {arm:<18} {a['status']}")
                continue

            def cell(m, a=a):
                return f"{a[m]:.3f}±{a[m + '_sd']:.3f}"

            v = "(reference)" if arm == GP_ARM else f"{a['verdict']} (rmse wins {a['rmse_wins_vs_gp']})"
            lines.append(f"{ds:<20} {arm:<18} {cell('rmse'):>14} {cell('spearman'):>14} "
                         f"{cell('picp'):>14} {cell('width'):>14}  {v}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kalos.bench.surrogate_arms", description=__doc__.splitlines()[0])
    p.add_argument("--suite", choices=("synthetic", "amii", "both"), default="synthetic")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--no-conformal", action="store_true")
    a = p.parse_args(argv)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    kw = {"arms": arms, "seeds": range(a.seeds), "alpha": a.alpha, "conformal": not a.no_conformal}
    print("Surrogate remains the default GP; this harness only reports.")
    print(f"arms: {arm_availability(arms)}  nominal coverage: {1 - a.alpha:.0%}")
    if a.suite in ("synthetic", "both"):
        print(format_table(run_synthetic(n=a.n, noise=a.noise, **kw)))
    if a.suite in ("amii", "both"):
        print(format_table(run_amii(**kw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
