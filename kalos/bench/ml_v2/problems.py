"""Randomized synthetic problem families (PROTOCOL.md section 3.1).

Every family is a generator: `make(family, seed)` draws one INSTANCE (shifted
optimum, rotation, dimension, noise level, categorical effects ...) from the
seed, so a result is an average over many surfaces, not one. Each instance
knows its optimum `f_star` (exact, or by multi-start local optimization, stated
in `meta["f_star_method"]`) and `f_med` (median of the truth over the feasible
box), so normalized simple regret is (f* - f) / (f* - f_med).

Signal families (closed loop and snapshot): hartmann6, bumps, bioprocess,
heteroscedastic, mixed, constrained, clones, batch.
Null families (snapshot only; no learnable input signal): null_noise,
null_batch_column, null_batch_rounds, null_batch_hidden, null_jitter.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .space import Problem, Space, f_median

SIGNAL_FAMILIES: tuple[str, ...] = (
    "hartmann6", "bumps", "bioprocess", "heteroscedastic", "mixed", "constrained", "clones", "batch",
)
NULL_FAMILIES: tuple[str, ...] = (
    "null_noise", "null_batch_column", "null_batch_rounds", "null_batch_hidden", "null_jitter",
)
FAMILY_INDEX = {f: i for i, f in enumerate((*SIGNAL_FAMILIES, *NULL_FAMILIES))}
NOISE_FRACTIONS = (0.02, 0.10, 0.20)


def _rng(family: str, seed: int, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), FAMILY_INDEX[family], 1729, stream])


def _unit_space(d: int, prefix: str = "x") -> Space:
    return Space(continuous={f"{prefix}{i + 1}": (0.0, 1.0) for i in range(d)})


def _random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def _maximize(f: Callable[[np.ndarray], np.ndarray], d: int, starts: np.ndarray,
              constraints: list[dict[str, Any]] | None = None) -> float:
    """Multi-start local maximization on [0,1]^d (SLSQP when constrained)."""
    best = float(np.max(f(starts)))
    for x0 in starts:
        res = minimize(lambda x: -float(f(x[None])[0]), x0, bounds=[(0.0, 1.0)] * d,  # type: ignore[call-overload]
                       method="SLSQP" if constraints else "L-BFGS-B", constraints=constraints or ())
        if res.success or not constraints:
            x = np.clip(res.x, 0, 1)
            if constraints and any(c["fun"](x) < -1e-7 for c in constraints):
                continue
            best = max(best, float(f(x[None])[0]))
    return best


# ------------------------------------------------------------------ hartmann
_H6_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
_H6_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
H6_MAX = 3.32237


def _h6(x: np.ndarray) -> np.ndarray:
    inner = ((x[:, None, :] - _H6_P[None]) ** 2 * _H6_A[None]).sum(-1)
    return (_H6_ALPHA[None] * np.exp(-inner)).sum(-1)


def _hartmann6(seed: int) -> Problem:
    rng = _rng("hartmann6", seed)
    big_d = int(rng.choice([6, 10]))
    active = np.sort(rng.choice(big_d, 6, replace=False))
    perm = rng.permutation(6)
    flip = rng.random(6) < 0.5
    noise = float(rng.choice(NOISE_FRACTIONS))
    space = _unit_space(big_d)
    names = list(space.continuous)

    def truth(df: pd.DataFrame) -> np.ndarray:
        x = df[names].to_numpy(float)[:, active][:, perm]
        x = np.where(flip, 1.0 - x, x)
        return _h6(x)

    f_med = f_median(space, truth, seed)
    sd = noise * (H6_MAX - f_med)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        return truth(df) + r.normal(0, sd, len(df))

    return Problem(f"hartmann6/s{seed}", "hartmann6", "synthetic", space, truth, observe, H6_MAX, f_med,
                   meta={"d": big_d, "active_dims": active.tolist(), "noise_frac": noise, "noise_sd": sd,
                         "f_star_method": "exact (Hartmann-6 maximum 3.32237)"})


# ------------------------------------------------------------------- bumps
def _bump_mixture(d: int, k: int, rng: np.random.Generator) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    centers = rng.uniform(0.15, 0.85, (k, d))
    amps = np.r_[1.0, rng.uniform(0.4, 0.85, k - 1)]
    precs = []
    for _ in range(k):
        rot = _random_rotation(d, rng)
        w = rng.uniform(0.12, 0.35, d) * np.sqrt(d / 2.0) ** 0.5
        precs.append(rot @ np.diag(1.0 / w ** 2) @ rot.T)
    prec = np.array(precs)

    def f(x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - centers[None]
        q = np.einsum("nki,kij,nkj->nk", diff, prec, diff)
        return 10.0 * (amps[None] * np.exp(-0.5 * q)).sum(1)

    return f, centers


def _bumps_core(family: str, seed: int, dims: tuple[int, ...]) -> tuple[Space, Callable, np.ndarray, dict, np.random.Generator]:
    rng = _rng(family, seed)
    d = int(rng.choice(dims))
    k = int(rng.integers(1, 4))
    f, centers = _bump_mixture(d, k, rng)
    space = _unit_space(d)
    meta = {"d": d, "n_bumps": k}
    return space, f, centers, meta, rng


def _bumps(seed: int) -> Problem:
    space, f, centers, meta, rng = _bumps_core("bumps", seed, (2, 4, 6, 10))
    d = meta["d"]
    names = list(space.continuous)
    noise = float(rng.choice(NOISE_FRACTIONS))
    starts = np.vstack([centers, rng.random((8, d))])
    f_star = _maximize(f, d, starts)

    def truth(df: pd.DataFrame) -> np.ndarray:
        return f(df[names].to_numpy(float))

    f_med = f_median(space, truth, seed)
    sd = noise * (f_star - f_med)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        return truth(df) + r.normal(0, sd, len(df))

    return Problem(f"bumps/s{seed}", "bumps", "synthetic", space, truth, observe, f_star, f_med,
                   meta={**meta, "noise_frac": noise, "noise_sd": sd,
                         "f_star_method": "multi-start L-BFGS-B from every bump center + 8 random starts"})


# --------------------------------------------------------------- bioprocess
BIO_SPECS = {
    "pH": (6.0, 7.6), "temperature_C": (30.0, 39.0), "glucose_gL": (2.0, 12.0),
    "glutamine_mM": (1.0, 8.0), "feed_rate_mLh": (0.1, 2.0), "DO_pct": (20.0, 80.0),
    "agitation_rpm": (100.0, 400.0), "inoculum_1e6": (0.2, 2.0), "antifoam_mgL": (0.0, 50.0),
    "osmolality_mOsm": (280.0, 360.0),
}


def _g(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def _bioprocess(seed: int) -> Problem:
    """The examples/synthetic_bioprocess generator with its optimum shifted:
    zero-inflated (non-viable region gives 0), multiplicative noise."""
    rng = _rng("bioprocess", seed)
    c = {"pH": rng.uniform(6.7, 7.2), "temperature_C": rng.uniform(34.5, 37.5), "DO_pct": rng.uniform(40, 62),
         "glutamine_mM": rng.uniform(3.0, 6.0), "feed_rate_mLh": rng.uniform(0.8, 1.5)}
    cv = float(rng.choice((0.06, 0.12, 0.20)))
    space = Space(continuous=dict(BIO_SPECS))

    def truth(df: pd.DataFrame) -> np.ndarray:
        d = {k: df[k].to_numpy(float) for k in BIO_SPECS}
        t = (3.2 * _g(d["pH"], c["pH"], 0.35) * _g(d["temperature_C"], c["temperature_C"], 1.6)
             * _g(d["DO_pct"], c["DO_pct"], 14.0) * (d["glucose_gL"] / (d["glucose_gL"] + 3.0))
             * (0.6 + 0.4 * _g(d["glutamine_mM"], c["glutamine_mM"], 2.0))
             * (0.7 + 0.3 * _g(d["feed_rate_mLh"], c["feed_rate_mLh"], 0.5)))
        viable = ((d["pH"] >= 6.25) & (d["pH"] <= 7.45) & (d["temperature_C"] < 38.4)
                  & (d["osmolality_mOsm"] < 352) & (d["DO_pct"] > 24))
        return np.where(viable, t, 0.0)

    f_star = 3.2 * 12.0 / 15.0  # every bump at its center (inside the viable region), glucose at 12
    f_med = f_median(space, truth, seed)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        f = truth(df)
        return np.clip(np.where(f > 0, f * r.normal(1.0, cv, len(f)), 0.0), 0, None)

    return Problem(f"bioprocess/s{seed}", "bioprocess", "synthetic", space, truth, observe, f_star, f_med,
                   meta={"d": 10, "centers": c, "noise_cv": cv, "f_star_method": "exact (2.56)"})


# ----------------------------------------------------------- heteroscedastic
def _heteroscedastic(seed: int) -> Problem:
    rng = _rng("heteroscedastic", seed)
    d = int(rng.choice([2, 3, 5]))
    center = rng.uniform(0.2, 0.8, d)
    beta = rng.uniform(4.0, 10.0)
    b = float(rng.choice((0.1, 0.2, 0.3)))
    space = _unit_space(d)
    names = list(space.continuous)

    def truth(df: pd.DataFrame) -> np.ndarray:
        x = df[names].to_numpy(float)
        return 5.0 * np.exp(-beta * ((x - center) ** 2).sum(1))

    f_med = f_median(space, truth, seed)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        f = truth(df)
        return f + r.normal(0, 1, len(f)) * (0.05 + b * f)

    return Problem(f"heteroscedastic/s{seed}", "heteroscedastic", "synthetic", space, truth, observe, 5.0, f_med,
                   meta={"d": d, "noise": f"sd = 0.05 + {b} * f", "f_star_method": "exact (5.0)"})


# -------------------------------------------------------------------- mixed
def _mixed(seed: int) -> Problem:
    """3 continuous inputs + a strain (3-5 levels) + an inducer (3 levels); each
    strain has its own offset, peak location and width (bioprocess_like, randomized)."""
    rng = _rng("mixed", seed)
    n_s = int(rng.integers(3, 6))
    off = rng.uniform(0.0, 2.0, n_s)
    gain = rng.uniform(0.0, 2.5, 3)
    centers = rng.uniform(0.15, 0.85, (n_s, 3))
    widths = rng.uniform(0.15, 0.35, n_s)
    noise = float(rng.choice(NOISE_FRACTIONS))
    strains = tuple(f"S{i + 1}" for i in range(n_s))
    inducers = ("I1", "I2", "I3")
    space = Space(continuous={f"u{i + 1}": (0.0, 1.0) for i in range(3)},
                  categorical={"strain": strains, "inducer": inducers})

    def truth(df: pd.DataFrame) -> np.ndarray:
        u = df[["u1", "u2", "u3"]].to_numpy(float)
        s = df["strain"].map({v: i for i, v in enumerate(strains)}).to_numpy(int)
        i = df["inducer"].map({v: j for j, v in enumerate(inducers)}).to_numpy(int)
        d2 = (((u - centers[s]) / widths[s, None]) ** 2).sum(1)
        return off[s] + gain[i] + 6.0 * np.exp(-0.5 * d2)

    f_star = float(off.max() + gain.max() + 6.0)
    f_med = f_median(space, truth, seed)
    sd = noise * (f_star - f_med)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        return truth(df) + r.normal(0, sd, len(df))

    return Problem(f"mixed/s{seed}", "mixed", "synthetic", space, truth, observe, f_star, f_med,
                   roles={"categoricals": ["strain", "inducer"]},
                   meta={"d": 5, "n_strains": n_s, "noise_frac": noise, "noise_sd": sd,
                         "f_star_method": "exact (max offset + max gain + 6)"})


# -------------------------------------------------------------- constrained
def _constrained(seed: int) -> Problem:
    """A bump mixture with one linear constraint that cuts off the unconstrained optimum."""
    space, f, centers, meta, rng = _bumps_core("constrained", seed, (3, 5))
    d = meta["d"]
    names = list(space.continuous)
    starts = np.vstack([centers, rng.random((8, d))])
    xs = [minimize(lambda x: -float(f(x[None])[0]), x0, bounds=[(0, 1)] * d,  # type: ignore[call-overload]
                   method="L-BFGS-B").x
          for x0 in starts]
    x_opt = np.clip(xs[int(np.argmax([f(np.atleast_2d(x))[0] for x in xs]))], 0, 1)
    k = int(rng.integers(2, min(3, d) + 1))
    sub = np.sort(rng.choice(d, k, replace=False))
    tot = float(x_opt[sub].sum())
    if tot >= k / 2:
        op, rhs = "<=", tot - 0.15 * k
        cons = [{"type": "ineq", "fun": lambda x, s=sub, r=rhs: r - x[s].sum()}]
    else:
        op, rhs = ">=", tot + 0.15 * k
        cons = [{"type": "ineq", "fun": lambda x, s=sub, r=rhs: x[s].sum() - r}]
    space.linear_constraints = [{"coefficients": {names[i]: 1.0 for i in sub}, "op": op, "rhs": round(rhs, 6)}]
    feas_starts = space.sample(12, rng).to_numpy(float)
    f_star = _maximize(f, d, feas_starts, cons)
    noise = float(rng.choice(NOISE_FRACTIONS))

    def truth(df: pd.DataFrame) -> np.ndarray:
        return f(df[names].to_numpy(float))

    f_med = f_median(space, truth, seed)
    sd = noise * (f_star - f_med)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        return truth(df) + r.normal(0, sd, len(df))

    return Problem(f"constrained/s{seed}", "constrained", "synthetic", space, truth, observe, f_star, f_med,
                   meta={**meta, "constraint": space.linear_constraints[0], "noise_frac": noise, "noise_sd": sd,
                         "f_star_method": "multi-start SLSQP from 12 feasible random starts"})


# ------------------------------------------------------------------- clones
def _clones(seed: int) -> Problem:
    """Three production clones (numeric-looking IDs) with offsets and clone x
    process interaction (clone-specific optimum shift); a passage covariate with a
    small effect. Historical runs use every clone; proposals are pinned to one."""
    rng = _rng("clones", seed)
    d = int(rng.choice([3, 4]))
    ids = tuple(str(v) for v in rng.choice(np.arange(100, 999), 3, replace=False))
    target_clone = ids[int(rng.integers(0, 3))]
    off = dict(zip(ids, rng.uniform(0.0, 1.5, 3)))
    c0 = rng.uniform(0.25, 0.75, d)
    shift = {c: np.clip(c0 + rng.normal(0, 0.08, d), 0.15, 0.85) for c in ids}
    w = rng.uniform(0.18, 0.35)
    noise = float(rng.choice(NOISE_FRACTIONS))
    space = Space(continuous={f"x{i + 1}": (0.0, 1.0) for i in range(d)}, categorical={"clone": ids},
                  fixed={"clone": target_clone}, covariates={"passage": (5.0, 25.0)})
    names = list(space.continuous)
    amp = 5.0

    def truth(df: pd.DataFrame) -> np.ndarray:
        x = df[names].to_numpy(float)
        cl = df["clone"].astype(str).to_numpy()
        mu = np.array([shift[c] for c in cl])
        o = np.array([off[c] for c in cl])
        return o + amp * np.exp(-0.5 * (((x - mu) / w) ** 2).sum(1))

    f_star = float(off[target_clone] + amp)
    f_med = f_median(space, truth, seed)
    sd = noise * (f_star - f_med)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        pas = df["passage"].to_numpy(float) if "passage" in df else np.full(len(df), 15.0)
        return truth(df) - 0.02 * amp * (pas - 15.0) / 10.0 + r.normal(0, sd, len(df))

    return Problem(f"clones/s{seed}", "clones", "synthetic", space, truth, observe, f_star, f_med,
                   roles={"strain": "clone", "strain_policy": "fixed", "strain_level": target_clone,
                          "covariates": ["passage"]},
                   meta={"d": d + 1, "clones": list(ids), "target_clone": target_clone, "noise_frac": noise,
                         "noise_sd": sd, "f_star_method": "exact (offset of the target clone + 5)",
                         "regret_at": "passage 15 (covariate effect excluded from truth)"})


# -------------------------------------------------------------------- batch
def _batch(seed: int) -> Problem:
    """A bump mixture measured in rounds; every round (batch) adds its own offset.
    The round index reaches the arm as `RoundContext.round_index`."""
    space, f, centers, meta, rng = _bumps_core("batch", seed, (3, 6))
    d = meta["d"]
    names = list(space.continuous)
    f_star = _maximize(f, d, np.vstack([centers, rng.random((6, d))]))
    noise = float(rng.choice((0.02, 0.10)))
    bsd = float(rng.choice((0.1, 0.3)))

    def truth(df: pd.DataFrame) -> np.ndarray:
        return f(df[names].to_numpy(float))

    f_med = f_median(space, truth, seed)
    scale = f_star - f_med
    offsets = rng.normal(0, bsd * scale, 64)

    def observe(df: pd.DataFrame, r: np.random.Generator, rounds: np.ndarray) -> np.ndarray:
        rr = np.asarray(rounds, int) % 64
        return truth(df) + offsets[rr] + r.normal(0, noise * scale, len(df))

    return Problem(f"batch/s{seed}", "batch", "synthetic", space, truth, observe, f_star, f_med,
                   meta={**meta, "noise_frac": noise, "batch_sd_frac": bsd,
                         "f_star_method": "multi-start L-BFGS-B (offsets excluded from truth)"})


# -------------------------------------------------------------------- nulls
def null_design(family: str, seed: int, n: int) -> tuple[pd.DataFrame, np.ndarray, Problem]:
    """A null scenario's sheet (inputs + `y`), its round index and the problem.

    No input carries signal; any `trusted` verdict is false trust."""
    rng = _rng(family, seed, 1)
    d = int(rng.choice([3, 6])) if family == "null_noise" else 3
    space = _unit_space(d)
    names = list(space.continuous)
    rounds = np.zeros(n, int)
    extra: dict[str, Any] = {}
    if family == "null_noise":
        x = rng.uniform(0, 1, (n, d))
        y = rng.normal(5, 1, n)
    elif family.startswith("null_batch"):
        k = max(3, n // 8)
        sizes = np.full(k, n // k)
        sizes[: n - sizes.sum()] += 1
        xs, ys = [], []
        for b, m in enumerate(sizes):
            c = rng.uniform(0.15, 0.85, d)
            xs.append(np.clip(c + rng.normal(0, 0.12, (m, d)), 0, 1))
            ys.append(rng.normal(0, 1.0) + rng.normal(0, 0.2, m))
            rounds[sizes[:b].sum(): sizes[:b].sum() + m] = b
        x, y = np.vstack(xs), np.concatenate(ys)
        if family == "null_batch_column":
            extra["batch"] = [f"B{b + 1}" for b in rounds]
    elif family == "null_jitter":
        m = max(3, n // 4)
        base = rng.uniform(0, 1, (m, d))
        idx = np.arange(n) % m
        x = base[idx] + rng.normal(0, 0.005, (n, d))
        y = rng.normal(0, 1.0, m)[idx] + rng.normal(0, 0.2, n)
    else:
        raise KeyError(family)
    df = pd.DataFrame(np.clip(x, 0, 1).round(6), columns=names)
    for col, vals in extra.items():
        df[col] = vals
    df["y"] = np.round(y, 6)
    sd_new = float(np.std(y)) or 1.0

    def truth(frame: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(frame))

    def observe(frame: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        return float(np.mean(y)) + r.normal(0, sd_new, len(frame))

    roles: dict[str, Any] = {}
    prob = Problem(f"{family}/s{seed}", family, "synthetic", space, truth, observe, 0.0, 0.0, roles=roles,
                   meta={"d": d, "n": n, "round_attr": family == "null_batch_rounds"}, signal=False)
    return df, rounds, prob


MAKERS: dict[str, Callable[[int], Problem]] = {
    "hartmann6": _hartmann6, "bumps": _bumps, "bioprocess": _bioprocess, "heteroscedastic": _heteroscedastic,
    "mixed": _mixed, "constrained": _constrained, "clones": _clones, "batch": _batch,
}


def make(family: str, seed: int) -> Problem:
    """One instance of a signal family."""
    if family not in MAKERS:
        raise KeyError(f"unknown family {family!r}; expected one of {sorted(MAKERS)}")
    return MAKERS[family](int(seed))
