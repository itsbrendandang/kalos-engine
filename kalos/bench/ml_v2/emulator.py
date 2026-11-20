"""Emulators of real datasets (PROTOCOL.md section 3.3), Olympus-style.

Pool replay can only "run" recipes that exist in the pool. An emulator is a
model fitted to ALL rows of a real dataset, queried at any recipe in the
dataset's box, plus Gaussian noise. Two emulator kinds are fitted per dataset so
no surrogate family is favoured by construction:

- `gp`: scikit-learn GaussianProcessRegressor (constant x Matern-5/2 ARD + white
  noise) on unit-scaled inputs (one-hot categoricals); fitted on at most 1000
  rows (seeded subsample) to bound cost.
- `rf`: scikit-learn RandomForestRegressor (300 trees, min_samples_leaf 2).

Both use the same observation noise: the GP's fitted white-noise sd. The
optimum `f_star` is the max of the emulator mean over 20,000 uniform box points
plus every dataset row (approximate: an arm can exceed it, giving a negative
regret, which is reported as is). CAVEAT (stated with every emulator result):
an emulator is a model of the data, not the lab. Away from the measured rows it
reverts to its prior (GP) or to the nearest leaf (RF); in 15-29 dimensions most
of the box is far from every row. Emulator results are never pooled with
pool-replay results.
"""
from __future__ import annotations

import pickle  # noqa: S403 - local cache written and read by this module only
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .datasets import RealDataset
from .space import Problem

EMULATOR_KINDS = ("gp", "rf")
GP_MAX_ROWS = 1000
N_FSTAR = 20_000


def _design(ds: RealDataset, df: pd.DataFrame) -> np.ndarray:
    sp = ds.space()
    parts = [sp.unit(df)] if ds.continuous else []
    for c in ds.categorical:
        levels = sp.categorical[c]
        parts.append(np.stack([(df[c].astype(str) == lv).to_numpy(float) for lv in levels], 1))
    return np.hstack(parts)


class Emulator:
    def __init__(self, ds: RealDataset, kind: str):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        self.ds, self.kind = ds, kind
        x = _design(ds, ds.frame)
        y = ds.y
        self.mu, self.sd = float(y.mean()), float(y.std() or 1.0)
        z = (y - self.mu) / self.sd
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(z), GP_MAX_ROWS, replace=False)) if len(z) > GP_MAX_ROWS else np.arange(len(z))
        kern = ConstantKernel(1.0) * Matern(length_scale=np.full(x.shape[1], 0.5), length_scale_bounds=(1e-2, 1e2),
                                            nu=2.5) + WhiteKernel(0.1, (1e-5, 1.0))
        gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=2, random_state=0)
        gp.fit(x[idx], z[idx])
        self.noise_sd = float(np.sqrt(gp.kernel_.k2.noise_level)) * self.sd
        if kind == "gp":
            self.model: Any = gp
        elif kind == "rf":
            self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1)
            self.model.fit(x, z)
        else:
            raise KeyError(kind)
        sp = ds.space()
        box = sp.sample(N_FSTAR, np.random.default_rng(1))
        self.f_star = float(max(self.mean(box).max(), self.mean(ds.frame).max()))
        self.f_med = float(np.median(self.mean(sp.sample(4096, np.random.default_rng(2)))))

    def mean(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(_design(self.ds, df)), float) * self.sd + self.mu


def get_emulator(ds: RealDataset, kind: str, cache_dir: Path | None) -> Emulator:
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"emulator_{ds.name.replace('/', '__')}_{kind}.pkl"
        if path.is_file():
            with path.open("rb") as fh:
                return pickle.load(fh)  # noqa: S301 - our own cache
    em = Emulator(ds, kind)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(em, fh)
        tmp.replace(path)
    return em


def emulator_problem(ds: RealDataset, kind: str, cache_dir: Path | None) -> Problem:
    em = get_emulator(ds, kind, cache_dir)
    group = "real_emulator" if ds.group == "real" else f"{ds.group}_emulator"

    def truth(df: pd.DataFrame) -> np.ndarray:
        return em.mean(df)

    def observe(df: pd.DataFrame, r: np.random.Generator, _rounds: np.ndarray) -> np.ndarray:
        return em.mean(df) + r.normal(0, em.noise_sd, len(df))

    roles = {"categoricals": list(ds.categorical)} if ds.categorical else {}
    return Problem(f"emu-{kind}/{ds.name}", f"emu-{kind}/{ds.name}", group, ds.space(), truth, observe,
                   em.f_star, em.f_med, roles=roles, target=ds.target,
                   meta={"d": len(ds.continuous) + len(ds.categorical), "emulator": kind, "noise_sd": em.noise_sd,
                         "f_star_method": "max of emulator mean over 20k uniform points + dataset rows (approximate)"})
