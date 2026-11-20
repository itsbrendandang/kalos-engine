"""Design spaces and problems shared by every suite (synthetic, emulator, pool)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

LIN_TOL = 1e-3  # amendment A2


@dataclass
class Space:
    """What a scientist would declare for a campaign.

    `continuous` and `categorical` are the inputs an optimizer may set (ordered).
    `fixed` pins a categorical to one level in every proposal (a production clone).
    `covariates` are recorded run properties (passage): drawn by the harness for
    every run, never proposed. `linear_constraints` use the engine's setup format
    `{"coefficients": {name: c}, "op": "<=" | ">=", "rhs": r}` over continuous inputs.
    """

    continuous: dict[str, tuple[float, float]]
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fixed: dict[str, str] = field(default_factory=dict)
    covariates: dict[str, tuple[float, float]] = field(default_factory=dict)
    linear_constraints: list[dict[str, Any]] = field(default_factory=list)

    @property
    def inputs(self) -> list[str]:
        return [*self.continuous, *self.categorical]

    @property
    def dim(self) -> int:
        return len(self.continuous) + len(self.categorical)

    def lo_hi(self) -> tuple[np.ndarray, np.ndarray]:
        b = np.array(list(self.continuous.values()), float).reshape(-1, 2)
        return b[:, 0], b[:, 1]

    def feasible(self, df: pd.DataFrame, tol: float = 1e-9, lin_tol: float = LIN_TOL) -> np.ndarray:
        """Box (relative `tol`), linear constraints (`lin_tol` x (1 + |rhs|): recipes
        are reported rounded, so a proposal ON an active constraint can miss it in
        the last displayed digit; amendment A2), levels and pinned levels."""
        ok = np.ones(len(df), bool)
        lo, hi = self.lo_hi()
        names = list(self.continuous)
        if names:
            x = df[names].to_numpy(float)
            span = np.where(hi - lo > 0, hi - lo, 1.0)
            ok &= ((x >= lo - tol * span) & (x <= hi + tol * span)).all(1)
        for c in self.linear_constraints:
            lhs = sum(float(v) * df[k].to_numpy(float) for k, v in c["coefficients"].items())
            scale = 1.0 + abs(float(c["rhs"]))
            ok &= (lhs <= c["rhs"] + lin_tol * scale) if c["op"] == "<=" else (lhs >= c["rhs"] - lin_tol * scale)
        for k, levels in self.categorical.items():
            ok &= df[k].astype(str).isin(levels).to_numpy()
        for k, v in self.fixed.items():
            ok &= (df[k].astype(str) == v).to_numpy()
        return np.asarray(ok, bool)

    def _frame(self, u: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
        lo, hi = self.lo_hi()
        df = pd.DataFrame(lo + u[:, : len(lo)] * (hi - lo), columns=list(self.continuous))
        for k, levels in self.categorical.items():
            df[k] = [levels[i] for i in rng.integers(0, len(levels), len(df))]
        for k, v in self.fixed.items():
            df[k] = v
        return df

    def sample(self, n: int, rng: np.random.Generator, *, lhs: bool = False,
               fixed: bool = True) -> pd.DataFrame:
        """n feasible points (uniform or Latin hypercube, rejection for constraints).

        `fixed=False` draws the pinned categorical uniformly too (the historical
        runs of a campaign used every clone, the proposals use one)."""
        d = len(self.continuous)
        out: list[pd.DataFrame] = []
        need, tries = n, 0
        while need > 0:
            m = max(need * (4 if self.linear_constraints else 1), need)
            if lhs and tries == 0:
                u = (np.argsort(rng.random((m, d)), axis=0) + rng.random((m, d))) / m
            else:
                u = rng.random((m, d))
            df = self._frame(u, rng)
            if not fixed:
                for k in self.fixed:
                    levels = self.categorical.get(k, (self.fixed[k],))
                    df[k] = [levels[i] for i in rng.integers(0, len(levels), len(df))]
            keep = self.feasible(df.assign(**{k: self.fixed[k] for k in self.fixed}))
            df = df[keep]
            if lhs and tries == 0 and len(df) > need:
                df = df.iloc[np.sort(rng.choice(len(df), need, replace=False))]
            out.append(df.head(need))
            need -= len(out[-1])
            tries += 1
            if tries > 200:
                raise RuntimeError("could not sample feasible points; constraints too tight")
        res = pd.concat(out, ignore_index=True)
        return res[[*self.continuous, *self.categorical]]

    def draw_covariates(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        return pd.DataFrame({k: rng.uniform(lo, hi, n) for k, (lo, hi) in self.covariates.items()})

    def project(self, recipe: dict[str, Any]) -> dict[str, Any]:
        """Clip continuous values into the box and apply fixed levels. Constraint
        violations are NOT repaired; they are counted by the suites."""
        out: dict[str, Any] = {}
        for k, (lo, hi) in self.continuous.items():
            v = recipe.get(k)
            try:
                fv = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                fv = float("nan")
            out[k] = float(np.clip(fv, lo, hi)) if np.isfinite(fv) else (lo + hi) / 2
        for k, levels in self.categorical.items():
            v = recipe.get(k)
            out[k] = str(v) if v is not None and str(v) in levels else levels[0]
        out.update(self.fixed)
        return out

    def unit(self, df: pd.DataFrame) -> np.ndarray:
        """Continuous inputs scaled to [0, 1] (for distances)."""
        lo, hi = self.lo_hi()
        x = df[list(self.continuous)].to_numpy(float)
        return (x - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


@dataclass
class Problem:
    """One problem instance. `truth(df)` is the noise-free objective (judged at
    reference covariates); `observe(df, rng, rounds)` is one noisy measurement
    per row (it may add round/batch effects and covariate effects)."""

    name: str
    family: str
    group: str
    space: Space
    truth: Callable[[pd.DataFrame], np.ndarray]
    observe: Callable[[pd.DataFrame, np.random.Generator, np.ndarray], np.ndarray]
    f_star: float
    f_med: float
    roles: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    signal: bool = True  # False: no learnable input signal (null scenario)
    target: str = "y"

    def norm_regret(self, f: float) -> float:
        den = self.f_star - self.f_med
        return float((self.f_star - f) / den) if den > 0 else float("nan")


def f_median(space: Space, truth: Callable[[pd.DataFrame], np.ndarray], seed: int, n: int = 4096) -> float:
    rng = np.random.default_rng([seed, 77])
    return float(np.median(truth(space.sample(n, rng))))
