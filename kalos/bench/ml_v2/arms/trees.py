"""Tree-ensemble baseline arms: random forest and XGBoost (PROTOCOL.md 6.3).

As in Bioqore's voyager-ml-pipeline (seeded ensemble members), with the
uncertainty the protocol needs to score them:

- `interval: "spread"`: mean +/- z * sd over a SEED ENSEMBLE of `members`
  models (model-based, not calibrated).
- `interval: "conformal"` (default): grouped split-conformal. K=5 folds grouped
  by recipe (identical inputs after rounding to 6 decimals; campaign round when
  the sheet has >= 3 rounds); out-of-fold normalized scores |y - mu| / max(sd, eps)
  give q at rank ceil((n + 1) * level); band = mu +/- q * max(sd, eps). Falls
  back to "unavailable" when n is too small for the rank.

Trust uses the SAME rule as the engine's verdict (grouped-CV Spearman >= 0.2, the
lower end of its 95% bootstrap interval > 0, at least 20 runs and 8 groups), so
false trust is compared like for like.

Acquisition (trees are piecewise constant, no gradients): expected improvement
with (mu, sd_ensemble) on 4096 uniform feasible candidates (pool replay: the
not-yet-run rows), batch by greedy selection with a Chebyshev exclusion radius of
0.1 in unit space. Hyperparameters are fixed here; tuning them (e.g. Bayesian
HPO as in voyager-ml-pipeline) is allowed on TUNING seeds only and the result
is frozen as an arm config.

Needs `scikit-learn` (core) and, for XGBoost, the optional `bench` extra.
Standalone arms: never shippable by themselves (PROTOCOL.md 8.2 tree rule).
"""
from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from ..adapter import ArmOutput, Band, Proposal, RoundContext, Verdict
from ..space import Space

N_CANDIDATES = 4096
EXCLUSION = 0.1
FOLDS = 5
TRUST_FLOOR, MIN_RUNS, MIN_GROUPS = 0.2, 20, 8


def _design(space: Space, df: pd.DataFrame, extra: list[str]) -> np.ndarray:
    parts = [space.unit(df)] if space.continuous else []
    for c in space.categorical:
        parts.append(np.stack([(df[c].astype(str) == lv).to_numpy(float) for lv in space.categorical[c]], 1))
    for c in extra:  # covariates, scaled by their declared range
        lo, hi = space.covariates[c]
        parts.append(((df[c].to_numpy(float) - lo) / ((hi - lo) or 1.0))[:, None])
    return np.hstack(parts) if parts else np.zeros((len(df), 0))


class TreeEnsemble:
    def __init__(self, kind: str, members: int, seed: int, params: Mapping[str, Any]):
        self.kind, self.members, self.seed, self.params = kind, members, seed, dict(params)
        self.models: list[Any] = []

    def _one(self, s: int) -> Any:
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestRegressor

            p = {"n_estimators": 200, "min_samples_leaf": 2, "max_features": 0.7, **self.params}
            return RandomForestRegressor(random_state=s, n_jobs=1, **p)
        try:
            import xgboost as xgb
        except Exception as exc:  # noqa: BLE001 - name the fix, then fail the arm call
            raise RuntimeError(
                "XGBoost is unavailable: install the `bench` extra; on macOS it also needs an OpenMP "
                "runtime (`brew install libomp`, or DYLD_FALLBACK_LIBRARY_PATH=<site-packages>/torch/lib)"
            ) from exc

        p = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 1.0, **self.params}
        return xgb.XGBRegressor(random_state=s, n_jobs=1, verbosity=0, **p)

    def fit(self, x: np.ndarray, y: np.ndarray) -> TreeEnsemble:
        self.mu_y, self.sd_y = float(np.mean(y)), float(np.std(y) or 1.0)
        z = (y - self.mu_y) / self.sd_y
        self.models = [self._one(self.seed + k).fit(x, z) for k in range(self.members)]
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([m.predict(x) for m in self.models], 0) * self.sd_y + self.mu_y
        return preds.mean(0), preds.std(0)


def _groups(df: pd.DataFrame, cols: list[str], rounds: tuple[int, ...] | None) -> np.ndarray:
    if rounds is not None and len(set(rounds)) >= 3:
        return np.asarray(rounds)
    key = df[cols].round(6).astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


class TreeArm:
    shippable = False

    def __init__(self, kind: str, config: Mapping[str, Any] | None = None):
        self.kind = kind
        self.config = dict(config or {})
        self.interval = str(self.config.get("interval", "conformal"))
        self.members = int(self.config.get("members", 5))
        self.params = dict(self.config.get("params", {}))
        self.name = str(self.config.get("name", f"{kind}_{self.interval}"))

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def propose(self, ctx: RoundContext) -> ArmOutput:
        sp = ctx.space
        cov = [c for c in sp.covariates if c in ctx.sheet]
        feats = [*sp.inputs, *cov]
        x = _design(sp, ctx.sheet, cov)
        y = ctx.sheet[ctx.target].to_numpy(float)
        groups = _groups(ctx.sheet, feats, ctx.round_index)
        seed = int(self.config.get("seed", 0))
        model = TreeEnsemble(self.kind, self.members, seed, self.params).fit(x, y)
        # grouped out-of-fold predictions: trust and conformal scores
        ug = np.unique(groups)
        k = min(FOLDS, len(ug))
        oof_mu, oof_sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
        if k >= 3:
            rng = np.random.default_rng(seed)
            fold_of = dict(zip(rng.permutation(ug), np.arange(len(ug)) % k))
            f = np.array([fold_of[g] for g in groups])
            for j in range(k):
                te = f == j
                if te.all() or not te.any():
                    continue
                m = TreeEnsemble(self.kind, self.members, seed, self.params).fit(x[~te], y[~te])
                oof_mu[te], oof_sd[te] = m.predict(x[te])
        ok = np.isfinite(oof_mu)
        rho, ci = None, None
        if ok.sum() >= 5 and np.ptp(y[ok]) > 0 and np.ptp(oof_mu[ok]) > 0:
            rho = float(spearmanr(oof_mu[ok], y[ok]).statistic)
            brng = np.random.default_rng(seed + 1)
            bs = []
            idx = np.flatnonzero(ok)
            for _ in range(400):
                b = brng.choice(idx, len(idx))
                if np.ptp(y[b]) > 0 and np.ptp(oof_mu[b]) > 0:
                    bs.append(spearmanr(oof_mu[b], y[b]).statistic)
            ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else None
        trusted = bool(rho is not None and ci is not None and rho >= TRUST_FLOOR and ci[0] > 0
                       and len(y) >= MIN_RUNS and len(ug) >= MIN_GROUPS)
        eps = 1e-6 * (float(np.std(y)) or 1.0)
        scores = np.abs(y[ok] - oof_mu[ok]) / np.maximum(oof_sd[ok], eps)
        n_cal = len(scores)

        def q_for(level: float) -> float | None:
            if self.interval == "spread":
                return float(norm.ppf(0.5 + level / 2))
            r = int(np.ceil((n_cal + 1) * level))
            return None if r > n_cal or n_cal == 0 else float(np.sort(scores)[r - 1])

        qs = {lv: q_for(lv) for lv in ctx.levels}

        def mu_sd(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            mu, sd = model.predict(_design(sp, df, cov))
            return mu, np.maximum(sd, eps)

        def band_at(df: pd.DataFrame, level: float) -> tuple[np.ndarray, np.ndarray] | None:
            q = qs.get(level)
            if q is None:
                return None
            mu, sd = mu_sd(df)
            return mu - q * sd, mu + q * sd

        # acquisition: EI on random feasible candidates (or the pool)
        crng = np.random.default_rng(ctx.arm_seed)
        cands = ctx.candidates if ctx.candidates is not None else sp.sample(N_CANDIDATES, crng)
        cands = cands.reset_index(drop=True)
        cfull = cands.copy()
        for c in cov:
            cfull[c] = float(np.median(ctx.sheet[c]))
        mu, sd = mu_sd(cfull)
        best = float(np.max(y))
        z = (mu - best) / sd
        ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
        cu = sp.unit(cands) if sp.continuous else np.zeros((len(cands), 0))
        order = np.argsort(-ei)
        picks: list[int] = []
        for i in order:
            if len(picks) == ctx.q:
                break
            if picks and sp.continuous:
                dist = np.abs(cu[picks] - cu[i]).max(1)
                same_cat = np.ones(len(picks), bool)
                for c in sp.categorical:
                    same_cat &= cands[c].to_numpy()[picks].astype(str) == str(cands[c].iloc[i])
                if ((dist < EXCLUSION) & same_cat).any():
                    continue
            picks.append(int(i))
        props = []
        for i in picks:
            bands = {}
            for lv in ctx.levels:
                q = qs.get(lv)
                bands[lv] = Band(float(mu[i] - q * sd[i]), float(mu[i] + q * sd[i])) if q is not None else Band(None, None)
            props.append(Proposal({k: cands[k].iloc[i] for k in sp.inputs}, float(mu[i]), bands,
                                  "trusted" if trusted else "exploration_untrusted"))
        verdict = Verdict(trusted, "reliable" if trusted else "not_yet", False, rho,
                          None if ci is None else bool(rho is not None and rho >= TRUST_FLOOR and ci[0] > 0))
        return ArmOutput(props, verdict, band_at=band_at, predict=lambda df: mu_sd(df)[0],
                         info={"n_calibration": n_cal, "cv_ci95": ci, "interval": self.interval,
                               "members": self.members})


def make_rf(config: Mapping[str, Any] | None = None) -> TreeArm:
    return TreeArm("rf", config)


def make_xgb(config: Mapping[str, Any] | None = None) -> TreeArm:
    return TreeArm("xgb", config)
