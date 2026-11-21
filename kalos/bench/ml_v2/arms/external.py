"""External baselines (standalone, never shippable; PROTOCOL.md sections 2 and 9).

    --arm kalos.bench.ml_v2.arms.external:make_ax        # Ax 1.3 Client, its default ("fast") strategy
    --arm kalos.bench.ml_v2.arms.external:make_baybe     # BayBE defaults (needs `baybe`; separate venv)
    --arm kalos.bench.ml_v2.arms.external:make_gp_ucb    # voyager-brain GP+UCB, configuration below

What every external arm is given (the same as a user would declare): the
declared box of every continuous input, the categorical levels, the linear
constraints, the pinned (production) clone and the covariates. Kalos, by
contrast, searches the OBSERVED range of each input (no campaign setup is passed
unless the task has linear constraints): a difference stated with every result.

- Ax (Meta, MIT): `Client`, `configure_generation_strategy()` defaults (method
  "fast": a center point + Sobol initialization of 5 trials, then the BoTorch
  modular model, SingleTaskGP or MixedSingleTaskGP with qLogNoisyEI), with
  `use_existing_trials_for_initialization=True` so the n0 attached runs replace the
  Sobol stage. Covariates are range parameters fixed at the observed median in
  every proposal (`fixed_parameters`), as is the pinned clone. Intervals:
  `Client.predict` returns (mean, sem) of the modelled mean only, so the bands
  are mean +/- z sem: POSTERIOR bands (labelled), expected to under-cover
  fresh measurements. They are reported, never compared with Kalos's predictive bands.
- BayBE (Merck, Apache-2.0): `Campaign` with the default recommender
  (TwoPhaseMetaRecommender: random until data exist, then BotorchRecommender with
  its default surrogate and acquisition). Categoricals one-hot; the pinned clone
  as a `TaskParameter` with `active_values=[clone]` (BayBE's transfer-learning
  device); covariates dropped (BayBE has no held covariate); linear constraints as
  `ContinuousLinearConstraint`. Pool replay: a discrete search space of the unrun
  rows. Bands from `posterior_stats` (mean, std; posterior, labelled).
- GP+UCB as in voyager-brain (`external/voyager-brain/app.py`, unlicensed code,
  re-implemented here from its documented settings, no code copied):
  RobustScaler on inputs; sklearn GP C(1, (1e-3, 1e3)) x Matern(nu = 2.5, one
  isotropic length scale 1, bounds (1e-2, 1e2)) + WhiteKernel(1e-6),
  normalize_y, 10 optimizer restarts, random_state 42; 1,000 Latin-hypercube
  candidates (seed 42) inside the OBSERVED range of each input; UCB kappa = 2;
  a batch is built the way its "suggest" button adds points: the chosen point is
  appended with the believed value mu + 0.8 sd (its k_std slider default) and the
  GP refit, q times. Categorical inputs are integer codes (voyager-brain has none).
  Bands: mu +/- z sd with the WhiteKernel noise included (predictive).
"""
from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..adapter import ArmOutput, Band, Proposal, RoundContext, Verdict


def _z(level: float) -> float:
    return float(norm.ppf(0.5 + level / 2))


def _bands(mu: np.ndarray, sd: np.ndarray, levels: tuple[float, ...]) -> list[dict[float, Band]]:
    return [{lv: Band(float(m - _z(lv) * s), float(m + _z(lv) * s)) for lv in levels} for m, s in zip(mu, sd)]


class _Base:
    shippable = False
    name = "external"

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})

    def engine_patch(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


# ============================================================================ Ax
class AxArm(_Base):
    name = "ext_ax"

    def propose(self, ctx: RoundContext) -> ArmOutput:
        import logging

        from ax.api.client import Client
        from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

        logging.getLogger("ax").setLevel(logging.WARNING)
        sp = ctx.space
        params: list[Any] = []
        for k, (lo, hi) in sp.continuous.items():
            params.append(RangeParameterConfig(name=k, bounds=(float(lo), float(hi)), parameter_type="float"))
        for k, lv in sp.categorical.items():
            params.append(ChoiceParameterConfig(name=k, values=[str(v) for v in lv], parameter_type="str",
                                                is_ordered=False))
        cov_names = [c for c in sp.covariates if c in ctx.sheet]
        for k in cov_names:
            lo, hi = sp.covariates[k]
            lo = min(float(lo), float(ctx.sheet[k].min()))
            hi = max(float(hi), float(ctx.sheet[k].max()))
            params.append(RangeParameterConfig(name=k, bounds=(lo, hi if hi > lo else lo + 1.0), parameter_type="float"))
        cons = []
        for c in sp.linear_constraints:
            terms = " + ".join(f"{float(v)} * {k}" for k, v in c["coefficients"].items())
            cons.append(f"{terms} {c['op']} {float(c['rhs'])}")
        client = Client(random_seed=int(ctx.arm_seed % (2 ** 31)))
        client.configure_experiment(parameters=params, parameter_constraints=cons or None, name="kalos_bench")
        client.configure_optimization(objective=("-" if ctx.goal == "minimize" else "") + "y")
        client.configure_generation_strategy(method=str(self.config.get("method", "fast")))
        names = [*sp.continuous, *sp.categorical, *cov_names]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _, row in ctx.sheet.iterrows():
                p = {}
                for k in names:
                    v = row[k]
                    p[k] = str(v) if k in sp.categorical else float(v)
                for k, (lo, hi) in sp.continuous.items():  # attached data must lie in the declared box
                    p[k] = float(np.clip(p[k], lo, hi))
                idx = client.attach_trial(parameters=p)
                client.complete_trial(idx, raw_data={"y": float(row[ctx.target])})
            fixed: dict[str, Any] = {k: str(v) for k, v in sp.fixed.items()}
            for k in cov_names:
                fixed[k] = float(np.median(ctx.sheet[k].to_numpy(float)))
            trials = client.get_next_trials(max_trials=int(ctx.q), fixed_parameters=fixed or None)
            recs = [dict(p) for p in trials.values()]
            preds = client.predict(points=recs) if recs else []

        def predict_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            pts = []
            for _, r in df.iterrows():
                p = {k: (str(r[k]) if k in sp.categorical else float(r[k])) for k in [*sp.continuous, *sp.categorical]}
                for k in cov_names:
                    p[k] = float(r[k]) if k in r else fixed[k]
                for k, (lo, hi) in sp.continuous.items():
                    p[k] = float(np.clip(p[k], lo, hi))
                pts.append(p)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = client.predict(points=pts)
            mu = np.array([o["y"][0] for o in out], float)
            se = np.array([o["y"][1] for o in out], float)
            return mu, se

        mu = np.array([o["y"][0] for o in preds], float)
        se = np.array([o["y"][1] for o in preds], float)
        props = [Proposal({k: v for k, v in r.items() if k not in cov_names}, float(m), b, "posterior_band")
                 for r, m, b in zip(recs, mu, _bands(mu, se, ctx.levels))]

        def band_at(df: pd.DataFrame, level: float):  # type: ignore[no-untyped-def]
            m, s = predict_df(df)
            return m - _z(level) * s, m + _z(level) * s

        return ArmOutput(props, Verdict(False, "baseline"), band_at=band_at, predict=lambda df: predict_df(df)[0],
                         info={"band": "posterior (sem), not predictive"})


# ============================================================================ BayBE
class BayBEArm(_Base):
    name = "ext_baybe"

    def propose(self, ctx: RoundContext) -> ArmOutput:
        from baybe import Campaign
        from baybe.constraints import ContinuousLinearConstraint
        from baybe.objectives import SingleTargetObjective
        from baybe.parameters import (
            CategoricalParameter,
            NumericalContinuousParameter,
            NumericalDiscreteParameter,
            TaskParameter,
        )
        from baybe.searchspace import SearchSpace, SubspaceDiscrete
        from baybe.targets import NumericalTarget
        from baybe.utils.random import set_random_seed

        set_random_seed(int(ctx.arm_seed % (2 ** 31)))
        sp = ctx.space
        target = NumericalTarget(name="y", minimize=(ctx.goal == "minimize"))
        obj = SingleTargetObjective(target)
        cont = list(sp.continuous)
        cats = [c for c in sp.categorical if c not in sp.fixed]
        pinned = [c for c in sp.fixed]
        if ctx.candidates is not None and len(ctx.candidates):
            # pool replay: a discrete space of every row (run and unrun), recommend among the unrun ones
            full = pd.concat([ctx.sheet[[*cont, *sp.categorical]], ctx.candidates[[*cont, *sp.categorical]]],
                             ignore_index=True).drop_duplicates()
            for c in sp.categorical:
                full[c] = full[c].astype(str)
            params: list[Any] = []
            for c in cont:
                vals = np.unique(full[c].to_numpy(float))
                params.append(NumericalDiscreteParameter(name=c, values=tuple(float(v) for v in vals))
                              if len(vals) > 1 else NumericalDiscreteParameter(name=c, values=(float(vals[0]), float(vals[0]) + 1e-9)))
            for c in sp.categorical:
                vals = sorted(full[c].unique())
                params.append(CategoricalParameter(name=c, values=tuple(vals) if len(vals) > 1 else (vals[0], "__none__")))
            disc = SubspaceDiscrete.from_dataframe(full, parameters=params)
            space = SearchSpace(discrete=disc)
        else:
            params = [NumericalContinuousParameter(name=c, bounds=(float(lo), float(hi)))
                      for c, (lo, hi) in sp.continuous.items()]
            for c in cats:
                params.append(CategoricalParameter(name=c, values=tuple(str(v) for v in sp.categorical[c])))
            for c in pinned:
                params.append(TaskParameter(name=c, values=tuple(str(v) for v in sp.categorical[c]),
                                            active_values=[str(sp.fixed[c])]))
            cons = [ContinuousLinearConstraint(parameters=list(c["coefficients"]), operator=c["op"],
                                               coefficients=[float(v) for v in c["coefficients"].values()],
                                               rhs=float(c["rhs"])) for c in sp.linear_constraints]
            space = SearchSpace.from_product(parameters=params, constraints=cons or None)
        camp = Campaign(searchspace=space, objective=obj)
        meas = ctx.sheet[[*cont, *sp.categorical]].copy()
        for c in sp.categorical:
            meas[c] = meas[c].astype(str)
        for c, (lo, hi) in sp.continuous.items():
            if ctx.candidates is None:
                meas[c] = meas[c].astype(float).clip(lo, hi)
        meas["y"] = ctx.sheet[ctx.target].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            camp.add_measurements(meas, numerical_measurements_must_be_within_tolerance=False)
            rec = camp.recommend(batch_size=int(ctx.q))
        recs = rec[[*cont, *sp.categorical]].to_dict("records") if len(rec) else []

        def stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            d = df[[*cont, *sp.categorical]].copy()
            for c in sp.categorical:
                d[c] = d[c].astype(str)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = camp.posterior_stats(d, stats=["mean", "std"])
            return st["y_mean"].to_numpy(float), st["y_std"].to_numpy(float)

        try:
            mu, sd = stats(pd.DataFrame(recs)) if recs else (np.empty(0), np.empty(0))
        except Exception:  # noqa: BLE001 - bands are optional for a baseline
            mu, sd = np.full(len(recs), np.nan), np.full(len(recs), np.nan)
        props = [Proposal({str(k): v for k, v in r.items()}, float(m), b, "posterior_band")
                 for r, m, b in zip(recs, mu, _bands(mu, sd, ctx.levels))]

        def band_at(df: pd.DataFrame, level: float):  # type: ignore[no-untyped-def]
            m, s = stats(df)
            return m - _z(level) * s, m + _z(level) * s

        return ArmOutput(props, Verdict(False, "baseline"), band_at=band_at, predict=lambda df: stats(df)[0],
                         info={"band": "posterior std, not predictive"})


# ============================================================================ voyager-brain GP+UCB
class GPUCBArm(_Base):
    name = "ext_gp_ucb_voyager"

    def propose(self, ctx: RoundContext) -> ArmOutput:
        from scipy.stats import qmc
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel as C
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
        from sklearn.preprocessing import RobustScaler

        sp = ctx.space
        kappa = float(self.config.get("kappa", 2.0))
        k_std = float(self.config.get("k_std", 0.8))
        n_cand = int(self.config.get("n_candidates", 1000))
        cont = list(sp.continuous)
        cats = list(sp.categorical)
        covs = [c for c in sp.covariates if c in ctx.sheet]
        codes = {c: {str(v): i for i, v in enumerate(sp.categorical[c])} for c in cats}

        def enc(df: pd.DataFrame) -> np.ndarray:
            cols = [df[c].to_numpy(float) for c in cont]
            cols += [df[c].astype(str).map(codes[c]).fillna(-1).to_numpy(float) for c in cats]
            cols += [df[c].to_numpy(float) if c in df else np.full(len(df), med[c]) for c in covs]
            return np.column_stack(cols) if cols else np.zeros((len(df), 0))

        med = {c: float(np.median(ctx.sheet[c].to_numpy(float))) for c in covs}
        X = enc(ctx.sheet)
        y = ctx.sheet[ctx.target].to_numpy(float) * (-1 if ctx.goal == "minimize" else 1)
        lo, hi = X.min(0), X.max(0)  # observed range, as voyager-brain's `bounds`
        if ctx.candidates is not None and len(ctx.candidates):
            Xc_full = enc(ctx.candidates)
        else:
            u = qmc.LatinHypercube(d=X.shape[1], seed=42).random(n_cand)
            Xc_full = lo + u * (hi - lo)
            nc = len(cont)
            for j, c in enumerate(cats):
                Xc_full[:, nc + j] = np.clip(np.floor(u[:, nc + j] * len(codes[c])), 0, len(codes[c]) - 1)
            for c, lvl in sp.fixed.items():
                Xc_full[:, nc + cats.index(c)] = codes[c][str(lvl)]
            for j, c in enumerate(covs):
                Xc_full[:, nc + len(cats) + j] = med[c]
            if sp.linear_constraints:
                df_c = self._decode(Xc_full, cont, cats, sp)
                Xc_full = Xc_full[sp.feasible(df_c)]
        scaler = RobustScaler().fit(X)

        def fit(Xa: np.ndarray, ya: np.ndarray) -> GaussianProcessRegressor:
            kern = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) \
                + WhiteKernel(noise_level=1e-6)
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=10, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(scaler.transform(Xa), ya)
            return gp

        gp0 = fit(X, y)
        Xa, ya = X.copy(), y.copy()
        chosen: list[int] = []
        gp = gp0
        for _ in range(int(ctx.q)):
            if len(chosen) >= len(Xc_full):
                break
            mu, sd = gp.predict(scaler.transform(Xc_full), return_std=True)
            acq = mu + kappa * sd
            acq[chosen] = -np.inf
            i = int(np.argmax(acq))
            chosen.append(i)
            Xa = np.vstack([Xa, Xc_full[i]])
            ya = np.append(ya, mu[i] + k_std * sd[i])
            gp = fit(Xa, ya)
        Xp = Xc_full[chosen]
        recs = self._decode(Xp, cont, cats, sp).to_dict("records")
        sign = -1 if ctx.goal == "minimize" else 1

        def post(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            m, s = gp0.predict(scaler.transform(enc(df)), return_std=True)
            return sign * m, s

        mu, sd = post(pd.DataFrame(recs)) if recs else (np.empty(0), np.empty(0))
        props = [Proposal(r, float(m), b, "predictive_band") for r, m, b in zip(recs, mu, _bands(mu, sd, ctx.levels))]

        def band_at(df: pd.DataFrame, level: float):  # type: ignore[no-untyped-def]
            m, s = post(df)
            return m - _z(level) * s, m + _z(level) * s

        return ArmOutput(props, Verdict(False, "baseline"), band_at=band_at, predict=lambda df: post(df)[0],
                         info={"config": "voyager-brain app.py: Matern2.5 iso + White, UCB k=2, believer k_std 0.8"})

    @staticmethod
    def _decode(Xp: np.ndarray, cont: list[str], cats: list[str], sp: Any) -> pd.DataFrame:
        df = pd.DataFrame(Xp[:, :len(cont)], columns=cont)
        for j, c in enumerate(cats):
            lv = sp.categorical[c]
            df[c] = [str(lv[int(np.clip(round(v), 0, len(lv) - 1))]) for v in Xp[:, len(cont) + j]]
        return df


def make_ax(config: Mapping[str, Any] | None = None) -> AxArm:
    return AxArm(config)


def make_baybe(config: Mapping[str, Any] | None = None) -> BayBEArm:
    return BayBEArm(config)


def make_gp_ucb(config: Mapping[str, Any] | None = None) -> GPUCBArm:
    return GPUCBArm(config)
