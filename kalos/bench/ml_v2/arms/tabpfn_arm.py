"""TabPFN as a surrogate: a STANDALONE, information-only arm (never shippable).

    --arm kalos.bench.ml_v2.arms.tabpfn_arm:make_arm

Licence check (2026-11-20, PyPI sdists and the Hugging Face model cards):

- tabpfn 0.1.x (TabPFN v1): classifier only; no regressor, so not usable here.
- tabpfn 2.0.x-2.2.x and 6.x-8.x: CODE under the "Prior Labs License" =
  Apache-2.0 plus a paragraph 10 attribution requirement ("Built with
  PriorLabs-TabPFN" displayed prominently on the product's website/UI/docs;
  "TabPFN" prefixed to the name of any distributed model trained on its outputs).
- tabpfn >= 9.0: CODE under plain Apache-2.0.
- WEIGHTS are licensed separately from the code. TabPFN-2 weights
  (Prior-Labs/TabPFN-v2-reg): Prior Labs License 1.1 (Apache-2.0 + the same
  attribution paragraph), not gated. TabPFN-2.5 / 2.6 / 3 / 3.5 / 3.5-Fast
  weights (the defaults from tabpfn 6.0 on): NON-COMMERCIAL licences, gated
  behind a licence acceptance; commercial use needs a Prior Labs agreement.

This arm therefore uses tabpfn 10.2.0 (Apache-2.0 code) with the TabPFN-2
weights ONLY (`ModelVersion.V2`), the one combination whose licence allows
commercial use, and only with the attribution paragraph honoured. It is NOT
adoptable into Kalos as is: it is a standalone arm (no engine patch, no
gradients for the engine's acquisition), the attribution clause would bind the
product, and the dependency is not in the engine.

Model: TabPFN-2 regressor (CPU, `n_estimators` 4 by default), predictive mean
and sd = (q84 - q16) / 2 from its predicted quantiles. Everything else (grouped
split-conformal bands, trust rule, EI on 4,096 random feasible candidates or the
pool, greedy batch with exclusion radius) is the tree baselines' code
(`arms/trees.py`), so TabPFN is compared like for like with RF/XGBoost.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from ..adapter import ArmOutput, RoundContext
from . import trees


class _TabPFNModel:
    """Duck-types `trees.TreeEnsemble` (fit(x, y) -> self; predict(x) -> (mu, sd))."""

    def __init__(self, kind: str, members: int, seed: int, params: Mapping[str, Any]):
        self.seed = int(seed)
        self.n_estimators = int(dict(params).get("n_estimators", 4))

    def fit(self, x: np.ndarray, y: np.ndarray) -> _TabPFNModel:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        self.r = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device="cpu", random_state=self.seed, n_estimators=self.n_estimators)
        self.const = None
        if np.ptp(y) == 0:
            self.const = float(y[0])
            return self
        self.r.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.const is not None:
            return np.full(len(x), self.const), np.full(len(x), 1e-6)
        mu_all, sd_all = [], []
        for i in range(0, len(x), 2048):
            o = self.r.predict(x[i:i + 2048], output_type="full", quantiles=[0.158655, 0.841345])
            mu_all.append(np.asarray(o["mean"], float))
            sd_all.append((np.asarray(o["quantiles"][1], float) - np.asarray(o["quantiles"][0], float)) / 2.0)
        return np.concatenate(mu_all), np.maximum(np.concatenate(sd_all), 1e-9)


class TabPFNArm(trees.TreeArm):
    shippable = False

    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = {"name": "tabpfn_v2_conformal", "members": 1, **dict(config or {})}
        super().__init__("tabpfn", cfg)

    @contextlib.contextmanager
    def _swap(self) -> Iterator[None]:
        old = trees.TreeEnsemble
        trees.TreeEnsemble = _TabPFNModel  # type: ignore[misc, assignment]
        try:
            yield
        finally:
            trees.TreeEnsemble = old  # type: ignore[misc]

    def propose(self, ctx: RoundContext) -> ArmOutput:
        import torch

        torch.set_num_threads(1)
        with self._swap():
            out = super().propose(ctx)
        out.info["model"] = "TabPFN-2 regressor weights (Prior Labs License 1.1), tabpfn code Apache-2.0"
        return out


def make_arm(config: Mapping[str, Any] | None = None) -> TabPFNArm:
    return TabPFNArm(config)
