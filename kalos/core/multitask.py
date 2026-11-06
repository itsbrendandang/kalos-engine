"""EXPERIMENTAL cross-campaign warm start: a BoTorch `MultiTaskGP` over
campaigns. OFF by default; nothing in the engine or portal uses it unless a
caller opts in (`warm_start_enabled()` reads `KALOS_EXPERIMENTAL_WARM_START`).

The question it answers: a tenant runs a NEW campaign on a process it has
optimized before (same factors, a new clone or site). The response surface
usually keeps its shape and shifts its level. A single-task GP fitted only on
the new campaign's first few runs knows nothing; a multi-task GP with an
intrinsic-coregionalization (ICM) kernel learns how campaigns correlate and
lends the old campaigns' shape to the new one.

Clean-room re-implementation of an idea (partial pooling across campaigns
with a three-regime evaluation) from a prior prototype; that prototype used a
scalar shrunk per-campaign offset on an sklearn GP, and named BoTorch
`MultiTaskGP` as the production path, which is what this module uses.

`MultiTaskSurrogate` exposes the same `.model` / `._X` / `posterior` surface as
`kalos.core.surrogate.Surrogate`, with `model` restricted to the TARGET
campaign's output (`output_tasks=[target]`), so `kalos.core.optimize.propose`
works on it unchanged.

The three-regime evaluation (within / new / warm-after-k) lives in
`kalos/bench/warm_start.py`. Honest limits: no evidence on real multi-campaign
data yet; a NEW campaign with zero runs cannot be predicted by task (its
offset is unknown), so the cold regime falls back to a pooled GP.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

# torch/botorch are imported inside the methods: kalos/core keeps torch out of
# `import kalos.core` (tests/test_core_torch_contract.py).

WARM_START_ENV = "KALOS_EXPERIMENTAL_WARM_START"


def warm_start_enabled() -> bool:
    """True only when `KALOS_EXPERIMENTAL_WARM_START` is set to 1/true/yes."""
    return os.environ.get(WARM_START_ENV, "").strip().lower() in {"1", "true", "yes"}


class MultiTaskSurrogate:
    """A `MultiTaskGP` over (X, campaign) with the target campaign's posterior
    as its single output. Campaign labels are any hashables; they are mapped to
    integer task ids internally with the target campaign first (task 0)."""

    def __init__(self, rank: int | None = None) -> None:
        self.rank = rank
        self.model: Any = None  # botorch MultiTaskGP once fitted
        self._X: Any = None  # torch tensor of the target campaign's rows
        self._y: np.ndarray | None = None
        self.task_ids: dict = {}

    def fit(self, X, y, tasks, bounds, *, target_task) -> MultiTaskSurrogate:
        import torch
        from botorch.models import MultiTaskGP
        from botorch.models.transforms.input import Normalize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        from .surrogate import DEVICE, DTYPE, _fit_mll_with_retry, sanitize_bounds

        Xa = np.asarray(X, float)
        ya = np.asarray(y, float).reshape(-1)
        ta = np.asarray(tasks, dtype=object).reshape(-1)
        if Xa.ndim != 2 or len(Xa) != len(ya) or len(ya) != len(ta):
            raise ValueError("X (n x d), y (n) and tasks (n) must align")
        if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
            raise ValueError("X and y must be finite")
        if not (ta == target_task).any():
            raise ValueError("the target campaign needs at least one run to warm-start")
        labels = [target_task, *sorted({t for t in ta.tolist() if t != target_task}, key=str)]
        self.task_ids = {t: i for i, t in enumerate(labels)}
        tid = np.array([self.task_ids[t] for t in ta], float)
        d = Xa.shape[1]
        lower, upper = sanitize_bounds(bounds)
        full = torch.as_tensor(np.column_stack([Xa, tid]), dtype=DTYPE, device=DEVICE)
        yt = torch.as_tensor(ya, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        box = torch.as_tensor(np.vstack([lower, upper]), dtype=DTYPE, device=DEVICE)
        normalize = Normalize(d=d + 1, indices=list(range(d)), bounds=box)
        self.model = MultiTaskGP(
            full, yt, task_feature=d, output_tasks=[0], rank=self.rank, input_transform=normalize,
        )
        _fit_mll_with_retry(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        target_rows = tid == 0
        self._X = torch.as_tensor(Xa[target_rows], dtype=DTYPE, device=DEVICE)
        self._y = ya[target_rows]
        return self

    def posterior(self, X, *, observation_noise: bool = False):
        """(mean, std) for the TARGET campaign at `X` (n x d, no task column)."""
        import torch

        from .surrogate import DEVICE, DTYPE

        assert self.model is not None, "fit() first"
        Xt = torch.as_tensor(np.asarray(X, float), dtype=DTYPE, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(Xt, observation_noise=observation_noise)
            mean = post.mean.reshape(-1).cpu().numpy()
            std = post.variance.clamp_min(1e-12).sqrt().reshape(-1).cpu().numpy()
        return mean, std

    @property
    def best_f(self) -> float:
        assert self._y is not None
        return float(self._y.max())


def campaign_surrogate(X, y, tasks, bounds, *, target_task, enabled: bool | None = None):
    """The flag point. Returns a `MultiTaskSurrogate` over every campaign when
    warm start is enabled (`enabled`, defaulting to `warm_start_enabled()`) and
    there is at least one other campaign; otherwise the default single-task
    `Surrogate` fitted on the target campaign's rows only, exactly as today."""
    from .surrogate import Surrogate

    ta = np.asarray(tasks, dtype=object).reshape(-1)
    on = warm_start_enabled() if enabled is None else bool(enabled)
    others = bool((ta != target_task).any())
    if on and others:
        return MultiTaskSurrogate().fit(X, y, ta, bounds, target_task=target_task)
    rows = ta == target_task
    return Surrogate().fit(np.asarray(X, float)[rows], np.asarray(y, float)[rows], bounds=bounds)


__all__ = ["WARM_START_ENV", "MultiTaskSurrogate", "campaign_surrogate", "warm_start_enabled"]
