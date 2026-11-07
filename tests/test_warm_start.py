"""Cross-campaign warm start (EXPERIMENTAL, behind KALOS_EXPERIMENTAL_WARM_START):
off by default; when on, a MultiTaskGP lends earlier campaigns' shape to a new
campaign, beating a cold single-task start after a few runs."""
from __future__ import annotations

import numpy as np

from kalos.bench.warm_start import BOUNDS, make_campaigns, warm_start_benchmark
from kalos.core.multitask import (
    MultiTaskSurrogate,
    campaign_surrogate,
    warm_start_enabled,
)
from kalos.core.optimize import propose
from kalos.core.surrogate import Surrogate


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("KALOS_EXPERIMENTAL_WARM_START", raising=False)
    assert not warm_start_enabled()
    X, y, c = make_campaigns(0)
    s = campaign_surrogate(X, y, c, BOUNDS, target_task=0)
    assert type(s) is Surrogate and len(s._y) == int((c == 0).sum())
    monkeypatch.setenv("KALOS_EXPERIMENTAL_WARM_START", "1")
    assert isinstance(campaign_surrogate(X, y, c, BOUNDS, target_task=0), MultiTaskSurrogate)


def test_propose_runs_on_the_multitask_surrogate():
    X, y, c = make_campaigns(1)
    keep = (c != 0) | (np.arange(len(c)) % 5 == 0)  # 4 target runs
    mt = MultiTaskSurrogate().fit(X[keep], y[keep], c[keep], BOUNDS, target_task=0)
    batch = propose(mt, BOUNDS, q=2, num_restarts=2, raw_samples=64, seed=0)
    assert batch.shape[1] == X.shape[1] and np.all((batch >= 0) & (batch <= 1))
    mean, std = mt.posterior(batch)
    assert mean.shape == (len(batch),) and np.all(std > 0)


def test_three_regimes_and_warm_k_beats_cold_start():
    r = warm_start_benchmark(range(3), ks=(4,))
    assert set(r) == {"within", "new", "warm_4"}
    assert "mtgp" not in r["new"]  # an unseen campaign cannot be addressed by task
    assert r["warm_4"]["mtgp"]["rmse"] < r["warm_4"]["st_target"]["rmse"]
    assert r["warm_4"]["mtgp"]["rmse"] < r["warm_4"]["pooled"]["rmse"]
