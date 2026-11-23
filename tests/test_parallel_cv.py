"""Cross-validation fold fits in worker processes give the same numbers as in
this process, so `KALOS_CV_WORKERS` changes only how long CV takes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalos.core import evaluation
from kalos.core.evaluation import cv_workers, grouped_cv_report


def _data(n=36, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(0, 4, n), rng.uniform(5, 7, n), rng.integers(0, 3, n)])
    y = 1.5 * X[:, 0] - 0.8 * (X[:, 1] - 6) ** 2 + 0.4 * X[:, 2] + rng.normal(0, 0.3, n)
    return X, y


def _sheet(n=40, seed=0):
    rng = np.random.default_rng(seed)
    methanol, ph = rng.uniform(0, 4, n), rng.uniform(5, 7, n)
    return pd.DataFrame({
        "medium": rng.choice(["A", "B", "C"], n),
        "Methanol": methanol.round(3),
        "pH": ph.round(2),
        "lipase_titer": (1.5 * methanol - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.3, n)).round(3),
    })


def _report(monkeypatch, workers, **kw):
    monkeypatch.setenv("KALOS_CV_WORKERS", str(workers))
    X, y = _data()
    bounds = np.vstack([X.min(0), X.max(0)])
    return grouped_cv_report(X, y, bounds=bounds, n_boot=200, n_repeats=2, **kw)


@pytest.mark.parametrize("cat_dims", [None, [2]])
def test_parallel_folds_match_serial_exactly(monkeypatch, cat_dims):
    serial = _report(monkeypatch, 1, cat_dims=cat_dims)
    parallel = _report(monkeypatch, 3, cat_dims=cat_dims)
    assert parallel["oof_pred"] == serial["oof_pred"]
    assert parallel["oof_std"] == serial["oof_std"]
    assert parallel["oof_index"] == serial["oof_index"]
    for key in ("spearman", "ci95", "spearman_per_repeat", "n_folds", "n_groups", "n_repeats"):
        assert parallel[key] == serial[key], key


def test_serial_folds_ignore_the_torch_thread_count(monkeypatch):
    import torch

    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one = _report(monkeypatch, 1)
        torch.set_num_threads(4)
        four = _report(monkeypatch, 1)
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(prev)
    assert one["oof_pred"] == four["oof_pred"] and one["oof_std"] == four["oof_std"]


def test_broken_pool_falls_back_to_serial(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool

    class Broken:
        def map(self, *_a, **_k):
            raise BrokenProcessPool("worker died")

    serial = _report(monkeypatch, 1)
    monkeypatch.setattr(evaluation, "_fold_pool", lambda: Broken())
    assert _report(monkeypatch, 3)["oof_pred"] == serial["oof_pred"]


def test_cv_workers_env(monkeypatch):
    monkeypatch.setenv("KALOS_CV_WORKERS", "1")
    assert cv_workers() == 1
    monkeypatch.setenv("KALOS_CV_WORKERS", "0")
    assert cv_workers() == 1
    monkeypatch.setenv("KALOS_CV_WORKERS", "many")
    with pytest.raises(ValueError, match="KALOS_CV_WORKERS"):
        cv_workers()


def test_analysis_numbers_do_not_change_with_parallel_cv(monkeypatch):
    from kalos.portal.analysis import _analyze

    keys = ("cv_spearman", "cv_ci95", "cv_spearman_per_repeat", "conformal_q", "reliability",
            "proposals", "oof", "best", "gp_shapes")
    monkeypatch.setenv("KALOS_CV_WORKERS", "1")
    serial = _analyze(_sheet())
    monkeypatch.setenv("KALOS_CV_WORKERS", "3")
    parallel = _analyze(_sheet())
    for key in keys:
        assert parallel[key] == serial[key], key


def test_fold_labels_name_each_fold(monkeypatch):
    from kalos.core.splits import make_splits, row_hash_groups

    rep = _report(monkeypatch, 3)
    X, y = _data()
    expected = []
    for k, (_tr, te) in enumerate(make_splits(X, y, row_hash_groups(X), n_splits=5)):
        expected.extend([k] * len(te))
    assert rep["oof_repeats"][0]["fold"] == expected
    assert len(set(expected)) == rep["n_folds"] > 1


def test_daemon_process_fits_folds_here(monkeypatch):
    import multiprocessing as mp

    class Daemon:
        daemon = True

    serial = _report(monkeypatch, 1)
    monkeypatch.setattr(mp, "current_process", lambda: Daemon())
    monkeypatch.setattr(evaluation, "_fold_pool", lambda: pytest.fail("pool started in a daemon"))
    assert _report(monkeypatch, 3)["oof_pred"] == serial["oof_pred"]
