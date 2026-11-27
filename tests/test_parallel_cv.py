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


@pytest.mark.parametrize("threads", [1, 2])
def test_parallel_folds_match_serial_at_the_callers_thread_count(monkeypatch, threads):
    import torch

    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        serial = _report(monkeypatch, 1)
        parallel = _report(monkeypatch, 3)
    finally:
        torch.set_num_threads(prev)
    assert parallel["oof_pred"] == serial["oof_pred"] and parallel["oof_std"] == serial["oof_std"]


def test_a_fold_that_draws_from_the_rng_gives_the_serial_numbers(monkeypatch):
    """A retry that draws from the torch RNG shifts what the next folds start from
    in a serial run; the parallel path refits those folds so the result and the
    RNG state it leaves both match the serial run."""
    import torch

    real_fit = evaluation._fit_fold

    def drawing_fit(*fold):
        mean, std = real_fit(*fold)
        return mean + torch.rand(1).item(), std

    class InProcessPool:
        def map(self, fn, tasks):
            out = []
            prev = torch.get_num_threads()
            with torch.random.fork_rng():
                for task in tasks:
                    out.append(fn(task))
            torch.set_num_threads(prev)
            return out

    monkeypatch.setattr(evaluation, "_fit_fold", drawing_fit)
    monkeypatch.setattr(evaluation, "_fold_pool", lambda: InProcessPool())
    torch.manual_seed(7)
    serial = _report(monkeypatch, 1)
    serial_state = torch.get_rng_state()
    torch.manual_seed(7)
    parallel = _report(monkeypatch, 3)
    assert parallel["oof_pred"] == serial["oof_pred"]
    assert torch.equal(torch.get_rng_state(), serial_state)


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


def test_failed_proposal_fit_waits_for_cv(monkeypatch):
    import threading
    import time

    from kalos.core import surrogate
    from kalos.portal.analysis import _analyze

    done = threading.Event()

    def slow_cv(*_a, **_k):
        time.sleep(0.5)
        done.set()
        raise RuntimeError("cv stopped")

    def broken_fit(*_a, **_k):
        raise RuntimeError("proposal fit failed")

    monkeypatch.setenv("KALOS_CV_WORKERS", "3")
    monkeypatch.setattr(evaluation, "grouped_cv_report", slow_cv)
    monkeypatch.setattr(surrogate.Surrogate, "fit", broken_fit)
    with pytest.raises(RuntimeError, match="proposal fit failed"):
        _analyze(_sheet())
    assert done.is_set()


def test_daemon_analysis_runs_cv_after_the_proposal_fit(monkeypatch):
    import multiprocessing as mp
    import threading

    from kalos.portal.analysis_parts.background_cv import cv_alongside

    class Daemon:
        daemon = True

    ran_in: list[str] = []
    monkeypatch.setenv("KALOS_CV_WORKERS", "3")
    monkeypatch.setattr(mp, "current_process", lambda: Daemon())
    with cv_alongside(lambda: ran_in.append(threading.current_thread().name) or {}) as cv:
        assert ran_in == []
    cv.result()
    assert ran_in == [threading.current_thread().name]


def _background_report(monkeypatch, between=lambda: None):
    import torch

    from kalos.portal.analysis_parts.background_cv import BackgroundCV

    monkeypatch.setenv("KALOS_CV_WORKERS", "3")
    X, y = _data()
    bounds = np.vstack([X.min(0), X.max(0)])
    torch.manual_seed(3)
    cv = BackgroundCV(lambda: grouped_cv_report(X, y, bounds=bounds, n_boot=200, n_repeats=2))
    between()
    return cv.result()


def _serial_report(monkeypatch, between=lambda: None):
    import torch

    torch.manual_seed(3)
    between()
    return _report(monkeypatch, 1)


def test_precomputed_folds_are_used_and_give_the_serial_numbers(monkeypatch):
    refit: list[int] = []
    real = evaluation._fit_folds_now
    monkeypatch.setattr(evaluation, "_fit_folds_now", lambda t: refit.append(len(t)) or real(t))
    background = _background_report(monkeypatch)
    assert sum(refit) == 0
    serial = _serial_report(monkeypatch)
    assert background["oof_pred"] == serial["oof_pred"] and background["oof_std"] == serial["oof_std"]


def test_folds_are_refit_when_the_proposal_fit_moves_the_rng(monkeypatch):
    """The proposal fit runs between the precompute and the CV; if it draws from
    the torch RNG, the CV starts from a different state than the precompute did,
    as it would without the pool."""
    import torch

    def draw():
        torch.rand(3)

    background = _background_report(monkeypatch, draw)
    serial = _serial_report(monkeypatch, draw)
    assert background["oof_pred"] == serial["oof_pred"] and background["oof_std"] == serial["oof_std"]


def test_a_failed_precompute_still_gives_the_serial_numbers(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("worker failed from the precompute state")

    monkeypatch.setattr(evaluation, "_precompute_folds", broken)
    background = _background_report(monkeypatch)
    serial = _serial_report(monkeypatch)
    assert background["oof_pred"] == serial["oof_pred"]


def test_analysis_with_a_retry_draw_in_the_proposal_fit_matches_serial(monkeypatch):
    """A proposal fit that draws from the torch RNG (a botorch retry) moves the
    state the CV starts from; the precomputed folds no longer match it, so the CV
    fits them again on the analysis thread and gives the serial numbers."""
    import torch

    from kalos.core import surrogate
    from kalos.portal.analysis import _analyze

    real_fit = surrogate.Surrogate.fit

    def fit_with_retry_draw(self, *a, **k):
        out = real_fit(self, *a, **k)
        if "restart_seed" in k:  # the proposal fit, not a fold
            torch.rand(1)
        return out

    refit: list[int] = []  # primary-CV folds fitted again instead of taken precomputed
    real_now = evaluation._fit_folds_now

    def counting_now(tasks):
        if getattr(evaluation._LOCAL, "cache", None):
            refit.append(len(tasks))
        return real_now(tasks)

    monkeypatch.setattr(evaluation, "_fit_folds_now", counting_now)
    monkeypatch.setattr(surrogate.Surrogate, "fit", fit_with_retry_draw)
    keys = ("cv_spearman", "cv_ci95", "cv_spearman_per_repeat", "conformal_q", "reliability",
            "proposals", "oof", "best", "gp_shapes")
    monkeypatch.setenv("KALOS_CV_WORKERS", "1")
    torch.manual_seed(0)
    serial = _analyze(_sheet())
    refit.clear()
    monkeypatch.setenv("KALOS_CV_WORKERS", "3")
    torch.manual_seed(0)
    parallel = _analyze(_sheet())
    assert sum(refit) > 0
    for key in keys:
        assert parallel[key] == serial[key], key
