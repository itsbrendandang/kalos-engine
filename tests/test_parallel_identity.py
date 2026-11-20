"""The analysis worker pool (kalos/core/parallel.py) changes speed, never results.

Claim under test: with `KALOS_ANALYSIS_WORKERS` > 1 (CV fold fits, the proposal
fit's extra starting points and the mixed acquisition's per-level solves run in
worker processes), the FULL `_analyze` response is byte-identical to the serial
run (`KALOS_ANALYSIS_WORKERS=1`, the code path that existed before), on the same
machine and stack: proposals, intervals, CV, verdict, every key except wall
clock. Compared as `json.dumps(..., sort_keys=True)` strings, not within a
tolerance. Each comparison also asserts the pool was really used, so a silent
fallback to serial cannot make the test pass vacuously.

Plus the fallbacks that keep that true when a task draws from the global RNG,
raises, cannot be pickled, or a mixed-acquisition solve would have retried.

Runs in the serial CI job (`serial`): it starts its own 2-process pool.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from kalos.core import parallel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic_bioprocess" / "synthetic_bioprocess.csv"
VOLATILE = {"timestamp", "fit_seconds", "created_at"}
WORKERS = "2"

pytestmark = [pytest.mark.serial]


@pytest.fixture
def workers(monkeypatch):
    """`workers("2")` / `workers("1")` switches the pool on and off; the pool is
    stopped after the test."""
    torch.set_num_threads(1)

    def _set(n: str) -> None:
        monkeypatch.setenv(parallel.ENV_WORKERS, n)
        parallel.reset_stats()

    yield _set
    parallel.shutdown()


def _sheets() -> dict:
    spec = importlib.util.spec_from_file_location("trust_golden", ROOT / "scripts" / "trust_golden.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SHEETS)


def _bioprocess_mixed() -> tuple[pd.DataFrame, dict]:
    """40 rows of the committed example with its 3-level `medium_base` factor and
    three continuous inputs: the mixed (categorical) acquisition on real-shaped data."""
    cols = ["medium_base", "pH", "temperature_C", "glucose_gL", "titer_g_L"]
    return pd.read_csv(EXAMPLE).head(40)[cols], {"target": "titer_g_L"}


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _analysis_json(make, seed: int, monkeypatch) -> str:
    import kalos.portal.analysis as analysis

    monkeypatch.setattr(analysis, "ANALYZE_SEED", seed)
    df, kw = make()
    kw = dict(kw)
    target = kw.pop("target")
    res = analysis._analyze(df, target=target, **kw)
    return json.dumps(_strip(json.loads(json.dumps(res, default=str))), sort_keys=True)


CASES = [
    # every trust-golden sheet at the engine's seed
    *[(name, 1234) for name in ("two_feature", "bioprocess_subset", "batch_text", "jitter_reps",
                                "pure_noise", "mixed_medium", "minimize", "campaign_rounds")],
    # other seeds where the parallel paths differ most: mixed levels, minimize, rounds
    ("mixed_medium", 7), ("minimize", 7), ("campaign_rounds", 7),
    ("bioprocess_mixed", 1234), ("bioprocess_mixed", 7),
]


@pytest.mark.slow
@pytest.mark.parametrize(("sheet", "seed"), CASES, ids=[f"{s}-seed{n}" for s, n in CASES])
def test_full_analysis_is_byte_identical_serial_vs_parallel(sheet, seed, workers, monkeypatch):
    make = _bioprocess_mixed if sheet == "bioprocess_mixed" else _sheets()[sheet]
    workers("1")
    serial = _analysis_json(make, seed, monkeypatch)
    assert parallel.stats["pool_tasks"] == 0
    workers(WORKERS)
    pooled = _analysis_json(make, seed, monkeypatch)
    assert parallel.pool_tasks_by_fn.get("_fit_fold", 0) >= 5, (
        f"the CV folds did not run in the pool: {parallel.stats}, {parallel.pool_tasks_by_fn}")
    if sheet in ("mixed_medium", "bioprocess_mixed"):
        # 3 levels x q=5 greedy slots at least (a near-duplicate repair adds
        # slots), all in the pool
        assert parallel.pool_tasks_by_fn.get("_solve_level", 0) >= 3 * 5, parallel.pool_tasks_by_fn
        assert parallel.pool_tasks_by_fn.get("_fit_start", 0) == 2, parallel.pool_tasks_by_fn
    assert parallel.stats["error_redo"] == parallel.stats["broken_pool"] == 0, parallel.stats
    assert pooled == serial
    assert '"proposals"' in serial


# --- the proposal fit: a winning extra start comes back from a worker -------- #

def test_multi_start_fit_identical_when_an_extra_start_wins(workers):
    from kalos.core.surrogate import Surrogate

    rng = np.random.default_rng(7)  # restart 1 wins on this draw (measured)
    X = rng.uniform(0, 1, (20, 3))
    y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2 + rng.normal(0, 0.3, 20)
    box = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    grid = np.random.default_rng(1).uniform(0, 1, (50, 3))

    def fit():
        s = Surrogate().fit(X, y, bounds=box, restarts=3, restart_seed=1234)
        diag = {k: v for k, v in s.fit_diagnostics.items() if k != "fit_seconds"}
        return diag, s.posterior(grid)

    workers("1")
    d1, (m1, s1) = fit()
    workers(WORKERS)
    d2, (m2, s2) = fit()
    assert parallel.pool_tasks_by_fn.get("_fit_start") == 2
    assert d1["restart_chosen"] == 1, "the data no longer exercises a winning extra start"
    assert d1 == d2
    assert np.array_equal(m1, m2) and np.array_equal(s1, s2)


# --- the ordering / RNG / error contract of run_ordered ---------------------- #

def _draw(k: int) -> float:
    """Draws from every global RNG, like a botorch fallback would."""
    return float(torch.rand(1)) + float(np.random.rand()) + random.random() + k


def _maybe_draw(k: int) -> float:
    return _draw(k) if k == 2 else float(k)


def _fail_at(k: int) -> int:
    if k == 2:
        raise ValueError(f"task {k} failed")
    return k


def _rng_states():
    return torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate()


def _same_states(a, b) -> bool:
    return torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


@pytest.mark.parametrize("fn", [_draw, _maybe_draw])
def test_tasks_that_draw_from_the_global_rng_match_serial(fn, workers):
    tasks = [(k,) for k in range(5)]

    def run():
        torch.manual_seed(3)
        np.random.seed(3)
        random.seed(3)
        return parallel.run_ordered(fn, tasks), _rng_states()

    workers("1")
    serial, s_state = run()
    workers(WORKERS)
    pooled, p_state = run()
    assert pooled == serial
    assert _same_states(s_state, p_state), "the RNG state after the tasks differs from serial"
    assert parallel.stats["rng_redo"] == 1


def test_a_task_error_is_the_serial_error(workers):
    workers(WORKERS)
    with pytest.raises(ValueError, match="task 2 failed"):
        parallel.run_ordered(_fail_at, [(k,) for k in range(5)])
    assert parallel.stats["error_redo"] == 1


def test_unpicklable_tasks_run_serially(workers):
    workers(WORKERS)
    lam = lambda k: k + 1  # noqa: E731 - deliberately not picklable
    assert parallel.run_ordered(lam, [(1,), (2,)]) == [2, 3]
    assert parallel.run_ordered(_fail_at, [(0,), (1,)]) == [0, 1]
    assert parallel.stats["local_tasks"] == 2 and parallel.stats["pool_tasks"] == 2


def test_rng_moved_between_submit_and_collect_recomputes(workers):
    workers(WORKERS)
    torch.manual_seed(5)
    np.random.seed(5)
    random.seed(5)
    pending = parallel.submit(_draw, [(0,), (1,)])
    torch.rand(1)  # the serial code would have started the tasks from here
    snap = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
    got = pending.collect()
    torch.set_rng_state(snap[0])
    np.random.set_state(snap[1])
    random.setstate(snap[2])
    assert got == [_draw(0), _draw(1)]
    assert parallel.stats["rng_redo"] == 1 and parallel.stats["pool_tasks"] == 0


def test_warnings_from_a_worker_are_replayed(workers):
    workers(WORKERS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parallel.run_ordered(_warn, [(1,), (2,)])
    assert [str(w.message) for w in caught if "task warns" in str(w.message)] == [
        "task warns 1", "task warns 2"]


def _warn(k: int) -> int:
    warnings.warn(f"task warns {k}", UserWarning, stacklevel=1)
    return k


# --- when the pool is not used --------------------------------------------- #

def test_serial_when_torch_runs_more_than_one_thread(workers):
    workers(WORKERS)
    torch.set_num_threads(2)
    try:
        assert parallel.active_workers() == 1
        assert "torch threads" in (parallel.serial_reason() or "")
    finally:
        torch.set_num_threads(1)
    assert parallel.active_workers() == 2


@pytest.mark.parametrize(("raw", "want"), [("1", 1), ("0", 1), ("3", 3), ("junk", 1)])
def test_configured_workers(raw, want, monkeypatch):
    monkeypatch.setenv(parallel.ENV_WORKERS, raw)
    assert parallel.configured_workers() == want


def test_auto_workers_share_the_cpus_between_concurrent_analyses(monkeypatch):
    monkeypatch.setenv(parallel.ENV_WORKERS, "auto")
    monkeypatch.setattr(parallel, "_cpu_count", lambda: 4)
    monkeypatch.setenv("KALOS_JOB_WORKERS", "2")
    assert parallel.configured_workers() == 2  # the 4-vCPU CI runner
    monkeypatch.setattr(parallel, "_cpu_count", lambda: 64)
    assert parallel.configured_workers() == parallel.MAX_AUTO_WORKERS
    monkeypatch.setattr(parallel, "_cpu_count", lambda: 1)
    assert parallel.configured_workers() == 1


@pytest.mark.parametrize("raw", ["-1", "many"])
def test_malformed_worker_count_refuses_startup(raw, monkeypatch):
    monkeypatch.setenv(parallel.ENV_WORKERS, raw)
    with pytest.raises(ValueError, match=parallel.ENV_WORKERS):
        parallel.validate_env()


# --- the mixed acquisition's slot redo -------------------------------------- #

def test_a_solve_that_would_have_retried_redoes_the_slot_serially(workers, monkeypatch):
    """An OptimizationWarning in a level's solve means botorch would have drawn new
    initial conditions from the RNG; the slot is then recomputed by botorch's own
    serial loop from the RNG state saved before it, so the batch is the serial one."""
    from botorch.exceptions.warnings import OptimizationWarning

    import kalos.core.mixed_parallel as mp_
    from kalos.core.optimize import propose_batch
    from kalos.core.surrogate import Surrogate

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(0, 1, (18, 2)), np.tile([0.0, 1.0, 2.0], 6)])
    y = X[:, 0] - (X[:, 1] - 0.5) ** 2 + 0.3 * X[:, 2] + rng.normal(0, 0.05, 18)
    box = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]])
    s = Surrogate().fit(X, y, bounds=box, cat_dims=[2])

    def propose():
        return propose_batch(s, box, q=2, num_restarts=2, raw_samples=16, cat_dims=[2],
                             cat_cardinalities=[3], seed=11).X

    workers("1")
    want = propose()

    # In-process stand-ins for the pool, so the redo path runs deterministically
    # here: every solve reports "would have retried".
    real_solve = mp_._solve_level

    def warned_solve(*args):
        cands, vals, _clean = real_solve(*args)
        warnings.warn("forced", OptimizationWarning, stacklevel=1)
        return cands, vals, False

    monkeypatch.setattr(mp_.parallel, "active_workers", lambda: 2)
    monkeypatch.setattr(mp_.parallel, "run_ordered", lambda fn, tasks, **_: [warned_solve(*t) for t in tasks])
    got = propose()
    assert np.array_equal(got, want)


def test_the_acquisition_pickles(workers):
    """The pool needs the acquisition picklable (no lambdas): minimize objective,
    black-box constraint and feasibility gate included."""
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.models.model_list_gp_regression import ModelListGP

    from kalos.core.optimize import _FloorConstraint, _negated_first_output
    from kalos.core.surrogate import Surrogate

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (12, 2))
    box = np.array([[0.0, 0.0], [1.0, 1.0]])
    a = Surrogate().fit(X, X.sum(1), bounds=box)
    b = Surrogate().fit(X, X[:, 0], bounds=box)
    acq = qLogNoisyExpectedImprovement(
        ModelListGP(a.model, b.model), X_baseline=a._X, prune_baseline=True,
        objective=GenericMCObjective(_negated_first_output), constraints=[_FloorConstraint(0.5)],
    )
    clone = pickle.loads(pickle.dumps(acq))
    Xq = torch.rand(4, 1, 2, dtype=torch.double)
    with torch.no_grad():
        assert torch.equal(acq(Xq), clone(Xq))


def test_an_idle_pool_is_stopped(workers, monkeypatch):
    import time

    monkeypatch.setattr(parallel, "POOL_IDLE_S", 0.3)
    workers(WORKERS)
    assert parallel.run_ordered(_fail_at, [(0,), (1,)]) == [0, 1]
    assert parallel._pool is not None
    deadline = time.monotonic() + 10
    while parallel._pool is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert parallel._pool is None
    assert parallel.run_ordered(_fail_at, [(0,), (1,)]) == [0, 1]  # a new pool starts
    assert parallel.stats["pool_tasks"] == 4
