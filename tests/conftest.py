"""Suite-wide safety net: no test may touch the real `~/.kalos/portal.db`.

Opening a `CampaignStore` on a schema-v1 database migrates it (after taking a
backup), so a test that reaches the lazily-built `get_campaign_store()`
singleton without pinning it would otherwise migrate a developer's real state
directory. Every test gets a private `KALOS_STATE_DIR` and an unbuilt
singleton; tests that pin their own store (`monkeypatch.setattr(campaign,
"_STORE", ...)`) still override this, since their fixtures run later.
"""
from __future__ import annotations

import os
import sys

import pytest

# --- one torch thread per test process ----------------------------------------- #
#
# The engine pins torch to KALOS_TORCH_THREADS threads on its first analysis
# (default min(4, cpus); kalos/portal/pipeline.py). Under `pytest -n auto` that
# is 4 threads in EVERY xdist worker: 16 busy threads on a 4-core CI runner. On
# linux the oversubscribed OpenMP pool slows GP fits by 10x or more (the same
# pathology deploy/compose.yaml documents for the container), which is what made
# single analyses take 5-10 minutes in CI and the full suite time out. Tests run
# with one thread per process unless the caller sets the variable (this module
# is imported before any test module imports kalos.portal). One vs four threads
# gives identical analysis output (measured; see also tests/test_golden_determinism.py).
os.environ.setdefault("KALOS_TORCH_THREADS", "1")
_TEST_TORCH_THREADS = int(os.environ["KALOS_TORCH_THREADS"].strip() or 0)


@pytest.fixture(autouse=True)
def _torch_thread_count_does_not_leak():
    """A test that pins torch to another count (tests/test_run_provenance.py
    pins 3) must not leave the whole worker process at that count."""
    yield
    torch = sys.modules.get("torch")
    if torch is not None and _TEST_TORCH_THREADS > 0 and torch.get_num_threads() != _TEST_TORCH_THREADS:
        torch.set_num_threads(_TEST_TORCH_THREADS)


@pytest.fixture(autouse=True)
def _isolated_kalos_state(tmp_path_factory, monkeypatch):
    state = tmp_path_factory.mktemp("kalos-state")
    monkeypatch.setenv("KALOS_STATE_DIR", str(state))
    try:
        from kalos.portal import campaign
    except ImportError:  # the portal extra is not installed
        return
    monkeypatch.setattr(campaign, "_STORE", None)


# --- one analysis shared by the tests that only read it ---------------------- #
#
# A full `_analyze` is 10+ GP fits plus the acquisition: tens of seconds on a CI
# core. Several modules run the SAME analysis (same sheet, same arguments) in
# many tests, each asserting a different part of the one response (data
# accounting, findings, drivers, the response contract). `shared_analysis` runs
# each distinct analysis once per worker process and hands every test a deep
# copy, so the assertions are unchanged and no test can see another's edits.
#
# Use it only where a test READS the result of a plain call. A test that checks
# determinism (two calls must agree), patches the engine or the environment
# first, or needs its own fresh run must call `_analyze` itself. Tests that share
# an analysis carry the same `xdist_group` mark, so under `pytest -n ...
# --dist loadgroup` (CI) they run on one worker and the analysis runs once.


def _frame_key(df) -> str:
    """A content key for a DataFrame: columns, dtypes, every cell's type and
    repr (so 6.5 and "6.5" differ), and the parser's attrs."""
    import hashlib

    cells = [[(type(v).__name__, repr(v)) for v in row] for row in df.itertuples(index=False)]
    blob = repr((
        [str(c) for c in df.columns], [str(t) for t in df.dtypes], [str(i) for i in df.index],
        cells, sorted((str(k), repr(v)) for k, v in df.attrs.items()),
    ))
    return hashlib.sha256(blob.encode()).hexdigest()


@pytest.fixture(scope="session")
def shared_analysis():
    import copy

    cache: dict[tuple, dict] = {}

    def run(df, target=None, **kwargs) -> dict:
        from kalos.portal.analysis import _analyze

        key = (_frame_key(df), target, repr(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = _analyze(df, target, **kwargs)
        return copy.deepcopy(cache[key])

    return run
