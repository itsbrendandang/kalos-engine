"""Suite-wide safety net: no test may touch the real `~/.kalos/portal.db`.

Opening a `CampaignStore` on a schema-v1 database migrates it (after taking a
backup), so a test that reaches the lazily-built `get_campaign_store()`
singleton without pinning it would otherwise migrate a developer's real state
directory. Every test gets a private `KALOS_STATE_DIR` and an unbuilt
singleton; tests that pin their own store (`monkeypatch.setattr(campaign,
"_STORE", ...)`) still override this, since their fixtures run later.
"""
from __future__ import annotations

import pytest


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
