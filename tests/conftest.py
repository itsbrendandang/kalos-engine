"""Suite-wide safety net: no test may touch the real `~/.kalos` state directory.

Opening a `CampaignStore` on a schema-v1 database migrates it (after taking a
backup), so a test that reaches the lazily-built `get_campaign_store()`
singleton without pinning it would otherwise migrate a developer's real state
directory. Three layers, outermost first:

1. GUARD (detector). A Python audit hook, installed when this conftest is
   imported (before collection), refuses every file open, mkdir, listdir,
   rename/replace/remove, copy and `sqlite3.connect` whose path lies under the
   REAL home state directory (`~/.kalos`, resolved at import time). The
   operation raises `RealStateDirTouched` before it happens, and the access is
   recorded: the test that made it FAILS at teardown even if the code under
   test swallowed the exception; an access outside any test (collection, a
   session fixture) fails the session.
2. SESSION default. `pytest_configure` points `KALOS_STATE_DIR` at a private
   temporary directory before collection, so module- and session-scoped
   fixtures (which run BEFORE the function-scoped fixture below) and any
   subprocess that inherits the environment never default to `~/.kalos`.
3. PER TEST. Every test gets its own `KALOS_STATE_DIR` and an unbuilt store
   singleton; tests that pin their own store (`monkeypatch.setattr(campaign,
   "_STORE", ...)`) still override this, since their fixtures run later.

A higher-scoped fixture that needs the portal must still isolate itself
(`tmp_path_factory` + `pytest.MonkeyPatch.context()`), as
`tests/test_web_contract.py::analyze_response` does; layer 2 only keeps it off
the real directory, it does not keep modules apart.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from typing import Any

import pytest

# Resolved ONCE, before any test can change HOME or KALOS_STATE_DIR.
_REAL_HOME_STATE = os.path.join(os.path.expanduser("~"), ".kalos")
_GUARDED_ROOTS = tuple(sorted({
    os.path.normpath(os.path.abspath(_REAL_HOME_STATE)),
    os.path.normpath(os.path.realpath(_REAL_HOME_STATE)),
}))

# Audit events that carry a path as their first (and, for the two-path ones,
# second) argument. `os.stat` has no audit event; a bare existence check is not
# a touch.
_ONE_PATH_EVENTS = frozenset({
    "open", "os.mkdir", "os.listdir", "os.scandir", "os.remove", "os.rmdir",
    "os.chmod", "os.chown", "os.truncate", "os.utime", "os.chdir", "shutil.rmtree",
    "sqlite3.connect",
})
_TWO_PATH_EVENTS = frozenset({
    "os.rename", "os.replace", "os.link", "os.symlink", "shutil.copyfile",
    "shutil.copytree", "shutil.move",
})


class RealStateDirTouched(RuntimeError):
    """A test reached the real `~/.kalos` state directory."""


class _Guard:
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.reported = 0
        self.expecting = 0  # >0 inside `expect()` (the guard's own self-test)

    def unreported(self) -> list[str]:
        new = self.violations[self.reported:]
        self.reported = len(self.violations)
        return new

    @contextlib.contextmanager
    def expect(self) -> Iterator[list[str]]:
        """Collect (and do not report) the accesses refused inside the block."""
        start = len(self.violations)
        self.expecting += 1
        caught: list[str] = []
        try:
            yield caught
        finally:
            self.expecting -= 1
            caught.extend(self.violations[start:])
            del self.violations[start:]
            self.reported = min(self.reported, len(self.violations))


_GUARD = _Guard()


def _guarded(path: Any) -> str | None:
    """The normalized path when it lies under the real state dir, else None."""
    if path is None or isinstance(path, (int, bool)):
        return None
    try:
        text = os.fsdecode(os.fspath(path))
    except TypeError:
        return None
    if text.startswith("file:"):  # sqlite URI
        text = text[5:].split("?", 1)[0]
    if not text or text == ":memory:":
        return None
    norm = os.path.normpath(text if os.path.isabs(text) else os.path.abspath(text))
    for root in _GUARDED_ROOTS:
        if norm == root or norm.startswith(root + os.sep):
            return norm
    return None


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event in _ONE_PATH_EVENTS:
        paths = args[:1]
    elif event in _TWO_PATH_EVENTS:
        paths = args[:2]
    else:
        return
    for path in paths:
        hit = _guarded(path)
        if hit is not None:
            _GUARD.violations.append(f"{event}: {hit}")
            raise RealStateDirTouched(
                f"test touched the real Kalos state directory ({event} {hit}); "
                "isolate it with KALOS_STATE_DIR (see tests/conftest.py)"
            )


sys.addaudithook(_audit_hook)

_SESSION_STATE: str | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _SESSION_STATE
    _SESSION_STATE = tempfile.mkdtemp(prefix="kalos-session-state-")
    os.environ["KALOS_STATE_DIR"] = _SESSION_STATE
    _preimport_warning_modules_in_xdist_controller(config)


# Warning modules the GP stack emits from workers (NumericalWarning,
# OptimizationWarning, NumericsWarning, InputDataWarning, ...).
_WORKER_WARNING_MODULES = (
    "linear_operator.utils.warnings", "gpytorch.utils.warnings",
    "botorch.exceptions.warnings", "sklearn.exceptions",
)


def _preimport_warning_modules_in_xdist_controller(config: pytest.Config) -> None:
    """Under xdist the controller re-creates each warning a worker reports by
    importing the warning's module (xdist `unserialize_warning_message`), on the
    receiver thread of THAT worker. Two workers reporting GP-stack warnings at
    the same moment made two threads import torch/linear_operator concurrently,
    and CI died with `_DeadlockError: deadlock detected by
    _ModuleLock('linear_operator.utils.warnings')` (INTERNALERROR, node down).
    Importing those modules once here, before any worker starts, removes the
    race. Workers and plain (non-xdist) runs skip this."""
    if hasattr(config, "workerinput") or not config.getoption("numprocesses", None):
        return
    import importlib

    for name in _WORKER_WARNING_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:  # the ml extra is not installed
            pass


def pytest_unconfigure(config: pytest.Config) -> None:
    if _SESSION_STATE is not None:
        shutil.rmtree(_SESSION_STATE, ignore_errors=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    leaked = _GUARD.unreported()
    if leaked:
        sys.stderr.write(
            "\nERROR: the real Kalos state directory was touched outside a test:\n  "
            + "\n  ".join(leaked) + "\n"
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope="session", autouse=True)
def _real_state_dir_guard() -> Iterator[_Guard]:
    leaked = _GUARD.unreported()
    if leaked:  # collection or conftest-time access
        pytest.fail("the real Kalos state directory was touched during collection:\n  "
                    + "\n  ".join(leaked), pytrace=False)
    yield _GUARD


@pytest.fixture
def real_state_guard(_real_state_dir_guard: _Guard) -> _Guard:
    """The guard itself, for the test that proves it works."""
    return _real_state_dir_guard


# --- one torch thread per test process ----------------------------------------- #
#
# The engine pins torch to KALOS_TORCH_THREADS threads on its first analysis
# (default min(4, cpus); kalos/portal/pipeline.py). Under `pytest -n auto` that
# is 4 threads in EVERY xdist worker, several times more busy threads than the
# CI runner has cores (2 workers x 4 threads on a 4-vCPU, 2-core runner; more on
# a workstation). On linux the oversubscribed OpenMP pool slows GP fits by 10x or
# more (the same pathology deploy/compose.yaml documents for the container),
# which is what made single analyses take 5-10 minutes in CI. Tests run
# with one thread per process unless the caller sets the variable (this module
# is imported before any test module imports kalos.portal). One vs four threads
# gives identical analysis output (measured; see also tests/test_golden_determinism.py).
os.environ.setdefault("KALOS_TORCH_THREADS", "1")
# The same reasoning for the analysis worker pool (kalos/core/parallel.py): under
# xdist every worker would start its own pool. Tests run serially unless the
# caller sets the variable; tests/test_parallel_identity.py sets it per test.
os.environ.setdefault("KALOS_ANALYSIS_WORKERS", "1")
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
def _isolated_kalos_state(tmp_path_factory, monkeypatch, _real_state_dir_guard):
    state = tmp_path_factory.mktemp("kalos-state")
    monkeypatch.setenv("KALOS_STATE_DIR", str(state))
    try:
        from kalos.portal import campaign
    except ImportError:  # the portal extra is not installed
        campaign = None
    if campaign is not None:
        monkeypatch.setattr(campaign, "_STORE", None)
    yield
    leaked = _real_state_dir_guard.unreported()
    if leaked:
        pytest.fail("this test (or a fixture it uses) touched the real Kalos state "
                    "directory:\n  " + "\n  ".join(leaked), pytrace=False)


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
