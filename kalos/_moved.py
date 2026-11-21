"""Module aliases for the Phase 2 reorg moves (~/.kalos-dev/REORG_PLAN.md section 4).

When a module moves (e.g. `kalos/portal/analysis.py` -> `kalos/analysis/run.py`),
its OLD path stays importable as the SAME module object, not as a copy and not as
a star re-export. That keeps every `monkeypatch.setattr(old_module, name, value)`
in the test suite and every `mock.patch("kalos.portal....")` string working: the
patch lands on the module the code actually reads. In-flight branches that still
import the old path keep working while they rebase.

How a move uses it, in one PR:

1. `git mv kalos/portal/analysis.py kalos/analysis/run.py` (no other edits in the
   moved file beyond import lines).
2. Add `"kalos.portal.analysis": "kalos.analysis.run"` to `MOVED` below.
3. Recreate the old file as a two-line shim:

       from kalos._moved import alias
       alias(__name__)

   For a moved PACKAGE (e.g. `kalos/portal/analysis_parts/` ->
   `kalos/analysis/parts/`) the shim is the old package's `__init__.py`, with the
   same two lines; every submodule (`kalos.portal.analysis_parts.columns`) then
   resolves to the new submodule object too.
4. Rewrite imports inside `kalos/` to the new path (mypy checks `kalos/` and only
   sees the shim at the old path); tests keep the old paths until the test-layout
   move (R21). tests/contract/test_module_aliases.py checks every entry.

Entry points that are run as a path (the job worker spawned as
`python -m kalos.portal.job_worker`, `python -m kalos.portal`) stay real files
that delegate; they are not aliased.

Torch-free and dependency-free: importing this module imports nothing else.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from collections.abc import Sequence
from types import ModuleType

# old dotted path -> new dotted path. Filled by the Phase 2 move PRs, one line
# per moved module or package; removed one release after the moves (D5).
MOVED: dict[str, str] = {}


class _AliasLoader(importlib.abc.Loader):
    """Hands back the already-imported module at the new path."""

    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(self.target)

    def exec_module(self, module: ModuleType) -> None:
        return None  # already executed under its new name


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves `old.pkg.sub` to `new.pkg.sub` for every aliased package. Sits
    first on `sys.meta_path`: otherwise the path finder would find the new file
    through the aliased package's `__path__` and execute it a second time under
    the old name, which is exactly the duplicate module aliasing must prevent."""

    def __init__(self) -> None:
        self.packages: dict[str, str] = {}

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        for old, new in self.packages.items():
            if fullname.startswith(old + "."):
                return importlib.machinery.ModuleSpec(fullname, _AliasLoader(new + fullname[len(old):]))
        return None


_FINDER = _AliasFinder()
_LOCK = threading.Lock()


def _install_finder() -> None:
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)


def alias(old: str, new: str | None = None) -> ModuleType:
    """Make `old` the same module object as `new` (default: `MOVED[old]`).

    Called from the shim file left at the old path with `alias(__name__)`. The
    import system returns `sys.modules[old]` after the shim runs, so the importer
    receives the new module. For a package, submodules of `old` resolve to the
    submodules of `new`."""
    target = MOVED.get(old) if new is None else new
    if target is None:
        raise KeyError(f"{old} is not in kalos._moved.MOVED; add the move there first")
    module = importlib.import_module(target)
    with _LOCK:
        sys.modules[old] = module
        if hasattr(module, "__path__"):  # a package: alias its submodules too
            _FINDER.packages[old] = target
            _install_finder()
    parent, _, child = old.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def aliased_packages() -> dict[str, str]:
    """The package prefixes currently aliased (old -> new); for the tests."""
    return dict(_FINDER.packages)


__all__ = ["MOVED", "alias", "aliased_packages"]
