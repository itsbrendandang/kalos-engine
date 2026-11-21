"""The module-alias harness for the Phase 2 moves (kalos/_moved.py, REORG_PLAN section 4).

Two parts:

1. The machinery, proven on a throwaway package in a fresh interpreter: an old
   path is the SAME module object as the new one (executed once), for a module
   and for a package with submodules, in either import order; `setattr` and
   `mock.patch` through the old path land on the module the code reads; `from
   old import name` works; reload keeps the identity.
2. The registry (`kalos._moved.MOVED`): every entry's old path imports as the
   very module at its new path, every submodule of a moved package does too,
   the file left at the old path is a pure two-line shim, and no shim exists
   without a registry entry. Empty until the first Phase 2 move.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from kalos import _moved

ROOT = Path(__file__).resolve().parents[2]
KALOS = ROOT / "kalos"

_SCENARIO = textwrap.dedent('''
    import importlib, json, sys
    from unittest import mock

    order = sys.argv[1]
    import aliaspkg
    if order == "new-first":
        import aliaspkg.new_home.run, aliaspkg.new_home.parts.columns
    from aliaspkg.old import analysis as old_mod
    import aliaspkg.old.analysis_parts.columns as old_cols
    from aliaspkg.old.analysis import read
    from aliaspkg.old.analysis_parts import columns as old_cols_b
    import aliaspkg.new_home.run as new_mod
    import aliaspkg.new_home.parts.columns as new_cols
    import aliaspkg.new_home.parts as new_parts
    import aliaspkg.old.analysis_parts as old_parts

    out = {
        "module_identity": old_mod is new_mod and sys.modules["aliaspkg.old.analysis"] is new_mod,
        "package_identity": old_parts is new_parts,
        "submodule_identity": old_cols is new_cols and old_cols_b is new_cols,
        "parent_attribute": aliaspkg.old.analysis is new_mod,
        "name_is_new": old_mod.__name__ == "aliaspkg.new_home.run",
        "from_import": read is new_mod.read,
        "executions": list(aliaspkg.EXECUTED),
    }
    old_mod.STATE = "set-through-old"
    out["setattr_reaches_code"] = new_mod.read() == "set-through-old"
    with mock.patch("aliaspkg.old.analysis.STATE", "mock-patched"):
        out["mock_patch_reaches_code"] = new_mod.read() == "mock-patched"
    with mock.patch("aliaspkg.old.analysis_parts.columns.WIDTH", 99):
        out["mock_patch_submodule"] = new_cols.width() == 99
    reloaded = importlib.reload(old_mod)
    out["reload_keeps_identity"] = reloaded is new_mod and sys.modules["aliaspkg.old.analysis"] is new_mod
    print(json.dumps(out))
''')


def _write_package(root: Path) -> None:
    files = {
        "aliaspkg/__init__.py": "EXECUTED = []\n",
        "aliaspkg/new_home/__init__.py": "",
        "aliaspkg/new_home/run.py": (
            "import aliaspkg\naliaspkg.EXECUTED.append(__name__)\nSTATE = 'real'\n\n"
            "def read():\n    return STATE\n"
        ),
        "aliaspkg/new_home/parts/__init__.py": "import aliaspkg\naliaspkg.EXECUTED.append(__name__)\n",
        "aliaspkg/new_home/parts/columns.py": (
            "import aliaspkg\naliaspkg.EXECUTED.append(__name__)\nWIDTH = 1\n\n"
            "def width():\n    return WIDTH\n"
        ),
        "aliaspkg/old/__init__.py": "",
        "aliaspkg/old/analysis.py": (
            "from kalos._moved import alias\nalias(__name__, 'aliaspkg.new_home.run')\n"
        ),
        "aliaspkg/old/analysis_parts/__init__.py": (
            "from kalos._moved import alias\nalias(__name__, 'aliaspkg.new_home.parts')\n"
        ),
    }
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.mark.parametrize("order", ["old-first", "new-first"])
def test_alias_machinery_in_a_fresh_interpreter(tmp_path: Path, order: str) -> None:
    _write_package(tmp_path)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    proc = subprocess.run([sys.executable, "-c", _SCENARIO, order], capture_output=True, text=True,
                          env=env, cwd=tmp_path, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    executions = out.pop("executions")
    # Each real module ran exactly once, under its new name only.
    assert sorted(executions) == sorted(
        ["aliaspkg.new_home.run", "aliaspkg.new_home.parts", "aliaspkg.new_home.parts.columns"]), executions
    assert all(out.values()), out


def test_unregistered_alias_is_refused() -> None:
    with pytest.raises(KeyError, match=r"not in kalos\._moved\.MOVED"):
        _moved.alias("kalos.portal.no_such_module_for_this_test")


def test_registry_is_well_formed() -> None:
    for old, new in _moved.MOVED.items():
        assert old.startswith("kalos.") and new.startswith("kalos."), (old, new)
        assert old != new
        assert new not in _moved.MOVED, f"chained move {old} -> {new} -> {_moved.MOVED[new]}: point at the final path"
    assert len(set(_moved.MOVED.values())) == len(_moved.MOVED), "two old paths alias one new module"


def _shim_files() -> dict[str, Path]:
    """Every file under kalos/ that calls `alias(...)`, by module name."""
    found: dict[str, Path] = {}
    for path in KALOS.rglob("*.py"):
        if path.name == "_moved.py" or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "kalos._moved" not in text:
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts[:-1]) if rel.name == "__init__" else list(rel.parts)
        found[".".join(parts)] = path
    return found


def test_every_shim_is_registered_and_pure() -> None:
    shims = _shim_files()
    assert sorted(set(shims) - set(_moved.MOVED)) == [], "shim files without a MOVED entry"
    for path in shims.values():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        assert len(body) == 2, f"{path}: a shim is exactly `from kalos._moved import alias` + `alias(__name__)`"
        imp, call = body
        assert isinstance(imp, ast.ImportFrom) and imp.module == "kalos._moved", path
        assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call), path
        assert ast.unparse(call.value) == "alias(__name__)", f"{path}: the target belongs in MOVED, not the shim"


@pytest.mark.parametrize("old,new", sorted(_moved.MOVED.items()) or [pytest.param(None, None, marks=pytest.mark.skip(
    reason="no module has moved yet (Phase 2 not started)"))])
def test_moved_module_is_the_same_object(old: str, new: str, monkeypatch: pytest.MonkeyPatch) -> None:
    assert old in _shim_files(), f"{old} is in MOVED but no shim file is left at its old path"
    new_mod = importlib.import_module(new)
    old_mod = importlib.import_module(old)
    assert old_mod is new_mod
    # A patch through the old path is visible through the new one.
    probe = object()
    monkeypatch.setattr(old_mod, "kalos_alias_probe", probe, raising=False)
    assert new_mod.kalos_alias_probe is probe
    if hasattr(new_mod, "__path__"):
        for info in pkgutil.iter_modules(new_mod.__path__):
            sub_new = importlib.import_module(f"{new}.{info.name}")
            assert importlib.import_module(f"{old}.{info.name}") is sub_new, info.name
