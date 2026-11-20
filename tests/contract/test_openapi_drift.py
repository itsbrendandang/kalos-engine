"""Reorg guard R01: docs/openapi.json is the engine's HTTP contract, checked in.

- the checked-in file equals the document regenerated from the live app, byte
  for byte (regenerate: `.venv/bin/python scripts/export_openapi.py`);
- the export is deterministic (two renders are identical);
- every operation the app serves in its schema is in the file, and nothing else.

A behaviour-preserving move (REORG_PLAN Phase 2) must leave the file unchanged.
A branch that adds or changes a route regenerates it in the same commit, the
same way it regenerates docs/route-scopes.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "docs" / "openapi.json"
REGENERATE = "docs/openapi.json is stale: run .venv/bin/python scripts/export_openapi.py and commit the result"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rendered() -> str:
    return str(_script().render())


def test_checked_in_openapi_has_not_drifted(rendered: str) -> None:
    assert DOCUMENT.exists(), REGENERATE
    current = DOCUMENT.read_text(encoding="utf-8")
    if current != rendered:
        old, new = json.loads(current), json.loads(rendered)
        added = sorted(set(new.get("paths", {})) - set(old.get("paths", {})))
        removed = sorted(set(old.get("paths", {})) - set(new.get("paths", {})))
        changed = sorted(p for p in set(new.get("paths", {})) & set(old.get("paths", {}))
                         if new["paths"][p] != old["paths"][p])
        schemas = sorted(k for k in set(new.get("components", {}).get("schemas", {}))
                         | set(old.get("components", {}).get("schemas", {}))
                         if new.get("components", {}).get("schemas", {}).get(k)
                         != old.get("components", {}).get("schemas", {}).get(k))
        pytest.fail(f"{REGENERATE}\n  paths added: {added}\n  paths removed: {removed}\n"
                    f"  paths changed: {changed}\n  schemas changed: {schemas}", pytrace=False)


def test_export_is_deterministic(rendered: str) -> None:
    assert _script().render() == rendered


# Routes deliberately hidden from the schema (`include_in_schema=False`). None
# today; one added later is listed here so the hiding is a reviewed decision.
HIDDEN: set[tuple[str, str]] = set()


def test_document_lists_exactly_the_served_routes(rendered: str) -> None:
    from kalos.portal.app import app
    from kalos.portal.route_scopes import build_manifest

    doc = json.loads(rendered)
    listed = {(m.upper(), p) for p, ops in doc["paths"].items() for m in ops}
    # build_manifest walks both router layouts (copied routes and FastAPI's lazily
    # included routers), so it sees every APIRoute the app serves.
    served = {(e["method"], e["path"]) for e in build_manifest(app)}
    assert listed == served - HIDDEN, (
        f"in the schema but not served: {sorted(listed - served)}; "
        f"served but not in the schema (hide deliberately via HIDDEN): {sorted(served - HIDDEN - listed)}"
    )
    assert doc["openapi"].startswith("3.")
    assert doc["info"]["title"] == "Kalos Engine API"
