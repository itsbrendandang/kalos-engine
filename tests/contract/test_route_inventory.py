"""Reorg guard R04: the route inventory (docs/routes.json), cross-checked.

- the checked-in inventory equals the one regenerated from the live app
  (regenerate: `.venv/bin/python scripts/export_routes.py`);
- it lists the same (method, path) set as docs/route-scopes.json, with the same
  scopes, auth mode and conditional scopes for every route;
- every route belongs to a module the engine owns, and a module that serves
  routes is one of the known router modules (a new router module is a reviewed
  addition to ROUTER_MODULES, and the import-layer rules place it in `api`);
- the OpenAPI snapshot (docs/openapi.json, guard R01), when present, lists the
  same operations as the inventory's schema routes.

A behaviour-preserving move (REORG_PLAN Phase 2) changes only the `module` of
the routes it moves; everything else in the file must stay identical.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("fastapi")

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs" / "routes.json"
SCOPES = ROOT / "docs" / "route-scopes.json"
OPENAPI = ROOT / "docs" / "openapi.json"
REGENERATE = "docs/routes.json is stale: run .venv/bin/python scripts/export_routes.py and commit the result"

# Modules that define HTTP endpoints today. A move (Phase 2) adds the new module
# path here in the same PR; R14 later collapses these into kalos.api.routers.*.
ROUTER_MODULES = frozenset({
    "kalos.portal.app",
    "kalos.portal.campaign_routes",
    "kalos.portal.campaigns_api",
    "kalos.portal.clones_api",
    "kalos.portal.exports_api",
    "kalos.portal.gates_api",
    "kalos.portal.jobs_api",
    "kalos.portal.scale_routes",
    "kalos.portal.setup_api",
    "kalos.assist.routes",
    "kalos.assist.sop_routes",
})
ROUTER_PACKAGES = ("kalos.api.",)  # target home of every router (REORG_PLAN 2.1)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("export_routes", ROOT / "scripts" / "export_routes.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inventory() -> list[dict]:
    return list(_script().build())


def _key(e: dict) -> tuple[str, str]:
    return (e["method"], e["path"])


def test_checked_in_inventory_has_not_drifted(inventory: list[dict]) -> None:
    assert INVENTORY.exists(), REGENERATE
    current = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if current != inventory:
        old = {_key(e): e for e in current}
        new = {_key(e): e for e in inventory}
        lines = [f"  + {k}" for k in sorted(set(new) - set(old))]
        lines += [f"  - {k}" for k in sorted(set(old) - set(new))]
        for k in sorted(set(old) & set(new)):
            diff = {f: (old[k].get(f), new[k].get(f)) for f in set(old[k]) | set(new[k])
                    if old[k].get(f) != new[k].get(f)}
            if diff:
                lines.append(f"  ~ {k}: {diff}")
        pytest.fail(REGENERATE + "\n" + "\n".join(lines), pytrace=False)
    assert INVENTORY.read_text(encoding="utf-8") == _script().render(inventory)


def test_inventory_is_sorted_unique_and_well_formed(inventory: list[dict]) -> None:
    assert inventory == sorted(inventory, key=lambda e: (e["path"], e["method"]))
    assert len({_key(e) for e in inventory}) == len(inventory)
    allowed = {"method", "path", "module", "endpoint", "scopes", "auth", "in_schema", "status_code",
               "response_model", "conditional_scopes"}
    for e in inventory:
        assert set(e) <= allowed, e
        assert e["method"] in {"GET", "POST", "PUT", "PATCH", "DELETE"}, e
        assert e["path"].startswith("/"), e
        assert e["auth"] in ("required", "public"), e


def test_inventory_matches_the_scope_manifest(inventory: list[dict]) -> None:
    """docs/route-scopes.json is what the web mints engine tokens from; the
    inventory must agree with it route for route."""
    manifest = {_key(e): e for e in json.loads(SCOPES.read_text(encoding="utf-8"))}
    routes = {_key(e): e for e in inventory}
    assert sorted(set(routes) - set(manifest)) == [], "routes missing from docs/route-scopes.json"
    assert sorted(set(manifest) - set(routes)) == [], "docs/route-scopes.json lists routes the app does not serve"
    for key, e in routes.items():
        m = manifest[key]
        assert (e["scopes"], e["auth"], e.get("conditional_scopes")) == (
            m["scopes"], m["auth"], m.get("conditional_scopes")), key


def test_every_route_lives_in_a_known_router_module(inventory: list[dict]) -> None:
    unknown = sorted({e["module"] for e in inventory
                      if e["module"] not in ROUTER_MODULES and not e["module"].startswith(ROUTER_PACKAGES)})
    assert unknown == [], (
        f"routes defined in unregistered modules {unknown}: add the module to ROUTER_MODULES "
        "(tests/contract/test_route_inventory.py) in the same PR"
    )


def test_public_routes_are_only_the_probes(inventory: list[dict]) -> None:
    public = sorted(_key(e) for e in inventory if e["auth"] == "public")
    assert public == [("GET", "/healthz"), ("GET", "/readyz")]


def test_openapi_snapshot_lists_the_schema_routes(inventory: list[dict]) -> None:
    if not OPENAPI.exists():
        pytest.skip("docs/openapi.json (guard R01) is not in this tree")
    doc = json.loads(OPENAPI.read_text(encoding="utf-8"))
    listed = {(m.upper(), p) for p, ops in doc["paths"].items() for m in ops}
    assert listed == {_key(e) for e in inventory if e["in_schema"]}
