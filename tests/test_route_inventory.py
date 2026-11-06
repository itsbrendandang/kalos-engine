"""The routes desktop-engine-2 adds exist with their methods and scopes.

Branch arch-safe-now pins the whole route list in docs/openapi.json
(tests/test_openapi_drift.py there). This branch is not merged with it, so this
small inventory stands in for the two routes added here until the merge
regenerates that spec.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.routing import APIRoute

from kalos.portal.app import app

EXPECTED = {
    ("/api/preflight", "POST"): "write",
    ("/api/campaigns/{campaign_id}/rows", "GET"): "read",
}


def _scopes(route: APIRoute) -> set[str]:
    """The scopes `require_scope(...)` demands anywhere in the route's dependencies."""
    out: set[str] = set()
    stack = [route.dependant]
    while stack:
        dep = stack.pop()
        call = dep.call
        if call is not None and getattr(call, "__qualname__", "").startswith("require_scope."):
            names = call.__code__.co_freevars
            cells = call.__closure__ or ()
            out |= {c.cell_contents for n, c in zip(names, cells) if n == "scope"}
        stack.extend(dep.dependencies)
    return out


def _api_routes(routes, prefix: str = ""):
    """Every APIRoute with its full path. FastAPI 0.16x keeps an included
    router as one entry (`original_router`, `include_context.prefix`)."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield prefix + route.path, route
        inner = getattr(route, "original_router", None)
        if inner is not None:
            ctx = getattr(route, "include_context", None)
            yield from _api_routes(inner.routes, prefix + (getattr(ctx, "prefix", "") or ""))


def _inventory() -> dict[tuple[str, str], set[str]]:
    return {(path, m): _scopes(r) for path, r in _api_routes(app.routes) for m in r.methods}


def test_desktop_engine_2_routes_exist_with_their_methods_and_scopes():
    found = _inventory()
    for key, scope in EXPECTED.items():
        assert key in found, f"missing route {key}"
        assert found[key] == {scope}, (key, found[key])


def test_the_scope_reader_sees_scopes():
    # Guards the helper: a known write route and a known read route.
    found = _inventory()
    assert found[("/api/run", "POST")] == {"write"}
    assert found[("/api/campaigns/{campaign_id}/upload", "POST")] == {"write"}
    assert found[("/api/latest", "GET")] == {"read"}
