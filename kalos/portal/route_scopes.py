"""Per-route scope manifest for the portal app (security review S1).

`build_manifest(app)` walks every FastAPI route and reports which scope(s) its
dependency tree enforces, read back from the marker `require_scope` puts on its
dependency (`kalos.portal.auth.scope_of`). It never guesses from the HTTP
method. A route is `auth: "required"` when its tree contains
`require_principal` (directly or through `require_scope`), else `"public"`.

Scopes enforced inside an endpoint body (e.g. `admin` to archive, via
`PATCH /api/campaigns/{campaign_id}`) cannot be seen in the dependency tree;
endpoints declare them with `@conditional_scope(scope, when)` and they appear
as `conditional_scopes`.

The checked-in copy is docs/route-scopes.json, written by
`scripts/export_scope_manifest.py`; tests/test_route_scopes.py fails on drift.
Import-time cost: none beyond the app itself (no torch).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from kalos.portal.auth import CONDITIONAL_SCOPES_ATTR, require_principal, scope_of

_SCOPE_ORDER = {"read": 0, "write": 1, "admin": 2}


def _walk(routes: Sequence[Any], prefix: str, extra: tuple[Any, ...]) -> Iterator[tuple[str, APIRoute, tuple[Any, ...]]]:
    """Yield (full path, route, router-level dependency callables). Handles
    both FastAPI layouts: routes copied into the app (older releases) and
    lazily included routers (`original_router` + `include_context`, 0.16x)."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield prefix + route.path, route, extra
            continue
        original = getattr(route, "original_router", None)
        ctx = getattr(route, "include_context", None)
        if original is not None and ctx is not None:
            deps = tuple(d.dependency for d in (ctx.dependencies or []) if d.dependency is not None)
            yield from _walk(original.routes, prefix + (ctx.prefix or ""), extra + deps)


def _calls(dependant: Dependant) -> Iterator[Any]:
    for sub in dependant.dependencies:
        if sub.call is not None:
            yield sub.call
        yield from _calls(sub)


def _route_entries(path: str, route: APIRoute, extra: tuple[Any, ...]) -> list[dict[str, Any]]:
    calls = [*extra, *_calls(route.dependant)]
    scopes = sorted({s for c in calls if (s := scope_of(c)) is not None}, key=_SCOPE_ORDER.__getitem__)
    authed = bool(scopes) or any(c is require_principal for c in calls)
    conditional = [
        {"scope": scope, "when": when} for scope, when in getattr(route.endpoint, CONDITIONAL_SCOPES_ATTR, ())
    ]
    out = []
    for method in sorted(route.methods or ()):
        entry: dict[str, Any] = {
            "method": method,
            "path": path,
            "scopes": scopes,
            "auth": "required" if authed else "public",
        }
        if conditional:
            entry["conditional_scopes"] = conditional
        out.append(entry)
    return out


def build_manifest(app: FastAPI) -> list[dict[str, Any]]:
    """Every API route as `{method, path, scopes, auth[, conditional_scopes]}`,
    sorted by (path, method). Raises if a route enforces a scope without
    authentication (would indicate a broken marker)."""
    app_deps = tuple(d.dependency for d in (app.router.dependencies or []) if d.dependency is not None)
    entries: list[dict[str, Any]] = []
    for path, route, extra in _walk(app.router.routes, "", app_deps):
        entries.extend(_route_entries(path, route, extra))
    entries.sort(key=lambda e: (e["path"], e["method"]))
    seen: set[tuple[str, str]] = set()
    for e in entries:
        key = (e["method"], e["path"])
        if key in seen:
            raise RuntimeError(f"duplicate route in manifest: {key}")
        seen.add(key)
    return entries


__all__ = ["build_manifest"]
