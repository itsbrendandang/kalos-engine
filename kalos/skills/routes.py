"""Read-only HTTP routes for skills: list, search, detail, files.

Mounted by `kalos.portal.app` when the `skills` extra is installed, and served
only when `KALOS_SKILLS=1` (default off: every route answers 404, as if absent).

- Auth: `require_scope(READ)`; tenant = `principal.tenant`.
- A skill outside the caller's visibility answers 404, byte-identical to an
  unknown slug (never 403), like the campaign routes.
- Any POST/PUT/PATCH/DELETE under `/api/skills` answers 405 with
  `Allow: GET`. There is no create, edit or run route in v1.
- Errors use the engine's envelope (`{"error", "code"}`) via the app's
  `HTTPException` handler.

Kalos-original code.
"""
from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from kalos.portal.auth import READ, Principal, require_scope

from .loader import TEXT_SUFFIXES, Skill
from .registry import SkillRegistry, get_registry
from .schema import CATEGORIES

FLAG_ENV = "KALOS_SKILLS"
MAX_LIMIT = 100
MAX_FILE_BYTES = 256 * 1024
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MEDIA = {".md": "text/markdown", ".json": "application/json", ".csv": "text/csv",
          ".yaml": "text/plain", ".yml": "text/plain", ".txt": "text/plain"}

router = APIRouter()


def skills_enabled() -> bool:
    return os.environ.get(FLAG_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _require_enabled() -> None:
    if not skills_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


def get_skill_registry() -> SkillRegistry:
    """Dependency (overridable in tests)."""
    return get_registry()


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="skill not found")


def _summary(s: Skill) -> dict[str, Any]:
    m = s.meta
    return {
        "slug": m.slug, "name": m.name, "version": m.version, "kind": m.kind,
        "category": m.category, "summary": m.summary, "status": m.status,
        "implementation": m.implementation.status, "owner": m.owner, "source": s.source,
        "scope": m.scope, "tools": list(m.tools), "content_hash": s.content_hash,
        "updated": m.updated.isoformat(),
    }


def _files(s: Skill) -> list[dict[str, Any]]:
    return [{"path": f, "size": (s.path / f).stat().st_size,
             "type": _MEDIA.get(PurePosixPath(f).suffix.lower(), "text/plain")}
            for f in s.files if PurePosixPath(f).suffix.lower() in TEXT_SUFFIXES]


def _get(reg: SkillRegistry, slug: str, tenant: str) -> Skill:
    if not _SLUG_RE.match(slug) or len(slug) > 64:
        raise _not_found()
    s = reg.get(slug, tenant)
    if s is None:
        raise _not_found()
    return s


@router.get("/api/skills", dependencies=[Depends(_require_enabled)])
def list_skills(
    category: str | None = Query(default=None, max_length=32),
    kind: str | None = Query(default=None, pattern="^(locked|guided)$"),
    status: str | None = Query(default=None, pattern="^(active|beta|planned|deprecated)$"),
    implementation: str | None = Query(default=None,
                                       pattern="^(implemented|partial|needs-new-code)$"),
    source: str | None = Query(default=None, pattern="^(kalos|team)$"),
    include_beta: bool = Query(default=False),
    include_planned: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=MAX_LIMIT),
    cursor: str | None = Query(default=None, max_length=64),
    principal: Principal = Depends(require_scope(READ)),
    reg: SkillRegistry = Depends(get_skill_registry),
) -> dict[str, Any]:
    """The caller's visible skills, by slug. `cursor` is the last slug of the
    previous page."""
    if category is not None and category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="unknown category")
    if cursor is not None and not _SLUG_RE.match(cursor):
        raise HTTPException(status_code=400, detail="invalid cursor")
    rows = reg.list_skills(principal.tenant, category=category, kind=kind, status=status,
                           implementation=implementation, source=source,
                           include_beta=include_beta, include_planned=include_planned)
    if cursor is not None:
        rows = [s for s in rows if s.slug > cursor]
    page = rows[:limit]
    more = len(rows) > limit
    return {"skills": [_summary(s) for s in page],
            "next_cursor": page[-1].slug if more and page else None}


@router.get("/api/skills/search", dependencies=[Depends(_require_enabled)])
def search_skills(
    q: str = Query(min_length=1, max_length=500),
    category: str | None = Query(default=None, max_length=32),
    include_beta: bool = Query(default=False),
    include_planned: bool = Query(default=False),
    limit: int = Query(default=10, ge=1, le=MAX_LIMIT),
    principal: Principal = Depends(require_scope(READ)),
    reg: SkillRegistry = Depends(get_skill_registry),
) -> dict[str, Any]:
    """Deterministic weighted search (docs/SKILLS.md, "Search")."""
    if category is not None and category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="unknown category")
    hits = reg.search(q, principal.tenant, category=category, include_beta=include_beta,
                      include_planned=include_planned, limit=limit)
    return {"results": [{"score": score, "skill": _summary(s)} for score, s in hits]}


@router.get("/api/skills/{slug}", dependencies=[Depends(_require_enabled)])
def get_skill(
    slug: str,
    request: Request,
    principal: Principal = Depends(require_scope(READ)),
    reg: SkillRegistry = Depends(get_skill_registry),
) -> Response:
    s = _get(reg, slug, principal.tenant)
    etag = f'"{s.content_hash}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    body = {**_summary(s),
            "frontmatter": s.meta.model_dump(mode="json", by_alias=True),
            "body": s.body, "files": _files(s)}
    return JSONResponse(body, headers=headers)


@router.get("/api/skills/{slug}/files", dependencies=[Depends(_require_enabled)])
def list_files(
    slug: str,
    principal: Principal = Depends(require_scope(READ)),
    reg: SkillRegistry = Depends(get_skill_registry),
) -> dict[str, Any]:
    return {"files": _files(_get(reg, slug, principal.tenant))}


@router.get("/api/skills/{slug}/files/{path:path}", dependencies=[Depends(_require_enabled)])
def get_file(
    slug: str,
    path: str,
    principal: Principal = Depends(require_scope(READ)),
    reg: SkillRegistry = Depends(get_skill_registry),
) -> Response:
    """One text file of the skill. Only paths that are part of the hashed
    package are served: no `..`, no absolute paths, no symlinks, text suffixes
    only, at most 256 KB."""
    s = _get(reg, slug, principal.tenant)
    p = PurePosixPath(path)
    if (path.startswith("/") or "\\" in path or "\x00" in path or ".." in p.parts
            or path not in s.files or p.suffix.lower() not in TEXT_SUFFIXES):
        raise HTTPException(status_code=404, detail="file not found")
    full = s.path / path
    if full.is_symlink() or not full.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        full.resolve().relative_to(s.path.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="file not found") from None
    if full.stat().st_size > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    media = _MEDIA.get(p.suffix.lower(), "text/plain")
    return Response(full.read_bytes(), media_type=f"{media}; charset=utf-8",
                    headers={"ETag": f'"{s.content_hash}"',
                             "X-Content-Type-Options": "nosniff"})


@router.get("/api/skills/{rest:path}", include_in_schema=False,
            dependencies=[Depends(_require_enabled)])
def unknown_path(rest: str) -> None:
    """Any other GET under /api/skills: 404 (and not the 405 of the write
    catch-all below, which would otherwise match the path)."""
    raise HTTPException(status_code=404, detail="Not Found")


# Auth runs before the 405, so an unauthenticated write gets 401 like every
# other route (and learns nothing about which paths exist).
_WRITE_DEPS = [Depends(_require_enabled), Depends(require_scope(READ))]


@router.api_route("/api/skills", methods=["POST", "PUT", "PATCH", "DELETE"],
                  include_in_schema=False, dependencies=_WRITE_DEPS)
@router.api_route("/api/skills/{rest:path}", methods=["POST", "PUT", "PATCH", "DELETE"],
                  include_in_schema=False, dependencies=_WRITE_DEPS)
def no_writes(rest: str = "") -> None:
    """Skills are read-only in v1: no create, edit, delete or run route."""
    raise HTTPException(status_code=405, detail="skills are read-only",
                        headers={"Allow": "GET"})


__all__ = ["FLAG_ENV", "get_skill_registry", "router", "skills_enabled"]
