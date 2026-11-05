"""Read-only, tenant-scoped skill registry: visibility, get, list, search.

Sources in v1:

- **Kalos** - the curated set shipped in `kalos/skills/builtin/`
  (`owner: system`), visible to every tenant.
- **Team** - `owner: tenant:<id>` skills read from `$KALOS_SKILLS_DIR/<id>/`,
  visible only to that tenant. The schema and loader support them; there is no
  route to create one (no tenant authoring in v1). A Team skill whose owner does
  not match its folder's tenant, or whose slug collides with a Kalos skill, is
  refused at load time.

`user:<id>` (Mine) skills are a v2 concept and are never visible in v1.

Search is deterministic so routing can be tested: query tokens (>= 3 chars,
minus stopwords, 5-character prefix stem) are matched against weighted fields
(slug+name x5, keywords x3, category x2, summary x1.5, body x0.25); ties break by
slug. Kalos-original code.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from .loader import LoadResult, Skill, load_dir

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"
SKILLS_DIR_ENV = "KALOS_SKILLS_DIR"
_TENANT_DIR_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")

STOPWORDS = frozenset(
    ["the", "and", "for", "with", "this", "that", "will", "can", "how", "are", "our", "from", "into", "what", "which", "use", "not", "its", "you", "your", "all", "any", "was", "has", "have", "been", "these", "those", "them", "they", "then", "than", "when", "where", "why", "who", "whom", "does", "did", "should", "would", "could", "about", "over", "under", "please", "want", "need"]
)
FIELD_WEIGHTS = (("name", 5.0), ("keywords", 3.0), ("category", 2.0), ("summary", 1.5),
                 ("body", 0.25))
_WORD = re.compile(r"[a-z0-9]+")


def _stem(t: str) -> str:
    return t[:5]


def tokens(text: str, *, drop_stopwords: bool = False) -> set[str]:
    return {
        _stem(t) for t in _WORD.findall(text.lower())
        if len(t) >= 3 and not (drop_stopwords and t in STOPWORDS)
    }


class SkillRegistry:
    """In-memory and immutable after construction."""

    def __init__(self, skills: Iterable[Skill], errors: list[tuple[str, str]] | None = None):
        self._by_slug: dict[str, Skill] = {}
        self.errors: list[tuple[str, str]] = list(errors or [])
        for s in skills:
            if s.slug in self._by_slug:
                self.errors.append((s.slug, "duplicate slug; later copy ignored"))
                continue
            self._by_slug[s.slug] = s
        # Precompute field token sets for search.
        self._fields: dict[str, dict[str, set[str]]] = {
            slug: {
                "name": tokens(s.meta.slug.replace("-", " ") + " " + s.meta.name),
                "keywords": tokens(" ".join(s.meta.keywords)),
                "category": tokens(s.meta.category.replace("-", " ")),
                "summary": tokens(s.meta.summary),
                "body": tokens(s.body),
            }
            for slug, s in self._by_slug.items()
        }

    # --- construction -------------------------------------------------------- #

    @classmethod
    def from_dirs(cls, builtin: Path | None = BUILTIN_DIR,
                  team_root: Path | None = None) -> SkillRegistry:
        res = LoadResult()
        if builtin is not None:
            b = load_dir(builtin, source="kalos")
            res.errors += b.errors
            for s in b.skills:
                if s.meta.owner != "system":
                    res.errors.append((s.slug, "a built-in skill must have owner=system"))
                else:
                    res.skills.append(s)
        system = {s.slug for s in res.skills}
        if team_root is not None and Path(team_root).is_dir():
            for tdir in sorted(p for p in Path(team_root).iterdir() if p.is_dir()):
                if not _TENANT_DIR_RE.match(tdir.name) or tdir.is_symlink():
                    res.errors.append((str(tdir), "not a tenant folder name"))
                    continue
                t = load_dir(tdir, source="team")
                res.errors += t.errors
                for s in t.skills:
                    if s.meta.owner != f"tenant:{tdir.name}":
                        res.errors.append(
                            (str(s.path), f"owner must be tenant:{tdir.name} in this folder"))
                    elif s.slug in system:
                        res.errors.append((str(s.path), "slug collides with a Kalos skill"))
                    else:
                        res.skills.append(s)
        return cls(res.skills, res.errors)

    # --- visibility ---------------------------------------------------------- #

    @staticmethod
    def _visible_to(s: Skill, tenant: str) -> bool:
        owner = s.meta.owner
        return owner == "system" or owner == f"tenant:{tenant}"

    def all(self) -> list[Skill]:
        """Every loaded skill, for tooling only (not tenant-filtered)."""
        return [self._by_slug[k] for k in sorted(self._by_slug)]

    def visible(self, tenant: str) -> list[Skill]:
        return [s for s in self.all() if self._visible_to(s, tenant)]

    def get(self, slug: str, tenant: str) -> Skill | None:
        """The skill, or None when it does not exist OR is not visible to this
        tenant (callers answer 404 for both, never 403)."""
        s = self._by_slug.get(slug)
        return s if s is not None and self._visible_to(s, tenant) else None

    # --- listing ------------------------------------------------------------- #

    @staticmethod
    def _status_ok(s: Skill, status: str | None, include_beta: bool,
                   include_planned: bool) -> bool:
        st = s.meta.status
        if status is not None:
            return st == status
        if st == "beta":
            return include_beta
        if st == "planned":
            return include_planned
        return st != "deprecated"

    def list_skills(self, tenant: str, *, category: str | None = None, kind: str | None = None,
             status: str | None = None, implementation: str | None = None,
             source: str | None = None, include_beta: bool = False,
             include_planned: bool = False) -> list[Skill]:
        out = []
        for s in self.visible(tenant):
            m = s.meta
            if category and m.category != category:
                continue
            if kind and m.kind != kind:
                continue
            if implementation and m.implementation.status != implementation:
                continue
            if source and s.source != source:
                continue
            if not self._status_ok(s, status, include_beta, include_planned):
                continue
            out.append(s)
        return out

    # --- search -------------------------------------------------------------- #

    def search(self, q: str, tenant: str, *, category: str | None = None,
               include_beta: bool = False, include_planned: bool = False,
               limit: int | None = None) -> list[tuple[float, Skill]]:
        qt = tokens(q, drop_stopwords=True)
        if not qt:
            return []
        out: list[tuple[float, Skill]] = []
        for s in self.list_skills(tenant, category=category, include_beta=include_beta,
                           include_planned=include_planned):
            f = self._fields[s.slug]
            score = sum(w * len(qt & f[name]) for name, w in FIELD_WEIGHTS)
            if score > 0:
                out.append((round(score, 4), s))
        out.sort(key=lambda x: (-x[0], x[1].slug))
        return out[:limit] if limit is not None else out


@lru_cache(maxsize=1)
def load_builtin() -> SkillRegistry:
    """The Kalos-curated set only (cached for the process)."""
    return SkillRegistry.from_dirs(BUILTIN_DIR, None)


@lru_cache(maxsize=8)
def _registry_for(team_root: str) -> SkillRegistry:
    return SkillRegistry.from_dirs(BUILTIN_DIR, Path(team_root) if team_root else None)


def get_registry() -> SkillRegistry:
    """Kalos skills plus Team skills from `$KALOS_SKILLS_DIR` (when set).
    Cached per directory; restart (or `reset_cache`) to pick up file changes."""
    return _registry_for(os.environ.get(SKILLS_DIR_ENV, "").strip())


def reset_cache() -> None:
    load_builtin.cache_clear()
    _registry_for.cache_clear()


__all__ = [
    "BUILTIN_DIR",
    "SKILLS_DIR_ENV",
    "SkillRegistry",
    "get_registry",
    "load_builtin",
    "reset_cache",
    "tokens",
]
