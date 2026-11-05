"""Kalos skills: typed, versioned, hash-identified skill packages.

A skill is a folder with a `SKILL.md` (YAML frontmatter in the `kalos-skill/1`
format plus a markdown body). The Kalos-curated set ships in
`kalos/skills/builtin/`. See docs/SKILLS.md.

Importing this package must never load torch: engine symbols are referenced by
string and imported only when a tool is first called
(tests/test_skills_builtin.py checks this in a fresh interpreter).
"""
from __future__ import annotations

from .schema import SkillMeta

__all__ = ["SkillMeta"]
