"""Keep the local skill, its mirror and its provenance lock in sync."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_skill_sources_mirrors_and_hashes() -> None:
    lock = json.loads((ROOT / "skills-lock.json").read_text())
    assert lock["version"] == 2
    assert set(lock["skills"]) == {"kalos-assist-review"}
    for name, entry in lock["skills"].items():
        assert entry["source"] == "itsbrendandang/kalos-engine"
        assert entry["sourceType"] == "github"
        assert entry["installMode"] == "copy"
        assert entry["attribution"]
        assert entry["skillPath"] == f".agents/skills/{name}/SKILL.md"
        source = (ROOT / entry["skillPath"]).read_bytes()
        assert source == (ROOT / ".claude/skills" / name / "SKILL.md").read_bytes()
        assert hashlib.sha256(source).hexdigest() == entry["computedHash"]
        frontmatter = source.decode().split("---", 2)
        assert frontmatter[0] == ""
        assert f"name: {name}" in frontmatter[1].splitlines()
        assert re.search(r"^description: \S.+$", frontmatter[1], re.MULTILINE)


def test_review_skill_cites_existing_repo_paths() -> None:
    text = (ROOT / ".agents/skills/kalos-assist-review/SKILL.md").read_text()
    paths = re.findall(r"`((?:kalos|tests|scripts|changes)/[A-Za-z0-9_./-]+)\`", text)
    assert paths
    for path in paths:
        assert (ROOT / path).exists(), path
    for path in ("AGENTS.md", "docs/ASSISTANT.md", "scripts/check.sh",
                 "tests/test_web_contract.py", "tests/test_agents_md.py", "tests/test_agent_skills.py"):
        assert (ROOT / path).is_file(), path
