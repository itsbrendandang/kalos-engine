"""The agent and contributor rules ship with the repo: AGENTS.md exists with its
sections, CLAUDE.md points at it, and neither cites paths outside the repo."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_claude_md_imports_agents_md() -> None:
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"


def test_agents_md_has_the_rules_agent_context_and_team_rules() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert headings[0] == "1. More real (honesty)"
    assert "9. Agent context" in headings
    assert "10. Team rules" in headings
    for layer in ("core", "analysis", "campaigns", "storage", "api"):
        assert re.search(rf"^- Layers:.*\b{layer}\b", text, flags=re.MULTILINE), layer
    assert "scripts/check.sh" in text


def test_agent_files_cite_no_private_paths() -> None:
    for name in ("AGENTS.md", "CLAUDE.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert ".kalos-dev" not in text, name


def test_agent_context_cites_files_that_exist() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    text = text.split("## 9. Agent context", 1)[1].split("\n## ", 1)[0]
    cited = set(re.findall(r"\b((?:docs|deploy|scripts|tests)/[A-Za-z0-9_./-]+\.(?:md|json|sh|py))\b", text))
    assert cited, "expected the agent context to cite repo files"
    missing = sorted(p for p in cited if not (ROOT / p).exists())
    assert not missing, missing
