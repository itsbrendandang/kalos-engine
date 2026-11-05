"""SKILL.md parsing, body checks, folder rules and the content hash
(kalos/skills/loader.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills.loader import (
    MAX_BODY_LINES,
    SkillFormatError,
    check_engine_refs,
    check_symbol,
    default_engine_root,
    load_dir,
    ngram_overlap,
    parse_skill,
    tree_hash,
)

FRONT = """---
schema: kalos-skill/1
slug: {slug}
name: Example skill
version: 0.1.0
kind: locked
category: analytics
summary: >-
  Use to compute an example summary of a run sheet for these tests only. Not for real data.
keywords: [example]
starting_prompt: Summarise this run sheet please.
inputs:
  - {{name: rows, type: table, description: "Rows of the run sheet."}}
outputs:
  - {{name: summary, kind: json, label: derived, min_bytes: 2, description: "A derived summary."}}
tools: [engine.noise_report]
entrypoint: "kalos.core.replicates:noise_report"
implementation:
  status: implemented
  engine:
    - {{symbol: "kalos.core.replicates:noise_report", ref: "{ref}"}}
scope: read
side_effects: none
refuses_when: [The table is empty.]
verification: {{deterministic: [same input same output]}}
author: Kalos
licence: {{spdx: Apache-2.0}}
created: 2026-11-05
updated: 2026-11-05
---
"""
SECTIONS = ["Purpose", "When to use", "When not to use", "Inputs", "Procedure", "Outputs",
            "Stop conditions", "Caveats", "Example"]


def _ref() -> str:
    from kalos.skills.loader import top_level_defs

    line = top_level_defs(default_engine_root() / "kalos/core/replicates.py")["noise_report"]
    return f"kalos/core/replicates.py:{line}"


def write_skill(root: Path, slug: str = "example-skill", *, sections=SECTIONS, extra_body: str = "",
                ref: str | None = None) -> Path:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(f"## {s}\nText for {s.lower()}.\n\n" for s in sections) + extra_body
    (d / "SKILL.md").write_text(FRONT.format(slug=slug, ref=ref or _ref()) + body, encoding="utf-8")
    return d


def test_parse_ok(tmp_path):
    sk = parse_skill(write_skill(tmp_path))
    assert sk.slug == "example-skill"
    assert sk.files == ("SKILL.md",)
    assert sk.content_hash.startswith("sha256:") and len(sk.content_hash) == 7 + 64
    assert check_engine_refs(sk) == []


def test_missing_frontmatter(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "SKILL.md").write_text("# no frontmatter\n")
    with pytest.raises(SkillFormatError):
        parse_skill(d)


def test_frontmatter_not_a_mapping(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    (d / "SKILL.md").write_text("---\n- a\n- b\n---\nbody\n")
    with pytest.raises(SkillFormatError, match="mapping"):
        parse_skill(d)


def test_crlf_and_bom_accepted(tmp_path):
    d = write_skill(tmp_path)
    text = (d / "SKILL.md").read_text()
    (d / "SKILL.md").write_bytes(("\ufeff" + text.replace("\n", "\r\n")).encode("utf-8"))
    assert parse_skill(d).slug == "example-skill"


def test_missing_section(tmp_path):
    d = write_skill(tmp_path, sections=[s for s in SECTIONS if s != "Stop conditions"])
    with pytest.raises(SkillFormatError, match="Stop conditions"):
        parse_skill(d)


def test_slug_must_equal_folder(tmp_path):
    d = write_skill(tmp_path)
    d2 = tmp_path / "other-name"
    d.rename(d2)
    with pytest.raises(SkillFormatError, match="folder name"):
        parse_skill(d2)


def test_line_cap(tmp_path):
    d = write_skill(tmp_path, extra_body="line\n" * (MAX_BODY_LINES + 1))
    with pytest.raises(SkillFormatError, match="exceeds"):
        parse_skill(d)


def test_non_text_file_rejected(tmp_path):
    d = write_skill(tmp_path)
    (d / "scripts").mkdir()
    (d / "scripts" / "run.py").write_text("print('x')\n")
    with pytest.raises(SkillFormatError, match="not allowed"):
        parse_skill(d)
    (d / "scripts" / "run.py").unlink()
    (d / "scripts").rmdir()
    (d / "references").mkdir()
    (d / "references" / "picture.png").write_bytes(b"x")
    with pytest.raises(SkillFormatError, match="non-text"):
        parse_skill(d)


@pytest.mark.parametrize("rel", [
    "tests/test_x.md", "references/expected_output.json", "examples/golden.csv",
    "references/uv.lock", "examples/requirements.txt", "references/run_skill.md",
    "references/runner.md", "examples/fixtures/a.csv", "notes.md", "SKILL.md.bak.md",
    "references/pyproject.toml", "examples/answers.json", "references/package-lock.json",
])
def test_evidence_and_machinery_rejected(tmp_path, rel):
    d = write_skill(tmp_path)
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    (d / rel).write_text("x\n")
    with pytest.raises(SkillFormatError):
        parse_skill(d)


def test_references_and_synthetic_examples_allowed(tmp_path):
    d = write_skill(tmp_path)
    (d / "references").mkdir()
    (d / "references" / "method.md").write_text("# Method\n")
    (d / "examples").mkdir()
    (d / "examples" / "synthetic_runs.csv").write_text("a,b\n1,2\n")
    assert parse_skill(d).files == ("SKILL.md", "examples/synthetic_runs.csv",
                                    "references/method.md")


def test_data_sources_md_must_match_frontmatter(tmp_path):
    from kalos.skills.loader import render_data_sources

    d = write_skill(tmp_path)
    meta = parse_skill(d).meta
    (d / "DATA_SOURCES.md").write_text(render_data_sources(meta))
    assert "None." in (d / "DATA_SOURCES.md").read_text()
    parse_skill(d)
    (d / "DATA_SOURCES.md").write_text("# hand-edited\n")
    with pytest.raises(SkillFormatError, match="DATA_SOURCES"):
        parse_skill(d)


def test_render_data_sources_rows():
    from kalos.skills.loader import render_data_sources
    from kalos.skills.schema import SkillMeta

    meta = SkillMeta.model_validate({**_meta_dict(), "licence": {"spdx": "Apache-2.0", "data_sources": [
        {"name": "A|B", "kind": "software", "version": "1.0", "uri": "https://x", "licence": "MIT",
         "commercial_status": "unknown", "bundled": True, "verified_by": "references/lic.md",
         "checked_on": "2026-11-05"}]}})
    out = render_data_sources(meta)
    assert "| A\\|B | software | 1.0 | MIT | unknown | yes |" in out
    assert render_data_sources(meta) == out


def _meta_dict():
    from kalos.skills.loader import split_frontmatter

    return split_frontmatter(FRONT.format(slug="example-skill", ref=_ref()) + "x")[0]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_symlink_rejected(tmp_path):
    d = write_skill(tmp_path)
    (tmp_path / "outside.md").write_text("secret\n")
    (d / "references").mkdir()
    (d / "references" / "link.md").symlink_to(tmp_path / "outside.md")
    with pytest.raises(SkillFormatError, match="symlink"):
        parse_skill(d)


def test_hash_stable_and_sensitive(tmp_path):
    a = write_skill(tmp_path / "a")
    b = write_skill(tmp_path / "b")
    h = parse_skill(a).content_hash
    assert parse_skill(b).content_hash == h  # same bytes, different location
    # dotfiles and caches do not count
    (b / ".DS_Store").write_bytes(b"x")
    (b / "__pycache__").mkdir()
    (b / "__pycache__" / "x.pyc").write_bytes(b"x")
    assert tree_hash(b)[0] == h
    # one byte in a reference file changes identity
    (b / "references").mkdir()
    (b / "references" / "note.md").write_text("a\n")
    h2 = parse_skill(b).content_hash
    assert h2 != h
    (b / "references" / "note.md").write_text("b\n")
    assert parse_skill(b).content_hash not in (h, h2)
    # renaming a file changes identity even with identical bytes
    (b / "references" / "note.md").rename(b / "references" / "note2.md")
    assert parse_skill(b).content_hash != h2


def test_load_dir_collects_all_errors_and_duplicates(tmp_path):
    write_skill(tmp_path, "good-one")
    bad = write_skill(tmp_path, "bad-one", sections=SECTIONS[:-1])
    assert bad.exists()
    res = load_dir(tmp_path)
    assert [s.slug for s in res.skills] == ["good-one"]
    assert len(res.errors) == 1 and "Example" in res.errors[0][1]


def test_load_dir_missing_root(tmp_path):
    res = load_dir(tmp_path / "nope")
    assert res.errors and not res.skills


def test_engine_ref_wrong_line_and_wrong_module(tmp_path):
    good = _ref()
    path, line = good.split(":")
    d = write_skill(tmp_path, ref=f"{path}:{int(line) + 1}")
    probs = check_engine_refs(parse_skill(d))
    assert probs and "ref says" in probs[0]
    d2 = write_skill(tmp_path / "x", ref=f"kalos/core/drivers.py:{line}")
    assert any("module path" in p for p in check_engine_refs(parse_skill(d2)))


def test_check_symbol_static():
    assert check_symbol("kalos.core.optimize:propose_batch", default_engine_root()) == []
    root = default_engine_root()
    assert check_symbol("kalos.data.anonymizer:Anonymizer.anonymize_meta", root) == []
    assert check_symbol("kalos.data.anonymizer:Anonymizer.nope", root)
    assert check_symbol("kalos.data.anonymizer:Anonymizer._salt", root)
    assert check_symbol("kalos.core.optimize:propose_batch.x", root)  # not a class
    assert check_symbol("kalos.core.optimize:no_such_thing", default_engine_root())
    assert check_symbol("kalos.core.nope:x", default_engine_root())


def test_ngram_overlap():
    corpus = {"a b c d e f g h i j"}
    assert ngram_overlap("x a b c d e f g h i j y", corpus) == ["a b c d e f g h i j"]
    assert ngram_overlap("a b c d e f g h i", corpus) == []
