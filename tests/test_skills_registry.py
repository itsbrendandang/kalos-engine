"""Registry visibility and listing (kalos/skills/registry.py): Kalos skills for
every tenant, Team skills only for their tenant, user skills never in v1, and
listing filters with beta/planned hidden by default."""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")

from kalos.skills import registry as registry_mod
from kalos.skills.registry import BUILTIN_DIR, SkillRegistry, get_registry


def team_skill(root: Path, tenant: str, slug: str = "acme-titer-notes", *, owner: str | None = None,
               visibility: str = "tenant") -> Path:
    """A Team skill made from a built-in: copy, then rewrite slug/owner/visibility."""
    src = BUILTIN_DIR / "driver-ranking"
    dst = root / tenant / slug
    shutil.copytree(src, dst)
    p = dst / "SKILL.md"
    text = p.read_text(encoding="utf-8")
    text = text.replace("slug: driver-ranking", f"slug: {slug}", 1)
    text = text.replace("owner: system", f"owner: {owner or 'tenant:' + tenant}", 1)
    text = text.replace("visibility: system", f"visibility: {visibility}", 1)
    p.write_text(text, encoding="utf-8")
    return dst


def test_team_skill_visible_only_to_its_tenant(tmp_path):
    team_skill(tmp_path, "acme")
    reg = SkillRegistry.from_dirs(BUILTIN_DIR, tmp_path)
    assert not reg.errors, reg.errors
    s = reg.get("acme-titer-notes", "acme")
    assert s is not None and s.source == "team" and s.meta.tenant == "acme"
    assert reg.get("acme-titer-notes", "globex") is None
    assert "acme-titer-notes" not in {x.slug for x in reg.visible("globex")}
    # Kalos skills stay visible to both
    assert reg.get("driver-ranking", "globex") is not None
    assert {x.slug for x in reg.list_skills("acme", source="team")} == {"acme-titer-notes"}
    # and search never leaks across tenants
    assert all(x.slug != "acme-titer-notes" for _, x in reg.search("drivers", "globex"))
    assert any(x.slug == "acme-titer-notes" for _, x in reg.search("drivers", "acme"))


def test_team_skill_owner_must_match_folder(tmp_path):
    team_skill(tmp_path, "acme", owner="tenant:globex")
    reg = SkillRegistry.from_dirs(BUILTIN_DIR, tmp_path)
    assert reg.get("acme-titer-notes", "acme") is None
    assert reg.get("acme-titer-notes", "globex") is None
    assert any("owner must be tenant:acme" in m for _, m in reg.errors)


def test_team_skill_cannot_shadow_kalos_skill(tmp_path):
    team_skill(tmp_path, "acme", slug="driver-ranking")
    reg = SkillRegistry.from_dirs(BUILTIN_DIR, tmp_path)
    assert reg.get("driver-ranking", "acme").source == "kalos"
    assert any("collides" in m for _, m in reg.errors)


def test_user_owned_skill_never_visible_in_v1(tmp_path):
    team_skill(tmp_path, "acme", owner="user:bob", visibility="private")
    reg = SkillRegistry.from_dirs(None, tmp_path)
    # refused (owner does not match the tenant folder) and invisible either way
    assert reg.get("acme-titer-notes", "acme") is None


def test_unknown_slug_and_hidden_return_none():
    reg = SkillRegistry.from_dirs(BUILTIN_DIR)
    assert reg.get("no-such-skill", "acme") is None


def test_list_filters_and_defaults():
    reg = SkillRegistry.from_dirs(BUILTIN_DIR)
    default = {s.slug for s in reg.list_skills("t0")}
    assert all(reg.get(s, "t0").meta.status == "active" for s in default)
    with_beta = {s.slug for s in reg.list_skills("t0", include_beta=True)}
    assert "virtual-run-simulation" in with_beta - default
    planned = {s.slug for s in reg.list_skills("t0", status="planned")}
    assert planned == {"dataset-licence-check", "distance-to-data", "hplc-peak-summary",
                       "space-filling-design", "time-series-endpoint-extraction"}
    assert not planned & with_beta
    assert {s.slug for s in reg.list_skills("t0", category="scale-up")} == {"scale-up-envelope-check"}
    assert {s.meta.kind for s in reg.list_skills("t0", kind="locked")} == {"locked"}
    assert reg.list_skills("t0", kind="guided") == []
    impl = reg.list_skills("t0", implementation="partial", include_beta=True)
    assert {s.slug for s in impl} == {"campaign-provenance-report", "trust-report",
                                      "virtual-run-simulation"}


def test_search_hides_planned_by_default():
    reg = SkillRegistry.from_dirs(BUILTIN_DIR)
    assert all(s.meta.status == "active" for _, s in reg.search("titer from chromatogram", "t0"))
    top = reg.search("titer from chromatogram", "t0", include_planned=True)[0][1]
    assert top.slug == "hplc-peak-summary"


def test_get_registry_reads_env(tmp_path, monkeypatch):
    team_skill(tmp_path, "acme")
    registry_mod.reset_cache()
    monkeypatch.setenv("KALOS_SKILLS_DIR", str(tmp_path))
    try:
        assert get_registry().get("acme-titer-notes", "acme") is not None
        monkeypatch.delenv("KALOS_SKILLS_DIR")
        assert get_registry().get("acme-titer-notes", "acme") is None
    finally:
        registry_mod.reset_cache()


def test_bad_tenant_folder_name_refused(tmp_path):
    (tmp_path / "bad tenant!").mkdir()
    reg = SkillRegistry.from_dirs(None, tmp_path)
    assert reg.errors


def test_cross_tenant_duplicate_slug_is_refused_by_name(tmp_path):
    team_skill(tmp_path, "acme", slug="shared-name")
    team_skill(tmp_path, "globex", slug="shared-name")
    reg = SkillRegistry.from_dirs(BUILTIN_DIR, tmp_path)
    # acme loads first (sorted); globex's copy is refused, never served to anyone
    assert reg.get("shared-name", "acme").meta.owner == "tenant:acme"
    assert reg.get("shared-name", "globex") is None
    assert any(slug == "shared-name" and m.startswith("DuplicateSlugError:")
               and "tenant:acme" in m and "tenant:globex" in m for slug, m in reg.errors)


def test_constructor_refuses_team_copy_of_kalos_slug():
    from kalos.skills.loader import load_dir

    base = {s.slug: s for s in load_dir(BUILTIN_DIR).skills}
    dr = base["driver-ranking"]
    shadow = dataclasses.replace(dr, meta=dr.meta.model_copy(
        update={"owner": "tenant:acme", "visibility": "tenant"}), source="team")
    reg = SkillRegistry([*base.values(), shadow])
    assert reg.get("driver-ranking", "acme").meta.owner == "system"
    assert any(m.startswith("DuplicateSlugError:") for _, m in reg.errors)


def test_lookup_is_by_owner_then_system(tmp_path):
    team_skill(tmp_path, "acme", slug="acme-only")
    reg = SkillRegistry.from_dirs(BUILTIN_DIR, tmp_path)
    assert reg.get("acme-only", "acme").source == "team"
    assert reg.get("driver-ranking", "acme").source == "kalos"
    assert reg.get("acme-only", "globex") is None
    assert len(reg.all()) == 20
