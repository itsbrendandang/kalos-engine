"""Read-only skill routes (kalos/skills/routes.py): list, search, detail, files;
tenant scoping with 404 (never 403) outside the caller's visibility; 405 on
every write; path traversal; ETag; feature flag off -> 404."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("yaml")
from fastapi.testclient import TestClient

from kalos.portal.app import app
from kalos.skills import registry as registry_mod
from kalos.skills.registry import BUILTIN_DIR, SkillRegistry
from kalos.skills.routes import get_skill_registry


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("c-none"), "subject": "x@acme", "tenant": "acme", "scopes": []},
]
A = {"Authorization": "Bearer a-read"}
B = {"Authorization": "Bearer b-admin"}
TEAM_SLUG = "acme-secret-recipe-notes"


def _team(root):
    dst = root / "acme" / TEAM_SLUG
    shutil.copytree(BUILTIN_DIR / "driver-ranking", dst)
    p = dst / "SKILL.md"
    t = p.read_text(encoding="utf-8")
    t = t.replace("slug: driver-ranking", f"slug: {TEAM_SLUG}", 1)
    t = t.replace("owner: system", "owner: tenant:acme", 1)
    t = t.replace("visibility: system", "visibility: tenant", 1)
    p.write_text(t, encoding="utf-8")
    (dst / "references").mkdir()
    (dst / "references" / "notes.md").write_text("# Notes\nAcme only.\n", encoding="utf-8")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setenv("KALOS_SKILLS", "1")
    _team(tmp_path / "team")
    reg = SkillRegistry.from_dirs(BUILTIN_DIR, tmp_path / "team")
    assert not reg.errors
    app.dependency_overrides[get_skill_registry] = lambda: reg
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_skill_registry, None)
        registry_mod.reset_cache()


def test_list_default_hides_beta_and_planned(client):
    r = client.get("/api/skills", headers=A)
    assert r.status_code == 200
    body = r.json()
    assert body["next_cursor"] is None
    statuses = {s["status"] for s in body["skills"]}
    assert statuses == {"active"}
    slugs = [s["slug"] for s in body["skills"]]
    assert slugs == sorted(slugs) and "run-sheet-checks" in slugs
    s0 = body["skills"][0]
    assert set(s0) >= {"slug", "name", "version", "kind", "category", "summary", "status",
                       "implementation", "owner", "source", "content_hash", "updated"}


def test_list_filters_and_paging(client):
    r = client.get("/api/skills", params={"status": "planned"}, headers=A).json()
    assert len(r["skills"]) == 5 and all(s["implementation"] == "needs-new-code" for s in r["skills"])
    page1 = client.get("/api/skills", params={"limit": 3, "include_beta": True,
                                              "include_planned": True}, headers=A).json()
    assert len(page1["skills"]) == 3 and page1["next_cursor"] == page1["skills"][-1]["slug"]
    seen = [s["slug"] for s in page1["skills"]]
    cursor = page1["next_cursor"]
    while cursor:
        p = client.get("/api/skills", params={"limit": 3, "cursor": cursor, "include_beta": True,
                                              "include_planned": True}, headers=A).json()
        seen += [s["slug"] for s in p["skills"]]
        cursor = p["next_cursor"]
    assert len(seen) == len(set(seen)) == 20  # 19 Kalos + 1 acme Team skill
    assert client.get("/api/skills", params={"category": "nope"}, headers=A).status_code == 400
    assert client.get("/api/skills", params={"cursor": "../x"}, headers=A).status_code == 400
    team = client.get("/api/skills", params={"source": "team"}, headers=A).json()["skills"]
    assert [s["slug"] for s in team] == [TEAM_SLUG]


def test_team_skill_is_tenant_scoped_everywhere(client):
    # acme sees it
    assert client.get(f"/api/skills/{TEAM_SLUG}", headers=A).status_code == 200
    # globex: 404 byte-identical to an unknown slug, in list, search, detail and files
    other = client.get(f"/api/skills/{TEAM_SLUG}", headers=B)
    missing = client.get("/api/skills/no-such-skill", headers=B)
    assert other.status_code == missing.status_code == 404
    assert other.content == missing.content
    assert other.json() == {"error": "skill not found", "code": "not_found"}
    assert client.get(f"/api/skills/{TEAM_SLUG}/files", headers=B).status_code == 404
    assert client.get(f"/api/skills/{TEAM_SLUG}/files/SKILL.md", headers=B).status_code == 404
    lst = client.get("/api/skills", params={"source": "team"}, headers=B).json()["skills"]
    assert lst == []
    hits = client.get("/api/skills/search", params={"q": "acme secret recipe drivers"},
                      headers=B).json()["results"]
    assert TEAM_SLUG not in {h["skill"]["slug"] for h in hits}
    assert "acme-secret" not in client.get("/api/skills", headers=B).text


def test_search(client):
    r = client.get("/api/skills/search", params={"q": "how noisy are replicates"}, headers=A)
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["skill"]["slug"] == "replicate-noise-estimate" and res[0]["score"] > 0
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)
    planned = client.get("/api/skills/search", params={"q": "titer from chromatogram",
                                                       "include_planned": True}, headers=A)
    assert planned.json()["results"][0]["skill"]["slug"] == "hplc-peak-summary"
    assert client.get("/api/skills/search", params={"q": ""}, headers=A).status_code == 422


def test_detail_etag_and_body(client):
    r = client.get("/api/skills/bo-next-batch", headers=A)
    assert r.status_code == 200
    d = r.json()
    assert r.headers["etag"] == f'"{d["content_hash"]}"'
    assert d["frontmatter"]["schema"] == "kalos-skill/1"
    assert "## Purpose" in d["body"]
    assert d["files"] == [{"path": "SKILL.md", "size": d["files"][0]["size"], "type": "text/markdown"}]
    again = client.get("/api/skills/bo-next-batch", headers={**A, "If-None-Match": r.headers["etag"]})
    assert again.status_code == 304


def test_files(client):
    r = client.get(f"/api/skills/{TEAM_SLUG}/files", headers=A).json()["files"]
    assert {f["path"] for f in r} == {"SKILL.md", "references/notes.md"}
    f = client.get(f"/api/skills/{TEAM_SLUG}/files/references/notes.md", headers=A)
    assert f.status_code == 200 and f.text.startswith("# Notes")
    assert f.headers["content-type"].startswith("text/markdown")


@pytest.mark.parametrize("path", [
    "%2e%2e/driver-ranking/SKILL.md", "references/%2e%2e/SKILL.md", "%2e%2e/%2e%2e/etc/passwd",
    "/etc/passwd", "references/missing.md", "references", "SKILL.md%00.txt",
    "..%2F..%2Fpyproject.toml",
])
def test_file_traversal_refused(client, path):
    r = client.get(f"/api/skills/{TEAM_SLUG}/files/{path}", headers=A)
    assert r.status_code == 404
    assert "error" in r.json()


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
@pytest.mark.parametrize("url", ["/api/skills", "/api/skills/bo-next-batch",
                                 "/api/skills/bo-next-batch/run", "/api/skills/bo-next-batch/files/SKILL.md"])
def test_writes_are_405(client, method, url):
    r = getattr(client, method)(url, headers=B)
    assert r.status_code == 405
    assert r.json() == {"error": "skills are read-only", "code": "method_not_allowed"}
    assert r.headers["allow"] == "GET"


def test_unknown_get_path_is_404_not_405(client):
    r = client.get("/api/skills/bo-next-batch/nope/deeper", headers=A)
    assert r.status_code == 404


def test_auth_required(client):
    assert client.get("/api/skills").status_code == 401
    assert client.get("/api/skills", headers={"Authorization": "Bearer c-none"}).status_code == 403


def test_flag_off_is_404_everywhere(client, monkeypatch):
    monkeypatch.delenv("KALOS_SKILLS")
    for url in ["/api/skills", "/api/skills/search?q=x", "/api/skills/bo-next-batch",
                "/api/skills/bo-next-batch/files", "/api/skills/bo-next-batch/files/SKILL.md"]:
        r = client.get(url, headers=A)
        assert r.status_code == 404, url
        assert r.json()["code"] == "not_found"
    assert client.post("/api/skills", headers=B).status_code == 404
    assert client.get("/api/skills").status_code == 404  # flag checked before auth


def test_default_registry_dependency_serves_builtins(monkeypatch, tmp_path):
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.setenv("KALOS_SKILLS", "1")
    monkeypatch.delenv("KALOS_SKILLS_DIR", raising=False)
    registry_mod.reset_cache()
    r = TestClient(app).get("/api/skills/run-sheet-checks", headers=A)
    assert r.status_code == 200 and r.json()["name"] == "Run-sheet checks"
    assert TestClient(app).get("/api/skills/run-sheet-validation", headers=A).status_code == 404
