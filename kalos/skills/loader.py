"""Parse `SKILL.md` folders, check their body sections, and hash them.

Identity: `content_hash` is sha256 over the sorted list of
(relative path, sha256 of the file bytes) for the whole skill folder
(`tree_hash`). The frontmatter never stores its own hash, so editing any byte
of any file in the folder gives a new identity. The `version` field is a human
label only.

Engine refs are checked statically (`check_engine_refs`): the file is parsed
with `ast` and never imported, so checking a torch-backed ref never loads torch.

Kalos-original code.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schema import ENTRYPOINT_RE, REF_RE, REQUIRED_SECTIONS, SkillMeta

FM_RE = re.compile(r"\A---\n(?P<fm>.*?)\n---\n(?P<body>.*)\Z", re.S)
HEADING_RE = re.compile(r"(?m)^## +(.+?)\s*$")
MAX_BODY_LINES = 300
MAX_FILE_BYTES = 256 * 1024
# A skill folder holds text only. No scripts: v1 has no free code execution, and a
# locked skill's computation is the engine function its tools are bound to.
TEXT_SUFFIXES = frozenset({".md", ".txt", ".json", ".yaml", ".yml", ".csv"})
SKILL_FILE = "SKILL.md"
DATA_SOURCES_FILE = "DATA_SOURCES.md"
# Top level of a skill folder: SKILL.md, an optional rendered DATA_SOURCES.md,
# and two optional folders. examples/ holds synthetic examples only.
ALLOWED_TOP = frozenset({SKILL_FILE, DATA_SOURCES_FILE, "references", "examples"})
# Evidence and machinery live outside the package: no tests, fixtures, expected
# answers, lockfiles, dependency pins or runner copies anywhere in the folder.
FORBIDDEN_NAME_RE = re.compile(
    r"(?i)(^|/)(tests?|fixtures?|expected[^/]*|golden[^/]*|answers?[^/]*|"
    r"[^/]*\.lock|[^/]*-lock\.json|requirements[^/]*\.txt|pipfile[^/]*|pyproject\.toml|"
    r"run_skill[^/]*|runner[^/]*|test_[^/]*|[^/]*_test\.[a-z]+)(/|$)")


class SkillFormatError(ValueError):
    """A skill folder that does not follow `kalos-skill/1`."""


@dataclass(frozen=True)
class Skill:
    meta: SkillMeta
    body: str
    path: Path
    content_hash: str  # "sha256:<hex>" over sorted (relpath, sha256(bytes)) of the folder
    files: tuple[str, ...]
    source: str = "kalos"  # "kalos" (curated, shipped) | "team" (tenant folder)

    @property
    def slug(self) -> str:
        return self.meta.slug


@dataclass
class LoadResult:
    skills: list[Skill] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (where, message)


def _skip(rel: str) -> bool:
    parts = rel.split("/")
    return any(p.startswith(".") or p == "__pycache__" for p in parts)


def tree_hash(folder: Path) -> tuple[str, tuple[str, ...]]:
    """Content hash of a skill folder and the relative paths it covers.

    Dotfiles and `__pycache__` are ignored. A symlink anywhere in the folder is an
    error: the bytes a link points at are not part of the package.
    """
    folder = Path(folder)
    h = hashlib.sha256()
    rels: list[str] = []
    for p in sorted(folder.rglob("*"), key=lambda x: x.relative_to(folder).as_posix()):
        rel = p.relative_to(folder).as_posix()
        if _skip(rel):
            continue
        if p.is_symlink():
            raise SkillFormatError(f"symlink not allowed in a skill folder: {rel}")
        if not p.is_file():
            continue
        rels.append(rel)
        h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return "sha256:" + h.hexdigest(), tuple(rels)


def split_frontmatter(text: str) -> tuple[dict, str]:
    text = text.removeprefix("\ufeff").replace("\r\n", "\n")
    m = FM_RE.match(text)
    if not m:
        raise SkillFormatError("SKILL.md must start with a '---' YAML frontmatter block")
    data = yaml.safe_load(m.group("fm"))
    if not isinstance(data, dict):
        raise SkillFormatError("frontmatter must be a YAML mapping")
    return data, m.group("body")


def check_body(body: str) -> None:
    heads = set(HEADING_RE.findall(body))
    missing = [s for s in REQUIRED_SECTIONS if s not in heads]
    if missing:
        raise SkillFormatError(f"missing body sections: {missing}")
    if body.count("\n") > MAX_BODY_LINES:
        raise SkillFormatError(f"body exceeds {MAX_BODY_LINES} lines; move detail to references/")


def check_files(folder: Path, files: tuple[str, ...]) -> None:
    for rel in files:
        top = rel.split("/", 1)[0]
        if top not in ALLOWED_TOP or (top in (SKILL_FILE, DATA_SOURCES_FILE) and "/" in rel):
            raise SkillFormatError(
                f"not allowed in a skill folder: {rel} (only SKILL.md, DATA_SOURCES.md, "
                "references/, examples/)")
        if FORBIDDEN_NAME_RE.search(rel):
            raise SkillFormatError(
                f"tests, fixtures, expected answers, lockfiles and runners live outside "
                f"the skill folder: {rel}")
        p = folder / rel
        if p.suffix.lower() not in TEXT_SUFFIXES:
            raise SkillFormatError(f"non-text file in skill folder: {rel}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise SkillFormatError(f"file over {MAX_FILE_BYTES} bytes: {rel}")


def parse_skill(folder: Path, *, source: str = "kalos") -> Skill:
    """Load one skill folder. Raises `SkillFormatError` or pydantic's
    `ValidationError` on any problem."""
    folder = Path(folder)
    if folder.is_symlink():
        raise SkillFormatError("a skill folder may not be a symlink")
    entry = folder / SKILL_FILE
    if not entry.is_file() or entry.is_symlink():
        raise SkillFormatError(f"{SKILL_FILE} missing")
    data, body = split_frontmatter(entry.read_text(encoding="utf-8"))
    meta = SkillMeta.model_validate(data)
    if meta.slug != folder.name:
        raise SkillFormatError(f"slug {meta.slug!r} != folder name {folder.name!r}")
    check_body(body)
    digest, files = tree_hash(folder)
    check_files(folder, files)
    ds = folder / DATA_SOURCES_FILE
    if ds.is_file() and ds.read_text(encoding="utf-8") != render_data_sources(meta):
        raise SkillFormatError(
            "DATA_SOURCES.md differs from the frontmatter rows (the frontmatter is "
            "authoritative; re-render it with `kalos skills data-sources`)")
    return Skill(meta=meta, body=body, path=folder, content_hash=digest, files=files, source=source)


def _first_line(exc: Exception) -> str:
    text = str(exc)
    if not text:
        return repr(exc)
    lines = text.splitlines()
    # pydantic: "1 validation error for SkillMeta\n<loc>\n  <msg> ..."; keep loc + msg
    if len(lines) >= 3 and "validation error" in lines[0]:
        msg = lines[2].strip().split(" [type=", 1)[0].removeprefix("Value error, ")
        return f"{lines[1].strip()}: {msg}"
    return lines[0]


def load_dir(root: Path, *, source: str = "kalos") -> LoadResult:
    """Load every `<root>/<slug>/SKILL.md`. Collects every problem instead of
    stopping at the first; duplicate slugs are an error."""
    res = LoadResult()
    root = Path(root)
    if not root.is_dir():
        res.errors.append((str(root), "not a directory"))
        return res
    real_root = root.resolve()
    for folder in sorted(p.parent for p in root.glob(f"*/{SKILL_FILE}")):
        # A symlinked folder, or one that resolves outside the root, could pull
        # arbitrary files into a tenant's or the curated skill set: skip it.
        if folder.is_symlink() or folder.resolve().parent != real_root:
            res.errors.append((str(folder), "skipped: symlinked or outside the skills root"))
            continue
        try:
            res.skills.append(parse_skill(folder, source=source))
        except Exception as exc:  # noqa: BLE001 - collect and report every problem
            res.errors.append((str(folder), _first_line(exc)))
    seen: dict[str, int] = {}
    for s in res.skills:
        seen[s.slug] = seen.get(s.slug, 0) + 1
    for dup in sorted(k for k, n in seen.items() if n > 1):
        res.errors.append((dup, "duplicate slug"))
    return res


def default_engine_root() -> Path:
    """The directory that holds the `kalos/` package (a checkout or site-packages)."""
    import kalos

    return Path(kalos.__file__).resolve().parent.parent


def render_data_sources(meta: SkillMeta) -> str:
    """DATA_SOURCES.md rendered from `licence.data_sources` (byte-stable)."""
    lines = [f"# Data sources: {meta.slug}", "",
             "Rendered from the SKILL.md frontmatter, which is authoritative. "
             "`unknown` never counts as allowed.", ""]
    rows = meta.licence.data_sources
    if not rows:
        lines.append("None.")
    else:
        lines += ["| Name | Kind | Version | Licence | Commercial status | Bundled | URI | "
                  "Checked by | Checked on |", "|---|---|---|---|---|---|---|---|---|"]
        for d in rows:
            cells = [d.name, d.kind, d.version, d.licence, d.commercial_status,
                     "yes" if d.bundled else "no", d.uri, d.verified_by, d.checked_on.isoformat()]
            lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def defined_extras(engine_root: Path | None = None) -> frozenset[str]:
    """Optional-dependency groups the engine defines: pyproject.toml in a
    checkout, else the installed distribution's Provides-Extra."""
    import tomllib

    root = Path(engine_root) if engine_root is not None else default_engine_root()
    pp = root / "pyproject.toml"
    if pp.is_file():
        data = tomllib.loads(pp.read_text(encoding="utf-8"))
        return frozenset(data.get("project", {}).get("optional-dependencies", {}))
    from importlib import metadata

    try:
        return frozenset(metadata.metadata("kalos").get_all("Provides-Extra") or [])
    except metadata.PackageNotFoundError:
        return frozenset()


def top_level_defs(path: Path) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        n.name: n.lineno
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def check_symbol(symbol: str, engine_root: Path, *, line: int | None = None,
                 ref_path: str | None = None) -> list[str]:
    """Statically check `pkg.mod:attr` (or `pkg.mod:Class.method`): the module
    file exists and defines `attr` at top level (at `line`, when given), and the
    class defines a public `method`. Never imports anything."""
    sm = ENTRYPOINT_RE.match(symbol)
    if not sm:
        return [f"{symbol}: not 'package.module:attr'"]
    mod_path = sm["module"].replace(".", "/") + ".py"
    problems: list[str] = []
    if ref_path is not None and mod_path != ref_path:
        problems.append(f"{symbol}: module path {mod_path} != ref file {ref_path}")
    path = Path(engine_root) / mod_path
    if not path.is_file():
        problems.append(f"{symbol}: file {mod_path} not found")
        return problems
    got = top_level_defs(path).get(sm["attr"])
    if got is None:
        problems.append(f"{symbol}: not defined at module top level")
    elif line is not None and got != line:
        problems.append(f"{symbol}: defined at line {got}, ref says {line}")
    if got is not None and sm["method"]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == sm["attr"]),
                   None)
        methods = {b.name for b in (cls.body if cls else [])
                   if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if cls is None:
            problems.append(f"{symbol}: {sm['attr']} is not a class")
        elif sm["method"] not in methods:
            problems.append(f"{symbol}: class has no method {sm['method']}")
        elif sm["method"].startswith("_"):
            problems.append(f"{symbol}: method is private")
    return problems


def check_engine_refs(skill: Skill, engine_root: Path | None = None) -> list[str]:
    """Each ref's file exists, the symbol's module path equals the file, and the
    attribute is a top-level def/class at exactly the cited line."""
    root = Path(engine_root) if engine_root is not None else default_engine_root()
    problems: list[str] = []
    for er in skill.meta.implementation.engine:
        rm = REF_RE.match(er.ref)
        assert rm is not None  # EngineRef already enforced the format
        problems += check_symbol(er.symbol, root, line=int(rm["line"]), ref_path=rm["path"])
    return problems


def ngram_overlap(text: str, corpus: set[str], n: int = 10) -> list[str]:
    """Word n-grams of `text` that also occur in `corpus` (used by `check
    --foreign-corpus` to show skill texts are our own words)."""
    w = re.findall(r"[a-z0-9']+", text.lower())
    grams = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return sorted(grams & corpus)


def corpus_ngrams(dirs: list[Path], n: int = 10) -> set[str]:
    grams: set[str] = set()
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.is_file() and p.suffix in {".md", ".txt", ".json"}:
                w = re.findall(r"[a-z0-9']+", p.read_text(errors="ignore").lower())
                grams |= {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return grams


__all__ = [
    "MAX_BODY_LINES",
    "LoadResult",
    "Skill",
    "SkillFormatError",
    "check_engine_refs",
    "check_symbol",
    "corpus_ngrams",
    "default_engine_root",
    "defined_extras",
    "load_dir",
    "ngram_overlap",
    "parse_skill",
    "render_data_sources",
    "split_frontmatter",
    "tree_hash",
]
