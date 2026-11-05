"""Command line for skills.

    kalos skills validate [DIR] [--engine-root P] [--foreign-corpus DIR ...] [--json]
    kalos skills list     [DIR] [--category C] [--json]
    kalos skills search   "query" [DIR] [--tenant T] [--all] [--limit N]
    kalos skills hash     DIR/slug
    kalos skills tools    [--json]

`python -m kalos.skills <cmd> ...` is the same. DIR defaults to the built-in set.

`validate` checks the schema, slug = folder, required sections, the line cap,
duplicate slugs, static engine-ref resolution at the cited line, that every
`tools[]` name is registered with a label the skill declares, and (with
`--foreign-corpus`) that no 10-word run is shared with the given texts.
Exit codes: 0 clean, 1 problems, 2 usage error. Kalos-original code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .loader import check_engine_refs, corpus_ngrams, load_dir, ngram_overlap, tree_hash
from .registry import BUILTIN_DIR, SkillRegistry


def _validate(a: argparse.Namespace) -> int:
    from .tools import check_tools

    root = Path(a.dir) if a.dir else BUILTIN_DIR
    res = load_dir(root)
    problems: list[tuple[str, str]] = list(res.errors)
    corpus = corpus_ngrams([Path(d) for d in a.foreign_corpus]) if a.foreign_corpus else set()
    engine_root = Path(a.engine_root) if a.engine_root else None
    rows: list[dict[str, Any]] = []
    for sk in res.skills:
        p = check_engine_refs(sk, engine_root)
        p += [x.split(": ", 1)[1] for x in check_tools([sk]) if x.startswith(f"{sk.slug}: ")]
        if corpus:
            text = "\n".join((sk.path / f).read_text(encoding="utf-8") for f in sk.files)
            overlap = ngram_overlap(text, corpus)
            if overlap:
                p.append(f"{len(overlap)} 10-word run(s) shared with the foreign corpus")
        problems += [(sk.slug, x) for x in p]
        rows.append({"slug": sk.slug, "category": sk.meta.category, "kind": sk.meta.kind,
                     "status": sk.meta.status, "implementation": sk.meta.implementation.status,
                     "refs": len(sk.meta.implementation.engine), "tools": len(sk.meta.tools),
                     "content_hash": sk.content_hash, "ok": not p})
    # Problems with the tool table itself (unresolvable symbol, missing adapter).
    problems += [("tools", x) for x in check_tools([]) if x]
    if a.json:
        print(json.dumps({"ok": not problems, "skills": rows,
                          "problems": [{"where": w, "message": m} for w, m in problems]},
                         indent=1))
    else:
        for r in rows:
            print(f"{'OK ' if r['ok'] else 'ERR'} {r['slug']:34} {r['category']:13} "
                  f"{r['status']:8} {r['implementation']:15} refs={r['refs']} tools={r['tools']} "
                  f"{r['content_hash'][:19]}")
        for where, msg in problems:
            print(f"  ! {where}: {msg}")
        print(f"{len(res.skills)} skills loaded, {len(problems)} problem(s)")
    return 1 if problems else 0


def _list(a: argparse.Namespace) -> int:
    res = load_dir(Path(a.dir) if a.dir else BUILTIN_DIR)
    skills = [s for s in res.skills if not a.category or s.meta.category == a.category]
    if a.json:
        print(json.dumps([{"slug": s.slug, "name": s.meta.name, "category": s.meta.category,
                           "status": s.meta.status,
                           "implementation": s.meta.implementation.status,
                           "content_hash": s.content_hash} for s in skills], indent=1))
    else:
        for s in skills:
            print(f"{s.slug:34} {s.meta.category:13} {s.meta.status:8} "
                  f"{s.meta.implementation.status:15} {s.content_hash[:19]}")
    for where, msg in res.errors:
        print(f"  ! {where}: {msg}", file=sys.stderr)
    return 0


def _search(a: argparse.Namespace) -> int:
    root = Path(a.dir) if a.dir else BUILTIN_DIR
    res = load_dir(root)
    reg = SkillRegistry(res.skills)
    hits = reg.search(a.query, a.tenant, include_beta=a.all, include_planned=a.all,
                      limit=a.limit)
    for score, sk in hits:
        print(f"{score:6.2f}  {sk.slug:34} [{sk.meta.category}, {sk.meta.status}]")
    return 0


def _hash(a: argparse.Namespace) -> int:
    folder = Path(a.folder)
    if not (folder / "SKILL.md").is_file():
        print(f"no SKILL.md in {folder}", file=sys.stderr)
        return 2
    print(tree_hash(folder)[0])
    return 0


def _tools(a: argparse.Namespace) -> int:
    from .tools import TOOLS

    rows = [{"name": t.name, "symbol": t.symbol, "label": t.label, "scope": t.scope,
             "side_effects": t.side_effects, "needs_torch": t.needs_torch,
             "timeout_s": t.timeout_s} for t in TOOLS.values()]
    if a.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{r['name']:34} {r['label']:10} {'torch' if r['needs_torch'] else '     '} "
                  f"{r['symbol']}")
    return 0


def build_parser(prog: str = "kalos skills") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Kalos skills tooling")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", aliases=["check"], help="check skill folders")
    v.add_argument("dir", nargs="?")
    v.add_argument("--engine-root")
    v.add_argument("--foreign-corpus", action="append", default=[])
    v.add_argument("--json", action="store_true")
    v.set_defaults(fn=_validate)
    li = sub.add_parser("list", help="list skills")
    li.add_argument("dir", nargs="?")
    li.add_argument("--category")
    li.add_argument("--json", action="store_true")
    li.set_defaults(fn=_list)
    s = sub.add_parser("search", help="deterministic search")
    s.add_argument("query")
    s.add_argument("dir", nargs="?")
    s.add_argument("--tenant", default="cli")
    s.add_argument("--all", action="store_true", help="include beta and planned skills")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(fn=_search)
    h = sub.add_parser("hash", help="content hash of one skill folder")
    h.add_argument("folder")
    h.set_defaults(fn=_hash)
    t = sub.add_parser("tools", help="the tool table")
    t.add_argument("--json", action="store_true")
    t.set_defaults(fn=_tools)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


def kalos_main(argv: list[str] | None = None) -> int:
    """`kalos` console script. Only the `skills` command group exists today."""
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] != "skills":
        print("usage: kalos skills {validate,list,search,hash,tools} ...", file=sys.stderr)
        return 2
    return main(args[1:])


__all__ = ["build_parser", "kalos_main", "main"]
