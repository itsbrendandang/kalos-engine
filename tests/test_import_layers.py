"""Import-layer rules (report-only; plan docs ARCHITECTURE §2.1).

Rules:
  core     numerics (kalos.core, validation, normalize, domains, scale, data,
           features, kit) import nothing from portal, runtime, services, the
           plug-ins or bench.
  storage  the storage layer (kalos.storage and today's storage modules in
           kalos/portal) never imports a route module.
  plug-ins kalos.integrations / assist / skills / twin import only themselves,
           core (numerics + envconfig/flags/versions), storage and the service
           facade (kalos.services).

"Report-only": these are not import-linter contracts yet (pyproject.toml must
wait for the live branches). Today's violations are listed in
`ALLOWED_VIOLATIONS`, which may only SHRINK: a new violation fails, and a
listed one that is gone fails until it is removed from the list. Every import
counts, including ones inside functions and under TYPE_CHECKING.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KALOS = ROOT / "kalos"

NUMERICS = ("kalos.core", "kalos.validation", "kalos.normalize", "kalos.domains", "kalos.scale",
            "kalos.data", "kalos.features", "kalos.kit")
CORE_EXTRAS = ("kalos.envconfig", "kalos.flags", "kalos.versions")
ABOVE_CORE = ("kalos.portal", "kalos.runtime", "kalos.services", "kalos.integrations", "kalos.assist",
              "kalos.skills", "kalos.twin", "kalos.bench")
STORAGE = ("kalos.storage", "kalos.portal.campaign", "kalos.portal.migrate", "kalos.portal.migrate_pg",
           "kalos.portal.serialization", "kalos.portal.storage")
ROUTES = ("kalos.portal.app", "kalos.portal.campaign_routes", "kalos.portal.campaigns_api",
          "kalos.portal.jobs_api", "kalos.portal.scale_routes", "kalos.portal.virtual_runs_api",
          "kalos.portal.routes")
ROUTE_SUFFIXES = (".routes", ".sop_routes")
PLUGINS = ("kalos.integrations", "kalos.assist", "kalos.skills", "kalos.twin")
SERVICE_FACADE = ("kalos.services",)

# (rule, importer, imported) found when this test was written. May only shrink.
# Empty on desktop-engine-2: no plug-in package is merged yet. A merging branch
# (assistant, skills, twin, benchling) adds its own current violations here in
# its merge commit, once; after that the list only shrinks.
ALLOWED_VIOLATIONS: set[tuple[str, str, str]] = set()


def _under(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _is_route(module: str) -> bool:
    return _under(module, ROUTES) or (module.startswith("kalos.") and module.endswith(ROUTE_SUFFIXES))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path, module: str) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""
            found.add(target)
            # `from kalos.portal import app` imports the submodule kalos.portal.app
            found.update(f"{target}.{alias.name}" for alias in node.names if alias.name != "*")
    return {m for m in found if m == "kalos" or m.startswith("kalos.")}


def _plugin_root(module: str) -> str | None:
    for prefix in PLUGINS:
        if _under(module, (prefix,)):
            rest = module[len(prefix) + 1:].split(".")
            # integrations: each connector is its own plug-in
            return f"{prefix}.{rest[0]}" if prefix == "kalos.integrations" and rest[0] else prefix
    return None


def _real(module: str) -> bool:
    """Whether `module` is a module/package in kalos/ (drops `from x import name` attributes)."""
    rel = Path(*module.split("."))
    return (ROOT / rel).with_suffix(".py").exists() or (ROOT / rel / "__init__.py").exists()


def violations() -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for path in sorted(KALOS.rglob("*.py")):
        importer = _module_name(path)
        for imported in (m for m in _imports(path, importer) if _real(m)):
            if _under(importer, NUMERICS) and _under(imported, ABOVE_CORE):
                found.add(("core", importer, imported))
            if _under(importer, STORAGE) and _is_route(imported):
                found.add(("storage", importer, imported))
            root = _plugin_root(importer)
            if root is not None and not (
                imported == "kalos"
                or _under(imported, (root,))
                or _under(imported, NUMERICS + CORE_EXTRAS + STORAGE + SERVICE_FACADE)
                or (root.startswith("kalos.integrations.") and imported == "kalos.integrations")
            ):
                found.add(("plugin", importer, imported))
    return found


def test_no_new_layer_violations():
    new = sorted(violations() - ALLOWED_VIOLATIONS)
    assert not new, (
        "new import-layer violations (rule, importer, imported); route the call through "
        f"the storage layer or a service instead: {new}"
    )


def test_allowlist_only_shrinks():
    gone = sorted(ALLOWED_VIOLATIONS - violations())
    assert not gone, f"fixed: remove from ALLOWED_VIOLATIONS in tests/test_import_layers.py: {gone}"


def test_rules_see_a_planted_violation(tmp_path, monkeypatch):
    """The scanner is not vacuous: a numerics module importing portal is caught."""
    fake = tmp_path / "kalos" / "core"
    fake.mkdir(parents=True)
    (fake / "bad.py").write_text("def f():\n    from kalos.portal import app\n", encoding="utf-8")
    (tmp_path / "kalos" / "portal").mkdir()
    (tmp_path / "kalos" / "portal" / "app.py").write_text("", encoding="utf-8")
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "KALOS", tmp_path / "kalos")
    assert ("core", "kalos.core.bad", "kalos.portal.app") in violations()
