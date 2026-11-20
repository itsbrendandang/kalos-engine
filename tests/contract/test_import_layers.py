"""Reorg guard R03: the import-layer contract (scripts/import_layers.py).

REPORT-ONLY for now (REORG_PLAN section 3): the violations of the layer rules
are printed and compared with the checked-in baseline
(tests/contract/import_layers_baseline.json), and NEW or FIXED edges are
reported as warnings, not failures. What fails today is only a broken map: a
module placed in no layer, rules that are not layered, or a baseline that is
not well formed. `ENFORCE` flips to True after the API move (R14): from then
on a new violation fails and the baseline may only shrink; it must be empty
after the assist decoupling (R19).
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENFORCE = False


def _layers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("import_layers", ROOT / "scripts" / "import_layers.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def layers() -> ModuleType:
    return _layers()


@pytest.fixture(scope="module")
def result(layers: ModuleType) -> dict:
    return dict(layers.scan())


def test_every_module_is_placed_in_a_layer(layers: ModuleType, result: dict) -> None:
    unplaced = sorted(m for m, layer in result["modules"].items() if layer is None)
    assert unplaced == [], f"add these modules to PLACEMENT in scripts/import_layers.py: {unplaced}"
    assert set(layers.PLACEMENT.values()) <= set(layers.LAYER_ORDER)


def test_rules_are_layered(layers: ModuleType) -> None:
    """Each layer may import only itself and layers listed before it, except the
    two top-level consumers (`compat` and `api`), so the rules cannot hide a cycle."""
    order = list(layers.LAYER_ORDER)
    assert set(layers.ALLOWED) == set(order)
    for layer, allowed in layers.ALLOWED.items():
        assert layer in allowed, layer
        if layer == "compat":
            continue
        above = {a for a in allowed if order.index(a) > order.index(layer)}
        assert above == set(), f"{layer} may import higher layers {sorted(above)}"
    for layer, allowed in layers.ALLOWED.items():
        if layer not in ("api", "compat"):
            assert "api" not in allowed, f"{layer} may import api"
        if layer not in ("bench", "compat"):
            assert "bench" not in allowed, f"{layer} may import bench"
    assert layers.ALLOWED["core"] <= {"util", "data", "config", "core"}


@pytest.mark.parametrize("module,layer", [
    ("kalos.core.surrogate", "core"),
    ("kalos.portal.analysis", "analysis"),
    ("kalos.portal.analysis_parts.columns", "analysis"),
    ("kalos.analysis.stages.trust", "analysis"),
    ("kalos.portal.campaign", "campaigns"),
    ("kalos.campaigns.service", "campaigns"),
    ("kalos.portal.migrate", "storage"),
    ("kalos.storage.migrations.v8", "storage"),
    ("kalos.portal.jobs", "jobs"),
    ("kalos.portal.app", "api"),
    ("kalos.api.routers.health", "api"),
    ("kalos.assist.routes", "api"),
    ("kalos.assist.index", "assist"),
    ("kalos.portal.uploads", "ingest"),
    ("kalos.portal.busy", "runtime"),
    ("kalos.bench.faults.runner", "bench"),
    ("kalos", "compat"),
])
def test_today_and_target_paths_share_a_layer(layers: ModuleType, module: str, layer: str) -> None:
    assert layers.layer_of(module) == layer


def test_baseline_is_well_formed(layers: ModuleType) -> None:
    data = json.loads(layers.BASELINE.read_text(encoding="utf-8"))
    keys = data["violations"]
    assert keys == sorted(set(keys)), "the baseline is sorted and unique"
    for k in keys:
        src, dst = k.split(" -> ")
        assert src.startswith("kalos.") and dst.startswith("kalos"), k


def test_import_layer_report(layers: ModuleType, result: dict) -> None:
    """Report-only: prints the layer report; warns on new or fixed edges."""
    baseline = layers.load_baseline()
    text = layers.report(result, baseline)
    print(text)
    keys = layers.violation_keys(result["violations"])
    new = sorted(set(keys) - set(baseline))
    fixed = sorted(set(baseline) - set(keys))
    if ENFORCE:
        assert new == [], f"new import-layer violations:\n{text}"
        assert fixed == [], "fixed violations: shrink the baseline (scripts/import_layers.py --write-baseline)"
    elif new or fixed:
        warnings.warn(f"import layers (report-only): {len(new)} new, {len(fixed)} fixed vs baseline; "
                      "run scripts/import_layers.py for the report", stacklevel=1)
