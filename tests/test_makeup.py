"""Tests for kalos.analysis.makeup: make-up arithmetic goldens, refusals, and purity.

Synthetic numbers only. The goldens in tests/fixtures/makeup_goldens.json were
derived by hand (each case carries its derivation), so they check the arithmetic
rather than record whatever the code returns.
"""
from __future__ import annotations

import ast
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from kalos.analysis import makeup
from kalos.analysis.makeup import LABEL, Stock, Target, make_up

GOLDENS = json.loads((Path(__file__).parent / "fixtures" / "makeup_goldens.json").read_text())["cases"]


def _run(case: dict) -> makeup.MakeUp:
    targets = [Target(n, c, u) for n, c, u in case["targets"]]
    stocks = [Stock(n, c, u, mw) for n, c, u, mw in case["stocks"]]
    return make_up(targets, stocks, case["final_volume_ml"])


@pytest.mark.parametrize("case", GOLDENS, ids=[c["id"] for c in GOLDENS])
def test_arithmetic_goldens(case: dict) -> None:
    result = _run(case)
    expect = case["expect"]
    assert result.status == expect["status"]
    assert result.label == LABEL
    got = {c.component: c.volume_ml for c in result.components}
    for name, vol in expect.get("volumes_ml", {}).items():
        # Exact: the module computes in fractions and rounds once to the nearest float.
        assert got[name] == float(Fraction(vol)), (name, got[name], vol)
    if expect["status"] == "ok":
        assert result.refusals == ()
        assert result.water_ml == float(Fraction(expect["water_ml"]))
    else:
        assert result.water_ml is None
        assert [r.code for r in result.refusals] == expect["refusal_codes"]
        for r in result.refusals:
            assert r.reason and r.what_would_fix_it


def test_goldens_cover_the_acceptance_units() -> None:
    units = {u for c in GOLDENS for _, _, u in c["targets"] + [s[:3] for s in c["stocks"]]}
    assert {"g/L", "mM", "% w/v"} <= units
    codes = {code for c in GOLDENS for code in c["expect"].get("refusal_codes", [])}
    assert {"negative_water", "stock_too_dilute", "unit_mismatch_no_mw"} <= codes


def test_c1v1_equals_c2v2_and_volumes_sum_to_final() -> None:
    rng = random.Random(20261120)
    for _ in range(200):
        v_final = round(rng.uniform(0.2, 2000.0), 3)
        n = rng.randint(1, 6)
        targets, stocks = [], []
        for i in range(n):
            stock = round(rng.uniform(50.0, 1000.0), 2)
            target = round(rng.uniform(0.0, stock / (2 * n)), 4)
            targets.append(Target(f"c{i}", target, "g/L"))
            stocks.append(Stock(f"c{i}", stock, "g/L"))
        r = make_up(targets, stocks, v_final)
        assert r.status == "ok", r.refusals
        for line, tg, st in zip(r.components, targets, stocks):
            assert line.volume_ml is not None
            assert math.isclose(st.concentration * line.volume_ml, tg.concentration * v_final, rel_tol=1e-12)
        assert r.water_ml is not None and r.water_ml >= 0
        assert math.isclose(sum(c.volume_ml or 0.0 for c in r.components) + r.water_ml, v_final, rel_tol=1e-12)


def test_deterministic_and_serialisable() -> None:
    a, b = _run(GOLDENS[2]).as_dict(), _run(GOLDENS[2]).as_dict()
    assert a == b
    assert json.loads(json.dumps(a)) == a
    assert a["label"] == "arithmetic from your stocks; not a model output"


def _codes(targets: list[Target], stocks: list[Stock], v: float = 100.0) -> list[str]:
    return [r.code for r in make_up(targets, stocks, v).refusals]


def test_negative_water_names_the_largest_additions() -> None:
    r = make_up([Target("a", 60, "g/L"), Target("b", 3, "g/L")], [Stock("a", 50, "% w/v"), Stock("b", 30, "g/L")], 10)
    assert r.status == "ok"  # 60 g/L from 500 g/L = 1.2 mL; 3 from 30 = 1 mL
    r = make_up([Target("a", 60, "g/L"), Target("b", 25, "g/L")], [Stock("a", 100, "g/L"), Stock("b", 50, "g/L")], 100)
    (ref,) = r.refusals
    assert ref.code == "negative_water"
    assert "10 mL more than the 100 mL" in ref.reason
    assert "a: 60 mL" in ref.what_would_fix_it


def test_unit_mismatch_needs_mw_on_the_stock() -> None:
    (ref,) = make_up([Target("glucose", 10, "mM")], [Stock("glucose", 200, "g/L")], 100).refusals
    assert ref.code == "unit_mismatch_no_mw"
    assert ref.component == "glucose"
    assert "molecular weight" in ref.what_would_fix_it


@pytest.mark.parametrize(
    ("t_unit", "s_unit", "code"),
    [
        ("%", "g/L", "ambiguous_percent"),
        ("g/L", "pct", "ambiguous_percent"),
        ("% v/v", "g/L", "incompatible_units"),
        ("mg/L", "g/L", "unknown_unit"),  # not in the engine's unit registry yet
        ("furlongs", "g/L", "unknown_unit"),
        ("MM", "mM", "unknown_unit"),  # molar prefixes are case-sensitive
        ("", "g/L", "unknown_unit"),
    ],
)
def test_unconvertible_units_refuse(t_unit: str, s_unit: str, code: str) -> None:
    assert _codes([Target("x", 1, t_unit)], [Stock("x", 100, s_unit, 100.0)]) == [code]


def test_same_unit_needs_no_conversion_even_if_unknown() -> None:
    r = make_up([Target("x", 2, "%")], [Stock("x", 50, " % ")], 100)
    assert r.status == "ok" and r.components[0].volume_ml == 4.0
    assert r.components[0].basis.startswith("same unit")


def test_micro_sign_variants_are_one_unit() -> None:
    for unit in ("µM", "μM", "uM"):
        r = make_up([Target("fe", 50, unit)], [Stock("fe", 10, "mM")], 1)
        assert r.components[0].volume_ml == 0.005


@pytest.mark.parametrize(
    ("targets", "stocks", "v", "codes"),
    [
        ([Target("a", 1, "g/L")], [Stock("a", 10, "g/L")], 0.0, ["invalid_final_volume"]),
        ([Target("a", 1, "g/L")], [Stock("a", 10, "g/L")], float("nan"), ["invalid_final_volume"]),
        ([Target("a", -1, "g/L")], [Stock("a", 10, "g/L")], 100.0, ["invalid_target"]),
        ([Target("a", float("inf"), "g/L")], [Stock("a", 10, "g/L")], 100.0, ["invalid_target"]),
        ([Target("a", 1, "g/L")], [Stock("a", 0, "g/L")], 100.0, ["invalid_stock"]),
        ([Target("a", 1, "mM")], [Stock("a", 10, "g/L", -5.0)], 100.0, ["invalid_mw"]),
        ([Target("a", 1, "g/L")], [Stock("b", 10, "g/L")], 100.0, ["no_stock"]),
        ([Target("a", 1, "g/L")], [Stock("a", 10, "g/L"), Stock("a", 20, "g/L")], 100.0, ["duplicate_stock"]),
        ([Target("a", 1, "g/L"), Target("a ", 1, "g/L")], [Stock("a", 10, "g/L")], 100.0, ["duplicate_component"]),
    ],
)
def test_invalid_inputs_refuse_with_a_fix(targets: list[Target], stocks: list[Stock], v: float, codes: list[str]) -> None:
    r = make_up(targets, stocks, v)
    assert r.status == "refused" and r.water_ml is None
    assert [x.code for x in r.refusals] == codes
    assert all(x.reason and x.what_would_fix_it for x in r.refusals)


def test_bool_is_not_a_number() -> None:
    assert _codes([Target("a", True, "g/L")], [Stock("a", 10, "g/L")]) == ["invalid_target"]  # type: ignore[arg-type]


def test_zero_target_needs_no_stock() -> None:
    r = make_up([Target("a", 0, "g/L"), Target("b", 5, "g/L")], [Stock("b", 50, "g/L")], 10)
    assert r.status == "ok"
    assert [c.volume_ml for c in r.components] == [0.0, 1.0]
    assert r.water_ml == 9.0


def test_all_refusals_reported_not_just_the_first() -> None:
    codes = _codes(
        [Target("a", 20, "g/L"), Target("b", 1, "mM"), Target("c", 1, "g/L")],
        [Stock("a", 10, "g/L"), Stock("b", 1, "g/L")],
    )
    assert codes == ["stock_too_dilute", "unit_mismatch_no_mw", "no_stock"]


def test_exact_decimal_arithmetic() -> None:
    # 0.1 + 0.2 in binary floats is 0.30000000000000004; read as decimals it is 0.3.
    r = make_up([Target("a", 0.1, "g/L"), Target("b", 0.2, "g/L")], [Stock("a", 1, "g/L"), Stock("b", 1, "g/L")], 1)
    assert r.water_ml == 0.7


def test_module_is_pure_and_respects_the_layer_rule() -> None:
    """analysis/makeup.py may import ingest, libs and core only (no I/O, portal, storage or api)."""
    tree = ast.parse(Path(makeup.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    kalos_mods = {m for m in imported if m.split(".")[0] == "kalos"}
    allowed = ("kalos.core.", "kalos.ingest.", "kalos.normalize.", "kalos.domains.", "kalos.validation.")
    assert all(m.startswith(allowed) for m in kalos_mods), kalos_mods
    stdlib = {m.split(".")[0] for m in imported - kalos_mods}
    assert stdlib <= set(sys.stdlib_module_names) | {"__future__"}, stdlib
    assert not stdlib & {"os", "io", "pathlib", "sqlite3", "socket", "subprocess", "shutil", "tempfile"}
