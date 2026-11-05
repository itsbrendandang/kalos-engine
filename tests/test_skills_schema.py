"""`kalos-skill/1` frontmatter rules (kalos/skills/schema.py): every cross-field
rule has a passing and a failing case, and unknown keys are rejected."""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("yaml")
from pydantic import ValidationError

from kalos.skills.schema import EngineRef, Parameter, SkillMeta


def base(**over) -> dict:
    d = {
        "schema": "kalos-skill/1",
        "slug": "example-skill",
        "name": "Example skill",
        "version": "0.1.0",
        "kind": "locked",
        "category": "analytics",
        "summary": "Use to compute an example summary of a run sheet for tests. Not for anything real.",
        "keywords": ["example"],
        "starting_prompt": "Summarise this run sheet please.",
        "inputs": [{"name": "rows", "type": "table", "description": "Rows of the run sheet."}],
        "parameters": [],
        "outputs": [{"name": "summary", "type": "json", "label": "derived",
                     "description": "A derived summary of the rows."}],
        "tools": ["engine.noise_report"],
        "implementation": {"status": "implemented",
                           "engine": [{"symbol": "kalos.core.replicates:noise_report",
                                       "ref": "kalos/core/replicates.py:354"}],
                           "gaps": []},
        "scope": "read",
        "side_effects": "none",
        "refuses_when": ["The table is empty."],
        "verification": {"deterministic": ["same input gives same output"], "receipt": True,
                         "review": "none"},
        "author": "Kalos",
        "owner": "system",
        "visibility": "system",
        "licence": {"spdx": "Apache-2.0"},
        "source": "kalos-engine",
        "status": "active",
        "created": "2026-11-05",
        "updated": "2026-11-05",
    }
    d = copy.deepcopy(d)
    d.update(over)
    return d


def ok(d: dict) -> SkillMeta:
    return SkillMeta.model_validate(d)


def bad(d: dict) -> None:
    with pytest.raises(ValidationError):
        SkillMeta.model_validate(d)


def test_base_is_valid_and_frozen():
    m = ok(base())
    assert m.schema_id == "kalos-skill/1"
    assert m.output_labels == frozenset({"derived"})
    with pytest.raises(ValidationError):
        m.slug = "other"  # type: ignore[misc]


def test_unknown_field_rejected():
    bad(base(prompt_injection="x"))
    d = base()
    d["outputs"][0]["colour"] = "red"
    bad(d)


def test_wrong_schema_id_rejected():
    bad(base(schema="kalos-skill/2"))


@pytest.mark.parametrize("slug,valid", [("a-b-c", True), ("A-b", False), ("a--b", False),
                                        ("a" * 65, False)])
def test_slug_format(slug, valid):
    (ok if valid else bad)(base(slug=slug))


def test_enum_parameter_needs_allowed_and_default_inside():
    Parameter(name="mode", type="enum", allowed=["a", "b"], default="a", effect="Picks a mode.")
    with pytest.raises(ValidationError):
        Parameter(name="mode", type="enum", default="a", effect="Picks a mode.")
    with pytest.raises(ValidationError):
        Parameter(name="mode", type="enum", allowed=["a", "b"], default="c", effect="Picks a mode.")


@pytest.mark.parametrize("ptype,default,valid", [
    ("int", 4, True), ("int", "4", False), ("int", True, False), ("float", 1, True),
    ("float", 0.5, True), ("bool", True, True), ("bool", 1, False), ("str", "x", True),
    ("str", 3, False),
])
def test_parameter_default_matches_type(ptype, default, valid):
    def make():
        return Parameter(name="p", type=ptype, default=default, effect="Changes something.")
    if valid:
        make()
    else:
        with pytest.raises(ValidationError):
            make()


def test_implemented_needs_engine_ref():
    bad(base(implementation={"status": "implemented", "engine": [], "gaps": []}))


def test_partial_needs_gaps():
    impl = base()["implementation"]
    ok(base(implementation={**impl, "status": "partial", "gaps": ["the export needs code"]}))
    bad(base(implementation={**impl, "status": "partial", "gaps": []}))


def test_planned_iff_needs_new_code_and_no_tools():
    new = {"status": "needs-new-code", "engine": [], "gaps": ["all of it"]}
    ok(base(status="planned", implementation=new, tools=[]))
    bad(base(status="planned", implementation=new, tools=["engine.noise_report"]))
    bad(base(status="active", implementation=new, tools=[]))
    bad(base(status="planned", tools=[]))  # implemented but planned


def test_summary_needs_negative_boundary():
    bad(base(summary="Use to compute an example summary of a run sheet for tests and anything else."))
    # 'note' must not count as 'not'
    bad(base(summary="Use to compute an example summary of a run sheet; note it is for the tests here."))


def test_locked_needs_refuses_when_guided_does_not():
    bad(base(refuses_when=[]))
    ok(base(kind="guided", refuses_when=[]))


def test_side_effects_need_write_scope():
    bad(base(side_effects="writes-campaign", scope="read"))
    ok(base(side_effects="writes-campaign", scope="write"))


def test_every_output_needs_a_label():
    d = base()
    del d["outputs"][0]["label"]
    bad(d)
    d["outputs"][0]["label"] = "guessed"
    bad(d)


def test_simulation_may_only_emit_simulated_or_synthetic():
    bad(base(category="simulation"))  # output is 'derived'
    d = base(category="simulation")
    d["outputs"][0]["label"] = "simulated"
    ok(d)
    d["outputs"][0]["label"] = "measured"
    bad(d)


def test_owner_and_visibility_agree():
    ok(base(owner="tenant:acme", visibility="tenant"))
    bad(base(owner="tenant:acme", visibility="system"))
    bad(base(owner="system", visibility="tenant"))
    bad(base(owner="root", visibility="system"))
    ok(base(owner="user:bob@acme", visibility="private"))
    assert ok(base(owner="tenant:acme", visibility="tenant")).tenant == "acme"


def test_updated_not_before_created():
    bad(base(updated="2026-11-04"))
    ok(base(updated="2026-11-06"))


def test_tool_name_shape_and_uniqueness():
    bad(base(tools=["propose_batch"]))
    bad(base(tools=["engine.noise_report", "engine.noise_report"]))


@pytest.mark.parametrize("symbol,ref,valid", [
    ("kalos.core.optimize:propose_batch", "kalos/core/optimize.py:408", True),
    ("kalos.core.optimize.propose_batch", "kalos/core/optimize.py:408", False),
    ("kalos.core.optimize:propose_batch", "kalos/core/optimize.py", False),
])
def test_engine_ref_format(symbol, ref, valid):
    if valid:
        EngineRef(symbol=symbol, ref=ref)
    else:
        with pytest.raises(ValidationError):
            EngineRef(symbol=symbol, ref=ref)
