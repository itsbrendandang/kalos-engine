"""Domain rules of the records (kalos/records), the request models
(kalos/api/schemas/records.py), the units registry and request ids. Torch-free,
no database.

Ports Voyager's `tests/test_validators.py` cases where Kalos keeps the
behaviour (trimming, required fields, maximum lengths, bounds order) and records
where Kalos is deliberately stricter (no string-to-number coercion, no equal
bounds, unknown keys refused).
"""
from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from kalos.api import request_context as rc
from kalos.api.schemas.records import (
    FactorPut,
    LinkCreate,
    LinkRetract,
    ProjectCreate,
    ProjectUpdate,
)
from kalos.normalize import unit_registry as ur
from kalos.records import factors, lineage, outcomes, projects
from kalos.records.projects import RecordsValidationError

CID_A, CID_B, CID_C = ("cmp_" + c * 32 for c in "abc")

# --- request models (Voyager validators/schemas.py) ----------------------------------- #


def test_project_model_trims_every_string():  # Voyager test_project_trimming
    m = ProjectCreate.model_validate({"name": "  P  ", "objective": "\tgoal\n", "sponsor": " S "})
    assert (m.name, m.objective, m.sponsor, m.status) == ("P", "goal", "S", "drafting")


@pytest.mark.parametrize("body", [
    {"objective": "o"}, {"name": "n"},                            # missing required fields
    {"name": "x" * 81, "objective": "o"},                         # name too long
    {"name": "n", "objective": "x" * 501},                        # objective too long
    {"name": "   ", "objective": "o"},                            # empty after trim
    {"name": "n", "objective": "o", "customer_id": "1"},          # no CRM field, no coercion
    {"name": 5, "objective": "o"},                                # strict: no number for text
])
def test_project_model_refusals(body):
    with pytest.raises(ValidationError):
        ProjectCreate.model_validate(body)


def test_project_update_is_partial():
    assert ProjectUpdate.model_validate({"status": "completed"}).model_dump(exclude_unset=True) == {"status": "completed"}


def test_factor_model_bounds_and_types():
    m = FactorPut.model_validate({"name": " glucose ", "lower_bound": 1, "upper_bound": 2.5, "unit": " g/L "})
    assert (m.name, m.lower_bound, m.upper_bound, m.unit, m.role) == ("glucose", 1.0, 2.5, "g/L", "setting")
    for bad in ({"lower_bound": "1"}, {"upper_bound": float("inf")}, {"role": "target"}, {"description": "x" * 301}):
        with pytest.raises(ValidationError):
            FactorPut.model_validate({"name": "g", "lower_bound": 0, "upper_bound": 1} | bad)


@pytest.mark.parametrize("body", [
    {}, {"parent_campaign_id": CID_A}, {"child_campaign_id": CID_B},          # Voyager missing fields
    {"parent_campaign_id": CID_A, "child_campaign_id": CID_B, "relation": "sibling"},
    {"parent_campaign_id": 1, "child_campaign_id": 2, "relation": "scale_up"},  # Voyager took ints
])
def test_link_model_refusals(body):
    with pytest.raises(ValidationError):
        LinkCreate.model_validate(body)


def test_retract_needs_a_reason():
    with pytest.raises(ValidationError):
        LinkRetract.model_validate({"reason": "   "})


def test_every_request_field_is_described():
    for model in (ProjectCreate, ProjectUpdate, FactorPut, LinkCreate, LinkRetract):
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name} has no description"


# --- projects ---------------------------------------------------------------------- #


def test_project_fields_and_merge():
    f = projects.project_fields(name=" A ", objective=" B ", status="drafting", sponsor="  ")
    assert (f.name, f.objective, f.sponsor) == ("A", "B", None)
    merged = projects.merge_update({"name": "A", "objective": "B", "status": "drafting", "sponsor": "S"},
                                   {"status": "terminated"})
    assert (merged.status, merged.sponsor) == ("terminated", "S")
    with pytest.raises(RecordsValidationError) as exc:
        projects.project_fields(name="A\x00", objective="B", status="drafting")
    assert exc.value.code == "invalid_body" and "\x00" not in exc.value.reason


def test_project_ids():
    pid = projects.new_project_id()
    assert projects.valid_project_id(pid)
    for bad in ("prj_", "prj_" + "g" * 32, "cmp_" + "0" * 32, pid.upper(), pid + "0"):
        assert not projects.valid_project_id(bad)


# --- factors ------------------------------------------------------------------------ #


def test_factor_fields_canonical_unit_and_dimension():
    f = factors.factor_fields(name=" glucose_g_L ", lower=0.5, upper=8, unit="g/l")
    assert (f.name, f.lower, f.upper, f.unit, f.dimension) == ("glucose_g_L", 0.5, 8.0, "g/L", "concentration")
    assert factors.factor_fields(name="n_feeds", lower=0, upper=4, unit="").unit is None


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"lower": 2, "upper": 2}, "invalid_bounds"),
    ({"lower": 3, "upper": 2}, "invalid_bounds"),
    ({"lower": float("nan"), "upper": 2}, "invalid_bounds"),
    ({"lower": True, "upper": 2}, "invalid_bounds"),
    ({"unit": "smoots"}, "unknown_unit"),
    ({"role": "outcome"}, "invalid_role"),
])
def test_factor_refusals(kwargs, code):
    args = {"name": "x", "lower": 0, "upper": 1} | kwargs
    with pytest.raises(RecordsValidationError) as exc:
        factors.factor_fields(**args)
    assert exc.value.code == code


def test_factor_physical_bounds_hook():
    def pb(name, unit):
        return factors.PhysicalBounds(0.0, 100.0, "percent") if unit == "%" else None

    assert factors.factor_fields(name="DO", lower=20, upper=60, unit="%", physical_bounds=pb).upper == 60
    with pytest.raises(RecordsValidationError) as exc:
        factors.factor_fields(name="DO", lower=20, upper=150, unit="%", physical_bounds=pb)
    assert exc.value.code == "impossible_bounds" and "0 to 100" in exc.value.reason


def test_setup_bounds_maps_only_settings_and_known_columns():
    rows = [
        {"name": "glucose", "lower_bound": 1, "upper_bound": 9, "unit": "g/L", "role": "setting"},
        {"name": "temp", "lower_bound": 30, "upper_bound": 37, "unit": "°C", "role": "setting"},
        {"name": "passage", "lower_bound": 1, "upper_bound": 40, "unit": None, "role": "covariate"},
    ]
    out = factors.setup_bounds(rows, columns=["glucose"])
    assert out["bounds"] == {"glucose": {"lo": 1.0, "hi": 9.0}} and out["applied"] is False
    assert [s["factor"] for s in out["skipped"]] == ["temp", "passage"]


# --- lineage ------------------------------------------------------------------------ #


def _row(i, p, c, pr="", cr="", rel="scale_up"):
    return {"id": i, "parent_campaign_id": p, "parent_run_id": pr, "child_campaign_id": c,
            "child_run_id": cr, "relation": rel}


def test_link_fields_rules():
    with pytest.raises(RecordsValidationError, match="two of its runs"):
        lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_A, relation="next_round")
    with pytest.raises(RecordsValidationError) as exc:
        lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_B, relation="scale_up",
                            parent_run_id="bad id!")
    assert exc.value.code == "invalid_run_id"
    with pytest.raises(RecordsValidationError) as exc:
        lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_B, relation="scale_up",
                            child_stage="stage three")
    assert exc.value.code == "invalid_stage"
    ok = lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_B, relation="derived",
                             note=" split from the fed-batch arm ", parent_stage="S2")
    assert (ok.note, ok.parent_stage, ok.parent_run_id) == ("split from the fed-batch arm", "S2", "")


def test_cycles_are_refused_across_and_within_campaigns():
    active = [_row(1, CID_A, CID_B), _row(2, CID_B, CID_C)]
    back = lineage.link_fields(parent_campaign_id=CID_C, child_campaign_id=CID_A, relation="scale_down")
    with pytest.raises(RecordsValidationError) as exc:
        lineage.check_new_link(back, active)
    assert exc.value.code == "cycle"
    sideways = lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_C, relation="follow_up")
    lineage.check_new_link(sideways, active)  # a DAG may have two paths
    inner = [_row(3, CID_A, CID_A, "r1", "r2", "next_round")]
    loop = lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_A, relation="next_round",
                               parent_run_id="r2", child_run_id="r1")
    with pytest.raises(RecordsValidationError):
        lineage.check_new_link(loop, inner)
    dup = lineage.link_fields(parent_campaign_id=CID_A, child_campaign_id=CID_B, relation="scale_up")
    with pytest.raises(RecordsValidationError) as exc:
        lineage.check_new_link(dup, active)
    assert exc.value.code == "duplicate_link"


def test_lineage_view_bounds_nodes(monkeypatch):
    star = [_row(i, CID_A, f"cmp_{i:032x}") for i in range(1, 11)]
    monkeypatch.setattr(lineage, "MAX_NODES", 4)
    view = lineage.lineage_view(CID_A, star, depth=3)
    assert len(view["nodes"]) == 4 and view["truncated"] == ["nodes"]
    assert all(link["child_campaign_id"] in {n["campaign_id"] for n in view["nodes"]} for link in view["links"])


def test_lineage_view_depth_is_clamped():
    assert lineage.lineage_view(CID_A, [], depth=10_000)["depth"] == lineage.MAX_DEPTH
    assert lineage.lineage_view(CID_A, [], depth=0)["depth"] == 1


# --- outcomes ----------------------------------------------------------------------- #


def test_run_outcome_never_mixes_measured_and_predicted():
    out = outcomes.run_outcome(
        CID_A, "r1",
        [{"version": 1, "value": 2.0}, {"version": 2, "value": 2.2, "previous_value": 2.0, "reason": "typo"}],
        {"available": 0, "unavailable_reason": "recipe was not a published proposal", "round": 1},
    )
    assert out["measured"]["value"] == 2.2 and out["measured"]["is_correction"] and out["measured"]["n_versions"] == 2
    assert out["prediction_at_start"] == {"available": False, "reason": "recipe was not a published proposal",
                                          "round": 1, "basis_kind": "predicted"}
    assert "mean" not in out["measured"] and "value" not in out["prediction_at_start"]


# --- units registry ------------------------------------------------------------------ #


@pytest.mark.parametrize(("text", "symbol"), [
    ("g/L", "g/L"), ("g/l", "g/L"), ("mg/ml", "mg/mL"), ("ug/mL", "µg/mL"), ("μg/mL", "µg/mL"),
    ("degC", "°C"), ("pct", "%"), ("hr", "h"), ("uM", "µM"), (" mM ", "mM"), ("1e6 cells/mL", "10^6 cells/mL"),
])
def test_unit_lookup_canonicalizes(text, symbol):
    u = ur.lookup(text)
    assert u is not None and u.symbol == symbol


@pytest.mark.parametrize("text", ["furlongs", "", "  ", "m", "x" * 40, None, 5])
def test_unit_lookup_refuses_unknown(text):
    assert ur.lookup(text) is None


def test_unit_registry_agrees_with_the_cell_converter():
    """Every unit marked convertible is one kalos.normalize.units converts, in the
    same dimension; the formulas stay only there."""
    from kalos.normalize import units as cell_units

    same = {"flow rate": "flow_rate", "ph": "dimensionless"}
    for u in ur.UNITS:
        if u.cell_token is None:
            continue
        assert cell_units.is_known_unit(u.cell_token), u.symbol
        dim = cell_units._UNIT_TABLE[cell_units._normalize_unit_token(u.cell_token)].dimension
        assert dim == same.get(u.dimension, u.dimension), u.symbol
    assert ur.lookup("L") is not None and ur.lookup("L").cell_token is None  # volume: never converted in a cell


def test_unit_registry_agrees_with_header_units_when_present():
    """On branch feat/data-checks `kalos.normalize.header_units` canonicalizes
    header units; each registered spelling it understands must land on the same
    symbol. Skipped until that module is on this branch."""
    hu = pytest.importorskip("kalos.normalize.header_units")
    for u in ur.UNITS:
        for spelling in (u.symbol, *u.aliases):
            parsed = hu.parse_header_unit(f"x ({spelling})")
            # pH/OD have no unit token; a power-of-ten scale is kept as written there
            if parsed is not None and u.dimension not in ("ph", "od", "cell density"):
                assert parsed == u.symbol, (spelling, parsed)
    atoms = set(hu._UNIT_ATOMS.values())
    assert atoms <= {u.symbol for u in ur.UNITS}, "a header unit atom is missing from the registry"


# --- request ids and logs -------------------------------------------------------------- #


@pytest.mark.parametrize(("incoming", "kept"), [
    ("web-1234", True), ("a" * 128, True), ("a" * 129, False), ("x\ny", False), ("<script>", False),
    ("", False), (None, False),
])
def test_request_id_acceptance(incoming, kept):
    rid = rc.accept_request_id(incoming)
    assert (rid == incoming) is kept
    if not kept:
        assert rid.startswith("req_") and len(rid) == 36


def test_json_log_lines_carry_the_request_id():
    record = logging.LogRecord("kalos.records", logging.WARNING, __file__, 1, "refused %s", ("x",), None)
    rc.RequestIdFilter().filter(record)
    line = json.loads(rc.JsonFormatter().format(record))
    assert line["request_id"] == "-" and line["message"] == "refused x" and line["level"] == "WARNING"


def test_configure_logging_is_idempotent(monkeypatch):
    monkeypatch.setenv("KALOS_LOG_FORMAT", "json")
    logger = logging.getLogger("kalos")
    before = list(logger.handlers)
    for h in before:  # a handler configured earlier in this worker
        if getattr(h, rc._CONFIGURED_ATTR, False):
            logger.removeHandler(h)
    try:
        h1 = rc.configure_logging()
        h2 = rc.configure_logging()
        assert h1 is h2 and isinstance(h1.formatter, rc.JsonFormatter)
    finally:
        for h in list(logger.handlers):
            if h not in before:
                logger.removeHandler(h)
        for h in before:
            if h not in logger.handlers:
                logger.addHandler(h)


def test_middleware_echoes_only_a_callers_usable_id():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    seen: list[str | None] = []
    app = FastAPI()

    @app.get("/x")
    def x() -> dict[str, str]:
        seen.append(rc.current_request_id())
        return {"ok": "1"}

    app.add_middleware(rc.RequestIdMiddleware)
    tc = TestClient(app)
    a, b = tc.get("/x"), tc.get("/x")
    assert "x-request-id" not in a.headers and a.content == b.content and a.headers == b.headers
    assert tc.get("/x", headers={"X-Request-ID": "web-1"}).headers["x-request-id"] == "web-1"
    assert tc.get("/x", headers={"X-Correlation-ID": "corr-2"}).headers["x-request-id"] == "corr-2"
    assert "x-request-id" not in tc.get("/x", headers={"X-Request-ID": "a b"}).headers
    assert seen[0].startswith("req_") and seen[0] != seen[1] and seen[2:4] == ["web-1", "corr-2"]
    assert seen[4].startswith("req_")  # the unsafe id was replaced, in the logs too
    assert rc.current_request_id() is None
