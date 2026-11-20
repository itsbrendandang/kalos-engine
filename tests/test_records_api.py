"""The records routes (kalos/api/routers/{projects,lineage,units}.py, schema v9).

Every test runs on SQLite, and again on PostgreSQL when KALOS_TEST_DATABASE_URL
is set (the `records_env` fixture). Covers Voyager's own test cases where Kalos
has the behaviour (tests/test_projects.py, test_experiment_links.py,
test_outcomes.py and test_validators.py of voyager-flask-back-end, NOTICE), plus
Kalos's rules: scopes on every route, tenant isolation with byte-identical 404s,
audit rows in the same transaction carrying the request id, append-only links,
bounded responses.
"""
from __future__ import annotations

import itertools

import pytest
import records_helpers
from records_helpers import H

# Fixtures shared with tests/test_storage_backend.py (assigned, not imported, so
# the parameter names below are the fixtures, not shadowed imports).
records_env = records_helpers.records_env
pg_schema_dsn = records_helpers.pg_schema_dsn

pytestmark = pytest.mark.xdist_group("records_api")


def _project(env, token="a-write", **body):
    payload = {"name": "Media optimisation", "objective": "Raise titer above 3 g/L"} | body
    return env.tc.post("/api/projects", headers=H(token), json=payload)


def _pid(env, **body) -> str:
    r = _project(env, **body)
    assert r.status_code == 201, r.text
    return str(r.json()["project"]["id"])


def _link(env, parent, child, token="a-write", **body):
    payload = {"parent_campaign_id": parent, "child_campaign_id": child, "relation": "scale_up"} | body
    return env.tc.post("/api/lineage/links", headers=H(token), json=payload)


# --- projects (Voyager tests/test_projects.py) -------------------------------------- #


def test_create_and_get_project_trims_strings(records_env):
    env = records_env
    r = _project(env, name="  Media optimisation  ", objective="  Raise titer  ", sponsor="  Lab B ")
    assert r.status_code == 201
    p = r.json()["project"]
    assert (p["name"], p["objective"], p["sponsor"], p["status"]) == (
        "Media optimisation", "Raise titer", "Lab B", "drafting")
    assert p["id"].startswith("prj_") and len(p["id"]) == 36
    got = env.tc.get(f"/api/projects/{p['id']}", headers=H("a-read")).json()
    assert got["project"]["name"] == "Media optimisation"
    assert got["campaigns"] == [] and got["factors"] == []


@pytest.mark.parametrize("body,field", [
    ({"name": ""}, "name"), ({"name": "   "}, "name"), ({"name": "x" * 81}, "name"),
    ({"objective": ""}, "objective"), ({"objective": "x" * 501}, "objective"),
    ({"status": "Running"}, "status"), ({"sponsor": "x" * 121}, "sponsor"),
    ({"customer_id": 1}, "customer_id"),  # Voyager's CRM field is not accepted
    ({"ml_config": {}}, "ml_config"),
])
def test_create_project_validation(records_env, body, field):
    r = _project(records_env, **body)
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"
    assert field in r.json()["error"]


def test_project_name_unique_per_tenant_not_across(records_env):
    env = records_env
    _pid(env)
    dup = _project(env)
    assert dup.status_code == 409 and dup.json()["code"] == "name_taken"
    assert _project(env, token="b-write").status_code == 201  # another tenant may reuse it


def test_list_projects_paged_and_filtered(records_env):
    env = records_env
    for i in range(5):
        _pid(env, name=f"P{i}", status="in_progress" if i % 2 else "drafting")
    page = env.tc.get("/api/projects?limit=2", headers=H("a-read")).json()
    assert len(page["projects"]) == 2 and page["has_more"] and page["next_offset"] == 2
    rest = env.tc.get("/api/projects?limit=10&offset=2", headers=H("a-read")).json()
    assert len(rest["projects"]) == 3 and not rest["has_more"]
    running = env.tc.get("/api/projects?status=in_progress", headers=H("a-read")).json()
    assert {p["name"] for p in running["projects"]} == {"P1", "P3"}
    assert env.tc.get("/api/projects?status=bogus", headers=H("a-read")).json()["code"] == "invalid_status"
    assert env.tc.get("/api/projects?limit=1000", headers=H("a-read")).status_code == 422


def test_patch_project_partial_and_status_audited(records_env):
    env = records_env
    pid = _pid(env, sponsor="Lab B")
    r = env.tc.patch(f"/api/projects/{pid}", headers=H("a-write"), json={"status": "in_progress"})
    assert r.status_code == 200
    p = r.json()["project"]
    assert p["status"] == "in_progress" and p["name"] == "Media optimisation" and p["sponsor"] == "Lab B"
    cleared = env.tc.patch(f"/api/projects/{pid}", headers=H("a-write"), json={"sponsor": None}).json()
    assert cleared["project"]["sponsor"] is None
    rows = [a for a in env.audit() if a[0] == "project.update"]
    assert rows[0][4].endswith("status=drafting->in_progress")
    assert env.tc.patch(f"/api/projects/{pid}", headers=H("a-write"), json={"status": "nope"}).status_code == 422


def test_project_statuses_need_read_scope(records_env):
    env = records_env
    assert env.tc.get("/api/project-statuses").status_code == 401  # public in Voyager
    body = env.tc.get("/api/project-statuses", headers=H("a-read")).json()
    assert [s["status"] for s in body["statuses"]] == ["drafting", "in_progress", "completed", "terminated"]
    assert [s["label"] for s in body["statuses"]] == ["Drafting", "In progress", "Completed", "Terminated"]


# --- project campaigns ----------------------------------------------------------------- #


def test_attach_detach_campaign(records_env):
    env = records_env
    pid, cid = _pid(env), env.campaign(name="plate")
    r = env.tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    assert r.json() == {"project_id": pid, "campaign_id": cid, "attached": True, "created": True}
    again = env.tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write")).json()
    assert again["created"] is False
    got = env.tc.get(f"/api/projects/{pid}", headers=H("a-read")).json()
    assert [c["name"] for c in got["campaigns"]] == ["plate"] and got["project"]["n_campaigns"] == 1
    other = _pid(env, name="Other")
    moved = env.tc.put(f"/api/projects/{other}/campaigns/{cid}", headers=H("a-write"))
    assert moved.status_code == 409 and moved.json()["code"] == "campaign_in_other_project"
    d = env.tc.delete(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    assert d.json()["attached"] is False
    assert env.tc.delete(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write")).status_code == 404
    assert env.store.campaign_info(tenant="acme", campaign_id=cid)["name"] == "plate"  # untouched


def test_cannot_attach_another_tenants_campaign(records_env):
    env = records_env
    pid = _pid(env)
    foreign = env.campaign(tenant="globex")
    r = env.tc.put(f"/api/projects/{pid}/campaigns/{foreign}", headers=H("a-write"))
    missing = env.tc.put(f"/api/projects/{pid}/campaigns/cmp_{'0' * 32}", headers=H("a-write"))
    malformed = env.tc.put(f"/api/projects/{pid}/campaigns/not-a-campaign", headers=H("a-write"))
    assert r.status_code == missing.status_code == malformed.status_code == 404
    assert r.content == missing.content == malformed.content


# --- factors (Voyager FactorCreate) ------------------------------------------------------- #


def _factor(env, pid, token="a-write", **body):
    payload = {"name": "glucose_g_L", "lower_bound": 1, "upper_bound": 10, "unit": "g/L"} | body
    return env.tc.put(f"/api/projects/{pid}/factors", headers=H(token), json=payload)


def test_factor_upsert_canonicalizes_unit(records_env):
    env = records_env
    pid = _pid(env)
    r = _factor(env, pid, unit="g/l", description="  feed glucose  ")
    assert r.status_code == 200 and r.json()["created"] is True
    f = r.json()["factor"]
    assert (f["unit"], f["dimension"], f["description"], f["role"]) == ("g/L", "concentration", "feed glucose", "setting")
    r2 = _factor(env, pid, upper_bound=12.5)
    assert r2.json()["created"] is False and r2.json()["factor"]["upper_bound"] == 12.5
    listed = env.tc.get(f"/api/projects/{pid}/factors", headers=H("a-read")).json()["factors"]
    assert [x["name"] for x in listed] == ["glucose_g_L"]


@pytest.mark.parametrize("body,code", [
    ({"lower_bound": 5, "upper_bound": 5}, "invalid_bounds"),   # Voyager allowed equal bounds
    ({"lower_bound": 6, "upper_bound": 5}, "invalid_bounds"),   # Voyager validate_bounds
    ({"unit": "furlongs"}, "unknown_unit"),
    ({"name": "DO_pct", "unit": "%", "lower_bound": 10, "upper_bound": 150}, "impossible_bounds"),
    ({"name": "pH", "unit": None, "lower_bound": 5, "upper_bound": 15}, "impossible_bounds"),
    ({"name": "glutamine_mM", "unit": "mM", "lower_bound": -1, "upper_bound": 4}, "impossible_bounds"),
])
def test_factor_refusals(records_env, body, code):
    env = records_env
    pid = _pid(env)
    r = _factor(env, pid, **body)
    assert r.status_code == 400 and r.json()["code"] == code
    assert "furlongs" not in r.text  # never echoes input
    assert env.audit()[-1][:2] == ("project.factor.put", "http_400")


@pytest.mark.parametrize("body", [
    {"lower_bound": "1"}, {"lower_bound": True}, {"upper_bound": None}, {"role": "outcome"}, {"extra": 1},
])
def test_factor_body_validation(records_env, body):
    env = records_env
    assert _factor(env, _pid(env), **body).status_code == 422


def test_factor_real_bounds_and_delete(records_env):
    env = records_env
    pid = _pid(env)
    assert _factor(env, pid, name="pH", unit="pH", lower_bound=6.8, upper_bound=7.2).status_code == 200
    assert env.tc.delete(f"/api/projects/{pid}/factors?name=pH", headers=H("a-write")).json()["deleted"]
    gone = env.tc.delete(f"/api/projects/{pid}/factors?name=pH", headers=H("a-write"))
    assert gone.status_code == 404 and gone.json()["code"] == "factor_not_found"


def test_setup_bounds_preview_maps_settings_only(records_env):
    env = records_env
    pid = _pid(env)
    _factor(env, pid)
    _factor(env, pid, name="passage", unit=None, lower_bound=1, upper_bound=30, role="covariate")
    cid = env.campaign()
    env.tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    r = env.tc.get(f"/api/projects/{pid}/factors/setup-bounds", headers=H("a-read")).json()
    assert r["bounds"] == {"glucose_g_L": {"lo": 1.0, "hi": 10.0}} and r["applied"] is False
    assert r["skipped"][0]["factor"] == "passage" and r["columns_checked"] is False
    _factor(env, pid, name="feed_rate_mL_h", unit="mL/h", lower_bound=0, upper_bound=5)
    env.store.save_analysis(
        {"target": "titer", "features": ["glucose_g_L", "temp_C"], "categorical_features": []},
        "sheet.csv", tenant="acme", campaign_id=cid)
    checked = env.tc.get(f"/api/projects/{pid}/factors/setup-bounds?campaign_id={cid}", headers=H("a-read")).json()
    assert checked["columns_checked"] is True and list(checked["bounds"]) == ["glucose_g_L"]
    assert {s["factor"] for s in checked["skipped"]} == {"passage", "feed_rate_mL_h"}
    foreign = env.campaign(tenant="globex")
    bad = env.tc.get(f"/api/projects/{pid}/factors/setup-bounds?campaign_id={foreign}", headers=H("a-read"))
    assert bad.status_code == 404 and bad.json()["code"] == "campaign_not_in_project"


# --- lineage (Voyager tests/test_experiment_links.py) ---------------------------------------- #


def test_link_and_lineage_both_directions(records_env):
    env = records_env
    plate, flask, bench, other = (env.campaign(name=n) for n in ("plate", "flask", "bench", "other"))
    assert _link(env, plate, flask).status_code == 201
    assert _link(env, flask, bench, relation="scale_up", child_stage="S3").status_code == 201
    view = env.tc.get(f"/api/campaigns/{flask}/lineage", headers=H("a-read")).json()
    roles = {n["name"]: (n["role"], n["generation_offset"]) for n in view["nodes"]}
    assert roles == {"plate": ("ancestor", -1), "flask": ("start", 0), "bench": ("descendant", 1)}
    assert len(view["links"]) == 2 and view["truncated"] == []
    assert view["links"][1]["child_stage"] == "S3"
    lonely = env.tc.get(f"/api/campaigns/{other}/lineage", headers=H("a-read")).json()
    assert [n["role"] for n in lonely["nodes"]] == ["start"] and lonely["links"] == []


def test_lineage_depth_is_bounded_and_says_so(records_env):
    env = records_env
    chain = [env.campaign(name=f"c{i}") for i in range(5)]
    for a, b in itertools.pairwise(chain):
        assert _link(env, a, b).status_code == 201
    view = env.tc.get(f"/api/campaigns/{chain[0]}/lineage?depth=2", headers=H("a-read")).json()
    assert [n["generation_offset"] for n in view["nodes"]] == [0, 1, 2]
    assert view["truncated"] == ["depth"]
    assert env.tc.get(f"/api/campaigns/{chain[0]}/lineage?depth=33", headers=H("a-read")).status_code == 400


@pytest.mark.parametrize("case", ["self", "cycle", "duplicate", "derived_without_note", "foreign", "bad_run"])
def test_link_refusals(records_env, case):
    env = records_env
    a, b, c = env.campaign(), env.campaign(), env.campaign()
    if case == "self":  # Voyager: "An experiment cannot link to itself"
        r = _link(env, a, a)
        assert r.status_code == 400 and r.json()["code"] == "self_link"
    elif case == "cycle":
        _link(env, a, b)
        _link(env, b, c)
        r = _link(env, c, a)
        assert r.status_code == 409 and r.json()["code"] == "cycle"
    elif case == "duplicate":
        _link(env, a, b)
        r = _link(env, a, b)
        assert r.status_code == 409 and r.json()["code"] == "duplicate_link"
    elif case == "derived_without_note":
        r = _link(env, a, b, relation="derived")
        assert r.status_code == 400 and r.json()["code"] == "note_required"
    elif case == "foreign":
        g = env.campaign(tenant="globex")
        r = _link(env, a, g)
        missing = _link(env, a, f"cmp_{'f' * 32}")
        assert r.status_code == 404 and r.content == missing.content
    else:
        r = _link(env, a, b, parent_run_id="0123456789abcdef0123456789abcdef")
        assert r.status_code == 404 and r.json()["code"] == "run_not_found"


def test_retract_link_is_append_only(records_env):
    env = records_env
    a, b = env.campaign(), env.campaign()
    link_id = _link(env, a, b).json()["link"]["id"]
    r = env.tc.post(f"/api/lineage/links/{link_id}/retract", headers=H("a-write"), json={"reason": "  wrong parent "})
    assert r.status_code == 200
    assert r.json()["link"]["retraction"]["reason"] == "wrong parent"
    again = env.tc.post(f"/api/lineage/links/{link_id}/retract", headers=H("a-write"), json={"reason": "x"})
    assert again.status_code == 409 and again.json()["code"] == "already_retracted"
    view = env.tc.get(f"/api/campaigns/{a}/lineage", headers=H("a-read")).json()
    assert view["links"] == []  # a retracted link no longer counts
    assert _link(env, b, a).status_code == 201  # and no longer blocks the reverse
    assert env.tc.get(f"/api/lineage/links/{link_id}", headers=H("a-read")).json()["link"]["id"] == link_id
    blank = env.tc.post(f"/api/lineage/links/{link_id}/retract", headers=H("a-write"), json={"reason": "  "})
    assert blank.status_code == 422


def test_link_rows_reject_update_and_delete_in_the_database(records_env):
    from kalos.api.routers._records_common import _backends
    from kalos.storage.backend import INTEGRITY_ERRORS

    env = records_env
    a, b = env.campaign(), env.campaign()
    _link(env, a, b)
    records, _ = _backends(env.store.state_dir)
    for sql in ("UPDATE lineage_links SET relation = 'derived'", "DELETE FROM lineage_links"):
        with pytest.raises(INTEGRITY_ERRORS), records.txn(write=True) as conn:
            conn.execute(sql)


def test_lineage_within_a_campaign_joins_runs_and_refuses_run_cycles(records_env):
    env = records_env
    cid = env.campaign()
    with env.store._txn(write=True) as conn:
        for run in ("r1aaaaaaaa", "r2bbbbbbbb", "r3cccccccc"):
            conn.execute(
                "INSERT INTO run_results (tenant, campaign_id, run_id, version, value, recorded_at) "
                "VALUES ('acme', ?, ?, 1, 2.5, 1.0)", (cid, run))
    no_runs = _link(env, cid, cid, relation="next_round")
    assert no_runs.status_code == 400 and no_runs.json()["code"] == "self_link"
    assert _link(env, cid, cid, relation="next_round", parent_run_id="r1aaaaaaaa",
                 child_run_id="r2bbbbbbbb").status_code == 201
    assert _link(env, cid, cid, relation="next_round", parent_run_id="r2bbbbbbbb",
                 child_run_id="r3cccccccc").status_code == 201
    loop = _link(env, cid, cid, relation="next_round", parent_run_id="r3cccccccc", child_run_id="r1aaaaaaaa")
    assert loop.status_code == 409 and loop.json()["code"] == "cycle"
    view = env.tc.get(f"/api/campaigns/{cid}/lineage?include_outcomes=true", headers=H("a-read")).json()
    measured = {o["run_id"]: o["measured"]["value"] for o in view["run_outcomes"]}
    assert measured == {"r1aaaaaaaa": 2.5, "r2bbbbbbbb": 2.5, "r3cccccccc": 2.5}


# --- outcomes (Voyager tests/test_outcomes.py) --------------------------------------------- #


def test_project_outcomes_keep_measured_and_predicted_apart(records_env):
    env = records_env
    pid, cid = _pid(env), env.campaign()
    env.tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    with env.store._txn(write=True) as conn:
        conn.execute("INSERT INTO run_results (tenant, campaign_id, run_id, version, value, recorded_at) "
                     "VALUES ('acme', ?, 'runA', 1, 2.0, 1.0)", (cid,))
        conn.execute("INSERT INTO run_results (tenant, campaign_id, run_id, version, value, previous_value, "
                     "reason, recorded_at) VALUES ('acme', ?, 'runA', 2, 2.4, 2.0, 'transcription', 2.0)", (cid,))
        conn.execute(
            "INSERT INTO run_predictions (tenant, campaign_id, run_id, round, source, pred_source, recipe, "
            "available, pred, lo_90, hi_90, method_90, basis_90, recorded_at) "
            "VALUES ('acme', ?, 'runB', 1, 'start', 'engine', '{}', 1, 3.1, 2.5, 3.7, 'conformal', "
            "'held_out_past_runs', 1.5)", (cid,))
    body = env.tc.get(f"/api/projects/{pid}/outcomes", headers=H("a-read")).json()
    runs = {r["run_id"]: r for r in body["runs"]}
    a, b = runs["runA"], runs["runB"]
    assert a["measured"]["value"] == 2.4 and a["measured"]["n_versions"] == 2 and a["measured"]["is_correction"]
    assert a["prediction_at_start"] is None and a["prediction_reason"]
    assert b["measured"] is None and b["measured_reason"] == "no result recorded yet"
    assert b["prediction_at_start"]["interval_90"] == {
        "lo": 2.5, "hi": 3.7, "coverage": 0.9, "method": "conformal", "basis": "held_out_past_runs"}
    assert b["prediction_at_start"]["basis_kind"] == "predicted" and a["measured"]["basis_kind"] == "measured"
    other = env.campaign(tenant="globex")
    with env.store._txn(write=True) as conn:
        conn.execute("INSERT INTO run_results (tenant, campaign_id, run_id, version, value, recorded_at) "
                     "VALUES ('globex', ?, 'runG', 1, 9.0, 1.0)", (other,))
    assert "runG" not in {r["run_id"] for r in
                          env.tc.get(f"/api/projects/{pid}/outcomes", headers=H("a-read")).json()["runs"]}


# --- units ----------------------------------------------------------------------------- #


def test_units_registry_read_only(records_env):
    env = records_env
    r = env.tc.get("/api/units", headers=H("a-read")).json()
    symbols = {u["symbol"] for u in r["units"]}
    assert {"g/L", "°C", "%", "mM", "rpm", "pH"} <= symbols
    assert env.tc.post("/api/units", headers=H("a-write"), json={"data_unit": "x"}).status_code == 405


# --- scopes, isolation, audit and request ids for EVERY new route ----------------------------- #

ROUTES = [
    ("GET", "/api/project-statuses", "read"),
    ("GET", "/api/projects", "read"),
    ("POST", "/api/projects", "write"),
    ("GET", "/api/projects/{pid}", "read"),
    ("PATCH", "/api/projects/{pid}", "write"),
    ("PUT", "/api/projects/{pid}/campaigns/{cid}", "write"),
    ("DELETE", "/api/projects/{pid}/campaigns/{cid}", "write"),
    ("GET", "/api/projects/{pid}/factors", "read"),
    ("PUT", "/api/projects/{pid}/factors", "write"),
    ("DELETE", "/api/projects/{pid}/factors?name=x", "write"),
    ("GET", "/api/projects/{pid}/factors/setup-bounds", "read"),
    ("GET", "/api/projects/{pid}/outcomes", "read"),
    ("GET", "/api/projects/{pid}/lineage", "read"),
    ("GET", "/api/lineage/relations", "read"),
    ("POST", "/api/lineage/links", "write"),
    ("GET", "/api/lineage/links/{lid}", "read"),
    ("POST", "/api/lineage/links/{lid}/retract", "write"),
    ("GET", "/api/campaigns/{cid}/lineage", "read"),
    ("GET", "/api/units", "read"),
]
BODIES = {
    ("POST", "/api/projects"): {"name": "n2", "objective": "o"},
    ("PATCH", "/api/projects/{pid}"): {"status": "completed"},
    ("PUT", "/api/projects/{pid}/factors"): {"name": "x", "lower_bound": 0, "upper_bound": 1},
    ("POST", "/api/lineage/links"): {"parent_campaign_id": "{cid}", "child_campaign_id": "{cid2}", "relation": "replicate"},
    ("POST", "/api/lineage/links/{lid}/retract"): {"reason": "r"},
}


def _fixture_ids(env) -> dict[str, str]:
    pid = _pid(env)
    cid, cid2, cid3 = env.campaign(), env.campaign(), env.campaign()
    env.tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    lid = _link(env, cid2, cid3).json()["link"]["id"]
    return {"pid": pid, "cid": cid, "cid2": cid2, "lid": str(lid)}


def _fill(template, ids):
    if isinstance(template, dict):
        return {k: _fill(v, ids) for k, v in template.items()}
    return template.format(**ids) if isinstance(template, str) else template


def test_route_table_matches_the_app_and_the_scope_manifest():
    """ROUTES is every records route, with the scope the app enforces."""
    from kalos.portal.app import app
    from kalos.portal.route_scopes import build_manifest

    mine = {(e["method"], e["path"], tuple(e["scopes"])) for e in build_manifest(app)
            if e["path"].startswith(("/api/project", "/api/lineage", "/api/units"))
            or e["path"].endswith("/lineage")}
    listed = {(m, p.split("?")[0].replace("{pid}", "{project_id}").replace("{cid}", "{campaign_id}")
               .replace("{lid}", "{link_id}"), (scope,)) for m, p, scope in ROUTES}
    assert mine == listed


@pytest.mark.parametrize("method,path,scope", ROUTES)
def test_every_route_needs_its_scope(records_env, method, path, scope):
    env = records_env
    ids = _fixture_ids(env)
    url = path.format(**ids)
    body = _fill(BODIES.get((method, path)), ids)
    anon = env.tc.request(method, url, json=body)
    assert anon.status_code == 401
    as_reader = env.tc.request(method, url, json=body, headers={**H("a-read"), "X-Request-ID": "web-7"})
    if scope == "write":
        assert as_reader.status_code == 403
    else:
        assert as_reader.status_code == 200, as_reader.text
    assert as_reader.headers["x-request-id"] == "web-7"  # a caller's id is echoed
    assert "x-request-id" not in anon.headers  # a server-minted one is not (byte-identical 404s)


@pytest.mark.parametrize("method,path,scope", [r for r in ROUTES if "{" in r[1]])
def test_every_id_route_hides_other_tenants(records_env, method, path, scope):
    """globex (a writer) addressing acme's project / campaign / link gets the
    same 404 as for an id that does not exist, and nothing changes."""
    env = records_env
    ids = _fixture_ids(env)
    url = path.format(**ids)
    body = _fill(BODIES.get((method, path)), ids)
    theirs = env.tc.request(method, url, json=body, headers=H("b-write"))
    fake = {"pid": "prj_" + "0" * 32, "cid": "cmp_" + "0" * 32, "cid2": "cmp_" + "1" * 32, "lid": "999999"}
    missing = env.tc.request(method, path.format(**fake), json=_fill(BODIES.get((method, path)), fake),
                             headers=H("b-write"))
    assert theirs.status_code == 404, theirs.text
    assert theirs.content == missing.content
    after = env.tc.get(f"/api/projects/{ids['pid']}", headers=H("a-read")).json()
    assert after["project"]["status"] == "drafting" and len(after["campaigns"]) == 1
    assert env.tc.get(f"/api/lineage/links/{ids['lid']}", headers=H("a-read")).json()["link"]["retraction"] is None


def test_lists_never_show_other_tenants(records_env):
    env = records_env
    _pid(env, name="acme-only")
    g = env.tc.get("/api/projects", headers=H("b-write")).json()
    assert g["projects"] == []


def test_successful_writes_audit_in_the_same_transaction_with_the_request_id(records_env):
    env = records_env
    r = env.tc.post("/api/projects", headers={**H("a-write"), "X-Request-ID": "web-1234"},
                    json={"name": "Audited", "objective": "o"})
    pid = r.json()["project"]["id"]
    assert r.headers["x-request-id"] == "web-1234"
    action, outcome, subject, _cid, detail, request_id = env.audit()[-1]
    assert (action, outcome, subject, request_id) == ("project.create", "ok", "bob@acme", "web-1234")
    assert detail == f"project={pid};status=drafting" and "Audited" not in detail
    bad = env.tc.post("/api/projects", headers={**H("a-write"), "X-Request-ID": "bad id\nINJECT"},
                      json={"name": "Audited", "objective": "o"})
    assert bad.status_code == 409 and "x-request-id" not in bad.headers  # unsafe id: not echoed
    action, outcome, *_rest, minted = env.audit()[-1]
    assert (action, outcome) == ("project.create", "http_409") and minted.startswith("req_")


def test_limits_are_enforced(records_env, monkeypatch):
    from kalos.records import factors as fr
    from kalos.records import projects as pr

    env = records_env
    monkeypatch.setattr(pr, "MAX_PROJECTS_PER_TENANT", 2)
    _pid(env, name="a")
    _pid(env, name="b")
    assert _project(env, name="c").json()["code"] == "too_many_projects"
    monkeypatch.setattr(fr, "MAX_FACTORS_PER_PROJECT", 1)
    pid = env.tc.get("/api/projects", headers=H("a-read")).json()["projects"][0]["id"]
    assert _factor(env, pid, name="f1").status_code == 200
    assert _factor(env, pid, name="f1", upper_bound=5).status_code == 200  # a replace is not a new factor
    assert _factor(env, pid, name="f2").json()["code"] == "too_many_factors"


@pytest.mark.parametrize("header", [  # Voyager's auth_failure_case set
    {}, {"Authorization": ""}, {"Authorization": "Basic YTpi"}, {"Authorization": "Bearer"},
    {"Authorization": "Bearer not-a-known-token"}, {"Authorization": "Bearer a-write extra"},
])
@pytest.mark.parametrize("method,path", [("GET", "/api/projects"), ("POST", "/api/projects"),
                                         ("POST", "/api/lineage/links"), ("GET", "/api/units")])
def test_auth_failures_are_401_without_detail(records_env, header, method, path):
    r = records_env.tc.request(method, path, headers=header, json={})
    assert r.status_code == 401
    assert set(r.json()) == {"error", "code"}


def test_a_campaign_that_can_no_longer_be_read_can_still_be_detached(records_env):
    """N4: detaching does not look the campaign up, so a campaign deleted from
    the campaign store (here: its row removed) does not stay counted forever."""
    env = records_env
    pid, cid = _pid(env), env.campaign()
    env.tc.put(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    with env.store._txn(write=True) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM campaigns WHERE id = ?", (cid,))
    got = env.tc.get(f"/api/projects/{pid}", headers=H("a-read")).json()
    assert got["campaigns"][0]["available"] is False
    r = env.tc.delete(f"/api/projects/{pid}/campaigns/{cid}", headers=H("a-write"))
    assert r.status_code == 200 and r.json()["attached"] is False
    assert env.tc.get(f"/api/projects/{pid}", headers=H("a-read")).json()["campaigns"] == []
