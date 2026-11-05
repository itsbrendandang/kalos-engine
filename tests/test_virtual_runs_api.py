"""`/api/campaigns/{id}/virtual-runs*` (kalos/portal/virtual_runs_api.py,
docs/VIRTUAL_RUNS.md): tenant scoping, scopes, storage, golden replay from the
stored record, the labelled export and its refusal of mixed exports. Every
database lives in tmp_path."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import busy
from kalos.portal import campaign as campaign_module
from kalos.portal import virtual_runs as vr
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.uploads import UploadRejected
from kalos.twin import generators as g


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-write"), "subject": "hank@globex", "tenant": "globex",
     "scopes": ["read", "write"]},
]


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    busy.reset_limiter(total=2, per_tenant=1)
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    portal_module.app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        tc = TestClient(portal_module.app)
        cid = tc.post("/api/campaigns", json={"name": "c"}, headers=H("a-write")).json()["id"]
        yield tc, store, cid
    finally:
        portal_module.app.dependency_overrides.pop(get_campaign_store, None)
        busy.reset_limiter()
        store.close()


def _create(tc, cid, token="a-write", **body):
    payload = {"generator": "bench.fed_batch", "n": 5, **body}
    return tc.post(f"/api/campaigns/{cid}/virtual-runs", json=payload, headers=H(token))


def _count(store: CampaignStore) -> int:
    return store._conn.execute("SELECT COUNT(*) FROM virtual_runs").fetchone()[0]


def test_create_stores_labelled_attributed_seeded_rows(env):
    tc, store, cid = env
    r = _create(tc, cid, params={"noise_sd_g": 1.0})
    assert r.status_code == 201
    body = r.json()
    assert re.fullmatch(r"vjb_[0-9a-f]{32}", body["batch_id"])
    assert body["origin"] == "synthetic" and body["label"] == "SYNTHETIC - not measured"
    assert body["requested_by"] == "bob@acme" and body["n"] == 5
    assert body["params"]["noise_sd_g"] == 1.0 and len(body["params_sha256"]) == 64
    assert body["solver"]["method"] == "LSODA"
    assert body["library_versions"]["numpy"]
    expected = g.seed_entropy(tenant="acme", campaign_id=cid, twin_version=None,
                              job_id=body["batch_id"])
    assert body["seed_entropy"] == [str(e) for e in expected]
    assert all(run["origin"] == "synthetic" for run in body["runs"])
    rows = store._conn.execute(
        "SELECT tenant, campaign_id, origin, requested_by, auth_method, generator_id "
        "FROM virtual_runs ORDER BY k").fetchall()
    assert rows == [("acme", cid, "synthetic", "bob@acme", "static", "bench.fed_batch")] * 5


def test_golden_replay_from_the_stored_record(env):
    """The stored entropy + generator + params reproduce the stored rows exactly."""
    tc, store, cid = env
    body = _create(tc, cid, generator="bench.bioprocess_like", n=8, params={"noise": 0.2}).json()
    stored = vr.list_runs(store, tenant="acme", campaign_id=cid)
    replay = g.generate(stored[0]["generator_id"], stored[0]["params_json"], len(stored),
                        tenant="acme", campaign_id=cid, job_id=body["batch_id"])
    assert [str(e) for e in replay.seed_entropy] == stored[0]["seed_entropy"]
    assert replay.pcg64_state == stored[0]["pcg64_state"]
    assert [(r.design, r.outcome) for r in replay.rows] == [
        (s["design_json"], s["outcome_json"]) for s in stored]
    assert replay.generator_version == stored[0]["generator_version"]


def test_scopes_and_auth(env):
    tc, store, cid = env
    assert _create(tc, cid, token="a-read").status_code == 403
    r = tc.post(f"/api/campaigns/{cid}/virtual-runs", json={"generator": "bench.fed_batch", "n": 1})
    assert r.status_code == 401
    assert tc.get(f"/api/campaigns/{cid}/virtual-runs", headers=H("a-read")).status_code == 200
    assert tc.get("/api/virtual-runs/generators", headers=H("a-read")).status_code == 200
    assert _count(store) == 0


def test_another_tenants_campaign_reads_exactly_like_a_missing_one(env):
    tc, store, cid = env
    _create(tc, cid)
    missing = "cmp_" + "0" * 32
    for path in ("virtual-runs", "virtual-runs/export"):
        foreign = tc.get(f"/api/campaigns/{cid}/{path}", headers=H("b-write"))
        absent = tc.get(f"/api/campaigns/{missing}/{path}", headers=H("b-write"))
        malformed = tc.get(f"/api/campaigns/nope/{path}", headers=H("b-write"))
        assert foreign.status_code == absent.status_code == malformed.status_code == 404
        assert foreign.content == absent.content == malformed.content
        assert cid not in foreign.text
    r = _create(tc, cid, token="b-write")
    assert r.status_code == 404 and r.content == _create(tc, missing, token="b-write").content
    assert _count(store) == 5  # only acme's own batch


def test_archived_campaign_refuses_new_batches(env):
    tc, store, cid = env
    store.archive_campaign(tenant="acme", campaign_id=cid)
    r = _create(tc, cid)
    assert r.status_code == 409 and r.json()["code"] == "campaign_archived"
    assert _count(store) == 0


@pytest.mark.parametrize("body", [
    {"generator": "nope"},
    {"params": {"evil": 1}},
    {"params": {"mu_max": -1.0}},
    {"params": {"mu_max": "0.3"}},
])
def test_bad_generator_requests_are_refused_and_write_nothing(env, body):
    tc, store, cid = env
    r = _create(tc, cid, **body)
    assert r.status_code == 400 and r.json()["code"] == "invalid_generator_request"
    assert "evil" not in r.text
    assert _count(store) == 0


@pytest.mark.parametrize("n", [0, g.MAX_ROWS_PER_BATCH + 1, "5", True, 2.5])
def test_n_is_bounded(env, n):
    tc, store, cid = env
    assert _create(tc, cid, n=n).status_code == 400
    assert _count(store) == 0


def test_list_pages_and_filters(env):
    tc, _, cid = env
    b1 = _create(tc, cid, n=3).json()["batch_id"]
    _create(tc, cid, n=2, generator="bench.gaussian_bump")
    r = tc.get(f"/api/campaigns/{cid}/virtual-runs?limit=2", headers=H("a-read")).json()
    assert len(r["runs"]) == 2 and r["next_after"] is not None
    assert all(x["label"] == "SYNTHETIC - not measured" for x in r["runs"])
    rest = tc.get(f"/api/campaigns/{cid}/virtual-runs?after={r['next_after']}",
                  headers=H("a-read")).json()
    assert len(rest["runs"]) == 3 and rest["next_after"] is None
    only = tc.get(f"/api/campaigns/{cid}/virtual-runs?batch_id={b1}", headers=H("a-read")).json()
    assert [x["k"] for x in only["runs"]] == [0, 1, 2]
    assert tc.get(f"/api/campaigns/{cid}/virtual-runs?origin=simulated",
                  headers=H("a-read")).json()["runs"] == []
    assert tc.get(f"/api/campaigns/{cid}/virtual-runs?origin=measured",
                  headers=H("a-read")).status_code == 400
    assert tc.get(f"/api/campaigns/{cid}/virtual-runs?batch_id=x",
                  headers=H("a-read")).status_code == 400


def test_export_is_labelled_and_a_reupload_is_refused(env):
    tc, _, cid = env
    _create(tc, cid, n=20, generator="bench.gaussian_bump")
    r = tc.get(f"/api/campaigns/{cid}/virtual-runs/export", headers=H("a-read"))
    assert r.status_code == 200
    assert r.headers["x-kalos-origin"] == "synthetic"
    assert f'filename="kalos-synthetic-virtual-runs-{cid}.csv"' in r.headers["content-disposition"]
    lines = r.text.splitlines()
    assert lines[0].startswith("# ORIGIN: SYNTHETIC") and "NOT lab measurements" in lines[0]
    assert any("generator bench.gaussian_bump 1+" in ln and "seed_entropy" in ln for ln in lines)
    header = next(ln for ln in lines if not ln.startswith("#"))
    assert header.split(",")[0] == "origin"
    data = [ln for ln in lines if not ln.startswith("#")][1:]
    assert len(data) == 20 and all(ln.startswith("synthetic,") for ln in data)
    # As exported, the parser already refuses the file (the comment lines are not a
    # table). With the comment lines stripped, the origin guard refuses it.
    up = tc.post("/api/run", files={"file": ("x.csv", r.content, "text/csv")},
                 data={"target": "y"}, headers=H("a-write"))
    assert up.status_code == 400
    stripped = "\n".join(ln for ln in lines if not ln.startswith("#")).encode()
    for path in ("/api/run", f"/api/campaigns/{cid}/upload", f"/api/campaigns/{cid}/jobs"):
        up2 = tc.post(path, files={"file": ("x.csv", stripped, "text/csv")},
                      data={"target": "y"}, headers=H("a-write"))
        assert up2.status_code == 400 and up2.json()["code"] == "virtual_data_refused", path


def _insert_simulated(store: CampaignStore, cid: str) -> None:
    store._conn.execute(
        "INSERT INTO virtual_runs (tenant, campaign_id, batch_id, k, origin, generator_id, "
        "generator_version, params_json, params_sha256, seed_entropy, pcg64_state, solver_json, "
        "library_versions, twin_version_id, design_json, design_sha256, outcome_json, "
        "requested_by, auth_method, created_at) VALUES ('acme', ?, ?, 0, 'simulated', 'test.sim', "
        "'1+x', '{}', ?, '[1,2,3,4]', '{}', '{}', '{}', 'tv_1', '{\"x1\": 0.5}', ?, "
        "'{\"y\": 1.0}', 'scheduler-for:bob@acme', 'bearer', 0)",
        (cid, "vjb_" + "f" * 32, "a" * 64, "b" * 64),
    )
    store._conn.commit()


def test_mixed_exports_are_refused(env):
    tc, store, cid = env
    _create(tc, cid, n=3, generator="bench.gaussian_bump")
    _insert_simulated(store, cid)
    r = tc.get(f"/api/campaigns/{cid}/virtual-runs/export", headers=H("a-read"))
    assert r.status_code == 409 and r.json()["code"] == "mixed_export_refused"
    syn = tc.get(f"/api/campaigns/{cid}/virtual-runs/export?origin=synthetic", headers=H("a-read"))
    assert syn.status_code == 200 and "simulated" not in syn.text.split("\n", 3)[-1]
    sim = tc.get(f"/api/campaigns/{cid}/virtual-runs/export?origin=simulated", headers=H("a-read"))
    assert sim.status_code == 200 and sim.headers["x-kalos-origin"] == "simulated"
    assert "SIMULATED - twin tv_1 - not measured" in sim.text
    assert "kalos-simulated-virtual-runs" in sim.headers["content-disposition"]
    both = tc.get(f"/api/campaigns/{cid}/virtual-runs/export?origin=synthetic&include_measured=true",
                  headers=H("a-read"))
    assert both.status_code == 409 and both.json()["code"] == "mixed_export_refused"


def test_export_of_nothing_is_a_404_with_a_code(env):
    tc, _, cid = env
    r = tc.get(f"/api/campaigns/{cid}/virtual-runs/export", headers=H("a-read"))
    assert r.status_code == 404 and r.json()["code"] == "no_virtual_runs"


def test_csv_cells_cannot_start_a_formula():
    assert vr._safe_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert vr._safe_cell("-1") == "'-1"
    assert vr._safe_cell(-1.0) == -1.0 and vr._safe_cell("ok") == "ok"


def test_stored_rows_cannot_reach_the_analysis(env):
    from kalos.portal.analysis import _analyze

    tc, store, cid = env
    _create(tc, cid, n=20, generator="bench.gaussian_bump")
    df = vr.rows_frame(vr.list_runs(store, tenant="acme", campaign_id=cid))
    with pytest.raises(UploadRejected) as info:
        _analyze(df.drop(columns=["y_true"]), "y")
    assert info.value.code == "virtual_data_refused"


def test_every_statement_on_virtual_runs_names_the_tenant(env):
    _, store, cid = env
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    vr.create_batch(store, tenant="acme", campaign_id=cid, generator_id="bench.gaussian_bump",
                    params=None, n=2, requested_by="bob")
    vr.list_runs(store, tenant="acme", campaign_id=cid, origin="synthetic")
    vr.export_csv(store, tenant="acme", campaign_id=cid)
    store._conn.set_trace_callback(None)
    hits = [s for s in seen if re.search(r"\b(virtual_runs|campaigns)\b", s)]
    assert hits
    for sql in hits:
        assert "tenant" in sql, sql


def test_store_rejects_bad_ids_before_sql(env):
    _, store, _ = env
    with pytest.raises(campaign_module.CampaignNotFound):
        vr.list_runs(store, tenant="acme", campaign_id="cmp_bad")
    with pytest.raises(ValueError):
        vr.list_runs(store, tenant="bad tenant!", campaign_id="cmp_" + "0" * 32)


def test_open_mode_actor_is_recorded(env):
    _, store, cid = env
    out = vr.create_batch(store, tenant="acme", campaign_id=cid, generator_id="bench.ackley",
                          params=None, n=1, requested_by=None)
    assert out["requested_by"] == "anonymous"
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("UPDATE virtual_runs SET requested_by = 'x'")
    store._conn.rollback()
