"""Scale-up stage track + go/no-go gate registry (kalos/portal/gates_api.py).

Covers: default stages, stage edits + validation, recorded current_stage,
append-only gate decisions (DB triggers), decider identity from the principal,
computed evidence kept separate (and never a verdict), archived campaigns,
scopes, cross-tenant isolation (byte-identical 404s), and audit rows.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store
from kalos.portal.stages import DEFAULT_STAGES, MAX_DECISIONS_PER_CAMPAIGN


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme",
     "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]
B_SECRET = "GLOBEX-SECRET-Project-Hera"


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    app.dependency_overrides[get_campaign_store] = lambda: store
    try:
        yield TestClient(app), store
    finally:
        app.dependency_overrides.pop(get_campaign_store, None)


def _cid(store: CampaignStore, tenant: str = "acme", name: str = "c") -> str:
    return store.create_campaign(tenant=tenant, name=name, created_by="x")["id"]


def _decide(tc, cid, gate="S1", token="a-write", **body):
    payload = {"decision": "go", "note": "titer cleared the screen"} | body
    return tc.put(f"/api/campaigns/{cid}/gates/{gate}", headers=H(token), json=payload)


def _stages_body(**over):
    body = {
        "stages": [
            {"key": "S1", "label": "Screen"},
            {"key": "S2", "label": "Micro", "min_L": 0.0001, "max_L": 0.003},
            {"key": "S3", "label": "Bioreactor", "min_L": 2, "max_L": 50, "nominal_L": [2, 10, 50]},
        ],
        "current_stage": "S2",
    }
    body.update(over)
    return body


def _audit_rows(store: CampaignStore, tenant: str) -> list[tuple]:
    with store._txn(write=False) as conn:
        return conn.execute(
            "SELECT action, outcome, subject, auth_method, campaign_id, detail FROM audit_log "
            "WHERE tenant = ? ORDER BY id", (tenant,),
        ).fetchall()


# --- stages ------------------------------------------------------------------- #


def test_default_stages_when_never_edited(env):
    tc, store = env
    cid = _cid(store)
    r = tc.get(f"/api/campaigns/{cid}/stages", headers=H("a-read"))
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "default"
    assert body["current_stage"] is None
    assert body["updated_at"] is None
    keys = [s["key"] for s in body["stages"]]
    assert keys == ["S1", "S2", "S3", "S4", "S5", "S6"]
    s = {st["key"]: st for st in body["stages"]}
    assert s["S1"]["min_L"] is None and s["S1"]["max_L"] is None and s["S1"]["nominal_L"] is None
    assert (s["S2"]["min_L"], s["S2"]["max_L"]) == (0.0001, 0.003)
    assert (s["S3"]["min_L"], s["S3"]["max_L"]) == (0.05, 0.25)
    assert (s["S4"]["min_L"], s["S4"]["max_L"]) == (0.5, 1.0)
    assert s["S5"]["nominal_L"] == [2.0, 10.0, 50.0]
    assert s["S6"]["label"] == "Data package and handoff"
    for st in body["stages"]:
        assert st["gate"] == {"key": st["key"], "decision": None, "state": "open", "basis": "recorded",
                              "current": None, "history_count": 0}
    assert body["orphaned_gates"] == []
    assert len(DEFAULT_STAGES) == 6


def test_put_stages_replaces_and_records_current_stage(env):
    tc, store = env
    cid = _cid(store)
    r = tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=_stages_body())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "campaign"
    assert body["current_stage"] == "S2" and body["current_stage_basis"] == "recorded"
    assert body["updated_by"] == "bob@acme"
    assert [s["key"] for s in body["stages"]] == ["S1", "S2", "S3"]
    assert body["stages"][2]["nominal_L"] == [2.0, 10.0, 50.0]
    again = tc.get(f"/api/campaigns/{cid}/stages", headers=H("a-read")).json()
    assert again == body
    # Explicit null clears current_stage; omitting it is refused.
    r = tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=_stages_body(current_stage=None))
    assert r.status_code == 200 and r.json()["current_stage"] is None
    nocur = _stages_body()
    del nocur["current_stage"]
    assert tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=nocur).status_code == 400
    rows = [r for r in _audit_rows(store, "acme") if r[0] == "campaign.stages.update"]
    assert rows[0][:5] == ("campaign.stages.update", "ok", "bob@acme", "static", cid)
    assert rows[0][5] == "n_stages=3"


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda b: b.update(stages=[]), "invalid_body"),
        (lambda b: b.update(stages=[{"key": f"K{i}", "label": "x"} for i in range(13)]), "invalid_body"),
        (lambda b: b["stages"][1].update(key="S1"), "duplicate_stage_key"),
        (lambda b: b["stages"][0].update(key="bad key!"), "invalid_stage_key"),
        (lambda b: b["stages"][0].update(key="X" * 18), "invalid_stage_key"),
        (lambda b: b["stages"][0].update(label="   "), "invalid_label"),
        (lambda b: b["stages"][0].update(label="a\x00b"), "invalid_label"),
        (lambda b: b["stages"][0].update(label="x" * 81), "invalid_label"),
        (lambda b: b["stages"][1].update(min_L=0.01, max_L=0.001), "invalid_volume"),
        (lambda b: b["stages"][1].update(min_L=-1.0), "invalid_volume"),
        (lambda b: b["stages"][1].update(min_L=0), "invalid_volume"),
        (lambda b: b["stages"][1].update(max_L=None), "invalid_volume"),
        (lambda b: b["stages"][1].update(min_L="1"), "invalid_body"),
        (lambda b: b["stages"][1].update(min_L=True), "invalid_body"),
        (lambda b: b["stages"][1].update(max_L=1e9), "invalid_volume"),
        (lambda b: b["stages"][2].update(nominal_L=[100]), "invalid_volume"),
        (lambda b: b["stages"][2].update(nominal_L=[10, 2]), "invalid_volume"),
        (lambda b: b["stages"][0].update(nominal_L=[1]), "invalid_volume"),
        (lambda b: b["stages"][2].update(min_L=0.00001, max_L=0.0001), "stages_out_of_order"),
        (lambda b: b.update(current_stage="S9"), "invalid_current_stage"),
        (lambda b: b["stages"][0].update(extra=1), "invalid_body"),
        (lambda b: b.update(tenant="globex"), "invalid_body"),
    ],
)
def test_put_stages_validation(env, mutate, code):
    tc, store = env
    cid = _cid(store)
    body = _stages_body()
    mutate(body)
    r = tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=body)
    assert r.status_code == 400, r.text
    assert r.json()["code"] == code
    assert "bad key!" not in r.text and "S9" not in r.text and "globex" not in r.text
    # Nothing stored: still the defaults.
    assert tc.get(f"/api/campaigns/{cid}/stages", headers=H("a-read")).json()["source"] == "default"


def test_non_finite_volume_refused(env):
    tc, store = env
    cid = _cid(store)
    raw = json.dumps(_stages_body()).replace('"max_L": 50', '"max_L": Infinity')
    r = tc.put(f"/api/campaigns/{cid}/stages", headers={**H("a-write"), "Content-Type": "application/json"},
               content=raw)
    assert r.status_code == 400


# --- gates -------------------------------------------------------------------- #


def test_record_decision_append_only_history(env):
    tc, store = env
    cid = _cid(store)
    r = _decide(tc, cid, "S1", decision="pending", note="waiting on replicate plate",
                evidence=[{"kind": "run", "id": "a" * 32, "label": "plate 3"},
                          {"kind": "analysis", "id": "gen-1"}])
    assert r.status_code == 200, r.text
    rec = r.json()["recorded"]
    assert rec["decision"] == "pending" and rec["decided_by"] == "bob@acme"
    assert rec["auth_method"] == "static" and rec["basis"] == "recorded"
    assert rec["evidence_linked"] == [{"kind": "run", "id": "a" * 32, "label": "plate 3"},
                                      {"kind": "analysis", "id": "gen-1"}]
    r2 = _decide(tc, cid, "S1", token="a-admin", decision="go", note="replicates agree")
    assert r2.status_code == 200
    gate = r2.json()["gate"]
    assert gate["decision"] == "go" and gate["state"] == "recorded"
    assert gate["current"]["decided_by"] == "alice@acme"
    assert gate["history_count"] == 2
    assert [h["decision"] for h in gate["history"]] == ["go", "pending"]
    g = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()
    gates = {x["key"]: x for x in g["gates"]}
    assert gates["S1"]["decision"] == "go" and len(gates["S1"]["history"]) == 2
    assert gates["S2"]["decision"] is None and gates["S2"]["state"] == "open"
    stages = tc.get(f"/api/campaigns/{cid}/stages", headers=H("a-read")).json()
    assert stages["stages"][0]["gate"]["history_count"] == 2
    assert "history" not in stages["stages"][0]["gate"]
    audits = [a for a in _audit_rows(store, "acme") if a[0] == "campaign.gate.decide"]
    assert len(audits) == 2 and audits[0][1] == "ok" and audits[0][4] == cid
    assert "waiting" not in (audits[0][5] or "")  # the note never lands in the audit row


def test_gate_decisions_table_rejects_update_and_delete(env):
    tc, store = env
    cid = _cid(store)
    assert _decide(tc, cid).status_code == 200
    with pytest.raises(sqlite3.DatabaseError, match="append-only"), store._txn(write=True) as conn:
        conn.execute("UPDATE gate_decisions SET decision = 'no_go' WHERE tenant = ?", ("acme",))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"), store._txn(write=True) as conn:
        conn.execute("DELETE FROM gate_decisions WHERE tenant = ?", ("acme",))
    g = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()
    assert g["gates"][0]["decision"] == "go"


def test_decider_identity_cannot_come_from_body(env):
    tc, store = env
    cid = _cid(store)
    for extra in ({"decided_by": "mallory"}, {"decided_at": 1.0}, {"auth_method": "jwt"},
                  {"tenant": "globex"}):
        r = _decide(tc, cid, **extra)
        assert r.status_code == 400 and r.json()["code"] == "invalid_body"
        assert "mallory" not in r.text
    assert tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()["gates"][0]["history_count"] == 0


@pytest.mark.parametrize(
    "body,code",
    [
        ({"decision": "maybe", "note": "x"}, "invalid_body"),
        ({"decision": "go"}, "invalid_body"),
        ({"decision": "go", "note": ""}, "note_required"),
        ({"decision": "go", "note": "   \n "}, "note_required"),
        ({"decision": "go", "note": "x" * 2001}, "invalid_body"),
        ({"decision": "go", "note": "bad\x07bell"}, "invalid_note"),
        ({"decision": "go", "note": 5}, "invalid_body"),
        ({"decision": "go", "note": "x", "evidence": [{"kind": "x", "id": "a"}]}, "invalid_body"),
        ({"decision": "go", "note": "x", "evidence": [{"kind": "run", "id": "../etc"}]}, "invalid_body"),
        ({"decision": "go", "note": "x", "evidence": [{"kind": "run", "id": "a" * 129}]}, "invalid_body"),
        ({"decision": "go", "note": "x", "evidence": [{"kind": "run", "id": "a", "label": "\x01"}]},
         "invalid_body"),
        ({"decision": "go", "note": "x", "evidence": [{"kind": "run", "id": "a"}] * 21}, "invalid_body"),
        ({"decision": "go", "note": "x", "evidence": [{"kind": "run", "id": "a", "x": 1}]}, "invalid_body"),
    ],
)
def test_decision_validation(env, body, code):
    tc, store = env
    cid = _cid(store)
    r = tc.put(f"/api/campaigns/{cid}/gates/S1", headers=H("a-write"), json=body)
    assert r.status_code == 400, r.text
    assert r.json()["code"] == code
    assert "../etc" not in r.text
    assert tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()["gates"][0]["history_count"] == 0


def test_unknown_gate_is_404_without_echo(env):
    tc, store = env
    cid = _cid(store)
    for gate in ("S7", "NOPE-SECRET", "S1%00", "x" * 300, "S6x"):
        r = _decide(tc, cid, gate)
        assert r.status_code == 404
        assert r.json() == {"error": "gate not found", "code": "gate_not_found"}
    # A gate key not in the campaign's own definitions.
    tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=_stages_body())
    assert _decide(tc, cid, "S5").status_code == 404
    assert _decide(tc, cid, "S3").status_code == 200


def test_stage_edit_keeps_history_as_orphaned_gate(env):
    tc, store = env
    cid = _cid(store)
    assert _decide(tc, cid, "S5", decision="no_go", note="titer collapsed at 10 L").status_code == 200
    r = tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=_stages_body())
    assert r.status_code == 200
    orphan = r.json()["orphaned_gates"]
    assert len(orphan) == 1 and orphan[0]["key"] == "S5" and orphan[0]["decision"] == "no_go"
    assert orphan[0]["in_stage_definitions"] is False and orphan[0]["history_count"] == 1
    g = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()
    assert g["orphaned_gates"][0]["history"][0]["note"] == "titer collapsed at 10 L"
    # Cannot record against the orphan until a stage with that key exists again.
    assert _decide(tc, cid, "S5").status_code == 404


def test_computed_evidence_absent_then_present_and_never_a_verdict(env):
    tc, store = env
    cid = _cid(store)
    g = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()
    ce = g["gates"][0]["computed_evidence"]
    assert ce["available"] is False and ce["basis"] == "computed" and ce["reason"]
    analysis = {
        "target": "titer", "n": 24, "best": 3.21,
        "reliability": {"spearman": 0.12, "ci95": [-0.2, 0.4], "spearman_floor": 0.3,
                        "clears_floor": False, "ci_excludes_zero": False},
        "promotion": {"passed": False, "failures": ["spearman below floor"], "summary": "not promoted"},
    }
    store.save_analysis(analysis, "runs.csv", tenant="acme", campaign_id=cid)
    g = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()
    gate = g["gates"][0]
    ce = gate["computed_evidence"]
    assert ce["available"] is True and ce["basis"] == "computed"
    assert ce["n_measured"] == 24 and ce["best_measured"] == {"value": 3.21, "basis": "measured"}
    assert ce["reliability"]["clears_floor"] is False and ce["reliability"]["ci95"] == [-0.2, 0.4]
    assert ce["promotion"]["passed"] is False
    # Failing computed evidence does NOT produce a decision.
    assert gate["decision"] is None and gate["state"] == "open"
    assert "decision" not in ce
    # A recorded go stands even though the computed promotion failed.
    r = _decide(tc, cid, "S1", decision="go", note="team accepts the risk")
    assert r.json()["gate"]["decision"] == "go"
    assert r.json()["gate"]["computed_evidence"]["promotion"]["passed"] is False
    assert r.json()["recorded"]["evidence_linked"] == []


def test_older_analysis_missing_fields_reads_null_not_fabricated(env):
    tc, store = env
    cid = _cid(store)
    store.save_analysis({"target": "titer"}, "runs.csv", tenant="acme", campaign_id=cid)
    ce = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()["gates"][0]["computed_evidence"]
    assert ce["available"] is True
    assert ce["n_measured"] is None and ce["best_measured"]["value"] is None
    assert ce["reliability"] is None and ce["promotion"] is None


# --- archived, scopes, tenants ------------------------------------------------ #


def test_archived_campaign_refuses_writes_but_reads(env):
    tc, store = env
    cid = _cid(store)
    assert _decide(tc, cid).status_code == 200
    store.update_campaign(tenant="acme", campaign_id=cid, archived=True)
    r = tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-write"), json=_stages_body())
    assert r.status_code == 409 and r.json()["code"] == "campaign_archived"
    r = _decide(tc, cid, decision="no_go", note="x")
    assert r.status_code == 409 and r.json()["code"] == "campaign_archived"
    assert tc.get(f"/api/campaigns/{cid}/stages", headers=H("a-read")).status_code == 200
    g = tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read")).json()
    assert g["gates"][0]["history_count"] == 1
    outcomes = [a[1] for a in _audit_rows(store, "acme") if a[0].startswith("campaign.")]
    assert outcomes.count("http_409") == 2


def test_read_token_cannot_write(env):
    tc, store = env
    cid = _cid(store)
    r = tc.put(f"/api/campaigns/{cid}/stages", headers=H("a-read"), json=_stages_body())
    assert r.status_code == 403
    r = _decide(tc, cid, token="a-read")
    assert r.status_code == 403
    # Same 403 for a foreign or malformed id: scope is checked first.
    assert _decide(tc, "cmp_nope", token="a-read").status_code == 403
    assert tc.get(f"/api/campaigns/{cid}/stages").status_code == 401
    assert tc.get(f"/api/campaigns/{cid}/gates").status_code == 401


def _routes():
    return [
        ("GET", "/api/campaigns/{cid}/stages", {}),
        ("PUT", "/api/campaigns/{cid}/stages", {"json": _stages_body()}),
        ("GET", "/api/campaigns/{cid}/gates", {}),
        ("PUT", "/api/campaigns/{cid}/gates/S1", {"json": {"decision": "go", "note": "x"}}),
        ("PUT", "/api/campaigns/{cid}/gates/NOPE", {"json": {"decision": "go", "note": "x"}}),
    ]


def _raw(resp):
    return resp.status_code, sorted(resp.headers.items()), resp.content


def test_cross_tenant_byte_identical_404_and_untouched(env):
    tc, store = env
    b_cid = _cid(store, "globex", B_SECRET)
    assert _decide(tc, b_cid, "S2", token="b-admin", note=B_SECRET).status_code == 200
    with store._txn(write=False) as conn:
        before = conn.execute("SELECT * FROM gate_decisions ORDER BY id").fetchall()
    for method, path, kw in _routes():
        foreign = tc.request(method, path.format(cid=b_cid), headers=H("a-admin"), **kw)
        missing = tc.request(method, path.format(cid="cmp_" + uuid.uuid4().hex), headers=H("a-admin"), **kw)
        malformed = tc.request(method, path.format(cid="cmp_nope"), headers=H("a-admin"), **kw)
        assert foreign.status_code == 404, (method, path, foreign.text)
        assert _raw(foreign) == _raw(missing) == _raw(malformed)
        assert foreign.json() == {"error": "campaign not found", "code": "not_found"}
        assert B_SECRET not in foreign.text
    with store._txn(write=False) as conn:
        after = conn.execute("SELECT * FROM gate_decisions ORDER BY id").fetchall()
    assert before == after
    # globex's stages untouched and its decision visible only to globex.
    own = tc.get(f"/api/campaigns/{b_cid}/stages", headers=H("b-admin")).json()
    assert own["source"] == "default" and own["stages"][1]["gate"]["decision"] == "go"
    # acme audit rows never carry globex's campaign id.
    assert all(a[4] != b_cid for a in _audit_rows(store, "acme"))


def test_decisions_are_tenant_scoped_in_sql(env):
    """A decision row for another tenant with the same campaign id string is
    never read: every SELECT is filtered by tenant."""
    tc, store = env
    cid = _cid(store)
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    try:
        tc.get(f"/api/campaigns/{cid}/gates", headers=H("a-read"))
        _decide(tc, cid)
    finally:
        store._conn.set_trace_callback(None)
    stmts = [s for s in seen if "gate_decisions" in s or "campaign_settings" in s]
    assert stmts
    for s in stmts:
        if s.startswith("INSERT INTO gate_decisions"):
            assert "VALUES ('acme'" in s  # tenant is the principal's, bound as a column
        else:
            assert "tenant = 'acme'" in s, s


def test_decision_limit(env, monkeypatch):
    tc, store = env
    cid = _cid(store)
    import kalos.portal.stages as stages_module
    monkeypatch.setattr(stages_module, "MAX_DECISIONS_PER_CAMPAIGN", 2)
    assert _decide(tc, cid).status_code == 200
    assert _decide(tc, cid).status_code == 200
    r = _decide(tc, cid)
    assert r.status_code == 409 and r.json()["code"] == "gate_decision_limit"
    assert MAX_DECISIONS_PER_CAMPAIGN >= 1000
