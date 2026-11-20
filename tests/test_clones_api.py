"""Label-only clone registry routes (kalos/portal/clones_api.py, schema v7).

Covers: registry CRUD (upsert, no campaign generation minted), the labels-only
allowlist (`sequence` and friends and any unknown key refused without echo), size
caps, the 200-clone cap, scopes, cross-tenant isolation (byte-identical 404s),
the append-only titer history (DB triggers, corrections need previous_id +
reason, unit/basis allowlist, finite numbers, paging), audit rows (no field
values), `lead_clone_label`, and the summary join (409 / unavailable / joined /
anonymized-not-joined).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kalos.portal import app as portal_module
from kalos.portal import campaign as campaign_module
from kalos.portal import clone_registry
from kalos.portal.app import app
from kalos.portal.campaign import CampaignNotFound, CampaignStore, get_campaign_store


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme",
     "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme",
     "scopes": ["read"]},
    {"token_sha256": _sha("b-write"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write"]},
]
SECRET = "GLOBEX-SECRET-Hera"
STRAIN = {"available": True, "column": "clone",
          "levels": [{"label": "CHO-A12", "n_runs": 4}, {"label": "CHO-B07", "n_runs": 5}]}


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


def _cid(store: CampaignStore, tenant: str = "acme") -> str:
    return store.create_campaign(tenant=tenant, name="c", created_by="x")["id"]


def _put(tc, cid, label="CHO-A12", token="a-write", **body):
    payload = {"status": "lead"} | body
    return tc.put(f"/api/campaigns/{cid}/clones/{label}", headers=H(token), json=payload)


def _titer(tc, cid, label="CHO-A12", token="a-write", **body):
    payload = {"basis": "harvest_titer", "value": 3.2, "unit": "g/L"} | body
    return tc.post(f"/api/campaigns/{cid}/clones/{label}/titer-history", headers=H(token), json=payload)


def _audit_rows(store: CampaignStore, tenant: str = "acme") -> list[tuple]:
    with store._txn(write=False) as conn:
        return conn.execute(
            "SELECT action, outcome, subject, campaign_id, detail FROM audit_log WHERE tenant = ? "
            "ORDER BY id", (tenant,),
        ).fetchall()


def _analysis(store, cid, strain=STRAIN, *, anonymize=False, tenant="acme"):
    result = {"target": "titer_g_L", "run": {"anonymize": anonymize}}
    if strain is not None:
        result["strain_summary"] = strain
    store.save_analysis(result, "runs.csv", tenant=tenant, campaign_id=cid)


# --- registry ------------------------------------------------------------------ #


def test_empty_registry_without_analysis(env):
    tc, store = env
    cid = _cid(store)
    r = tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read"))
    assert r.status_code == 200
    body = r.json()
    assert body["clones"] == [] and body["seen_in_data"] == [] and body["unregistered_labels"] == []
    assert body["max_clones"] == 200 and body["seen_in_data_reason"]


def test_put_creates_then_replaces_without_minting_a_generation(env):
    tc, store = env
    import pandas as pd

    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6], "y": [1.0, 2, 3, 4, 5, 7]})
    store.seed(df, "y", ["x"], tenant="acme")
    cid = store.default_campaign_id(tenant="acme")
    gen = store.generation_of(tenant="acme", campaign_id=cid)
    r = _put(tc, cid, host="CHO-K1 GS-KO", construct_id="pK-mAb7-v3", passage_at_banking="P12",
             note="line 1\nline 2")
    assert r.status_code == 200, r.text
    clone = r.json()["clone"]
    assert clone["status"] == "lead" and clone["created"] is True and clone["updated_by"] == "bob@acme"
    assert clone["fields"] == {"host": "CHO-K1 GS-KO", "construct_id": "pK-mAb7-v3",
                               "passage_at_banking": "P12", "note": "line 1\nline 2"}
    r = _put(tc, cid, status="backup", passage_at_banking=14)
    assert r.status_code == 200 and r.json()["clone"]["created"] is False
    listed = tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read")).json()["clones"]
    assert len(listed) == 1
    assert {k: listed[0][k] for k in ("label", "status", "fields")} == {
        "label": "CHO-A12", "status": "backup", "fields": {"passage_at_banking": 14.0}}
    assert set(listed[0]) == {"label", "status", "fields", "updated_at", "updated_by"}
    assert store.generation_of(tenant="acme", campaign_id=cid) == gen


@pytest.mark.parametrize("key", ["sequence", "fasta", "smiles", "pdb", "uniprot", "tenant", "notes",
                                 "Sequence", "dna_" + SECRET])
def test_put_rejects_non_label_keys_without_echo(env, key):
    tc, store = env
    cid = _cid(store)
    r = _put(tc, cid, **{key: "ATGC" + SECRET})
    assert r.status_code == 400
    body = r.json()
    assert body["code"] == "invalid_body" and "labels only" in body["error"]
    assert SECRET not in r.text and "ATGC" not in r.text
    assert clone_registry.list_clones(store, tenant="acme", campaign_id=cid) == []
    assert _audit_rows(store)[-1][:2] == ("campaign.clone.put", "http_400")


@pytest.mark.parametrize(
    "body",
    [
        {},  # status is required
        {"status": "primary"},
        {"status": "lead", "host": "x" * 65},
        {"status": "lead", "product": 7},
        {"status": "lead", "note": "n" * 501},
        {"status": "lead", "passage_at_banking": "p" * 65},
        {"status": "lead", "passage_at_banking": True},
        {"status": "lead", "host": "a\x00b"},
    ],
)
def test_put_validation_and_size_caps(env, body):
    tc, store = env
    cid = _cid(store)
    r = tc.put(f"/api/campaigns/{cid}/clones/CHO-A12", headers=H("a-write"), json=body)
    assert r.status_code == 400, r.text
    assert r.json()["code"] in ("invalid_body", "invalid_field")
    assert "x" * 65 not in r.text and "primary" not in r.text
    assert clone_registry.list_clones(store, tenant="acme", campaign_id=cid) == []


def test_put_accepts_the_limits_and_refuses_non_finite_passage(env):
    tc, store = env
    cid = _cid(store)
    assert _put(tc, cid, host="h" * 64, note="n" * 500, passage_at_banking=12.5).status_code == 200
    raw = '{"status": "lead", "passage_at_banking": NaN}'
    r = tc.put(f"/api/campaigns/{cid}/clones/CHO-A12",
               headers={**H("a-write"), "Content-Type": "application/json"}, content=raw)
    assert r.status_code == 400
    r = tc.put(f"/api/campaigns/{cid}/clones/CHO-A12",
               headers={**H("a-write"), "Content-Type": "application/json"}, content="[1, 2]")
    assert r.status_code == 400 and r.json()["code"] == "invalid_body"


@pytest.mark.parametrize("label", ["x" * 65, "a%01b", "%20%20", "a%0Ab"])
def test_label_path_param_is_checked_without_echo(env, label):
    tc, store = env
    cid = _cid(store)
    for r in (_put(tc, cid, label=label), _titer(tc, cid, label=label),
              tc.get(f"/api/campaigns/{cid}/clones/{label}/titer-history", headers=H("a-read"))):
        assert r.status_code == 400 and r.json()["code"] == "invalid_label"
        assert "x" * 65 not in r.text


def test_label_of_64_characters_is_accepted(env):
    tc, store = env
    cid = _cid(store)
    assert _put(tc, cid, label="L" * 64).status_code == 200


def test_body_size_cap(env):
    tc, store = env
    cid = _cid(store)
    big = json.dumps({"status": "lead", "note": "x" * (1024 * 1024 + 10)})
    r = tc.put(f"/api/campaigns/{cid}/clones/CHO-A12", content=big,
               headers={**H("a-write"), "content-type": "application/json"})
    assert r.status_code == 413


def test_two_hundred_clone_cap(env):
    tc, store = env
    cid = _cid(store)
    for i in range(clone_registry.MAX_CLONES_PER_CAMPAIGN):
        clone_registry.put_clone(store, tenant="acme", campaign_id=cid, label=f"C{i:03d}",
                                 status="candidate", fields={}, updated_by="u")
    r = _put(tc, cid, label="C-new")
    assert r.status_code == 409 and r.json()["code"] == "too_many_clones"
    assert r.json()["max_clones"] == 200 and "C-new" not in r.text
    assert _put(tc, cid, label="C000", status="retired").status_code == 200  # edits still allowed
    assert len(clone_registry.list_clones(store, tenant="acme", campaign_id=cid)) == 200


def test_put_summary_label_does_not_shadow_the_summary_route(env):
    tc, store = env
    cid = _cid(store)
    assert _put(tc, cid, label="summary").status_code == 200
    r = tc.get(f"/api/campaigns/{cid}/clones/summary", headers=H("a-read"))
    assert r.status_code == 409 and r.json()["code"] == "no_analysis"


def test_archived_campaign_reads_but_refuses_writes(env):
    tc, store = env
    cid = _cid(store)
    assert _put(tc, cid).status_code == 200
    store.archive_campaign(tenant="acme", campaign_id=cid)
    for r in (_put(tc, cid, status="retired"), _titer(tc, cid)):
        assert r.status_code == 409 and r.json()["code"] == "campaign_archived"
    assert tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read")).json()["clones"][0]["status"] == "lead"


# --- scopes and tenants ------------------------------------------------------------- #


def test_scopes(env):
    tc, store = env
    cid = _cid(store)
    assert _put(tc, cid, token="a-read").status_code == 403
    assert _titer(tc, cid, token="a-read").status_code == 403
    assert _put(tc, "cmp_nope", token="a-read").status_code == 403  # scope first
    assert tc.get(f"/api/campaigns/{cid}/clones").status_code == 401
    assert _put(tc, cid).status_code == 200
    assert tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read")).status_code == 200
    assert tc.get(f"/api/campaigns/{cid}/clones/CHO-A12/titer-history",
                  headers=H("a-read")).status_code == 200
    assert clone_registry.list_clones(store, tenant="acme", campaign_id=cid)[0]["updated_by"] == "bob@acme"


def test_cross_tenant_is_byte_identical_to_missing(env):
    tc, store = env
    g = _cid(store, tenant="globex")
    clone_registry.put_clone(store, tenant="globex", campaign_id=g, label="CHO-A12", status="lead",
                             fields={"host": SECRET}, updated_by="gina")
    clone_registry.append_titer(store, tenant="globex", campaign_id=g, label="CHO-A12",
                                row={"basis": "harvest_titer", "value": 1.0, "unit": "g/L"}, actor="g")
    _analysis(store, g, tenant="globex")
    missing = "cmp_" + "0" * 32
    calls = [
        ("GET", "/api/campaigns/{c}/clones", None),
        ("GET", "/api/campaigns/{c}/clones/summary", None),
        ("PUT", "/api/campaigns/{c}/clones/CHO-A12", {"status": "retired"}),
        ("POST", "/api/campaigns/{c}/clones/CHO-A12/titer-history",
         {"basis": "qP", "value": 1.0, "unit": "pg/cell/day"}),
        ("GET", "/api/campaigns/{c}/clones/CHO-A12/titer-history", None),
    ]
    for method, path, body in calls:
        kw = {"json": body} if body is not None else {}
        foreign = tc.request(method, path.format(c=g), headers=H("a-write"), **kw)
        absent = tc.request(method, path.format(c=missing), headers=H("a-write"), **kw)
        malformed = tc.request(method, path.format(c="cmp_nope"), headers=H("a-write"), **kw)
        assert foreign.status_code == 404, (method, path)
        assert foreign.content == absent.content == malformed.content
        assert SECRET not in foreign.text
    got = clone_registry.list_clones(store, tenant="globex", campaign_id=g)
    assert got[0]["status"] == "lead" and got[0]["fields"] == {"host": SECRET}
    page = clone_registry.list_titer_history(store, tenant="globex", campaign_id=g, label="CHO-A12")
    assert len(page["items"]) == 1
    with pytest.raises(CampaignNotFound):
        clone_registry.list_clones(store, tenant="acme", campaign_id=g)
    with pytest.raises(CampaignNotFound):
        clone_registry.lead_clone_label(store, tenant="acme", campaign_id=g)
    # probes of a foreign id are audited without the id
    assert all(row[3] is None for row in _audit_rows(store) if row[1] == "http_404")


# --- titer history ---------------------------------------------------------------- #


def test_titer_needs_a_registered_clone(env):
    tc, store = env
    cid = _cid(store)
    r = _titer(tc, cid, label="CHO-" + SECRET[:10])
    assert r.status_code == 404 and r.json()["code"] == "clone_not_registered"
    assert SECRET[:10] not in r.text
    r = tc.get(f"/api/campaigns/{cid}/clones/CHO-X/titer-history", headers=H("a-read"))
    assert r.status_code == 404 and r.json()["code"] == "clone_not_registered"


def test_titer_append_and_paging_newest_first(env):
    tc, store = env
    cid = _cid(store)
    _put(tc, cid)
    _put(tc, cid, label="CHO-B07", status="backup")
    ids = []
    for i in range(5):
        r = _titer(tc, cid, value=2.0 + i, measured_on=f"2025-03-0{i + 1}", scale_label="ambr15",
                   age_kind="passage", age_value=10 + i, source_note="harvest assay")
        assert r.status_code == 200, r.text
        row = r.json()["row"]
        assert row["basis_kind"] == "measured" and row["actor"] == "bob@acme" and not row["is_correction"]
        ids.append(row["id"])
    _titer(tc, cid, label="CHO-B07", basis="qP", unit="pg/cell/day", value=20.0)
    url = f"/api/campaigns/{cid}/clones/CHO-A12/titer-history"
    page = tc.get(url, params={"limit": 2}, headers=H("a-read")).json()["history"]
    assert [i["id"] for i in page["items"]] == [ids[4], ids[3]]
    assert page["has_more"] is True and page["next_before_id"] == ids[3]
    assert all(i["basis_kind"] == "measured" and i["label"] == "CHO-A12" for i in page["items"])
    page2 = tc.get(url, params={"limit": 2, "before_id": page["next_before_id"]},
                   headers=H("a-read")).json()["history"]
    assert [i["id"] for i in page2["items"]] == [ids[2], ids[1]]
    last = tc.get(url, params={"limit": 2, "before_id": ids[1]}, headers=H("a-read")).json()["history"]
    assert [i["id"] for i in last["items"]] == [ids[0]] and last["has_more"] is False
    assert last["next_before_id"] is None
    assert tc.get(url, params={"limit": 501}, headers=H("a-read")).status_code == 400


@pytest.mark.parametrize(
    ("over", "code"),
    [
        ({"unit": "mg/mL"}, "invalid_body"),
        ({"unit": "pg/cell/day"}, "unit_basis_mismatch"),
        ({"basis": "qP", "unit": "g/L"}, "unit_basis_mismatch"),
        ({"basis": "space_time_yield", "unit": "g/L"}, "unit_basis_mismatch"),
        ({"basis": "titer"}, "invalid_body"),
        ({"value": "3.2"}, "invalid_body"),
        ({"value": True}, "invalid_body"),
        ({"age_value": "P12"}, "invalid_body"),
        ({"age_kind": "days"}, "invalid_body"),
        ({"measured_on": "03/01/2025"}, "invalid_date"),
        ({"measured_on": "2025-02-30"}, "invalid_date"),
        ({"source_note": "s" * 201}, "invalid_body"),
        ({"scale_label": "s" * 65}, "invalid_body"),
        ({"sequence": "ATGC"}, "invalid_body"),
        ({"recorded_at": 1.0}, "invalid_body"),
        ({"previous_id": 1}, "reason_required"),
        ({"reason": "typo"}, "previous_id_required"),
        ({"previous_id": 999, "reason": "typo"}, "previous_not_found"),
        ({"previous_id": 0, "reason": "typo"}, "invalid_body"),
        ({"previous_id": 1, "reason": "   "}, "reason_required"),
    ],
)
def test_titer_validation(env, over, code):
    tc, store = env
    cid = _cid(store)
    _put(tc, cid)
    r = _titer(tc, cid, **over)
    assert r.status_code == 400, r.text
    assert r.json()["code"] == code
    assert "ATGC" not in r.text and "03/01/2025" not in r.text
    assert clone_registry.list_titer_history(store, tenant="acme", campaign_id=cid,
                                             label="CHO-A12")["items"] == []


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_titer_values_must_be_finite(env, token):
    tc, store = env
    cid = _cid(store)
    _put(tc, cid)
    for field in ("value", "age_value"):
        raw = '{"basis": "harvest_titer", "value": 1.0, "unit": "g/L", "' + field + '": ' + token + "}"
        raw = raw.replace('"value": 1.0, ', "") if field == "value" else raw
        r = tc.post(f"/api/campaigns/{cid}/clones/CHO-A12/titer-history", content=raw,
                    headers={**H("a-write"), "Content-Type": "application/json"})
        assert r.status_code == 400, (field, r.text)


@pytest.mark.parametrize(("basis", "unit"), [("harvest_titer", "g/L"), ("harvest_titer", "mg/L"),
                                             ("qP", "pg/cell/day"), ("space_time_yield", "g/L/day")])
def test_titer_unit_basis_allowlist(env, basis, unit):
    tc, store = env
    cid = _cid(store)
    _put(tc, cid)
    assert _titer(tc, cid, basis=basis, unit=unit).status_code == 200


def test_correction_needs_previous_id_of_the_same_clone_and_a_reason(env):
    tc, store = env
    cid = _cid(store)
    other = _cid(store)
    _put(tc, cid)
    _put(tc, cid, label="CHO-B07", status="backup")
    _put(tc, other)
    first = _titer(tc, cid).json()["row"]["id"]
    b07 = _titer(tc, cid, label="CHO-B07").json()["row"]["id"]
    elsewhere = _titer(tc, other).json()["row"]["id"]
    for prev in (b07, elsewhere):
        r = _titer(tc, cid, previous_id=prev, reason="re-assayed")
        assert r.status_code == 400 and r.json()["code"] == "previous_not_found"
    r = _titer(tc, cid, value=3.4, previous_id=first, reason="re-assayed on fresh standard")
    assert r.status_code == 200
    row = r.json()["row"]
    assert row["previous_id"] == first and row["is_correction"] and row["reason"].startswith("re-assayed")
    items = clone_registry.list_titer_history(store, tenant="acme", campaign_id=cid, label="CHO-A12")["items"]
    assert [(i["value"], i["previous_id"]) for i in items] == [(3.4, first), (3.2, None)]


def test_titer_history_is_append_only_in_sqlite(env):
    tc, store = env
    cid = _cid(store)
    _put(tc, cid)
    _titer(tc, cid)
    conn = sqlite3.connect(str(store._path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only: UPDATE"):
            conn.execute("UPDATE clone_titer_history SET value = 99")
        with pytest.raises(sqlite3.IntegrityError, match="append-only: DELETE"):
            conn.execute("DELETE FROM clone_titer_history")
    finally:
        conn.close()
    assert [i["value"] for i in clone_registry.list_titer_history(
        store, tenant="acme", campaign_id=cid, label="CHO-A12")["items"]] == [3.2]


# --- audit ---------------------------------------------------------------------- #


def test_every_write_is_audited_without_field_values(env):
    tc, store = env
    cid = _cid(store)
    _put(tc, cid, host="CHO-K1-" + SECRET, note="private " + SECRET)
    _titer(tc, cid, source_note="note " + SECRET, scale_label="ambr-" + SECRET)
    _put(tc, cid, token="a-write", sequence="x")
    rows = [r for r in _audit_rows(store) if r[0].startswith("campaign.clone.")]
    assert [(r[0], r[1]) for r in rows] == [
        ("campaign.clone.put", "ok"), ("campaign.clone.titer_append", "ok"),
        ("campaign.clone.put", "http_400")]
    assert rows[0][2] == "bob@acme" and rows[0][3] == cid
    assert "status=lead" in rows[0][4] and "keys=host,note" in rows[0][4]
    assert "basis=harvest_titer" in rows[1][4]
    assert all(SECRET not in (r[4] or "") for r in _audit_rows(store))


# --- lead_clone_label ---------------------------------------------------------------- #


def test_lead_clone_label(env):
    _tc, store = env
    cid = _cid(store)

    def put(label: str, status: str) -> None:
        clone_registry.put_clone(store, tenant="acme", campaign_id=cid, label=label, status=status,
                                 fields={}, updated_by="u")

    assert clone_registry.lead_clone_label(store, tenant="acme", campaign_id=cid) is None
    put("CHO-A12", "backup")
    assert clone_registry.lead_clone_label(store, tenant="acme", campaign_id=cid) is None
    put("CHO-B07", "lead")
    assert clone_registry.lead_clone_label(store, tenant="acme", campaign_id=cid) == "CHO-B07"
    put("CHO-A12", "lead")
    assert clone_registry.lead_clone_label(store, tenant="acme", campaign_id=cid) is None  # ambiguous
    put("CHO-A12", "retired")
    assert clone_registry.lead_clone_label(store, tenant="acme", campaign_id=cid) == "CHO-B07"


# --- seen_in_data + summary ------------------------------------------------------------- #


def test_seen_in_data_and_unregistered_labels(env):
    tc, store = env
    cid = _cid(store)
    _analysis(store, cid)
    _put(tc, cid, label="CHO-B07")
    _put(tc, cid, label="CHO-Z99", status="retired")
    body = tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read")).json()
    assert body["seen_in_data"] == ["CHO-A12", "CHO-B07"]
    assert body["unregistered_labels"] == ["CHO-A12"]
    assert "seen_in_data_reason" not in body
    _analysis(store, cid, anonymize=True)
    body = tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read")).json()
    assert body["seen_in_data"] == [] and body["unregistered_labels"] == []
    assert "anonymized" in body["seen_in_data_reason"]
    _analysis(store, cid, strain=None)
    body = tc.get(f"/api/campaigns/{cid}/clones", headers=H("a-read")).json()
    assert body["seen_in_data"] == [] and body["seen_in_data_reason"]
    assert [c["label"] for c in body["clones"]] == ["CHO-B07", "CHO-Z99"]


def test_summary_no_analysis_is_409(env):
    tc, store = env
    cid = _cid(store)
    r = tc.get(f"/api/campaigns/{cid}/clones/summary", headers=H("a-read"))
    assert r.status_code == 409
    assert r.json()["code"] == "no_analysis" and r.json()["what_would_fix_it"]


@pytest.mark.parametrize("strain", [None, {"available": False, "reason": "no clone column"},
                                    {"available": False}])
def test_summary_unavailable_is_200(env, strain):
    tc, store = env
    cid = _cid(store)
    _analysis(store, cid, strain=strain)
    r = tc.get(f"/api/campaigns/{cid}/clones/summary", headers=H("a-read"))
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and isinstance(body["reason"], str) and body["reason"]
    if strain and strain.get("reason"):
        assert body["reason"] == "no clone column"


def test_summary_joins_the_registry_and_orders_by_status_never_by_score(env):
    tc, store = env
    cid = _cid(store)
    strain = {"available": True, "column": "clone", "levels": [
        {"label": "CHO-A12", "outcomes": {"titer_g_L": {"mean": 9.9}}},
        {"label": "CHO-B07", "outcomes": {"titer_g_L": {"mean": 1.0}}},
        {"label": "CHO-C33"}, {"label": "CHO-D01"}, {"label": "CHO-0X"}, {"label": "CHO-E5"}]}
    _analysis(store, cid, strain=strain)
    _put(tc, cid, label="CHO-A12", status="retired", host="CHO-K1")
    _put(tc, cid, label="CHO-B07", status="lead")
    _put(tc, cid, label="CHO-D01", status="candidate")
    _put(tc, cid, label="CHO-C33", status="backup")
    _put(tc, cid, label="CHO-NOTDATA", status="lead")
    body = tc.get(f"/api/campaigns/{cid}/clones/summary", headers=H("a-read")).json()
    assert body["available"] is True and body["column"] == "clone"
    assert body["registry_join"] == {"applied": True, "reason": None}
    levels = {lvl["label"]: lvl for lvl in body["levels"]}
    assert levels["CHO-A12"]["registry"] == {"status": "retired", "fields": {"host": "CHO-K1"}}
    assert levels["CHO-A12"]["outcomes"] == {"titer_g_L": {"mean": 9.9}}  # analysis untouched
    assert levels["CHO-B07"]["registry"]["status"] == "lead"
    assert levels["CHO-0X"]["registry"] is None
    assert body["unregistered_labels"] == ["CHO-0X", "CHO-E5"]
    assert body["registry_order"] == ["CHO-B07", "CHO-C33", "CHO-D01", "CHO-A12", "CHO-0X", "CHO-E5"]
    # the stored analysis itself is not rewritten by the join
    assert "registry" not in store.load_analysis(tenant="acme", campaign_id=cid)["strain_summary"]["levels"][0]


def test_summary_on_an_anonymized_analysis_is_not_joined(env):
    tc, store = env
    cid = _cid(store)
    aliased = {"available": True, "column": "clone",
               "levels": [{"label": "strain_1"}, {"label": "strain_2"}]}
    _analysis(store, cid, strain=aliased, anonymize=True)
    _put(tc, cid, label="strain_1", status="lead", host="should-not-join")
    body = tc.get(f"/api/campaigns/{cid}/clones/summary", headers=H("a-read")).json()
    assert body["registry_join"] == {"applied": False, "reason": "anonymized analysis: labels are aliases"}
    assert all("registry" not in lvl for lvl in body["levels"])
    assert "registry_order" not in body and "should-not-join" not in json.dumps(body)
