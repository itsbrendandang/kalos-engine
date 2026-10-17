"""Cross-tenant release gate for the multi-campaign API (`/api/campaigns*`, E4).

Two tenants authenticated with STATIC bearer tokens (`KALOS_AUTH_TOKENS`):
`acme` (admin, write-only and read-only tokens) and `globex` (admin token).
JWT tenants are not covered here because the E2 verifier is not on this base;
see tests/test_cross_tenant_isolation_jwt.py (a marked TODO stub).

What this file proves, route by route:

- another tenant's campaign id gets the byte-identical 404 (status, headers,
  body) a missing id gets, on every route and method, and runs the same SQL;
- lists, analyses and audit rows are scoped to the caller's tenant;
- campaign ids are uuid4-random, not guessable;
- the legacy single-campaign routes touch only the caller's default campaign;
- odd ids (traversal, encoded slashes, unicode, huge) fail safely and are
  never echoed back;
- scopes: read cannot write, write cannot archive, admin can;
- production mode serves the API with the same isolation;
- a reanalyze in one tenant never shows up in the other's analysis;
- concurrent creation by two tenants stays separated;
- body-size limits and the single analysis slot still apply;
- no response body or log line seen by one tenant carries the other's names.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import uuid

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kalos.portal import app as portal_module  # noqa: E402
from kalos.portal import busy  # noqa: E402
from kalos.portal import campaign as campaign_module  # noqa: E402
from kalos.portal.app import app  # noqa: E402
from kalos.portal.campaign import CampaignStore, get_campaign_store  # noqa: E402
from kalos.portal.campaigns_api import MAX_NAME_CHARS, CampaignUploadSizeLimit  # noqa: E402
from kalos.portal.uploads import MAX_UPLOAD_BYTES  # noqa: E402

A_SECRET = "ACME-SECRET-Project-Zeus"
B_SECRET = "GLOBEX-SECRET-Project-Hera"
B_DATASET = "globex-secret-runs.csv"
CID = re.compile(r"cmp_[0-9a-f]{32}")


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


def H(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _df(n: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 4, n)
    ph = rng.uniform(5, 7, n)
    return pd.DataFrame({
        "Methanol": m.round(3), "pH": ph.round(2),
        "lipase_titer": (1.5 * m - 0.8 * (ph - 6) ** 2 + rng.normal(0, 0.1, n)).round(3),
    })


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _recipe(m: float = 2.0, ph: float = 6.0) -> dict:
    return {"recipe": {"Methanol": m, "pH": ph}, "pred": 1.0, "std": 0.1,
            "mode": "explore", "reason": "x"}


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


def _globex_campaign(store: CampaignStore) -> tuple[str, str]:
    """A seeded globex campaign with a published analysis and one pending run.
    Returns (campaign id, run id)."""
    cid = store.create_campaign(tenant="globex", name=B_SECRET, created_by="gina@globex")["id"]
    store.seed(_df(8, 7), "lipase_titer", ["Methanol", "pH"], tenant="globex",
               campaign_id=cid, analysis={"target": "lipase_titer", "marker": B_SECRET},
               dataset=B_DATASET)
    run = store.start([_recipe()], tenant="globex", campaign_id=cid)[0]
    return cid, run["id"]


def _snapshot(store: CampaignStore, cid: str) -> tuple:
    return (store.campaign_info(tenant="globex", campaign_id=cid),
            store.get(tenant="globex", campaign_id=cid),
            store.load_analysis(tenant="globex", campaign_id=cid))


def _missing_id() -> str:
    return "cmp_" + uuid.uuid4().hex


def _route_matrix(run_id: str) -> list[tuple[str, str, dict]]:
    """Every campaign-scoped route + method, with a well-formed request."""
    return [
        ("GET", "/api/campaigns/{cid}", {}),
        ("PATCH", "/api/campaigns/{cid}", {"json": {"name": "hijacked"}}),
        ("PATCH", "/api/campaigns/{cid}", {"json": {"archived": True}}),
        ("PATCH", "/api/campaigns/{cid}", {"json": {"archived": False}}),
        ("POST", "/api/campaigns/{cid}/upload",
         {"files": {"file": ("runs.csv", _csv(_df()), "text/csv")}}),
        ("GET", "/api/campaigns/{cid}/analysis", {}),
        ("POST", "/api/campaigns/{cid}/runs", {"json": {"recipes": [_recipe()]}}),
        ("POST", "/api/campaigns/{cid}/runs/" + run_id + "/result", {"json": {"value": 9.9}}),
        ("POST", "/api/campaigns/{cid}/reanalyze", {}),
    ]


def _raw(resp) -> tuple[int, list[tuple[str, str]], bytes]:
    return resp.status_code, sorted(resp.headers.items()), resp.content


# --- 1. foreign id == missing id, on every route ------------------------------- #


def test_foreign_campaign_id_is_byte_identical_to_a_missing_one_on_every_route(env):
    tc, store = env
    b_cid, b_run = _globex_campaign(store)
    before = _snapshot(store, b_cid)
    for method, path, kw in _route_matrix(b_run):
        foreign = tc.request(method, path.format(cid=b_cid), headers=H("a-admin"), **kw)
        missing = tc.request(method, path.format(cid=_missing_id()), headers=H("a-admin"), **kw)
        malformed = tc.request(method, path.format(cid="cmp_nope"), headers=H("a-admin"), **kw)
        assert foreign.status_code == 404, (method, path, foreign.text)
        assert _raw(foreign) == _raw(missing) == _raw(malformed), (method, path)
        assert foreign.json() == {"error": "campaign not found"}
    # nothing about globex's campaign changed
    assert _snapshot(store, b_cid) == before
    # and the owner still sees it normally
    assert tc.get(f"/api/campaigns/{b_cid}", headers=H("b-admin")).status_code == 200


def test_foreign_and_missing_ids_run_the_same_sql(env):
    """Same code path, not just the same bytes: the statements the store runs
    for a foreign id and for a missing id are identical modulo the id."""
    tc, store = env
    b_cid, _ = _globex_campaign(store)
    missing = _missing_id()

    def trace(cid: str, method: str, path: str, **kw) -> list[str]:
        seen: list[str] = []
        store._conn.set_trace_callback(seen.append)
        try:
            tc.request(method, path.format(cid=cid), headers=H("a-admin"), **kw)
        finally:
            store._conn.set_trace_callback(None)
        return [re.sub(r"[0-9]+\.[0-9]+", "<num>", s.replace(cid, "<cid>")) for s in seen]

    for method, path, kw in [
        ("GET", "/api/campaigns/{cid}", {}),
        ("GET", "/api/campaigns/{cid}/analysis", {}),
        ("POST", "/api/campaigns/{cid}/runs", {"json": {"recipes": [_recipe()]}}),
        ("PATCH", "/api/campaigns/{cid}", {"json": {"name": "x"}}),
        ("POST", "/api/campaigns/{cid}/reanalyze", {}),
    ]:
        a, b = trace(b_cid, method, path, **kw), trace(missing, method, path, **kw)
        assert a and a == b, (method, path)


# --- 2. lists ---------------------------------------------------------------- #


def test_list_shows_only_the_callers_tenant(env):
    tc, store = env
    _globex_campaign(store)
    r = tc.post("/api/campaigns", json={"name": A_SECRET}, headers=H("a-write"))
    assert r.status_code == 201
    a_ids = {r.json()["id"]}
    a_list = tc.get("/api/campaigns", headers=H("a-read")).json()["campaigns"]
    assert {c["id"] for c in a_list} == a_ids
    assert B_SECRET not in json.dumps(a_list)
    b_list = tc.get("/api/campaigns?include_archived=true", headers=H("b-admin")).json()["campaigns"]
    assert [c["name"] for c in b_list] == [B_SECRET]
    assert A_SECRET not in json.dumps(b_list)


def test_include_archived(env):
    tc, _ = env
    cid = tc.post("/api/campaigns", json={"name": "old"}, headers=H("a-admin")).json()["id"]
    assert tc.patch(f"/api/campaigns/{cid}", json={"archived": True},
                    headers=H("a-admin")).status_code == 200
    assert tc.get("/api/campaigns", headers=H("a-read")).json()["campaigns"] == []
    listed = tc.get("/api/campaigns", params={"include_archived": "true"},
                    headers=H("a-read")).json()["campaigns"]
    assert [c["id"] for c in listed] == [cid] and listed[0]["archived"] is True


# --- 3. ids are not guessable ------------------------------------------------- #


def test_campaign_ids_are_random_uuid4(env):
    tc, _ = env
    ids = [tc.post("/api/campaigns", json={"name": f"c{i}"}, headers=H("a-write")).json()["id"]
           for i in range(20)]
    assert len(set(ids)) == 20
    for cid in ids:
        assert CID.fullmatch(cid)
        assert uuid.UUID(hex=cid[4:]).version == 4
    # no shared prefix beyond what chance allows (sequential/time-based ids would)
    assert len({cid[4:12] for cid in ids}) == 20


# --- 4. legacy routes act on the default campaign only ------------------------- #


def test_legacy_routes_touch_only_the_callers_default_campaign(env):
    tc, store = env
    b_cid, _ = _globex_campaign(store)
    b_before = _snapshot(store, b_cid)
    # acme: a named campaign X (seeded) and, separately, the legacy default
    x = tc.post("/api/campaigns", json={"name": A_SECRET}, headers=H("a-write")).json()["id"]
    store.seed(_df(8, 1), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=x,
               analysis={"target": "lipase_titer", "marker": "X"}, dataset="x.csv")
    store.seed(_df(8, 2), "lipase_titer", ["Methanol", "pH"], tenant="acme",
               analysis={"target": "lipase_titer", "marker": "DEFAULT"}, dataset="default.csv")
    x_before = store.get(tenant="acme", campaign_id=x)
    default_id = store.default_campaign_id(tenant="acme")
    assert default_id and default_id != x

    r = tc.post("/api/campaign/start", json={"recipes": [_recipe()]}, headers=H("a-write"))
    assert r.status_code == 200
    run_id = r.json()["started"][0]["id"]
    assert tc.post("/api/campaign/result", json={"id": run_id, "value": 3.0},
                   headers=H("a-write")).status_code == 200
    latest = tc.get("/api/latest", headers=H("a-read")).json()
    assert latest["marker"] == "DEFAULT"
    assert tc.get("/api/campaign", headers=H("a-read")).json()["n_measured"] == 1

    assert store.get(tenant="acme", campaign_id=x) == x_before  # X untouched
    assert store.summary(tenant="acme", campaign_id=default_id)["n_measured"] == 1
    assert _snapshot(store, b_cid) == b_before  # globex untouched
    # the default campaign is visible (and flagged) in the new API too
    listed = tc.get("/api/campaigns", headers=H("a-read")).json()["campaigns"]
    assert {c["id"]: c["is_default"] for c in listed} == {x: False, default_id: True}
    # globex's legacy view is its own (it has no default campaign at all)
    assert tc.get("/api/latest", headers=H("b-admin")).json() == {"has_data": False}


# --- 5. analyses and audit rows carry the right tenant (real upload) ----------- #


def _xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


def test_upload_to_a_campaign_stores_tenant_scoped_analysis_and_audit(env):
    tc, store = env
    b_cid, _ = _globex_campaign(store)
    cid = tc.post("/api/campaigns", json={"name": A_SECRET}, headers=H("a-write")).json()["id"]
    book = _xlsx({"Notes": pd.DataFrame({"about": ["synthetic"]}), "Runs": _df(20, 3)})
    r = tc.post(f"/api/campaigns/{cid}/upload", params={"sheet": "Runs"},
                files={"file": ("acme-runs.xlsx", book, "application/octet-stream")},
                headers=H("a-write"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sheet_used"] == "Runs" and body["sheets"] == ["Notes", "Runs"]

    got = tc.get(f"/api/campaigns/{cid}/analysis", headers=H("a-read")).json()
    assert got["has_data"] is True and got["dataset"] == "acme-runs.xlsx"
    detail = tc.get(f"/api/campaigns/{cid}", headers=H("a-read")).json()
    assert detail["summary"]["has_campaign"] is True
    assert detail["summary"]["analysis_in_sync"] is True
    # the upload did not land in the legacy default, nor anywhere in globex
    assert store.default_campaign_id(tenant="acme") is None
    assert tc.get("/api/latest", headers=H("a-read")).json() == {"has_data": False}
    assert store.load_analysis(tenant="globex", campaign_id=b_cid)["marker"] == B_SECRET

    rows = store._conn.execute(
        "SELECT campaign_id, tenant FROM analyses WHERE campaign_id IN (?, ?) AND tenant IS NOT ?",
        (cid, b_cid, None),
    ).fetchall()
    assert sorted(rows) == sorted([(cid, "acme"), (b_cid, "globex")])

    audit = store.audit_entries(tenant="acme")
    actions = {(e["action"], e["outcome"]) for e in audit}
    assert {("campaign.create", "ok"), ("campaign.upload", "ok")} <= actions
    for e in audit:
        assert e["tenant"] == "acme" and e["subject"] == "bob@acme"
        assert e["auth_method"] == "bearer"
        assert e["campaign_id"] == cid
        # no data payloads: never the filename, a sheet, a column, or a value
        assert e["detail"] is None or re.fullmatch(r"runs=\d+", e["detail"])
        assert "acme-runs" not in json.dumps(e) and "Methanol" not in json.dumps(e)
    assert all(e["subject"] != "bob@acme" for e in store.audit_entries(tenant="globex"))


def test_every_mutation_is_audited_with_subject_and_auth_method(env):
    tc, store = env
    cid = tc.post("/api/campaigns", json={"name": "a"}, headers=H("a-admin")).json()["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    tc.patch(f"/api/campaigns/{cid}", json={"name": "b"}, headers=H("a-admin"))
    run = tc.post(f"/api/campaigns/{cid}/runs", json={"recipes": [_recipe()]},
                  headers=H("a-admin")).json()["started"][0]
    tc.post(f"/api/campaigns/{cid}/runs/{run['id']}/result", json={"value": 1.0},
            headers=H("a-admin"))
    busy._slot.acquire()  # keep the reanalyze cheap: it answers 503 without fitting
    try:
        tc.post(f"/api/campaigns/{cid}/reanalyze", headers=H("a-admin"))
    finally:
        busy._slot.release()
    tc.patch(f"/api/campaigns/{cid}", json={"archived": True}, headers=H("a-admin"))
    tc.patch(f"/api/campaigns/{cid}", json={"archived": False}, headers=H("a-admin"))
    got = [(e["action"], e["outcome"]) for e in reversed(store.audit_entries(tenant="acme"))]
    assert got == [
        ("campaign.create", "ok"), ("campaign.rename", "ok"), ("campaign.start", "ok"),
        ("campaign.result", "ok"), ("campaign.reanalyze", "http_503"),
        ("campaign.archive", "ok"), ("campaign.unarchive", "ok"),
    ]
    for e in store.audit_entries(tenant="acme"):
        assert (e["tenant"], e["subject"], e["auth_method"], e["campaign_id"]) == (
            "acme", "alice@acme", "bearer", cid)


def test_a_probe_of_a_foreign_id_is_audited_without_the_id(env):
    tc, store = env
    b_cid, _ = _globex_campaign(store)
    tc.post(f"/api/campaigns/{b_cid}/runs", json={"recipes": [_recipe()]}, headers=H("a-write"))
    (entry,) = store.audit_entries(tenant="acme")
    assert entry["outcome"] == "http_404" and entry["campaign_id"] is None
    assert all(e["action"] != "campaign.start" for e in store.audit_entries(tenant="globex"))


# --- 6. odd ids ------------------------------------------------------------------ #

_ODD_IDS = [
    "..", "..%2f..%2fetc%2fpasswd", "%2e%2e%2fZZPROBE", "ZZPROBE%00", "cmp_ZZPROBE",
    "cmp_" + "A" * 32, "cmp_" + "0" * 31, "cmp_" + "0" * 33, "cmp_%C3%BC" + "0" * 30,
    "%E2%80%AEZZPROBE", "ZZPROBE" * 2000, "cmp_" + "0" * 32 + "%0a", " cmp_" + "0" * 32,
    "cmp_" + "0" * 32 + "%2f..",
]


@pytest.mark.parametrize("odd", _ODD_IDS, ids=lambda s: s[:24])
def test_odd_ids_fail_safely_without_echo(env, odd):
    tc, store = env
    _globex_campaign(store)
    for method, path, kw in _route_matrix("0" * 32):
        url = path.replace("{cid}", odd)
        r = tc.request(method, url, headers=H("a-admin"), **kw)
        assert r.status_code in (400, 404, 405, 422), (method, url[:80], r.status_code)
        assert "ZZPROBE" not in r.text and "passwd" not in r.text
        assert B_SECRET not in r.text
    assert store.list_campaigns(tenant="acme", include_archived=True) == []


def test_odd_run_ids_fail_safely(env):
    tc, store = env
    cid = tc.post("/api/campaigns", json={"name": "a"}, headers=H("a-write")).json()["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    for run_id in ["ZZPROBE", "0" * 31, "0" * 32, "F" * 32, "..%2fZZPROBE"]:
        r = tc.post(f"/api/campaigns/{cid}/runs/{run_id}/result", json={"value": 1.0},
                    headers=H("a-write"))
        assert r.status_code == 404 and "ZZPROBE" not in r.text
    # a real run id of ANOTHER tenant's campaign is just "run not found" here
    _, b_run = _globex_campaign(store)
    r = tc.post(f"/api/campaigns/{cid}/runs/{b_run}/result", json={"value": 1.0},
                headers=H("a-write"))
    assert r.status_code == 404 and r.json() == {"error": "run not found"}
    assert b_run not in r.text
    # a non-finite value is refused before anything is looked up
    r = tc.post(f"/api/campaigns/{cid}/runs/{'0' * 32}/result",
                content=b'{"value": NaN}', headers={**H("a-write"), "content-type": "application/json"})
    assert r.status_code == 400


# --- 7. scopes ------------------------------------------------------------------- #


def test_scopes_read_write_admin(env):
    tc, store = env
    # no token at all
    assert tc.get("/api/campaigns").status_code == 401
    assert tc.post("/api/campaigns", json={"name": "x"}).status_code == 401
    # read cannot write
    assert tc.post("/api/campaigns", json={"name": "x"}, headers=H("a-read")).status_code == 403
    cid = tc.post("/api/campaigns", json={"name": "x"}, headers=H("a-write")).json()["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    for method, path, kw in _route_matrix("0" * 32):
        if method == "GET":
            assert tc.request(method, path.format(cid=cid), headers=H("a-read"), **kw).status_code == 200
        else:
            r = tc.request(method, path.format(cid=cid), headers=H("a-read"), **kw)
            assert r.status_code == 403, (method, path)
    # write can rename but not archive/unarchive
    assert tc.patch(f"/api/campaigns/{cid}", json={"name": "y"},
                    headers=H("a-write")).status_code == 200
    for archived in (True, False):
        r = tc.patch(f"/api/campaigns/{cid}", json={"archived": archived}, headers=H("a-write"))
        assert r.status_code == 403 and r.json() == {"detail": "requires 'admin' scope"}
    # ... and the 403 is decided before the id is looked at (no existence oracle)
    for target in (_missing_id(), "cmp_bad"):
        assert tc.patch(f"/api/campaigns/{target}", json={"archived": True},
                        headers=H("a-write")).status_code == 403
    assert store.campaign_info(tenant="acme", campaign_id=cid)["archived"] is False
    # admin can
    r = tc.patch(f"/api/campaigns/{cid}", json={"archived": True}, headers=H("a-admin"))
    assert r.status_code == 200 and r.json()["archived"] is True


def test_archived_campaign_refuses_every_write_with_409(env):
    tc, store = env
    cid = tc.post("/api/campaigns", json={"name": "x"}, headers=H("a-admin")).json()["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    run = store.start([_recipe()], tenant="acme", campaign_id=cid)[0]
    store.set_result(run["id"], 2.0, tenant="acme", campaign_id=cid)
    assert tc.patch(f"/api/campaigns/{cid}", json={"archived": True},
                    headers=H("a-admin")).status_code == 200
    before = store.get(tenant="acme", campaign_id=cid)
    writes = [
        ("PATCH", f"/api/campaigns/{cid}", {"json": {"name": "renamed"}}),
        ("POST", f"/api/campaigns/{cid}/upload",
         {"files": {"file": ("r.csv", _csv(_df(20)), "text/csv")}}),
        ("POST", f"/api/campaigns/{cid}/runs", {"json": {"recipes": [_recipe()]}}),
        ("POST", f"/api/campaigns/{cid}/runs/{run['id']}/result", {"json": {"value": 1.0}}),
        ("POST", f"/api/campaigns/{cid}/reanalyze", {}),
    ]
    for method, url, kw in writes:
        r = tc.request(method, url, headers=H("a-admin"), **kw)
        assert r.status_code == 409, (method, url, r.text)
    assert store.get(tenant="acme", campaign_id=cid) == before
    # reads still work; unarchive + rename in one request
    assert tc.get(f"/api/campaigns/{cid}", headers=H("a-read")).status_code == 200
    r = tc.patch(f"/api/campaigns/{cid}", json={"archived": False, "name": "back"},
                 headers=H("a-admin"))
    assert r.status_code == 200 and (r.json()["archived"], r.json()["name"]) == (False, "back")


@pytest.mark.parametrize("name", [
    "", "   ", "x" * (MAX_NAME_CHARS + 1), "bad\x00name", "line\nbreak", "tab\there",
    "sep\u2028x", "del\x7f", 123, None, ["x"],
])
def test_create_rejects_bad_names_without_echo(env, name):
    tc, store = env
    r = tc.post("/api/campaigns", json={"name": name}, headers=H("a-write"))
    assert r.status_code == 400, r.text
    if isinstance(name, str) and name.strip():
        assert name not in r.text
    assert store.list_campaigns(tenant="acme") == []


def test_create_accepts_good_names_and_rejects_extra_fields(env):
    tc, _ = env
    for good in ["a", "x" * MAX_NAME_CHARS, "Lipase \u00fcber-run \U0001f9ea", "  padded  "]:
        r = tc.post("/api/campaigns", json={"name": good}, headers=H("a-write"))
        assert r.status_code == 201 and r.json()["name"] == good.strip()
    r = tc.post("/api/campaigns", json={"name": "x", "tenant": "globex"}, headers=H("a-write"))
    assert r.status_code == 400  # the tenant can never come from the body
    assert tc.patch(f"/api/campaigns/{_missing_id()}", json={},
                    headers=H("a-write")).status_code == 400


def test_tenant_hints_in_the_request_are_ignored(env):
    tc, store = env
    _globex_campaign(store)
    r = tc.get("/api/campaigns?tenant=globex", headers={**H("a-read"), "X-Tenant": "globex"})
    assert r.status_code == 200 and r.json() == {"campaigns": []}


# --- 8. production mode (subprocess: the posture is decided at import) --------- #


def test_production_mode_serves_the_api_with_the_same_isolation(tmp_path):
    script = """
        import json, uuid
        from fastapi.testclient import TestClient
        from kalos.portal.app import app
        A = {"Authorization": "Bearer a-admin"}
        B = {"Authorization": "Bearer b-admin"}
        with TestClient(app) as c:
            out = {}
            out["anon"] = c.get("/api/campaigns").status_code
            r = c.post("/api/campaigns", json={"name": "zeus"}, headers=B)
            out["create"] = r.status_code
            bid = r.json()["id"]
            f = c.get(f"/api/campaigns/{bid}", headers=A)
            m = c.get("/api/campaigns/cmp_" + uuid.uuid4().hex, headers=A)
            out["same404"] = [f.status_code, f.content.decode(), sorted(f.headers.items())] == \\
                             [m.status_code, m.content.decode(), sorted(m.headers.items())]
            out["foreign"] = f.status_code
            out["a_list"] = c.get("/api/campaigns", headers=A).json()
            out["docs"] = c.get("/openapi.json").status_code
        print(json.dumps(out))
    """
    env = {k: v for k, v in os.environ.items()
           if k not in {"KALOS_AUTH_TOKENS", "KALOS_AUTH_TOKENS_FILE",
                        "KALOS_ALLOW_OPEN_ACCESS", "KALOS_ENV"}}
    env.update(KALOS_STATE_DIR=str(tmp_path), KALOS_ENV="production",
               KALOS_AUTH_TOKENS=json.dumps(TOKENS))
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"anon": 401, "create": 201, "same404": True, "foreign": 404,
                   "a_list": {"campaigns": []}, "docs": 404}


# --- 9. reanalyze never crosses tenants (real fit) ------------------------------- #


def test_one_tenants_reanalyze_never_appears_in_the_others_analysis(env):
    tc, store = env
    b_cid, _ = _globex_campaign(store)
    b_before = _snapshot(store, b_cid)
    a_cid = tc.post("/api/campaigns", json={"name": A_SECRET}, headers=H("a-write")).json()["id"]
    store.seed(_df(8, 1), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=a_cid)
    run = tc.post(f"/api/campaigns/{a_cid}/runs", json={"recipes": [_recipe(2.5, 6.2)]},
                  headers=H("a-write")).json()["started"][0]
    assert tc.post(f"/api/campaigns/{a_cid}/runs/{run['id']}/result", json={"value": 4.2},
                   headers=H("a-write")).status_code == 200

    r = tc.post(f"/api/campaigns/{a_cid}/reanalyze", headers=H("a-write"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["campaign"]["round"] == 1 and body["campaign"]["n_base"] == 9
    assert body["campaign"]["analysis_in_sync"] is True
    assert body["analysis"]["dataset"] == "campaign round 1"

    # acme's new analysis is acme's alone
    assert tc.get(f"/api/campaigns/{a_cid}/analysis", headers=H("a-read")).json()["dataset"] \
        == "campaign round 1"
    assert _snapshot(store, b_cid) == b_before
    b_view = tc.get(f"/api/campaigns/{b_cid}/analysis", headers=H("b-admin")).json()
    assert b_view["marker"] == B_SECRET and b_view["dataset"] == B_DATASET
    assert A_SECRET not in json.dumps(b_view)
    # neither tenant's legacy /api/latest picked it up
    assert tc.get("/api/latest", headers=H("a-read")).json() == {"has_data": False}
    assert tc.get("/api/latest", headers=H("b-admin")).json() == {"has_data": False}


# --- 10. concurrent creation --------------------------------------------------- #


def test_concurrent_creation_by_two_tenants(env):
    _, store = env
    errors: list[BaseException] = []
    made: dict[str, list[str]] = {"acme": [], "globex": []}
    barrier = threading.Barrier(2)

    def worker(token: str, tenant: str) -> None:
        try:
            c = TestClient(app)
            barrier.wait()
            for i in range(15):
                r = c.post("/api/campaigns", json={"name": f"{tenant}-{i}"}, headers=H(token))
                assert r.status_code == 201, r.text
                made[tenant].append(r.json()["id"])
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=("a-write", "acme")),
               threading.Thread(target=worker, args=("b-admin", "globex"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    for tenant in ("acme", "globex"):
        listed = store.list_campaigns(tenant=tenant)
        assert {c["id"] for c in listed} == set(made[tenant])
        assert all(c["name"].startswith(tenant + "-") for c in listed)
    assert not set(made["acme"]) & set(made["globex"])


# --- 11. limits still apply -------------------------------------------------------- #


def test_size_limits_and_the_analysis_slot_still_apply(env):
    tc, store = env
    cid = tc.post("/api/campaigns", json={"name": "a"}, headers=H("a-write")).json()["id"]
    store.seed(_df(), "lipase_titer", ["Methanol", "pH"], tenant="acme", campaign_id=cid)
    # JSON routes keep the small default cap
    big = json.dumps({"name": "x" * (1024 * 1024 + 10)})
    r = tc.post("/api/campaigns", content=big,
                headers={**H("a-write"), "content-type": "application/json"})
    assert r.status_code == 413
    # the campaign upload route has the upload cap: 2 MiB is past the JSON cap but
    # reaches the route (rejected there as not a table), while > upload cap is 413
    r = tc.post(f"/api/campaigns/{cid}/upload",
                files={"file": ("x.csv", b"\x00" * (2 * 1024 * 1024), "text/csv")},
                headers=H("a-write"))
    assert r.status_code == 400, r.status_code
    assert "code" in r.json()
    r = tc.post(f"/api/campaigns/{cid}/upload",
                files={"file": ("x.csv", b"0" * (MAX_UPLOAD_BYTES + 2 * 1024 * 1024), "text/csv")},
                headers=H("a-write"))
    assert r.status_code == 413
    # start bounds identical to /api/campaign/start (max 64 recipes)
    r = tc.post(f"/api/campaigns/{cid}/runs", json={"recipes": [_recipe()] * 65},
                headers=H("a-write"))
    assert r.status_code == 400
    # the single analysis slot: busy -> 503 + Retry-After on upload and reanalyze
    run = store.start([_recipe()], tenant="acme", campaign_id=cid)[0]
    store.set_result(run["id"], 2.0, tenant="acme", campaign_id=cid)
    busy._slot.acquire()
    try:
        r1 = tc.post(f"/api/campaigns/{cid}/upload",
                     files={"file": ("r.csv", _csv(_df(20)), "text/csv")}, headers=H("a-write"))
        r2 = tc.post(f"/api/campaigns/{cid}/reanalyze", headers=H("a-write"))
    finally:
        busy._slot.release()
    for r in (r1, r2):
        assert r.status_code == 503 and r.headers["Retry-After"]


def test_upload_size_cap_only_widens_for_well_formed_campaign_upload_paths():
    mw = CampaignUploadSizeLimit(app, default_cap=10, upload_cap=1000,
                                 upload_paths={"/api/run"})
    cid = "cmp_" + "a" * 32
    assert mw.cap_for(f"/api/campaigns/{cid}/upload") == 1000
    assert mw.cap_for(f"/api/campaigns/{cid}/upload/") == 1000
    assert mw.cap_for("/api/run") == 1000
    for path in ["/api/campaigns/cmp_x/upload", f"/api/campaigns/{cid}/runs",
                 f"/api/campaigns/{cid}/upload/x", "/api/campaigns", f"/x/api/campaigns/{cid}/upload"]:
        assert mw.cap_for(path) == 10, path


def test_upload_error_contract_matches_legacy(env):
    tc, _ = env
    cid = tc.post("/api/campaigns", json={"name": "a"}, headers=H("a-write")).json()["id"]
    book = _xlsx({"Notes": pd.DataFrame({"about": ["synthetic"]}), "Runs": _df(20)})
    cases = [
        ({"file": ("empty.csv", b"", "text/csv")}, {}),
        ({"file": ("notes.txt", b"just some prose\nnothing tabular", "text/plain")}, {}),
        ({"file": ("tiny.csv", _csv(_df(4)), "text/csv")}, {}),
        ({"file": ("bad.xlsx", b"PK\x03\x04garbage", "application/octet-stream")}, {}),
        ({"file": ("b.xlsx", book, "application/octet-stream")}, {"sheet": "Missing"}),
    ]
    for files, data in cases:
        legacy = tc.post("/api/run", files=files, data=data, headers=H("a-write"))
        scoped = tc.post(f"/api/campaigns/{cid}/upload", files=files, data=data,
                         headers=H("a-write"))
        assert legacy.status_code == scoped.status_code == 400
        assert legacy.json() == scoped.json()
        assert "code" in scoped.json()


# --- 12. no foreign names in bodies or logs --------------------------------------- #


def test_no_foreign_names_in_bodies_or_logs(env, caplog):
    tc, store = env
    b_cid, b_run = _globex_campaign(store)
    texts: list[str] = []
    with caplog.at_level("DEBUG"):
        for token in ("a-admin", "a-write", "a-read"):
            for method, path, kw in _route_matrix(b_run):
                texts.append(tc.request(method, path.format(cid=b_cid), headers=H(token), **kw).text)
            texts.append(tc.get("/api/campaigns?include_archived=true", headers=H(token)).text)
            texts.append(tc.get("/api/latest", headers=H(token)).text)
            texts.append(tc.get("/api/campaign", headers=H(token)).text)
        # acme's own activity, which logs (e.g. a rejected start)
        cid = tc.post("/api/campaigns", json={"name": A_SECRET}, headers=H("a-admin")).json()["id"]
        texts.append(tc.post(f"/api/campaigns/{cid}/runs", json={"recipes": [_recipe()]},
                             headers=H("a-admin")).text)
        texts.append(tc.post("/api/campaign/start", json={"recipes": [_recipe()]},
                             headers=H("a-admin")).text)
        texts.append(tc.post(f"/api/campaigns/{cid}/upload",
                             files={"file": ("t.csv", _csv(_df(4)), "text/csv")},
                             headers=H("a-admin")).text)
    blob = "\n".join(texts)
    # Server-side log lines only: the test client's own httpx request log
    # naturally carries the URL the attacker typed.
    server_log = "\n".join(
        r.getMessage() + (r.exc_text or "") for r in caplog.records
        if not r.name.startswith(("httpx", "asyncio"))
    )
    assert any(r.name.startswith("kalos") for r in caplog.records)  # logs were captured
    for secret in (B_SECRET, B_DATASET, b_cid, b_run, "gina@globex"):
        assert secret not in blob, secret
        assert secret not in server_log, secret
    # and acme's activity never logs acme's own campaign name either
    assert A_SECRET not in server_log
    assert all(e["tenant"] == "acme" for e in store.audit_entries(tenant="acme"))
