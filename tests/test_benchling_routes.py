"""Routes under /api/integrations/benchling: feature flag, scopes, tenant
isolation, connect (secret sealed, never returned), import through the regular
upload path, draft-only proposal pushes, results only with a delegated token,
webhook signature checks, and the audit rows each push leaves."""
from __future__ import annotations

import hashlib
import json
import logging

import pytest

pytest.importorskip("fastapi")
from benchling_fake import (
    BASE,
    RESULT_SCHEMA,
    STUDY_ID,
    TITER_FIELD,
    FakeBenchling,
    bioprocess_study,
)
from fastapi.testclient import TestClient
from test_benchling_sync import APP_DEF, make_key, payload, sign

from kalos.integrations.benchling import routes as bl_routes
from kalos.integrations.benchling import secrets
from kalos.integrations.benchling.client import reset_buckets
from kalos.integrations.benchling.store import integration_store_at
from kalos.integrations.benchling.sync import JwksCache
from kalos.portal import app as portal_module
from kalos.portal import busy
from kalos.portal import campaign as campaign_module
from kalos.portal.app import app
from kalos.portal.campaign import CampaignStore, get_campaign_store

P = "/api/integrations/benchling"
APP_SECRET = "app-client-secret-SHOULD-NOT-LEAK"  # noqa: S105 - test value, matches the fake


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


TOKENS = [
    {"token_sha256": _sha("a-admin"), "subject": "alice@acme", "tenant": "acme", "scopes": ["read", "write", "admin"]},
    {"token_sha256": _sha("a-write"), "subject": "bob@acme", "tenant": "acme", "scopes": ["read", "write"]},
    {"token_sha256": _sha("a-read"), "subject": "carol@acme", "tenant": "acme", "scopes": ["read"]},
    {"token_sha256": _sha("b-admin"), "subject": "gina@globex", "tenant": "globex",
     "scopes": ["read", "write", "admin"]},
]


def H(token: str, **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", **extra}


def _stub_analyze(df, target, *, profile=None, anonymize=False, progress=None, roles=None, pending=None, **_):
    inputs = [c for c in df.columns if c not in (target, "run_id")]
    first = df.iloc[0]
    recipes = [{"recipe": {c: (first[c].item() if hasattr(first[c], "item") else first[c]) for c in inputs},
                "pred": 2.0 + i, "std": 0.2, "mode": "explore", "reason": "stub"} for i in range(3)]
    return {"target": target, "proposal_features": inputs[:4], "d": len(inputs), "n": len(df),
            "proposals": recipes, "marker": "stub-result", "seed": 1,
            "promotion": {"status": "stub"}, "proposal_gating": {"gated": False}, "validation": {"status": "pass"}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = bioprocess_study(FakeBenchling(), n_runs=2, conditions_per_run=4, replicates=2)
    key, jwk = make_key()
    store = CampaignStore(tmp_path)
    monkeypatch.setenv("KALOS_AUTH_TOKENS", json.dumps(TOKENS))
    monkeypatch.delenv("KALOS_AUTH_TOKENS_FILE", raising=False)
    monkeypatch.setenv("KALOS_INTEGRATIONS", "benchling")
    assert portal_module.mount_integrations()  # the app was imported with the flag unset
    monkeypatch.setenv(secrets.SECRET_KEY_VAR, "s" * 48)
    monkeypatch.setattr(portal_module, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(portal_module, "_LATEST_DIR", tmp_path / "latest")
    monkeypatch.setattr(campaign_module, "_STORE", store)
    monkeypatch.setattr(portal_module, "_analyze", _stub_analyze)
    busy.reset_limiter(total=2, per_tenant=1)
    reset_buckets()
    bl_routes.reset_token_cache()
    app.dependency_overrides[get_campaign_store] = lambda: store
    app.dependency_overrides[bl_routes.get_benchling_transport] = fake.transport
    app.dependency_overrides[bl_routes.get_jwks_cache] = lambda: JwksCache(lambda _a: {"keys": [jwk]})
    try:
        yield TestClient(app), store, fake, key
    finally:
        app.dependency_overrides.clear()
        busy.reset_limiter()
        reset_buckets()
        bl_routes.reset_token_cache()


def _connect(tc, token="a-admin", **over):
    body = {"base_url": BASE, "client_id": "app-client-id", "client_secret": APP_SECRET,
            "app_definition_id": APP_DEF, **over}
    return tc.post(f"{P}/connect", json=body, headers=H(token))


def _import(tc, token="a-write", **over):
    body = {"study_id": STUDY_ID, "target_field": TITER_FIELD, "result_schema_id": RESULT_SCHEMA, **over}
    return tc.post(f"{P}/import", json=body, headers=H(token))


# --- feature flag ------------------------------------------------------------------- #


def test_flag_off_every_route_is_the_unknown_path_404(env, monkeypatch):
    tc, *_ = env
    monkeypatch.delenv("KALOS_INTEGRATIONS")
    unknown = tc.get("/api/no-such-route", headers=H("a-admin"))
    for method, path, kw in [
        ("POST", "/connect", {"json": {}}), ("GET", "/status", {}), ("GET", "/studies", {}),
        ("POST", "/import", {"json": {}}), ("POST", "/push-proposals", {"json": {}}),
        ("POST", "/push-results", {"content": b"not json"}), ("POST", "/poll", {"json": {}}),
        ("POST", "/webhook/acme", {"content": b"{}"}), ("GET", "", {}),
    ]:
        for headers in (H("a-admin"), {}):
            r = tc.request(method, P + path, headers=headers, **kw)
            assert r.status_code == 404, (method, path)
            assert r.content == unknown.content, (method, path)
            assert r.headers["content-type"] == unknown.headers["content-type"]
    monkeypatch.setenv("KALOS_INTEGRATIONS", "something-else")
    assert tc.get(f"{P}/status", headers=H("a-read")).status_code == 404


def test_flag_on_status_needs_auth(env):
    tc, *_ = env
    assert tc.get(f"{P}/status").status_code == 401
    assert tc.get(f"{P}/status", headers=H("a-read")).json() == {
        "enabled": True, "connected": False, "secret_key_configured": True}


# --- connect / status ------------------------------------------------------------------ #


def test_connect_needs_admin_and_never_returns_or_logs_the_secret(env, caplog):
    tc, store, fake, _ = env
    caplog.set_level(logging.DEBUG)
    assert _connect(tc, token="a-write").status_code == 403
    r = _connect(tc)
    assert r.status_code == 200, r.text
    assert APP_SECRET not in r.text
    status = tc.get(f"{P}/status", headers=H("a-read"))
    assert status.json()["connected"] is True and status.json()["credentials_readable"] is True
    assert APP_SECRET not in status.text
    assert APP_SECRET not in caplog.text
    istore = integration_store_at(store.state_dir)
    assert APP_SECRET.encode() not in istore.path.read_bytes()
    audits = store.audit_entries(tenant="acme")
    assert audits[0]["action"] == "benchling.connect" and APP_SECRET not in json.dumps(audits)
    assert [c.path for c in fake.calls] == ["/oauth/token"]  # verified once


def test_connect_with_refused_credentials_stores_nothing(env):
    tc, store, _, _ = env
    r = _connect(tc, client_secret="wrong-secret")
    assert r.status_code == 502 and r.json()["code"] == "benchling_auth_failed"
    assert "wrong-secret" not in r.text
    assert integration_store_at(store.state_dir).connection(tenant="acme") is None


def test_connect_refuses_non_benchling_hosts_and_missing_key(env, monkeypatch):
    tc, *_ = env
    r = _connect(tc, base_url="https://evil.example")
    assert r.status_code == 400 and r.json()["code"] == "benchling_bad_request"
    monkeypatch.setenv(secrets.SECRET_KEY_VAR, "")
    assert _connect(tc).json()["code"] == "secret_key_not_configured"


def test_studies_lists_v3_studies_and_v2_projects(env):
    tc, *_ = env
    _connect(tc)
    body = tc.get(f"{P}/studies", headers=H("a-read")).json()
    assert [s["id"] for s in body["studies"]] == [STUDY_ID, "stdy_other0001"]
    assert body["projects"][0]["id"] == "src_YzU5p4dR"


# --- import ----------------------------------------------------------------------------- #


def test_import_goes_through_the_upload_path_and_records_provenance(env):
    tc, store, _, _ = env
    _connect(tc)
    assert _import(tc, token="a-read").status_code == 403
    r = _import(tc)
    assert r.status_code == 200, r.text
    body = r.json()
    cid = body["campaign_id"]
    assert body["rows"] == 16 and body["target_column"] == "Titer [g/L]"
    assert body["filename"].startswith(f"benchling-{STUDY_ID}-") and body["filename"].endswith("Z.csv")
    assert {p["benchling_id"] for p in body["provenance"]} >= {"schver_Rm4nTvWz", TITER_FIELD}
    assert body["report"]["rows_kept"] == 16
    stored = store.load_analysis(tenant="acme", campaign_id=cid)
    assert stored["run"]["filename"] == body["filename"]  # the upload record names study + fetch time
    assert stored["marker"] == "stub-result"
    actions = [a["action"] for a in store.audit_entries(tenant="acme")]
    assert "benchling.import" in actions and "campaign.upload" not in actions
    assert integration_store_at(store.state_dir).latest_import(tenant="acme", campaign_id=cid)["source_id"] == STUDY_ID


def test_import_errors_from_the_upload_checks_pass_through(env):
    tc, store, _, _ = env
    _connect(tc)
    cid = _import(tc).json()["campaign_id"]
    store.start([{"recipe": {"Feed rate [mL/h]": 2.0}}], tenant="acme", campaign_id=cid)
    again = _import(tc, campaign_id=cid)
    assert again.status_code == 409 and again.json()["analysis"]["code"] == "campaign_has_runs"


def test_import_body_must_name_one_source(env):
    tc, *_ = env
    _connect(tc)
    r = _import(tc, assay_run_schema_id="assaysch_run0001")
    assert r.status_code == 400 and r.json()["code"] == "invalid_body"
    r = tc.post(f"{P}/import", json={"study_id": "../x", "target_field": TITER_FIELD}, headers=H("a-write"))
    assert r.status_code == 422


# --- pushes ------------------------------------------------------------------------------ #


def test_push_proposals_is_a_draft_with_an_audit_row(env):
    tc, store, fake, _ = env
    _connect(tc)
    cid = _import(tc).json()["campaign_id"]
    r = tc.post(f"{P}/push-proposals", json={"campaign_id": cid, "study_id": STUDY_ID}, headers=H("a-write"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["promoted"] is False and len(body["condition_ids"]) == 3
    assert body["benchling_identity"] == "app"
    assert not any("promote" in c.path for c in fake.calls)
    audit = integration_store_at(store.state_dir).push_audit(tenant="acme")[0]
    assert audit["action"] == "benchling.push_proposals" and audit["subject"] == "bob@acme"
    assert audit["benchling_ids"]["run_plan_id"] == body["run_plan_id"]
    assert audit["benchling_ids"]["condition_ids"] == body["condition_ids"]
    main = store.audit_entries(tenant="acme")[0]
    assert main["action"] == "benchling.push_proposals" and f"push_audit_id={audit['id']}" in main["detail"]


def test_push_proposals_to_another_study_is_refused(env):
    tc, *_ = env
    _connect(tc)
    cid = _import(tc).json()["campaign_id"]
    r = tc.post(f"{P}/push-proposals", json={"campaign_id": cid, "study_id": "stdy_other0001"}, headers=H("a-write"))
    assert r.status_code == 409 and r.json()["code"] == "benchling_study_mismatch"


def _measured_campaign(tc, store) -> str:
    cid = _import(tc).json()["campaign_id"]
    runs = store.start([{"recipe": {"Feed rate [mL/h]": 2.0}}, {"recipe": {"Feed rate [mL/h]": 3.0}}],
                       tenant="acme", campaign_id=cid)
    for i, run in enumerate(runs):
        store.set_result(run["id"], 1.5 + i, tenant="acme", campaign_id=cid, actor="bob@acme")
    return cid


def test_push_results_refuses_without_a_delegated_token(env):
    tc, store, fake, _ = env
    _connect(tc)
    cid = _measured_campaign(tc, store)
    before = len(fake.calls)
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=H("a-write"))
    assert r.status_code == 400 and r.json()["code"] == "delegated_token_required"
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid},
                headers=H("a-write", **{"X-Benchling-Delegated-Token": "   "}))
    assert r.status_code == 400
    assert len(fake.calls) == before  # no Benchling call at all
    assert fake.committed_results == []


def test_push_results_writes_under_the_user_and_is_idempotent(env, caplog):
    tc, store, fake, _ = env
    caplog.set_level(logging.DEBUG)
    _connect(tc)
    cid = _measured_campaign(tc, store)
    fake.delegated_tokens["user-token-bob"] = "ent_bob"
    hdr = H("a-write", **{"X-Benchling-Delegated-Token": "user-token-bob"})
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid, "report_entry_id": "etr_Hds1XAaq",
                                           "report_field": "kalos_report"}, headers=hdr)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["results_written"] == 2 and body["benchling_identity"] == "delegated"
    assert [x["creator"]["id"] for x in fake.committed_results] == ["ent_bob", "ent_bob"]
    assert {x["fields"]["titer"]["value"] for x in fake.committed_results} == {1.5, 2.5}
    assert fake.entry_patches[0]["body"]["fields"]["kalos_report"]["value"] == body["report_blob_id"]
    writes = [c for c in fake.calls if c.method in ("POST", "PATCH") and c.path != "/oauth/token"]
    assert all(c.headers["authorization"] == "Bearer user-token-bob" for c in writes)
    audit = integration_store_at(store.state_dir).push_audit(tenant="acme")[0]
    assert audit["benchling_identity"] == "delegated" and audit["subject"] == "bob@acme"
    assert audit["benchling_ids"]["result_ids"] == body["result_ids"]
    assert "user-token-bob" not in caplog.text and "user-token-bob" not in json.dumps(audit)
    again = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr)
    assert again.status_code == 409 and again.json()["code"] == "nothing_to_push"
    assert len(fake.committed_results) == 2


def test_rejected_delegated_token_is_reported_and_audited(env):
    tc, store, _fake, _ = env
    _connect(tc)
    cid = _measured_campaign(tc, store)
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid},
                headers=H("a-write", **{"X-Benchling-Delegated-Token": "expired-token"}))
    assert r.status_code == 502 and r.json()["code"] == "benchling_auth_failed"
    assert "delegated" in r.json()["error"]
    audit = integration_store_at(store.state_dir).push_audit(tenant="acme")[0]
    assert audit["outcome"] == "http_502" and audit["benchling_identity"] == "delegated"


# --- tenant isolation ----------------------------------------------------------------------- #


def test_tenants_are_isolated(env):
    tc, store, _fake, key = env
    _connect(tc)
    cid = _import(tc).json()["campaign_id"]
    # globex sees no connection and cannot use acme's
    assert tc.get(f"{P}/status", headers=H("b-admin")).json()["connected"] is False
    assert tc.get(f"{P}/studies", headers=H("b-admin")).json()["code"] == "benchling_not_connected"
    r = tc.post(f"{P}/import", json={"study_id": STUDY_ID, "target_field": TITER_FIELD}, headers=H("b-admin"))
    assert r.json()["code"] == "benchling_not_connected"
    # acme's campaign id is not found for globex, byte-identical to a missing one
    foreign = tc.post(f"{P}/push-proposals", json={"campaign_id": cid, "study_id": STUDY_ID}, headers=H("b-admin"))
    missing = tc.post(f"{P}/push-proposals", json={"campaign_id": "cmp_" + "0" * 32, "study_id": STUDY_ID},
                      headers=H("b-admin"))
    assert foreign.status_code == 404 and foreign.content == missing.content
    foreign = tc.post(f"{P}/push-results", json={"campaign_id": cid},
                      headers=H("b-admin", **{"X-Benchling-Delegated-Token": "t"}))
    assert foreign.status_code == 404
    # an acme-signed delivery sent to globex's webhook path is rejected
    body = payload()
    r = tc.post(f"{P}/webhook/globex", content=body, headers=sign(key, body))
    assert r.status_code == 401
    # globex connecting does not change acme's connection
    tc.post(f"{P}/connect", json={"base_url": "https://globex.benchling.com", "client_id": "app-client-id",
                                  "client_secret": APP_SECRET, "verify": False}, headers=H("b-admin"))
    istore = integration_store_at(store.state_dir)
    assert istore.connection(tenant="acme")["base_url"] == BASE
    assert istore.connection(tenant="globex")["base_url"] == "https://globex.benchling.com"
    assert istore.imports(tenant="globex") == [] and istore.push_audit(tenant="globex") == []


# --- webhook ----------------------------------------------------------------------------------- #


def test_webhook_accepts_signed_events_once_and_rejects_tampering(env):
    tc, _store, _, key = env
    _connect(tc)
    body = payload("v3.run.created", resource="run_21f50003")
    headers = sign(key, body, wid="msg_42")
    first = tc.post(f"{P}/webhook/acme", content=body, headers=headers)
    assert first.status_code == 200 and first.json()["status"] == "accepted"
    again = tc.post(f"{P}/webhook/acme", content=body, headers=headers)
    assert again.json()["status"] == "duplicate"
    tampered = tc.post(f"{P}/webhook/acme", content=body.replace(b"run_21f50003", b"run_99999999"),
                       headers=headers)
    unknown = tc.post(f"{P}/webhook/nobody", content=body, headers=headers)
    bad_tenant = tc.post(f"{P}/webhook/a%2Fb", content=body, headers=headers)
    assert tampered.status_code == unknown.status_code == 401
    assert tampered.content == unknown.content
    assert bad_tenant.status_code in (401, 404)
    changes = tc.get(f"{P}/status", headers=H("a-read")).json()["changes"]
    assert [(c["event_type"], c["resource_id"]) for c in changes] == [("v3.run.created", "run_21f50003")]


def test_poll_records_study_changes(env):
    tc, *_ = env
    _connect(tc)
    r = tc.post(f"{P}/poll", json={"study_id": STUDY_ID}, headers=H("a-write"))
    assert r.status_code == 200 and r.json()["new_changes"] == 2
    assert tc.post(f"{P}/poll", json={"study_id": STUDY_ID}, headers=H("a-write")).json()["new_changes"] == 0
    assert tc.post(f"{P}/poll", json={"study_id": STUDY_ID}, headers=H("a-read")).status_code == 403


# --- review fixes: concurrent pushes, recording failures ---------------------------------------- #


def test_concurrent_push_results_write_each_result_once(env):
    import threading
    import time as _time

    tc, store, fake, _ = env
    _connect(tc)
    cid = _measured_campaign(tc, store)
    fake.delegated_tokens["user-token-bob"] = "ent_bob"
    orig = fake.handle

    def slow(req):  # widen the window between reading what to push and writing it
        if req.url.path == "/api/v2/result-transactions":
            _time.sleep(0.3)
        return orig(req)

    fake.handle = slow  # type: ignore[method-assign]
    hdr = H("a-write", **{"X-Benchling-Delegated-Token": "user-token-bob"})
    statuses: list[int] = []
    gate = threading.Barrier(2)

    def push():
        gate.wait()
        statuses.append(tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr).status_code)

    threads = [threading.Thread(target=push) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(statuses) == [200, 409]
    assert len(fake.committed_results) == 2


def test_claims_are_released_when_benchling_wrote_nothing(env):
    tc, store, fake, _ = env
    _connect(tc)
    cid = _measured_campaign(tc, store)
    fake.delegated_tokens["user-token-bob"] = "ent_bob"
    hdr = H("a-write", **{"X-Benchling-Delegated-Token": "user-token-bob"})
    fake.script["/api/v2/result-transactions/txn_00000001/results"] = [(500, {}, {"title": "boom"})]
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr)
    assert r.status_code == 502 and r.json()["created"]["aborted"] is True
    again = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr)
    assert again.status_code == 200 and len(fake.committed_results) == 2


def test_recording_failure_after_a_benchling_write_leaves_a_fallback_row(env, monkeypatch):
    tc, store, fake, _ = env
    _connect(tc)
    cid = _measured_campaign(tc, store)
    fake.delegated_tokens["user-token-bob"] = "ent_bob"
    istore = integration_store_at(store.state_dir)
    real = istore.record_push

    def broken(**kw):
        if kw.get("pushed_results"):
            raise RuntimeError("disk full")
        return real(**kw)

    monkeypatch.setattr(istore, "record_push", broken)
    hdr = H("a-write", **{"X-Benchling-Delegated-Token": "user-token-bob"})
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr)
    assert r.status_code == 200
    row = istore.push_audit(tenant="acme")[0]
    assert row["outcome"] == "ok_index_not_recorded"
    assert row["benchling_ids"]["result_ids"] == r.json()["result_ids"]
    # the claims are held: the same results are not sent again
    again = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr)
    assert again.status_code == 409 and len(fake.committed_results) == 2


def test_report_failure_after_committed_results_keeps_them_recorded(env):
    tc, store, fake, _ = env
    _connect(tc)
    cid = _measured_campaign(tc, store)
    fake.delegated_tokens["user-token-bob"] = "ent_bob"
    fake.script["/api/v2/blobs"] = [(500, {}, {"error": {"message": "x"}})]
    hdr = H("a-write", **{"X-Benchling-Delegated-Token": "user-token-bob"})
    r = tc.post(f"{P}/push-results", json={"campaign_id": cid, "report_entry_id": "etr_Hds1XAaq",
                                           "report_field": "kalos_report"}, headers=hdr)
    assert r.status_code == 200 and r.json()["report_attached"] is False
    assert integration_store_at(store.state_dir).push_audit(tenant="acme")[0]["outcome"] == "ok_report_failed"
    again = tc.post(f"{P}/push-results", json={"campaign_id": cid}, headers=hdr)
    assert again.status_code == 409 and len(fake.committed_results) == 2


# --- review fixes: flag off imports nothing; refuse modes ------------------------------------------ #


def test_flag_off_does_not_import_httpx_or_the_connector():
    import os
    import subprocess
    import sys

    code = ("import sys, kalos.portal.app as a; "
            "bad = [m for m in sys.modules if m == 'httpx' or m.startswith('httpx.') "
            "or m.startswith('kalos.integrations.benchling')]; "
            "print(bad, a.mount_integrations())")
    env = {k: v for k, v in os.environ.items() if k != "KALOS_INTEGRATIONS"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[] False"


@pytest.mark.parametrize("flag", ["", "benchling"])
def test_refusals_before_routing_match_unknown_paths(env, monkeypatch, flag):
    """Open mode refuses remote callers with a 503; a connector path must get the
    same bytes as an unknown path, flag on or off."""
    monkeypatch.delenv("KALOS_AUTH_TOKENS")
    monkeypatch.setenv("KALOS_INTEGRATIONS", flag)
    with TestClient(app, client=("203.0.113.9", 51000)) as remote:
        unknown = remote.get("/api/no-such-route")
        for method, path in (("GET", "/status"), ("POST", "/webhook/acme"), ("POST", "/connect")):
            r = remote.request(method, P + path)
            assert r.status_code == unknown.status_code == 503
            assert r.content == unknown.content


def test_connect_without_verify_is_refused_when_another_tenant_uses_the_host(env):
    tc, store, _, _ = env
    assert _connect(tc).status_code == 200  # acme, verified
    r = tc.post(f"{P}/connect", json={"base_url": BASE, "client_id": "x", "client_secret": "y", "verify": False},
                headers=H("b-admin"))
    assert r.status_code == 400 and r.json()["code"] == "verify_required"
    assert integration_store_at(store.state_dir).connection(tenant="globex") is None
    # with verify the credentials are checked, so the host may be shared
    assert _connect(tc, token="b-admin").status_code == 200


def test_status_reports_the_tenant_own_budget(env):
    tc, *_ = env
    _connect(tc)
    tc.get(f"{P}/studies", headers=H("a-read"))
    _connect(tc, token="b-admin")
    acme = tc.get(f"{P}/status", headers=H("a-read")).json()["rate_limit"]["available"]
    globex = tc.get(f"{P}/status", headers=H("b-admin")).json()["rate_limit"]["available"]
    assert globex > acme
