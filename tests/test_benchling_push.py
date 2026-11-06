"""Push: proposals become a DRAFT run plan (never promoted); recorded results go in
one atomic result transaction under the user's delegated token only; the report is
a blob on an entry. Plus the encrypted credential store and its append-only
push audit."""
from __future__ import annotations

import sqlite3

import pytest
from benchling_fake import (
    BASE,
    METHOD_VERSION,
    RESULT_SCHEMA,
    STUDY_ID,
    TITER_FIELD,
    FakeBenchling,
    bioprocess_study,
)

from kalos.integrations.benchling import secrets
from kalos.integrations.benchling.client import (
    AppCredentials,
    BenchlingClient,
    BenchlingConfigError,
    ClientCredentialsToken,
    DelegatedToken,
    TokenBucket,
)
from kalos.integrations.benchling.pull import fetch_study, map_study
from kalos.integrations.benchling.push import (
    DelegatedTokenRequired,
    PushFailed,
    PushRefused,
    attach_report,
    plan_conditions,
    push_proposals,
    push_results,
)
from kalos.integrations.benchling.store import IntegrationStore

KEY = "k" * 40


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv(secrets.SECRET_KEY_VAR, KEY)


def _app(fake: FakeBenchling) -> BenchlingClient:
    return BenchlingClient(BASE, ClientCredentialsToken(AppCredentials(fake.client_id, fake.client_secret)),
                           transport=fake.transport(), bucket=TokenBucket(10_000, 30.0))


def _user(fake: FakeBenchling, token: str = "user-token-alice") -> BenchlingClient:  # noqa: S107
    fake.delegated_tokens[token] = "ent_alice"
    return BenchlingClient(BASE, DelegatedToken(token), transport=fake.transport(), bucket=TokenBucket(10_000, 30.0))


def _provenance(fake: FakeBenchling) -> list[dict]:
    with _app(fake) as c:
        data = fetch_study(c, STUDY_ID, target_field_id=TITER_FIELD, result_schema_id=RESULT_SCHEMA)
    return map_study(data, target_field_id=TITER_FIELD).provenance_table()


def _proposals(n: int) -> list[dict]:
    return [{"recipe": {"Temperature [degC]": 30.0 + i * 0.1, "pH setpoint": 7.0, "Feed rate [mL/h]": 2.5,
                        "Medium": "CD-CHO", "Dissolved oxygen (measured) [%]": 40.0},
             "pred": 2.1, "std": 0.3, "mode": "exploit"} for i in range(n)]


# --- proposals ---------------------------------------------------------------------- #


def test_proposals_become_a_draft_run_plan_and_promote_is_never_called():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    prov = _provenance(fake)
    with _app(fake) as c:
        out = push_proposals(c, study_id=STUDY_ID, proposals=_proposals(30), provenance=prov,
                             description="Kalos batch 1", flowchart_config_version_id="wffccv_Lk3mQpXn")
    assert out.run_plan_id.startswith("prp_") and len(out.condition_ids) == 30
    assert not any("promote" in p for p in fake.paths())
    batch_calls = [x for x in fake.calls if x.path.endswith(":batch-create")]
    assert [len(x.body["items"]) for x in batch_calls] == [25, 5]
    plan_body = fake.run_plans[0]["_body"]
    assert plan_body == {"studyId": STUDY_ID, "description": "Kalos batch 1",
                         "workflowFlowchartConfigVersionId": "wffccv_Lk3mQpXn"}
    cond = fake.plan_conditions[0]
    assert cond["templateMethodVersionId"] == METHOD_VERSION
    assert cond["replicates"] == [{"replicateNumber": 1}]
    values = {v["parameterId"]: v["value"]["value"] for v in cond["methodParametersPlan"]["dataParameterValues"]}
    assert values["schver_Rm4nTvWz"] == {"type": "ft_float", "floatValue": 30.0, "value": "30.0"}
    assert values["schver_media001"] == {"type": "ft_string", "value": "CD-CHO"}
    assert "schver_dooo0001" not in values  # measured-only: cannot be planned
    assert out.columns_not_plannable == ["Dissolved oxygen (measured) [%]"]
    assert "not a measured result" in cond["description"] and len(cond["description"]) <= 255
    assert out.identity == "app" and out.creator == "app_XXXXXXXXXXXXXXXX"


def test_client_refuses_promote_outright():
    fake = FakeBenchling()
    with _app(fake) as c, pytest.raises(BenchlingConfigError):
        c.post("/api/v3/procedure-run-plan/prp_Xk9mLpQr:promote", {})
    assert fake.calls == []


def test_proposal_push_with_delegated_token_names_the_user():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    prov = _provenance(fake)
    with _user(fake) as c:
        out = push_proposals(c, study_id=STUDY_ID, proposals=_proposals(2), provenance=prov, description="d")
    assert out.identity == "delegated" and out.creator == "ent_alice"


def test_plan_limits_and_unmappable_proposals_are_refused_before_any_call():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    prov = _provenance(fake)
    with pytest.raises(PushRefused):
        plan_conditions(_proposals(101), prov)  # > 100 conditions for one method
    with pytest.raises(PushRefused):
        plan_conditions([{"recipe": {"Unknown column": 1.0}}], prov)
    with pytest.raises(PushRefused):
        plan_conditions([{"recipe": {}}], prov)
    with _app(fake) as c, pytest.raises(PushRefused):
        push_proposals(c, study_id=STUDY_ID, proposals=[], provenance=prov, description="d")
    assert not any(p.startswith("/api/v3/procedure-run-plan") for p in fake.paths())


def test_failed_condition_batch_reports_the_plan_already_created():
    fake = bioprocess_study(FakeBenchling(), n_runs=1, conditions_per_run=2, replicates=1)
    prov = _provenance(fake)
    fake.script["/api/v3/procedure-run-plan-condition:batch-create"] = [(400, {}, {"title": "bad parameter"})]
    with _app(fake) as c, pytest.raises(PushFailed) as err:
        push_proposals(c, study_id=STUDY_ID, proposals=_proposals(3), provenance=prov, description="d")
    assert err.value.created["run_plan_id"].startswith("prp_")


# --- results --------------------------------------------------------------------------- #


def _results(n: int) -> list[dict]:
    return [{"run_id": f"{i:032x}", "version": 1, "value": 1.0 + i} for i in range(n)]


def test_results_refuse_the_app_token_without_any_call():
    fake = FakeBenchling()
    with _app(fake) as c, pytest.raises(DelegatedTokenRequired):
        push_results(c, schema_id=RESULT_SCHEMA, target_system_name="titer", project_id=None,
                     results=_results(2))
    with _app(fake) as c, pytest.raises(DelegatedTokenRequired):
        attach_report(c, entry_id="etr_Hds1XAaq", field_name="report", content=b"{}", name="r.json")
    assert fake.calls == []


def test_results_are_one_atomic_transaction_under_the_user():
    fake = FakeBenchling()
    with _user(fake) as c:
        out = push_results(c, schema_id=RESULT_SCHEMA, target_system_name="titer", project_id="src_YzU5p4dR",
                           results=_results(150), schema_fields=["titer", "kalos_run_id"])
    paths = fake.paths("POST")
    assert paths[0] == "/api/v2/result-transactions"
    assert paths.count(f"/api/v2/result-transactions/{out.transaction_id}/results") == 2  # 100 + 50
    assert paths[-1] == f"/api/v2/result-transactions/{out.transaction_id}:commit"
    assert len(fake.committed_results) == 150
    first = fake.committed_results[0]
    assert first["fields"] == {"titer": {"value": 1.0}, "kalos_run_id": {"value": f"{0:032x}"}}
    assert first["schemaId"] == RESULT_SCHEMA and first["projectId"] == "src_YzU5p4dR"
    assert all(r["creator"]["id"] == "ent_alice" for r in fake.committed_results)
    assert all(x.headers["authorization"] == "Bearer user-token-alice" for x in fake.calls)
    assert len(out.pushed) == 150 and out.identity == "delegated"


def test_a_failed_results_call_aborts_the_transaction():
    fake = FakeBenchling()
    fake.script["/api/v2/result-transactions/txn_00000001/results"] = [
        (200, {}, {"assayResults": [], "errors": {"0": "bad"}})]
    with _user(fake) as c, pytest.raises(PushFailed) as err:
        push_results(c, schema_id=RESULT_SCHEMA, target_system_name="titer", project_id=None,
                     results=_results(3))
    assert err.value.created["aborted"] is True
    assert fake.paths("POST")[-1] == "/api/v2/result-transactions/txn_00000001:abort"
    assert fake.committed_results == []


def test_non_finite_result_is_refused_before_any_call():
    fake = FakeBenchling()
    with _user(fake) as c, pytest.raises(PushRefused):
        push_results(c, schema_id=RESULT_SCHEMA, target_system_name="titer", project_id=None,
                     results=[{"run_id": "a", "version": 1, "value": float("nan")}])
    assert fake.calls == []


def test_report_goes_up_as_a_blob_and_onto_the_entry():
    fake = FakeBenchling()
    with _user(fake) as c:
        out = attach_report(c, entry_id="etr_Hds1XAaq", field_name="kalos_report",
                            content=b'{"analysis": 1}', name="kalos-analysis.json")
    assert fake.blobs[0]["_size"] == 15 and fake.blobs[0]["type"] == "RAW_FILE"
    assert fake.entry_patches == [{"entry": "etr_Hds1XAaq", "creator": {"__typename": "User", "id": "ent_alice"},
                                   "body": {"fields": {"kalos_report": {"value": out["blob_id"]}}}}]
    with _user(fake) as c, pytest.raises(BenchlingConfigError):
        attach_report(c, entry_id="../projects", field_name="f", content=b"{}", name="x")


# --- secrets and store -------------------------------------------------------------------- #


def test_secret_is_sealed_per_tenant_and_needs_the_key(tmp_path, monkeypatch):
    sealed = secrets.seal("s3cret-value", tenant="acme")
    assert "s3cret-value" not in sealed
    assert secrets.open_sealed(sealed, tenant="acme") == "s3cret-value"
    with pytest.raises(secrets.SecretUnreadable):
        secrets.open_sealed(sealed, tenant="globex")
    monkeypatch.setenv(secrets.SECRET_KEY_VAR, "z" * 40)
    with pytest.raises(secrets.SecretUnreadable):
        secrets.open_sealed(sealed, tenant="acme")
    monkeypatch.setenv(secrets.SECRET_KEY_VAR, "short")
    with pytest.raises(secrets.SecretKeyUnavailable):
        secrets.seal("x", tenant="acme")


def test_store_keeps_the_secret_encrypted_and_per_tenant(tmp_path):
    store = IntegrationStore(tmp_path)
    store.save_connection(tenant="acme", base_url=BASE, client_id="cid", client_secret="TOP-SECRET-123",
                          app_definition_id="appdef_x", connected_by="alice", verified=True)
    assert b"TOP-SECRET-123" not in store.path.read_bytes()
    assert "TOP-SECRET-123" not in str(store.connection(tenant="acme"))
    assert store.client_secret(tenant="acme") == "TOP-SECRET-123"
    assert store.connection(tenant="globex") is None and store.client_secret(tenant="globex") is None
    assert oct(store.path.stat().st_mode & 0o777) == "0o600"


def test_push_audit_is_append_only_and_tenant_scoped(tmp_path):
    store = IntegrationStore(tmp_path)
    aid = store.record_push(tenant="acme", subject="alice", kalos_auth_method="static",
                            benchling_identity="delegated", benchling_creator="ent_alice",
                            action="benchling.push_results", campaign_id="cmp_x", source_id=STUDY_ID,
                            outcome="ok", benchling_ids={"transaction_id": "txn_1", "result_ids": ["r1"]},
                            pushed_results=[("run1", 1, "r1")])
    rows = store.push_audit(tenant="acme")
    assert rows[0]["id"] == aid and rows[0]["benchling_ids"]["result_ids"] == ["r1"]
    assert store.push_audit(tenant="globex") == []
    assert store.pushed_versions(tenant="acme", campaign_id="cmp_x") == {("run1", 1)}
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("UPDATE benchling_push_audit SET outcome = 'x'")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("DELETE FROM benchling_push_audit")
    with pytest.raises(ValueError):
        store.record_push(tenant="acme", subject=None, kalos_auth_method=None, benchling_identity="root",
                          action="x", campaign_id=None, source_id=None, outcome="ok", benchling_ids={})


def test_change_notices_are_append_only(tmp_path):
    store = IntegrationStore(tmp_path)
    assert store.record_change(tenant="acme", origin="poll", event_type="procedureRun.modified",
                               resource_id="prun_1", marker="2026-10-01T00:00:00+00:00")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("UPDATE benchling_changes SET marker = 'x'")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute("DELETE FROM benchling_changes")
    assert len(store.changes(tenant="acme")) == 1
    # an existing database gets the trigger when it is opened again
    store.close()
    again = IntegrationStore(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        again._conn.execute("DELETE FROM benchling_changes WHERE tenant = 'acme'")
