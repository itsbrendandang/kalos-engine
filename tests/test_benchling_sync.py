"""Webhook signature verification per Benchling's "Webhook Security" page (ECDSA
P-256 over `{id}.{timestamp}.{body}`, JWKS keys, 5-minute window), idempotent
re-delivery, and polling-based change detection for Bioprocess studies."""
from __future__ import annotations

import base64
import json
import time

import pytest
from benchling_fake import BASE, STUDY_ID, FakeBenchling, bioprocess_study
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from kalos.integrations.benchling.client import (
    AppCredentials,
    BenchlingClient,
    ClientCredentialsToken,
    TokenBucket,
)
from kalos.integrations.benchling.store import IntegrationStore
from kalos.integrations.benchling.sync import (
    JwksCache,
    WebhookRejected,
    handle_webhook,
    poll_study,
    public_keys,
    verify_webhook,
)

APP_DEF = "appdef_XXXXXXXXXX"


def _b64url(n: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(32, "big")).rstrip(b"=").decode()


def make_key() -> tuple[ec.EllipticCurvePrivateKey, dict]:
    key = ec.generate_private_key(ec.SECP256R1())
    nums = key.public_key().public_numbers()
    return key, {"kty": "EC", "crv": "P-256", "use": "sig", "kid": "k1", "x": _b64url(nums.x), "y": _b64url(nums.y)}


def sign(key: ec.EllipticCurvePrivateKey, body: bytes, *, wid: str = "msg_1", ts: float | None = None,
         raw_too: bool = True) -> dict[str, str]:
    stamp = str(int(ts if ts is not None else time.time()))
    der = key.sign(f"{wid}.{stamp}.".encode() + body, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    sigs = [f"v1bder,{base64.b64encode(der).decode()}"]
    if raw_too:
        sigs.insert(0, f"v1b,{base64.b64encode(raw).decode()}")
    return {"webhook-id": wid, "webhook-timestamp": stamp, "webhook-signature": " ".join(sigs)}


def payload(etype: str = "v3.run.created", *, resource: str = "run_21f50003", evt: str = "evt_1",
            base: str = BASE, app: str = APP_DEF) -> bytes:
    message: dict = {"type": etype, "id": evt, "createdAt": "2026-08-27T22:37:37+00:00", "deprecated": False}
    if etype == "v2.assayRun.created":
        message["assayRun"] = {"id": "77af3205-65af-457f-87f5-75462b85075a",
                               "schema": {"id": "assaysch_fFLKmdmG", "name": "Assay Run"}}
    else:
        message.update({"resourceId": resource, "stability": "beta"})
    return json.dumps({"version": "0", "baseURL": base, "tenantId": "ten_XXXXXXXXXX", "app": {"id": "app_X"},
                       "appDefinition": {"id": app, "versionNumber": "0.1.1"}, "channel": "events",
                       "message": message}).encode()


@pytest.fixture
def keyed():
    key, jwk = make_key()
    return key, JwksCache(lambda _app: {"keys": [jwk]})


def _conn() -> dict:
    return {"app_definition_id": APP_DEF, "base_url": BASE}


def test_valid_signature_verifies_with_der_or_raw(keyed):
    key, jwks = keyed
    body = payload()
    keys = jwks.keys(APP_DEF)
    assert verify_webhook(sign(key, body), body, keys) == "msg_1"
    only_raw = sign(key, body)
    only_raw["webhook-signature"] = only_raw["webhook-signature"].split(" ")[0]
    assert verify_webhook(only_raw, body, keys) == "msg_1"


@pytest.mark.parametrize("tamper", ["body", "id", "timestamp", "signature", "old", "future", "missing", "wrongkey"])
def test_tampering_is_rejected(keyed, tamper):
    key, jwks = keyed
    body = payload()
    headers = sign(key, body)
    keys = jwks.keys(APP_DEF)
    if tamper == "body":
        body = body.replace(b"run_21f50003", b"run_21f50004")
    elif tamper == "id":
        headers["webhook-id"] = "msg_2"
    elif tamper == "timestamp":
        headers["webhook-timestamp"] = str(int(headers["webhook-timestamp"]) + 1)
    elif tamper == "signature":
        headers["webhook-signature"] = "v1bder," + base64.b64encode(b"\x30\x06\x02\x01\x01\x02\x01\x01").decode()
    elif tamper == "old":
        headers = sign(key, body, ts=time.time() - 301)
    elif tamper == "future":
        headers = sign(key, body, ts=time.time() + 301)
    elif tamper == "missing":
        headers.pop("webhook-signature")
    elif tamper == "wrongkey":
        other, _ = make_key()
        headers = sign(other, body)
    with pytest.raises(WebhookRejected):
        verify_webhook(headers, body, keys)


def test_header_names_are_case_insensitive(keyed):
    key, jwks = keyed
    body = payload()
    headers = {k.title(): v for k, v in sign(key, body).items()}
    assert verify_webhook(headers, body, jwks.keys(APP_DEF)) == "msg_1"


def test_jwks_skips_non_p256_keys():
    _, jwk = make_key()
    assert len(public_keys({"keys": [jwk, {"kty": "RSA", "n": "x", "e": "AQAB"}, {"kty": "EC", "crv": "P-384"}]})) == 1


def test_rotated_key_triggers_one_refetch(tmp_path):
    _old_key, old_jwk = make_key()
    new_key, new_jwk = make_key()
    clock = {"t": 0.0}
    served = [[old_jwk]]
    fetches = []

    def fetch(_app):
        fetches.append(1)
        return {"keys": served[0]}

    jwks = JwksCache(fetch, clock=lambda: clock["t"])
    jwks.keys(APP_DEF)
    served[0] = [new_jwk]
    clock["t"] = 120.0
    store = IntegrationStore(tmp_path)
    out = handle_webhook(store, tenant="acme", connection=_conn(), headers=sign(new_key, payload()),
                         body=payload(), jwks=jwks)
    assert out.status == "accepted" and len(fetches) == 2


def test_handle_records_v3_run_created_once_on_redelivery(tmp_path, keyed):
    key, jwks = keyed
    store = IntegrationStore(tmp_path)
    body = payload("v3.run.created", resource="run_21f50003")
    headers = sign(key, body, wid="msg_7")
    first = handle_webhook(store, tenant="acme", connection=_conn(), headers=headers, body=body, jwks=jwks)
    again = handle_webhook(store, tenant="acme", connection=_conn(), headers=headers, body=body, jwks=jwks)
    assert first.status == "accepted" and first.resource_id == "run_21f50003"
    assert again.status == "duplicate"
    # the same event re-sent under a new webhook id is still one change notice
    resent = handle_webhook(store, tenant="acme", connection=_conn(), headers=sign(key, body, wid="msg_8"),
                            body=body, jwks=jwks)
    assert resent.status == "accepted"
    changes = store.changes(tenant="acme")
    assert [(c["event_type"], c["resource_id"]) for c in changes] == [("v3.run.created", "run_21f50003")]
    assert store.changes(tenant="globex") == []


def test_handle_v2_assay_run_created_records_the_schema(tmp_path, keyed):
    key, jwks = keyed
    store = IntegrationStore(tmp_path)
    body = payload("v2.assayRun.created")
    out = handle_webhook(store, tenant="acme", connection=_conn(), headers=sign(key, body), body=body, jwks=jwks)
    assert out.status == "accepted"
    change = store.changes(tenant="acme")[0]
    assert change["resource_id"] == "77af3205-65af-457f-87f5-75462b85075a"
    assert change["scope_id"] == "assaysch_fFLKmdmG"


def test_unhandled_event_is_acknowledged_and_ignored(tmp_path, keyed):
    key, jwks = keyed
    store = IntegrationStore(tmp_path)
    body = payload("v3.entry.created")
    out = handle_webhook(store, tenant="acme", connection=_conn(), headers=sign(key, body), body=body, jwks=jwks)
    assert out.status == "ignored" and store.changes(tenant="acme") == []


@pytest.mark.parametrize("field, value", [("app", "appdef_OTHER"), ("base", "https://globex.benchling.com"),
                                          ("base", "https://evil.example")])
def test_payload_for_another_app_or_tenant_is_rejected(tmp_path, keyed, field, value):
    key, jwks = keyed
    store = IntegrationStore(tmp_path)
    body = payload(**{field: value})
    with pytest.raises(WebhookRejected):
        handle_webhook(store, tenant="acme", connection=_conn(), headers=sign(key, body), body=body, jwks=jwks)
    assert store.changes(tenant="acme") == []


def test_polling_records_each_modified_run_once(tmp_path):
    fake = bioprocess_study(FakeBenchling(), n_runs=3, conditions_per_run=1, replicates=1)
    store = IntegrationStore(tmp_path)
    client = BenchlingClient(BASE, ClientCredentialsToken(AppCredentials(fake.client_id, fake.client_secret)),
                             transport=fake.transport(), bucket=TokenBucket(10_000, 30.0))
    first = poll_study(client, store, tenant="acme", study_id=STUDY_ID, result_schema_id="assaysch_fFLKmdmG")
    assert first["new_changes"] == 3 + 3  # 3 runs + 3 results
    again = poll_study(client, store, tenant="acme", study_id=STUDY_ID)
    assert again["new_changes"] == 0
    fake.runs[1]["modifiedAt"] = "2026-11-01T00:00:00+00:00"
    third = poll_study(client, store, tenant="acme", study_id=STUDY_ID)
    assert third["new_changes"] == 1
    polls = [c for c in fake.calls if c.path == "/api/v3/procedure-run/items"]
    assert "modifiedAt.gte" not in polls[0].params and polls[1].params["modifiedAt.gte"].startswith("2026-10-01")
    assert all(c.params["sort"] == "modifiedAt:asc" for c in polls)
