"""Change detection: webhooks for what Benchling emits, polling for Bioprocess.

**Webhooks** (docs "Webhook Security", fetched 2026-11-06). Benchling signs each
delivery with an ECDSA P-256 key from a JWKS published per app definition at
`https://apps.benchling.com/api/v1/apps/{app_definition_id}/jwks`, rotated often;
the docs ask for a refetch at least every 6 hours. Verification:

1. `webhook-timestamp` (seconds since epoch) within 5 minutes of now;
2. signed content = `{webhook-id}.{webhook-timestamp}.{raw body}`, unaltered;
3. `webhook-signature` is a space-separated list of `v<n><b|bder>,<base64>`; the
   delivery is genuine if ANY signature verifies with ANY current key (SHA-256).
   `...der` values are DER-encoded; `v1b` values are raw r||s (64 bytes).

Then the payload must name this tenant's app definition and Benchling URL. The
handled events are `v2.assayRun.created` and `v3.run.created`; others are
acknowledged and ignored. A delivery is recorded under its `webhook-id`, so a
re-delivery is a no-op; the change notice is keyed on the event id too.

**Polling.** The V3 events list has no procedure-run or result events, so
Bioprocess changes are found by listing a study's procedure runs (and optionally
a result schema's results) with `modifiedAt.gte` the last high-water mark. Each
(id, modifiedAt) becomes one change notice; seeing it again adds nothing.

Change notices tell a person "this study has new data"; nothing is imported
automatically.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from kalos.integrations.benchling.client import (
    BenchlingClient,
    BenchlingConfigError,
    api_id,
    normalize_base_url,
)
from kalos.integrations.benchling.store import IntegrationStore

JWKS_URL = "https://apps.benchling.com/api/v1/apps/{app_definition_id}/jwks"
JWKS_MAX_AGE_S = 6 * 3600.0
JWKS_REFRESH_MIN_INTERVAL_S = 60.0
TIMESTAMP_TOLERANCE_S = 300.0
MAX_WEBHOOK_BYTES = 1024 * 1024
HANDLED_EVENTS = frozenset({"v2.assayRun.created", "v3.run.created"})
MAX_SIGNATURES = 10


class WebhookRejected(Exception):
    """The delivery failed verification. The reason is for the server log only;
    the caller answers with one generic body."""


# --- keys ------------------------------------------------------------------------- #


def _b64url(data: str) -> int:
    return int.from_bytes(base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)), "big")


def public_keys(jwks: Mapping[str, Any]) -> list[ec.EllipticCurvePublicKey]:
    """EC P-256 keys from a JWKS document; other key types are skipped."""
    keys = []
    for jwk in jwks.get("keys") or []:
        if not isinstance(jwk, dict) or jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
            continue
        try:
            keys.append(ec.EllipticCurvePublicNumbers(_b64url(jwk["x"]), _b64url(jwk["y"]),
                                                      ec.SECP256R1()).public_key())
        except (KeyError, ValueError, TypeError):
            continue
    return keys


def fetch_jwks(app_definition_id: str, *, transport: httpx.BaseTransport | None = None) -> dict[str, Any]:
    url = JWKS_URL.format(app_definition_id=api_id(app_definition_id))
    with httpx.Client(transport=transport, timeout=10.0, follow_redirects=False) as http:
        resp = http.get(url, headers={"Accept": "application/json"})
    if resp.status_code != 200:
        raise WebhookRejected(f"jwks fetch returned {resp.status_code}")
    body = resp.json()
    if not isinstance(body, dict):
        raise WebhookRejected("jwks is not an object")
    return body


class JwksCache:
    """Keys per app definition, refetched after JWKS_MAX_AGE_S, or early (at most
    once a minute) when no cached key verifies a delivery (rotation)."""

    def __init__(self, fetch: Callable[[str], Mapping[str, Any]] = fetch_jwks, *,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._fetch = fetch
        self._clock = clock
        self._cache: dict[str, tuple[float, list[ec.EllipticCurvePublicKey]]] = {}
        self._lock = threading.Lock()

    def keys(self, app_definition_id: str, *, force: bool = False) -> list[ec.EllipticCurvePublicKey]:
        now = self._clock()
        with self._lock:
            hit = self._cache.get(app_definition_id)
            fresh = hit is not None and now - hit[0] < JWKS_MAX_AGE_S
            recent = hit is not None and now - hit[0] < JWKS_REFRESH_MIN_INTERVAL_S
            if hit is not None and ((fresh and not force) or (force and recent)):
                return hit[1]
        keys = public_keys(self._fetch(app_definition_id))
        with self._lock:
            self._cache[app_definition_id] = (now, keys)
        return keys


# --- verification ------------------------------------------------------------------- #


def _signatures(header: str) -> list[tuple[str, bytes]]:
    out = []
    for part in header.split(" ")[:MAX_SIGNATURES]:
        version, sep, value = part.partition(",")
        if not sep or not version.startswith("v"):
            continue
        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError:
            continue
        if version.endswith("der"):
            out.append(("der", raw))
        elif version.endswith("b") and len(raw) == 64:
            r, s = int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
            out.append(("raw", encode_dss_signature(r, s)))
    return out


def verify_webhook(
    headers: Mapping[str, str], body: bytes, keys: list[ec.EllipticCurvePublicKey], *,
    now: float | None = None,
) -> str:
    """Check timestamp and signature. Returns the webhook id. Raises
    WebhookRejected. `headers` keys are matched case-insensitively."""
    h = {k.lower(): v for k, v in headers.items()}
    wid, ts, sig = h.get("webhook-id"), h.get("webhook-timestamp"), h.get("webhook-signature")
    if not wid or not ts or not sig:
        raise WebhookRejected("missing signature headers")
    if len(wid) > 128 or len(sig) > 4096:
        raise WebhookRejected("oversized signature headers")
    try:
        sent = float(ts)
    except ValueError:
        raise WebhookRejected("bad timestamp") from None
    current = time.time() if now is None else now
    if abs(current - sent) > TIMESTAMP_TOLERANCE_S:
        raise WebhookRejected("timestamp outside tolerance")
    signed = f"{wid}.{ts}.".encode() + body
    for _kind, der in _signatures(sig):
        for key in keys:
            try:
                key.verify(der, signed, ec.ECDSA(hashes.SHA256()))
                return wid
            except InvalidSignature:
                continue
    raise WebhookRejected("no signature verified")


@dataclass
class WebhookOutcome:
    status: str          # "accepted" | "duplicate" | "ignored"
    event_type: str | None = None
    resource_id: str | None = None


def handle_webhook(
    store: IntegrationStore, *, tenant: str, connection: Mapping[str, Any], headers: Mapping[str, str],
    body: bytes, jwks: JwksCache, now: float | None = None,
) -> WebhookOutcome:
    """Verify, deduplicate and record one delivery for `tenant`. Raises
    WebhookRejected on any failed check."""
    if len(body) > MAX_WEBHOOK_BYTES:
        raise WebhookRejected("body too large")
    app_def = connection.get("app_definition_id")
    if not isinstance(app_def, str) or not app_def:
        raise WebhookRejected("tenant has no app definition id")
    try:
        wid = verify_webhook(headers, body, jwks.keys(app_def), now=now)
    except WebhookRejected:
        # One early refetch covers a key rotated since the cache was filled.
        wid = verify_webhook(headers, body, jwks.keys(app_def, force=True), now=now)
    try:
        payload = json.loads(body)
    except ValueError:
        raise WebhookRejected("body is not JSON") from None
    if not isinstance(payload, dict):
        raise WebhookRejected("body is not an object")
    named_app = (payload.get("appDefinition") or {}).get("id") if isinstance(payload.get("appDefinition"), dict) else None
    if named_app != app_def:
        raise WebhookRejected("payload names another app definition")
    try:
        base = normalize_base_url(str(payload.get("baseURL") or ""))
    except BenchlingConfigError:
        raise WebhookRejected("payload base URL is not a Benchling tenant") from None
    if base != connection.get("base_url"):
        raise WebhookRejected("payload names another Benchling tenant")
    message: dict[str, Any] = payload["message"] if isinstance(payload.get("message"), dict) else {}
    etype = message.get("type") if isinstance(message.get("type"), str) else None
    if not store.claim_delivery(tenant=tenant, webhook_id=wid, event_type=etype):
        return WebhookOutcome("duplicate", etype)
    if etype not in HANDLED_EVENTS:
        return WebhookOutcome("ignored", etype)
    if etype == "v2.assayRun.created":
        run: dict[str, Any] = message["assayRun"] if isinstance(message.get("assayRun"), dict) else {}
        resource = run.get("id") or message.get("resourceId")
        schema: dict[str, Any] = run["schema"] if isinstance(run.get("schema"), dict) else {}
        scope = schema.get("id")
    else:
        resource = message.get("resourceId")
        scope = None
    if not isinstance(resource, str) or not resource:
        raise WebhookRejected("event names no resource")
    marker: str = message["id"] if isinstance(message.get("id"), str) else wid
    store.record_change(tenant=tenant, origin="webhook", event_type=etype, resource_id=resource,
                        marker=marker, scope_id=scope if isinstance(scope, str) else None)
    return WebhookOutcome("accepted", etype, resource)


# --- polling ------------------------------------------------------------------------------ #


def poll_study(
    client: BenchlingClient, store: IntegrationStore, *, tenant: str, study_id: str,
    result_schema_id: str | None = None,
) -> dict[str, Any]:
    """Record procedure runs (and results of `result_schema_id`) modified since
    the last poll of this study. Returns counts. Safe to repeat."""
    api_id(study_id)
    new = 0
    cursors: dict[str, str] = {}
    sources: list[tuple[str, str, dict[str, Any], str]] = [
        (f"study:{study_id}:runs", "/api/v3/procedure-run/items", {"study.eq": study_id}, "procedureRun.modified"),
    ]
    if result_schema_id:
        api_id(result_schema_id)
        sources.append((f"schema:{result_schema_id}:results", "/api/v3/result/items",
                        {"schema.eq": result_schema_id}, "result.modified"))
    for key, path, params, etype in sources:
        since = store.cursor(tenant=tenant, key=key)
        query = {**params, "sort": "modifiedAt:asc"}
        if since:
            query["modifiedAt.gte"] = since
        high = since or ""
        for item in client.paginate(path, query):
            rid, modified = item.get("id"), item.get("modifiedAt")
            if not isinstance(rid, str) or not isinstance(modified, str):
                continue
            if store.record_change(tenant=tenant, origin="poll", event_type=etype, resource_id=rid,
                                   marker=modified, scope_id=study_id):
                new += 1
            high = max(high, modified)
        if high:
            store.set_cursor(tenant=tenant, key=key, value=high)
            cursors[key] = high
    return {"new_changes": new, "cursors": cursors}
