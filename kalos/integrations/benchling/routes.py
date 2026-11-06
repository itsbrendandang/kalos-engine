"""`/api/integrations/benchling/*`: the Benchling connector over HTTP.

Off unless `KALOS_INTEGRATIONS` names `benchling`: every route then answers the
same 404 an unknown path gets. Rules, for every route:

- The tenant is always `principal.tenant` (the webhook route, which Benchling
  calls without a Kalos token, takes it from the path and proves it with the
  signature). No route reads another tenant's connection.
- Scopes: `read` for status and study lists; `write` for import, pushes and
  polling; `admin` for connect (it sets the tenant-wide Benchling credentials).
- The app secret comes in once, on connect, and is stored sealed with
  `KALOS_SECRET_KEY`. It is never returned, logged or audited. A delegated user
  token arrives in the `X-Benchling-Delegated-Token` header, is used for that one
  request, and is never stored or logged.
- Result and report writes need that delegated header (400 without it, before any
  Benchling call), so Benchling's own audit log names the person. Proposal pushes
  use it when given, else the app identity; either way the plan is a draft.
- Every push writes a `benchling_push_audit` row with the Benchling ids returned
  (also on failure, with whatever was created) and a row in the main audit log.
- Imports go through the regular upload path (`kalos.portal.app._run_uploaded`):
  the same parser, caps, checks, analysis slot and error codes as a file upload.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from kalos.integrations import enabled
from kalos.integrations.benchling import NAME
from kalos.integrations.benchling.client import (
    AppCredentials,
    BenchlingAuthError,
    BenchlingClient,
    BenchlingConfigError,
    BenchlingError,
    BenchlingForbidden,
    BenchlingNotFound,
    BenchlingRateLimited,
    ClientCredentialsToken,
    DelegatedToken,
    api_id,
    bucket_for,
    normalize_base_url,
)
from kalos.integrations.benchling.pull import (
    ImportRefused,
    RunSheet,
    fetch_assay,
    fetch_study,
    list_projects,
    list_studies,
    map_assay,
    map_study,
)
from kalos.integrations.benchling.push import (
    PushFailed,
    PushRefused,
    attach_report,
    push_proposals,
    push_results,
)
from kalos.integrations.benchling.secrets import (
    SecretKeyUnavailable,
    SecretUnreadable,
    key_configured,
)
from kalos.integrations.benchling.store import IntegrationStore, integration_store_at
from kalos.integrations.benchling.sync import (
    JwksCache,
    WebhookRejected,
    handle_webhook,
    poll_study,
)
from kalos.portal.auth import ADMIN, READ, WRITE, Principal, require_scope
from kalos.portal.campaign import CampaignError, CampaignStore, get_campaign_store
from kalos.portal.campaign_routes import (
    campaign_error_response,
    campaign_not_found_response,
)
from kalos.portal.migrate import CAMPAIGN_ID_PATTERN
from kalos.portal.tenancy import is_valid_tenant_id

log = logging.getLogger("kalos.integrations.benchling")

PREFIX = "/api/integrations/benchling"
DELEGATED_HEADER = "X-Benchling-Delegated-Token"
MAX_ID = 128
MAX_PROPOSALS = 100


def require_enabled() -> None:
    """404, byte-identical to an unknown path, unless the integration is on."""
    if not enabled(NAME):
        raise HTTPException(status_code=404)


router = APIRouter(prefix=PREFIX, dependencies=[Depends(require_enabled)])

_NOT_FOUND = b'{"error":"Not Found","code":"not_found"}'


class BenchlingGate:
    """ASGI gate in front of the router: with the integration off, any request
    under PREFIX gets the app's unknown-path 404 before its body is read or
    validated (the router dependency alone runs after body parsing)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and not enabled(NAME):
            path = scope.get("path", "")
            if path == PREFIX or path.startswith(PREFIX + "/"):
                await send({"type": "http.response.start", "status": 404, "headers": [
                    (b"content-length", str(len(_NOT_FOUND)).encode()), (b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": _NOT_FOUND})
                return
        await self.app(scope, receive, send)


# --- dependencies (overridden in tests) ------------------------------------------ #


def get_benchling_transport() -> Any:
    """The httpx transport for Benchling calls: None = the real network."""
    return None


def get_integration_store(store: CampaignStore = Depends(get_campaign_store)) -> IntegrationStore:
    return integration_store_at(store.state_dir)


_JWKS = JwksCache()


def get_jwks_cache() -> JwksCache:
    return _JWKS


# App tokens are cached per tenant and reused until they near expiry, keyed on a
# hash of the credentials so a reconnect starts fresh.
_APP_TOKENS: dict[str, tuple[str, ClientCredentialsToken]] = {}
_APP_TOKENS_LOCK = threading.Lock()


def _app_token(tenant: str, creds: AppCredentials) -> ClientCredentialsToken:
    fp = hashlib.sha256(f"{creds.client_id}\x00{creds.client_secret}".encode()).hexdigest()
    with _APP_TOKENS_LOCK:
        hit = _APP_TOKENS.get(tenant)
        if hit is not None and hit[0] == fp:
            return hit[1]
        token = ClientCredentialsToken(creds)
        _APP_TOKENS[tenant] = (fp, token)
        return token


def reset_token_cache() -> None:
    with _APP_TOKENS_LOCK:
        _APP_TOKENS.clear()


# --- helpers ---------------------------------------------------------------------- #


def _auth_method(principal: Principal) -> str:
    method = getattr(principal, "auth_method", None)
    if isinstance(method, str) and method:
        return method
    return "anonymous" if principal.anonymous else "bearer"


def _err(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    headers = extra.pop("headers", None)
    return JSONResponse({"error": message, "code": code, **extra}, status_code=status, headers=headers)


def _not_connected() -> JSONResponse:
    return _err(409, "benchling_not_connected", "this workspace has no Benchling connection; connect first")


def _benchling_error(exc: BenchlingError, *, delegated: bool = False) -> JSONResponse:
    """Map a Benchling failure to a response. Benchling's own text is not echoed
    (it can quote names); the status is."""
    if isinstance(exc, (ImportRefused, PushRefused)):
        return _err(400, "benchling_refused", str(exc))
    if isinstance(exc, BenchlingConfigError):
        return _err(400, "benchling_bad_request", str(exc))
    if isinstance(exc, BenchlingRateLimited):
        wait = int(exc.retry_after) if exc.retry_after else 30
        return _err(503, "benchling_rate_limited", "Benchling's rate limit was reached; try again later",
                    headers={"Retry-After": str(max(1, wait))})
    if isinstance(exc, BenchlingAuthError):
        who = "the delegated user token" if delegated else "the app credentials"
        return _err(502, "benchling_auth_failed", f"Benchling refused {who}", benchling_status=exc.status)
    if isinstance(exc, BenchlingForbidden):
        return _err(502, "benchling_forbidden", "Benchling refused access to that object",
                    benchling_status=exc.status)
    if isinstance(exc, BenchlingNotFound):
        return _err(404, "benchling_not_found", "Benchling has no such object (or it is not shared with the app)",
                    benchling_status=exc.status)
    return _err(502, "benchling_unavailable", "the Benchling call failed", benchling_status=exc.status)


def _connection_and_creds(
    istore: IntegrationStore, tenant: str,
) -> tuple[dict[str, Any], AppCredentials] | JSONResponse:
    conn = istore.connection(tenant=tenant)
    if conn is None:
        return _not_connected()
    try:
        secret = istore.client_secret(tenant=tenant)
    except (SecretUnreadable, SecretKeyUnavailable):
        return _err(503, "benchling_secret_unreadable",
                    "the stored Benchling credentials cannot be read with the current KALOS_SECRET_KEY; reconnect")
    if not secret:
        return _not_connected()
    return conn, AppCredentials(conn["client_id"], secret)


def _client(conn: dict[str, Any], auth: Any, transport: Any) -> BenchlingClient:
    return BenchlingClient(conn["base_url"], auth, transport=transport)


def _delegated(token: str | None) -> DelegatedToken | None:
    if token is None or not token.strip():
        return None
    return DelegatedToken(token.strip())


def _audit(
    store: CampaignStore, principal: Principal, action: str, status: int, *, campaign_id: str | None = None,
    detail: str | None = None,
) -> None:
    try:
        store.record_audit(
            tenant=principal.tenant, subject=principal.subject, auth_method=_auth_method(principal),
            action=action, campaign_id=campaign_id, outcome="ok" if status < 400 else f"http_{status}",
            detail=detail,
        )
    except Exception:  # noqa: BLE001 - auditing must not break the audited request
        log.exception("failed to write an audit entry for %s", action)


_Id = Annotated[str, StringConstraints(min_length=1, max_length=MAX_ID, pattern=r"^[A-Za-z0-9_-]+$")]
_Url = Annotated[str, StringConstraints(min_length=1, max_length=256)]
_Secret = Annotated[str, StringConstraints(min_length=1, max_length=512)]


# --- connect / status ------------------------------------------------------------------- #


class _ConnectBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: _Url
    client_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    client_secret: _Secret = Field(repr=False)
    app_definition_id: _Id | None = None
    verify: bool = True


@router.post("/connect")
async def connect(
    body: _ConnectBody,
    principal: Principal = Depends(require_scope(ADMIN)),
    store: CampaignStore = Depends(get_campaign_store),
    istore: IntegrationStore = Depends(get_integration_store),
    transport: Any = Depends(get_benchling_transport),
) -> JSONResponse:
    """Store this workspace's Benchling app credentials (secret sealed at rest).
    With `verify` (default) a token is requested first; refused credentials are
    not stored."""
    if not key_configured():
        response = _err(503, "secret_key_not_configured",
                        "KALOS_SECRET_KEY is not set on the engine, so credentials cannot be stored")
        _audit(store, principal, "benchling.connect", response.status_code)
        return response
    try:
        base_url = normalize_base_url(body.base_url)
        creds = AppCredentials(body.client_id.strip(), body.client_secret)
    except BenchlingConfigError as exc:
        response = _err(400, "benchling_bad_request", str(exc))
        _audit(store, principal, "benchling.connect", response.status_code)
        return response
    if body.verify:
        try:
            with BenchlingClient(base_url, ClientCredentialsToken(creds), transport=transport) as c:
                await run_in_threadpool(c.authenticate)
        except BenchlingError as exc:
            response = _benchling_error(exc)
            _audit(store, principal, "benchling.connect", response.status_code)
            return response
    istore.save_connection(tenant=principal.tenant, base_url=base_url, client_id=creds.client_id,
                           client_secret=creds.client_secret, app_definition_id=body.app_definition_id,
                           connected_by=principal.subject, verified=body.verify)
    with _APP_TOKENS_LOCK:
        _APP_TOKENS.pop(principal.tenant, None)
    _audit(store, principal, "benchling.connect", 200, detail=f"verified={str(body.verify).lower()}")
    return JSONResponse({"connected": True, **_public_connection(istore, principal.tenant)})


def _public_connection(istore: IntegrationStore, tenant: str) -> dict[str, Any]:
    conn = istore.connection(tenant=tenant) or {}
    return {k: conn.get(k) for k in ("base_url", "client_id", "app_definition_id", "connected_by",
                                     "connected_at", "verified_at")}


@router.get("/status")
def status(
    principal: Principal = Depends(require_scope(READ)),
    istore: IntegrationStore = Depends(get_integration_store),
) -> JSONResponse:
    """Connection (never the secret), whether the secret can be opened, recent
    imports, recent pushes with the Benchling ids, and change notices."""
    tenant = principal.tenant
    conn = istore.connection(tenant=tenant)
    if conn is None:
        return JSONResponse({"enabled": True, "connected": False, "secret_key_configured": key_configured()})
    try:
        readable = istore.client_secret(tenant=tenant) is not None
    except (SecretUnreadable, SecretKeyUnavailable):
        readable = False
    bucket = bucket_for(conn["base_url"])
    return JSONResponse({
        "enabled": True, "connected": True, "secret_key_configured": key_configured(),
        "credentials_readable": readable, **_public_connection(istore, tenant),
        "rate_limit": {"requests": bucket.capacity, "per_seconds": bucket.period_s, "available": bucket.available},
        "imports": istore.imports(tenant=tenant, limit=5),
        "pushes": istore.push_audit(tenant=tenant, limit=5),
        "changes": istore.changes(tenant=tenant, limit=50),
    })


@router.get("/studies")
async def studies(
    principal: Principal = Depends(require_scope(READ)),
    istore: IntegrationStore = Depends(get_integration_store),
    transport: Any = Depends(get_benchling_transport),
) -> JSONResponse:
    """Bioprocess studies (v3) and projects (v2) the app can see."""
    got = _connection_and_creds(istore, principal.tenant)
    if isinstance(got, JSONResponse):
        return got
    conn, creds = got

    def _work() -> dict[str, Any]:
        with _client(conn, _app_token(principal.tenant, creds), transport) as c:
            out: dict[str, Any] = {"projects": list_projects(c)}
            try:
                out["studies"] = list_studies(c)
            except (BenchlingNotFound, BenchlingForbidden):
                out["studies"] = []
                out["studies_note"] = "the study API answered 403/404: Bioprocess may not be enabled on this tenant"
            return out

    try:
        return JSONResponse(await run_in_threadpool(_work))
    except BenchlingError as exc:
        return _benchling_error(exc)


# --- import -------------------------------------------------------------------------------- #


class _ImportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: _Id | None = None
    target_field: _Id
    result_schema_id: _Id | None = None
    assay_run_schema_id: _Id | None = None
    campaign_id: Annotated[str, StringConstraints(max_length=64)] | None = None
    replace: bool = False


@router.post("/import")
async def import_study(
    body: _ImportBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
    istore: IntegrationStore = Depends(get_integration_store),
    transport: Any = Depends(get_benchling_transport),
) -> JSONResponse:
    """Fetch a study (`study_id`, v3) or assay runs (`assay_run_schema_id` +
    `result_schema_id`, v2), map them to a run sheet, and upload it into
    `campaign_id` (a new campaign when omitted) through the regular upload path."""
    from kalos.portal.app import _run_uploaded  # deferred: app mounts this router

    tenant = principal.tenant
    if (body.study_id is None) == (body.assay_run_schema_id is None):
        return _err(400, "invalid_body", "give exactly one of study_id or assay_run_schema_id")
    if body.assay_run_schema_id is not None and body.result_schema_id is None:
        return _err(400, "invalid_body", "assay_run_schema_id needs result_schema_id")
    if body.campaign_id is not None:
        if CAMPAIGN_ID_PATTERN.fullmatch(body.campaign_id) is None:
            return campaign_not_found_response()
        try:
            info = store.campaign_info(tenant=tenant, campaign_id=body.campaign_id)
        except CampaignError as exc:
            return campaign_error_response(exc)
        if info["archived"]:
            return _err(409, "campaign_archived", "the campaign is archived; it can be read but not changed")
    got = _connection_and_creds(istore, tenant)
    if isinstance(got, JSONResponse):
        return got
    conn, creds = got

    def _fetch() -> RunSheet:
        with _client(conn, _app_token(tenant, creds), transport) as c:
            if body.study_id is not None:
                data = fetch_study(c, body.study_id, target_field_id=body.target_field,
                                   result_schema_id=body.result_schema_id)
                return map_study(data, target_field_id=body.target_field)
            assert body.assay_run_schema_id is not None and body.result_schema_id is not None
            adata = fetch_assay(c, body.assay_run_schema_id, body.result_schema_id)
            return map_assay(adata, target_field=body.target_field)

    try:
        sheet = await run_in_threadpool(_fetch)
    except BenchlingError as exc:
        response = _benchling_error(exc)
        _audit(store, principal, "benchling.import", response.status_code, campaign_id=body.campaign_id)
        return response

    campaign_id = body.campaign_id
    if campaign_id is None:
        try:
            campaign_id = store.create_campaign(
                tenant=tenant, name=f"Benchling {sheet.source_id}"[:120], created_by=principal.subject,
            )["id"]
        except CampaignError as exc:
            return campaign_error_response(exc)
    raw = sheet.to_csv()
    upload = UploadFile(file=io.BytesIO(raw), filename=sheet.filename)
    response = await _run_uploaded(
        upload, sheet.target_column, False, "", "", None, principal, store,
        campaign_id=campaign_id, replace=body.replace,
    )
    summary = {**sheet.summary(), "row_sources_count": len(sheet.row_sources),
               "flowchart_config_version_ids": sheet.flowchart_config_version_ids}
    import_id = istore.record_import(
        tenant=tenant, campaign_id=campaign_id, summary=summary, sheet=raw,
        outcome="ok" if response.status_code == 200 else f"http_{response.status_code}",
        created_by=principal.subject,
    )
    _audit(store, principal, "benchling.import", response.status_code, campaign_id=campaign_id,
           detail=f"import_id={import_id} rows={len(sheet.rows)}")
    try:
        analysis = json.loads(bytes(response.body))
    except ValueError:
        analysis = None
    return JSONResponse({
        "campaign_id": campaign_id, "import_id": import_id, "filename": sheet.filename,
        "rows": len(sheet.rows), "report": summary["report"], "provenance": summary["provenance"],
        "target_column": sheet.target_column, "analysis": analysis,
    }, status_code=response.status_code)


# --- pushes ------------------------------------------------------------------------------------ #


class _PushProposalsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: Annotated[str, StringConstraints(max_length=64)]
    study_id: _Id
    description: Annotated[str, StringConstraints(max_length=255)] | None = None
    proposal_indices: list[Annotated[int, Field(ge=0, lt=MAX_PROPOSALS)]] | None = Field(
        default=None, max_length=MAX_PROPOSALS)
    flowchart_config_version_id: _Id | None = None


def _campaign_import(
    store: CampaignStore, istore: IntegrationStore, tenant: str, campaign_id: str,
) -> dict[str, Any] | JSONResponse:
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        return campaign_not_found_response()
    try:
        store.campaign_info(tenant=tenant, campaign_id=campaign_id)
    except CampaignError as exc:
        return campaign_error_response(exc)
    imp = istore.latest_import(tenant=tenant, campaign_id=campaign_id)
    if imp is None:
        return _err(409, "benchling_not_imported",
                    "this campaign was not imported from Benchling, so its columns cannot be mapped back")
    return imp


@router.post("/push-proposals")
async def push_proposals_route(
    body: _PushProposalsBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
    istore: IntegrationStore = Depends(get_integration_store),
    transport: Any = Depends(get_benchling_transport),
    delegated_token: str | None = Header(default=None, alias=DELEGATED_HEADER),
) -> JSONResponse:
    """Write the campaign's current proposals as a DRAFT run plan in `study_id`
    (never promoted). Uses the delegated user token when the header is present."""
    tenant = principal.tenant
    imp = _campaign_import(store, istore, tenant, body.campaign_id)
    if isinstance(imp, JSONResponse):
        return imp
    if imp.get("source") != "v3_study" or imp.get("source_id") != body.study_id:
        return _err(409, "benchling_study_mismatch",
                    "proposals can only go to the study this campaign was imported from")
    analysis = store.load_analysis(tenant=tenant, campaign_id=body.campaign_id) or {}
    proposals = [p for p in analysis.get("proposals") or [] if isinstance(p, dict)]
    if body.proposal_indices is not None:
        proposals = [proposals[i] for i in body.proposal_indices if i < len(proposals)]
    got = _connection_and_creds(istore, tenant)
    if isinstance(got, JSONResponse):
        return got
    conn, creds = got
    user = _delegated(delegated_token)
    auth: Any = user if user is not None else _app_token(tenant, creds)
    flowchart = body.flowchart_config_version_id or next(iter(imp.get("flowchart_config_version_ids") or []), None)
    description = body.description or f"Kalos proposals for campaign {body.campaign_id} (draft)"

    def _work() -> Any:
        with _client(conn, auth, transport) as c:
            return push_proposals(c, study_id=body.study_id, proposals=proposals,
                                  provenance=imp.get("provenance") or [], description=description,
                                  flowchart_config_version_id=flowchart)

    identity = "delegated" if user is not None else "app"
    try:
        out = await run_in_threadpool(_work)
    except BenchlingError as exc:
        created = exc.created if isinstance(exc, PushFailed) else {}
        response = _benchling_error(exc.cause if isinstance(exc, PushFailed) else exc, delegated=user is not None)
        if isinstance(exc, PushFailed):
            response = _err(502, "benchling_push_incomplete", str(exc), created=created)
        audit_id = istore.record_push(
            tenant=tenant, subject=principal.subject, kalos_auth_method=_auth_method(principal),
            benchling_identity=identity, action="benchling.push_proposals", campaign_id=body.campaign_id,
            source_id=body.study_id, outcome=f"http_{response.status_code}", benchling_ids=created,
        )
        _audit(store, principal, "benchling.push_proposals", response.status_code, campaign_id=body.campaign_id,
               detail=f"push_audit_id={audit_id}")
        return response
    ids = {**out.ids, "columns_not_plannable": out.columns_not_plannable}
    audit_id = istore.record_push(
        tenant=tenant, subject=principal.subject, kalos_auth_method=_auth_method(principal),
        benchling_identity=out.identity, benchling_creator=out.creator, action="benchling.push_proposals",
        campaign_id=body.campaign_id, source_id=body.study_id, outcome="ok", benchling_ids=ids,
    )
    _audit(store, principal, "benchling.push_proposals", 200, campaign_id=body.campaign_id,
           detail=f"push_audit_id={audit_id} conditions={len(out.condition_ids)}")
    return JSONResponse({
        "run_plan_id": out.run_plan_id, "condition_ids": out.condition_ids, "promoted": False,
        "benchling_identity": out.identity, "benchling_creator": out.creator,
        "columns_sent": out.columns_sent, "columns_not_plannable": out.columns_not_plannable,
        "push_audit_id": audit_id,
    })


class _PushResultsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: Annotated[str, StringConstraints(max_length=64)]
    report_entry_id: _Id | None = None
    report_field: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None


@router.post("/push-results")
async def push_results_route(
    body: _PushResultsBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
    istore: IntegrationStore = Depends(get_integration_store),
    transport: Any = Depends(get_benchling_transport),
    delegated_token: str | None = Header(default=None, alias=DELEGATED_HEADER),
) -> JSONResponse:
    """Write the campaign's recorded results that are not in Benchling yet as
    Benchling results, in one transaction, under the signed-in user's delegated
    token (required). Optionally attach the analysis as a blob on an entry."""
    tenant = principal.tenant
    user = _delegated(delegated_token)
    if user is None:
        response = _err(400, "delegated_token_required",
                        f"writing results needs the signed-in user's Benchling token in {DELEGATED_HEADER}; "
                        "the app's own credentials are never used for results")
        _audit(store, principal, "benchling.push_results", 400)
        return response
    if (body.report_entry_id is None) != (body.report_field is None):
        return _err(400, "invalid_body", "give both report_entry_id and report_field, or neither")
    imp = _campaign_import(store, istore, tenant, body.campaign_id)
    if isinstance(imp, JSONResponse):
        return imp
    target = imp.get("target") or {}
    if not target.get("schema_id") or not target.get("system_name"):
        return _err(409, "benchling_no_result_schema", "the import did not record a result schema and field")
    got = _connection_and_creds(istore, tenant)
    if isinstance(got, JSONResponse):
        return got
    conn, _creds = got
    state = store.get(tenant=tenant, campaign_id=body.campaign_id) or {}
    done = istore.pushed_versions(tenant=tenant, campaign_id=body.campaign_id)
    todo = [
        {"run_id": r["id"], "version": int(r.get("result_version") or 1), "value": r["result"]}
        for r in state.get("pending") or []
        if r.get("result") is not None and (r["id"], int(r.get("result_version") or 1)) not in done
    ]
    report: bytes | None = None
    if body.report_entry_id is not None:
        analysis = store.load_analysis(tenant=tenant, campaign_id=body.campaign_id)
        if analysis:
            report = json.dumps(analysis, sort_keys=True, default=str).encode()

    def _work() -> tuple[Any, dict[str, str] | None]:
        with _client(conn, user, transport) as c:
            res = push_results(c, schema_id=target["schema_id"], target_system_name=target["system_name"],
                               project_id=target.get("project_id"), results=todo,
                               schema_fields=target.get("schema_fields") or []) if todo else None
            attached = None
            if report is not None and body.report_entry_id and body.report_field:
                attached = attach_report(c, entry_id=api_id(body.report_entry_id), field_name=body.report_field,
                                         content=report, name=f"kalos-analysis-{body.campaign_id}.json")
            return res, attached

    if not todo and report is None:
        return _err(409, "nothing_to_push", "every recorded result of this campaign is already in Benchling")
    try:
        res, attached = await run_in_threadpool(_work)
    except BenchlingError as exc:
        created = exc.created if isinstance(exc, PushFailed) else {}
        response = (_err(502, "benchling_push_incomplete", str(exc), created=created)
                    if isinstance(exc, PushFailed) else _benchling_error(exc, delegated=True))
        audit_id = istore.record_push(
            tenant=tenant, subject=principal.subject, kalos_auth_method=_auth_method(principal),
            benchling_identity="delegated", action="benchling.push_results", campaign_id=body.campaign_id,
            source_id=target.get("schema_id"), outcome=f"http_{response.status_code}", benchling_ids=created,
        )
        _audit(store, principal, "benchling.push_results", response.status_code, campaign_id=body.campaign_id,
               detail=f"push_audit_id={audit_id}")
        return response
    ids: dict[str, Any] = {}
    if res is not None:
        ids.update({"transaction_id": res.transaction_id, "result_ids": res.result_ids})
    if attached is not None:
        ids.update({"report_blob_id": attached["blob_id"], "report_entry_id": attached["entry_id"]})
    audit_id = istore.record_push(
        tenant=tenant, subject=principal.subject, kalos_auth_method=_auth_method(principal),
        benchling_identity="delegated", action="benchling.push_results", campaign_id=body.campaign_id,
        source_id=target.get("schema_id"), outcome="ok", benchling_ids=ids,
        pushed_results=res.pushed if res is not None else None,
    )
    _audit(store, principal, "benchling.push_results", 200, campaign_id=body.campaign_id,
           detail=f"push_audit_id={audit_id} results={len(todo)}")
    return JSONResponse({**ids, "results_written": len(todo), "benchling_identity": "delegated",
                         "push_audit_id": audit_id})


# --- polling and webhooks --------------------------------------------------------------------------- #


class _PollBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: _Id
    result_schema_id: _Id | None = None


@router.post("/poll")
async def poll(
    body: _PollBody,
    principal: Principal = Depends(require_scope(WRITE)),
    istore: IntegrationStore = Depends(get_integration_store),
    transport: Any = Depends(get_benchling_transport),
) -> JSONResponse:
    """Record what changed in a study since the last poll (Bioprocess has no events)."""
    got = _connection_and_creds(istore, principal.tenant)
    if isinstance(got, JSONResponse):
        return got
    conn, creds = got

    def _work() -> dict[str, Any]:
        with _client(conn, _app_token(principal.tenant, creds), transport) as c:
            return poll_study(c, istore, tenant=principal.tenant, study_id=body.study_id,
                              result_schema_id=body.result_schema_id)

    try:
        return JSONResponse(await run_in_threadpool(_work))
    except BenchlingError as exc:
        return _benchling_error(exc)


_WEBHOOK_REJECTED = {"error": "webhook not verified", "code": "webhook_rejected"}


@router.post("/webhook/{tenant}")
async def webhook(
    tenant: str, request: Request,
    istore: IntegrationStore = Depends(get_integration_store),
    jwks: JwksCache = Depends(get_jwks_cache),
) -> JSONResponse:
    """Benchling event delivery for one workspace. No Kalos token: the signature
    is the proof. Any failed check answers the same 401 body (an unknown tenant
    included). Acknowledged quickly; nothing is imported here."""
    body = await request.body()
    conn = istore.connection(tenant=tenant) if is_valid_tenant_id(tenant) else None
    if conn is None:
        return JSONResponse(_WEBHOOK_REJECTED, status_code=401)
    try:
        out = await run_in_threadpool(
            handle_webhook, istore, tenant=tenant, connection=conn, headers=dict(request.headers),
            body=body, jwks=jwks,
        )
    except WebhookRejected as exc:
        log.warning("benchling webhook rejected: %s", exc)
        return JSONResponse(_WEBHOOK_REJECTED, status_code=401)
    return JSONResponse({"status": out.status, "event_type": out.event_type})
