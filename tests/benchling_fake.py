"""A small in-memory Benchling for the connector tests (no network, no tenant).

`FakeBenchling.transport()` is an `httpx.MockTransport` that answers the v2/v3
paths the connector calls, with response shapes taken from the public OpenAPI
specs (v2 `openapi.yaml`, v3 `openapi.yaml`). Object ids reuse the spec examples
where the spec has one (`stdy_9CYiCggY`, `prp_Xk9mLpQr`, `schver_Rm4nTvWz`,
`pcmdv_Hn5kRtYw`, `wffccv_Lk3mQpXn`, `assaysch_fFLKmdmG`, `etr_Hds1XAaq`,
`src_YzU5p4dR`). Every request is recorded in `calls` so tests can assert what
was (and was not) sent.

This is a fixture of the documented shapes, not of Benchling's behaviour: where
the spec leaves something open (how a v3 result links to a procedure run, the
read shape of a parameter value) the fake encodes the connector's assumption,
and docs/BENCHLING.md lists it as unverified.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

import httpx

BASE = "https://acme.benchling.com"
STUDY_ID = "stdy_9CYiCggY"
FLOWCHART_CV = "wffccv_Lk3mQpXn"
METHOD_VERSION = "pcmdv_Hn5kRtYw"
RESULT_SCHEMA = "assaysch_fFLKmdmG"
TITER_FIELD = "tsf_titer01"
LINK_FIELD = "tsf_runlink1"
UNIT_GL = "unit_J2USvfuP"
PROJECT_ID = "src_YzU5p4dR"
ENTRY_ID = "etr_Hds1XAaq"
RUN_SCHEMA = "assaysch_run0001"


@dataclass
class Call:
    method: str
    path: str
    params: dict[str, str]
    headers: dict[str, str]
    body: Any


@dataclass
class FakeBenchling:
    client_id: str = "app-client-id"
    client_secret: str = "app-client-secret-SHOULD-NOT-LEAK"  # noqa: S105 - test value
    token_lifetime: int = 900
    calls: list[Call] = field(default_factory=list)
    tokens_issued: list[str] = field(default_factory=list)
    delegated_tokens: dict[str, str] = field(default_factory=dict)  # token -> user id
    revoked: set[str] = field(default_factory=set)
    # scripted failures: path -> list of (status, headers, body) to return first
    script: dict[str, list[tuple[int, dict[str, str], Any]]] = field(default_factory=dict)
    # data
    studies: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    conditions: list[dict] = field(default_factory=list)
    replicates: list[dict] = field(default_factory=list)
    meis: list[dict] = field(default_factory=list)
    planned: list[dict] = field(default_factory=list)
    measured: list[dict] = field(default_factory=list)
    param_versions: list[dict] = field(default_factory=list)
    units: list[dict] = field(default_factory=list)
    result_schemas: list[dict] = field(default_factory=list)
    field_defs: dict[str, list[dict]] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    assay_runs: list[dict] = field(default_factory=list)
    assay_results: list[dict] = field(default_factory=list)
    v2_result_schemas: dict[str, dict] = field(default_factory=dict)
    # writes
    run_plans: list[dict] = field(default_factory=list)
    plan_conditions: list[dict] = field(default_factory=list)
    transactions: dict[str, dict] = field(default_factory=dict)
    committed_results: list[dict] = field(default_factory=list)
    blobs: list[dict] = field(default_factory=list)
    entry_patches: list[dict] = field(default_factory=list)
    _seq: int = 0

    # --- plumbing --------------------------------------------------------- #

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}_{self._seq:08d}"

    def paths(self, method: str | None = None) -> list[str]:
        return [c.path for c in self.calls if method is None or c.method == method]

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = {k: v[-1] for k, v in parse_qs(request.url.query.decode()).items()}
        body: Any = None
        if request.content:
            ctype = request.headers.get("content-type", "")
            if "json" in ctype:
                body = json.loads(request.content)
            else:
                body = {k: v[-1] for k, v in parse_qs(request.content.decode()).items()}
        self.calls.append(Call(request.method, path, params, dict(request.headers), body))
        scripted = self.script.get(path)
        if scripted:
            status, headers, payload = scripted.pop(0)
            return httpx.Response(status, headers=headers, json=payload)
        if path == "/oauth/token":
            return self._token(body or {})
        auth = request.headers.get("authorization", "")
        token = auth.removeprefix("Bearer ")
        if token in self.revoked or (token not in self.tokens_issued and token not in self.delegated_tokens):
            return _problem(401, "Unauthorized")
        return self._route(request.method, path, params, body, token)

    def _token(self, form: dict) -> httpx.Response:
        grant = form.get("grant_type")
        if form.get("client_id") != self.client_id or form.get("client_secret") != self.client_secret:
            return httpx.Response(401, json={"error": "invalid_client"})
        if grant == "client_credentials":
            tok = f"app-token-{len(self.tokens_issued) + 1}"
            self.tokens_issued.append(tok)
            return httpx.Response(200, json={"token_type": "Bearer", "access_token": tok,
                                             "expires_in": self.token_lifetime})
        if grant == "refresh_token":
            tok = f"user-token-r{len(self.delegated_tokens) + 1}"
            self.delegated_tokens[tok] = "ent_refreshed"
            return httpx.Response(200, json={
                "token_type": "Bearer", "access_token": tok, "expires_in": self.token_lifetime,
                "refresh_token": f"refresh-{len(self.delegated_tokens)}",
                "refresh_token_expires_in": 2592000})
        return httpx.Response(400, json={"error": "unsupported_grant_type"})

    # --- routing ---------------------------------------------------------- #

    def _route(self, method: str, path: str, params: dict, body: Any, token: str) -> httpx.Response:
        user = self.delegated_tokens.get(token)
        creator = {"__typename": "User" if user else "App", "id": user or "app_XXXXXXXXXXXXXXXX"}
        v3 = "/api/v3"
        if method == "GET":
            if path == f"{v3}/study/items":
                return self._page(self.studies, params)
            if path == "/api/v2/projects":
                return self._page(self.projects, params, key="projects")
            if path == f"{v3}/procedure-run/items":
                items = [r for r in self.runs if r["_study"] == params.get("study.eq")]
                items = _modified_filter(items, params)
                if params.get("sort") == "modifiedAt:asc":
                    items = sorted(items, key=lambda r: r["modifiedAt"])
                return self._page(items, params)
            if path.startswith(f"{v3}/procedure-run/") and path.endswith("/conditions/items"):
                rid = path.split("/")[4]
                return _ok({"items": [_pub(c) for c in self.conditions if c["procedureRun"]["id"] == rid]})
            if path.startswith(f"{v3}/procedure-run/") and path.endswith("/method-execution-instances/items"):
                rid = path.split("/")[4]
                return _ok({"items": [_pub(m) for m in self.meis if m["procedureRun"]["id"] == rid]})
            if path == f"{v3}/procedure-run-condition-replicate/items":
                return self._page(_any_of(self.replicates, params), params)
            if path == f"{v3}/procedure-parameter-planned-value/items":
                return self._page(_mei_filter(self.planned, params), params)
            if path == f"{v3}/procedure-parameter-measured-value/items":
                return self._page(_mei_filter(self.measured, params), params)
            if path == f"{v3}/procedure-parameter-version/items":
                return self._page(_any_of(self.param_versions, params), params)
            if path == f"{v3}/unit/items":
                return self._page(_any_of(self.units, params), params)
            if path == f"{v3}/result-schema/items":
                return self._page(self.result_schemas, params)
            if path.startswith(f"{v3}/result-schema/") and path.endswith("/field-definitions/items"):
                sid = path.split("/")[4]
                return _ok({"items": self.field_defs.get(sid, [])})
            if path == f"{v3}/result/items":
                items = [r for r in self.results if r["schema"]["id"] == params.get("schema.eq")]
                if "schemaFieldLinkValue.linkedItemIds" in params:
                    wanted = set(params["schemaFieldLinkValue.linkedItemIds"].split(","))
                    assert len(wanted) <= 1000, "linkedItemIds maxItems is 1000"
                    items = [r for r in items if wanted & _result_links(r)]
                items = _modified_filter(items, params)
                return self._page(items, params)
            if path == "/api/v2/assay-runs":
                items = [r for r in self.assay_runs if r["schema"]["id"] == params.get("schemaId")]
                return self._page(items, params, key="assayRuns")
            if path == "/api/v2/assay-results":
                items = [r for r in self.assay_results if r["schema"]["id"] == params.get("schemaId")]
                return self._page(items, params, key="assayResults")
            if path.startswith("/api/v2/assay-result-schemas/"):
                sid = path.rsplit("/", 1)[1]
                if sid in self.v2_result_schemas:
                    return _ok(self.v2_result_schemas[sid])
                return _v2_error(404, "Not found")
            if path.startswith("/api/v2/assay-runs/"):
                rid = path.rsplit("/", 1)[1]
                for r in self.assay_runs:
                    if r["id"] == rid:
                        return _ok(r)
                return _v2_error(404, "Not found")
            if path.startswith(f"{v3}/run/"):
                return _ok({"__typename": "Run", "id": path.rsplit("/", 1)[1]})
        if method == "POST":
            if path == f"{v3}/procedure-run-plan":
                if set(body) - {"studyId", "description", "workflowFlowchartConfigVersionId",
                                "dataEntryMode", "confirmationMethodByParamVersionId"}:
                    return _problem(400, "Unknown field")
                if not body.get("studyId") or not body.get("description"):
                    return _problem(400, "studyId and description are required")
                if len(body["description"]) > 255:
                    return _problem(400, "description too long")
                plan = {"__typename": "ProcedureRunPlan", "id": self._id("prp"),
                        "study": {"id": body["studyId"]}, "description": body["description"],
                        "procedureRun": None, "creator": creator, "_body": body}
                self.run_plans.append(plan)
                return _ok(_pub(plan), 201)
            if path == f"{v3}/procedure-run-plan-condition:batch-create":
                items = body.get("items") or []
                if not 1 <= len(items) <= 25:
                    return _problem(400, "items must hold 1-25 conditions")
                out = []
                for it in items:
                    missing = {"name", "description", "procedureRunPlanId", "methodParametersPlan",
                               "replicates", "templateMethodVersionId"} - set(it)
                    if missing:
                        return _problem(400, f"missing {sorted(missing)}")
                    cond = {"__typename": "ProcedureRunPlanCondition", "id": self._id("prpc"),
                            **it, "creator": creator}
                    self.plan_conditions.append(cond)
                    out.append(cond)
                return _ok({"items": out}, 201)
            if path.endswith(":promote"):
                return _ok({"taskId": "task_promote"}, 202)
            if path == "/api/v2/result-transactions":
                tid = self._id("txn")
                self.transactions[tid] = {"results": [], "state": "open", "user": user}
                return _ok({"id": tid})
            if path.startswith("/api/v2/result-transactions/") and path.endswith("/results"):
                tid = path.split("/")[4]
                txn = self.transactions.get(tid)
                if txn is None or txn["state"] != "open":
                    return _v2_error(400, "transaction not open")
                ids = []
                for res in body["assayResults"]:
                    rid = self._id("res")
                    txn["results"].append({**res, "id": rid, "creator": creator})
                    ids.append(rid)
                return _ok({"assayResults": ids, "errors": None})
            if path.startswith("/api/v2/result-transactions/") and path.endswith(":commit"):
                tid = path.split("/")[4].split(":")[0]
                txn = self.transactions[tid]
                txn["state"] = "committed"
                self.committed_results.extend(txn["results"])
                return _ok({"id": tid})
            if path.startswith("/api/v2/result-transactions/") and path.endswith(":abort"):
                tid = path.split("/")[4].split(":")[0]
                self.transactions[tid]["state"] = "aborted"
                return _ok({"id": tid})
            if path == "/api/v2/blobs":
                data = base64.b64decode(body["data64"])
                if hashlib.md5(data).hexdigest() != body["md5"]:  # noqa: S324 - spec field
                    return _v2_error(400, "md5 mismatch")
                blob = {"id": self._id("blob"), "name": body["name"], "type": body["type"],
                        "mimeType": body.get("mimeType"), "uploadStatus": "COMPLETE",
                        "_size": len(data), "_creator": creator}
                self.blobs.append(blob)
                return _ok(_pub(blob))
        if method == "PATCH" and path.startswith("/api/v2/entries/"):
            eid = path.rsplit("/", 1)[1]
            self.entry_patches.append({"entry": eid, "body": body, "creator": creator})
            return _ok({"id": eid, "fields": body.get("fields", {})})
        return _problem(404, "Not Found")

    def _page(self, items: list[dict], params: dict, key: str = "items") -> httpx.Response:
        size = int(params.get("pageSize", 50))
        start = int(params.get("nextToken", "0") or 0)
        chunk = items[start:start + size]
        nxt = str(start + size) if start + size < len(items) else None
        out: dict[str, Any] = {key: [_pub(i) for i in chunk], "nextToken": nxt}
        if key != "items" and nxt is None:
            out["nextToken"] = ""
        return _ok(out)


def _pub(obj: dict) -> dict:
    return {k: v for k, v in obj.items() if not k.startswith("_")}


def _ok(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _problem(status: int, title: str) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "application/problem+json"},
                          content=json.dumps({"type": "about:blank", "title": title, "detail": None,
                                              "status": status, "instance": "/x"}).encode())


def _v2_error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error",
                                                  "userMessage": message}})


def _result_links(result: dict) -> set[str]:
    out = set()
    for f in result.get("schemaFields") or []:
        v = (f.get("value") or {}).get("value")
        for item in v if isinstance(v, list) else [v]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                out.add(item["id"])
    return out


def _modified_filter(items: list[dict], params: dict) -> list[dict]:
    if "modifiedAt.gt" in params:
        items = [r for r in items if r["modifiedAt"] > params["modifiedAt.gt"]]
    if "modifiedAt.gte" in params:
        items = [r for r in items if r["modifiedAt"] >= params["modifiedAt.gte"]]
    return items


def _any_of(items: list[dict], params: dict) -> list[dict]:
    if "id.anyOf" not in params:
        return items
    wanted = params["id.anyOf"].split(",")
    assert len(wanted) <= 100, "id.anyOf maxItems is 100"
    return [i for i in items if i["id"] in wanted]


def _mei_filter(items: list[dict], params: dict) -> list[dict]:
    wanted = params.get("methodExecutionInstance.anyOf", "").split(",")
    assert len(wanted) <= 100, "methodExecutionInstance.anyOf maxItems is 100"
    return [i for i in items if (i.get("methodExecutionInstance") or {}).get("id") in wanted]


# --- fixtures ------------------------------------------------------------------ #


def _typed(value: Any) -> dict:
    """The TypedValue shape the v3 spec documents for plan inputs
    (`{"type": "ft_float", "floatValue": 10.0, "value": "10.0"}`), which the fake
    also uses for read values."""
    if isinstance(value, str):
        return {"type": "ft_string", "value": value}
    return {"type": "ft_float", "floatValue": float(value), "value": str(value)}


PARAMS = [
    # (version id, name, unit, kind) - kind planned|measured
    ("schver_Rm4nTvWz", "Temperature", "unit_degC0001", "planned"),
    ("schver_pH000001", "pH setpoint", None, "planned"),
    ("schver_feed0001", "Feed rate", "unit_mLh00001", "planned"),
    ("schver_media001", "Medium", None, "planned"),
    ("schver_dooo0001", "Dissolved oxygen (measured)", "unit_pct000001", "measured"),
]


def bioprocess_study(
    fake: FakeBenchling, *, n_runs: int = 3, conditions_per_run: int = 4,
    replicates: int = 2, seed: int = 0, missing_target_every: int = 0,
    duplicate_planned_every: int = 0, text_target_every: int = 0,
) -> FakeBenchling:
    """Fill `fake` with one study: runs x conditions x replicates rows, each with
    one method execution instance holding planned values for 4 parameters
    (one categorical) and a measured value for one more, and a result per
    replicate in a result schema whose titer field carries a unit (g/L)."""
    rng = random.Random(seed)
    fake.studies.append({"__typename": "Study", "id": STUDY_ID, "name": "Fed-batch DoE 7",
                         "archived": False, "phase": "EXECUTION",
                         "modifiedAt": "2026-10-01T00:00:00+00:00"})
    fake.studies.append({"__typename": "Study", "id": "stdy_other0001", "name": "Seed train",
                         "archived": False, "phase": "DESIGN", "modifiedAt": "2026-09-01T00:00:00+00:00"})
    fake.projects.append({"id": PROJECT_ID, "name": "Process development", "archiveRecord": None})
    for vid, name, unit, _kind in PARAMS:
        fake.param_versions.append({
            "__typename": "ProcedureParameterVersion", "id": vid, "name": name,
            "procedureParameter": {"id": vid.replace("schver", "pparam")},
            "dataType": {"type": "dropdown" if name == "Medium" else "decimal",
                         "unit": {"id": unit} if unit else None}})
    fake.units.extend([
        {"__typename": "Unit", "id": UNIT_GL, "name": "gram per liter", "symbol": "g/L"},
        {"__typename": "Unit", "id": "unit_degC0001", "name": "degree Celsius", "symbol": "degC"},
        {"__typename": "Unit", "id": "unit_mLh00001", "name": "millilitre per hour", "symbol": "mL/h"},
        {"__typename": "Unit", "id": "unit_pct000001", "name": "percent", "symbol": "%"},
    ])
    fake.result_schemas.append({"__typename": "ResultSchema", "id": RESULT_SCHEMA, "name": "Titer",
                                "systemName": "titer_result", "archived": False})
    fake.field_defs[RESULT_SCHEMA] = [
        {"__typename": "DecimalFieldDefinition", "id": TITER_FIELD, "name": "Titer",
         "systemName": "titer", "unit": {"id": UNIT_GL}, "archived": False, "isRequired": True},
        {"__typename": "RunLinkFieldDefinition", "id": LINK_FIELD, "name": "Replicate",
         "systemName": "replicate", "archived": False, "isRequired": False, "isMulti": False},
        {"__typename": "TextFieldDefinition", "id": "tsf_kalos001", "name": "Kalos run",
         "systemName": "kalos_run_id", "archived": False, "isRequired": False},
    ]
    row = 0
    for r in range(n_runs):
        rid = f"prun_{r:06d}"
        fake.runs.append({"__typename": "ProcedureRun", "id": rid, "name": f"Run {r + 1}",
                          "_study": STUDY_ID, "archived": False, "procedureType": "RECIPE",
                          "procedureFlowchartConfigVersion": {"id": FLOWCHART_CV},
                          "modifiedAt": f"2026-10-01T00:{r % 60:02d}:00+00:00",
                          "createdAt": "2026-09-01T00:00:00+00:00"})
        for c in range(conditions_per_run):
            cid = f"prc_{r:04d}{c:04d}"
            fake.conditions.append({"__typename": "ProcedureRunCondition", "id": cid,
                                    "name": f"Condition {c + 1}", "procedureRun": {"id": rid}})
            temp = round(rng.uniform(28, 37), 1)
            ph = round(rng.uniform(6.4, 7.4), 2)
            feed = round(rng.uniform(1, 5), 2)
            medium = rng.choice(["CD-CHO", "ActiPro"])
            for k in range(replicates):
                row += 1
                repid = f"prcr_{r:04d}{c:04d}{k:02d}"
                meid = f"pcmei_{r:04d}{c:04d}{k:02d}"
                fake.replicates.append({"__typename": "ProcedureRunConditionReplicate", "id": repid,
                                        "replicateNumber": k + 1, "archived": False,
                                        "procedureRunCondition": {"id": cid}})
                fake.meis.append({"__typename": "ProcedureMethodExecutionInstance", "id": meid,
                                  "archived": False, "procedureRun": {"id": rid},
                                  "procedureRunConditionReplicate": {"id": repid},
                                  "templateMethodVersion": {"id": METHOD_VERSION}})
                for (vid, _n, _u, kind), value in zip(PARAMS, [temp, ph, feed, medium, None]):
                    if kind == "measured":
                        do = round(rng.uniform(20, 60), 1)
                        fake.measured.append({"__typename": "ProcedureParameterMeasuredValue",
                                              "id": f"ppmv_{meid}", "parameterVersion": {"id": vid},
                                              "methodExecutionInstance": {"id": meid},
                                              "value": _typed(do), "recordedAt": None})
                        continue
                    fake.planned.append({"__typename": "ProcedureParameterPlannedValue",
                                         "id": f"pppv_{meid}_{vid[-4:]}", "parameterVersion": {"id": vid},
                                         "methodExecutionInstance": {"id": meid}, "value": _typed(value)})
                    if duplicate_planned_every and row % duplicate_planned_every == 0 and vid == PARAMS[0][0]:
                        fake.planned.append({"__typename": "ProcedureParameterPlannedValue",
                                             "id": f"pppv_{meid}_dup", "parameterVersion": {"id": vid},
                                             "methodExecutionInstance": {"id": meid},
                                             "value": _typed(value + 1)})
                if missing_target_every and row % missing_target_every == 0:
                    continue
                titer = 1.5 + 0.1 * (temp - 30) - (ph - 7) ** 2 + 0.2 * feed + rng.gauss(0, 0.05)
                tval: dict = ({"__typename": "TextValue", "value": "n/a"}
                              if text_target_every and row % text_target_every == 0
                              else {"__typename": "DecimalValue", "value": f"{titer:.4f}",
                                    "numericValue": round(titer, 4)})
                fake.results.append({
                    "__typename": "Result", "id": f"result_{repid}", "archived": False,
                    "schema": {"id": RESULT_SCHEMA}, "project": {"id": PROJECT_ID},
                    "modifiedAt": "2026-10-02T00:00:00+00:00",
                    "schemaFields": [
                        {"__typename": "SchemaFieldValue", "id": f"sfv_{repid}_t",
                         "fieldDefinition": {"id": TITER_FIELD}, "value": tval},
                        {"__typename": "SchemaFieldValue", "id": f"sfv_{repid}_l",
                         "fieldDefinition": {"id": LINK_FIELD},
                         "value": {"__typename": "ObjectLinkValue",
                                   "value": {"__typename": "ProcedureRunConditionReplicate", "id": repid}}},
                    ]})
    return fake


def v2_assay(fake: FakeBenchling, *, n_runs: int = 12, seed: int = 1, multi_result_every: int = 0) -> FakeBenchling:
    """v2 shapes: assay runs (fields carry the inputs) + assay results linked to
    the run through an `assay_run_link` field. Run ids are UUIDs like the spec
    example `77af3205-65af-457f-87f5-75462b85075a`."""
    rng = random.Random(seed)
    fake.projects.append({"id": PROJECT_ID, "name": "Process development", "archiveRecord": None})
    fake.v2_result_schemas["assaysch_v2res001"] = {
        "id": "assaysch_v2res001", "name": "Yield", "type": "assay_result", "systemName": "yield_result",
        "fieldDefinitions": [
            {"id": "tsf_yield001", "name": "Yield", "type": "float", "isMulti": False, "isRequired": True,
             "unit": {"id": UNIT_GL, "name": "gram per liter", "symbol": "g/L"}},
            {"id": "tsf_run00001", "name": "Run", "type": "assay_run_link", "isMulti": False, "isRequired": False},
        ]}
    for i in range(n_runs):
        rid = f"77af3205-65af-457f-87f5-{i:012d}"
        temp = round(rng.uniform(28, 37), 1)
        ph = round(rng.uniform(6.4, 7.4), 2)
        fake.assay_runs.append({
            "id": rid, "schema": {"id": RUN_SCHEMA, "name": "Assay Run"}, "projectId": PROJECT_ID,
            "entryId": ENTRY_ID, "createdAt": "2021-05-06T13:43:25.752597+00:00", "isReviewed": False,
            "v3Id": f"run_{i:08d}",
            "fields": {
                "temperature": {"type": "float", "value": temp, "displayValue": str(temp), "isMulti": False},
                "ph": {"type": "float", "value": ph, "displayValue": str(ph), "isMulti": False},
                "operator": {"type": "text", "value": "alice", "displayValue": "alice", "isMulti": False},
            }})
        values = [1.0 + 0.1 * temp - (ph - 7) ** 2 + rng.gauss(0, 0.05)]
        if multi_result_every and i % multi_result_every == 0:
            values.append(values[0] + 0.3)
        for j, v in enumerate(values):
            fake.assay_results.append({
                "id": f"{rid[:-4]}{j:04d}-res", "schema": {"id": "assaysch_v2res001", "name": "Yield"},
                "projectId": PROJECT_ID, "entryId": ENTRY_ID,
                "fields": {
                    "yield": {"type": "float", "value": round(v, 4), "displayValue": f"{v:.4f}", "isMulti": False},
                    "run": {"type": "assay_run_link", "value": rid, "displayValue": rid, "isMulti": False},
                }})
    return fake
