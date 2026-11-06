"""Write Kalos output back to Benchling. Drafts only.

- **Proposals -> a draft run plan (v3).** `POST /api/v3/procedure-run-plan`, then
  the conditions with `POST /api/v3/procedure-run-plan-condition:batch-create`
  (25 per call, the spec's `maxItems`). The plan is never promoted: promoting starts
  a run, and a person does that in Benchling. (The client also refuses any
  `:promote` path.) The plan's creator is whoever's token is used: the delegated
  user when one is given, else the app.
- **Recorded results -> Benchling results (v2, atomic).** `POST
  /api/v2/result-transactions`, `POST .../{id}/results` (100 per call, the
  documented limit), then `:commit`; any failure aborts the transaction, so either
  all results land or none. Only with a delegated token: Benchling then names the
  person in its audit log and `createdBy`, not the app.
- **Analysis report -> a blob on an entry (v2).** `POST /api/v2/blobs`, then `PATCH
  /api/v2/entries/{id}` setting a blob-link field to the new blob. Delegated token.

Inputs come from the campaign's last Benchling import: its column provenance maps
a Kalos column back to a Benchling parameter version (and the method it belongs
to), and its target record names the result schema and field. Each function
returns the Benchling ids it got back; on a failure part-way it raises `PushFailed`
carrying the ids already created, so the caller's audit row lists them too.
"""
from __future__ import annotations

import base64
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kalos.integrations.benchling.client import BenchlingClient, BenchlingError, api_id

V2 = "/api/v2"
V3 = "/api/v3"
MAX_CONDITIONS_PER_CALL = 25       # procedure-run-plan-condition:batch-create maxItems
MAX_CONDITIONS_PER_METHOD = 100    # "Each template method in a run plan can contain at most 100 conditions."
MAX_REPLICATES_PER_PLAN = 500      # "The total replicates across all conditions ... cannot exceed 500."
MAX_RESULTS_PER_CALL = 100         # "Create 1 or more Result: 100 per request"
MAX_DESCRIPTION = 255
MAX_REPORT_BYTES = 10 * 1024 * 1024


class PushRefused(BenchlingError):
    """Nothing was sent: the inputs cannot become a valid write."""


class DelegatedTokenRequired(PushRefused):
    """This write must carry the logged-in user's Benchling token, not the app's."""


class PushFailed(BenchlingError):
    """Benchling failed part-way. `created` holds the ids made before the failure."""

    def __init__(self, message: str, *, created: dict[str, Any], cause: BenchlingError) -> None:
        super().__init__(message, status=cause.status, title=cause.title)
        self.created = created
        self.cause = cause


def _creator(obj: Any) -> str | None:
    ref = obj.get("creator") if isinstance(obj, dict) else None
    if isinstance(ref, dict) and isinstance(ref.get("id"), str):
        return ref["id"]
    return None


# --- proposals -> draft run plan ----------------------------------------------------- #


@dataclass
class PlanPush:
    run_plan_id: str
    condition_ids: list[str]
    proposals_sent: int
    columns_sent: list[str]
    columns_not_plannable: list[str]
    creator: str | None
    identity: str
    ids: dict[str, Any] = field(default_factory=dict)


def _typed(value: Any) -> dict[str, Any] | None:
    """The plan TypedValue (spec `DataParameterValuePlanInput.value`)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return {"value": {"type": "ft_float", "floatValue": f, "value": repr(f)}}
    if isinstance(value, str) and value.strip():
        return {"value": {"type": "ft_string", "value": value.strip()}}
    return None


def _describe(proposal: Mapping[str, Any]) -> str:
    parts = ["Kalos proposal (model output, not a measured result)"]
    mode = proposal.get("mode")
    if isinstance(mode, str) and mode:
        parts.append(f"mode {mode[:32]}")
    pred, std = proposal.get("pred"), proposal.get("std")
    if isinstance(pred, (int, float)) and not isinstance(pred, bool) and math.isfinite(pred):
        text = f"predicted {pred:.4g}"
        if isinstance(std, (int, float)) and not isinstance(std, bool) and math.isfinite(std):
            text += f" +/- {std:.3g}"
        parts.append(text)
    return "; ".join(parts)[:MAX_DESCRIPTION]


def plan_conditions(
    proposals: Sequence[Mapping[str, Any]], provenance: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """The condition bodies (without `procedureRunPlanId`), the columns sent, and
    the recipe columns that cannot be planned (no planned-parameter provenance:
    measured-only parameters, columns typed in Kalos). One condition per
    (proposal, template method), one replicate each."""
    planned = {
        str(p["column"]): p for p in provenance
        if p.get("kind") == "planned_parameter" and p.get("template_method_version_ids")
    }
    sent: set[str] = set()
    not_plannable: set[str] = set()
    items: list[dict[str, Any]] = []
    for i, proposal in enumerate(proposals):
        recipe = proposal.get("recipe")
        if not isinstance(recipe, Mapping) or not recipe:
            raise PushRefused(f"proposal {i + 1} has no recipe")
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for column, value in recipe.items():
            src = planned.get(str(column))
            if src is None:
                not_plannable.add(str(column))
                continue
            typed = _typed(value)
            if typed is None:
                continue
            method = sorted(src["template_method_version_ids"])[0]
            by_method[method].append({"parameterId": src["benchling_id"], "value": typed})
            sent.add(str(column))
        for method, values in sorted(by_method.items()):
            items.append({
                "name": f"Kalos proposal {i + 1}",
                "description": _describe(proposal),
                "templateMethodVersionId": method,
                "methodParametersPlan": {"dataParameterValues": values, "materialInputValues": [],
                                         "equipmentValues": []},
                "replicates": [{"replicateNumber": 1}],
            })
    if not items:
        raise PushRefused("no proposal column maps to a planned Benchling parameter; import the study first")
    per_method: dict[str, int] = defaultdict(int)
    for it in items:
        per_method[it["templateMethodVersionId"]] += 1
    if max(per_method.values()) > MAX_CONDITIONS_PER_METHOD:
        raise PushRefused(f"a run plan holds at most {MAX_CONDITIONS_PER_METHOD} conditions per method")
    if sum(len(it["replicates"]) for it in items) > MAX_REPLICATES_PER_PLAN:
        raise PushRefused(f"a run plan holds at most {MAX_REPLICATES_PER_PLAN} replicates")
    return items, sorted(sent), sorted(not_plannable)


def push_proposals(
    client: BenchlingClient, *, study_id: str, proposals: Sequence[Mapping[str, Any]],
    provenance: Iterable[Mapping[str, Any]], description: str,
    flowchart_config_version_id: str | None = None,
) -> PlanPush:
    """Create a DRAFT run plan with one condition per proposal (per method).
    Never promoted."""
    if not proposals:
        raise PushRefused("there are no proposals to push")
    api_id(study_id)
    items, sent, not_plannable = plan_conditions(proposals, provenance)
    body: dict[str, Any] = {"studyId": study_id, "description": description[:MAX_DESCRIPTION]}
    if flowchart_config_version_id:
        body["workflowFlowchartConfigVersionId"] = flowchart_config_version_id
    plan = client.post(f"{V3}/procedure-run-plan", body)
    plan_id = plan.get("id") if isinstance(plan, dict) else None
    if not isinstance(plan_id, str):
        raise BenchlingError("Benchling did not return a run plan id")
    condition_ids: list[str] = []
    for start in range(0, len(items), MAX_CONDITIONS_PER_CALL):
        chunk = [{**it, "procedureRunPlanId": plan_id} for it in items[start:start + MAX_CONDITIONS_PER_CALL]]
        try:
            resp = client.post(f"{V3}/procedure-run-plan-condition:batch-create", {"items": chunk})
        except BenchlingError as exc:
            raise PushFailed(
                "Benchling refused the run plan conditions; the draft plan exists without all of them",
                created={"run_plan_id": plan_id, "condition_ids": condition_ids}, cause=exc,
            ) from None
        for it in (resp or {}).get("items", []):
            if isinstance(it, dict) and isinstance(it.get("id"), str):
                condition_ids.append(it["id"])
    return PlanPush(plan_id, condition_ids, len(proposals), sent, not_plannable, _creator(plan), client.mode,
                    {"run_plan_id": plan_id, "condition_ids": condition_ids})


# --- results -> Benchling results (atomic) --------------------------------------------- #


@dataclass
class ResultsPush:
    transaction_id: str
    result_ids: list[str]
    pushed: list[tuple[str, int, str]]   # (kalos run id, version, benchling result id)
    identity: str


def _require_delegated(client: BenchlingClient) -> None:
    if client.mode != "delegated":
        raise DelegatedTokenRequired(
            "writing results needs the logged-in user's Benchling token, so Benchling records the person"
        )


def push_results(
    client: BenchlingClient, *, schema_id: str, target_system_name: str, project_id: str | None,
    results: Sequence[Mapping[str, Any]], schema_fields: Iterable[str] = (),
) -> ResultsPush:
    """Write each `{run_id, version, value}` as one Benchling result in a single
    transaction. Adds `kalos_run_id` when the schema has a field of that name."""
    _require_delegated(client)
    if not results:
        raise PushRefused("there are no new recorded results to push")
    fields_known = set(schema_fields)
    bodies = []
    for r in results:
        value = r.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PushRefused("a recorded result is not a finite number")
        fields: dict[str, Any] = {target_system_name: {"value": float(value)}}
        if "kalos_run_id" in fields_known:
            fields["kalos_run_id"] = {"value": str(r["run_id"])}
        body: dict[str, Any] = {"schemaId": schema_id, "fields": fields}
        if project_id:
            body["projectId"] = project_id
        bodies.append(body)
    txn = client.post(f"{V2}/result-transactions", {})
    tid = txn.get("id") if isinstance(txn, dict) else None
    if not isinstance(tid, str):
        raise BenchlingError("Benchling did not return a result transaction id")
    tid = api_id(tid)
    ids: list[str] = []
    try:
        for start in range(0, len(bodies), MAX_RESULTS_PER_CALL):
            resp = client.post(f"{V2}/result-transactions/{tid}/results",
                               {"assayResults": bodies[start:start + MAX_RESULTS_PER_CALL]})
            got = (resp or {}).get("assayResults") or []
            if (resp or {}).get("errors") or len(got) != len(bodies[start:start + MAX_RESULTS_PER_CALL]):
                raise BenchlingError("Benchling did not accept every result in the transaction")
            ids.extend(str(x) for x in got)
        client.post(f"{V2}/result-transactions/{tid}:commit", {})
    except BenchlingError as exc:
        try:
            client.post(f"{V2}/result-transactions/{tid}:abort", {})
            aborted = True
        except BenchlingError:
            aborted = False
        raise PushFailed(
            "the result transaction was aborted; no result was written" if aborted
            else "the result transaction failed and could not be aborted; check Benchling",
            created={"transaction_id": tid, "aborted": aborted, "result_ids_before_abort": ids}, cause=exc,
        ) from None
    pushed = [(str(r["run_id"]), int(r["version"]), rid) for r, rid in zip(results, ids)]
    return ResultsPush(tid, ids, pushed, client.mode)


# --- report -> blob on an entry ------------------------------------------------------------ #


def attach_report(
    client: BenchlingClient, *, entry_id: str, field_name: str, content: bytes, name: str,
    mime_type: str = "application/json",
) -> dict[str, str]:
    """Upload `content` as a v2 blob and set the entry's blob-link field
    `field_name` to it. Delegated token only."""
    _require_delegated(client)
    entry_id = api_id(entry_id)
    if len(content) > MAX_REPORT_BYTES:
        raise PushRefused("the report is larger than a single-part blob upload allows here")
    blob = client.post(f"{V2}/blobs", {
        "name": name[:200], "type": "RAW_FILE", "mimeType": mime_type,
        "data64": base64.b64encode(content).decode(),
        "md5": hashlib.md5(content).hexdigest(),  # noqa: S324 - integrity field the API requires, not security
    })
    blob_id = blob.get("id") if isinstance(blob, dict) else None
    if not isinstance(blob_id, str):
        raise BenchlingError("Benchling did not return a blob id")
    try:
        client.patch(f"{V2}/entries/{entry_id}", {"fields": {field_name: {"value": blob_id}}})
    except BenchlingError as exc:
        raise PushFailed("the report was uploaded but could not be attached to the entry",
                         created={"blob_id": blob_id}, cause=exc) from None
    return {"blob_id": blob_id, "entry_id": entry_id}
