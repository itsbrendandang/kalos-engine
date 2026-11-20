"""Kalos portal - `POST /api/campaigns/{id}/design`: a cold-start exploration plan.

A space-filling plan of recipes for a campaign with no or few measured runs
(`kalos.portal.design_plan`, generator `kalos.core.design`). No model is used:
no row carries a predicted value or an interval, every row is labelled
"exploration design, no model yet", and the response says why the model is not
used. Runs already measured or in flight are augmented (gaps filled), never
ignored. `format=csv` returns the plan as a run sheet in the upload layout.
API reference: docs/CAMPAIGNS_API.md, "Exploration design"; method and default
size: docs/COLD_START_DESIGN.md.

Security rules (the `kalos.portal.campaigns_api` pattern): scope `write`,
checked before the id is looked at; the tenant is always `principal.tenant`;
malformed, missing and foreign ids give the byte-identical
`campaign_not_found_response()`; errors never echo caller input; an archived
campaign answers 409; the size is capped (at most 96 rows, 64 factors, 16
categorical columns of at most 64 levels; JSON body cap 1 MiB); every call
writes an audit row (sizes, method and format only, never the body).
"""
from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, StringConstraints
from pydantic import Strict as _Strict

from kalos.core.design import DesignError
from kalos.portal.auth import WRITE, Principal, require_scope
from kalos.portal.campaign import (
    CampaignError,
    CampaignNotFound,
    CampaignStore,
    get_campaign_store,
)
from kalos.portal.campaign_routes import (
    campaign_error_response,
    campaign_not_found_response,
)
from kalos.portal.campaigns_api import _audit, _valid_campaign_id
from kalos.portal.design_plan import (
    API_METHODS,
    DEFAULT_METHOD,
    DEFAULT_N,
    MAX_CENTRE_POINTS,
    MAX_DESIGN_ROWS,
    MAX_REPLICATE_SLOTS,
    campaign_design,
    design_csv,
)
from kalos.portal.setup_settings import MAX_FEATURE_NAME_CHARS, SetupInvalid, load_setup

router = APIRouter()

MAX_FACTORS = 64
MAX_CATEGORICALS = 16
MAX_LEVELS = 64
MAX_LEVEL_CHARS = 64

_Num = Annotated[float, _Strict(), AllowInfNan(False), Field(ge=-1e12, le=1e12)]
_Pos = Annotated[float, _Strict(), AllowInfNan(False), Field(gt=0, le=1e12)]
_Int = Annotated[int, _Strict()]
_Name = Annotated[str, _Strict(), StringConstraints(min_length=1, max_length=MAX_FEATURE_NAME_CHARS)]
_Level = Annotated[str, _Strict(), StringConstraints(min_length=1, max_length=MAX_LEVEL_CHARS)]
_Unit = Annotated[str, _Strict(), StringConstraints(min_length=1, max_length=32)]

_ARCHIVED_BODY = {
    "error": "the campaign is archived; it can be read but not changed", "code": "campaign_archived",
}


class _FactorSpec(BaseModel):
    """Per-factor declarations. `lo`/`hi` only for a factor the campaign setup
    does not bound (setup bounds are the one source of truth); `step` or
    `levels` (not both) snap values to real setpoints; `unit` enables the
    physical-range check (e.g. "°C", "%", "g/L")."""

    model_config = ConfigDict(extra="forbid")

    lo: _Num | None = None
    hi: _Num | None = None
    step: _Pos | None = None
    levels: list[_Num] | None = Field(default=None, min_length=1, max_length=MAX_LEVELS)
    unit: _Unit | None = None


class _DesignBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: Annotated[_Int, Field(ge=1, le=MAX_DESIGN_ROWS)] = DEFAULT_N
    method: Literal["sobol", "lhs_maximin"] = DEFAULT_METHOD  # type: ignore[assignment]
    seed: Annotated[_Int, Field(ge=0, le=2**31 - 1)] = 0
    centre_points: Annotated[_Int, Field(ge=0, le=MAX_CENTRE_POINTS)] = 0
    replicate_slots: Annotated[_Int, Field(ge=0, le=MAX_REPLICATE_SLOTS)] = 0
    factors: dict[_Name, _FactorSpec] = Field(default_factory=dict, max_length=MAX_FACTORS)
    categoricals: dict[_Name, Annotated[list[_Level], Field(min_length=1, max_length=MAX_LEVELS)]] = Field(
        default_factory=dict, max_length=MAX_CATEGORICALS,
    )
    target: _Name | None = None
    include_pending: Annotated[bool, _Strict()] = True
    format: Literal["json", "csv"] = "json"


assert set(API_METHODS) == {"sobol", "lhs_maximin"}


def _anonymized(analysis: dict[str, Any] | None) -> bool:
    run = (analysis or {}).get("run")
    return isinstance(run, dict) and bool(run.get("anonymize"))


def _alias_design(design: dict[str, Any], analysis: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Identifier-type column names and the levels of identifier-type and clone
    columns get the analysis's own pseudonyms (`anonymize=true` campaigns)."""
    from kalos.data.anonymizer import _hash
    from kalos.portal.analysis_parts.identifiers import _is_identifier_name

    ss = analysis.get("strain_summary")
    clone = ss.get("column") if isinstance(ss, dict) else None
    changed = False

    def col(name: str) -> str:
        nonlocal changed
        if _is_identifier_name(name):
            changed = True
            return f"col_{_hash(name)[:8]}"
        return name

    def level(column: str, value: Any) -> Any:
        nonlocal changed
        if isinstance(value, str) and (column == clone or _is_identifier_name(column)):
            changed = True
            return f"level_{_hash(column + chr(31) + value)[:8]}"
        return value

    out = dict(design)
    out["rows"] = [{**r, "recipe": {col(k): level(k, v) for k, v in r["recipe"].items()}} for r in design["rows"]]
    out["factors"] = [
        {**f, "name": col(f["name"]),
         **({"levels": [level(f["name"], v) for v in f["levels"]]} if f.get("kind") in ("categorical", "clone") else {}),
         **({"held_level": level(f["name"], f.get("held_level"))} if f.get("held_level") is not None else {})}
        for f in design["factors"]
    ]
    out["levels"] = {
        col(k): {**v,
                 "held_level": level(k, v.get("held_level")),
                 "design_counts": {level(k, lv): c for lv, c in (v.get("design_counts") or {}).items()},
                 "existing_counts": ({level(k, lv): c for lv, c in v["existing_counts"].items()}
                                     if v.get("existing_counts") is not None else None)}
        for k, v in design["levels"].items()
    }
    out["template"] = {**design["template"], "columns": [col(c) for c in design["template"]["columns"]],
                       "target": col(design["template"]["target"])}
    out["not_designed"] = [{**e, "column": col(e["column"])} for e in design.get("not_designed") or []]
    out["anonymized"] = True
    return out, changed


@router.post("/api/campaigns/{campaign_id}/design")
def design_campaign(
    campaign_id: str,
    body: _DesignBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> Response:
    """A cold-start exploration plan for this campaign: space-filling recipes,
    no model, no predicted values (see the module docstring)."""
    action = "campaign.design"
    detail = (f"n={body.n} centre={body.centre_points} replicates={body.replicate_slots} "
              f"method={body.method} format={body.format}")
    if not _valid_campaign_id(campaign_id):
        _audit(store, principal, action, 404, None)
        return campaign_not_found_response()
    tenant = principal.tenant
    try:
        info = store.campaign_info(tenant=tenant, campaign_id=campaign_id)
        if info.get("archived"):
            _audit(store, principal, action, 409, campaign_id)
            return JSONResponse(_ARCHIVED_BODY, status_code=409)
        try:
            setup_row = load_setup(store, tenant=tenant, campaign_id=campaign_id)
        except SetupInvalid as exc:
            _audit(store, principal, action, 409, campaign_id, detail="code=setup_unreadable")
            return JSONResponse(exc.to_body(), status_code=409)
        analysis = store.load_analysis(tenant=tenant, campaign_id=campaign_id)
        state = store.get(tenant=tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        _audit(store, principal, action, 404, campaign_id)
        return campaign_not_found_response()
    except CampaignError as exc:
        response = campaign_error_response(exc)
        _audit(store, principal, action, response.status_code, campaign_id)
        return response
    seeded_target = (state or {}).get("target")
    if body.target is not None and seeded_target is not None and body.target != seeded_target:
        _audit(store, principal, action, 400, campaign_id, detail="code=target_mismatch")
        return JSONResponse({"error": "target names a different column than the campaign's target; "
                                      "omit it for a campaign that has data", "code": "target_mismatch"},
                            status_code=400)
    try:
        design = campaign_design(
            setup=setup_row["value"] if setup_row else None, analysis=analysis, state=state,
            n=body.n, method=body.method, seed=body.seed, centre_points=body.centre_points,
            replicate_slots=body.replicate_slots,
            factors={k: v.model_dump() for k, v in body.factors.items()},
            categoricals=body.categoricals, include_pending=body.include_pending,
        )
    except DesignError as exc:
        _audit(store, principal, action, 400, campaign_id, detail=f"code={exc.code}")
        return JSONResponse(exc.to_body(), status_code=400)
    if body.target is not None and seeded_target is None:
        cols = design["template"]["columns"]
        old = design["template"]["target"]
        if body.target in cols and body.target != old:
            _audit(store, principal, action, 400, campaign_id, detail="code=target_is_factor")
            return JSONResponse({"error": "target names a design factor", "code": "target_is_factor"},
                                status_code=400)
        design["template"] = {**design["template"], "target": body.target,
                              "columns": [body.target if c == old else c for c in cols]}
    aliased = False
    if analysis and _anonymized(analysis):
        design, aliased = _alias_design(design, analysis)
    if body.format == "csv":
        if aliased:
            _audit(store, principal, action, 409, campaign_id, detail="code=anonymized_template_unavailable")
            return JSONResponse({
                "error": "this campaign is anonymized and its plan names identifier columns or clone labels, "
                         "so a run-sheet template with pseudonyms could not be uploaded back",
                "code": "anonymized_template_unavailable",
                "what_would_fix_it": "use the JSON plan, or re-analyze without anonymize",
            }, status_code=409)
        body_bytes = design_csv(design).encode("utf-8")
        _audit(store, principal, action, 200, campaign_id, detail=detail)
        return Response(body_bytes, media_type="text/csv; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="{campaign_id}-exploration-design-no-model.csv"',
            "X-Kalos-Design-Label": str(design["label"]),
            "X-Kalos-Design-Method": str(design["method"]),
            "X-Kalos-Design-Seed": str(design["seed"]),
            "X-Kalos-Content-Sha256": hashlib.sha256(body_bytes).hexdigest(),
            "X-Kalos-Csv-Formula-Neutralized": "apostrophe-prefix",
            "Cache-Control": "no-store",
        })
    _audit(store, principal, action, 200, campaign_id, detail=detail)
    return JSONResponse({"campaign_id": campaign_id, "design": design})
