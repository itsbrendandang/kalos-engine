"""`/api/assist/sop*`: SOP drafts from campaign records (`kalos.assist.sop`).

Same rules as `kalos.portal.campaigns_api`: scope is checked before any id
is looked at; the tenant is always `principal.tenant`; a malformed, missing
or other tenant's campaign id gets `campaign_not_found_response()`; a
malformed, missing or other tenant's draft id gets one byte-identical 404.
Audit rows carry the action and the campaign id, never the draft text.
"""
from __future__ import annotations

import io
import re
import unicodedata
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from kalos.assist import sop
from kalos.assist.http import assist_disabled_response, provider_unavailable_response
from kalos.assist.llm import ChatModel, ModelUnavailable, get_chat_model
from kalos.portal.auth import READ, WRITE, Principal, require_scope
from kalos.portal.campaign import CampaignNotFound, CampaignStore, get_campaign_store
from kalos.portal.campaign_routes import campaign_not_found_response
from kalos.portal.campaigns_api import _audit, _valid_campaign_id

router = APIRouter()

_DRAFT_NOT_FOUND = {"error": "draft not found", "code": "not_found"}
MAX_TITLE_CHARS = 200
MAX_PLACEHOLDERS_PER_FILL = 64
_PLACEHOLDER_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _draft_not_found() -> JSONResponse:
    return JSONResponse(_DRAFT_NOT_FOUND, status_code=404)


def _draft_json(draft: sop.Draft, status_code: int = 200) -> JSONResponse:
    return JSONResponse(draft.to_dict(), status_code=status_code)


def _check_title(value: str) -> str:
    title = value.strip()
    if not 1 <= len(title) <= MAX_TITLE_CHARS:
        raise ValueError(f"title must be 1-{MAX_TITLE_CHARS} characters")
    if any(unicodedata.category(ch) in {"Cc", "Cs", "Zl", "Zp"} for ch in title):
        raise ValueError("title must not contain control characters")
    if "[[" in title or "]]" in title:
        raise ValueError("title must not contain [[ or ]]")
    return title


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_ids: list[Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]] | None = Field(
        default=None, min_length=1, max_length=sop.MAX_RECIPES
    )
    proposal_index: list[Annotated[int, Field(ge=0, le=10_000)]] | None = Field(
        default=None, min_length=1, max_length=sop.MAX_RECIPES
    )
    round: Annotated[int, Field(ge=0, le=100_000)] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class _DraftBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Omitted: the tenant's default campaign (as `/api/assist/next`).
    campaign_id: Annotated[str, StringConstraints(max_length=64)] | None = None
    source: _Source
    template: Literal["batch_recipe", "media_prep", "custom"]
    title: str | None = None

    @field_validator("title")
    @classmethod
    def _title_ok(cls, v: str | None) -> str | None:
        return None if v is None else _check_title(v)


class _ReviseBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class _FillBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, str] = Field(min_length=1, max_length=MAX_PLACEHOLDERS_PER_FILL)

    @field_validator("values")
    @classmethod
    def _names_ok(cls, v: dict[str, str]) -> dict[str, str]:
        if any(_PLACEHOLDER_NAME.fullmatch(k) is None for k in v):
            raise ValueError("placeholder names are lower-case words with underscores")
        return v


# --- routes -------------------------------------------------------------------- #


@router.post("/api/assist/sop/draft")
def draft_sop(
    body: _DraftBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Version 1 of a draft from this campaign's records. 201 with the draft."""
    campaign_id = body.campaign_id or store.default_campaign_id(tenant=principal.tenant)
    source = body.source.as_dict()
    if campaign_id is None or not _valid_campaign_id(campaign_id):
        response = campaign_not_found_response()
    elif len(source) != 1:
        response = JSONResponse(
            {"error": "source must hold exactly one of run_ids, proposal_index, round",
             "code": "invalid_source"},
            status_code=400,
        )
    else:
        try:
            store.campaign_info(tenant=principal.tenant, campaign_id=campaign_id)
            out = sop.draft(
                store, principal.tenant, campaign_id, source, body.template, body.title,
                principal.subject,
            )
            response = _draft_json(out, 201)
        except CampaignNotFound:
            response = campaign_not_found_response()
        except sop.SopError as exc:
            status = 409 if isinstance(exc, sop.LimitReached) else 400
            response = JSONResponse({"error": str(exc), "code": exc.code}, status_code=status)
    _audit(store, principal, "sop.draft", response.status_code,
           campaign_id if response.status_code != 404 else None)
    return response


@router.post("/api/assist/sop/{draft_id}/revise")
def revise_sop(
    draft_id: str,
    body: _ReviseBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
    chat_model: ChatModel | None = Depends(get_chat_model),
) -> JSONResponse:
    """The model edits the prose; the result is checked and stored as N+1, or
    refused (422, nothing stored)."""
    current = sop.load(store, principal.tenant, draft_id)
    if current is None:
        _audit(store, principal, "sop.revise", 404, None)
        return _draft_not_found()
    if chat_model is None:
        response = assist_disabled_response()
    else:
        try:
            out = sop.revise(store, principal.tenant, draft_id, body.instruction, chat_model, principal.subject)
            response = _draft_json(out)
        except sop.RevisionRefused as exc:
            response = JSONResponse(
                {"error": "the revision changed something a revision may not change",
                 "code": exc.code, "reasons": exc.reasons},
                status_code=422,
            )
        except ModelUnavailable:
            response = provider_unavailable_response()
        except sop.SopError as exc:
            status = 409 if isinstance(exc, (sop.VersionConflict, sop.LimitReached)) else 400
            response = JSONResponse({"error": str(exc), "code": exc.code}, status_code=status)
    _audit(store, principal, "sop.revise", response.status_code, current.campaign_id)
    return response


@router.post("/api/assist/sop/{draft_id}/fill")
def fill_sop(
    draft_id: str,
    body: _FillBody,
    principal: Principal = Depends(require_scope(WRITE)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """Replace `[[TO FILL: name]]` tokens with the lab's text; stored as N+1."""
    current = sop.load(store, principal.tenant, draft_id)
    if current is None:
        _audit(store, principal, "sop.fill", 404, None)
        return _draft_not_found()
    try:
        out = sop.fill(store, principal.tenant, draft_id, body.values, principal.subject)
        response = _draft_json(out)
    except sop.UnknownPlaceholder as exc:
        response = JSONResponse(
            {"error": "the draft holds no open placeholder with that name", "code": exc.code,
             "names": exc.names},
            status_code=400,
        )
    except sop.SopError as exc:
        status = 409 if isinstance(exc, (sop.VersionConflict, sop.LimitReached)) else 400
        response = JSONResponse({"error": str(exc), "code": exc.code}, status_code=status)
    _audit(store, principal, "sop.fill", response.status_code, current.campaign_id)
    return response


@router.get("/api/assist/sop")
def list_sops(
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> dict[str, Any]:
    """The caller's tenant's drafts, newest first."""
    return {"drafts": sop.list_drafts(store, principal.tenant)}


@router.get("/api/assist/sop/{draft_id}")
def get_sop(
    draft_id: str,
    version: int | None = Query(default=None, ge=1, le=1_000_000),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> JSONResponse:
    """One version of a draft (the latest by default)."""
    out = sop.load(store, principal.tenant, draft_id, version)
    if out is None:
        return _draft_not_found()
    return _draft_json(out)


@router.get("/api/assist/sop/{draft_id}/export")
def export_sop(
    draft_id: str,
    format: Literal["md", "docx"] = Query(default="md"),
    version: int | None = Query(default=None, ge=1, le=1_000_000),
    principal: Principal = Depends(require_scope(READ)),
    store: CampaignStore = Depends(get_campaign_store),
) -> Response:
    """The draft as a file: markdown, or .docx when python-docx is installed."""
    out = sop.load(store, principal.tenant, draft_id, version)
    if out is None:
        return _draft_not_found()
    filename = f"{out.draft_id}-v{out.version}.{format}"
    disposition = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if format == "md":
        return Response(
            out.markdown.encode("utf-8"), media_type="text/markdown; charset=utf-8",
            headers=disposition,
        )
    try:
        data = _to_docx(out.markdown)
    except ImportError:
        return JSONResponse(
            {"error": "the .docx export needs the python-docx package", "code": "docx_unavailable"},
            status_code=406,
        )
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=disposition,
    )


# --- .docx ----------------------------------------------------------------------- #


def _to_docx(markdown: str) -> bytes:
    """The draft's markdown as a Word document: headings, bullets, numbered
    steps and tables; everything else a paragraph. Imported lazily (the
    `assist` extra)."""
    import docx  # the python-docx package

    document = docx.Document()
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-*", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                width = max(len(r) for r in rows)
                table = document.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                for r, cells in enumerate(rows):
                    for c, text in enumerate(cells):
                        table.cell(r, c).text = text
            continue
        if line.startswith("# "):
            document.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            document.add_heading(line[3:], level=1)
        elif line.startswith("> "):
            document.add_paragraph(line[2:]).runs[0].italic = True
        elif line.startswith("- "):
            document.add_paragraph(line[2:], style="List Bullet")
        elif re.match(r"\d+\. ", line):
            document.add_paragraph(line.split(". ", 1)[1], style="List Number")
        elif line.strip():
            document.add_paragraph(line)
        i += 1
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


__all__ = ["router"]
