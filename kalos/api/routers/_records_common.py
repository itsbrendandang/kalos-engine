"""Shared plumbing of the records routers (projects, lineage, units).

- `Records`: the two databases a records request may touch: `records` (the v9
  tables; PostgreSQL when `KALOS_DATABASE_URL` is set, else `portal.db`) and
  `campaigns` (`portal.db`, where the campaign and result tables still live until
  REORG_PLAN R12). On SQLite they are one database.
- the security rules of the campaign routers, applied the same way: the scope is
  checked by the dependency before any id is read; the tenant is always the
  verified token's (`principal.tenant`); malformed, missing and foreign ids get a
  byte-identical 404; errors never echo caller input; every successful write has
  its audit row in the SAME transaction, a refused write a best-effort one.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends
from fastapi.responses import JSONResponse

from kalos.api.request_context import current_request_id
from kalos.portal.auth import Principal
from kalos.portal.campaign import CampaignNotFound, CampaignStore, get_campaign_store
from kalos.portal.campaigns_api import _auth_method, _valid_campaign_id
from kalos.records.factors import PhysicalBounds
from kalos.storage import records_repo
from kalos.storage.backend import (
    DATABASE_URL_ENV,
    SQLITE,
    SQLITE_FILENAME,
    Backend,
    BackendConfig,
    open_backend,
)

log = logging.getLogger("kalos.records")

PROJECT_NOT_FOUND = {"error": "project not found", "code": "project_not_found"}
LINK_NOT_FOUND = {"error": "lineage link not found", "code": "link_not_found"}


@dataclass
class Records:
    records: Backend
    campaigns: Backend
    store: CampaignStore

    @property
    def split(self) -> bool:
        """True when the records live in another database than the campaigns."""
        return self.records is not self.campaigns


_CACHE: OrderedDict[tuple[str, str], tuple[Backend, Backend]] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 8


def _backends(state_dir: Path) -> tuple[Backend, Backend]:
    key = (str(state_dir), os.environ.get(DATABASE_URL_ENV, "").strip())
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
        records = open_backend(state_dir)
        campaigns = (
            records if records.dialect == SQLITE
            else Backend(BackendConfig(SQLITE), sqlite_path=state_dir / SQLITE_FILENAME)
        )
        _CACHE[key] = (records, campaigns)
        while len(_CACHE) > _CACHE_MAX:
            _, (old_r, old_c) = _CACHE.popitem(last=False)
            for b in {id(old_r): old_r, id(old_c): old_c}.values():
                b.close()
        return records, campaigns


def reset_backends() -> None:
    """Close and forget every cached backend (tests)."""
    with _CACHE_LOCK:
        while _CACHE:
            _, (r, c) = _CACHE.popitem()
            for b in {id(r): r, id(c): c}.values():
                b.close()


def get_records(store: CampaignStore = Depends(get_campaign_store)) -> Records:
    records, campaigns = _backends(store.state_dir)
    return Records(records=records, campaigns=campaigns, store=store)


def error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": message, "code": code, **extra}, status_code=status)


def not_found_project() -> JSONResponse:
    return JSONResponse(PROJECT_NOT_FOUND, status_code=404)


def actor(principal: Principal) -> str | None:
    return principal.subject[:256] if isinstance(principal.subject, str) else None


def audit_ok(conn: Any, principal: Principal, action: str, *, detail: str,
             campaign_id: str | None = None, now: float | None = None) -> None:
    """The success audit row, in the caller's write transaction."""
    records_repo.insert_audit(
        conn, tenant=principal.tenant, action=action, outcome="ok", subject=principal.subject,
        auth_method=_auth_method(principal), now=now if now is not None else time.time(),
        campaign_id=campaign_id, request_id=current_request_id(), detail=detail,
    )


def audit_refused(rec: Records, principal: Principal, action: str, status: int, *,
                  code: str, campaign_id: str | None = None) -> None:
    """Best-effort audit of a refused write (never fails the request). Only
    the refusal code is kept, never what was sent."""
    try:
        with rec.records.txn(write=True) as conn:
            records_repo.insert_audit(
                conn, tenant=principal.tenant, action=action, outcome=f"http_{status}",
                subject=principal.subject, auth_method=_auth_method(principal), now=time.time(),
                campaign_id=campaign_id, request_id=current_request_id(), detail=f"code={code}",
            )
    except Exception:  # noqa: BLE001 - auditing a refusal must not turn it into a 500
        log.warning("records: could not audit a refused %s", action)


def campaign_info(store: CampaignStore, tenant: str, campaign_id: str) -> dict[str, Any] | None:
    """The caller's campaign, or None when the id is malformed, missing or
    another tenant's (one answer for all three)."""
    if not isinstance(campaign_id, str) or not _valid_campaign_id(campaign_id):
        return None
    try:
        return store.campaign_info(tenant=tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        return None


def current_stage(store: CampaignStore, tenant: str, campaign_id: str) -> str | None:
    """The campaign's RECORDED current stage key (never computed), or None."""
    from kalos.portal.stages import load_stages

    try:
        doc = load_stages(store, tenant=tenant, campaign_id=campaign_id)
    except CampaignNotFound:
        return None
    stage = doc.get("current_stage") if isinstance(doc, dict) else None
    return stage if isinstance(stage, str) else None


def physical_bounds(name: str, unit: str | None) -> PhysicalBounds | None:
    """The engine's physical-bounds table for a factor (kalos.validation.bounds):
    `column_bounds` when the unit-aware table is present (branch feat/data-checks),
    else the name-hint dimensions on main. None when nothing is known."""
    from kalos.validation import bounds as vb

    header = f"{name} ({unit})" if unit else name
    column_bounds = getattr(vb, "column_bounds", None)
    if callable(column_bounds):
        cb = column_bounds(header)
        if cb is None:
            return None
        return PhysicalBounds(float(cb.hard_lo), float(cb.hard_hi), str(cb.source))
    dim = vb.infer_dimension(header)
    if dim is None:
        return None
    b = vb.DIMENSION_BOUNDS[dim]
    words = dim.replace("_", " ")
    # A non-negative floor holds in any unit; a two-sided range only in the
    # dimension's base unit (a factor in another unit is not judged rather than
    # judged wrongly).
    if b.hard_lo == 0.0 and b.hard_hi == float("inf"):
        return PhysicalBounds(0.0, float("inf"), f"a {words} cannot be negative")
    base_units: dict[str, tuple[str | None, ...]] = {
        "ph": (None, "pH"), "percent": ("%",), "temperature_c": ("°C",), "od": (None, "OD"),
    }
    if unit not in base_units.get(dim, ()):
        return None
    return PhysicalBounds(float(b.hard_lo), float(b.hard_hi), f"read as {words}")
