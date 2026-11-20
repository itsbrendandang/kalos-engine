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

from fastapi import Depends, HTTPException
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
_CACHE_LOCK = threading.Lock()  # guards the dicts only; never held while connecting
_CACHE_MAX = 8
_OPEN_LOCKS: dict[tuple[str, str], threading.Lock] = {}
# A failed open is remembered for a short, growing backoff, so a down or
# misconfigured database costs one connection attempt per window instead of one
# per request (each attempt can take KALOS_DB_CONNECT_TIMEOUT seconds).
BACKOFF_FIRST_S = 5.0
BACKOFF_MAX_S = 300.0


@dataclass
class _Failure:
    error: Exception
    until: float
    attempts: int


_FAILURES: dict[tuple[str, str], _Failure] = {}


class RecordsStorageUnavailable(RuntimeError):
    """The records database could not be opened (cached for `retry_after` s).
    `cause` is the redacted storage error; it never carries the URL's password."""

    def __init__(self, cause: Exception, retry_after: float) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.retry_after = retry_after


def _key(state_dir: Path) -> tuple[str, str]:
    return (str(state_dir), os.environ.get(DATABASE_URL_ENV, "").strip())


def _open(state_dir: Path) -> tuple[Backend, Backend]:
    """Open both backends. Database driver errors are re-raised as
    `StorageConfigError` naming only their type (their text could name the host)."""
    from kalos.storage.backend import DB_ERRORS, StorageConfigError

    try:
        records = open_backend(state_dir)
    except DB_ERRORS as exc:
        raise StorageConfigError(
            f"the records database could not be opened ({type(exc).__name__})") from None
    campaigns = (
        records if records.dialect == SQLITE
        else Backend(BackendConfig(SQLITE), sqlite_path=state_dir / SQLITE_FILENAME)
    )
    return records, campaigns


def _backends(state_dir: Path) -> tuple[Backend, Backend]:
    """The cached backends for this state dir and URL. Raises
    `RecordsStorageUnavailable` while a recent failure is in its backoff."""
    key = _key(state_dir)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
        failure = _FAILURES.get(key)
        if failure is not None and time.monotonic() < failure.until:
            raise RecordsStorageUnavailable(failure.error, failure.until - time.monotonic())
        open_lock = _OPEN_LOCKS.setdefault(key, threading.Lock())
    with open_lock:  # one opener per key; other keys and cache hits are not blocked
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                return hit
            failure = _FAILURES.get(key)
            if failure is not None and time.monotonic() < failure.until:
                raise RecordsStorageUnavailable(failure.error, failure.until - time.monotonic())
        try:
            opened = _open(state_dir)
        except Exception as exc:
            attempts = (failure.attempts + 1) if failure is not None else 1
            wait = min(BACKOFF_FIRST_S * 2 ** (attempts - 1), BACKOFF_MAX_S)
            with _CACHE_LOCK:
                _FAILURES[key] = _Failure(exc, time.monotonic() + wait, attempts)
            log.warning("records storage unavailable (%s); next attempt in %.0fs",
                        type(exc).__name__, wait)
            raise RecordsStorageUnavailable(exc, wait) from None
        with _CACHE_LOCK:
            _FAILURES.pop(key, None)
            _CACHE[key] = opened
            evicted = []
            while len(_CACHE) > _CACHE_MAX:
                evicted.append(_CACHE.popitem(last=False)[1])
        for old_r, old_c in evicted:
            for b in {id(old_r): old_r, id(old_c): old_c}.values():
                b.close()
        return opened


def open_records_storage(state_dir: Path) -> None:
    """Engine startup (kalos/portal/app.py `_lifespan`): with
    `KALOS_DATABASE_URL` set, open the PostgreSQL backend now, which creates an
    empty schema or verifies the current one (pg_runner.ensure_schema). A
    foreign, damaged, older or newer schema, or an unreachable server, raises
    (`PgMigrationError` / `StorageConfigError`) and the boot fails. SQLite needs
    nothing here: `CampaignStore` has already migrated `portal.db`."""
    if not _key(state_dir)[1]:
        return
    with _CACHE_LOCK:
        _FAILURES.pop(_key(state_dir), None)  # a restart always tries again
    try:
        _backends(state_dir)
    except RecordsStorageUnavailable as exc:
        raise exc.cause from None


def reset_backends() -> None:
    """Close and forget every cached backend and failure (tests)."""
    with _CACHE_LOCK:
        _FAILURES.clear()
        items = list(_CACHE.values())
        _CACHE.clear()
    for r, c in items:
        for b in {id(r): r, id(c): c}.values():
            b.close()


def get_records(store: CampaignStore = Depends(get_campaign_store)) -> Records:
    try:
        records, campaigns = _backends(store.state_dir)
    except RecordsStorageUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="records storage is unavailable; try again later",
            headers={"Retry-After": str(max(1, int(exc.retry_after + 0.999)))},
        ) from None
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
