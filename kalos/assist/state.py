"""Reading a campaign's record for the assistant, and the refusal note.

A refused upload (a 400 with a code, e.g. `too_many_fit_rows`) leaves no trace
in the campaign: nothing is stored. So the guide could not say "fix the sheet"
after the user navigates away. `record_refusal` keeps the latest refusal code
and its numbers (never the file, its name, or any cell) in the assistant's own
folder, and a later successful upload clears it. Failed background jobs carry
their code in the jobs table already and are read from there.

Neither is campaign state: the campaign record is never written here.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from kalos.assist.paths import tenant_dir
from kalos.portal.migrate import CAMPAIGN_ID_PATTERN

log = logging.getLogger("kalos.assist")

# Context keys a refusal may carry into the note: counts and limits only.
_SAFE_CONTEXT = ("n_rows", "max_fit_rows", "min_rows", "max_rows", "n_dropped_incomplete",
                 "n_censored_excluded")


def _refusal_path(store: Any, tenant: str, campaign_id: str) -> Any:
    if CAMPAIGN_ID_PATTERN.fullmatch(campaign_id) is None:
        raise ValueError("bad campaign id")
    return tenant_dir(store.state_dir, tenant) / "refusals" / f"{campaign_id}.json"


def record_refusal(store: Any, tenant: str, campaign_id: str | None, body: dict[str, Any] | None) -> None:
    """Keep the latest refusal for `campaign_id` (its tenant default when None).
    Best-effort; never raises."""
    try:
        cid = campaign_id or store.default_campaign_id(tenant=tenant)
        body = body or {}
        code = body.get("code")
        if not cid or not isinstance(code, str) or not code:
            return
        context = {k: body[k] for k in _SAFE_CONTEXT if isinstance(body.get(k), int)}
        path = _refusal_path(store, tenant, cid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"code": code[:64], "context": context, "at": time.time()}))
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - a note for the guide, never fails the request
        log.exception("assist: could not keep the refusal note")


def clear_refusal(store: Any, tenant: str, campaign_id: str | None) -> None:
    try:
        cid = campaign_id or store.default_campaign_id(tenant=tenant)
        if cid:
            _refusal_path(store, tenant, cid).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        log.exception("assist: could not clear the refusal note")


def _job_refusal(store: Any, tenant: str, campaign_id: str) -> dict[str, Any] | None:
    try:
        from kalos.portal import jobs

        rows = jobs.list_jobs(store, tenant=tenant, campaign_id=campaign_id, limit=1)
    except Exception:  # noqa: BLE001 - no jobs table, or no jobs: nothing to report
        return None
    if not rows or rows[0].get("state") != "failed" or not rows[0].get("error"):
        return None
    finished = rows[0].get("finished_at")
    return {"code": rows[0]["error"].get("code"), "context": {},
            "at": float(finished) if isinstance(finished, (int, float)) else None}


def latest_refusal(store: Any, tenant: str, campaign_id: str) -> dict[str, Any] | None:
    """The newest of the refusal note and the latest failed job, or None."""
    found: list[dict[str, Any]] = []
    try:
        data = json.loads(_refusal_path(store, tenant, campaign_id).read_text())
        if isinstance(data, dict) and data.get("code"):
            found.append(data)
    except (OSError, ValueError):
        pass
    job = _job_refusal(store, tenant, campaign_id)
    if job and job.get("code") and job.get("at") is not None:
        found.append(job)
    if not found:
        return None
    return max(found, key=lambda r: float(r.get("at") or 0))


def campaign_inputs(store: Any, tenant: str, campaign_id: str | None) -> tuple[
    str | None, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None,
]:
    """(campaign id, summary, analysis, refusal) for the guide. `campaign_id`
    None means the tenant default (which may not exist yet). Raises
    CampaignNotFound for an id that is not this tenant's."""
    cid = campaign_id or store.default_campaign_id(tenant=tenant)
    if cid is None:
        return None, {"has_campaign": False}, None, None
    analysis = store.load_analysis(tenant=tenant, campaign_id=cid)
    summary = store.summary(
        tenant=tenant, campaign_id=cid,
        analysis_generation=(analysis or {}).get("campaign_generation"),
    )
    return cid, summary, analysis, latest_refusal(store, tenant, cid)


__all__ = ["campaign_inputs", "clear_refusal", "latest_refusal", "record_refusal"]
