"""Called by the portal after an upload, a recorded result, or a re-analysis.

- A refused upload/re-analysis (a 4xx whose body carries a `code`) leaves a
  refusal note for the guide (`kalos.assist.state.record_refusal`).
- A successful upload clears that note.
- Any success refreshes the tenant's index for that one campaign, if the
  index exists yet (`kalos.assist.index.refresh_quietly`).

Best-effort and never raises: the assistant's files are derived data and must
not fail or slow the request that changed the campaign beyond a quick refresh.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("kalos.assist")


def after_write(store: Any, tenant: str, campaign_id: str | None, response: Any, *, kind: str) -> None:
    """`kind` is "upload", "result" or "reanalyze"."""
    try:
        from kalos.assist.index import refresh_quietly
        from kalos.assist.state import clear_refusal, record_refusal

        status = int(getattr(response, "status_code", 500))
        if 400 <= status < 500 and kind in ("upload", "reanalyze"):
            try:
                body = json.loads(bytes(response.body).decode("utf-8"))
            except (ValueError, AttributeError, TypeError):
                body = None
            if isinstance(body, dict) and body.get("code") not in (None, "not_found", "campaign_has_runs",
                                                                  "conflict", "campaign_archived"):
                record_refusal(store, tenant, campaign_id, body)
                refresh_quietly(store, tenant, campaign_id)
            return
        if status < 300:
            if kind == "upload":
                clear_refusal(store, tenant, campaign_id)
            refresh_quietly(store, tenant, campaign_id)
    except Exception:  # noqa: BLE001 - never fails the audited request
        log.exception("assist: after-write hook failed")


__all__ = ["after_write"]
