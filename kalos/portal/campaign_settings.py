"""Per-campaign settings documents (schema v6 `campaign_settings`).

One JSON document per (campaign, kind). Kinds:

- ``setup``: the campaign's goal and operating boundaries (goal direction,
  user-entered bounds, linear constraints, target unit). Owned by
  `kalos.portal.setup_api`.
- ``stages``: the campaign's editable scale-up stage definitions. Owned by
  `kalos.portal.gates_api`.

Settings live OUTSIDE the campaign state blob, so editing them never mints a
new campaign generation and never races a fold. Every statement carries
``tenant = ?`` and resolves the campaign with ``WHERE id = ? AND tenant = ?``
first, so another tenant's id raises `CampaignNotFound` exactly like a missing id.
"""
from __future__ import annotations

import json
import time
from typing import Any

from kalos.portal.campaign import CampaignNotFound, CampaignStore

SETTING_KINDS = ("setup", "stages")
# A settings document is small structured data (bounds, constraints, stage
# definitions). Anything larger is refused before it is written.
MAX_SETTING_JSON_CHARS = 64 * 1024


class SettingTooLarge(ValueError):
    """The serialized settings document exceeds MAX_SETTING_JSON_CHARS."""


def _check_kind(kind: str) -> None:
    if kind not in SETTING_KINDS:
        raise ValueError(f"unknown settings kind; expected one of {SETTING_KINDS}")


def get_setting(
    store: CampaignStore, *, tenant: str, campaign_id: str, kind: str
) -> dict[str, Any] | None:
    """The stored document `{"value", "updated_at", "updated_by"}` or None when
    never set. Raises `CampaignNotFound` for a missing or foreign campaign."""
    _check_kind(kind)
    store._check(tenant, campaign_id)
    with store._txn(write=False) as conn:
        if store._row(conn, tenant, campaign_id) is None:
            raise CampaignNotFound()
        row = conn.execute(
            "SELECT value, updated_at, updated_by FROM campaign_settings "
            "WHERE campaign_id = ? AND tenant = ? AND kind = ?",
            (campaign_id, tenant, kind),
        ).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row[0])
    except ValueError:
        return None
    return {"value": value, "updated_at": row[1], "updated_by": row[2]}


def put_setting(
    store: CampaignStore, *, tenant: str, campaign_id: str, kind: str,
    value: dict[str, Any], updated_by: str | None, audit_action: str | None = None,
    auth_method: str | None = None, audit_detail: str | None = None,
) -> dict[str, Any]:
    """Replace the document for (campaign, kind). Refuses archived campaigns
    (`CampaignArchived`) and missing/foreign ones (`CampaignNotFound`). When
    `audit_action` is given, the audit row is written in the SAME transaction,
    so an unauditable change is not made."""
    from kalos.portal.campaign import CampaignArchived

    _check_kind(kind)
    store._check(tenant, campaign_id)
    blob = json.dumps(value, sort_keys=True, allow_nan=False)
    if len(blob) > MAX_SETTING_JSON_CHARS:
        raise SettingTooLarge(f"settings document exceeds {MAX_SETTING_JSON_CHARS} characters")
    now = time.time()
    who = updated_by[:256] if isinstance(updated_by, str) else None
    with store._txn(write=True) as conn:
        row = store._row(conn, tenant, campaign_id)
        if row is None:
            raise CampaignNotFound()
        if row.archived_at is not None:
            raise CampaignArchived()
        conn.execute(
            "INSERT INTO campaign_settings (campaign_id, tenant, kind, value, updated_at, updated_by) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(campaign_id, tenant, kind) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at, "
            "updated_by = excluded.updated_by WHERE campaign_settings.tenant = excluded.tenant",
            (campaign_id, tenant, kind, blob, now, who),
        )
        if audit_action is not None:
            store._insert_audit(
                conn, tenant=tenant, action=audit_action, outcome="ok", subject=updated_by,
                auth_method=auth_method, campaign_id=campaign_id, detail=audit_detail, now=now,
            )
    return {"value": value, "updated_at": now, "updated_by": who}
