"""Campaign-store loading for campaign setup documents (campaigns layer)."""
from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import Any

from kalos.portal.campaign import CampaignNotFound
from kalos.portal.campaign_settings import get_setting
from kalos.portal.clone_registry import lead_clone_label
from kalos.portal.setup_settings import SetupInvalid, analysis_setup, normalize_setup

log = logging.getLogger("kalos.portal")


def load_setup(store: Any, *, tenant: str, campaign_id: str) -> dict[str, Any] | None:
    """The stored setting row `{value (normalized), updated_at, updated_by}` or
    None when never set. Raises `CampaignNotFound`; raises `SetupInvalid` when the
    stored document no longer normalizes (fail closed, never silently ignored)."""
    row = get_setting(store, tenant=tenant, campaign_id=campaign_id, kind="setup")
    if row is None:
        return None
    value = row.get("value")
    if not isinstance(value, Mapping):
        raise SetupInvalid("setup_unreadable", "the stored setup could not be read")
    return {**row, "value": normalize_setup(value)}


def setup_for_analysis(
    store: Any, *, tenant: str, campaign_id: str | None
) -> dict[str, Any] | None:
    """The `_analyze(setup=...)` mapping for a campaign (`campaign_id=None` = the
    tenant's default campaign), or None when nothing is stored."""
    cid = campaign_id or store.default_campaign_id(tenant=tenant)
    if cid is None:
        return None
    try:
        row = load_setup(store, tenant=tenant, campaign_id=cid)
    except CampaignNotFound:
        return None
    except SetupInvalid:
        log.warning("stored campaign setup is unreadable; proposals will be withheld")
        return {"unreadable": True, "hash": None}
    if row is None:
        return None
    out = analysis_setup(row["value"])
    if out.get("strain_column") and not out.get("strain_level"):
        try:
            lead = lead_clone_label(store, tenant=tenant, campaign_id=cid)
        except CampaignNotFound:
            lead = None
        if lead:
            out["strain_lead"] = lead
    return out


def with_campaign_setup(
    analyze: Any, store: Any, *, tenant: str, campaign_id: str | None
) -> Any:
    """`analyze` (None = `kalos.portal.analysis._analyze`) bound to the campaign's
    setup, or `analyze` unchanged when the campaign has no stored setup."""
    setup = setup_for_analysis(store, tenant=tenant, campaign_id=campaign_id)
    if setup is None:
        return analyze
    if analyze is None:
        from kalos.portal.analysis import _analyze as analyze
    return functools.partial(analyze, setup=setup)


__all__ = ["load_setup", "setup_for_analysis", "with_campaign_setup"]
