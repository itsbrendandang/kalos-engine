"""Publish an upload analysis and seed its campaign atomically (campaigns layer)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

from kalos.portal.campaign import CampaignError
from kalos.portal.pipeline import stored_dataset_label
from kalos.portal.uploads import scrubbed_failure

log = logging.getLogger("kalos.portal")

_ANY = object()


def publish_upload_analysis(
    store: Any, result: dict, filename: str, *, tenant: str, campaign_id: str | None,
    df: pd.DataFrame | None = None, base_rows: list[dict[str, Any]] | None = None,
    expected_generation: Any = _ANY, on_commit: Callable[[Any, str], None] | None = None,
    fallback_unstamped: bool = True,
) -> str | None:
    """Seed the campaign with the upload and publish `result` in the same
    transaction (docs/CAMPAIGN_LOOP.md, "Seeding"). Returns the new generation,
    or None if only the unstamped fallback was published."""
    filename = stored_dataset_label(result, filename)
    seed_kwargs: dict[str, Any] = {}
    if expected_generation is not _ANY:
        seed_kwargs["expected_generation"] = expected_generation
    if on_commit is not None:
        seed_kwargs["on_commit"] = on_commit
    if base_rows is not None:
        seed_kwargs["base_rows"] = base_rows
    try:
        return store.seed(
            df, result["target"], result["proposal_features"], tenant=tenant,
            campaign_id=campaign_id, analysis=result, dataset=filename, **seed_kwargs,
        )
    except CampaignError as exc:
        if campaign_id is not None or not fallback_unstamped:
            raise
        log.error("failed to seed the campaign from an uploaded run sheet: %s", scrubbed_failure(exc))
        _publish_unstamped(store, result, filename, tenant=tenant)
    except Exception as exc:  # noqa: BLE001 - seeding must never fail a sync upload
        if not fallback_unstamped:
            raise
        log.error("failed to seed the campaign from an uploaded run sheet: %s", scrubbed_failure(exc))
        _publish_unstamped(store, result, filename, tenant=tenant, campaign_id=campaign_id)
    return None


def _publish_unstamped(
    store: Any, result: dict, filename: str, *, tenant: str, campaign_id: str | None = None,
) -> None:
    try:
        store.save_analysis(
            result, filename, tenant=tenant, campaign_id=campaign_id, campaign_generation=None,
        )
    except Exception as exc:  # noqa: BLE001 - publishing is best-effort too
        log.error("failed to publish the analysis of an uploaded run sheet: %s", scrubbed_failure(exc))


__all__ = ["publish_upload_analysis"]
