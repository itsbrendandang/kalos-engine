"""The upload-analysis pipeline, shared by the sync upload routes
(`/api/run`, `/api/campaigns/{id}/upload`) and the background-job worker
(kalos/portal/job_worker.py), so both paths compute and publish an analysis
with the SAME code.

Three steps:

- `preflight_upload`: parse + every pre-fit refusal `_analyze` would raise
  (shape, target, constant target, no process inputs, row caps). Cheap (pandas
  only, no torch), so a background job is refused with the sync route's 400
  BEFORE it is queued.
- `compute_upload_analysis`: parse + `_analyze` (the CPU-heavy part). Pure: it
  writes nothing.
- `publish_upload_analysis`: seed the campaign and publish the analysis in one
  store transaction (`CampaignStore.seed`). A job passes `expected_generation`
  (compare-and-swap) and `on_commit` (marks the job succeeded in the same
  transaction), so a stale or cancelled job can never overwrite newer state.

Torch-free at import time on purpose: the job worker imports this module in a
freshly spawned interpreter, and torch is imported only when `_analyze` runs.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any

import pandas as pd

from kalos.portal.uploads import ParsedUpload, UploadRejected, parse_upload

log = logging.getLogger("kalos.portal")

ProgressFn = Callable[[str, float], None]

# --- CPU thread budget (moved from app.py; re-exported there) ---------------- #
# See the KALOS_TORCH_THREADS comment in kalos/portal/app.py's history: 0 means
# DO NOT PIN (the container default), the bare-metal default is min(4, cpus).
TORCH_THREADS = int(
    os.environ.get("KALOS_TORCH_THREADS", "").strip() or str(min(4, os.cpu_count() or 1))
)
_torch_threads_configured = False


def ensure_torch_threads() -> None:
    """Apply `TORCH_THREADS` on first use (idempotent; deferred so importing the
    portal stays torch-free until a fit actually runs)."""
    global _torch_threads_configured
    if _torch_threads_configured:
        return
    if TORCH_THREADS <= 0:
        _torch_threads_configured = True
        return
    import torch

    torch.set_num_threads(TORCH_THREADS)
    _torch_threads_configured = True


def _roles_and_profile(roles_json: str) -> tuple[Any, Any]:
    from kalos.domains import BIOPROCESS_PROFILE, GENERIC_PROFILE, ColumnRoles

    if roles_json.strip():
        return ColumnRoles.from_dict(json.loads(roles_json)), GENERIC_PROFILE
    return None, BIOPROCESS_PROFILE


def _annotated(parsed: ParsedUpload, exc: BaseException, anonymize: bool) -> BaseException:
    """The error the sync route has always raised for a failed analysis: an
    `UploadRejected` gets the sheet context (and pseudonyms with `anonymize`),
    anything else carries `upload_context` for the generic error body."""
    from kalos.portal.analysis import anonymize_rejection

    if isinstance(exc, UploadRejected):
        refused = anonymize_rejection(exc) if anonymize else exc
        return parsed.annotate(refused)
    try:
        exc.upload_context = parsed.context()  # type: ignore[attr-defined]
    except AttributeError:  # an exception type that forbids new attributes
        pass
    return exc


def preflight_upload(
    raw: bytes, target: str | None, anonymize: bool, filename: str, roles_json: str = "",
    *, sheet: str | None = None,
) -> ParsedUpload:
    """Parse the upload and run every refusal `_analyze` raises before its first
    model fit (`_analyze(..., preflight_only=True)`: the same code, stopped
    early). Raises exactly what the sync path would (`UploadRejected` with
    sheet context, `ValueError`, parser errors). Returns the parsed upload."""
    from kalos.portal.analysis import _analyze

    parsed = parse_upload(raw, filename=filename, sheet=sheet)
    try:
        roles, profile = _roles_and_profile(roles_json)
        _analyze(parsed.df, target, anonymize=anonymize, roles=roles, profile=profile,
                 preflight_only=True)
    except Exception as exc:
        annotated = _annotated(parsed, exc, anonymize)
        if annotated is exc:
            raise
        raise annotated from exc
    return parsed


def compute_upload_analysis(
    raw: bytes, target: str | None, anonymize: bool, filename: str, roles_json: str = "",
    *, sheet: str | None = None, progress: ProgressFn | None = None,
    analyze: Callable[..., dict] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """parse -> `_analyze` -> the response dict. Returns `(df, result)`; writes
    nothing. `analyze` defaults to `kalos.portal.analysis._analyze` (the sync
    route passes `kalos.portal.app._analyze` so its tests can stub it)."""
    if analyze is None:
        from kalos.portal.analysis import _analyze

        analyze = _analyze
    ensure_torch_threads()
    if progress is not None:
        progress("parsing", 0.05)
    parsed = parse_upload(raw, filename=filename, sheet=sheet)
    kwargs: dict[str, Any] = {"anonymize": anonymize}
    if progress is not None:
        kwargs["progress"] = progress
    try:
        roles, profile = _roles_and_profile(roles_json)
        if roles is not None:
            kwargs["roles"] = roles
        result = analyze(parsed.df, target, profile=profile, **kwargs)
    except Exception as exc:
        annotated = _annotated(parsed, exc, anonymize)
        if annotated is exc:
            raise
        raise annotated from exc
    # Excel uploads only: `sheets` (every sheet, workbook order) and
    # `sheet_used` (the one analyzed), so a client can offer a sheet selector.
    return parsed.df, {**result, **parsed.context()}


_ANY = object()


def publish_upload_analysis(
    store: Any, result: dict, filename: str, *, tenant: str, campaign_id: str | None,
    df: pd.DataFrame | None = None, base_rows: list[dict[str, Any]] | None = None,
    expected_generation: Any = _ANY, on_commit: Callable[[Any, str], None] | None = None,
    fallback_unstamped: bool = True,
) -> str | None:
    """Seed the campaign with the upload and publish `result` in the same
    transaction (docs/CAMPAIGN_LOOP.md, "Seeding"). Returns the new generation,
    or None if only the unstamped fallback was published.

    `expected_generation` (jobs): the seed is a compare-and-swap on it and
    raises `CampaignConflict` when the campaign changed meanwhile. `on_commit`
    runs inside the transaction with `(conn, new_generation)`; raising there
    rolls the whole write back.

    Errors: an explicit `campaign_id` that vanished, was archived, or changed
    raises its `CampaignError` (404/409). With `fallback_unstamped` (the sync
    routes), any OTHER seeding failure is logged and the analysis is still
    published unstamped, so the campaign reports itself out of sync instead of
    the two silently describing different run sheets. Jobs pass False: a job
    either commits atomically or writes nothing.
    """
    from kalos.portal.campaign import CampaignError

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
    except CampaignError:
        if campaign_id is not None or not fallback_unstamped:
            raise
        log.exception("failed to seed the campaign from an uploaded run sheet")
        _publish_unstamped(store, result, filename, tenant=tenant)
    except Exception:  # noqa: BLE001 - seeding must never fail a sync upload
        if not fallback_unstamped:
            raise
        log.exception("failed to seed the campaign from an uploaded run sheet")
        _publish_unstamped(store, result, filename, tenant=tenant, campaign_id=campaign_id)
    return None


def _publish_unstamped(
    store: Any, result: dict, filename: str, *, tenant: str, campaign_id: str | None = None,
) -> None:
    try:
        store.save_analysis(
            result, filename, tenant=tenant, campaign_id=campaign_id, campaign_generation=None,
        )
    except Exception:  # noqa: BLE001 - publishing is best-effort too
        log.exception("failed to publish the analysis of an uploaded run sheet")
