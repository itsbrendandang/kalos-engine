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

import copy
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import pandas as pd

from kalos.portal.lineage import new_analysis_id, roles_used
from kalos.portal.uploads import (
    ParsedUpload,
    UploadRejected,
    parse_upload,
    scrubbed_failure,
)

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


def _parse(raw: bytes, filename: str, sheet: str | None, anonymize: bool) -> ParsedUpload:
    """`parse_upload`, with sheet names aliased under `anonymize`: the refusal
    (`sheet_not_found`, `no_usable_sheet`) and the parsed workbook both carry
    `sheet_<hash>` names, never the uploader's sheet names (kalos/portal/analysis.py)."""
    from kalos.portal.analysis import anonymize_parsed, anonymize_sheet_rejection

    try:
        parsed = parse_upload(raw, filename=filename, sheet=sheet)
    except UploadRejected as rej:
        if anonymize:
            raise anonymize_sheet_rejection(rej) from rej
        raise
    return anonymize_parsed(parsed) if anonymize else parsed


def torch_threads_actual() -> int | None:
    """What torch reports as its thread count now, or None when torch has not
    been loaded (a stub analysis, or a path that never fit). Read at record time,
    after the fit, so it is the count the fit actually ran with."""
    import sys

    torch = sys.modules.get("torch")
    return int(torch.get_num_threads()) if torch is not None else None


def _roles_and_profile(roles_json: str) -> tuple[Any, Any]:
    from kalos.domains import (
        BIOPROCESS_PROFILE,
        GENERIC_PROFILE,
        ColumnRoles,
        InvalidAllowBeyond,
        InvalidApplicabilityDistance,
        InvalidDirection,
    )

    if roles_json.strip():
        try:
            return ColumnRoles.from_dict(json.loads(roles_json)), GENERIC_PROFILE
        except (InvalidDirection, InvalidApplicabilityDistance, InvalidAllowBeyond) as exc:
            raise UploadRejected(str(exc), code=exc.code) from None
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
    *, sheet: str | None = None, direction: str | None = None,
) -> ParsedUpload:
    """Parse the upload and run every refusal `_analyze` raises before its first
    model fit (`_analyze(..., preflight_only=True)`: the same code, stopped
    early). Raises exactly what the sync path would (`UploadRejected` with
    sheet context, `ValueError`, parser errors). Returns the parsed upload."""
    from kalos.portal.analysis import _analyze

    parsed = _parse(raw, filename, sheet, anonymize)
    try:
        roles, profile = _roles_and_profile(roles_json)
        _analyze(parsed.df, target, anonymize=anonymize, roles=roles, profile=profile,
                 preflight_only=True, direction=direction or None)
    except Exception as exc:
        annotated = _annotated(parsed, exc, anonymize)
        if annotated is exc:
            raise
        raise annotated from exc
    return parsed


# --- same upload, same analysis: an in-process result cache ------------------ #
# A re-upload of identical bytes with identical settings (the "replace" retry, a
# page reload, the desktop app re-sending its sheet) returned the same seeded
# analysis after a full refit: 6-60 s on a laptop for nothing. The analysis is
# now kept for the last few distinct inputs and returned on a repeat, so the
# same upload is not refit and gives the same proposals every time.
#
# The key covers the tenant (one tenant never gets another's analysis, even for
# the same bytes) and everything the analysis reads: the uploaded bytes, the file
# name (it picks the parser), the sheet, target, roles, anonymize, the profile,
# the engine version, the analyze function, and every KALOS_* setting (the
# validation mode, the row caps, ...). Only successful analyses are kept; a
# refusal is recomputed. Results are deep-copied in and out. Per process: the
# background-job workers do not share it. `KALOS_ANALYSIS_CACHE` is the number
# of entries (default 4; 0 turns it off; at most `ANALYSIS_CACHE_MAX`, since each
# entry is a whole analysis held in memory).
ANALYSIS_CACHE_MAX = 32


def _cache_size() -> int:
    raw = os.environ.get("KALOS_ANALYSIS_CACHE", "").strip()
    try:
        return min(ANALYSIS_CACHE_MAX, max(0, int(raw))) if raw else 4
    except ValueError:
        return 4


class _AnalysisCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[Callable[..., dict], dict]] = OrderedDict()

    @staticmethod
    def key(raw: bytes, *parts: object) -> str:
        from kalos import __version__

        h = hashlib.sha256()
        h.update(hashlib.sha256(raw).digest())
        for part in (*parts, __version__):
            h.update(b"\x00" + repr(part).encode("utf-8"))
        for name in sorted(k for k in os.environ if k.startswith("KALOS_")):
            h.update(b"\x01" + name.encode() + b"=" + os.environ[name].encode("utf-8", "replace"))
        return h.hexdigest()

    def get(self, key: str, analyze: Callable[..., dict]) -> dict | None:
        if _cache_size() == 0:
            return None
        with self._lock:
            entry = self._entries.get(key)
            # The entry holds the function, so its identity cannot be reused by a
            # different function while the entry exists.
            if entry is None or entry[0] is not analyze:
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(entry[1])

    def put(self, key: str, analyze: Callable[..., dict], result: dict) -> None:
        size = _cache_size()
        if size == 0:
            return
        with self._lock:
            self._entries[key] = (analyze, copy.deepcopy(result))
            self._entries.move_to_end(key)
            while len(self._entries) > size:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


ANALYSIS_CACHE = _AnalysisCache()


def compute_upload_analysis(
    raw: bytes, target: str | None, anonymize: bool, filename: str, roles_json: str = "",
    *, sheet: str | None = None, progress: ProgressFn | None = None,
    analyze: Callable[..., dict] | None = None, direction: str | None = None,
    tenant: str | None = None,
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
    parsed = _parse(raw, filename, sheet, anonymize)
    kwargs: dict[str, Any] = {"anonymize": anonymize}
    if progress is not None:
        kwargs["progress"] = progress
    if direction:
        # The plain `direction` form field (/api/run). A different roles.direction is refused.
        kwargs["direction"] = direction
    try:
        roles, profile = _roles_and_profile(roles_json)
        if roles is not None:
            kwargs["roles"] = roles
        profile_name = profile.name
        # Every argument the analysis receives is in the key (progress aside),
        # and the tenant, so a cached analysis is never served across tenants.
        cache_key = ANALYSIS_CACHE.key(
            raw, tenant, filename, sheet, target, roles_json, bool(anonymize), profile_name,
            direction or None,
        )
        cached = ANALYSIS_CACHE.get(cache_key, analyze)
        if cached is not None:
            if progress is not None:
                progress("proposing", 0.8)
            result = {**cached, "analysis_cache": {"hit": True}}
        else:
            # The id is minted with the fit and cached with it: a repeat of the
            # same upload is the same analysis (kalos/portal/lineage.py).
            result = {**analyze(parsed.df, target, profile=profile, **kwargs),
                      "analysis_id": new_analysis_id()}
            ANALYSIS_CACHE.put(cache_key, analyze, result)
            result = {**result, "analysis_cache": {"hit": False}}
    except Exception as exc:
        annotated = _annotated(parsed, exc, anonymize)
        if annotated is exc:
            raise
        raise annotated from exc
    # Excel uploads only: `sheets` (every sheet, workbook order) and
    # `sheet_used` (the one analyzed), so a client can offer a sheet selector.
    # `run` (kalos/portal/run_record.py) is attached here, after the parse and
    # the analysis, so the sync routes and the job worker record the same thing.
    # `n_features` is the fitted feature count (`result["d"]`), NOT
    # `proposal_features`, which is only the top drivers the proposals vary.
    from kalos.portal import run_record
    from kalos.portal.analysis import ANALYZE_SEED

    record = run_record.build(
        raw, filename, parsed.sheet_used, result.get("target", target), roles_json,
        anonymize, profile_name, TORCH_THREADS, int(result.get("seed", ANALYZE_SEED)),
        len(parsed.df), int(result["d"]), sheet_index=parsed.sheet_index,
        sheet_hash=parsed.sheet_hash, torch_threads_actual=torch_threads_actual(),
        data_drops=run_record.drops_summary(result), header_row=result.get("header_row"),
        direction=result.get("direction"),
    )
    run = record.to_dict()
    out = {**result, **parsed.context(), "run": run,
           "proposal_provenance": _provenance(run, result)}
    return parsed.df, {**out, "roles_used": roles_used(out)}


def attach_reanalyze_run(
    df: pd.DataFrame, result: dict, *, anonymize: bool, pending_runs: int,
    roles_json: str = "", profile: str = "bioprocess",
) -> dict:
    """The `run` block for a campaign re-analysis (`kind="reanalyze"`): the same
    fields an upload gets, with the input hashed over the folded dataset `df`
    and the number of in-flight runs the batch was told about. `roles_json` and
    `profile` are the campaign's stored roles and the profile they imply.
    Returns a new dict; `result` is not mutated."""
    from kalos.portal import run_record
    from kalos.portal.analysis import ANALYZE_SEED

    record = run_record.build(
        run_record.fold_bytes(df), None, None, result.get("target"), roles_json, anonymize,
        profile, TORCH_THREADS, int(result.get("seed", ANALYZE_SEED)),
        len(df), int(result["d"]), kind="reanalyze", pending_runs=pending_runs,
        torch_threads_actual=torch_threads_actual(),
        data_drops=run_record.drops_summary(result),
        direction=result.get("direction"),
    )
    run = record.to_dict()
    out = {**result, "run": run, "proposal_provenance": _provenance(run, result),
           "analysis_id": new_analysis_id()}
    return {**out, "roles_used": roles_used(out)}


def _provenance(run: dict, result: dict) -> dict:
    """`proposal_provenance` (kalos/portal/run_record.py): engine and library
    versions, seed, input hash, config hash and timestamp, stored with the
    analysis and returned with it. No model artefact is stored."""
    from kalos import __version__ as engine_version
    from kalos.portal import run_record

    return run_record.proposal_provenance(
        run, engine_version=str(result.get("engine_version") or engine_version),
        timestamp=result.get("timestamp"),
        proposal_fit_restarts=(result.get("model_diagnostics") or {}).get("restarts"),
    )


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

    # The label the campaign and /api/latest store for this upload. Under
    # anonymize the filename is never stored (the run record has no filename
    # either), only a label derived from the input hash.
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
        # Exception type + location only (#87): the message can quote cell values.
        log.error("failed to seed the campaign from an uploaded run sheet: %s", scrubbed_failure(exc))
        _publish_unstamped(store, result, filename, tenant=tenant, campaign_id=campaign_id)
    return None


def stored_dataset_label(result: dict, filename: str) -> str:
    """The `dataset` label an analysis is stored under. The raw filename, except
    when the result's run record says `anonymize`: then a label built from the
    input hash, so a stored analysis never carries the uploader's filename."""
    run = result.get("run")
    if isinstance(run, dict) and run.get("anonymize"):
        digest = str(run.get("input_sha256") or "")[:12]
        return f"anonymized upload {digest}" if digest else "anonymized upload"
    return filename


def _publish_unstamped(
    store: Any, result: dict, filename: str, *, tenant: str, campaign_id: str | None = None,
) -> None:
    try:
        store.save_analysis(
            result, filename, tenant=tenant, campaign_id=campaign_id, campaign_generation=None,
        )
    except Exception as exc:  # noqa: BLE001 - publishing is best-effort too
        log.error("failed to publish the analysis of an uploaded run sheet: %s", scrubbed_failure(exc))
