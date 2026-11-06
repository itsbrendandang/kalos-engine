"""Which stored analysis this is, and what it replaced (docs/CAMPAIGNS_API.md,
"Analysis identity").

Every analysis gets an `analysis_id` when it is computed. The id travels with
the stored record, so `/api/run`, `/api/latest`, `/api/campaign`,
`/api/campaigns/{id}/analysis`, the reanalyze response and the job view all
name the same analysis with the same id. An identical re-upload answered from
the in-process cache returns the cached analysis, id included.

When a campaign gets a new analysis with a different id, the new one says what
it replaced and why in `supersedes`. Pure functions: no store, no torch.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def new_analysis_id() -> str:
    """A fresh id for a newly computed analysis."""
    return "an_" + uuid.uuid4().hex[:16]


def derived_analysis_id(stored: dict[str, Any]) -> str:
    """The id of an analysis stored before ids existed: a hash of what
    identifies it (its provenance hash and timestamps), so it is the same on
    every read."""
    prov = stored.get("proposal_provenance")
    prov_sha = prov.get("provenance_sha256") if isinstance(prov, dict) else None
    basis = json.dumps(
        [prov_sha, stored.get("timestamp"), stored.get("updated"), stored.get("dataset")],
        default=str,
    )
    return "an_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def with_id(stored: dict[str, Any]) -> dict[str, Any]:
    """`stored` with an `analysis_id` (derived for an old record)."""
    if isinstance(stored.get("analysis_id"), str) and stored["analysis_id"]:
        return stored
    return {**stored, "analysis_id": derived_analysis_id(stored)}


def _run(analysis: dict[str, Any]) -> dict[str, Any]:
    run = analysis.get("run")
    return run if isinstance(run, dict) else {}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def why_refit(new: dict[str, Any], previous: dict[str, Any], *, n_folded: int | None) -> tuple[str, str]:
    """(code, plain sentence): why `new` replaced `previous`."""
    run, prev_run = _run(new), _run(previous)
    if n_folded is not None or run.get("kind") == "reanalyze":
        n = int(n_folded or 0)
        return "results_added", f"Kalos fit the model again with {_plural(n, 'recorded result')} added."
    same_input = bool(run.get("input_sha256")) and run.get("input_sha256") == prev_run.get("input_sha256")
    if not same_input:
        return "new_upload", "A new run sheet was uploaded, so Kalos fit a new model."
    if new.get("target") != previous.get("target"):
        return "target_changed", "The target changed, so Kalos fit the model again."
    if run.get("roles_declared") != prev_run.get("roles_declared"):
        return "roles_changed", "The column choices changed, so Kalos fit the model again."
    if run.get("direction") != prev_run.get("direction"):
        return "direction_changed", "The direction changed, so Kalos fit the model again."
    return "refit", "The same sheet was fit again with different settings."


def link(
    new: dict[str, Any], previous: dict[str, Any] | None, *, n_folded: int | None = None,
) -> dict[str, Any]:
    """`new` as it will be stored in a campaign whose current analysis is
    `previous` (None: the campaign had none). Adds `analysis_id` if missing and
    `supersedes`. The same id as `previous` (an identical re-upload) keeps the
    previous record's lineage. Returns a new dict."""
    out = dict(new)
    if not out.get("analysis_id"):
        out["analysis_id"] = new_analysis_id()
    if not previous:
        out["supersedes"] = None
        return out
    previous = with_id(previous)
    if previous["analysis_id"] == out["analysis_id"]:
        out["supersedes"] = previous.get("supersedes")
        return out
    code, why = why_refit(out, previous, n_folded=n_folded)
    out["supersedes"] = {"analysis_id": previous["analysis_id"], "code": code, "why": why}
    return out
