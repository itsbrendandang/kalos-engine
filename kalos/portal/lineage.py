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


def roles_used(result: dict[str, Any]) -> dict[str, Any]:
    """`roles_used`: the roles this analysis was fit with. `declared` when the
    upload (or the campaign, on a reanalyze) gave roles, else `inferred` from
    the column names, reported as what the fit used."""
    declared = _run(result).get("roles_declared")
    if isinstance(declared, dict) and declared:
        return {"source": "declared", "roles": dict(declared)}
    return {"source": "inferred", "roles": {
        "target": result.get("target"),
        "features": list(result.get("features") or []),
        "categoricals": list(result.get("categorical_features") or []),
        "direction": result.get("direction"),
    }}


def _names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def inputs_changed(new: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any] | None:
    """The `inputs_changed` finding when the fitted inputs differ from the
    previous analysis of the campaign, else None."""
    before = [str(f) for f in previous.get("features") or []]
    after = [str(f) for f in new.get("features") or []]
    added = [f for f in after if f not in before]
    removed = [f for f in before if f not in after]
    if not added and not removed:
        return None
    parts = []
    if added:
        parts.append(f"{_names(added)} {'is' if len(added) == 1 else 'are'} now used as "
                     f"{'an input' if len(added) == 1 else 'inputs'}.")
    if removed:
        parts.append(f"{_names(removed)} {'is' if len(removed) == 1 else 'are'} no longer used.")
    return {
        "code": "inputs_changed", "severity": "warn",
        "message": "The inputs changed since the last model. " + " ".join(parts),
        "added": added, "removed": removed,
    }


# The fields `link` sets; an identical re-upload keeps the previous record's.
_LINEAGE_FIELDS = ("supersedes", "campaign_findings")


def link(
    new: dict[str, Any], previous: dict[str, Any] | None, *, n_folded: int | None = None,
) -> dict[str, Any]:
    """`new` as it will be stored in a campaign whose current analysis is
    `previous` (None: the campaign had none). Adds `analysis_id` if missing,
    `supersedes` and `campaign_findings`. The same id as `previous` (an
    identical re-upload) keeps the previous record's lineage. Returns a new dict."""
    out = dict(new)
    if not out.get("analysis_id"):
        out["analysis_id"] = new_analysis_id()
    out.setdefault("roles_used", roles_used(out))
    if not previous:
        out["supersedes"] = None
        out["campaign_findings"] = []
        return out
    previous = with_id(previous)
    if previous["analysis_id"] == out["analysis_id"]:
        for key in _LINEAGE_FIELDS:
            out[key] = previous.get(key)
        return out
    code, why = why_refit(out, previous, n_folded=n_folded)
    out["supersedes"] = {"analysis_id": previous["analysis_id"], "code": code, "why": why}
    changed = inputs_changed(out, previous)
    out["campaign_findings"] = [changed] if changed else []
    return out
